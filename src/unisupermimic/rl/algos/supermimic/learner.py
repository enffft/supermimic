"""SuperMimic training over the injected environment and MDP protocols."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from unilab.base.contracts.env import EnvProtocol

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    DirectGoalMdpProtocol,
    State,
    get_supermimic_mdp,
)
from unisupermimic.rl.logging import SuperMimicLogger

from . import networks
from .geometry import unflatten_state
from .networks import Critic, Policy, WorldModel, mlp
from .optim import MuonAdam
from .replay import ReplayBuffer


@dataclass
class SuperMimicConfig:
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    goal_mode: str = "encoded"
    hist: int = 10
    z_dim: int = 32
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    wm_structured: bool = False
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 4e-4
    pi_opt: str = "muon"
    wm_opt: str = "muon"
    no_sync: bool = True
    pi_ema: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    numbered_ckpt_every: int = 0  # 0 retains the original checkpoint cadence
    log_every: int = 50
    log_backend: str = "tensorboard"
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    test_hidden_dims: tuple[int, ...] | None = None
    profile: bool = False
    init_actor_from: str | None = None
    init_wm_from: str | None = None
    driver_enabled: bool = False
    terrain_training: bool = False  # explicit opt-in, including no-scan ablation
    terrain_conditioning: bool = False
    terrain_sensor_grad: bool = True
    critic_terrain_conditioning: bool = False  # reserved, not silently enabled
    actor_lr_scale: float = 1.0
    actor_ramp_iters: int = 0
    critic_warmup_iters: int = 0
    checkpoint_replay: bool = False
    strict_finite: bool = False


_LOG_FIELDS = (
    "iteration",
    "wall_h",
    "sim_steps",
    "wm_loss",
    "wm_err8",
    "pi_loss",
    "v_loss",
    "imag_rew",
    "pi_gradnorm",
    "imag_survival",
    "sim_rew",
    "ep_len",
    "ep_ret_norm",
    "term_frac",
    "it_per_s",
    "lr",
)

# ROCm's flattened small-matrix bmm path is numerically unreliable above a batch
# boundary (~32x4096 rollout); chunk the geometry-heavy reward/critic inputs
# there. Other backends keep a single fused call so their hot path is unchanged.
_SAFE_GEOMETRY_BATCH = 4096 if getattr(torch.version, "hip", None) is not None else 0


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class SuperMimicLearner:
    def __init__(
        self,
        env: EnvProtocol,
        cfg: SuperMimicConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        backend: str = "",
    ):
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.pi_opt not in ("adam", "muon"):
            raise ValueError(f"pi_opt must be 'adam' or 'muon', got {cfg.pi_opt!r}")
        if cfg.wm_opt not in ("adam", "muon"):
            raise ValueError(f"wm_opt must be 'adam' or 'muon', got {cfg.wm_opt!r}")
        if cfg.test_hidden_dims is not None and not cfg.test_hidden_dims:
            raise ValueError("test_hidden_dims must be nonempty when provided")

        self.env = env
        self.cfg = cfg
        if cfg.goal_mode not in ("encoded", "direct"):
            raise ValueError("goal_mode must be encoded or direct")
        if cfg.goal_mode == "direct" and (cfg.goal_frames != 1 or cfg.test_hidden_dims is not None):
            raise ValueError(
                "direct mode requires goal_frames=1 and explicit production network widths"
            )
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        # HOST_BRIDGE backends (e.g. MuJoCo) keep env tensors on CPU while the
        # learner may run on an accelerator (MPS on macOS); when the two differ,
        # the learner works on its own MDP copy and bridges tensors at the env
        # boundary. On CUDA/ROCm the devices always match and nothing changes.
        self._bridged = self.device != self.env_device
        mdp = get_supermimic_mdp(env)
        self.mdp = mdp.to(self.device) if self._bridged else mdp
        if cfg.goal_mode == "direct" and not isinstance(self.mdp, DirectGoalMdpProtocol):
            raise TypeError("direct goals require DirectGoalMdpProtocol")
        generated_terrain = (
            getattr(getattr(getattr(env, "cfg", None), "scene", None), "terrain", None) is not None
        )
        if cfg.critic_terrain_conditioning:
            raise NotImplementedError(
                "Privileged terrain critic is not implemented; use the original critic"
            )
        if generated_terrain and cfg.goal_mode == "direct":
            if not cfg.terrain_training or getattr(self.mdp, "terrain", None) is None:
                raise ValueError(
                    "Go2 terrain training is gated: requires explicit opt-in and collision-aligned terrain MDP"
                )
        if cfg.terrain_conditioning and (
            cfg.goal_mode != "direct" or getattr(self.mdp, "terrain", None) is None
        ):
            raise ValueError("Terrain WM requires Go2 direct mode and a bound heightfield")
        self.profile_ms: dict[str, list[float]] = {}

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16, while on ROCm a TF32 bmm over the
        # misaligned strided views produced by unflatten_state triggers per-call
        # rocBLAS kernel selection costing ~100 ms per call.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        networks.RunningNorm.no_sync = cfg.no_sync

        self.wm = self._make_world_model().to(self.device)
        self.pi = self._make_policy().to(self.device)
        self.collection_driver = None
        self.initialization = {"actor": "fresh", "wm": "fresh", "critic": "fresh"}
        if cfg.init_actor_from is not None:
            if cfg.goal_mode != "direct":
                raise ValueError(
                    "Actor-only initialization is implemented only for Go2 direct mode"
                )
            from unisupermimic.tasks.velocity.go2.checkpoint import initialize_actor

            self.initialization["actor"] = initialize_actor(
                self.pi, cfg.init_actor_from, env.supermimic_contract
            )
        if cfg.init_wm_from is not None:
            self._initialize_world_model(cfg.init_wm_from)
        self.critic = Critic(self._critic_input_dim(), hidden=cfg.critic_hidden).to(self.device)
        if cfg.test_hidden_dims is not None:
            self.critic.net = mlp(self.critic.norm.mean.numel(), cfg.test_hidden_dims, 1).to(
                self.device
            )
        self.pi_avg = None
        if cfg.pi_ema > 0:
            # Polyak average of the policy weights; checkpoints and evaluation use it while the
            # raw ``self.pi`` keeps being trained. Buffers (running normalizers) are averaged too.
            self.pi_avg = AveragedModel(
                self.pi,
                multi_avg_fn=get_ema_multi_avg_fn(cfg.pi_ema),
                use_buffers=True,
            )
        fused_adam = self.device.type == "cuda"
        if cfg.wm_opt == "muon":
            self.opt_wm = MuonAdam(
                self.wm,
                [self.wm.net],
                lr=cfg.lr,
                betas=(0.9, 0.999),
                fused=fused_adam,
            )
        else:
            self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr, fused=fused_adam)
        if cfg.pi_opt == "muon":
            self.opt_pi = MuonAdam(
                self.pi,
                ([self.pi.dec] if cfg.goal_mode == "direct" else [self.pi.enc, self.pi.dec]),
                lr=cfg.lr,
                betas=(0.9, 0.95),
                fused=fused_adam,
            )
        else:
            self.opt_pi = torch.optim.Adam(
                self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95), fused=fused_adam
            )
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, fused=fused_adam)
        self._v_mean = torch.zeros((), device=self.device)
        self._v_std = torch.ones((), device=self.device)
        self._v_ema = 0.01
        self._v_init = False

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
            goal_dim=self.mdp.goal_frame_dim if cfg.goal_mode == "direct" else 0,
            terrain_metadata=cfg.checkpoint_replay
            and generated_terrain
            and cfg.goal_mode == "direct",
        )
        self.obs_hist = torch.zeros(env.num_envs, cfg.hist, self.mdp.obs_dim, device=self.device)
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(self.device)

        self.on_iteration = None  # Optional validation/guard callback; one shared train loop.
        self.stop_requested = False
        self.extra_state = {}
        self.update_counts = {"wm": 0, "actor": 0, "critic": 0}
        self.step_accumulator = 0.0
        self.resume_mode = "new_run"
        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats = _empty_stats()
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_input_fn = self._compile(self._critic_input)

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)
        (self.log_dir / "initialization.json").write_text(json.dumps(self.initialization, indent=2))

    def _initialize_world_model(self, path):
        import hashlib

        values = torch.load(path, map_location=self.device, weights_only=True)
        source_contract = values.get("task_contract")
        target_contract = getattr(self.env, "supermimic_contract", None)
        source_args = values.get("args", {})
        if not source_contract or not target_contract:
            raise ValueError("WM initialization requires explicit UniLab task protocol metadata")
        for key in ("actor", "joints", "context", "action", "sim_dt", "ctrl_dt", "state"):
            if source_contract.get(key) != target_contract.get(key):
                raise ValueError(f"WM protocol mismatch: {key}")
        # Explicitly permit only the known nominal flat -> audited same Go2 model.
        # Absolute XY is absent from phi(s); the known flat vertical datum is zero.
        from unisupermimic.tasks.velocity.go2.contract import MODEL

        if source_contract.get("model_sha256") != hashlib.sha256(MODEL.read_bytes()).hexdigest():
            raise ValueError("WM expansion supports only the known nominal UniLab flat robot")
        if source_contract.get("terrain") is not None or source_args.get(
            "terrain_conditioning", False
        ):
            raise ValueError("WM expansion source must be an unconditioned flat UniLab model")
        if (
            tuple(source_args.get("wm_hidden", ())) != tuple(self.cfg.wm_hidden)
            or bool(source_args.get("wm_structured", False)) != self.cfg.wm_structured
        ):
            raise ValueError("WM widths or integration protocol differ")
        if not self.cfg.terrain_conditioning:
            raise ValueError(
                "WM-only transfer is reserved for explicit terrain expansion; use resume for flat"
            )
        source = WorldModel(
            hidden=self.cfg.wm_hidden,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
            structured=self.cfg.wm_structured,
        ).to(self.device)
        source.load_state_dict(values["wm"], strict=True)
        self.wm.initialize_from_flat(source)
        self.initialization["wm"] = {
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "method": "same_UniLab_protocol; appended_zero_columns; fresh_optimizer",
        }

    def set_collection_driver(self, driver):
        if not self.cfg.driver_enabled or self.cfg.goal_mode != "direct":
            raise ValueError("Driver requires explicit Go2 opt-in")
        from unisupermimic.tasks.velocity.go2.contract import DEFAULT_Q, JOINTS

        from .driver import DriverActionContract

        expected = DriverActionContract(tuple(JOINTS), tuple(DEFAULT_Q), 0.25, 0.02, 40.0, 1.0, 1.0)
        if driver.contract != expected:
            raise ValueError("Driver joint/default/scale/PD/frequency/coordinate contract mismatch")
        self.collection_driver = driver

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
        return torch.compile(function)

    @contextmanager
    def profile_section(self, name):
        """Opt-in synchronized timings, with operator ranges for torch.profiler."""
        if not self.cfg.profile:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.profiler.record_function(name):
            yield
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.profile_ms.setdefault(name, []).append(1000 * (time.perf_counter() - started))

    def _make_world_model(self) -> WorldModel:
        if self.cfg.test_hidden_dims is not None:
            return WorldModel(
                hidden=self.cfg.test_hidden_dims,
                num_joints=self.mdp.num_joints,
                context_dim=self.mdp.context_dim,
                structured=self.cfg.wm_structured,
            )
        return WorldModel(
            hidden=self.cfg.wm_hidden,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
            structured=self.cfg.wm_structured,
            terrain_dim=77 if self.cfg.terrain_conditioning else 0,
        )

    def _make_policy(self) -> Policy:
        goal_dim = self.mdp.goal_frame_dim * self.cfg.goal_frames
        policy = Policy(
            self.mdp.obs_dim,
            goal_dim,
            hist=self.cfg.hist,
            z_dim=self.cfg.z_dim,
            num_joints=self.mdp.num_joints,
            hidden=self.cfg.policy_hidden,
            goal_mode=self.cfg.goal_mode,
        )
        if self.cfg.test_hidden_dims is not None:
            enc_output = policy.enc[-1]
            assert isinstance(enc_output, nn.Linear)
            z_dim = enc_output.out_features
            policy.enc = mlp(goal_dim, self.cfg.test_hidden_dims, z_dim)
            policy.dec = mlp(
                z_dim + self.mdp.obs_dim * self.cfg.hist,
                self.cfg.test_hidden_dims,
                self.mdp.num_joints,
            )
            dec_output = policy.dec[-1]
            assert isinstance(dec_output, nn.Linear)
            with torch.no_grad():
                dec_output.weight.mul_(0.01)
                dec_output.bias.zero_()
        return policy

    def _state_info(self) -> dict[str, torch.Tensor]:
        state = self.env.state
        if state is None:
            state = self.env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    def _critic_input_dim(self) -> int:
        info = self._state_info()
        with torch.no_grad():
            features = self.mdp.critic_features(
                unflatten_state(info["s"]),
                info["f"],
                self.mdp.normalize_context(info["xi"]),
            )
        return int(features.shape[-1] + self.mdp.goal_frame_dim * self.cfg.goal_frames)

    def _terrain_scan(self, state):
        if not self.cfg.terrain_conditioning:
            return None
        return self.mdp.terrain.local_scan(
            state[0], state[1], sensor_grad=self.cfg.terrain_sensor_grad
        )

    def _wm_step(self, state, action, normalized_context):
        # Every autoregressive H8/H32 step queries its own predicted WORLD pose.
        return self.wm.step(state, action, normalized_context, self._terrain_scan(state))

    def _wm_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor
    ) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            state = self._wm_step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(state)
        return (
            torch.stack([prediction[0] for prediction in predictions], 1),
            torch.stack([prediction[1] for prediction in predictions], 1),
            torch.stack([prediction[2] for prediction in predictions], 1),
            torch.stack([prediction[3] for prediction in predictions], 1),
            torch.stack([prediction[4] for prediction in predictions], 1),
            torch.stack([prediction[5] for prediction in predictions], 1),
        )

    def _wm_err(
        self, predicted: State, next_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.maximum(self.wm.inc_norm.std(), self.err_floor)
        error = (self.wm.state_error(predicted, unflatten_state(next_states)) / std) ** 2
        error = error.mean(-1)
        return (error * mask).sum() / next_states.shape[0], error[:, -1]

    def _imag_step(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs_hist, goal)
        next_state = self._wm_step(state, action, normalized_context)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = (
            goal
            if self.cfg.goal_mode == "direct"
            else self.mdp.goal(next_frames, self.cfg.goal_frames)
        )
        return next_state, action, next_hist, next_frames, next_goal

    def _reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mdp.reward(state, frames, action, previous_action), self.mdp.soft_termination(
            state, frames
        )

    def _critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((self.mdp.critic_features(state, frames, normalized_context), goal), -1)

    def _batched_reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _SAFE_GEOMETRY_BATCH or len(frames) <= _SAFE_GEOMETRY_BATCH:
            reward, done = self._reward_term_fn(state, frames, action, previous_action)
            return reward, done
        reward_parts, done_parts = [], []
        for start in range(0, len(frames), _SAFE_GEOMETRY_BATCH):
            stop = start + _SAFE_GEOMETRY_BATCH
            reward, done = self._reward_term_fn(
                cast(State, tuple(item[start:stop] for item in state)),
                frames[start:stop],
                action[start:stop],
                previous_action[start:stop],
            )
            reward_parts.append(reward)
            done_parts.append(done)
        return torch.cat(reward_parts), torch.cat(done_parts)

    def _batched_critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        if not _SAFE_GEOMETRY_BATCH or len(frames) <= _SAFE_GEOMETRY_BATCH:
            return self._critic_input_fn(state, frames, normalized_context, goal)
        parts = []
        for start in range(0, len(frames), _SAFE_GEOMETRY_BATCH):
            stop = start + _SAFE_GEOMETRY_BATCH
            parts.append(
                self._critic_input_fn(
                    cast(State, tuple(item[start:stop] for item in state)),
                    frames[start:stop],
                    normalized_context[start:stop],
                    goal[start:stop],
                )
            )
        return torch.cat(parts)

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            info = self._state_info()
            state = unflatten_state(info["s"])
            obs = self.mdp.add_noise(self.mdp.obs(state, info["a_prev"], info["xi"], info["f"]))
            fresh = info["ep_step"] == 0
            shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            self.obs_hist = torch.where(
                fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
            )
            goal = (
                info["g"]
                if self.cfg.goal_mode == "direct"
                else self.mdp.goal(info["f"], self.cfg.goal_frames)
            )
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            if self.cfg.driver_enabled:
                if self.collection_driver is None:
                    raise ValueError("driver_enabled requires a validated collection driver")
                self.collection_driver.reset(fresh)
                action = self.collection_driver(self.obs_hist, goal)
                if action.shape != info["a_prev"].shape or not torch.isfinite(action).all():
                    raise ValueError("Driver returned invalid action")
            else:
                action = self.pi(self.obs_hist, goal)
            bound = 1 if self.cfg.goal_mode == "direct" else 20
            if self.cfg.goal_mode == "direct" and hasattr(self.env, "set_collection_exploration"):
                self.env.set_collection_exploration(self.cfg.action_noise)
                action = action.clamp(-bound, bound)
            else:
                action = torch.clamp(
                    action + self.cfg.action_noise * torch.randn_like(action), -bound, bound
                )
            with self.profile_section("simulation"):
                if self._bridged:
                    step_state = self.env.step(action.to(self.env_device))
                else:
                    step_state = self.env.step(action)
            transition = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            if self._bridged:
                transition = {key: value.to(self.device) for key, value in transition.items()}
            if self.cfg.terrain_training:
                self.buffer.mark_reset_boundary(transition["ep_step"] == 0)
            self.buffer.add(transition)
            initial = unflatten_state(transition["s"])
            next_state = unflatten_state(transition["s1"])
            self.wm.in_norm.update(
                self.wm.inputs(
                    initial,
                    transition["a"],
                    self.mdp.normalize_context(transition["xi"]),
                    self._terrain_scan(initial),
                )
            )
            target = self.wm.targets(initial, next_state)
            self.wm.out_norm.update(target)
            if self.cfg.wm_structured:
                self.wm.inc_norm.update(self.wm.increments(initial, next_state))
            else:
                self.wm.inc_norm.update(target)
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            self.stats["term"].append(transition["term"].float().sum())
            self.stats["done"].append(transition["done"].float().sum())
            if "ep_len" in transition:
                self.stats["ep_len"].append(transition["ep_len"].mean())
                self.stats["ep_ret"].append(transition["ep_ret_norm"].mean())

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        timing = self.profile_section("wm_fit")
        timing.__enter__()
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.wm.parameters(), 1.0, error_if_nonfinite=self.cfg.strict_finite
        )
        self.opt_wm.step()
        self.update_counts["wm"] += 1
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        timing.__exit__(None, None, None)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _sample_valid_starts(
        self, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.cfg.no_sync:
            # Fixed-size selection without reading the valid count on the host: oversample by 2x,
            # stable-sort the valid starts first, and keep the leading ``batch_size``. Only starts
            # within ``horizon`` frames of a clip end are invalid (< 1 %), so 2x is enough.
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = torch.sort((~valid).to(torch.uint8), stable=True).indices[:batch_size]
            return (
                {key: value[indices] for key, value in start.items()},
                {key: value[indices] for key, value in history.items()},
            )

        starts = []
        histories = []
        count = 0
        while count < batch_size:
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def policy_update(
        self,
        update_actor: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.mdp.normalize_context(context)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)

        batch_size = self.cfg.pi_batch
        initial_goal = (
            start["g"]
            if self.cfg.goal_mode == "direct"
            else self.mdp.goal(frames, self.cfg.goal_frames)
        )
        goal = initial_goal
        initial_state = state
        states, actions, previous_actions, all_frames, goals = [], [], [], [], []
        actor_timing = self.profile_section("actor_bptt")
        actor_timing.__enter__()
        try:
            for _ in range(self.cfg.horizon):
                previous_actions.append(previous_action)
                state, previous_action, obs_hist, frames, goal = self._imag_step_fn(
                    state,
                    obs_hist,
                    previous_action,
                    frames,
                    context,
                    normalized_context,
                    goal,
                )
                states.append(state)
                actions.append(previous_action)
                all_frames.append(frames)
                goals.append(goal)

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states = cast(
                State, tuple(flatten([item[index] for item in states]) for index in range(6))
            )
            flat_frames = flatten(all_frames)
            if self.cfg.goal_mode == "direct":
                # Commands are sampled from immutable replay values and held over the
                # entire imagined rollout. Acceleration uses the preceding state.
                predecessors = [initial_state, *states[:-1]]
                previous_states = cast(
                    State, tuple(flatten([item[i] for item in predecessors]) for i in range(6))
                )
                rewards = self.mdp.reward_transition(
                    previous_states,
                    all_states,
                    flatten(goals),
                    flatten(actions),
                    flatten(previous_actions),
                )
                dones = self.mdp.soft_termination(all_states, flat_frames)
            else:
                rewards, dones = self._batched_reward_term(
                    all_states, flat_frames, flatten(actions), flatten(previous_actions)
                )
            rewards = rewards.view(self.cfg.horizon, batch_size)
            dones = dones.view(self.cfg.horizon, batch_size)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
            discount = self.cfg.gamma ** torch.arange(
                self.cfg.horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            objective = (discount * continuation * rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()
            if not self.cfg.no_critic:
                bootstrap = self.critic(
                    self._critic_input_fn(state, frames, normalized_context, goal)
                )
                objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * (
                    bootstrap * self._v_std + self._v_mean
                )
            loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if self.cfg.strict_finite and not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite actor gradient; optimizer not stepped")
            if not update_actor:
                pass  # Critic warmup must not mutate actor Adam moments or weights.
            elif self.cfg.no_sync:
                # Skip a non-finite update without reading the norm on the host: zero the
                # gradients on-device instead of branching on ``torch.isfinite``.
                finite = torch.isfinite(grad_norm)
                for parameter in self.pi.parameters():
                    if parameter.grad is not None:
                        parameter.grad.copy_(torch.where(finite, parameter.grad, 0.0))
                self.opt_pi.step()
            elif torch.isfinite(grad_norm):
                self.opt_pi.step()
            if update_actor:
                self.update_counts["actor"] += 1
            if update_actor and self.pi_avg is not None:
                self.pi_avg.update_parameters(self.pi)
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)
            actor_timing.__exit__(None, None, None)

        critic_timing = self.profile_section("critic_update")
        critic_timing.__enter__()
        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            with torch.no_grad():
                critic_states = cast(
                    State,
                    tuple(
                        torch.cat(
                            (
                                initial_state[index].unsqueeze(0),
                                torch.stack([item[index].detach() for item in states]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for index in range(6)
                    ),
                )
                critic_frames = torch.cat(
                    (start["f"].unsqueeze(0), torch.stack(all_frames)), 0
                ).flatten()
                critic_goals = torch.cat(
                    (initial_goal.unsqueeze(0), torch.stack(goals)), 0
                ).flatten(0, 1)
                critic_context = (
                    normalized_context.unsqueeze(0)
                    .expand(self.cfg.horizon + 1, -1, -1)
                    .flatten(0, 1)
                )
                critic_input = self._batched_critic_input(
                    critic_states, critic_frames, critic_context, critic_goals
                ).view(self.cfg.horizon + 1, batch_size, -1)
            self.critic.norm.update(critic_input)
            with torch.no_grad():
                values = self.critic(critic_input) * self._v_std + self._v_mean
                target = values[-1]
                targets = []
                for index in range(self.cfg.horizon - 1, -1, -1):
                    target = detached_rewards[index] + self.cfg.gamma * (
                        1 - detached_dones[index]
                    ) * ((1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target)
                    targets.append(target)
                targets = torch.stack(targets[::-1])
                target_mean = targets.mean()
                target_std = targets.std().clamp(min=1e-2)
                if not self._v_init:
                    self._v_mean = target_mean.detach()
                    self._v_std = target_std.detach()
                    self._v_init = True
                else:
                    self._v_mean = (1 - self._v_ema) * self._v_mean + self._v_ema * target_mean
                    self._v_std = (
                        ((1 - self._v_ema) * self._v_std**2 + self._v_ema * target_std**2)
                        .sqrt()
                        .clamp(min=1e-2)
                    )
                targets = ((targets - self._v_mean) / self._v_std).clamp(-20.0, 20.0)
            flat_input = critic_input[:-1].reshape(-1, critic_input.shape[-1])
            flat_targets = targets.reshape(-1)
            num_samples = flat_input.shape[0]
            minibatch_size = num_samples // self.cfg.critic_minibatches
            if minibatch_size == 0:
                raise ValueError("critic_minibatches exceeds the number of imagined transitions")
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(num_samples, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                    value_loss = (
                        (self.critic(flat_input[indices]) - flat_targets[indices]) ** 2
                    ).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), 1.0, error_if_nonfinite=self.cfg.strict_finite
                    )
                    self.opt_v.step()
                    self.update_counts["critic"] += 1

        imagined_reward = (reward_sum / self.cfg.horizon / self.mdp.max_step_reward).mean()
        critic_timing.__exit__(None, None, None)
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * 0.03
        return self.cfg.lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        )

    def actor_scale_at(self, iteration: int) -> float:
        start = self.cfg.wm_only_iters + self.cfg.critic_warmup_iters
        if iteration < start:
            return 0.0
        ramp = min(1.0, (iteration - start + 1) / max(1, self.cfg.actor_ramp_iters))
        return self.cfg.actor_lr_scale * ramp

    def set_learning_rates(self, iteration: int) -> float:
        rate = self.lr_at(iteration)
        for optimizer, scale in (
            (self.opt_wm, 1.0),
            (self.opt_v, 1.0),
            (self.opt_pi, self.actor_scale_at(iteration)),
        ):
            for group in optimizer.param_groups:
                group["lr"] = rate * scale
        return rate

    def _checkpoint(self, include_optimizers: bool) -> dict[str, Any]:
        policy_state = (
            self.pi_avg.module.state_dict() if self.pi_avg is not None else self.pi.state_dict()
        )
        checkpoint = {
            "wm": self.wm.state_dict(),
            "pi": policy_state,
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "args": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.cfg.goal_mode == "direct":
            checkpoint["task_contract"] = self.env.supermimic_contract
        if self.pi_avg is not None:
            checkpoint["pi_live"] = self.pi.state_dict()
        if include_optimizers:
            checkpoint.update(
                {
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        if self.cfg.checkpoint_replay:
            checkpoint.update(
                extra_state=self.extra_state,
                update_counts=self.update_counts.copy(),
                step_accumulator=self.step_accumulator,
                value_normalization={
                    "mean": self._v_mean,
                    "std": self._v_std,
                    "initialized": self._v_init,
                },
                resume_semantics="weights_optimizers_replay_curriculum; fresh_environment; not_step_equivalent",
            )
        if include_optimizers and self.cfg.checkpoint_replay:
            checkpoint["replay"] = self.buffer.state_dict()
            checkpoint["torch_rng"] = torch.get_rng_state()
            if self.device.type == "cuda":
                checkpoint["cuda_rng"] = torch.cuda.get_rng_state(self.device)
            command = getattr(self.env, "command_manager", None)
            curriculum = command.get_term("velocity").curriculum if command else None
            if curriculum is not None:
                checkpoint["curriculum"] = curriculum.state_dict()
        return checkpoint

    def save(self, path: str | Path | None = None, include_optimizers: bool = True) -> Path:
        destination = Path(path) if path is not None else self.log_dir / "last.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint(include_optimizers), temporary)
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        required = {"critic", "opt_wm", "opt_pi", "opt_v"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"SuperMimic resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        checkpoint_args = checkpoint.get("args", {})
        if checkpoint_args.get("goal_mode", "encoded") != self.cfg.goal_mode:
            raise ValueError("Cannot resume with a different goal_mode")
        if self.cfg.goal_mode == "direct":
            if checkpoint.get("task_contract") != getattr(self.env, "supermimic_contract", None):
                raise ValueError(
                    "Cannot resume Go2 with changed state/action/reward/model/terrain semantics"
                )
            for key in (
                "hist",
                "goal_frames",
                "policy_hidden",
                "wm_hidden",
                "wm_structured",
                "action_noise",
            ):
                if checkpoint_args.get(key) != asdict(self.cfg)[key]:
                    raise ValueError(f"Cannot resume with changed {key}")
        for key, default in (
            ("terrain_training", False),
            ("terrain_conditioning", False),
            ("terrain_sensor_grad", True),
            ("critic_terrain_conditioning", False),
        ):
            if checkpoint_args.get(key, default) != getattr(self.cfg, key):
                raise ValueError(f"Cannot resume with changed {key}")
        checkpoint_structured = bool(checkpoint_args.get("wm_structured", False))
        if checkpoint_structured != self.cfg.wm_structured:
            raise ValueError(
                "Cannot resume with a different algo.wm_structured setting "
                f"(checkpoint={checkpoint_structured}, requested={self.cfg.wm_structured})"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        if "pi_live" in checkpoint:
            self.pi.load_state_dict(checkpoint["pi_live"])
        else:
            self.pi.load_state_dict(checkpoint["pi"])
        if self.pi_avg is not None:
            self.pi_avg.module.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opt_wm.load_state_dict(checkpoint["opt_wm"])
        self.opt_pi.load_state_dict(checkpoint["opt_pi"])
        self.opt_v.load_state_dict(checkpoint["opt_v"])
        self.iteration = checkpoint["iteration"]
        self.wall_time = checkpoint["wall_time"]
        self.sim_steps = checkpoint["sim_steps"]
        self.extra_state = checkpoint.get("extra_state", {})
        self.update_counts = checkpoint.get("update_counts", {"wm": 0, "actor": 0, "critic": 0})
        self.step_accumulator = checkpoint.get("step_accumulator", 0.0)
        if "value_normalization" in checkpoint:
            values = checkpoint["value_normalization"]
            self._v_mean, self._v_std, self._v_init = (
                values["mean"],
                values["std"],
                values["initialized"],
            )
        if "replay" in checkpoint:
            self.buffer.load_state_dict(checkpoint["replay"])
            # Physics and command-manager RNG are deliberately restarted. Close every
            # prior trajectory and clear partial curriculum episodes before new reset.
            self.buffer.mark_reset_boundary(
                torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
            )
            command = getattr(self.env, "command_manager", None)
            curriculum = command.get_term("velocity").curriculum if command else None
            if curriculum is not None and "curriculum" in checkpoint:
                curriculum.load_state_dict(checkpoint["curriculum"])
                curriculum.steps.zero_()
                curriculum.episode_fall.zero_()
            self.env.reset()
            self.obs_hist.zero_()
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if self.device.type == "cuda" and "cuda_rng" in checkpoint:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), self.device)
            self.resume_mode = (
                "weights_optimizers_replay_curriculum_restored; environment_resampled; not_exact"
            )
        else:
            self.resume_mode = (
                "weights_optimizers_only; replay_environment_curriculum_resampled; not_exact"
            )

    @staticmethod
    def _mean(values: list[torch.Tensor]) -> float:
        return float(sum(float(value) for value in values) / max(len(values), 1))

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        if resume is not None:
            self.load(resume)
        start_iteration = self.iteration
        self.stats = _empty_stats()
        logger = SuperMimicLogger(
            env_name=type(self.env).__name__,
            backend=self.backend,
            max_iterations=self.cfg.iterations,
            wm_only_iters=self.cfg.wm_only_iters,
            start_iteration=start_iteration,
            num_envs=self.env.num_envs,
            log_dir=str(self.log_dir),
            log_backend=self.cfg.log_backend,
        )
        logger.start(status="Warming up...")
        try:
            if start_iteration == 0:
                self.collect(self.cfg.warmup_steps)
            required_steps = max(self.cfg.hist, self.cfg.wm_window)
            if start_iteration > 0:
                required_steps = max(required_steps, self.cfg.warmup_steps)
            if self.buffer.size < required_steps:
                self.collect(required_steps - self.buffer.size)
            logger.log_buffer_fill(self.buffer.size, self.cfg.buffer_steps)
            logger.start_training_timer()

            log_path = self.log_dir / "train_log.csv"
            with log_path.open("a", encoding="utf-8", newline="") as log_file:
                if log_file.tell() == 0:
                    log_file.write(",".join(_LOG_FIELDS) + "\n")
                train_start = time.time() - self.wall_time
                log_start = time.time()

                aggregate: dict[str, list[torch.Tensor]] = {
                    key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")
                }
                last_metrics: dict[str, float | int] = {}
                for iteration in range(start_iteration, self.cfg.iterations):
                    if self.cfg.compile and iteration == start_iteration:
                        logger.log_status("Compiling world model (one-off, slow on first run)")
                    if self.cfg.compile and iteration == self.cfg.wm_only_iters:
                        logger.log_status("Compiling policy update (one-off, slow on first run)")
                    iteration_start = time.time()
                    learning_rate = self.set_learning_rates(iteration)
                    self.step_accumulator += 1.0 / self.cfg.utd
                    collect_steps = int(self.step_accumulator)
                    self.step_accumulator -= collect_steps
                    stats_mark = len(self.stats["r"])
                    self.collect(collect_steps)
                    train_phase_start = time.time()
                    wm_loss = torch.zeros((), device=self.device)
                    wm_error = torch.zeros((), device=self.device)
                    for _ in range(self.cfg.wm_batches):
                        wm_loss, wm_error = self.wm_update()
                        for name, metric in (("wm_loss", wm_loss), ("wm_err8", wm_error)):
                            if not torch.isfinite(metric).all():
                                raise FloatingPointError(
                                    f"non-finite {name} at iteration {iteration + 1}"
                                )
                    aggregate["wm"].append(wm_loss)
                    aggregate["wm8"].append(wm_error)
                    pi_loss: float | None = None
                    v_loss: float | None = None
                    imag_rew: float | None = None
                    pi_gradnorm: float | None = None
                    imag_survival: float | None = None
                    if iteration >= self.cfg.wm_only_iters:
                        policy_loss, value_loss, imagined_reward, grad_norm, survival = (
                            self.policy_update(update_actor=False)
                            if iteration < self.cfg.wm_only_iters + self.cfg.critic_warmup_iters
                            else self.policy_update()
                        )
                        # Fail fast on divergence instead of checkpointing a diverged model.
                        # ``grad_norm`` is deliberately excluded: ``no_sync`` training already
                        # handles a non-finite gradient by zeroing it and stepping, so a non-finite
                        # gradient is not a divergence signal on its own.
                        for name, metric in (
                            ("pi_loss", policy_loss),
                            ("v_loss", value_loss),
                            ("imag_rew", imagined_reward),
                        ):
                            if not torch.isfinite(metric).all():
                                raise FloatingPointError(
                                    f"non-finite {name} at iteration {iteration + 1}"
                                )
                        aggregate["pi"].append(policy_loss)
                        aggregate["v"].append(value_loss)
                        aggregate["ir"].append(imagined_reward)
                        aggregate["gn"].append(grad_norm)
                        aggregate["surv"].append(survival)
                        pi_loss = float(policy_loss)
                        v_loss = float(value_loss)
                        imag_rew = float(imagined_reward)
                        pi_gradnorm = float(grad_norm)
                        imag_survival = float(survival)
                    train_time = time.time() - train_phase_start
                    self.iteration = iteration + 1
                    self.wall_time = time.time() - train_start
                    iteration_time = time.time() - iteration_start

                    if self.on_iteration is not None:
                        self.on_iteration(self)
                    iter_rewards = self.stats["r"][stats_mark:]
                    mean_reward = self._mean(iter_rewards) if iter_rewards else None
                    ep_len = self._mean(self.stats["ep_len"]) if self.stats["ep_len"] else None
                    ep_ret_norm = self._mean(self.stats["ep_ret"]) if self.stats["ep_ret"] else None
                    total_done = sum(float(value) for value in self.stats["done"])
                    term_frac = (
                        sum(float(value) for value in self.stats["term"]) / total_done
                        if total_done > 0
                        else None
                    )
                    logger.log_step(
                        self.iteration,
                        wm_loss=float(wm_loss),
                        wm_err8=float(wm_error),
                        pi_loss=pi_loss,
                        v_loss=v_loss,
                        imag_rew=imag_rew,
                        imag_survival=imag_survival,
                        pi_gradnorm=pi_gradnorm,
                        lr=learning_rate,
                        sim_rew=mean_reward,
                        ep_len=ep_len,
                        ep_ret_norm=ep_ret_norm,
                        term_frac=term_frac,
                        iter_sim_steps=collect_steps * self.env.num_envs,
                        collect_time=train_phase_start - iteration_start,
                        learn_time=train_time,
                        iteration_time=iteration_time,
                    )

                    if self.iteration % self.cfg.log_every == 0:
                        iterations_per_second = self.cfg.log_every / (time.time() - log_start)
                        log_start = time.time()
                        row = [
                            self.iteration,
                            self.wall_time / 3600,
                            self.sim_steps,
                            self._mean(aggregate["wm"]),
                            self._mean(aggregate["wm8"]),
                            self._mean(aggregate["pi"]),
                            self._mean(aggregate["v"]),
                            self._mean(aggregate["ir"]),
                            self._mean(aggregate["gn"]),
                            self._mean(aggregate["surv"]),
                            self._mean(self.stats["r"]),
                            ep_len,
                            ep_ret_norm,
                            term_frac,
                            iterations_per_second,
                            learning_rate,
                        ]
                        log_file.write(
                            ",".join(
                                f"{value:.6g}" if isinstance(value, float) else str(value)
                                for value in row
                            )
                            + "\n"
                        )
                        log_file.flush()
                        last_metrics = dict(zip(_LOG_FIELDS, row, strict=True))
                        self.stats = _empty_stats()
                        aggregate = {key: [] for key in aggregate}

                    if (
                        self.iteration % self.cfg.ckpt_every == 0
                        or self.iteration == self.cfg.iterations
                        or self.stop_requested
                    ):
                        self.wall_time = time.time() - train_start
                        self.save()
                        if (
                            not self.cfg.numbered_ckpt_every
                            or self.iteration % self.cfg.numbered_ckpt_every == 0
                            or self.iteration == self.cfg.iterations
                            or self.stop_requested
                        ):
                            numbered = self.save(
                                self.log_dir / f"ckpt_{self.iteration:06d}.pt",
                                include_optimizers=self.cfg.checkpoint_replay,
                            )
                            logger.log_save(str(numbered))
                    if self.stop_requested:
                        break

            self.wall_time = time.time() - train_start
            logger.finish()
            return {
                **last_metrics,
                "completed": self.iteration >= self.cfg.iterations,
                "iteration": self.iteration,
                "sim_steps": self.sim_steps,
                "wall_time": self.wall_time,
            }
        finally:
            logger.close()


__all__ = ["SuperMimicConfig", "SuperMimicLearner"]
