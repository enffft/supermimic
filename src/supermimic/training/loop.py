"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.tasks import MotionTrackingTask, VelocityCommandTask
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        robot_spec = get_robot_spec(cfg.env.robot_name)
        is_motion = cfg.env.task_name == "motion_tracking"
        if cfg.env.task_name not in ("motion_tracking", "velocity_command"):
            raise ValueError(f"unknown task {cfg.env.task_name!r}")
        if cfg.env.task_name == "velocity_command" and robot_spec.name != "go2":
            raise ValueError("velocity_command is currently supported only for Go2")
        if robot_spec.name == "go2" and is_motion:
            raise ValueError("Go2 has no Motion Library; select env.task_name=velocity_command")
        if robot_spec.name != "go2" and not is_motion:
            raise ValueError("velocity_command is currently supported only for Go2")
        if cfg.env.anchor_obs is None:
            cfg.env.anchor_obs = robot_spec.name == "m1_piper"

        # Build every network dimension from the selected robot/task while leaving the
        # default G1 values and initialization path unchanged.
        n_joints = robot_spec.num_joints
        goal_term_sizes = (robot_spec.amp_dim, robot_spec.amp_dim, 3, 3, 3, 1) if is_motion else (3,)
        goal_term_scale = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0) if is_motion else (1.0,)
        goal_dim = cfg.env.future_frames * (2 * robot_spec.amp_dim + 10) + (9 if cfg.env.anchor_obs else 0) if is_motion else 3
        cfg.policy.num_joints = n_joints
        cfg.policy.history = cfg.env.history_length
        cfg.policy.goal_dim = goal_dim
        cfg.policy.goal_interface = "motion" if is_motion else "velocity"
        cfg.policy.goal_term_sizes = goal_term_sizes
        cfg.policy.goal_term_scale = goal_term_scale
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        cfg.world_model.num_joints = n_joints
        cfg.world_model.action_dim = n_joints
        cfg.world_model.xi_dim = n_joints + 4
        cfg.world_model.termination_base_height = robot_spec.default_base_height
        cfg.world_model.state_spec.num_joints = n_joints
        cfg.world_model.state_spec.dt = robot_spec.control_dt
        cfg.world_model.state_spec.wheel_indices = tuple(robot_spec.joint_names.index(name) for name in robot_spec.wheel_joint_names)
        cfg.world_model.state_spec.xi_offset = (0.75,) + (0.0,) * n_joints + (0.0,) * 3
        cfg.world_model.state_spec.xi_scale = (0.45,) + (0.01,) * n_joints + (0.05,) * 3

        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if not is_motion:
            files = []
            if cfg.data.library is not None:
                raise ValueError("Go2 velocity-command task does not consume a motion library")
        elif cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            npz_dir = cfg.data.npz_dir
            if robot_spec.name == "m1_piper" and npz_dir == Path("data/npz/lafan1"):
                from supermimic.data.m1_motion_convert import DEFAULT_BANK, DEFAULT_OUTPUT, convert_m1_motion_bank

                npz_dir = DEFAULT_OUTPUT
                source_names = sorted(p.stem for p in (DEFAULT_BANK / "motions").glob("*.txt"))
                selected = [name for name in source_names if cfg.data.clips is None or name in set(cfg.data.clips)]
                if cfg.data.limit is not None:
                    selected = selected[: cfg.data.limit]
                if any(not (npz_dir / f"{name}.npz").exists() for name in selected):
                    convert_m1_motion_bank(DEFAULT_BANK, npz_dir, names=selected)
                cfg.data.npz_dir = npz_dir
                cfg.data.manifest = None
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if is_motion and not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot_name=robot_spec.name,
                task_name=cfg.env.task_name,
                num_envs=cfg.env.num_envs,
                history_length=cfg.env.history_length,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        command_name = "motion" if is_motion else "velocity"
        self.command = env.command_manager.get_term(command_name)
        self.lib = getattr(self.command, "library", None)
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_robot_spec_model(robot_spec.name), device=self.device)
        heads = AnalyticHeads(
            fk, robot_spec.tracked_body_names, robot_spec.anchor_body_name, robot_spec.root_body_name,
            robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
            default_base_height=robot_spec.default_base_height,
        )
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, feat_dim=self.wm.spec.feat_dim, goal_dim=goal_dim, xi_dim=n_joints + 4, pcfg=cfg.policy).to(self.device)
        if is_motion:
            self.task = MotionTrackingTask(self.lib, cfg.env.future_frames, cfg.env.anchor_obs)
        else:
            self.task = VelocityCommandTask(
                debug_objective=cfg.env.velocity_debug_objective,
                default_joint_pos=robot.data.default_joint_pos[0].detach().clone(),
                dt=robot_spec.control_dt,
            )
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.task, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward if is_motion else 1.0
        self.has_tracking_return = is_motion
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        state = {
            "iteration": self.iteration,
            "num_updates": self.num_updates,
            "wm": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "config": asdict(self.cfg),
        }
        if hasattr(self.command, "bin_return"):
            state["bin_return"] = self.command.bin_return
            state["bin_memory"] = self.command.bin_memory
        torch.save(state, self.log_dir / name)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if hasattr(self.command, "bin_return") and "bin_return" in ck:
            self.command.bin_return.copy_(ck["bin_return"])
            self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                if self.has_tracking_return:
                    norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                    finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    m_pi = {}
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    if sch.policy_updates_per_wm_update:
                        torch.cuda.synchronize()
                        timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    if self.cfg.env.task_name == "motion_tracking":
                        self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    if "sampling_entropy" in self.command.metrics:
                        self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    if hasattr(self.command, "bin_return"):
                        self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
