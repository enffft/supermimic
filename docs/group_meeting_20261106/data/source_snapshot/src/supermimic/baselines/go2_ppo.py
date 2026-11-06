"""Go2 comparison adapter: existing SuperMimic task + upstream mjlab/RSL-RL PPO.

PPO, GAE, rollout storage, optimization, logging and checkpoints are upstream.
The downloaded giangdao1402/go2_velocity supplies the PPO hyperparameters; its
older combined actor-critic config is translated to the installed mjlab API.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.state_io import StateIO
from supermimic.envs.sim_context import SimContext
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.go2_gait import Go2GaitCfg
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt"
UPSTREAM = ROOT / "external/go2_velocity"


class FixedScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", scale)

    def forward(self, x):
        return x * self.scale


class AlignedActor(MLPModel):
    """Upstream Gaussian model with exactly the SuperMimic deterministic mean."""
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        policy = Policy(PolicyCfg(
            history=5, num_joints=12, goal_dim=3, goal_interface="velocity",
            goal_encoding="direct", action_output="tanh",
            goal_term_sizes=(3,), goal_term_scale=(1.0,),
        ))
        self.mlp = nn.Sequential(
            FixedScale(torch.cat((policy.decoder.scale, policy.decoder.goal_scale))),
            policy.decoder.mlp.net, nn.Tanh(),
        )


def critic_observation(env):
    # Same privileged canonical state, goal, xi and zero root-error inputs as
    # the velocity SuperMimic critic. No privileged features reach the Actor.
    if not hasattr(env, "_ppo_critic_io"):
        spec = get_robot_spec("go2")
        env._ppo_critic_io = (
            StateIO(env), SimContext(env, spec.foot_geom_regex, spec.root_body_name),
            StateSpec(StateSpecCfg(num_joints=12), env.device),
        )
    state_io, context, state_spec = env._ppo_critic_io
    s = state_io.read()
    return torch.cat((state_spec.canonical(s), env.command_manager.get_term("velocity").command,
                      state_spec.normalize_xi(context.read()), s.new_zeros((len(s), 4))), -1)


def env_args(reference_config, num_envs, seed=2026, evaluation=False):
    names = {f.name for f in fields(SuperMimicEnvCfgArgs)}
    kw = {k: v for k, v in reference_config["env"].items() if k in names}
    kw.update(npz_files=[], num_envs=num_envs, seed=seed, record_transitions=False)
    kw["go2_gait"] = Go2GaitCfg(**kw["go2_gait"])
    if evaluation:
        kw.update(domain_randomization=False, obs_noise=False, reset_noise=False,
                  velocity_gaussian_noise_std=0., velocity_ou_env_fraction=0.,
                  velocity_ou_noise_std=0., action_noise_std=0.)
    return SuperMimicEnvCfgArgs(**kw)


def upstream_ppo_config():
    source = UPSTREAM / "src/go2_velocity/rl_cfg.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "RslRlPpoAlgorithmCfg")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def runner_config(num_envs, steps=32, minibatch=1024):
    batch = num_envs * steps
    if min(num_envs, steps, minibatch) <= 0 or batch % minibatch:
        raise ValueError("rollout batch must divide into exact logical minibatches")
    algo = upstream_ppo_config()
    algo.update(num_mini_batches=batch // minibatch, learning_rate=2e-4)
    cfg = RslRlOnPolicyRunnerCfg(
        seed=2026, num_steps_per_env=steps, logger="tensorboard", upload_model=False,
        save_interval=10, obs_groups={"actor": ("policy", "goal"), "critic": ("critic",)},
        actor=RslRlModelCfg(
            class_name="supermimic.baselines.go2_ppo:AlignedActor",
            hidden_dims=(2048, 1024, 512, 256), activation="swish", obs_normalization=False,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                              "std_type": "log"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128), activation="swish", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(**algo),
    )
    return cfg


class BudgetReached(Exception):
    pass


class ComparisonRunner(MjlabOnPolicyRunner):
    """Only checkpoint metadata/export and wall-clock stop; learning is upstream."""
    def elapsed_training_seconds(self):
        return getattr(self, "comparison_elapsed_before", 0.) + time.monotonic() - self.comparison_started

    def resume_training(self, path, command_name):
        infos = self.load(str(path), map_location=self.device)
        self.current_learning_iteration += 1  # native checkpoint stores last completed index
        infos = infos or {}
        self.comparison_elapsed_before = float(infos.get("comparison_training_wall_seconds", 0.))
        if not self.comparison_elapsed_before:
            progress = Path(path).parent / "progress.json"
            if progress.exists():
                saved = json.loads(progress.read_text())
                if Path(saved["checkpoint"]).resolve() == Path(path).resolve():
                    self.comparison_elapsed_before = float(saved["training_wall_seconds"])
        command = self.env.unwrapped.command_manager.get_term(command_name)
        key = "velocity_command" if command_name == "velocity" else "m1_command"
        if key in infos and hasattr(command, "load_state_dict"):
            command.load_state_dict(infos[key])

    def install_budget_hook(self):
        # Upstream logs after every complete PPO update. Use that existing boundary
        # without copying its learn(), rollout collection, GAE or optimizer loop.
        log = self.logger.log

        def log_and_check_budget(**kwargs):
            log(**kwargs)
            if (self.comparison_max_seconds > 0
                    and self.elapsed_training_seconds() >= self.comparison_max_seconds):
                self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))

        self.logger.log = log_and_check_budget

    def save(self, path, infos=None):
        command = self.env.unwrapped.command_manager.get_term("velocity")
        temporary = str(path) + ".tmp"
        super().save(temporary, {**(infos or {}), "velocity_command": command.state_dict(),
                            "comparison_training_wall_seconds": self.elapsed_training_seconds()})
        Path(temporary).replace(path)
        elapsed = self.elapsed_training_seconds()
        progress = {
            "completed_ppo_updates": self.current_learning_iteration + 1,
            "real_env_steps_per_env": self.env.unwrapped.common_step_counter,
            "real_transitions": self.env.unwrapped.common_step_counter * self.env.num_envs,
            "training_wall_seconds": elapsed,
            "checkpoint": str(Path(path).resolve()),
        }
        (Path(path).parent / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        # Always keep an exportable latest snapshot for deployment/evaluation.
        torch.jit.script(self.alg.get_policy().as_jit().cpu()).save(str(Path(path).parent / "actor_latest.pt"))
        if self.comparison_max_seconds > 0 and elapsed >= self.comparison_max_seconds:
            raise BudgetReached()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--num-envs", type=int, default=3072)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--max-hours", type=float, default=2.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/go2_ppo/comparison_20261104")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    ckpt = torch.load(args.reference, map_location="cpu", weights_only=False, mmap=True)
    expected = PolicyCfg(history=5, num_joints=12, goal_dim=3, goal_interface="velocity",
                         goal_encoding="direct", action_output="tanh", goal_term_sizes=(3,), goal_term_scale=(1.,))
    for name in ("history", "num_joints", "goal_dim", "goal_interface", "goal_encoding", "action_output",
                 "decoder_hidden", "obs_term_scale", "goal_term_scale", "goal_term_sizes", "decoder_last_gain"):
        saved = ckpt["config"]["policy"][name]
        wanted = getattr(expected, name)
        if (tuple(saved) if isinstance(saved, list) else saved) != wanted:
            raise ValueError(f"reference actor {name}={saved!r} does not match aligned PPO actor {wanted!r}")
    cfg = runner_config(args.num_envs, args.steps_per_env, args.minibatch)
    cfg.max_iterations = args.iterations
    ecfg_args = env_args(ckpt["config"], args.num_envs)
    out = args.log_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "mjlab MjlabOnPolicyRunner / RSL-RL PPO (not reimplemented)",
        "upstream_repo": "https://github.com/giangdao1402/go2_velocity",
        "upstream_sha": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "reference_checkpoint": str(args.reference.resolve()),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "reference_config": ckpt["config"], "env_args": asdict(ecfg_args),
        "runner_config": asdict(cfg), "max_hours": args.max_hours,
        "budget": {"primary": "fresh initialization, equal training wall clock (2.5h each)",
                   "ceiling_steps_per_env": args.iterations * args.steps_per_env,
                   "ceiling_real_transitions": args.iterations * args.steps_per_env * args.num_envs,
                   "wall_clock_cap_hours": args.max_hours,
                   "note": "Stop at the next completed PPO update; periodic saves every 10 updates. SM env iteration != PPO optimizer iteration; report actual samples separately. Reference supplies configuration only; no weights loaded."},
    }
    (out / "comparison_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    if args.plan_only:
        print(json.dumps(metadata["budget"], indent=2))
        return
    ecfg = make_env_cfg(ecfg_args)
    ecfg.observations["critic"] = ObservationGroupCfg(
        terms={"privileged": ObservationTermCfg(func=critic_observation)},
        concatenate_terms=True, enable_corruption=False,
    )
    env = ManagerBasedRlEnv(ecfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = ComparisonRunner(wrapped, asdict(cfg), log_dir=str(out), device=args.device)
    runner.comparison_started = time.monotonic()
    runner.comparison_max_seconds = args.max_hours * 3600
    if args.resume is not None:
        runner.resume_training(args.resume, "velocity")
    runner.install_budget_hook()
    try:
        if runner.elapsed_training_seconds() >= runner.comparison_max_seconds:
            status = "wall_clock_cap_already_reached"
        else:
            runner.learn(args.iterations, init_at_random_ep_len=False)
            status = "sample_budget_complete"
    except BudgetReached:
        status = "wall_clock_cap_reached"
        runner.logger.stop_logging_writer()
    finally:
        env.close()
    (out / "completion.json").write_text(json.dumps({"status": status, "wall_seconds": time.monotonic()-runner.comparison_started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
