"""Prior-free M1 task adapter for the existing upstream mjlab/RSL-RL PPO.

Only environment/Actor contracts, checkpoint export and budget handling live
here. Rollouts, GAE and PPO optimization remain upstream.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from supermimic.baselines.go2_ppo import (
    ROOT, UPSTREAM, FixedScale, ComparisonRunner, BudgetReached, runner_config,
)
from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.state_io import StateIO
from supermimic.envs.sim_context import SimContext
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.action_contract import serialize_m1_action_contract
from supermimic.tasks.m1_reward import M1RewardCfg, M1TargetCfg, m1_actor_goal
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

REFERENCE = ROOT / "logs/m1_reward/lm50_2048_2p5h_seed2026/ckpt_latest.pt"


def policy_config(saved):
    cfg = PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v
                       for k, v in saved.items()})
    if (cfg.history, cfg.num_joints, cfg.goal_dim, cfg.goal_interface,
            cfg.goal_encoding, cfg.action_output) != (10, 22, 8, "m1_reward", "direct", "tanh"):
        raise ValueError("Requires M1 reward-only 10-frame / Goal8 / Action22 direct tanh Actor")
    return cfg


class AlignedM1Actor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, policy_cfg, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        policy = Policy(policy_config(policy_cfg))
        if output_dim != 22:
            raise ValueError("M1 requires 22 actions")
        self.mlp = nn.Sequential(
            FixedScale(torch.cat((policy.decoder.scale, policy.decoder.goal_scale))),
            policy.decoder.mlp.net, nn.Tanh(),
        )


def env_args(config, num_envs, seed, task_type="balanced"):
    saved = config["env"]
    if (saved["robot_name"], saved["task_name"], saved["m1_action_contract"]) != ("m1_piper", "m1_reward", "v3"):
        raise ValueError("Requires m1_piper/m1_reward/v3 reference")
    values = {f.name: saved[f.name] for f in fields(SuperMimicEnvCfgArgs) if f.name in saved}
    # Keep the recorder for exact pre-reset reward state/command snapshots.
    # Its unused WM replay can be tiny: PPO has its own upstream rollout storage.
    values.update(npz_files=[], num_envs=num_envs, seed=seed, record_transitions=True,
                  replay_capacity=2, m1_joint_task_type=task_type,
                  m1_joint_balanced_fixed_quota=task_type == "balanced")
    values["m1_reward"] = M1RewardCfg(**saved["m1_reward"])
    values["m1_target"] = M1TargetCfg(**saved.get("m1_target", {}))
    return SuperMimicEnvCfgArgs(**values)


def critic_observation(env, state_spec_cfg, pose_profile):
    if not hasattr(env, "_m1_ppo_critic"):
        spec = get_robot_spec("m1_piper", m1_pose_profile=pose_profile,
                              m1_action_contract="v3")
        env._m1_ppo_critic = (
            StateIO(env), SimContext(env, spec.foot_geom_regex, spec.root_body_name),
            StateSpec(StateSpecCfg(**state_spec_cfg), env.device),
        )
    io, context, state_spec = env._m1_ppo_critic
    state = io.read()
    goal = m1_actor_goal(state, env.command_manager.get_term("m1_prior_joint").goal)
    return torch.cat((state_spec.canonical(state), goal,
                      state_spec.normalize_xi(context.read()), state.new_zeros((len(state), 4))), -1)


def ppo_config(config, num_envs, steps=32, minibatch=1024, seed=2026):
    cfg = runner_config(num_envs, steps, minibatch)
    cfg.seed = seed
    cfg.save_interval = 10
    cfg.actor.class_name = "supermimic.baselines.m1_ppo:AlignedM1Actor"
    cfg.critic.hidden_dims = tuple(config["critic"]["hidden"])
    # A custom Actor kwarg is passed through upstream's model factory.
    result = asdict(cfg)
    result["actor"]["policy_cfg"] = config["policy"]
    return result


def inference_checkpoint(actor_state, config):
    cfg = policy_config(config["policy"])
    policy = Policy(cfg)
    split = cfg.history * (3 * cfg.num_joints + 6)
    state = {"decoder.scale": actor_state["mlp.0.scale"][:split],
             "decoder.goal_scale": actor_state["mlp.0.scale"][split:]}
    state.update({"decoder.mlp.net." + k.removeprefix("mlp.1."): v
                  for k, v in actor_state.items() if k.startswith("mlp.1.")})
    policy.load_state_dict(state, strict=True)
    return {"policy": policy.state_dict(), "config": config, "source_algorithm": "PPO",
            "inference_only": True, "actuator_model_version": config["actuator_model_version"]}


class M1Runner(ComparisonRunner):
    def save(self, path, infos=None):
        command = self.env.unwrapped.command_manager.get_term("m1_prior_joint")
        temporary = str(path) + ".tmp"
        MjlabOnPolicyRunner.save(self, temporary, {**(infos or {}), "m1_command": command.state_dict(),
                                           "comparison_training_wall_seconds": self.elapsed_training_seconds()})
        Path(temporary).replace(path)
        elapsed = self.elapsed_training_seconds()
        out = Path(path).parent
        actor = self.alg.get_policy()
        torch.jit.script(actor.as_jit().cpu()).save(str(out / "actor_latest.pt"))
        inference = inference_checkpoint(actor.state_dict(), self.inference_config)
        inference["source_checkpoint"] = str(Path(path).resolve())
        # Existing play/evaluation consumes this inference-only SM schema.
        torch.save(inference, out / "policy_inference.pt")
        (out / "progress.json").write_text(json.dumps({
            "completed_ppo_updates": self.current_learning_iteration + 1,
            "real_env_steps_per_env": self.env.unwrapped.common_step_counter,
            "real_transitions": self.env.unwrapped.common_step_counter * self.env.num_envs,
            "training_wall_seconds": elapsed, "checkpoint": str(Path(path).resolve()),
        }, indent=2) + "\n")
        if self.comparison_max_seconds > 0 and elapsed >= self.comparison_max_seconds:
            raise BudgetReached()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--hours", type=float, default=3.5)
    p.add_argument("--task-version", choices=("v1", "v2"), default="v2")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--task-type", choices=("balanced", "L", "M"), default="balanced")
    p.add_argument("--steps-per-env", type=int, default=32)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    if a.num_envs < 2 or a.hours <= 0 or a.iterations < 1 or (a.task_type == "balanced" and a.num_envs % 2):
        p.error("positive budget/iterations; >=2 envs, balanced requires even env count")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    ref = torch.load(a.reference, map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(ref["config"])
    del ref  # Reference supplies configuration only, never learned weights.
    policy_config(config["policy"])
    spec = get_robot_spec("m1_piper", m1_pose_profile=config["env"]["m1_pose_profile"], m1_action_contract="v3")
    if config["m1_reward_action_mapping"] != serialize_m1_action_contract(spec):
        raise ValueError("Reference action/pose contract differs from current robot")
    if hashlib.sha256(spec.model_path.read_bytes()).hexdigest() != config["robot_model_sha256"]:
        raise ValueError("Reference robot model SHA differs from current robot")
    if a.task_version == "v2":
        from supermimic.training.m1_reward import reward_config
        shared = reward_config(a.log_dir, num_envs=a.num_envs, hours=a.hours, seed=a.seed, task_type=a.task_type)
        config["env"] = asdict(shared.env)
    ecfg_args = env_args(config, a.num_envs, a.seed, a.task_type)
    cfg = ppo_config(config, a.num_envs, a.steps_per_env, a.minibatch, a.seed)
    cfg["max_iterations"] = a.iterations
    out = a.log_dir.resolve()
    if a.resume is None and any(out.glob("*.pt")):
        raise FileExistsError(f"Checkpoints already exist in {out}")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "mjlab / RSL-RL PPO (not reimplemented)", "motion_prior_used": False,
        "world_model_used": False, "fresh_initialization": a.resume is None,
        "resume_from": str(a.resume) if a.resume is not None else None,
        "upstream_repo": "https://github.com/giangdao1402/go2_velocity",
        "upstream_sha": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "mjlab", "rsl-rl-lib")},
        "reference_checkpoint": str(a.reference.resolve()),
        "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "reference_config": config, "env_args": asdict(ecfg_args), "runner_config": cfg,
        "budget": {"primary": "equal fresh training wall time", "hours": a.hours,
                   "excludes": "initialization and post-training evaluation",
                   "stop_boundary": "next completed PPO update",
                   "note": "PPO updates/real samples differ from SuperMimic; report separately"},
        "environment_changes": {"replay_capacity": "256 -> 2, unused WM storage only"},
    }
    (out / "comparison_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    if a.plan_only:
        print(json.dumps(metadata["budget"], indent=2))
        return
    config["env"].update(asdict(ecfg_args))
    config["seed"] = a.seed
    config["run_name"], config["log_root"] = out.name, str(out.parent)
    ecfg = make_env_cfg(ecfg_args)
    ecfg.observations["critic"] = ObservationGroupCfg(
        terms={"privileged": ObservationTermCfg(func=critic_observation,
               params={"state_spec_cfg": config["world_model"]["state_spec"],
                       "pose_profile": ecfg_args.m1_pose_profile})},
        concatenate_terms=True, enable_corruption=False,
    )
    worker_started = time.monotonic()
    env = ManagerBasedRlEnv(ecfg, device=a.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = M1Runner(wrapped, cfg, log_dir=str(out), device=a.device)
        runner.inference_config = config
        runner.comparison_started = time.monotonic()
        runner.comparison_max_seconds = a.hours * 3600
        if a.resume is not None:
            runner.resume_training(a.resume, "m1_prior_joint")
        runner.install_budget_hook()
        try:
            if runner.elapsed_training_seconds() >= runner.comparison_max_seconds:
                # Recreate exports if a process died after the native save.
                runner.save(str(a.resume))
                status = "wall_clock_cap_already_reached"
            else:
                runner.learn(a.iterations, init_at_random_ep_len=False)
                status = "iteration_cap_reached"
        except BudgetReached:
            runner.logger.stop_logging_writer()
            status = "wall_clock_cap_reached"
        command = env.command_manager.get_term("m1_prior_joint")
        cache = env._supermimic_m1_reward_cache
        expected_reward = sum(cache["terms"].values()) * env.step_dt
        reward_error = float((expected_reward - env.reward_buf).abs().max())
        inference = torch.load(out / "policy_inference.pt", map_location="cpu", weights_only=False)
        exported_policy = Policy(policy_config(config["policy"])).to(a.device)
        exported_policy.load_state_dict(inference["policy"], strict=True)
        jit = torch.jit.load(str(out / "actor_latest.pt"), map_location=a.device)
        observation = wrapped.get_observations()
        with torch.no_grad():
            expected_action = runner.alg.get_policy()(observation)
            export_error = float((exported_policy(observation["policy"], observation["goal"]) - expected_action).abs().max())
            jit_error = float((jit(torch.cat((observation["policy"], observation["goal"]), -1)) - expected_action).abs().max())
        validation = {"L_envs": int((command.task_type == 0).sum()), "M_envs": int((command.task_type == 1).sum()),
                      "reward_dt_integration_max_error": reward_error,
                      "inference_action_max_error": export_error, "torchscript_action_max_error": jit_error,
                      "motion_prior_used": False, "world_model_used": False}
        (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        if reward_error > 1e-5 or export_error > 2e-6 or jit_error > 2e-6:
            raise RuntimeError(f"Reward/export validation failed: {validation}")
        (out / "completion.json").write_text(json.dumps({
            "status": status, "algorithm": "PPO", "task": "m1_reward",
            "training_wall_seconds": runner.elapsed_training_seconds(),
            "worker_wall_seconds_including_init": time.monotonic() - worker_started,
            **json.loads((out / "progress.json").read_text()),
        }, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
