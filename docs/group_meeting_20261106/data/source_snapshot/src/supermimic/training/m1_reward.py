"""Configuration and worker for the prior-free M1 SuperMimic experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from supermimic.config import TrainConfig


def reward_config(log_dir: Path, *, num_envs=2048, hours=3.5, seed=2026, task_type="balanced", task_version="v2"):
    if num_envs < 2 or hours <= 0:
        raise ValueError("num_envs >= 2 and hours > 0 are required")
    if task_type not in ("balanced", "L", "M"):
        raise ValueError("task_type must be balanced, L or M")
    if task_type == "balanced" and num_envs % 2:
        raise ValueError("balanced training needs an even environment count")
    if task_version not in ("v1", "v2"):
        raise ValueError("task_version must be v1 or v2")
    cfg = TrainConfig()
    cfg.run_name, cfg.log_root, cfg.seed = log_dir.name, log_dir.parent, seed
    cfg.env.robot_name, cfg.env.task_name = "m1_piper", "m1_reward"
    cfg.env.num_envs = num_envs
    cfg.env.m1_pose_profile = "dataset_v3_3_l_high_carry"
    cfg.env.m1_action_contract = "v3"
    cfg.env.anchor_obs = False
    cfg.env.m1_joint_task_type = task_type
    cfg.env.m1_joint_l_task_fraction = 0.5
    cfg.env.m1_joint_balanced_fixed_quota = task_type == "balanced"
    cfg.env.obs_noise = cfg.env.reset_noise = False
    cfg.env.action_noise_std = 0.03
    cfg.env.domain_randomization = True
    cfg.env.replay_capacity = 256
    cfg.env.history_length = 10
    cfg.env.episode_length_s = 10.0
    cfg.resume = cfg.init_wm_from = None
    cfg.m1_joint_training.enabled = False
    cfg.schedule.max_hours = hours
    cfg.schedule.total_iterations = 1_000_000
    # A constant LR avoids treating unknown iterations/hour as a decay budget.
    cfg.schedule.lr_schedule = "constant"
    cfg.schedule.m1_wm_only_iterations = 300
    cfg.schedule.actor_ramp_iterations = 500
    cfg.schedule.checkpoint_interval = 10_000
    cfg.schedule.checkpoint_interval_minutes = 5.0
    cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = 128
    cfg.wm_trainer.micro_batch_size = cfg.policy_trainer.micro_batch_size = 32
    cfg.wm_trainer.batches_per_update = 4
    cfg.policy_trainer.obs_noise = False
    cfg.policy_trainer.horizon = 8 if task_version == "v1" else 32
    if task_version == "v2":
        from supermimic.tasks.m1_reward import M1TargetCfg
        cfg.env.m1_target = M1TargetCfg(sampling="base_disk", resample_on_success=True)
        cfg.env.m1_reward.reach_coarse_sigma = 1.5
        cfg.env.m1_reward.distance_time_weight = 0.5
        cfg.env.m1_reward.success_bonus = 12.0
        cfg.env.m1_reward.success_distance_m = cfg.env.m1_success_distance
        cfg.env.episode_length_s = 30.0
        cfg.policy_trainer.m1_terminal_distance_weight = 1.0
        # Preserve the existing logical/micro batches; verify H32 under the watchdog.
        cfg.policy_trainer.micro_batch_size = 32
    # Avoid many compiler subprocesses / giant FK compilation on a 16 GiB host.
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    return cfg


def evaluation_args(config, *, num_envs=16, task_type="balanced", seed=123):
    """Restore the saved physical/command contract, disabling training noise."""
    from dataclasses import fields
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.tasks.m1_reward import M1RewardCfg, M1TargetCfg

    saved = config["env"]
    if saved.get("task_name") != "m1_reward" or saved.get("robot_name") != "m1_piper":
        raise ValueError("Requires a m1_piper/m1_reward checkpoint")
    values = {f.name: saved[f.name] for f in fields(SuperMimicEnvCfgArgs) if f.name in saved}
    values.update(npz_files=[], num_envs=num_envs, obs_noise=False, reset_noise=False,
                  action_noise_std=0., domain_randomization=False, record_transitions=True,
                  replay_capacity=64, seed=seed, m1_joint_task_type=task_type,
                  m1_joint_balanced_fixed_quota=task_type == "balanced")
    values["m1_reward"] = M1RewardCfg(**saved.get("m1_reward", {}))
    values["m1_target"] = M1TargetCfg(**saved.get("m1_target", {}))
    return SuperMimicEnvCfgArgs(**values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--hours", type=float, default=3.5)
    p.add_argument("--iterations", type=int, help="Optional absolute environment-step cap for short validation")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--task-type", choices=("balanced", "L", "M"), default="balanced")
    p.add_argument("--task-version", choices=("v1", "v2"), default="v2")
    p.add_argument("--smoke", action="store_true", help="4 envs, short real WM and Actor updates")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--resume-total-budget", action="store_true", help="Resume hours is TOTAL budget, for queue recovery")
    p.add_argument("--resume", type=Path, help="Resume this task in a new output directory; hours is an additional budget")
    a = p.parse_args()
    cfg = reward_config(a.log_dir.resolve(), num_envs=4 if a.smoke else a.num_envs,
                        hours=a.hours, seed=a.seed, task_type=a.task_type, task_version=a.task_version)
    if a.resume is not None:
        if a.smoke:
            p.error("--resume and --smoke cannot be combined")
        import torch
        from supermimic.baselines.go2_supermimic import apply_config
        saved = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
        if saved.get("task_name") != "m1_reward":
            p.error("--resume requires a prior-free m1_reward checkpoint")
        cfg = apply_config(TrainConfig(), saved["config"])
        cfg.log_root, cfg.run_name = a.log_dir.resolve().parent, a.log_dir.name
        cfg.resume, cfg.init_wm_from = a.resume.resolve(), None
        cfg.schedule.max_hours = max(0., a.hours - float(saved.get("training_wall_seconds", 0.)) / 3600) if a.resume_total_budget else a.hours
        del saved
    if a.smoke:
        cfg.env.replay_capacity = 64
        cfg.env.domain_randomization = False
        cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = 4
        cfg.wm_trainer.micro_batch_size = cfg.policy_trainer.micro_batch_size = 4
        cfg.wm_trainer.batches_per_update = 1
        cfg.schedule.warmup_iterations = 12
        cfg.schedule.m1_wm_only_iterations = 2
        cfg.schedule.actor_ramp_iterations = 1
        cfg.schedule.total_iterations = 24
        cfg.schedule.log_interval = 12
    if a.iterations is not None:
        if a.iterations < 1:
            p.error("iterations must be positive")
        cfg.schedule.total_iterations = a.iterations
    if a.plan_only:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return
    out = cfg.log_root / cfg.run_name
    if a.resume is None and out.exists() and any(out.glob("*.pt")):
        raise FileExistsError(f"Existing checkpoint in {out}; choose a new --tag")
    from supermimic.training.loop import Trainer
    import torch
    started = time.monotonic()
    trainer = Trainer(cfg)
    starting_iteration, starting_actor_updates = trainer.iteration, trainer.actor_updates
    try:
        trainer.run()
        if a.smoke:
            import torch
            from supermimic.motion_prior.features import task_goal8
            from supermimic.tasks.world_point_reward import tcp_position_world
            cache = trainer.env._supermimic_m1_reward_cache
            imagined_terms = trainer.task.reward_terms(
                cache["state_prev"], cache["state_next"], cache["context"],
                cache["action"], cache["previous_action"],
            )
            reward_error = max(float((imagined_terms[k] - v * trainer.env.step_dt).abs().max())
                               for k, v in cache["terms"].items())
            state = trainer.recorder.state_io.read()
            command = trainer.command
            goal = trainer.task.actor_goal(state, command.goal)
            goal_error = float((goal - task_goal8(state, command.task_type, command.velocity_command_body,
                                                 command.target_position_world)).abs().max())
            tcp = tcp_position_world(state, trainer.task.fk, trainer.task.tcp_body_indices)
            robot = trainer.env.scene["robot"]
            ids = [robot.body_names.index(n) for n in ("piper_x_link7", "piper_x_link8")]
            sim_tcp = robot.data.body_link_pos_w[:, ids].mean(1) - trainer.env.scene.env_origins
            fk_error = float((tcp - sim_tcp).abs().max())
            saved = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False, mmap=True)
            assert "motion_prior" not in saved and trainer.m1_joint_trainer is None
            assert trainer.actor_updates > 0 and trainer.num_updates > trainer.actor_updates
            assert reward_error < 1e-6 and goal_error < 1e-6 and fk_error < 1e-4
            validation = {"reward_real_imag_max_error": reward_error, "goal8_max_error": goal_error,
                          "tcp_fk_mujoco_max_error_m": fk_error,
                          "L_envs": int((command.task_type == 0).sum()), "M_envs": int((command.task_type == 1).sum()),
                          "prior_absent": True, "actor_updates": trainer.actor_updates}
            (out / "smoke_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        (out / "completion.json").write_text(json.dumps({
            "algorithm": "SuperMimic", "task": "m1_reward", "motion_prior_used": False,
            "resume_from": str(cfg.resume) if cfg.resume is not None else None,
            "iterations": trainer.iteration, "real_transitions": trainer.iteration * cfg.env.num_envs,
            "segment_real_transitions": (trainer.iteration - starting_iteration) * cfg.env.num_envs,
            "actor_updates": trainer.actor_updates, "wm_update_events": trainer.num_updates,
            "segment_actor_updates": trainer.actor_updates - starting_actor_updates,
            "worker_wall_seconds_including_init": time.monotonic() - started,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "checkpoint": str(out / "ckpt_latest.pt"),
        }, indent=2) + "\n")
    finally:
        trainer.env.close()


if __name__ == "__main__":
    main()
