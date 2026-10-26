"""Deterministically evaluate one M1 checkpoint on each selected reference clip.

This script only runs the policy in MuJoCo; it does not update model parameters.
Example: ``python -m supermimic.scripts.evaluate_m1 --checkpoint logs/run/ckpt_latest.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


def evaluate(checkpoint: Path, motion_set: str = "C", device: str = "cuda:0", seed: int = 0) -> dict:
    cfg = TrainConfig(run_name=f"eval_m1_{checkpoint.parent.name}", seed=seed, device=device)
    cfg.resume = checkpoint
    cfg.env.robot_name = "m1_piper"
    cfg.env.task_name = "motion_tracking"
    cfg.env.num_envs = {"A": 2, "B": 3, "C": 6}[motion_set.upper()]
    cfg.env.action_noise_std = 0.0
    cfg.env.obs_noise = False
    cfg.env.reset_noise = False
    cfg.env.domain_randomization = False
    cfg.env.sampling_mode = "start"
    cfg.env.episode_length_s = 120.0
    cfg.data.m1_motion_set = motion_set.upper()
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.compile = None
    trainer = Trainer(cfg)
    env = trainer.env
    command = trainer.command
    library = trainer.lib
    num_clips = library.num_clips
    if env.num_envs != num_clips:
        raise RuntimeError(f"expected one environment per selected clip, got {env.num_envs} envs / {num_clips} clips")

    # Pin env i to clip i. The command reset writes the exact reference state with no perturbation.
    command.forced_frame.copy_(library.clip_start)
    obs, _ = env.reset()
    active = torch.ones(num_clips, dtype=torch.bool, device=env.device)
    ep_return = torch.zeros(num_clips, device=env.device)
    ep_steps = torch.zeros(num_clips, device=env.device)
    ep_fall = torch.zeros(num_clips, dtype=torch.bool, device=env.device)
    accum = {k: torch.zeros(num_clips, device=env.device) for k in (
        "root_pos_error", "root_ori_error", "arm_joint_error", "tcp_pos_error",
    )}
    arm_ids = torch.tensor(
        [env.scene["robot"].joint_names.index(f"piper_x_joint{i}") for i in range(1, 7)],
        dtype=torch.long, device=env.device,
    )
    tcp_idx = library.body_names.index("piper_x_link7")
    from mjlab.utils.lab_api.math import quat_error_magnitude

    max_steps = int((library.clip_last - library.clip_start).max().item()) + 8
    try:
        for _ in range(max_steps):
            robot = env.scene["robot"]
            values = {
                "root_pos_error": torch.linalg.vector_norm(command.root_pos_w - command.robot_root_pos_w, dim=-1),
                "root_ori_error": quat_error_magnitude(command.root_quat_w, command.robot_root_quat_w),
                "arm_joint_error": torch.linalg.vector_norm(
                    command.joint_pos[:, arm_ids] - robot.data.joint_pos[:, arm_ids], dim=-1
                ) / arm_ids.numel() ** 0.5,
                "tcp_pos_error": torch.linalg.vector_norm(
                    command.body_pos_w[:, tcp_idx] - command.robot_body_pos_w[:, tcp_idx], dim=-1
                ),
            }
            ep_steps += active
            for key, value in values.items():
                accum[key] += value * active
            action = trainer.act(obs)
            obs, reward, terminated, timed_out, _ = env.step(action)
            ep_return += reward * active
            ep_fall |= terminated & active
            finished = active & (terminated | timed_out)
            active &= ~finished
            if not active.any():
                break

        result = {"checkpoint": str(checkpoint), "motion_set": motion_set.upper(), "motions": {}}
        for i, clip in enumerate(library.clips):
            count = max(float(ep_steps[i]), 1.0)
            mean = {key: float(value[i] / count) for key, value in accum.items()}
            success = (
                mean["root_pos_error"] < 0.10
                and mean["root_ori_error"] < 0.25
                and mean["arm_joint_error"] < 0.20
                and mean["tcp_pos_error"] < 0.08
                and not bool(ep_fall[i])
                and not bool(active[i])
            )
            result["motions"][clip.name] = {
                "return": float(ep_return[i]),
                "steps": int(ep_steps[i]),
                "completed": not bool(active[i]),
                "success": success,
                **mean,
            }
        completed = [x for x in result["motions"].values() if x["completed"]]
        result["success_rate"] = (
            sum(x["success"] for x in completed) / len(completed) if completed else None
        )
        return result
    finally:
        env.close()
        if trainer.writer is not None:
            trainer.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-set", choices=("A", "B", "C"), default="C")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.motion_set, args.device, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
