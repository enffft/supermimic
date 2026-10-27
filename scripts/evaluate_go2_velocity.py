"""Evaluate Go2 velocity tracking with fixed body-frame commands in real MuJoCo."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.spec import get_robot_spec


COMMANDS = ((0.3, 0.0, 0.0), (0.5, 0.0, 0.0))


def evaluate_command(
    checkpoint: Path,
    command_value: tuple[float, float, float],
    *,
    device: str,
    num_envs: int,
    episodes: int,
) -> dict[str, float | list[float]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    env_cfg = config.get("env", {})
    if env_cfg.get("robot_name") != "go2" or env_cfg.get("task_name") != "velocity_command":
        raise ValueError("checkpoint must use go2/velocity_command")
    policy_cfg = config.get("policy", {})
    policy_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in policy_cfg.items()}
    policy = Policy(PolicyCfg(**policy_cfg)).to(device)
    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[],
            robot_name="go2",
            task_name="velocity_command",
            num_envs=num_envs,
            history_length=int(env_cfg.get("history_length", policy.cfg.history)),
            episode_length_s=float(env_cfg.get("episode_length_s", 10.0)),
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            record_transitions=True,
            replay_capacity=1000,
            velocity_gaussian_noise_std=0.0,
            velocity_ou_env_fraction=0.0,
            velocity_ou_noise_std=0.0,
            velocity_curriculum=False,
            velocity_fixed_command=command_value,
            velocity_fall_height=float(env_cfg.get("velocity_fall_height", 0.17)),
            velocity_fall_tilt_deg=float(env_cfg.get("velocity_fall_tilt_deg", 70.0)),
            seed=2026,
        ),
        device=device,
    )
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("velocity").command
        expected = command.new_tensor(command_value).expand_as(command)
        if not torch.allclose(command, expected):
            raise RuntimeError(f"fixed command mismatch after reset: {command[0].tolist()} != {command_value}")
        state_io = StateIO(env)
        recorder = env.recorder_manager.get_term("replay")
        dt = float(env.step_dt)
        nj = get_robot_spec("go2").num_joints
        target = torch.tensor(command_value, device=device, dtype=torch.float32)

        count = torch.zeros(num_envs, device=device)
        sum_vel = torch.zeros(num_envs, 2, device=device)
        sum_vel_sq_error = torch.zeros(num_envs, device=device)
        sum_yaw = torch.zeros(num_envs, device=device)
        sum_yaw_sq_error = torch.zeros(num_envs, device=device)
        sum_height = torch.zeros(num_envs, device=device)
        sum_action_abs = torch.zeros(num_envs, device=device)
        sum_action_saturated = torch.zeros(num_envs, device=device)
        episode_start = state_io.read()[:, :3].clone()
        rows: list[dict[str, float]] = []

        while len(rows) < episodes:
            with torch.no_grad():
                action = policy(obs["policy"], obs["goal"])
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            # Replay's terminal row preserves the final physical state before the
            # simulator resets a completed environment.
            state = state_io.read()
            ids = done.nonzero().squeeze(-1)
            if ids.numel():
                terminal_rows = (recorder.buffer.ptr[ids] - 1) % recorder.buffer.T
                state[ids] = recorder.buffer.s[ids, terminal_rows]

            root_quat = state[:, 3:7]
            lin_vel_world = state[:, 7 + nj:10 + nj]
            lin_vel_body = quat_apply_inverse(root_quat, lin_vel_world)[:, :2]
            yaw_rate = state[:, 12 + nj]
            action_term = env.action_manager.get_term("joint_pos")
            executed = action_term.executed_raw.clone()
            if ids.numel():
                transition_rows = (recorder.buffer.ptr[ids] - 2) % recorder.buffer.T
                executed[ids] = recorder.buffer.a_exec[ids, transition_rows]
            sum_vel += lin_vel_body
            sum_vel_sq_error += (lin_vel_body - target[:2]).square().sum(-1)
            sum_yaw += yaw_rate
            sum_yaw_sq_error += (yaw_rate - target[2]).square()
            sum_height += state[:, 2]
            sum_action_abs += executed.abs().mean(-1)
            sum_action_saturated += (executed.abs() >= 1.0 - 1e-6).float().mean(-1)
            count += 1

            if ids.numel() == 0:
                continue
            for env_id in ids.tolist():
                n = count[env_id].clamp_min(1.0)
                displacement = state[env_id, :3] - episode_start[env_id]
                rows.append({
                    "mean_vx": float(sum_vel[env_id, 0] / n),
                    "mean_vy": float(sum_vel[env_id, 1] / n),
                    "mean_yaw_rate": float(sum_yaw[env_id] / n),
                    "velocity_rmse": float((sum_vel_sq_error[env_id] / n).sqrt()),
                    "yaw_rate_rmse": float((sum_yaw_sq_error[env_id] / n).sqrt()),
                    "displacement_x": float(displacement[0]),
                    "displacement_y": float(displacement[1]),
                    "displacement_xy": float(torch.linalg.vector_norm(displacement[:2])),
                    "mean_base_height": float(sum_height[env_id] / n),
                    "mean_abs_executed_action": float(sum_action_abs[env_id] / n),
                    "action_saturation_fraction": float(sum_action_saturated[env_id] / n),
                    "fall": float(terminated[env_id]),
                    "timeout": float(truncated[env_id]),
                })
                count[env_id] = 0
                sum_vel[env_id] = 0
                sum_vel_sq_error[env_id] = 0
                sum_yaw[env_id] = 0
                sum_yaw_sq_error[env_id] = 0
                sum_height[env_id] = 0
                sum_action_abs[env_id] = 0
                sum_action_saturated[env_id] = 0
            reset_state = state_io.read()
            episode_start[ids] = reset_state[ids, :3]

        def mean(name: str) -> float:
            return sum(row[name] for row in rows[:episodes]) / max(len(rows[:episodes]), 1)

        return {
            "command_body_xy_yaw": list(command_value),
            "episodes": min(len(rows), episodes),
            "mean_vx": mean("mean_vx"),
            "mean_vy": mean("mean_vy"),
            "mean_yaw_rate": mean("mean_yaw_rate"),
            "velocity_rmse": mean("velocity_rmse"),
            "yaw_rate_rmse": mean("yaw_rate_rmse"),
            "mean_displacement_xy": mean("displacement_xy"),
            "mean_displacement_x": mean("displacement_x"),
            "mean_base_height": mean("mean_base_height"),
            "mean_abs_executed_action": mean("mean_abs_executed_action"),
            "action_saturation_fraction": mean("action_saturation_fraction"),
            "fall_rate": mean("fall"),
            "timeout_rate": mean("timeout"),
        }
    finally:
        env.close()


def main(
    checkpoint: Path,
    device: str = "cuda:0",
    num_envs: int = 64,
    episodes_per_command: int = 64,
    output: Path | None = None,
) -> None:
    if num_envs < 1 or episodes_per_command < 1:
        raise ValueError("num_envs and episodes_per_command must be positive")
    results = {
        "checkpoint": str(checkpoint),
        "real_mujoco": [
            evaluate_command(
                checkpoint, command, device=device,
                num_envs=num_envs, episodes=episodes_per_command,
            )
            for command in COMMANDS
        ],
    }
    payload = json.dumps(results, indent=2)
    print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")


if __name__ == "__main__":
    tyro.cli(main)
