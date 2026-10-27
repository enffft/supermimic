"""Evaluate real MuJoCo performance of an M1 ``world_point_reach`` checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.commands import WorldPointCommand
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.tasks.world_point_reward import tcp_position_world


def evaluate(
    checkpoint: Path,
    device: str = "cuda:0",
    num_envs: int = 128,
    episodes: int = 256,
    debug_vis: bool = False,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    env_cfg = config.get("env", {})
    if env_cfg.get("robot_name") != "m1_piper" or env_cfg.get("task_name") != "world_point_reach":
        raise ValueError("checkpoint must be trained with m1_piper/world_point_reach")
    policy_cfg = config.get("policy", {})
    policy_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in policy_cfg.items()}
    policy = Policy(PolicyCfg(**policy_cfg)).to(device)
    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    args = SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="world_point_reach",
        num_envs=num_envs,
        history_length=int(env_cfg.get("history_length", policy.cfg.history)),
        future_frames=int(env_cfg.get("future_frames", 10)),
        episode_length_s=float(env_cfg.get("episode_length_s", 10.0)),
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=True, replay_capacity=256,
        debug_vis=debug_vis,
        m1_fall_height=float(env_cfg.get("m1_fall_height", 0.20)),
        m1_success_distance=float(env_cfg.get("m1_success_distance", 0.08)),
        m1_near_target_fraction=float(env_cfg.get("m1_near_target_fraction", 0.60)),
        m1_near_x=tuple(env_cfg.get("m1_near_x", (0.05, 0.30))),
        m1_near_y=tuple(env_cfg.get("m1_near_y", (-0.20, 0.20))),
        m1_near_z=tuple(env_cfg.get("m1_near_z", (-0.15, 0.15))),
        m1_far_x=tuple(env_cfg.get("m1_far_x", (1.00, 1.40))),
        m1_far_y=tuple(env_cfg.get("m1_far_y", (-0.40, 0.40))),
        m1_far_z=tuple(env_cfg.get("m1_far_z", (-0.10, 0.25))),
        seed=123,
    )
    env = make_env(args, device=device)
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("world_point")
        if not isinstance(command, WorldPointCommand):
            raise TypeError("environment did not construct WorldPointCommand")
        state_io = StateIO(env)
        spec = load_robot_spec_model("m1_piper")
        fk = G1Kinematics(spec, device=device)
        tcp_ids = (fk.body_index("piper_x_link7"), fk.body_index("piper_x_link8"))
        recorder = env.recorder_manager.get_term("replay")
        dt = float(env.step_dt)
        arm_slice = slice(7 + 16, 7 + 22)

        initial_state = state_io.read().clone()
        initial_target = command.target_position_world.clone()
        initial_tcp = tcp_position_world(initial_state, fk, tcp_ids)
        target_range = torch.linalg.vector_norm(initial_target - initial_tcp, dim=-1)
        step_count = torch.zeros(num_envs, device=device)
        hold = torch.zeros(num_envs, device=device)
        max_hold = torch.zeros(num_envs, device=device)
        success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        dist_sum = torch.zeros(num_envs, device=device)
        dist_count = torch.zeros(num_envs, device=device)
        episodes_done = torch.zeros(num_envs, dtype=torch.long, device=device)
        rows: list[dict[str, float]] = []
        fk_error_sum = 0.0
        fk_error_count = 0

        while len(rows) < episodes:
            state = state_io.read()
            target = command.target_position_world
            tcp = tcp_position_world(state, fk, tcp_ids)
            distance = torch.linalg.vector_norm(tcp - target, dim=-1)
            hold = torch.where(distance < command.cfg.success_distance, hold + dt, torch.zeros_like(hold))
            max_hold = torch.maximum(max_hold, hold)
            success |= hold >= 0.5
            dist_sum += distance
            dist_count += 1
            step_count += 1

            # Verify the differentiable FK against the simulator body frames.
            measured = env.scene["robot"].data.body_link_pos_w[:, command._tcp_body_ids].mean(dim=1)
            measured_local = measured - env.scene.env_origins
            fk_error_sum += float(torch.linalg.vector_norm(tcp - measured_local, dim=-1).sum())
            fk_error_count += num_envs

            with torch.no_grad():
                action = policy(obs["policy"], obs["goal"])
            obs, reward, terminated, truncated, extras = env.step(action)
            done = terminated | truncated
            ids = done.nonzero().squeeze(-1)
            if ids.numel():
                for env_id in ids.tolist():
                    terminal_idx = int((recorder.buffer.ptr[env_id] - 1) % recorder.buffer.T)
                    final_state = recorder.buffer.s[env_id, terminal_idx]
                    final_target = recorder.buffer.goal[env_id, terminal_idx]
                    final_tcp = tcp_position_world(final_state[None], fk, tcp_ids)[0]
                    final_distance = float(torch.linalg.vector_norm(final_tcp - final_target))
                    final_hold = float(max_hold[env_id])
                    final_success = bool(success[env_id] or (final_distance < command.cfg.success_distance and hold[env_id] + dt >= 0.5))
                    root_displacement = float(torch.linalg.vector_norm(final_state[:2] - initial_state[env_id, :2]))
                    arm_motion = float(torch.linalg.vector_norm(final_state[arm_slice] - initial_state[env_id, arm_slice]))
                    rows.append({
                        "initial_target_distance": float(target_range[env_id]),
                        "mean_tcp_distance": float(dist_sum[env_id] / dist_count[env_id].clamp_min(1)),
                        "final_tcp_distance": final_distance,
                        "success": float(final_success),
                        "success_hold_time": final_hold,
                        "base_displacement": root_displacement,
                        "arm_joint_motion": arm_motion,
                        "fall": float(terminated[env_id]),
                        "timeout": float(truncated[env_id]),
                    })
                    episodes_done[env_id] += 1
                    if len(rows) >= episodes:
                        break
                # A reset installs a new target. Reset episode-local accumulators
                # and snapshot its start pose after env.step's automatic reset.
                reset_state = state_io.read()
                for env_id in ids.tolist():
                    initial_state[env_id] = reset_state[env_id]
                    initial_target[env_id] = command.target_position_world[env_id]
                    initial_tcp[env_id] = tcp_position_world(reset_state[env_id:env_id + 1], fk, tcp_ids)[0]
                    target_range[env_id] = torch.linalg.vector_norm(initial_target[env_id] - initial_tcp[env_id])
                    step_count[env_id] = 0
                    hold[env_id] = 0
                    max_hold[env_id] = 0
                    success[env_id] = False
                    dist_sum[env_id] = 0
                    dist_count[env_id] = 0

        def mean(key: str, selected: list[dict[str, float]] | None = None) -> float:
            values = [r[key] for r in (selected if selected is not None else rows)]
            return sum(values) / max(len(values), 1)

        near = [r for r in rows if r["initial_target_distance"] <= 0.40]
        far = [r for r in rows if r["initial_target_distance"] > 0.40]
        result = {
            "checkpoint": str(checkpoint),
            "episodes": len(rows),
            "mean_initial_target_distance": mean("initial_target_distance"),
            "mean_tcp_distance": mean("mean_tcp_distance"),
            "final_tcp_distance": mean("final_tcp_distance"),
            "success_rate": mean("success"),
            "success_hold_time": mean("success_hold_time"),
            "base_displacement": mean("base_displacement"),
            "arm_joint_motion": mean("arm_joint_motion"),
            "fall_rate": mean("fall"),
            "timeout_rate": mean("timeout"),
            "fk_mujoco_tcp_error_mean": fk_error_sum / max(fk_error_count, 1),
            "near_target_episodes": len(near),
            "near_target_initial_distance": mean("initial_target_distance", near),
            "near_target_success_rate": mean("success", near),
            "far_target_episodes": len(far),
            "far_target_initial_distance": mean("initial_target_distance", far),
            "far_target_success_rate": mean("success", far),
        }
        print(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def main(
    checkpoint: Path,
    device: str = "cuda:0",
    num_envs: int = 128,
    episodes: int = 256,
    debug_vis: bool = False,
) -> None:
    evaluate(checkpoint, device, num_envs, episodes, debug_vis)


if __name__ == "__main__":
    tyro.cli(main)
