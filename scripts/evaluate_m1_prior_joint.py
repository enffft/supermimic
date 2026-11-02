"""Fixed-seed real-MuJoCo evaluation for M1 L velocity and M EE reach checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.rotations import quat_apply_inverse, quat_error_magnitude
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.metrics import root_tilt_and_stability
from supermimic.motion_prior.reference import decode_future31
from supermimic.policy.nets import M1PriorActorCfg, M1PriorConditionedActor
from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS

L_COMMANDS = torch.tensor((
    (0.30, 0.00, 0.00),
    (0.00, 0.15, 0.00),
    (0.00, 0.00, 0.30),
    (0.30, 0.10, 0.20),
), dtype=torch.float32)
L_NAMES = ("forward", "lateral", "yaw", "combined")
M_OFFSETS = torch.tensor((
    (0.20, 0.00, 0.05),
    (0.10, 0.15, 0.05),
    (0.12, 0.00, 0.12),
    (0.18, 0.10, 0.08),
), dtype=torch.float32)
M_NAMES = ("forward", "side", "up", "combined")


def evaluate(
    checkpoint: Path,
    *,
    task_type: str = "L",
    num_envs: int = 64,
    steps: int = 500,
    seed: int = 20261102,
    device: str = "cuda:0",
    out: Path | None = None,
) -> dict:
    if task_type not in ("L", "M"):
        raise ValueError("task_type must be L or M")
    if num_envs < 4 or num_envs % 4:
        raise ValueError("num_envs must be a positive multiple of four")
    if steps <= 0:
        raise ValueError("steps must be positive")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "supermimic_m1_prior_actor_joint_v2":
        raise ValueError("checkpoint is not an M1 Prior/Actor joint checkpoint")
    if ckpt.get("action_mapping", {}).get("contract_version") != "m1_mixed_action_v2":
        raise ValueError("checkpoint uses an incompatible M1 Mixed Action contract")
    actor_cfg = dict(ckpt["actor_schema"])
    if isinstance(actor_cfg.get("hidden"), list):
        actor_cfg["hidden"] = tuple(actor_cfg["hidden"])
    actor = M1PriorConditionedActor(M1PriorActorCfg(**actor_cfg)).to(device)
    actor.load_state_dict(ckpt["actor"], strict=True)
    actor.eval()
    source = ckpt["prior_source"]
    prior = load_m1_motion_prior(
        source["path"], device=device, expected_sha256=source["sha256"],
    )
    prior.load_state_dict(ckpt["motion_prior"], strict=True)
    prior.eval()

    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        num_envs=num_envs, history_length=int(actor.cfg.history),
        episode_length_s=10.0, action_noise_std=0.0, obs_noise=False,
        reset_noise=False, domain_randomization=False, record_transitions=False,
        replay_capacity=32, seed=seed, m1_joint_task_type=task_type,
    ), device=device)
    try:
        obs, _ = env.reset(seed=seed)
        command = env.command_manager.get_term("m1_prior_joint")
        state_io = StateIO(env)
        robot = env.scene["robot"]
        state = state_io.read()
        task_codes = torch.zeros(num_envs, dtype=torch.long, device=device)
        category = torch.arange(num_envs, device=device) % 4
        if task_type == "L":
            velocity = L_COMMANDS.to(device)[category]
            target = torch.zeros(num_envs, 3, device=device)
            names = L_NAMES
        else:
            task_codes.fill_(1)
            velocity = torch.zeros(num_envs, 3, device=device)
            tcp_id = command._tcp_body_ids
            tcp = robot.data.body_link_pos_w[:, tcp_id].mean(dim=1) - env.scene.env_origins
            target = tcp + M_OFFSETS.to(device)[category]
            names = M_NAMES
        command.task_type[:] = task_codes

        history = state[:, None].expand(-1, 4, -1).clone()
        reference = torch.zeros(num_envs, 10, 31, device=device, dtype=state.dtype)
        ref_state = torch.zeros(num_envs, 10, 57, device=device, dtype=state.dtype)
        phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_start = state[:, :3].clone()
        tcp_id = command._tcp_body_ids

        wheel_ids = torch.tensor([
            M1_JOINTS.index(n) for n in (
                "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
            )
        ], device=device)
        arm_ids = torch.tensor([M1_JOINTS.index(n) for n in M1_AMP_JOINTS[12:]], device=device)
        leg_ids = torch.tensor([M1_JOINTS.index(n) for n in M1_AMP_JOINTS[:12]], device=device)
        scalar_keys = (
            "samples", "xy_sq_sum", "xy_count", "yaw_sq_sum", "yaw_count",
            "ee_distance_sum", "ee_progress_sum", "base_displacement_sum",
            "wheel_rate_sum", "arm_sq_sum", "arm_count", "root_pos_sq_sum",
            "root_rot_sq_sum", "leg_sq_sum", "leg_count", "wheel_track_sq_sum",
            "wheel_count", "action_saturated", "falls", "episodes",
            "success_episodes", "distance_at_end_sum", "distance_at_end_count",
            "base_height_sum", "base_tilt_sum",
        )
        slots = []
        for _ in range(4):
            row = {key: torch.zeros((), device=device) for key in scalar_keys}
            row["hold"] = torch.zeros(num_envs, device=device)
            slots.append(row)

        tcp = robot.data.body_link_pos_w[:, tcp_id].mean(dim=1) - env.scene.env_origins
        episode_initial_distance = torch.linalg.vector_norm(tcp - target, dim=-1)
        for i in range(4):
            slots[i]["hold"].zero_()

        with torch.inference_mode():
            for step in range(steps):
                # Reassert the fixed per-group command/target after any auto-reset.
                command.task_type[:] = task_codes
                command.velocity_command_body[:] = velocity
                command.target_position_world[:] = target
                goal = task_goal8(state, task_codes, velocity, target)
                obs["goal"] = goal
                due = phase == 0
                candidate_reference = prior(history, goal, mode="prior_mean")
                candidate_state = decode_future31(state, candidate_reference)
                reference = torch.where(due[:, None, None], candidate_reference, reference)
                if step == 0:
                    ref_state = candidate_state
                else:
                    ref_state = torch.where(due[:, None, None], candidate_state, ref_state)
                ref_next = ref_state[torch.arange(num_envs, device=device), phase]
                action = actor(
                    obs["policy"], goal, reference,
                    phase.to(dtype=state.dtype),
                )
                next_obs, _, terminated, truncated, _ = env.step(action)
                next_state = state_io.read()
                done = terminated | truncated
                valid = ~done
                next_tcp = robot.data.body_link_pos_w[:, tcp_id].mean(dim=1) - env.scene.env_origins
                next_distance = torch.linalg.vector_norm(next_tcp - target, dim=-1)
                tilt, stable = root_tilt_and_stability(
                    next_state[:, 3:7], next_state[:, 2],
                    minimum_height_m=0.22, maximum_tilt_rad=0.90,
                )
                body_v = quat_apply_inverse(next_state[:, 3:7], next_state[:, 29:32])
                xy_sq = (body_v[:, :2] - velocity[:, :2]).square()
                yaw_sq = (next_state[:, 34] - velocity[:, 2]).square()
                displacement = torch.linalg.vector_norm(next_state[:, :3] - episode_start, dim=-1)
                predicted_next = ref_next
                arm_err = next_state[:, 7:29].index_select(-1, arm_ids) - predicted_next[:, 7:29].index_select(-1, arm_ids)
                leg_err = next_state[:, 7:29].index_select(-1, leg_ids) - predicted_next[:, 7:29].index_select(-1, leg_ids)
                wheel_err = next_state[:, 35:57].index_select(-1, wheel_ids) - predicted_next[:, 35:57].index_select(-1, wheel_ids)
                root_pos_err = next_state[:, :3] - predicted_next[:, :3]
                root_rot_err = quat_error_magnitude(next_state[:, 3:7], predicted_next[:, 3:7])

                for i in range(4):
                    mask = (category == i) & valid
                    weight = mask.to(state.dtype)
                    row = slots[i]
                    row["samples"] += weight.sum()
                    row["xy_sq_sum"] += (xy_sq.sum(-1) * weight).sum()
                    row["xy_count"] += 2 * weight.sum()
                    row["yaw_sq_sum"] += (yaw_sq * weight).sum()
                    row["yaw_count"] += weight.sum()
                    row["wheel_rate_sum"] += (next_state[:, 35:57].index_select(-1, wheel_ids).abs().mean(-1) * weight).sum()
                    row["base_displacement_sum"] += (displacement * weight).sum()
                    row["root_pos_sq_sum"] += (root_pos_err.square().sum(-1) * weight).sum()
                    row["root_rot_sq_sum"] += (root_rot_err.square() * weight).sum()
                    row["arm_sq_sum"] += (arm_err.square().sum(-1) * weight).sum()
                    row["arm_count"] += weight.sum() * arm_err.shape[-1]
                    row["leg_sq_sum"] += (leg_err.square().sum(-1) * weight).sum()
                    row["leg_count"] += weight.sum() * leg_err.shape[-1]
                    row["wheel_track_sq_sum"] += (wheel_err.square().sum(-1) * weight).sum()
                    row["wheel_count"] += weight.sum() * wheel_err.shape[-1]
                    row["action_saturated"] += ((action.abs() >= 0.98).float().sum(-1) * weight).sum()
                    row["ee_distance_sum"] += (next_distance * weight).sum()
                    row["ee_progress_sum"] += ((episode_initial_distance - next_distance) * weight).sum()
                    row["base_height_sum"] += (next_state[:, 2] * weight).sum()
                    row["base_tilt_sum"] += (tilt * weight).sum()
                    if step >= steps - 25:
                        row["distance_at_end_sum"] += (next_distance * weight).sum()
                        row["distance_at_end_count"] += weight.sum()

                if task_type == "M":
                    stable_reach = (next_distance < 0.08) & stable
                    for i in range(4):
                        mask_i = category == i
                        row = slots[i]
                        active_hold = stable_reach & valid
                        ended = mask_i & done
                        row["episodes"] += ended.sum()
                        row["success_episodes"] += ((row["hold"] >= 0.5) & ended).sum()
                        row["falls"] += (terminated & ended).sum()
                        row["hold"] = torch.where(
                            mask_i & active_hold, row["hold"] + env.step_dt,
                            torch.where(mask_i & valid, torch.zeros_like(row["hold"]), row["hold"]),
                        )
                        row["hold"] = torch.where(ended, torch.zeros_like(row["hold"]), row["hold"])
                else:
                    for i in range(4):
                        mask_i = category == i
                        ended = mask_i & done
                        slots[i]["episodes"] += ended.sum()
                        slots[i]["falls"] += (terminated & ended).sum()

                history = torch.cat((history[:, 1:], next_state[:, None]), dim=1)
                history[done] = next_state[done, None, :].expand(-1, 4, -1)
                episode_start[done] = next_state[done, :3]
                episode_initial_distance[done] = next_distance[done]
                phase = (phase + 1) % 10
                phase[done] = 0
                state, obs = next_state, next_obs

        result: dict[str, object] = {
            "checkpoint": str(checkpoint), "iteration": int(ckpt["iteration"]),
            "task_type": task_type, "num_envs": num_envs, "steps": steps,
            "seed": seed, "domain_randomization": False,
            "action_contract": ckpt["action_mapping"]["contract_version"],
            "groups": {},
        }
        for i, name in enumerate(names):
            row = slots[i]
            n = max(float(row["samples"]), 1)
            metrics = {
                "samples": int(row["samples"]),
                "fall_rate": float(row["falls"] / row["episodes"].clamp_min(1)),
                "episode_count": int(row["episodes"]),
                "tracking_root_position_rmse_m": float((row["root_pos_sq_sum"] / n) ** 0.5),
                "tracking_root_rotation_rmse_rad": float((row["root_rot_sq_sum"] / n) ** 0.5),
                "tracking_arm_rmse_rad": float((row["arm_sq_sum"] / row["arm_count"].clamp_min(1)) ** 0.5),
                "tracking_leg_rmse_rad": float((row["leg_sq_sum"] / row["leg_count"].clamp_min(1)) ** 0.5),
                "tracking_wheel_rmse_rad_s": float((row["wheel_track_sq_sum"] / row["wheel_count"].clamp_min(1)) ** 0.5),
                "action_saturation_fraction": float(row["action_saturated"] / (row["samples"] * 22).clamp_min(1)),
                "wheel_rate_abs_mean_rad_s": float(row["wheel_rate_sum"] / n),
                "base_displacement_mean_m": float(row["base_displacement_sum"] / n),
                "base_height_mean_m": float(row["base_height_sum"] / n),
                "base_tilt_mean_rad": float(row["base_tilt_sum"] / n),
            }
            if task_type == "L":
                metrics.update({
                    "base_velocity_xy_rmse_m_s": float((row["xy_sq_sum"] / row["xy_count"].clamp_min(1)) ** 0.5),
                    "yaw_rate_rmse_rad_s": float((row["yaw_sq_sum"] / row["yaw_count"].clamp_min(1)) ** 0.5),
                })
            else:
                metrics.update({
                    "mean_ee_distance_m": float(row["ee_distance_sum"] / n),
                    "mean_ee_progress_m": float(row["ee_progress_sum"] / n),
                    "end_window_ee_distance_m": float(row["distance_at_end_sum"] / row["distance_at_end_count"].clamp_min(1)),
                    "stable_hold_success_rate": float(row["success_episodes"] / row["episodes"].clamp_min(1)),
                })
            result["groups"][name] = metrics

        encoded = json.dumps(result, indent=2)
        print(encoded)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(encoded + "\n")
        return result
    finally:
        env.close()


def main(
    checkpoint: Path,
    task_type: str = "L",
    num_envs: int = 64,
    steps: int = 500,
    seed: int = 20261102,
    device: str = "cuda:0",
    out: Path | None = None,
) -> None:
    evaluate(checkpoint, task_type=task_type, num_envs=num_envs, steps=steps,
             seed=seed, device=device, out=out)


if __name__ == "__main__":
    tyro.cli(main)
