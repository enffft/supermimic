"""Compare M1 CVAE references with real MuJoCo MixedM1Action reachability."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.reference import decode_future31
from supermimic.policy.nets import M1PriorActorCfg, M1PriorConditionedActor
from supermimic.robots.spec import M1_JOINTS, M1_LEG_JOINTS, get_robot_spec


def _summarize(rows: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    result = {}
    for key, values in rows.items():
        if not values:
            continue
        tensor = torch.cat([value.reshape(-1).float() for value in values])
        result[key] = float(tensor.mean())
    return result


def audit(
    actor_checkpoint: Path,
    *,
    task_type: str = "L",
    num_envs: int = 32,
    device: str = "cuda:0",
    seed: int = 20261102,
    out: Path | None = None,
) -> dict:
    if task_type not in ("L", "M"):
        raise ValueError("task_type must be L or M")
    checkpoint = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("robot_identity") != "m1_piper":
        raise ValueError("Actor checkpoint is not an M1-Piper joint-training checkpoint")
    actor_cfg = checkpoint.get("actor_schema")
    if not isinstance(actor_cfg, dict):
        raise TypeError("checkpoint does not contain the versioned M1 Actor schema")
    actor_cfg = {key: tuple(value) if isinstance(value, list) and key == "hidden" else value
                 for key, value in actor_cfg.items()}
    actor = M1PriorConditionedActor(M1PriorActorCfg(**actor_cfg)).to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()

    source = checkpoint["prior_source"]
    prior = load_m1_motion_prior(
        source["path"], device=device, expected_sha256=source["sha256"],
    )
    spec = get_robot_spec("m1_piper")
    args = SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        num_envs=num_envs, history_length=actor.cfg.history,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, replay_capacity=32, seed=seed,
        m1_joint_task_type=task_type,
    )
    env = make_env(args, device=device)
    try:
        command = env.command_manager.get_term("m1_prior_joint")
        state_io = StateIO(env)
        scales = torch.tensor(spec.action_scales, device=device)
        default = env.scene["robot"].data.default_joint_pos[0].clone()
        limits = env.scene["robot"].data.soft_joint_pos_limits[0].clone()
        wheel_mask = torch.tensor([name.endswith("FOOT_JOINT") for name in M1_JOINTS], device=device)
        amp_joint_names = tuple(
            name for name in M1_LEG_JOINTS if not name.endswith("FOOT_JOINT")
        ) + tuple(f"piper_x_joint{i}" for i in range(1, 7))
        amp_ids = torch.tensor([M1_JOINTS.index(name) for name in amp_joint_names], device=device)
        arm_ids = torch.tensor([M1_JOINTS.index(f"piper_x_joint{i}") for i in range(1, 7)], device=device)
        leg_ids = torch.tensor([M1_JOINTS.index(name) for name in amp_joint_names[:12]], device=device)
        wheel_ids = torch.tensor([M1_JOINTS.index(name) for name in (
            "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
        )], device=device)

        result: dict[str, object] = {
            "task_type": task_type,
            "num_envs": num_envs,
            "control_dt_s": float(env.step_dt),
            "action_scales": {name: float(scale) for name, scale in zip(M1_JOINTS, spec.action_scales)},
            "scenarios": {},
        }
        for scenario in ("actor", "inverse_action", "projected_reference_actor"):
            observations, _ = env.reset(seed=seed)
            state = state_io.read()
            state_history = state[:, None].expand(-1, 4, -1).contiguous()
            goal = task_goal8(
                state, command.task_type, command.velocity_command_body,
                command.target_position_world,
            )
            with torch.no_grad():
                reference = prior(state_history, goal, mode="prior_mean")
                ref_state = decode_future31(state, reference)
            initial_root_position = state[:, :3].clone()

            if scenario == "projected_reference_actor":
                q = ref_state[..., 7:29].clone()
                q[..., amp_ids] = torch.maximum(
                    torch.minimum(q[..., amp_ids], default[amp_ids] + scales[amp_ids]),
                    default[amp_ids] - scales[amp_ids],
                )
                q[..., amp_ids] = torch.maximum(
                    torch.minimum(q[..., amp_ids], limits[amp_ids, 1]), limits[amp_ids, 0],
                )
                projected = reference.clone()
                projected[..., 9:27] = q[..., amp_ids]
                projected[..., 27:31] = reference[..., 27:31].clamp(-5.0, 5.0)
                actor_reference = projected
                ref_state = decode_future31(state, projected)
            else:
                actor_reference = reference

            sums: dict[str, list[torch.Tensor]] = {
                "root_position_error_m": [], "leg_position_rmse_rad": [],
                "arm_position_rmse_rad": [], "wheel_rate_rmse_rad_s": [],
                "action_saturation_fraction": [], "fall_fraction": [],
                "base_displacement_m": [], "wheel_rate_abs_mean_rad_s": [],
            }
            for phase in range(10):
                with torch.no_grad():
                    if scenario == "inverse_action":
                        q_target = ref_state[:, phase, 7:29]
                        action = torch.where(
                            wheel_mask,
                            ref_state[:, phase, 35:57] / scales,
                            (q_target - default) / scales,
                        ).clamp(-1.0, 1.0)
                    else:
                        action = actor(
                            observations["policy"], observations["goal"], actor_reference,
                            torch.full((num_envs,), phase, device=device, dtype=state.dtype),
                        )
                next_obs, _, terminated, truncated, _ = env.step(action)
                measured = state_io.read()
                valid = ~(terminated | truncated)
                if valid.any():
                    sums["root_position_error_m"].append(
                        torch.linalg.vector_norm(measured[valid, :3] - ref_state[valid, phase, :3], dim=-1),
                    )
                    for key, indices, source, scale in (
                        ("leg_position_rmse_rad", leg_ids, measured[..., 7:29], 1.0),
                        ("arm_position_rmse_rad", arm_ids, measured[..., 7:29], 1.0),
                    ):
                        err = (source[valid].index_select(-1, indices)
                               - ref_state[valid, phase, 7:29].index_select(-1, indices))
                        sums[key].append(err.square().mean(-1).sqrt() / scale)
                    wheel_error = measured[valid, 35:57].index_select(-1, wheel_ids) - ref_state[valid, phase, 35:57].index_select(-1, wheel_ids)
                    sums["wheel_rate_rmse_rad_s"].append(wheel_error.square().mean(-1).sqrt())
                    sums["base_displacement_m"].append(
                        torch.linalg.vector_norm(measured[valid, :2] - initial_root_position[valid, :2], dim=-1),
                    )
                    sums["wheel_rate_abs_mean_rad_s"].append(
                        measured[valid, 35:57].index_select(-1, wheel_ids).abs().mean(-1),
                    )
                sums["action_saturation_fraction"].append((action.abs() >= 0.98).float().mean().reshape(1))
                sums["fall_fraction"].append(terminated.float().mean().reshape(1))
                observations = next_obs
                state = measured

            result["scenarios"][scenario] = _summarize(sums)
        # Static reference-to-setpoint reachability measured on each initial plan.
        observations, _ = env.reset(seed=seed)
        initial = state_io.read()
        history = initial[:, None].expand(-1, 4, -1).contiguous()
        goal = task_goal8(initial, command.task_type, command.velocity_command_body, command.target_position_world)
        with torch.no_grad():
            ref = prior(history, goal)
        q_reference = torch.zeros(num_envs, 10, 22, device=device)
        q_reference[..., amp_ids] = ref[..., 9:27]
        position_delta = q_reference[..., amp_ids] - default[amp_ids]
        amp_scales = scales[amp_ids]
        amp_limits = limits[amp_ids]
        result["reference_reachability"] = {
            "position_delta_p5_p50_p95_rad": torch.quantile(position_delta.flatten(), torch.tensor([0.05, 0.5, 0.95], device=device)).cpu().tolist(),
            "position_outside_setpoint_fraction": float((position_delta.abs() > amp_scales).float().mean()),
            "position_near_setpoint_saturation_fraction": float((position_delta.abs() > 0.9 * amp_scales).float().mean()),
            "position_outside_soft_limit_fraction": float(((q_reference[..., amp_ids] < amp_limits[:, 0]) | (q_reference[..., amp_ids] > amp_limits[:, 1])).float().mean()),
            "wheel_rate_p5_p50_p95_rad_s": torch.quantile(ref[..., 27:31].flatten(), torch.tensor([0.05, 0.5, 0.95], device=device)).cpu().tolist(),
            "wheel_outside_setpoint_fraction": float((ref[..., 27:31].abs() > 5.0).float().mean()),
            "wheel_near_setpoint_saturation_fraction": float((ref[..., 27:31].abs() > 4.5).float().mean()),
        }
        joint_stats: dict[str, dict[str, object]] = {}
        for future_index, joint_name in enumerate(amp_joint_names):
            joint_id = M1_JOINTS.index(joint_name)
            joint_ref = ref[..., 9 + future_index]
            delta = joint_ref - default[joint_id]
            joint_limit = limits[joint_id]
            joint_stats[joint_name] = {
                "reference_position_p5_p50_p95_rad": torch.quantile(
                    joint_ref.flatten(), torch.tensor([0.05, 0.5, 0.95], device=device),
                ).cpu().tolist(),
                "delta_from_default_p5_p50_p95_rad": torch.quantile(
                    delta.flatten(), torch.tensor([0.05, 0.5, 0.95], device=device),
                ).cpu().tolist(),
                "outside_setpoint_fraction": float((delta.abs() > scales[joint_id]).float().mean()),
                "near_setpoint_saturation_fraction": float((delta.abs() > 0.9 * scales[joint_id]).float().mean()),
                "outside_soft_limit_fraction": float(
                    ((joint_ref < joint_limit[0]) | (joint_ref > joint_limit[1])).float().mean(),
                ),
            }
        wheel_stats: dict[str, dict[str, object]] = {}
        wheel_names = ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")
        for index, joint_name in enumerate(wheel_names):
            rate = ref[..., 27 + index]
            wheel_stats[joint_name] = {
                "rate_p5_p50_p95_rad_s": torch.quantile(
                    rate.flatten(), torch.tensor([0.05, 0.5, 0.95], device=device),
                ).cpu().tolist(),
                "outside_setpoint_fraction": float((rate.abs() > 5.0).float().mean()),
                "near_setpoint_saturation_fraction": float((rate.abs() > 4.5).float().mean()),
            }
        result["per_joint_reference_reachability"] = {
            "position_joints": joint_stats,
            "wheel_velocity_joints": wheel_stats,
        }
        encoded = json.dumps(result, indent=2)
        print(encoded)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(encoded + "\n")
        return result
    finally:
        env.close()


def main(actor_checkpoint: Path, task_type: str = "L", num_envs: int = 32,
         device: str = "cuda:0", seed: int = 20261102, out: Path | None = None) -> None:
    audit(actor_checkpoint, task_type=task_type, num_envs=num_envs, device=device, seed=seed, out=out)


if __name__ == "__main__":
    tyro.cli(main)
