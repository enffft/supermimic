"""Compare M1 CVAE references with real MuJoCo MixedM1Action reachability."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.reference import decode_future31, future31_to_m1_action22
from supermimic.policy.nets import (
    LegacyM1FutureConditionedActor, LegacyM1PriorActorCfg,
    M1ActionReferenceActor, M1ActionReferenceActorCfg,
)
from supermimic.robots.spec import M1_JOINTS, M1_LEG_JOINTS, get_robot_spec
from supermimic.tasks.action_contract import inverse_m1_setpoint_action, m1_robot_spec_from_checkpoint

PIPER_SDK_POSITION_LIMITS = (
    (-2.6179, 2.6179), (0.0, 3.14), (-2.967, 0.0),
    (-1.745, 1.745), (-1.22, 1.22), (-2.09439, 2.09439),
)
PIPER_SDK_MAX_SPEED_RAD_S = 3.0
PIPER_SDK_POSITION_LIMITS_BY_NAME = {
    f"piper_x_joint{i}": value for i, value in enumerate(PIPER_SDK_POSITION_LIMITS, start=1)
}


def _urdf_joint_limits() -> dict[str, tuple[float, float, float]]:
    path = Path("/home/yifeiy/AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf")
    result = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        node = joint.find("limit")
        if node is not None:
            result[str(joint.get("name"))] = (
                float(node.get("lower", "-inf")), float(node.get("upper", "inf")),
                float(node.get("velocity", "inf")),
            )
    return result


def _classify_references(ref_state, *, spec, urdf_limits):
    counts = {
        "position_joint_reference_frames": 0,
        "velocity_joint_reference_frames": 0,
        "out_of_action_envelope_position_frames": 0,
        "urdf_position_violation_frames": 0,
        "sdk_position_violation_frames": 0,
        "urdf_velocity_violation_frames": 0,
        "sdk_velocity_violation_frames": 0,
        "wheel_out_of_action_envelope_frames": 0,
    }
    per_joint = {}
    bounds = ref_state.new_tensor(spec.position_target_bounds) if spec.position_target_bounds else None
    wheel_names = {"FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT"}
    for joint_id, name in enumerate(M1_JOINTS):
        qd = ref_state[..., 35 + joint_id]
        hard_lo, hard_hi, hard_speed = urdf_limits.get(
            name, (-float("inf"), float("inf"), float("inf")),
        )
        urdf_velocity = qd.abs() > hard_speed
        if name in wheel_names:
            outside_envelope = qd.abs() > spec.action_scales[joint_id]
            counts["wheel_out_of_action_envelope_frames"] += int(outside_envelope.sum())
            sdk_position = torch.zeros_like(urdf_velocity)
            sdk_velocity = torch.zeros_like(urdf_velocity)
            urdf_position = torch.zeros_like(urdf_velocity)
            pos_frames = 0
        else:
            q = ref_state[..., 7 + joint_id]
            if bounds is None:
                scale, default = spec.action_scales[joint_id], spec.default_joint_pos[joint_id]
                outside_envelope = (q < default - scale) | (q > default + scale)
            else:
                outside_envelope = (q < bounds[joint_id, 0]) | (q > bounds[joint_id, 1])
            urdf_position = (q < hard_lo) | (q > hard_hi)
            sdk = PIPER_SDK_POSITION_LIMITS_BY_NAME.get(name)
            sdk_position = torch.zeros_like(urdf_position) if sdk is None else (q < sdk[0]) | (q > sdk[1])
            sdk_velocity = torch.zeros_like(urdf_velocity) if sdk is None else qd.abs() > min(hard_speed, PIPER_SDK_MAX_SPEED_RAD_S)
            pos_frames = q.numel()
            counts["position_joint_reference_frames"] += pos_frames
            counts["out_of_action_envelope_position_frames"] += int(outside_envelope.sum())
            counts["urdf_position_violation_frames"] += int(urdf_position.sum())
            counts["sdk_position_violation_frames"] += int(sdk_position.sum())
        counts["velocity_joint_reference_frames"] += qd.numel()
        counts["urdf_velocity_violation_frames"] += int(urdf_velocity.sum())
        counts["sdk_velocity_violation_frames"] += int(sdk_velocity.sum())
        per_joint[name] = {
            "position_frame_count": int(pos_frames),
            "position_out_of_action_envelope_fraction": float(outside_envelope.float().mean()),
            "urdf_position_violation_fraction": float(urdf_position.float().mean()),
            "sdk_position_violation_fraction": float(sdk_position.float().mean()),
            "urdf_velocity_violation_fraction": float(urdf_velocity.float().mean()),
            "sdk_velocity_violation_fraction": float(sdk_velocity.float().mean()),
        }
    position_n = max(counts["position_joint_reference_frames"], 1)
    velocity_n = max(counts["velocity_joint_reference_frames"], 1)
    counts["out_of_action_envelope_fraction"] = counts["out_of_action_envelope_position_frames"] / position_n
    counts["urdf_limit_violation_fraction"] = (
        counts["urdf_position_violation_frames"] + counts["urdf_velocity_violation_frames"]
    ) / (position_n + velocity_n)
    counts["sdk_limit_violation_fraction"] = (
        counts["sdk_position_violation_frames"] + counts["sdk_velocity_violation_frames"]
    ) / (position_n + velocity_n)
    counts["velocity_limit_violation_fraction"] = counts["urdf_velocity_violation_frames"] / velocity_n
    counts["per_joint"] = per_joint
    return counts


def _reference_to_action(q_target, wheel_rate, *, spec, default, scales, wheel_mask):
    del default, scales, wheel_mask
    return inverse_m1_setpoint_action(q_target, wheel_rate, robot_spec=spec)


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
    is_action22_v4 = checkpoint.get("format") == "supermimic_m1_action22_reference_v1"
    if is_action22_v4 and checkpoint.get("reference_adapter_schema") != "future31_to_interleaved_action22_v1":
        raise ValueError("unsupported Action22 reference adapter schema")
    actor = (
        M1ActionReferenceActor(M1ActionReferenceActorCfg(**actor_cfg)).to(device)
        if is_action22_v4 else
        LegacyM1FutureConditionedActor(LegacyM1PriorActorCfg(**actor_cfg)).to(device)
    )
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()

    source = checkpoint["prior_source"]
    spec = m1_robot_spec_from_checkpoint(checkpoint)
    training_cfg = checkpoint.get("config", {}).get("m1_joint_training", {})
    input_compatibility = checkpoint.get(
        "prior_input_compatibility", training_cfg.get("prior_input_compatibility", "train_constant_compat_v1"),
    )
    prior = load_m1_motion_prior(
        source["path"], device=device, expected_sha256=source["sha256"],
        input_compatibility=input_compatibility,
    )
    if source.get("input_compatibility", input_compatibility) != input_compatibility:
        raise ValueError("saved Motion Prior input compatibility differs from source provenance")
    urdf_joint_limits = _urdf_joint_limits()
    sampler = checkpoint.get("task_sampler_state", {})
    profile = {
        "m1_pose_legacy_manipulation_v1": "legacy_manipulation",
        "m1_pose_dataset_carry_v1": "dataset_carry",
        "m1_pose_dataset_v3_3_l_high_carry_v1": "dataset_v3_3_l_high_carry",
    }[spec.default_pose_profile]
    contract = "v3" if spec.action_contract_version.endswith("v3") else "v2"
    args = SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        m1_pose_profile=profile, m1_action_contract=contract,
        num_envs=num_envs, history_length=actor.cfg.history,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, replay_capacity=32, seed=seed,
        m1_joint_task_type=task_type,
        m1_joint_vx_range=tuple(sampler.get("vx_range", (-0.09, 0.145))),
        m1_joint_vx_stop_fraction=float(sampler.get("vx_stop_fraction", 0.15)),
        m1_joint_vx_reverse_fraction=float(sampler.get("vx_reverse_fraction", 0.35)),
        m1_joint_vy_range=tuple(sampler.get("vy_range", (-0.05, 0.055))),
        m1_joint_wz_range=tuple(sampler.get("wz_range", (-0.17, 0.18))),
        m1_joint_target_x_range=tuple(sampler.get("target_x_range", (0.12, 0.36))),
        m1_joint_target_y_range=tuple(sampler.get("target_y_range", (-0.20, 0.20))),
        m1_joint_target_z_range=tuple(sampler.get("target_z_range", (-0.15, 0.15))),
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
            "motion_prior_source": source,
            "prior_input_compatibility": input_compatibility,
            "default_pose_profile": spec.default_pose_profile,
            "action_mapping": checkpoint["action_mapping"],
            "action_scales": {name: float(scale) for name, scale in zip(M1_JOINTS, spec.action_scales)},
            "reference_not_dynamically_validated": True,
            "prior_assurance_mode": source.get("assurance_mode"),
            "strict_fcl_only": source.get("strict_fcl_only"),
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
                if spec.position_target_bounds:
                    bounds = torch.tensor(spec.position_target_bounds, device=device)
                    lower, upper = bounds[:, 0], bounds[:, 1]
                else:
                    lower, upper = default - scales, default + scales
                q[..., amp_ids] = torch.maximum(
                    torch.minimum(q[..., amp_ids], upper[amp_ids]), lower[amp_ids],
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
            actor_actions_ref = (
                future31_to_m1_action22(actor_reference, robot_spec=spec)
                if is_action22_v4 else None
            )

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
                        action = _reference_to_action(
                            q_target, ref_state[:, phase, 35:57], spec=spec,
                            default=default, scales=scales, wheel_mask=wheel_mask,
                        )
                    else:
                        if is_action22_v4:
                            action = actor(
                                observations["policy"], observations["goal"], actor_actions_ref[:, phase],
                            )
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
            reference_states = decode_future31(initial, ref)
        result["reference_safety_classification"] = _classify_references(
            reference_states, spec=spec, urdf_limits=urdf_joint_limits,
        )
        q_reference = torch.zeros(num_envs, 10, 22, device=device)
        q_reference[..., amp_ids] = ref[..., 9:27]
        position_delta = q_reference[..., amp_ids] - default[amp_ids]
        if spec.position_target_bounds:
            target_bounds = torch.tensor(spec.position_target_bounds, device=device)[amp_ids]
            action_upper = target_bounds[:, 1] - default[amp_ids]
            action_lower = default[amp_ids] - target_bounds[:, 0]
            normalized_position_action = torch.where(
                position_delta >= 0.0,
                position_delta / action_upper.clamp_min(1.0e-8),
                position_delta / action_lower.clamp_min(1.0e-8),
            )
            envelope_lower, envelope_upper = target_bounds[:, 0], target_bounds[:, 1]
        else:
            normalized_position_action = position_delta / scales[amp_ids]
            envelope_lower = default[amp_ids] - scales[amp_ids]
            envelope_upper = default[amp_ids] + scales[amp_ids]
        amp_limits = limits[amp_ids]
        result["reference_reachability"] = {
            "position_delta_p5_p50_p95_rad": torch.quantile(position_delta.flatten(), torch.tensor([0.05, 0.5, 0.95], device=device)).cpu().tolist(),
            "position_outside_setpoint_fraction": float(((q_reference[..., amp_ids] < envelope_lower) | (q_reference[..., amp_ids] > envelope_upper)).float().mean()),
            "position_near_setpoint_saturation_fraction": float((normalized_position_action.abs() > 0.9).float().mean()),
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
                "action_bounds_rad": [float(envelope_lower[amp_joint_names.index(joint_name)]), float(envelope_upper[amp_joint_names.index(joint_name)])],
                "outside_setpoint_fraction": float(((joint_ref < envelope_lower[amp_joint_names.index(joint_name)]) | (joint_ref > envelope_upper[amp_joint_names.index(joint_name)])).float().mean()),
                "near_setpoint_saturation_fraction": float((normalized_position_action[..., amp_joint_names.index(joint_name)].abs() > 0.9).float().mean()),
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
