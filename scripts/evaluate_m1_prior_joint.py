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
from supermimic.motion_prior.reference import decode_future31, future31_to_m1_action22
from supermimic.policy.nets import (
    LegacyM1FutureConditionedActor, LegacyM1PriorActorCfg,
    M1ActionReferenceActor, M1ActionReferenceActorCfg,
)
from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS
from supermimic.tasks.action_contract import inverse_m1_setpoint_action, m1_robot_spec_from_checkpoint

L_COMMANDS = torch.tensor((
    (0.30, 0.00, 0.00),
    (0.00, 0.00, 0.20),
    (-0.30, 0.00, 0.00),
    (0.00, 0.00, 0.00),
    (0.12, 0.00, 0.00),
    (-0.06, 0.00, 0.00),
), dtype=torch.float32)
L_NAMES = ("forward_ood", "yaw_only_ood", "backward_ood", "stop_id", "forward_id", "backward_id")
M_OFFSETS = torch.tensor((
    (0.20, 0.00, 0.05),
    (0.10, 0.15, 0.05),
    (0.12, 0.00, 0.12),
    (0.18, 0.10, 0.08),
), dtype=torch.float32)
M_NAMES = ("forward", "side", "up", "combined")


def _inverse_reference_action(q_target, wheel_rate, *, spec, default, scales, wheel_mask):
    del default, scales, wheel_mask
    return inverse_m1_setpoint_action(q_target, wheel_rate, robot_spec=spec)


def _command_in_saved_range(command, command_state):
    tolerance = 1e-6
    return all(
        float(bounds[0]) - tolerance <= float(command[index]) <= float(bounds[1]) + tolerance
        for index, bounds in enumerate((
            command_state.get("vx_range", (-float("inf"), float("inf"))),
            command_state.get("vy_range", (-float("inf"), float("inf"))),
            command_state.get("wz_range", (-float("inf"), float("inf"))),
        ))
    )


def _target_offset_in_saved_range(offset, command_state):
    tolerance = 1e-6
    return all(
        float(bounds[0]) - tolerance <= float(offset[axis]) <= float(bounds[1]) + tolerance
        for axis, bounds in enumerate((
            command_state.get("target_x_range", (0.12, 0.36)),
            command_state.get("target_y_range", (-0.20, 0.20)),
            command_state.get("target_z_range", (-0.15, 0.15)),
        ))
    )


def _resolve_saved_task_type(checkpoint: dict, requested: str | None = None) -> str:
    """Select L or M from a fixed or balanced checkpoint without weakening its contract."""
    if requested is not None and requested not in ("L", "M"):
        raise ValueError("task_type must be L or M when explicitly set")
    sampler = checkpoint.get("task_sampler_state", {})
    saved = sampler.get("task_type")
    if saved == "balanced":
        if requested is None:
            raise ValueError("balanced checkpoint evaluation requires task_type='L' or task_type='M'")
        return requested
    if saved not in ("L", "M"):
        saved = checkpoint.get("config", {}).get("m1_joint_training", {}).get("task_type")
    if saved == "balanced":
        if requested is None:
            raise ValueError("balanced checkpoint evaluation requires task_type='L' or task_type='M'")
        return requested
    if saved not in ("L", "M"):
        raise ValueError("checkpoint does not contain a supported L/M or balanced task definition")
    if requested is not None and requested != saved:
        raise ValueError(f"requested task_type {requested!r} differs from saved checkpoint task {saved!r}")
    return saved


def evaluate(
    checkpoint: Path,
    *,
    task_type: str | None = None,
    prior_variant: str = "trained",
    num_envs: int = 64,
    steps: int = 500,
    seed: int = 20261102,
    device: str = "cuda:0",
    controller: str = "actor",
    out: Path | None = None,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if prior_variant not in ("trained", "initial"):
        raise ValueError("prior_variant must be 'trained' or 'initial'")
    if controller not in ("actor", "prior_direct"):
        raise ValueError("controller must be 'actor' or 'prior_direct'")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_format = ckpt.get("format")
    is_action22_v4 = checkpoint_format == "supermimic_m1_action22_reference_v1"
    is_legacy = checkpoint_format in (
        "supermimic_m1_prior_actor_joint_v2", "supermimic_m1_prior_actor_joint_v3",
    )
    if not (is_action22_v4 or is_legacy):
        raise ValueError("checkpoint is not a supported M1 Prior/Actor joint checkpoint")
    if is_action22_v4 and ckpt.get("reference_adapter_schema") != "future31_to_interleaved_action22_v1":
        raise ValueError("v4 checkpoint has an unsupported Future31 to Action22 adapter schema")
    spec = m1_robot_spec_from_checkpoint(ckpt)
    reset_contract = ckpt.get("reset_contract")
    if reset_contract is not None:
        if reset_contract.get("profile") != spec.default_pose_profile:
            raise ValueError("checkpoint reset pose profile differs from its saved Action Contract")
        if reset_contract.get("task_conditional") is not False:
            raise ValueError("task-conditional reset is unsupported by this Action22 evaluator")
        if tuple(reset_contract.get("default_joint_pos", ())) != tuple(spec.default_joint_pos):
            raise ValueError("checkpoint reset joint pose differs from its saved Action Contract")
        if abs(float(reset_contract.get("default_base_height_m", float("nan")))
               - float(spec.default_base_height)) > 1e-8:
            raise ValueError("checkpoint reset base height differs from its saved Action Contract")
    contract_type = "v3" if spec.action_contract_version.endswith("v3") else "v2"
    actor_cfg = dict(ckpt["actor_schema"])
    if isinstance(actor_cfg.get("hidden"), list):
        actor_cfg["hidden"] = tuple(actor_cfg["hidden"])
    actor = (
        M1ActionReferenceActor(M1ActionReferenceActorCfg(**actor_cfg)).to(device)
        if is_action22_v4 else
        LegacyM1FutureConditionedActor(LegacyM1PriorActorCfg(**actor_cfg)).to(device)
    )
    actor.load_state_dict(ckpt["actor"], strict=True)
    actor.eval()
    source = ckpt["prior_source"]
    training_cfg = ckpt.get("config", {}).get("m1_joint_training", {})
    prior_input_compatibility = ckpt.get(
        "prior_input_compatibility", training_cfg.get("prior_input_compatibility", "train_constant_compat_v1"),
    )
    prior = load_m1_motion_prior(
        source["path"], device=device, expected_sha256=source["sha256"],
        input_compatibility=prior_input_compatibility,
    )
    if source.get("input_compatibility", prior_input_compatibility) != prior_input_compatibility:
        raise ValueError("saved Motion Prior input compatibility differs from its source provenance")
    prior_state_key = "motion_prior" if prior_variant == "trained" else "prior_initial"
    if prior_state_key not in ckpt:
        raise ValueError(f"checkpoint does not contain {prior_state_key!r} weights for the requested diagnostic")
    prior.load_state_dict(ckpt[prior_state_key], strict=True)
    prior.eval()

    sampler = ckpt.get("task_sampler_state", {})
    task_type = _resolve_saved_task_type(ckpt, task_type)
    if num_envs < (len(L_NAMES) if task_type == "L" else len(M_NAMES)):
        raise ValueError("num_envs must provide at least one environment per fixed command group")
    ranges = {
        "vx_range": tuple(sampler.get("vx_range", (-0.09, 0.145))),
        "vy_range": tuple(sampler.get("vy_range", (-0.05, 0.055))),
        "wz_range": tuple(sampler.get("wz_range", (-0.17, 0.18))),
    }
    profile = {
        "m1_pose_legacy_manipulation_v1": "legacy_manipulation",
        "m1_pose_dataset_carry_v1": "dataset_carry",
        "m1_pose_dataset_v3_3_l_high_carry_v1": "dataset_v3_3_l_high_carry",
    }[spec.default_pose_profile]

    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        m1_pose_profile=profile, m1_action_contract=contract_type,
        num_envs=num_envs, history_length=int(actor.cfg.history),
        episode_length_s=10.0, action_noise_std=0.0, obs_noise=False,
        reset_noise=False, domain_randomization=False, record_transitions=False,
        replay_capacity=32, seed=seed, m1_joint_task_type=task_type,
        m1_joint_vx_range=ranges["vx_range"], m1_joint_vy_range=ranges["vy_range"],
        m1_joint_vx_stop_fraction=float(sampler.get("vx_stop_fraction", 0.15)),
        m1_joint_vx_reverse_fraction=float(sampler.get("vx_reverse_fraction", 0.35)),
        m1_joint_wz_range=ranges["wz_range"],
        m1_joint_target_x_range=tuple(sampler.get("target_x_range", (0.12, 0.36))),
        m1_joint_target_y_range=tuple(sampler.get("target_y_range", (-0.20, 0.20))),
        m1_joint_target_z_range=tuple(sampler.get("target_z_range", (-0.15, 0.15))),
        m1_success_distance=float(sampler.get("success_distance", 0.08)),
        m1_fall_height=float(sampler.get("success_minimum_base_height_m", 0.22)),
    ), device=device)
    try:
        obs, _ = env.reset(seed=seed)
        command = env.command_manager.get_term("m1_prior_joint")
        state_io = StateIO(env)
        robot = env.scene["robot"]
        state = state_io.read()
        task_codes = torch.zeros(num_envs, dtype=torch.long, device=device)
        names = L_NAMES if task_type == "L" else M_NAMES
        category = torch.arange(num_envs, device=device) % len(names)
        if task_type == "L":
            velocity = L_COMMANDS.to(device)[category]
            target = torch.zeros(num_envs, 3, device=device)
        else:
            task_codes.fill_(1)
            velocity = torch.zeros(num_envs, 3, device=device)
            tcp_id = command._tcp_body_ids
            tcp = robot.data.body_link_pos_w[:, tcp_id].mean(dim=1) - env.scene.env_origins
            target = tcp + M_OFFSETS.to(device)[category]
        command.task_type[:] = task_codes

        history = state[:, None].expand(-1, 4, -1).clone()
        reference = torch.zeros(num_envs, 10, 31, device=device, dtype=state.dtype)
        reference_actions = torch.zeros(num_envs, 10, 22, device=device, dtype=state.dtype)
        ref_state = torch.zeros(num_envs, 10, 57, device=device, dtype=state.dtype)
        phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_start = state[:, :3].clone()
        episode_start_quat = state[:, 3:7].clone()
        episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        tcp_id = command._tcp_body_ids

        wheel_ids = torch.tensor([
            M1_JOINTS.index(n) for n in (
                "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
            )
        ], device=device)
        arm_ids = torch.tensor([M1_JOINTS.index(n) for n in M1_AMP_JOINTS[12:]], device=device)
        leg_ids = torch.tensor([M1_JOINTS.index(n) for n in M1_AMP_JOINTS[:12]], device=device)
        scalar_keys = (
            "samples", "xy_sq_sum", "xy_count", "vx_sq_sum", "vy_sq_sum", "yaw_sq_sum", "yaw_count",
            "ee_distance_sum", "ee_progress_sum", "base_displacement_sum",
            "wheel_rate_sum", "arm_sq_sum", "arm_count", "root_pos_sq_sum",
            "root_rot_sq_sum", "leg_sq_sum", "leg_count", "wheel_track_sq_sum",
            "wheel_count", "action_saturated", "falls", "episodes",
            "success_episodes", "distance_at_end_sum", "distance_at_end_count",
            "base_height_sum", "base_tilt_sum", "forward_displacement_sum",
            "positive_command_negative_vx", "episode_duration_s_sum",
            "prior_reference_root_dx_sum", "prior_reference_count",
            "prior_reference_wrong_direction_count", "prior_reference_stop_abs_dx_sum",
            "prior_reference_first_dx_sum", "prior_reference_first_wrong_direction_count",
            "prior_wheel_wrong_direction_count", "prior_root_wheel_disagreement_count",
            "prior_stop_abs_wheel_rate_sum",
        )
        slots = []
        for _ in names:
            row = {key: torch.zeros((), device=device) for key in scalar_keys}
            row["hold"] = torch.zeros(num_envs, device=device)
            row["wheel_reference_sum"] = torch.zeros(4, device=device)
            row["prior_wheel_reference_sum"] = torch.zeros(4, device=device)
            row["wheel_actual_sum"] = torch.zeros(4, device=device)
            row["wheel_track_sq_by_joint"] = torch.zeros(4, device=device)
            slots.append(row)

        tcp = robot.data.body_link_pos_w[:, tcp_id].mean(dim=1) - env.scene.env_origins
        episode_initial_distance = torch.linalg.vector_norm(tcp - target, dim=-1)
        for i in range(len(names)):
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
                candidate_actions = (
                    future31_to_m1_action22(candidate_reference, robot_spec=spec)
                    if is_action22_v4 else None
                )
                candidate_state = decode_future31(state, candidate_reference)
                plan_dx_first_heading = quat_apply_inverse(
                    state[:, 3:7], candidate_state[:, 0, :3] - state[:, :3],
                )[:, 0]
                plan_dx_heading = quat_apply_inverse(
                    state[:, 3:7], candidate_state[:, -1, :3] - state[:, :3],
                )[:, 0]
                plan_wheel_mean = candidate_state[..., 35:57].index_select(-1, wheel_ids).mean(1)
                plan_wheel_scalar = plan_wheel_mean.mean(-1)
                reference = torch.where(due[:, None, None], candidate_reference, reference)
                if is_action22_v4:
                    reference_actions = torch.where(due[:, None, None], candidate_actions, reference_actions)
                if step == 0:
                    ref_state = candidate_state
                else:
                    ref_state = torch.where(due[:, None, None], candidate_state, ref_state)
                ref_next = ref_state[torch.arange(num_envs, device=device), phase]
                row_ids = torch.arange(num_envs, device=device)
                current_reference_action = reference_actions[row_ids, phase]
                if controller == "prior_direct":
                    if not is_action22_v4:
                        raise ValueError("prior_direct evaluation requires an Action22-reference v4 checkpoint")
                    action = current_reference_action
                elif is_action22_v4:
                    action = actor(obs["policy"], goal, current_reference_action)
                else:
                    action = actor(obs["policy"], goal, reference, phase.to(dtype=state.dtype))
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
                xy_error = body_v[:, :2] - velocity[:, :2]
                xy_sq = xy_error.square()
                yaw_sq = (next_state[:, 34] - velocity[:, 2]).square()
                displacement = torch.linalg.vector_norm(next_state[:, :3] - episode_start, dim=-1)
                signed_forward = quat_apply_inverse(
                    episode_start_quat, next_state[:, :3] - episode_start,
                )[:, 0]
                predicted_next = ref_next
                arm_err = next_state[:, 7:29].index_select(-1, arm_ids) - predicted_next[:, 7:29].index_select(-1, arm_ids)
                leg_err = next_state[:, 7:29].index_select(-1, leg_ids) - predicted_next[:, 7:29].index_select(-1, leg_ids)
                wheel_err = next_state[:, 35:57].index_select(-1, wheel_ids) - predicted_next[:, 35:57].index_select(-1, wheel_ids)
                root_pos_err = next_state[:, :3] - predicted_next[:, :3]
                root_rot_err = quat_error_magnitude(next_state[:, 3:7], predicted_next[:, 3:7])

                for i in range(len(names)):
                    mask = (category == i) & valid
                    weight = mask.to(state.dtype)
                    row = slots[i]
                    plan_weight = ((category == i) & due).to(state.dtype)
                    if task_type == "L":
                        row["prior_reference_root_dx_sum"] += (plan_dx_heading * plan_weight).sum()
                        row["prior_reference_count"] += plan_weight.sum()
                        row["prior_reference_first_dx_sum"] += (plan_dx_first_heading * plan_weight).sum()
                        row["prior_reference_wrong_direction_count"] += (
                            (((velocity[:, 0] > 0.05) & (plan_dx_heading <= 0.0))
                             | ((velocity[:, 0] < -0.05) & (plan_dx_heading >= 0.0))).to(state.dtype)
                            * plan_weight
                        ).sum()
                        moving_forward = velocity[:, 0] > 0.05
                        moving_backward = velocity[:, 0] < -0.05
                        moving = moving_forward | moving_backward
                        row["prior_reference_first_wrong_direction_count"] += (
                            (
                                (moving_forward & (plan_dx_first_heading <= 0.0))
                                | (moving_backward & (plan_dx_first_heading >= 0.0))
                            ).to(state.dtype) * plan_weight
                        ).sum()
                        row["prior_wheel_wrong_direction_count"] += (
                            (
                                (moving_forward & (plan_wheel_scalar <= 0.0))
                                | (moving_backward & (plan_wheel_scalar >= 0.0))
                            ).to(state.dtype) * plan_weight
                        ).sum()
                        row["prior_root_wheel_disagreement_count"] += (
                            (
                                moving & ((plan_dx_heading > 0.0) != (plan_wheel_scalar > 0.0))
                            ).to(state.dtype) * plan_weight
                        ).sum()
                        row["prior_reference_stop_abs_dx_sum"] += (
                            plan_dx_heading.abs() * plan_weight
                            * (velocity[:, 0].abs() <= 0.05).to(state.dtype)
                        ).sum()
                        row["prior_stop_abs_wheel_rate_sum"] += (
                            plan_wheel_scalar.abs() * plan_weight
                            * (velocity[:, 0].abs() <= 0.05).to(state.dtype)
                        ).sum()
                        row["prior_wheel_reference_sum"] += (plan_wheel_mean * plan_weight[:, None]).sum(0)
                    row["samples"] += weight.sum()
                    row["xy_sq_sum"] += (xy_sq.sum(-1) * weight).sum()
                    row["xy_count"] += 2 * weight.sum()
                    row["vx_sq_sum"] += (xy_sq[:, 0] * weight).sum()
                    row["vy_sq_sum"] += (xy_sq[:, 1] * weight).sum()
                    row["yaw_sq_sum"] += (yaw_sq * weight).sum()
                    row["yaw_count"] += weight.sum()
                    row["wheel_rate_sum"] += (next_state[:, 35:57].index_select(-1, wheel_ids).abs().mean(-1) * weight).sum()
                    row["wheel_reference_sum"] += (ref_next[:, 35:57].index_select(-1, wheel_ids) * weight[:, None]).sum(0)
                    row["wheel_actual_sum"] += (next_state[:, 35:57].index_select(-1, wheel_ids) * weight[:, None]).sum(0)
                    row["base_displacement_sum"] += (displacement * weight).sum()
                    row["root_pos_sq_sum"] += (root_pos_err.square().sum(-1) * weight).sum()
                    row["root_rot_sq_sum"] += (root_rot_err.square() * weight).sum()
                    row["arm_sq_sum"] += (arm_err.square().sum(-1) * weight).sum()
                    row["arm_count"] += weight.sum() * arm_err.shape[-1]
                    row["leg_sq_sum"] += (leg_err.square().sum(-1) * weight).sum()
                    row["leg_count"] += weight.sum() * leg_err.shape[-1]
                    row["wheel_track_sq_sum"] += (wheel_err.square().sum(-1) * weight).sum()
                    row["wheel_count"] += weight.sum() * wheel_err.shape[-1]
                    row["wheel_track_sq_by_joint"] += (wheel_err.square() * weight[:, None]).sum(0)
                    row["action_saturated"] += ((action.abs() >= 0.98).float().sum(-1) * weight).sum()
                    row["ee_distance_sum"] += (next_distance * weight).sum()
                    row["ee_progress_sum"] += ((episode_initial_distance - next_distance) * weight).sum()
                    row["base_height_sum"] += (next_state[:, 2] * weight).sum()
                    row["base_tilt_sum"] += (tilt * weight).sum()
                    row["forward_displacement_sum"] += (signed_forward * weight).sum()
                    if task_type == "L":
                        row["positive_command_negative_vx"] += (
                            ((velocity[:, 0] > 0.05) & (body_v[:, 0] < 0.0)).to(state.dtype) * weight
                        ).sum()
                    if step >= steps - 25:
                        row["distance_at_end_sum"] += (next_distance * weight).sum()
                        row["distance_at_end_count"] += weight.sum()

                if task_type == "M":
                    stable_reach = (next_distance < 0.08) & stable
                    for i in range(len(names)):
                        mask_i = category == i
                        row = slots[i]
                        active_hold = stable_reach & valid
                        ended = mask_i & done
                        row["episodes"] += ended.sum()
                        row["success_episodes"] += ((row["hold"] >= 0.5) & ended).sum()
                        row["falls"] += (terminated & ended).sum()
                        row["episode_duration_s_sum"] += (episode_steps[ended].to(state.dtype) + 1.0).sum() * env.step_dt
                        row["hold"] = torch.where(
                            mask_i & active_hold, row["hold"] + env.step_dt,
                            torch.where(mask_i & valid, torch.zeros_like(row["hold"]), row["hold"]),
                        )
                        row["hold"] = torch.where(ended, torch.zeros_like(row["hold"]), row["hold"])
                else:
                    for i in range(len(names)):
                        mask_i = category == i
                        ended = mask_i & done
                        slots[i]["episodes"] += ended.sum()
                        slots[i]["falls"] += (terminated & ended).sum()
                        slots[i]["episode_duration_s_sum"] += (episode_steps[ended].to(state.dtype) + 1.0).sum() * env.step_dt

                episode_steps += 1
                history = torch.cat((history[:, 1:], next_state[:, None]), dim=1)
                history[done] = next_state[done, None, :].expand(-1, 4, -1)
                episode_start[done] = next_state[done, :3]
                episode_start_quat[done] = next_state[done, 3:7]
                episode_steps[done] = 0
                episode_initial_distance[done] = next_distance[done]
                phase = (phase + 1) % 10
                phase[done] = 0
                state, obs = next_state, next_obs

        result: dict[str, object] = {
            "checkpoint": str(checkpoint), "iteration": int(ckpt["iteration"]),
            "task_type": task_type, "num_envs": num_envs, "steps": steps,
            "seed": seed, "domain_randomization": False,
            "controller": controller,
            "prior_weights_variant": prior_variant,
            "prior_swap_diagnostic_uses_same_actor": prior_variant == "initial",
            "prior_source": source,
            "prior_input_compatibility": prior_input_compatibility,
            "default_pose_profile": spec.default_pose_profile,
            "reset_contract": reset_contract,
            "action_mapping": ckpt["action_mapping"],
            "task_sampler_state": sampler,
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
                "wheel_reference_mean_rad_s_by_joint": (row["wheel_reference_sum"] / n).cpu().tolist(),
                "wheel_actual_mean_rad_s_by_joint": (row["wheel_actual_sum"] / n).cpu().tolist(),
                "wheel_tracking_rmse_rad_s_by_joint": (row["wheel_track_sq_by_joint"] / n).clamp_min(0).sqrt().cpu().tolist(),
                "action_saturation_fraction": float(row["action_saturated"] / (row["samples"] * 22).clamp_min(1)),
                "wheel_rate_abs_mean_rad_s": float(row["wheel_rate_sum"] / n),
                "base_displacement_mean_m": float(row["base_displacement_sum"] / n),
                "signed_forward_displacement_mean_m": float(row["forward_displacement_sum"] / n),
                "mean_episode_duration_s": float(row["episode_duration_s_sum"] / row["episodes"].clamp_min(1)),
                "base_height_mean_m": float(row["base_height_sum"] / n),
                "base_tilt_mean_rad": float(row["base_tilt_sum"] / n),
            }
            if task_type == "L":
                command = L_COMMANDS[i]
                n_plans = row["prior_reference_count"].clamp_min(1)
                metrics.update({
                    "command_vx_vy_wz": command.tolist(),
                    "command_in_saved_training_range": _command_in_saved_range(command, sampler),
                    "ood_test": not _command_in_saved_range(command, sampler),
                    "base_velocity_xy_vector_rmse_m_s": float((row["xy_sq_sum"] / row["samples"].clamp_min(1)) ** 0.5),
                    "base_velocity_xy_per_axis_rmse_m_s": float((row["xy_sq_sum"] / row["xy_count"].clamp_min(1)) ** 0.5),
                    "base_velocity_xy_rmse_m_s": float((row["xy_sq_sum"] / row["samples"].clamp_min(1)) ** 0.5),
                    "base_vx_rmse_m_s": float((row["vx_sq_sum"] / row["samples"].clamp_min(1)) ** 0.5),
                    "base_vy_rmse_m_s": float((row["vy_sq_sum"] / row["samples"].clamp_min(1)) ** 0.5),
                    "positive_command_negative_vx_fraction": float(
                        row["positive_command_negative_vx"] / row["samples"].clamp_min(1)
                    ),
                    "prior_reference_root_dx_0p2s_mean_m": float(
                        row["prior_reference_root_dx_sum"] / n_plans
                    ),
                    "prior_reference_wrong_direction_fraction": float(
                        row["prior_reference_wrong_direction_count"] / n_plans
                    ),
                    "prior_reference_first_dx_0p02s_mean_m": float(
                        row["prior_reference_first_dx_sum"] / n_plans
                    ),
                    "prior_reference_first_wrong_direction_fraction": float(
                        row["prior_reference_first_wrong_direction_count"] / n_plans
                    ),
                    "prior_reference_stop_abs_dx_0p2s_mean_m": float(
                        row["prior_reference_stop_abs_dx_sum"] / n_plans
                    ),
                    "prior_reference_wheel_wrong_direction_fraction": float(
                        row["prior_wheel_wrong_direction_count"] / n_plans
                    ),
                    "prior_root_wheel_direction_disagreement_fraction": float(
                        row["prior_root_wheel_disagreement_count"] / n_plans
                    ),
                    "prior_stop_abs_wheel_rate_mean_rad_s": float(
                        row["prior_stop_abs_wheel_rate_sum"] / n_plans
                    ),
                    "prior_wheel_reference_mean_rad_s_by_joint": (
                        row["prior_wheel_reference_sum"] / n_plans
                    ).cpu().tolist(),
                    "yaw_rate_rmse_rad_s": float((row["yaw_sq_sum"] / row["yaw_count"].clamp_min(1)) ** 0.5),
                })
            else:
                offset = M_OFFSETS[i]
                target_in_range = _target_offset_in_saved_range(offset, sampler)
                metrics.update({
                    "target_offset_m": offset.tolist(),
                    "command_in_saved_training_range": target_in_range,
                    "ood_test": not target_in_range,
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
    task_type: str | None = None,
    prior_variant: str = "trained",
    num_envs: int = 64,
    steps: int = 500,
    seed: int = 20261102,
    device: str = "cuda:0",
    controller: str = "actor",
    out: Path | None = None,
) -> None:
    evaluate(checkpoint, task_type=task_type, prior_variant=prior_variant,
             num_envs=num_envs, steps=steps,
             seed=seed, device=device, controller=controller, out=out)


if __name__ == "__main__":
    tyro.cli(main)
