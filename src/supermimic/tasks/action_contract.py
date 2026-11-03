"""Small action mappings shared by simulator collection and imagination."""

from __future__ import annotations

import torch


_M1_V3_POSITION_MAPPING = "asymmetric_default_to_absolute_bounds_v3;clamp_soft_limits"
_M1_WHEEL_MAPPING = "clip(action,-1,1)*wheel_scale_rad_s"


def serialize_m1_action_contract(robot_spec) -> dict:
    """Return the single checkpoint/resume serialization of an M1 action map."""
    if robot_spec.name != "m1_piper":
        raise ValueError("M1 action contract serialization requires the m1_piper RobotSpec")
    wheel_names = tuple(name for name in robot_spec.joint_names if name.endswith("FOOT_JOINT"))
    wheel_count = len(wheel_names)
    return {
        "contract_version": robot_spec.action_contract_version,
        "default_pose_profile": robot_spec.default_pose_profile,
        "default_joint_pos": tuple(robot_spec.default_joint_pos),
        "joint_names": tuple(robot_spec.joint_names),
        "wheel_joint_names": wheel_names,
        "position_control_count": robot_spec.num_joints - wheel_count,
        "wheel_velocity_control_count": wheel_count,
        "action_modes": tuple(robot_spec.action_modes),
        "action_scales": tuple(robot_spec.action_scales),
        "position_target_bounds": tuple(robot_spec.position_target_bounds),
        "position_mapping": _M1_V3_POSITION_MAPPING if robot_spec.position_target_bounds else "default_plus_action_scale_v2",
        "wheel_mapping": _M1_WHEEL_MAPPING,
    }


def m1_robot_spec_from_checkpoint(checkpoint):
    """Rebuild the M1 spec selected by a saved Actor/Action Contract checkpoint."""
    from dataclasses import replace

    from supermimic.robots.spec import get_robot_spec

    mapping = checkpoint.get("action_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("M1 checkpoint is missing its saved action_mapping")
    contract_version = mapping.get("contract_version")
    contract = {"m1_mixed_action_v2": "v2", "m1_mixed_action_v3": "v3"}.get(contract_version)
    if contract is None:
        raise ValueError(f"unsupported saved M1 action contract: {contract_version!r}")
    profile_names = {
        "m1_pose_legacy_manipulation_v1": "legacy_manipulation",
        "m1_pose_dataset_carry_v1": "dataset_carry",
        "m1_pose_dataset_v3_3_l_high_carry_v1": "dataset_v3_3_l_high_carry",
    }
    profile = profile_names.get(mapping.get("default_pose_profile"))
    if profile is None:
        profile = checkpoint.get("config", {}).get("env", {}).get("m1_pose_profile")
    if profile is None:
        profile = "legacy_manipulation"
    spec = get_robot_spec("m1_piper", m1_pose_profile=profile, m1_action_contract=contract)
    fields = {
        "default_joint_pos": "default_joint_pos",
        "joint_names": "joint_names",
        "action_modes": "action_modes",
        "action_scales": "action_scales",
        "position_target_bounds": "position_target_bounds",
    }
    updates = {}
    for key, field_name in fields.items():
        if key in mapping:
            value = mapping[key]
            if key in ("default_joint_pos", "action_scales"):
                value = tuple(float(v) for v in value)
            elif key in ("joint_names", "action_modes"):
                value = tuple(value)
            elif key == "position_target_bounds":
                value = tuple(tuple(float(x) for x in bounds) for bounds in value)
            updates[field_name] = value
    spec = replace(spec, **updates)
    reconstructed = serialize_m1_action_contract(spec)
    if contract == "v3" and reconstructed != mapping:
        raise ValueError("saved v3 M1 action mapping differs from the supported pose/profile contract")
    return spec


def clamp_executed_action(policy_action: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """Deterministic deployment mapping from policy output to the actuator input."""
    return policy_action.clamp(-float(clip), float(clip))


def add_bounded_action_exploration(
    policy_action: torch.Tensor,
    noise: torch.Tensor,
    environment_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply explicitly sampled exploration to selected envs, then clamp.

    The returned action is the actual command passed to the environment and
    therefore the value that must be written to replay and used by the WM.
    """
    if policy_action.shape != noise.shape:
        raise ValueError("exploration noise must match the policy-action shape")
    if environment_mask.shape != (policy_action.shape[0],):
        raise ValueError("exploration mask must have one value per environment")
    perturbation = noise * environment_mask.to(policy_action.dtype).unsqueeze(-1)
    return clamp_executed_action(policy_action + perturbation)


def position_setpoint(
    executed_action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scale: torch.Tensor,
) -> torch.Tensor:
    """Go2 joint target in radians for a clipped/executed normalized action."""
    return default_joint_pos + executed_action * action_scale


def mixed_m1_setpoints(
    executed_action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scale: torch.Tensor,
    wheel_mask: torch.Tensor,
    soft_joint_limits: torch.Tensor,
    position_target_bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map M1's interleaved 22D action to physical joint targets.

    Without bounds this keeps Contract v2 ``q_default + a*scale`` behavior.
    Contract v3 maps position actions asymmetrically between an absolute safe
    interval and the default pose, so -1/0/+1 map to lower/default/upper.
    Wheel joints always retain angular-velocity semantics ``a*scale``.
    """
    if executed_action.shape[-1] != 22:
        raise ValueError("M1 Mixed Action must have 22 dimensions")
    if default_joint_pos.shape[-1] != 22 or action_scale.shape[-1] != 22:
        raise ValueError("M1 defaults and action scales must have 22 dimensions")
    if wheel_mask.shape != (22,) or wheel_mask.dtype != torch.bool:
        raise ValueError("M1 wheel mask must be a bool vector of length 22")
    if soft_joint_limits.shape[-2:] != (22, 2):
        raise ValueError("M1 soft joint limits must end in [22,2]")
    action = clamp_executed_action(executed_action)
    if position_target_bounds is None:
        position_target = default_joint_pos + action * action_scale
    else:
        if position_target_bounds.shape[-2:] != (22, 2):
            raise ValueError("M1 v3 position bounds must end in [22,2]")
        lower, upper = position_target_bounds[..., 0], position_target_bounds[..., 1]
        positive = default_joint_pos + action * (upper - default_joint_pos)
        negative = default_joint_pos + action * (default_joint_pos - lower)
        position_target = torch.where(action >= 0.0, positive, negative)
        # Keep endpoint rounding inside the serialized action envelope before
        # applying the simulator's final URDF soft-limit guard.
        position_target = torch.maximum(torch.minimum(position_target, upper), lower)
    position_target = torch.maximum(
        torch.minimum(position_target, soft_joint_limits[..., 1]), soft_joint_limits[..., 0],
    )
    wheel_target = action * action_scale
    return torch.where(wheel_mask, wheel_target, position_target)


def inverse_m1_setpoint_action(
    joint_target: torch.Tensor,
    wheel_rate_target: torch.Tensor,
    *,
    robot_spec,
) -> torch.Tensor:
    """Invert the saved M1 v2/v3 setpoint map for a physical reference frame."""
    if joint_target.shape[-1] != robot_spec.num_joints:
        raise ValueError(f"joint_target must end in {robot_spec.num_joints}, got {tuple(joint_target.shape)}")
    wheel_ids = tuple(robot_spec.joint_names.index(name) for name in robot_spec.wheel_joint_names)
    if wheel_rate_target.shape != (*joint_target.shape[:-1], len(wheel_ids)):
        raise ValueError("wheel_rate_target must match the leading batch shape and saved wheel order")
    default = joint_target.new_tensor(robot_spec.default_joint_pos)
    scales = joint_target.new_tensor(robot_spec.action_scales)
    wheel_mask = torch.zeros(robot_spec.num_joints, dtype=torch.bool, device=joint_target.device)
    wheel_mask[list(wheel_ids)] = True
    if robot_spec.position_target_bounds:
        bounds = joint_target.new_tensor(robot_spec.position_target_bounds)
        lower, upper = bounds[..., 0], bounds[..., 1]
        positive = (joint_target - default) / (upper - default).clamp_min(1e-8)
        negative = (joint_target - default) / (default - lower).clamp_min(1e-8)
        action = torch.where(joint_target >= default, positive, negative)
    else:
        action = (joint_target - default) / torch.where(wheel_mask, torch.ones_like(scales), scales)
    for slot, joint_id in enumerate(wheel_ids):
        action[..., joint_id] = wheel_rate_target[..., slot] / scales[joint_id].clamp_min(1e-8)
    return action.clamp(-1.0, 1.0)
