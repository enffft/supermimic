"""Small action mappings shared by simulator collection and imagination."""

from __future__ import annotations

import torch


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
    position_target = torch.maximum(
        torch.minimum(position_target, soft_joint_limits[..., 1]), soft_joint_limits[..., 0],
    )
    wheel_target = action * action_scale
    return torch.where(wheel_mask, wheel_target, position_target)
