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
) -> torch.Tensor:
    """Map M1's interleaved 22D action to physical joint targets.

    Position-controlled joints use ``q_default + a*scale`` and are clamped to
    soft limits; wheel joints use velocity targets ``a*scale``. The returned
    tensor retains RobotSpec order so callers can split it by actuator type.
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
    position_target = default_joint_pos + action * action_scale
    position_target = torch.maximum(
        torch.minimum(position_target, soft_joint_limits[..., 1]), soft_joint_limits[..., 0],
    )
    wheel_target = action * action_scale
    return torch.where(wheel_mask, wheel_target, position_target)
