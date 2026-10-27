"""Small action mappings shared by simulator collection and imagination."""

from __future__ import annotations

import torch


def clamp_executed_action(policy_action: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """Deterministic deployment mapping from policy output to the actuator input."""
    return policy_action.clamp(-float(clip), float(clip))


def position_setpoint(
    executed_action: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scale: torch.Tensor,
) -> torch.Tensor:
    """Go2 joint target in radians for a clipped/executed normalized action."""
    return default_joint_pos + executed_action * action_scale
