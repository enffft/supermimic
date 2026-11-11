"""Opt-in collision-safe spawn for the expanded stairs atlas, without reward changes."""

import torch

from .feet import GO2_FOOT_RADIUS, go2_feet_world


def safe_spawn_lift(root, joint_position, terrain, margin=0.005):
    """Minimal upward translation putting all four foot spheres above collision terrain.

    Uses the final jittered XY, yaw and joint posture. Never lowers a spawn.
    Does not change terrain, orientation, dynamics, rewards or termination.
    """
    state = torch.zeros(*root.shape[:-1], 37, device=root.device, dtype=root.dtype)
    state[..., :7] = root[..., :7]
    state[..., 7:19] = joint_position
    feet, _ = go2_feet_world(state)
    clearance = feet[..., 2] - terrain.ground_height(feet[..., :2]) - GO2_FOOT_RADIUS
    return (margin - clearance.amin(-1)).clamp_min(0)
