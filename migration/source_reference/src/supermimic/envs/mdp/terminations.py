"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end


def free_root_too_low(env: ManagerBasedRlEnv, asset_cfg, threshold: float) -> torch.Tensor:
    """Reference-free low-base failure label for locomotion tasks."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < threshold


def free_root_tilted(env: ManagerBasedRlEnv, asset_cfg, threshold: float = 0.8) -> torch.Tensor:
    """Reference-free tilt label: upright gravity z is -1, so 1 + z measures tilt."""
    asset = env.scene[asset_cfg.name]
    gravity = asset.data.gravity_vec_w
    gravity_local = quat_apply_inverse(asset.data.root_link_quat_w, gravity)
    return (1.0 + gravity_local[:, 2]).abs() > threshold


def chase_tag(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Success timeout, distinct from the physical fall termination label."""
    from supermimic.envs.mdp.commands import ChaseBallCommand
    from supermimic.envs.state_io import StateIO
    from supermimic.tasks.chase_reward import chase_front_position

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ChaseBallCommand):
        raise TypeError("chase_tag termination requires ChaseBallCommand")
    state_io = getattr(env, "_supermimic_chase_state_io", None)
    if state_io is None:
        state_io = StateIO(env)
        env._supermimic_chase_state_io = state_io
    state = state_io.read()
    front = chase_front_position(state)
    distance = torch.linalg.vector_norm(front[..., :2] - command.ball_position_world[..., :2], dim=-1)
    return distance <= command.cfg.tag_radius
