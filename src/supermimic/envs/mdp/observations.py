from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.kinematics.rotations import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def velocity_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return command.goal


def chase_ball_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """6-D ball position/velocity in the robot's yaw-only navigation frame."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    root_local = robot.data.root_link_pos_w - env.scene.env_origins
    from supermimic.tasks.chase_reward import chase_goal_from_pose

    return chase_goal_from_pose(root_local, robot.data.root_link_quat_w, command.goal)


def world_point_goal_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fixed env-local world target expressed relative to the current robot base."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    root_local = robot.data.root_link_pos_w - env.scene.env_origins
    delta_world = command.target_position_world - root_local
    return quat_apply_inverse(robot.data.root_link_quat_w, delta_world)


def m1_prior_goal8(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """AMP Goal8; L command and M fixed env-local world target are explicit."""
    from supermimic.envs.state_io import StateIO
    from supermimic.motion_prior.features import task_goal8

    command = env.command_manager.get_term(command_name)
    state_io = getattr(env, "_supermimic_prior_state_io", None)
    if state_io is None:
        state_io = StateIO(env)
        env._supermimic_prior_state_io = state_io
    return task_goal8(
        state_io.read(), command.task_type,
        command.velocity_command_body, command.target_position_world,
    )


def joint_pos_rel_masked(
    env: ManagerBasedRlEnv,
    wheel_joint_names: tuple[str, ...] = (),
    biased: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint-position error with selected rolling wheel angles hidden from the actor."""
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    position = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    result = position[:, asset_cfg.joint_ids] - default
    if wheel_joint_names:
        joint_ids = asset_cfg.joint_ids
        selected_names = asset.joint_names[joint_ids] if isinstance(joint_ids, slice) else [asset.joint_names[int(i)] for i in joint_ids]
        for name in wheel_joint_names:
            if name in selected_names:
                # asset_cfg joint order is also the policy/action order.
                result[:, selected_names.index(name)] = 0.0
    return result
