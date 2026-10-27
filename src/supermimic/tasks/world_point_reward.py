"""Pure dense reward and MuJoCo RewardManager adapter for M1 world-point reach."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse


# The Piper URDF has no named TCP frame. Keep the phase-one gripper-center
# proxy (midpoint of these fixed finger frames) identical across commands,
# reward, evaluation, and visualization.
M1_TCP_PROXY_BODIES = ("piper_x_link7", "piper_x_link8")


def m1_tcp_proxy_indices(fk: G1Kinematics) -> tuple[int, int]:
    return tuple(fk.body_index(name) for name in M1_TCP_PROXY_BODIES)


@dataclass
class WorldPointRobotParams:
    default_joint_pos: torch.Tensor
    soft_joint_limits: torch.Tensor
    dt: float = 0.02
    fall_height: float = 0.20
    tilt_feature_threshold: float = 1.0 - math.cos(math.radians(70.0))
    soft_term_temperature: float = 0.015
    success_distance: float = 0.08
    distance_weight: float = 1.5
    approach_weight: float = 0.25
    near_weight: float = 0.6
    upright_weight: float = 0.25
    base_height_weight: float = -4.0
    joint_limit_weight: float = -0.3
    action_rate_weight: float = -0.005
    tcp_speed_weight: float = -0.02
    tcp_speed_threshold: float = 0.8


WORLD_POINT_REWARD_NAMES = (
    "tcp_distance",
    "tcp_approach",
    "tcp_near",
    "upright",
    "base_height_low",
    "joint_limit",
    "action_rate",
    "tcp_speed",
)


def tcp_position_world(
    state: torch.Tensor,
    fk: G1Kinematics,
    tcp_body_indices: tuple[int, int],
) -> torch.Tensor:
    """Differentiable gripper-center proxy: midpoint of the two fixed finger frames.

    The Piper URDF has no named TCP frame; link7/link8 are the two fixed gripper
    finger frames attached at a common origin. Their midpoint is used as the
    first-phase point-task proxy, with no claim that this equals an external tool
    calibration point.
    """
    nj = (state.shape[-1] - 13) // 2
    qpos_dim = 7 + nj
    bodies = fk(state[..., :qpos_dim], state[..., qpos_dim:])
    pos = bodies["pos"]
    return 0.5 * (pos[..., tcp_body_indices[0], :] + pos[..., tcp_body_indices[1], :])


def world_point_reward_terms(
    state_prev: torch.Tensor,
    state_next: torch.Tensor,
    target_position_world: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    params: WorldPointRobotParams,
    fk: G1Kinematics,
    tcp_body_indices: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Reward rates from two states and a fixed env-local world target.

    ``RewardManager`` and imagined rollouts both call this function and apply the
    control ``dt`` exactly once at their adapter boundary.
    """
    nj = action.shape[-1]
    tcp_prev = tcp_position_world(state_prev, fk, tcp_body_indices)
    tcp_next = tcp_position_world(state_next, fk, tcp_body_indices)
    d_prev = torch.linalg.vector_norm(tcp_prev - target_position_world, dim=-1)
    d_next = torch.linalg.vector_norm(tcp_next - target_position_world, dim=-1)
    tcp_speed = torch.linalg.vector_norm(tcp_next - tcp_prev, dim=-1) / float(params.dt)

    quat = state_next[..., 3:7]
    gravity_w = torch.zeros_like(quat[..., :3])
    gravity_w[..., 2] = -1.0
    gravity_b = quat_apply_inverse(quat, gravity_w)
    q = state_next[..., 7:7 + nj]
    below = F.relu(params.soft_joint_limits[..., 0] - q)
    above = F.relu(q - params.soft_joint_limits[..., 1])
    tilt_feature = (1.0 + gravity_b[..., 2]).abs()

    return {
        "tcp_distance": -params.distance_weight * d_next,
        # Signed, bounded progress prevents oscillating away and back from earning
        # positive progress twice. It remains useful when far from the target.
        # Bound progress by displacement, not nominal velocity. The old dt / 0.25
        # scaling saturated after roughly 5 mm per step and removed gradients on
        # ordinary arm motion.
        "tcp_approach": params.approach_weight * torch.tanh((d_prev - d_next) / 0.05),
        "tcp_near": params.near_weight * torch.exp(-(d_next / 0.10).square()),
        "upright": params.upright_weight * torch.exp(-gravity_b[..., :2].square().sum(-1) / 0.16),
        "base_height_low": params.base_height_weight * F.relu(params.fall_height + 0.04 - state_next[..., 2]).square(),
        "joint_limit": params.joint_limit_weight * (below.square() + above.square()).sum(-1),
        "action_rate": params.action_rate_weight * (action - previous_action).square().sum(-1),
        "tcp_speed": params.tcp_speed_weight * F.relu(tcp_speed - params.tcp_speed_threshold).square(),
    }


def m1_world_point_reward_term(env, term_name: str) -> torch.Tensor:
    """Cached simulator adapter; FK uses the same compiled URDF as World Model."""
    from supermimic.envs.mdp.commands import WorldPointCommand
    from supermimic.envs.state_io import StateIO
    from supermimic.robots.mjlab_assets import load_robot_spec_model
    from supermimic.robots.spec import get_robot_spec

    step = int(env.common_step_counter)
    cache = getattr(env, "_supermimic_world_point_reward_cache", None)
    if cache is None or cache["step"] != step:
        robot = env.scene["robot"]
        command = env.command_manager.get_term("world_point")
        if not isinstance(command, WorldPointCommand):
            raise TypeError("world_point reward requires WorldPointCommand")
        state_io = getattr(env, "_supermimic_world_point_state_io", None)
        if state_io is None:
            state_io = StateIO(env)
            env._supermimic_world_point_state_io = state_io
        fk = getattr(env, "_supermimic_world_point_fk", None)
        if fk is None:
            from supermimic.kinematics.g1_fk import G1Kinematics

            fk = G1Kinematics(load_robot_spec_model("m1_piper"), device=env.device)
            env._supermimic_world_point_fk = fk
        spec = get_robot_spec("m1_piper")
        tcp_indices = m1_tcp_proxy_indices(fk)
        action = env.action_manager.action.clone()
        previous_action = env.action_manager.prev_action.clone()
        state_next = state_io.read()
        state_prev = state_next
        recorder_manager = getattr(env, "recorder_manager", None)
        if recorder_manager is not None and "replay" in getattr(recorder_manager, "active_terms", []):
            state_prev = recorder_manager.get_term("replay").prev_s.clone()
        default = robot.data.default_joint_pos[0]
        params = WorldPointRobotParams(
            default_joint_pos=default,
            soft_joint_limits=robot.data.soft_joint_pos_limits[0],
            dt=float(env.step_dt),
        )
        terms = world_point_reward_terms(
            state_prev, state_next, command.target_position_world, action,
            previous_action, params, fk, tcp_indices,
        )
        tcp_next = tcp_position_world(state_next, fk, tcp_indices)
        command.record_real_step(torch.linalg.vector_norm(tcp_next - command.target_position_world, dim=-1))
        cache = {"step": step, "terms": terms}
        env._supermimic_world_point_reward_cache = cache
    if term_name not in cache["terms"]:
        raise KeyError(f"unknown world_point_reach reward {term_name!r}")
    return cache["terms"][term_name]
