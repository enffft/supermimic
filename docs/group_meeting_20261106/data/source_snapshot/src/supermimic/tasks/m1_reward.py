"""Prior-free M1 loco/reach rewards, shared by real and imagined dynamics.

Reward design references (formula inspiration, not a copied implementation):
RoboDuet, go1_gym/envs/rewards/rewards.py (velocity kernels, reach, regularizers);
Learning Whole-Body Loco-Manipulation ... Wheeled-Quadrupedal-Manipulator (RAL).
All returned terms are rates; adapters integrate control dt exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

from supermimic.kinematics.rotations import quat_apply_inverse, heading_quat
from supermimic.motion_prior.velocity import velocity_command_errors
from supermimic.tasks.world_point_reward import tcp_position_world


@dataclass
class M1TargetCfg:
    # Legacy checkpoints retain their original TCP-offset, episode-fixed task.
    sampling: str = "tcp_offset"
    radius_m: float = 3.0
    minimum_height_m: float = 0.03
    maximum_height_m: float = 1.10
    resample_on_success: bool = False

    def __post_init__(self):
        if self.sampling not in ("tcp_offset", "base_disk"):
            raise ValueError("M1 target sampling must be tcp_offset or base_disk")
        if self.radius_m <= 0 or not 0 <= self.minimum_height_m < self.maximum_height_m:
            raise ValueError("Invalid M1 target radius/heights")


def sample_m1_disk_targets(root_position_local, cfg: M1TargetCfg):
    """Uniform AREA density in a base-centred disk; z is ground-relative."""
    n = root_position_local.shape[0]
    random = torch.rand(n, 3, device=root_position_local.device, dtype=root_position_local.dtype)
    radius = cfg.radius_m * random[:, 0].sqrt()
    theta = 2 * torch.pi * random[:, 1]
    return torch.stack((root_position_local[:, 0] + radius * theta.cos(),
                        root_position_local[:, 1] + radius * theta.sin(),
                        cfg.minimum_height_m + random[:, 2] * (cfg.maximum_height_m - cfg.minimum_height_m)), -1)


@dataclass
class M1RewardCfg:
    linear_tracking: float = 3.0
    yaw_tracking: float = 1.0
    linear_sigma: float = 0.15
    yaw_sigma: float = 0.25
    reach_coarse: float = 2.0
    reach_fine: float = 1.0
    reach_progress: float = 0.25
    reach_hold: float = 1.0
    reach_coarse_sigma: float = 0.50
    reach_fine_sigma: float = 0.08
    hold_speed_sigma: float = 0.10
    upright: float = 0.5
    height: float = 0.5
    vertical_velocity: float = -0.2
    roll_pitch_rate: float = -0.05
    leg_pose: float = -0.02
    arm_carry: float = -0.01
    action_rate: float = -0.01
    joint_limit: float = -1.0
    joint_velocity: float = -1.0e-5
    joint_acceleration: float = -1.0e-7
    # Explicitly a control-effort surrogate; no claim of torque/energy parity.
    control_effort: float = -0.005
    mani_base_speed: float = -0.1
    # Rates, except success_bonus which is a one-time reward per completed goal.
    distance_time_weight: float = 0.0
    distance_time_scale_m: float = 0.5
    success_bonus: float = 0.0
    success_distance_m: float = 0.08

    def __post_init__(self):
        for name in ("linear_sigma", "yaw_sigma", "reach_coarse_sigma", "reach_fine_sigma", "hold_speed_sigma",
                     "distance_time_scale_m", "success_distance_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.distance_time_weight < 0 or self.success_bonus < 0:
            raise ValueError("M1 time cost and success bonus must be nonnegative")


M1_REWARD_NAMES = (
    "l_linear", "l_yaw", "m_reach_coarse", "m_reach_fine", "m_progress", "m_hold",
    "upright", "height", "vertical_velocity", "roll_pitch_rate", "leg_pose", "l_arm_carry",
    "action_rate", "joint_limit", "joint_velocity", "joint_acceleration", "control_effort", "m_base_speed",
    "m_distance_time", "m_success",
)


def m1_goal_reached(state, context, fk, tcp_body_indices, distance_m, fall_height=.20, tilt_feature_threshold=0.6579798566743313):
    """Same first stable arrival event in real simulation and imagination."""
    distance = (tcp_position_world(state, fk, tcp_body_indices) - context[..., 4:7]).norm(dim=-1)
    gravity = state.new_zeros(state.shape[:-1] + (3,)); gravity[..., 2] = -1.
    gravity_b = quat_apply_inverse(state[..., 3:7], gravity)
    stable = (state[..., 2] >= fall_height) & ((1. + gravity_b[..., 2]) < tilt_feature_threshold)
    return (context[..., 0] >= .5) & (distance < distance_m) & stable


def m1_actor_goal(state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Replay context [task(0=L,1=M), vx,vy,wz, local-world xyz] -> AMP Goal8.

    Matches task_goal8 without data-dependent Python checks inside compiled BPTT.
    World targets stay fixed; heading-relative position is recomputed each step.
    """
    is_l = context[..., :1] < 0.5
    delta = quat_apply_inverse(heading_quat(state[..., 3:7]), context[..., 4:7] - state[..., :3])
    return torch.cat((torch.where(is_l, context[..., 1:4], 0.0),
                      torch.where(is_l, 0.0, delta), is_l.to(state.dtype), (~is_l).to(state.dtype)), -1)


def m1_reward_terms(s_prev, s_next, context, action, previous_action, *,
                    default_joint_pos, soft_limits, base_height, fk, tcp_body_indices,
                    cfg: M1RewardCfg, dt: float = 0.02,
                    fall_height: float = .20, tilt_feature_threshold: float = 0.6579798566743313):
    """Signed progress + multi-scale reaching avoids a sparse-only objective.

    No per-wheel sign heuristic: motion is scored using physical base velocity.
    Wheels are excluded from position-limit/pose penalties, retained in rate costs.
    """
    is_l = (context[..., 0] < 0.5).to(s_next.dtype)
    is_m = 1.0 - is_l
    xy_error, xy_sq, yaw_error, vz = velocity_command_errors(s_next, context[..., 1:4])
    tcp_prev = tcp_position_world(s_prev, fk, tcp_body_indices)
    tcp_next = tcp_position_world(s_next, fk, tcp_body_indices)
    # Inactive targets must not introduce arbitrary gradients into L episodes.
    target = context[..., 4:7]
    d_prev = (tcp_prev - target).norm(dim=-1)
    d_next = (tcp_next - target).norm(dim=-1)
    tcp_speed_sq = ((tcp_next - tcp_prev) / dt).square().sum(-1)
    near = torch.exp(-d_next.square() / cfg.reach_fine_sigma**2)
    gravity = torch.zeros_like(s_next[..., :3]); gravity[..., 2] = -1.0
    gravity_b = quat_apply_inverse(s_next[..., 3:7], gravity)
    q, qd = s_next[..., 7:29], s_next[..., 35:57]
    qdd = (qd - s_prev[..., 35:57]) / dt
    position_mask = q.new_ones(22); position_mask[[3, 7, 11, 15]] = 0.0
    leg_ids = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    limit_error = (F.relu(soft_limits[..., 0] - q).square() + F.relu(q - soft_limits[..., 1]).square()) * position_mask
    return {
        "l_linear": is_l * cfg.linear_tracking * torch.exp(-xy_sq / cfg.linear_sigma**2),
        "l_yaw": is_l * cfg.yaw_tracking * torch.exp(-yaw_error.square() / cfg.yaw_sigma**2),
        "m_reach_coarse": is_m * cfg.reach_coarse * torch.exp(-d_next.square() / cfg.reach_coarse_sigma**2),
        "m_reach_fine": is_m * cfg.reach_fine * near,
        "m_progress": is_m * cfg.reach_progress * torch.tanh((d_prev - d_next) / (dt * 0.5)),
        "m_hold": is_m * cfg.reach_hold * near * torch.exp(-tcp_speed_sq / cfg.hold_speed_sigma**2),
        "upright": cfg.upright * torch.exp(-(1.0 + gravity_b[..., 2]) / 0.1),
        "height": cfg.height * torch.exp(-((s_next[..., 2] - base_height) / 0.10).square()),
        "vertical_velocity": cfg.vertical_velocity * vz.square(),
        "roll_pitch_rate": cfg.roll_pitch_rate * s_next[..., 32:34].square().sum(-1),
        "leg_pose": cfg.leg_pose * (q[..., leg_ids] - default_joint_pos[..., leg_ids]).square().sum(-1),
        "l_arm_carry": is_l * cfg.arm_carry * (q[..., 16:22] - default_joint_pos[..., 16:22]).square().sum(-1),
        "action_rate": cfg.action_rate * (action - previous_action).square().mean(-1),
        "joint_limit": cfg.joint_limit * limit_error.sum(-1),
        "joint_velocity": cfg.joint_velocity * qd.square().mean(-1),
        "joint_acceleration": cfg.joint_acceleration * qdd.square().mean(-1),
        "control_effort": cfg.control_effort * action.square().mean(-1),
        "m_base_speed": is_m * cfg.mani_base_speed * (xy_error.square().sum(-1) + s_next[..., 34].square()),
        "m_distance_time": -is_m * cfg.distance_time_weight * d_next / (d_next + cfg.distance_time_scale_m),
        "m_success": (cfg.success_bonus / dt) * m1_goal_reached(
            s_next, context, fk, tcp_body_indices, cfg.success_distance_m, fall_height, tilt_feature_threshold).to(s_next.dtype),
    }


def m1_reward_term(env, term_name: str, cfg: M1RewardCfg, base_height: float) -> torch.Tensor:
    """RewardManager adapter, including pre-reset terminal transitions."""
    from supermimic.envs.state_io import StateIO
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.mjlab_assets import load_robot_spec_model
    from supermimic.tasks.world_point_reward import m1_tcp_proxy_indices

    step = int(env.common_step_counter)
    cache = getattr(env, "_supermimic_m1_reward_cache", None)
    if cache is None or cache["step"] != step:
        if not hasattr(env, "_supermimic_m1_reward_io"):
            env._supermimic_m1_reward_io = StateIO(env)
            env._supermimic_m1_reward_fk = G1Kinematics(load_robot_spec_model("m1_piper"), device=env.device)
        robot = env.scene["robot"]
        command = env.command_manager.get_term("m1_prior_joint")
        state_next = env._supermimic_m1_reward_io.read()
        recorder = env.recorder_manager.get_term("replay") if "replay" in env.recorder_manager.active_terms else None
        state_prev = recorder.prev_s if recorder is not None else getattr(env, "_supermimic_m1_reward_previous", state_next)
        if recorder is None:
            done = env.reset_terminated | env.reset_time_outs
            state_prev = torch.where(done[:, None], state_next, state_prev)
        # Recorder updates its source snapshot after reward/reset processing.
        # Keep the command that actually produced this transition for diagnostics.
        context = (recorder.prev_goal if recorder is not None else command.goal).clone()
        fk = env._supermimic_m1_reward_fk
        terms = m1_reward_terms(
            state_prev, state_next, context, env.action_manager.action, env.action_manager.prev_action,
            default_joint_pos=robot.data.default_joint_pos[0], soft_limits=robot.data.soft_joint_pos_limits[0],
            base_height=base_height, fk=fk, tcp_body_indices=m1_tcp_proxy_indices(fk),
            cfg=cfg, dt=float(env.step_dt),
            fall_height=command.cfg.success_minimum_base_height_m,
            tilt_feature_threshold=1. - math.cos(command.cfg.success_maximum_tilt_rad),
        )
        cache = {"step": step, "terms": terms, "state_prev": state_prev.clone(),
                 "state_next": state_next, "context": context,
                 "action": env.action_manager.action.clone(), "previous_action": env.action_manager.prev_action.clone()}
        env._supermimic_m1_reward_cache = cache
        env._supermimic_m1_reward_previous = state_next.clone()
    return cache["terms"][term_name]
