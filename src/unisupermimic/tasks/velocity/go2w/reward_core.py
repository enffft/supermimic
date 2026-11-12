"""Reward rates shared by measured transitions and differentiable imagination.

Uses only compact state, normalized executed actions, commands and optional
measured motor torque. No contacts, wheel angles or gait/rolling heuristics.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from unisupermimic.rl.algos.supermimic.geometry import matvec

from .contract import mixed_torque


@dataclass
class VelocityRewardCfg:
    track_vx: float = 2.5
    track_vy: float = 1.5
    track_yaw: float = 2.0
    sigma_x: float = 0.25
    sigma_y: float = 0.25
    sigma_yaw: float = 0.30
    zero_speed_baseline: bool = True
    stand_still: float = 0.5
    stop_command_sigma: float = 0.05
    stop_velocity_sigma: float = 0.15
    terrain_alignment: float = 0.4
    terrain_alignment_sigma: float = 0.16
    max_target_tilt_deg: float = 35.0
    base_lin_vel_z: tuple[float, ...] = (-0.50, -0.35, -0.15, -0.15, -0.08, -0.08)
    base_ang_vel_xy: float = -0.08
    base_height_low: float = -5.0
    min_safe_height: float = 0.22
    leg_joint_deviation: tuple[float, ...] = (-0.030, -0.020, -0.015, -0.015, -0.008, -0.008)
    leg_joint_limit: float = -1.0
    leg_action_rate: float = -0.01
    wheel_action_rate: float = -0.003
    leg_torque: float = -2e-5
    wheel_torque: float = -1e-5
    leg_joint_acceleration: float = -2.5e-7
    wheel_joint_acceleration: float = -2.5e-9

    def validate(self):
        import math
        from dataclasses import asdict

        if any(
            not math.isfinite(x)
            for v in asdict(self).values()
            for x in (v if isinstance(v, (tuple, list)) else (v,))
        ):
            raise ValueError("Reward parameters must be finite")
        if (
            min(
                self.sigma_x,
                self.sigma_y,
                self.sigma_yaw,
                self.stop_command_sigma,
                self.stop_velocity_sigma,
                self.terrain_alignment_sigma,
            )
            <= 0
        ):
            raise ValueError("Reward sigma is a positive standard velocity/attitude scale")

        if not 0 < self.max_target_tilt_deg < 60:
            raise ValueError("Terrain target tilt must be between zero and 60 degrees")
        for values in (self.base_lin_vel_z, self.leg_joint_deviation):
            if len(values) != 6 or any(x > 0 for x in values):
                raise ValueError("Expected six nonpositive domain penalty weights")


def velocity_reward_terms(
    previous,
    state,
    command,
    action,
    previous_action,
    params,
    *,
    executed_action=None,
    measured_torque=None,
    base_height=None,
    terrain_normal=None,
    command_domain=None,
):
    p, rotation, velocity, angular, q, qd = state
    cfg, control, default, limits = params
    domain = (
        torch.zeros_like(p[..., 0], dtype=torch.long) if command_domain is None else command_domain
    )
    vertical_weight = p.new_tensor(cfg.base_lin_vel_z)[domain]
    deviation_weight = p.new_tensor(cfg.leg_joint_deviation)[domain]
    if terrain_normal is None:
        terrain_normal = torch.zeros_like(p)
        terrain_normal[..., 2] = 1
    alignment_error = (1 - (rotation[..., :, 2] * terrain_normal).sum(-1)).clamp_min(0)
    velocity_body = matvec(rotation.transpose(-1, -2), velocity)
    angular_body = matvec(rotation.transpose(-1, -2), angular)
    observed = torch.cat((velocity_body[..., :2], angular_body[..., 2:3]), -1)
    sigma = command.new_tensor((cfg.sigma_x, cfg.sigma_y, cfg.sigma_yaw))
    track = torch.exp(-((observed - command) / sigma).square())
    if cfg.zero_speed_baseline:
        # Axis-wise subtraction cannot change the optimum at a fixed command.
        # Restore the stationary maximum continuously near zero, including zero.
        baseline = torch.exp(-(command / sigma).square())
        restore = torch.exp(-(command / cfg.stop_command_sigma).square())
        track = track - baseline * (1 - restore)
    a = (action if executed_action is None else executed_action).clamp(-1, 1)
    torque = (
        measured_torque
        if measured_torque is not None
        else mixed_torque(q, qd, a, control, default, limits)
    )
    acceleration = (qd - previous[5]) / control.dt
    rate = a - previous_action.clamp(-1, 1)
    stop_gate = torch.exp(-(command / cfg.stop_command_sigma).square().sum(-1))
    stop = stop_gate * torch.exp(
        -velocity_body[..., :2].square().sum(-1) / cfg.stop_velocity_sigma**2
        - angular_body[..., 2].square() / cfg.sigma_yaw**2
    )
    below, above = F.relu(limits[:, 0] - q), F.relu(q - limits[:, 1])
    return {
        "track_vx": cfg.track_vx * track[..., 0],
        "track_vy": cfg.track_vy * track[..., 1],
        "track_yaw": cfg.track_yaw * track[..., 2],
        "stand_still": cfg.stand_still * stop,
        "terrain_alignment": cfg.terrain_alignment
        * torch.exp(-alignment_error / cfg.terrain_alignment_sigma),
        "base_lin_vel_z": vertical_weight * velocity[..., 2].square(),
        "base_ang_vel_xy": cfg.base_ang_vel_xy * angular_body[..., :2].square().sum(-1),
        "base_height_low": cfg.base_height_low
        * F.relu(
            cfg.min_safe_height - (p[..., 2] if base_height is None else base_height)
        ).square(),
        "leg_joint_deviation": deviation_weight * (q - default).square().sum(-1),
        "leg_joint_limit": cfg.leg_joint_limit * (below.square() + above.square()).sum(-1),
        "leg_action_rate": cfg.leg_action_rate * rate[..., :12].square().sum(-1),
        "wheel_action_rate": cfg.wheel_action_rate * rate[..., 12:].square().sum(-1),
        "leg_torque": cfg.leg_torque * torque[..., :12].square().sum(-1),
        "wheel_torque": cfg.wheel_torque * torque[..., 12:].square().sum(-1),
        "leg_joint_acceleration": cfg.leg_joint_acceleration
        * acceleration[..., :12].square().sum(-1),
        "wheel_joint_acceleration": cfg.wheel_joint_acceleration
        * acceleration[..., 12:].square().sum(-1),
    }


def velocity_reward_total(terms):
    return torch.stack(tuple(terms.values()), 0).sum(0)
