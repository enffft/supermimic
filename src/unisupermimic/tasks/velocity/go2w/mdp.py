"""Task-owned proprioception, compact state semantics, and shared reward heads."""

from copy import deepcopy
from dataclasses import replace

import torch

from unisupermimic.rl.algos.supermimic.geometry import invariant_features, matvec
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp

from .contract import DEFAULT_Q, LAYOUT, POLICY_SPEC, MixedControlCfg, soft_limits
from .reward_core import VelocityRewardCfg, velocity_reward_terms, velocity_reward_total
from .terrain_plane import terrain_plane_normal


class Go2WVelocityMdp(Go2VelocityMdp):
    # num_joints is the historical trainer's action width; StateLayout owns q/qd.
    num_joints = 16
    state_layout = LAYOUT
    policy_spec = POLICY_SPEC
    context_dim = 0  # Nominal v1 physics: no unimplemented randomization context.
    obs_dim = 50
    goal_frame_dim = 3

    def __init__(
        self,
        device="cpu",
        terrain=None,
        control=None,
        reward=None,
        terrain_modes=False,
        unified=False,
    ):
        self.device = torch.device(device)
        if not isinstance(terrain_modes, bool):
            raise ValueError("terrain_modes must be a boolean")
        self.terrain_modes = terrain_modes
        self.unified = unified
        self.goal_frame_dim = 7 if terrain_modes else 3
        self.policy_spec = replace(POLICY_SPEC, goal_dim=self.goal_frame_dim)
        self.terrain = terrain
        self.control = deepcopy(control) if control is not None else MixedControlCfg()
        self.reward_cfg = deepcopy(reward) if reward is not None else VelocityRewardCfg()
        self.control.validate()
        self.reward_cfg.validate()
        self.default_q = torch.tensor(DEFAULT_Q, device=device)
        self.limits = soft_limits(device, self.control.leg_limit_fraction)
        # Upper bound on positive rate terms, including zero-command restoration.
        c = self.reward_cfg
        self.max_step_reward = self.control.dt * sum(
            max(0, x)
            for x in (c.track_vx, c.track_vy, c.track_yaw, c.stand_still, c.terrain_alignment)
        )

    def to(self, device):
        return type(self)(
            device,
            None if self.terrain is None else self.terrain.to(device),
            self.control,
            self.reward_cfg,
            self.terrain_modes,
            self.unified,
        )

    @staticmethod
    def validate_learner_config(cfg):
        required = dict(
            goal_mode="direct",
            hist=5,
            goal_frames=1,
            wm_structured=True,
            terrain_training=True,
            terrain_conditioning=True,
            terrain_sensor_grad=True,
            checkpoint_replay=True,
        )
        for name, expected in required.items():
            if getattr(cfg, name) != expected:
                raise ValueError(f"Go2W v2 requires algo.{name}={expected!r}")
        if cfg.driver_enabled:
            raise ValueError("Go2 collection drivers cannot control Go2W mixed actions")

    def obs(self, state, previous_action, context, frames):
        _, rotation, _, angular, q, qd = state
        return torch.cat(
            (
                q - self.default_q,
                qd[..., :12],
                qd[..., 12:],
                matvec(rotation.transpose(-1, -2), angular),
                -rotation[..., 2, :],
                previous_action,
            ),
            -1,
        )

    def terrain_normal(self, state):
        scan = (
            state[0].new_zeros((*state[0].shape[:-1], 77))
            if self.terrain is None
            else self.terrain.local_scan(state[0], state[1], sensor_grad=True)
        )
        return terrain_plane_normal(scan, state[1], self.reward_cfg.max_target_tilt_deg)

    def terms(self, previous, state, goal, action, previous_action, **measurements):
        if goal.shape[-1] != self.goal_frame_dim:
            raise ValueError("Go2W goal width differs from task mode protocol")
        normal = measurements.pop("terrain_normal", None)
        if normal is None:
            normal = self.terrain_normal(state)
        traversal = measurements.pop("traversal", None)
        peak = measurements.pop("success_peak", None)
        horizon = measurements.pop("rollout_horizon", 1)
        domain = measurements.get("command_domain")
        if self.unified and domain is not None:
            measurements["command_domain"] = domain.new_tensor([0, 1, 2, 3, 4, 5, 4, 5])[domain]
        terms = velocity_reward_terms(
            previous,
            state,
            goal[..., :3],
            action,
            previous_action,
            (self.reward_cfg, self.control, self.default_q, self.limits),
            base_height=self.local_height(state),
            terrain_normal=normal,
            **measurements,
        )
        if self.unified:
            from unisupermimic.rl.algos.supermimic.geometry import matvec

            from .traversal import reward_terms, wheel_centers

            if traversal is None or peak is None or domain is None:
                raise ValueError("Unified reward requires immutable traversal and domain metadata")
            cfg = self.reward_cfg
            stair = (domain >= 4).float()
            side = (goal[..., 1].abs() > 0.05) & (goal[..., 0].abs() < 0.08)
            terms["track_vx"] *= 1 - 0.4 * stair
            terms["leg_action_rate"] *= torch.where(side, 0.35, 1.0) * (1 - 0.6 * stair)
            terms["leg_joint_deviation"] *= torch.where(side, 0.35, 1.0) * (1 - 0.6 * stair)
            terms["terrain_alignment"] *= 1 - stair
            body = matvec(state[1].transpose(-1, -2), state[2])
            angular = matvec(state[1].transpose(-1, -2), state[3])
            terms["side_drift"] = (
                -cfg.side_drift * side * (body[..., 0].square() + angular[..., 2].square())
            )
            wheels = wheel_centers(state)
            ground = self.terrain.ground_height(wheels[..., :2])
            geometry_gate = torch.exp(
                -((wheels[..., 2] - ground - cfg.wheel_radius) / 0.04).square()
            ).mean(-1)
            y = matvec(state[1].transpose(-1, -2).unsqueeze(-3), wheels - state[0][..., None, :])[
                ..., 1
            ]
            target = (goal[..., 0, None] - goal[..., 2, None] * y) / cfg.wheel_radius
            rolling = torch.exp(-((state[5][..., 12:] - target) / 3.0).square()).mean(-1)
            gate = (domain == 0) * (~side) * geometry_gate
            terms["wheel_drive_consistency"] = cfg.wheel_consistency * gate * rolling
            terms.update(
                reward_terms(
                    previous,
                    state,
                    traversal,
                    peak,
                    cfg.progress_gamma,
                    cfg.stair_progress,
                    cfg.stair_success,
                    self.control.dt,
                    horizon,
                )
            )
        return terms

    def reward_transition(self, previous, state, goal, action, previous_action, **measurements):
        return self.control.dt * velocity_reward_total(
            self.terms(previous, state, goal, action, previous_action, **measurements)
        )

    @staticmethod
    def imagination_reward_kwargs(start, horizon):
        # Learner flattens trajectories in time-major H,B order.
        values = {
            "command_domain": start["command_domain"].unsqueeze(0).expand(horizon, -1).flatten()
        }
        if "traversal" in start:
            for name in ("traversal", "success_peak"):
                value = start[name]
                values[name] = value.unsqueeze(0).expand(horizon, *value.shape).flatten(0, 1)
            values["rollout_horizon"] = horizon
        return values

    @staticmethod
    def validate_resume_config(previous, cfg):
        for name in ("wm_only_iters", "wm_window", "horizon"):
            if previous.get(name) != getattr(cfg, name):
                raise ValueError(
                    f"Go2W resume cannot change the coverage/imagination schedule: {name}"
                )

    def normalize_context(self, context):
        return context

    def critic_features(self, state, frames, normalized_context):
        return torch.cat((invariant_features(state), normalized_context), -1)
