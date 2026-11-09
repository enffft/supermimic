"""Direct velocity MDP: no motion reference, encoder, phase or privileged actor input."""

from dataclasses import replace

import torch

from unisupermimic.rl.algos.supermimic.geometry import invariant_features, mat_to_quat, matvec

from .contract import DEFAULT_Q, DT, EFFORT, soft_limits
from .reward_core import (
    VelocityRewardCfg,
    VelocityRobotParams,
    velocity_reward_terms,
    velocity_reward_total,
)


class Go2VelocityMdp:
    num_joints = 12
    context_dim = 16
    obs_dim = 42
    goal_frame_dim = 3
    max_step_reward = 5.2 * DT

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.default_q = torch.tensor(DEFAULT_Q, device=device)
        self.params = VelocityRobotParams(
            self.default_q,
            soft_limits(device),
            torch.full((12,), 0.25, device=device),
            torch.full((12,), 40.0, device=device),
            torch.ones(12, device=device),
            torch.tensor(EFFORT, device=device),
            dt=DT,
            cfg=VelocityRewardCfg(tracking_sigma_sq=0.09, zero_speed_baseline=True),
        )

    def to(self, device):
        return type(self)(device)

    def obs(self, state, previous_action, context, frames):
        p, R, v, w, q, qd = state
        # w is world in UniSuperMimic State, body in source State37.
        w_body = matvec(R.transpose(-1, -2), w)
        gravity = -R[..., 2, :]
        return torch.cat(
            (q + context[..., 1:13] - self.default_q, qd, w_body, gravity, previous_action), -1
        )

    def add_noise(self, obs):
        return obs

    def goal(self, frames, num_frames):
        raise RuntimeError("Velocity commands must come from replay g, never motion frame indices")

    def reward(self, state, frames, action, previous_action):
        raise RuntimeError("Velocity reward requires explicit command and preceding state")

    @staticmethod
    def source_state(state):
        p, R, v, w, q, qd = state
        return torch.cat((p, mat_to_quat(R), q, v, matvec(R.transpose(-1, -2), w), qd), -1)

    def terms(self, previous, state, goal, action, previous_action, **measurements):
        return velocity_reward_terms(
            self.source_state(previous),
            self.source_state(state),
            goal,
            action,
            previous_action,
            self.params,
            **measurements,
        )

    def reward_transition(self, previous, state, goal, action, previous_action, **measurements):
        return DT * velocity_reward_total(
            self.terms(previous, state, goal, action, previous_action, **measurements)
        )

    def hard_termination(self, state):
        return (state[0][..., 2] < 0.17) | (state[1][..., 2, 2] < 0.3420201433256688)

    def soft_termination(self, state, frames):
        height = torch.sigmoid((0.17 - state[0][..., 2]) / 0.02)
        tilt = torch.sigmoid((0.3420201433256688 - state[1][..., 2, 2]) / 0.05)
        return 1 - (1 - height) * (1 - tilt)

    def critic_features(self, state, frames, normalized_context):
        return torch.cat((invariant_features(state), normalized_context), -1)

    def normalize_context(self, context):
        # Same physical meaning as source: friction, 12 encoder biases, base COM xyz.
        return torch.cat(
            (
                (context[..., :1] - 0.75) / 0.45,
                context[..., 1:13] / 0.01,
                context[..., 13:16] / 0.025,
            ),
            -1,
        )

    def valid_imagination_start(self, frames, horizon):
        return torch.ones_like(frames, dtype=torch.bool)
