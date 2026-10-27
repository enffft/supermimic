"""The same replayed 3-D velocity goal is held fixed across imagined BPTT."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.imagination.rollout import imagine
from supermimic.imagination.tasks import VelocityCommandTask
from supermimic.tasks.velocity_reward import VelocityRewardCfg, VelocityRobotParams
from supermimic.world_model.heads import analytic_observation


class _Heads:
    anchor_index = 0

    @staticmethod
    def free_termination_features(root_pos, root_quat, base_height):
        z = root_pos[..., 2]
        return torch.stack([float(base_height) - z, torch.zeros_like(z)], dim=-1)


class _GoalSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.goals: list[torch.Tensor] = []

    def forward(self, obs_hist, goal):
        self.goals.append(goal.detach().clone())
        return obs_hist.new_zeros(obs_hist.shape[0], 12)


def test_replayed_velocity_goal_is_the_3d_goal_for_every_imagination_step():
    torch.manual_seed(4)
    device = torch.device("cpu")
    batch, nj = 3, 12
    fixed_goal = torch.tensor([0.3, -0.05, 0.2])
    buf = ReplayBuffer(num_envs=1, capacity=32, state_dim=37, act_dim=nj, xi_dim=16,
                      device=device, goal_dim=3)
    state = torch.zeros(1, 37)
    state[:, 2] = 0.29
    state[:, 3] = 1.0
    action = torch.zeros(1, nj)
    xi = torch.zeros(1, 16)
    for t in range(12):
        buf.add(
            torch.tensor([0]), s=state, a=action, a_exec=action, xi=xi,
            r=torch.zeros(1), done=torch.zeros(1), frame=torch.zeros(1, dtype=torch.long),
            clip=torch.zeros(1, dtype=torch.long), terminal=torch.zeros(1, dtype=torch.bool),
            ep_start=torch.tensor([t == 0]), goal=fixed_goal[None],
        )
    start = buf.sample_starts(batch, hist=10)
    torch.testing.assert_close(start.goal, fixed_goal.expand(batch, -1))

    q0 = torch.zeros(nj)
    params = VelocityRobotParams(
        default_joint_pos=q0,
        soft_joint_limits=torch.stack([torch.full((nj,), -2.0), torch.full((nj,), 2.0)], dim=-1),
        action_scales=torch.full((nj,), 0.25), kp=torch.full((nj,), 40.0),
        kd=torch.ones(nj), effort_limits=torch.full((nj,), 23.5),
        cfg=VelocityRewardCfg(),
    )
    task = VelocityCommandTask(params=params, dt=0.02)
    wm = SimpleNamespace(
        spec=SimpleNamespace(nj=nj, wheel_indices=()),
        heads=_Heads(),
        cfg=SimpleNamespace(termination_base_height=0.42),
    )
    spy = _GoalSpy()

    def step_fn(s, a, a_prev, context, goal):
        s_next = s.clone()
        r_terms = task.reward_terms(s, s_next, goal, a, a_prev)
        reward = torch.stack([r_terms[name] for name in task.reward_term_names], dim=0).sum(0)
        terms = torch.stack([r_terms[name] for name in task.reward_term_names], dim=0)
        obs_clean = analytic_observation(s_next, a, context, q0)
        return s_next, reward, terms, torch.zeros_like(reward), obs_clean, s.new_zeros(batch, 7)

    rollout = imagine(wm, spy, task, start, horizon=32, obs_noise=False, step_fn=step_fn)
    assert rollout.goals.shape == (33, batch, 3)
    torch.testing.assert_close(rollout.goals, fixed_goal.expand(33, batch, -1))
    assert len(spy.goals) == 32
    torch.testing.assert_close(torch.stack(spy.goals), fixed_goal.expand(32, batch, -1))
