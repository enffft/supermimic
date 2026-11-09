import pytest
import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, so3_exp, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp


def state(n=4):
    mdp = Go2VelocityMdp()
    return (
        torch.tensor([[0.0, 0.0, 0.42]]).expand(n, 3).clone(),
        torch.eye(3).expand(n, 3, 3).clone(),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        mdp.default_q.expand(n, 12).clone(),
        torch.zeros(n, 12),
    )


def test_state_world_to_source_and_observation():
    mdp = Go2VelocityMdp()
    s = list(state())
    s[1] = so3_exp(torch.tensor([[0.3, -0.2, 0.4]]).expand(4, 3))
    s[3] = torch.randn(4, 3)
    xi = torch.randn(4, 16) * 0.01
    a = torch.randn(4, 12)
    o = mdp.obs(tuple(s), a, xi, torch.zeros(4, dtype=torch.long))
    assert o.shape == (4, 42)
    torch.testing.assert_close(o[:, :12], s[4] + xi[:, 1:13] - mdp.default_q)
    torch.testing.assert_close(
        o[:, 24:27], (s[1].transpose(-1, -2) @ s[3].unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(o[:, 27:30], -s[1][:, 2, :])
    torch.testing.assert_close(mdp.source_state(tuple(s))[:, 22:25], o[:, 24:27])
    torch.testing.assert_close(
        flatten_state(unflatten_state(flatten_state(tuple(s)))), flatten_state(tuple(s))
    )


def test_reward_transition_acceleration_and_gradients():
    mdp = Go2VelocityMdp()
    s = state()
    nxt = list(s)
    nxt[5] = torch.ones(4, 12) * 0.1
    a = torch.zeros(4, 12, requires_grad=True)
    goal = torch.tensor([[0.5, 0.0, 0.0]]).expand(4, 3)
    terms = mdp.terms(s, tuple(nxt), goal, a, torch.zeros_like(a))
    torch.testing.assert_close(terms["joint_acceleration"], torch.full((4,), -2.5e-7 * 12 * 25))
    reward = mdp.reward_transition(s, tuple(nxt), goal, a, torch.zeros_like(a))
    reward.sum().backward()
    assert torch.isfinite(a.grad).all()
    with pytest.raises(RuntimeError, match="replay g"):
        mdp.goal(torch.zeros(4), 1)


def test_replay_command_values_survive_resampling_and_ring_wrap():
    rb = ReplayBuffer(5, 2, "cpu", num_joints=12, context_dim=16, goal_dim=3)
    for t in range(8):
        row = {
            k: torch.zeros((2,) + (tuple(v.shape[2:])), dtype=v.dtype) for k, v in rb.data.items()
        }
        row["g"][:] = t
        row["ep_step"][:] = t % 3
        row["s"][:, 0] = t
        rb.add(row)
        row["g"].fill_(-99)
    start, hist = rb.sample_starts(200, 5)
    torch.testing.assert_close(start["g"][:, 0], start["s"][:, 0])
    assert (start["g"] >= 3).all()
    assert (hist["s"][:, -1, 0] == start["s"][:, 0]).all()


def test_actor_history_layout_and_range():
    pi = Policy(42, 3, hist=5, num_joints=12, hidden=(32, 16), goal_mode="direct")
    assert not hasattr(pi, "enc")
    hist = torch.randn(4, 5, 42, requires_grad=True)
    goal = torch.randn(4, 3)
    parts = hist.split((12, 12, 3, 3, 12), -1)
    expected = torch.tanh(
        pi.dec(torch.cat((torch.cat([p.flatten(1) for p in parts], -1) * pi.obs_scale, goal), -1))
    )
    actual = pi(hist, goal)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (actual.abs() <= 1).all()
    actual.sum().backward()
    assert torch.isfinite(hist.grad).all()


@pytest.mark.parametrize("horizon", [1, 8, 32])
def test_wm_prediction_and_gradients(horizon):
    s = state()
    wm = WorldModel(hidden=(32, 32), num_joints=12, context_dim=16)
    torch.nn.init.normal_(wm.net[-1].weight, std=0.0001)
    a = torch.zeros(4, 12, requires_grad=True)
    for _ in range(horizon):
        s = wm.step(s, a, torch.zeros(4, 16))
    assert all(torch.isfinite(t).all() for t in s)
    (s[0].square().sum() + s[4].square().sum()).backward()
    assert torch.isfinite(a.grad).all()
