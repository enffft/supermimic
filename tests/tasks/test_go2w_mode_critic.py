"""Task mode privacy, reset labels and differentiable imagined Critic scans."""

from types import SimpleNamespace

import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.rl.algos.supermimic.networks import Critic, Policy, WorldModel
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2w.contract import DEFAULT_Q, LAYOUT
from unisupermimic.tasks.velocity.go2w.manager_terms import VelocityCommand
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp


def state(batch=6):
    return (
        torch.tensor([[0.0, 0.0, 0.42]]).expand(batch, -1).clone(),
        torch.eye(3).expand(batch, 3, 3).clone(),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.tensor([DEFAULT_Q]).expand(batch, -1).clone(),
        torch.zeros(batch, 16),
    )


def test_actor_mode_privacy():
    mdp = Go2WVelocityMdp(terrain=TerrainHeightField.flat(), terrain_modes=True)
    assert mdp.goal_frame_dim == mdp.policy_spec.goal_dim == 7
    assert mdp.to("cpu").goal_frame_dim == 7
    s = state()
    frame = torch.zeros(6, dtype=torch.long)
    prior = torch.zeros(6, 16)
    obs = mdp.obs(s, prior, torch.empty(6, 0), frame)
    mdp.terrain = TerrainHeightField.flat(height=100)
    torch.testing.assert_close(obs, mdp.obs(s, prior, torch.empty(6, 0), frame), atol=0, rtol=0)
    policy = Policy(
        50, 7, hist=5, num_joints=16, hidden=(8,), goal_mode="direct", direct_spec=mdp.policy_spec
    )
    assert policy.dec[0].in_features == 257
    goal = torch.tensor([[0.3, 0, 0, 1, 0, 0, 0.0]]).expand(6, -1).clone()
    history = obs[:, None].expand(-1, 5, -1)
    result = policy(history, goal)
    # Hidden target and terrain changes cannot enter the Actor's exact inputs.
    mdp.traversal_target = torch.randn(6, 3) * 100
    mdp.terrain = TerrainHeightField.flat(height=-200)
    torch.testing.assert_close(result, policy(history, goal), atol=0, rtol=0)
    buffer = ReplayBuffer(
        8, 6, "cpu", num_joints=16, context_dim=0, goal_dim=7, state_layout=LAYOUT
    )
    transition = dict(
        s=flatten_state(s),
        s1=flatten_state(s),
        a=prior,
        a_prev=prior,
        xi=torch.empty(6, 0),
        r=torch.zeros(6),
        f=frame,
        ep_step=frame.int(),
        done=frame.bool(),
        g=goal,
    )
    for _ in range(8):
        buffer.add(transition)
    windows = buffer.sample_windows(16, 8)
    torch.testing.assert_close(windows["g"], goal[:1, None].expand(16, 8, 7))
    learner = object.__new__(SuperMimicLearner)
    learner.cfg = SimpleNamespace(goal_mode="direct")
    learner.pi = policy
    learner.mdp = mdp
    learner._wm_step = lambda state, action, context: state
    imagined = learner._imag_step(
        s, history, prior, frame, torch.empty(6, 0), torch.empty(6, 0), goal
    )
    torch.testing.assert_close(imagined[-1], goal, atol=0, rtol=0)


def test_reset_mode_onehot_and_partial_reset():
    command = SimpleNamespace(
        curriculum=None,
        f=torch.zeros(6, dtype=torch.long),
        g=torch.zeros(6, 7),
        command_domain=torch.arange(6),
        command_mode=torch.zeros(6, dtype=torch.long),
        crossing_episode=torch.zeros(6, dtype=torch.bool),
        time_left=torch.zeros(6),
        cfg=SimpleNamespace(fixed_command=None, terrain_mode_conditioning=True),
        sampler=SimpleNamespace(
            sample=lambda domain, level: (
                torch.zeros(6, 3),
                torch.zeros(6, dtype=torch.long),
                torch.zeros(6, dtype=torch.bool),
                torch.ones(6),
            )
        ),
    )
    VelocityCommand._resample_command(command, torch.ones(6, dtype=torch.bool))
    expected = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 2, 3, 3]), 4).float()
    torch.testing.assert_close(command.g[:, 3:], expected)
    command.command_domain[:2] = 5
    VelocityCommand._resample_command(
        command, torch.tensor([True, True, False, False, False, False])
    )
    expected[:2] = expected[-1].clone()
    torch.testing.assert_close(command.g[:, 3:], expected)


def test_critic_scan_and_actor_bootstrap_gradient():
    grid = torch.linspace(-2, 2, 5)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    field = TerrainHeightField(0.1 * x + 0.2 * y, (0.0, 0.0, 0.0), (2.0, 2.0))
    mdp = Go2WVelocityMdp(terrain=field, terrain_modes=True)
    learner = object.__new__(SuperMimicLearner)
    learner.cfg = SimpleNamespace(critic_terrain_conditioning=True)
    learner.mdp = mdp
    s = list(state(batch=1))
    s[0].requires_grad_()
    frame = torch.zeros(1, dtype=torch.long)
    context = torch.empty(1, 0)
    goal = torch.tensor([[0.3, 0, 0, 1, 0, 0, 0.0]])
    features = learner._critic_input(tuple(s), frame, context, goal)
    assert features.shape == (1, 125)
    torch.testing.assert_close(features[:, -77:], field.local_scan(s[0], s[1]))
    scan_gradient = torch.autograd.grad(features[:, -77:].sum(), s[0])[0]
    assert torch.isfinite(scan_gradient).all() and (scan_gradient.abs() > 0).all()
    s[0] = s[0].detach() + torch.tensor([[0.2, 0.0, 0.1]])
    updated = learner._critic_input(tuple(s), frame, context, goal)
    assert not torch.allclose(features[:, -77:], updated[:, -77:])
    policy = Policy(
        50, 7, hist=5, num_joints=16, hidden=(8,), goal_mode="direct", direct_spec=mdp.policy_spec
    )
    wm = WorldModel(
        hidden=(8,),
        num_joints=16,
        context_dim=0,
        structured=True,
        terrain_dim=77,
        state_layout=LAYOUT,
    )
    with torch.no_grad():
        wm.net[-1].weight.normal_(std=0.05)
    wm.requires_grad_(False)
    critic = Critic(125, hidden=(8,))
    critic.requires_grad_(False)
    obs = mdp.obs(tuple(s), torch.zeros(1, 16), context, frame)
    action = policy(obs[:, None].expand(-1, 5, -1), goal)
    predicted = wm.step(tuple(s), action, context, field.local_scan(s[0], s[1]))
    value = critic(learner._critic_input(predicted, frame + 1, context, goal))
    value.sum().backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in critic.parameters())
