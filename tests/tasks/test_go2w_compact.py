"""Small CPU tensor tests. Written for later execution; no physics fixtures."""

from types import SimpleNamespace

import pytest
import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.rl.contracts.layout import StateLayout
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2w.contract import (
    DEFAULT_Q, JOINTS, LAYOUT, MODEL, POLICY_SPEC, WHEEL_JOINTS, MixedControlCfg,
    mixed_torque, soft_limits,
)
from unisupermimic.tasks.velocity.go2w.env import validate_model
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp


def state():
    return (torch.tensor([[0., 0., 0.42]]), torch.eye(3)[None],
            torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([DEFAULT_Q]), torch.zeros(1, 16))


def test_explicit_serialization_and_legacy_layouts():
    assert StateLayout.legacy(12).state_dim == 37
    assert StateLayout.legacy(29).state_dim == 71
    assert (LAYOUT.state_dim, LAYOUT.increment_dim, LAYOUT.action_dim) == (41, 40, 16)
    s = list(state())
    s[5] = torch.arange(16, dtype=torch.float32)[None]
    value = flatten_state(tuple(s), LAYOUT)
    assert value.shape == (1, 41)
    torch.testing.assert_close(value[:, 13:25], s[4])
    torch.testing.assert_close(value[:, 25:41], s[5])
    for expected, actual in zip(s, unflatten_state(value, LAYOUT)):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        unflatten_state(value, StateLayout.legacy(12))
    buffer = ReplayBuffer(2, 1, "cpu", num_joints=16, context_dim=0, goal_dim=3, state_layout=LAYOUT)
    assert buffer.data["s"].shape == (2, 1, 41)
    assert buffer.data["a"].shape == (2, 1, 16)


def test_structured_world_drift_and_wheel_speed():
    wm = WorldModel(hidden=(8,), num_joints=16, context_dim=0, terrain_dim=77,
                    state_layout=LAYOUT, structured=True)
    assert (wm.in_dim, wm.out_dim) == (134, 40)
    s = list(state())
    s[1] = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    s[2] = torch.tensor([[0., 1., 0.]])  # world +Y, body +X
    s[5] = torch.cat((torch.ones(1, 12), torch.full((1, 4), 100.)), -1)
    inc = torch.zeros(1, 40, requires_grad=True)
    result = wm.integrate(tuple(s), inc)
    torch.testing.assert_close(result[0], s[0] + torch.tensor([[0., .02, 0.]]))
    torch.testing.assert_close(result[4], s[4] + .02)
    torch.testing.assert_close(result[5][..., 12:], s[5][..., 12:])
    assert result[4].shape[-1] == 12
    torch.testing.assert_close(wm.targets(tuple(s), result), inc, atol=1e-5, rtol=1e-5)
    for _ in range(31):
        result = wm.integrate(result, inc)
    result[0].sum().backward()
    assert inc.grad is not None and torch.isfinite(inc.grad).all()
    assert inc.grad[..., 6:9].abs().sum() > 0
    # Historical input/output widths and q/qd integration stay unchanged.
    old = WorldModel(hidden=(8,), num_joints=12, context_dim=16)
    assert (old.in_dim, old.out_dim) == (65, 36)


def test_actor_term_major_history_is_proprioceptive():
    mdp = Go2WVelocityMdp()
    s = list(state())
    prior = torch.arange(16, dtype=torch.float32)[None] / 16
    obs = mdp.obs(tuple(s), prior, torch.zeros(1, 0), torch.zeros(1, dtype=torch.long))
    assert obs.shape == (1, 50)
    torch.testing.assert_close(obs[..., -16:], prior)
    s[0] = s[0] + 1000
    s[2] = s[2] + 500
    torch.testing.assert_close(obs, mdp.obs(tuple(s), prior, torch.zeros(1, 0), None))
    policy = Policy(50, 3, hist=5, num_joints=16, hidden=(8,), goal_mode="direct", direct_spec=POLICY_SPEC)
    assert policy.dec[0].in_features == 253
    assert not hasattr(policy, "enc")
    history = torch.arange(250, dtype=torch.float32).reshape(1, 5, 50)
    captured = []
    handle = policy.dec[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    action = policy(history, torch.tensor([[.1, .2, .3]]))
    handle.remove()
    expected = torch.cat([part.flatten(-2) for part in history.split(POLICY_SPEC.widths, -1)], -1)
    torch.testing.assert_close(captured[0][..., :250], expected * policy.obs_scale)
    assert action.shape == (1, 16) and (action.abs() <= 1).all()
    legacy = Policy(42, 3, hist=5, num_joints=12, hidden=(8,), goal_mode="direct")
    assert legacy.dec[0].in_features == 213


def test_mixed_torque_and_reward_properties():
    mdp = Go2WVelocityMdp()
    s = state()
    a = torch.zeros(1, 16)
    a[..., 12:] = 1
    torque = mixed_torque(s[4], s[5], a, MixedControlCfg(), mdp.default_q, soft_limits())
    torch.testing.assert_close(torque[..., :12], torch.zeros(1, 12))
    torch.testing.assert_close(torque[..., 12:], torch.full((1, 4), 7.5))
    goal = torch.tensor([[.4, .1, .2]])
    matched = list(s)
    matched[2] = torch.tensor([[.4, .1, 0.]])
    matched[3] = torch.tensor([[0., 0., .2]])
    tracked = mdp.terms(s, tuple(matched), goal, a, a)
    stopped = mdp.terms(s, s, goal, a, a)
    for name in ("track_vx", "track_vy", "track_yaw"):
        assert tracked[name] > stopped[name]
    zero = mdp.terms(s, s, torch.zeros_like(goal), a, a)
    assert zero["stand_still"] > .49 and zero["track_vx"] > 2.49
    assert stopped["stand_still"] < 1e-6
    # Changing wheel velocity cannot introduce any wheel-position regularizer.
    fast = list(s)
    fast[5] = torch.cat((s[5][..., :12], torch.full((1, 4), 100.)), -1)
    fast_terms = mdp.terms(s, tuple(fast), goal, a, a)
    for name in ("leg_joint_limit", "leg_joint_deviation"):
        torch.testing.assert_close(fast_terms[name], stopped[name])
    measured = mdp.terms(s, s, goal, a, a, measured_torque=torque)
    torch.testing.assert_close(measured["wheel_torque"], stopped["wheel_torque"])
    torch.testing.assert_close(mdp.reward_transition(s, s, goal, a, a),
                               mdp.control.dt * torch.stack(tuple(stopped.values())).sum(0))


def test_heightfield_depends_on_predicted_pose_every_step():
    grid = torch.linspace(-2., 2., 5)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    terrain = TerrainHeightField(.2 * x + .1 * y, (0., 0., 0.), (2., 2.))
    s = list(state())
    s[0] = s[0].requires_grad_()
    first = terrain.local_scan(s[0], s[1])
    moved = terrain.local_scan(s[0] + torch.tensor([[.2, 0., 0.]]), s[1])
    torch.testing.assert_close(moved - first, torch.full_like(first, -.04))
    rotated = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    assert not torch.allclose(first, terrain.local_scan(s[0], rotated))
    first.sum().backward()
    assert s[0].grad[..., :2].abs().sum() > 0

    class RecordingWM:
        def __init__(self):
            self.scans = []
        def step(self, current, action, context, scan):
            self.scans.append(scan)
            return (current[0] + current[0].new_tensor([[.02, 0., 0.]]), *current[1:])

    # Exercise the actual learner _wm_step path without __init__, backend or GPU.
    learner = object.__new__(SuperMimicLearner)
    learner.cfg = SimpleNamespace(terrain_conditioning=True, terrain_sensor_grad=True)
    learner.mdp = SimpleNamespace(terrain=terrain)
    learner.wm = RecordingWM()
    for _ in range(32):
        s = learner._wm_step(s, torch.zeros(1, 16), torch.zeros(1, 0))
    assert len(learner.wm.scans) == 32
    assert not torch.allclose(learner.wm.scans[0], learner.wm.scans[-1])
    learner.wm.scans.clear()
    learner.cfg.wm_window = 8
    learner.state_layout = LAYOUT
    learner.mdp.normalize_context = lambda context: context
    serialized = flatten_state(state(), LAYOUT)[:, None].expand(1, 8, 41)
    rollout = learner._wm_rollout(serialized, torch.zeros(1, 8, 16), torch.zeros(1, 8, 0))
    assert rollout[4].shape == (1, 8, 12) and rollout[5].shape == (1, 8, 16)
    assert len(learner.wm.scans) == 8
    assert not torch.allclose(learner.wm.scans[0], learner.wm.scans[-1])


def test_official_model_mapping_without_mujoco():
    validate_model(MODEL)
    validate_model(MODEL.with_name("terrain_robot.xml"))
    assert JOINTS[-4:] == WHEEL_JOINTS
