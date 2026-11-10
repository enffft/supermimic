"""Collision, spatial-gradient, blind-observation and shared-training contracts."""

import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, so3_exp
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
from unisupermimic.tasks.velocity.go2.curriculum import TerrainCurriculum
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp


def state(n=2):
    m = Go2VelocityMdp()
    return (
        torch.tensor([[0.0, 0.0, 0.3]]).expand(n, 3).clone(),
        torch.eye(3).expand(n, 3, 3).clone(),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        m.default_q.expand(n, 12).clone(),
        torch.zeros(n, 12),
    )


def test_collision_triangles_rotation_and_autograd():
    import mujoco

    m = mujoco.MjModel.from_xml_string(
        """<mujoco><asset><hfield name="h" nrow="3" ncol="4" size="2 1 .8 .1"/></asset><worldbody><geom name="go2_terrain" type="hfield" hfield="h" pos="3 -2 -.2" euler="0 0 90"/></worldbody></mujoco>"""
    )
    m.hfield_data[:] = np.random.default_rng(4).uniform(0, 1, 12)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    field = TerrainHeightField.from_mujoco(m)
    xy = torch.tensor(
        [[3.15, -1.73], [2.43, -2.58], [3.6, -3.4]], dtype=torch.float64, requires_grad=True
    )
    h = field.ground_height(xy)
    rays = []
    for x, y in xy.detach().numpy():
        distance = mujoco.mj_rayHfield(m, d, 0, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]))
        rays.append(5 - distance)
    assert np.max(np.abs(h.detach().numpy() - rays)) < 2e-7
    grad = torch.autograd.grad(h.sum(), xy)[0]
    eps = 1e-5
    fd = torch.stack(
        [
            (
                field.ground_height(xy.detach() + torch.eye(2, dtype=xy.dtype)[i] * eps)
                - field.ground_height(xy.detach() - torch.eye(2, dtype=xy.dtype)[i] * eps)
            )
            / (2 * eps)
            for i in range(2)
        ],
        -1,
    )
    torch.testing.assert_close(grad, fd, atol=1e-8, rtol=1e-6)
    assert not field.contains(torch.tensor([[100.0, 100.0]])).any()
    assert torch.isfinite(field.ground_height(torch.tensor([[100.0, 100.0]]))).all()


def test_scan_slope_yaw_tiles_and_sensor_grad():
    x = torch.linspace(-2, 2, 41)
    field = TerrainHeightField(0.1 * x.expand(41, 41), (0, 0, 0), (2, 2))
    s = state()
    p = s[0].requires_grad_()
    scan = field.local_scan(p, s[1])
    assert scan.shape == (2, 77)
    assert float(scan[0, 0].detach()) == pytest.approx(0.3 + 0.042)
    rotation = so3_exp(torch.tensor([[0.0, 0.0, torch.pi / 2]])).expand(2, 3, 3)
    scan90 = field.local_scan(p, rotation)
    assert float(scan90[0, 0].detach()) == pytest.approx(0.3 - 0.036)
    assert not field.local_scan(p, rotation, sensor_grad=False).requires_grad
    grad = torch.autograd.grad(scan.sum(), p)[0]
    torch.testing.assert_close(grad[:, 0], torch.full((2,), -7.7))
    # Same global atlas, distinct world positions, no hidden env-index selection.
    p2 = p.detach().clone()
    p2[:, 0] = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(
        field.local_scan(p2, s[1])[0] - field.local_scan(p2, s[1])[1], torch.full((77,), 0.2)
    )
    assert torch.equal(TerrainHeightField.flat().ground_height(p2[:, :2]), torch.zeros(2))


def test_stairs_edge_and_local_reward_translation():
    field = TerrainHeightField(
        torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.1, 0.1]]), (0, 0, 0.5), (1.5, 1)
    )
    xy = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(field.ground_height(xy), torch.tensor([0.5, 0.55, 0.6]))
    original = Go2VelocityMdp()
    elevated = Go2VelocityMdp(terrain=TerrainHeightField.flat(height=2.0))
    s = state()
    high = tuple(t.clone() for t in s)
    high[0][:, 2] += 2
    a = torch.zeros(2, 12)
    g = torch.tensor([[0.5, 0, 0.0]]).expand(2, 3)
    terms0 = original.terms(s, s, g, a, a)
    terms1 = elevated.terms(high, high, g, a, a)
    for key in terms0:
        torch.testing.assert_close(terms0[key], terms1[key], atol=2e-6, rtol=2e-5)
    assert torch.equal(original.hard_termination(s), elevated.hard_termination(high))
    # Foot clearance samples each foot independently across the edge.
    feet = torch.tensor([[[-1.0, 0.0, 0.55], [1.0, 0.0, 0.65]]])
    torch.testing.assert_close(
        field.foot_clearance(feet, radius=0.02), torch.full((1, 2), 0.03), atol=1e-7, rtol=1e-5
    )


def test_wm_expansion_nontrivial_norms_and_blind_actor():
    torch.manual_seed(7)
    old = WorldModel(hidden=(32, 32), num_joints=12, context_dim=16)
    with torch.no_grad():
        old.net[-1].weight.normal_(std=0.01)
        old.in_norm.mean.normal_()
        old.in_norm.var.uniform_(0.2, 2)
        old.out_norm.mean.normal_(std=0.01)
        old.out_norm.var.uniform_(0.001, 0.01)
    new = WorldModel(hidden=(32, 32), num_joints=12, context_dim=16, terrain_dim=77)
    new.initialize_from_flat(old)
    s = state()
    a = torch.randn(2, 12)
    xi = torch.randn(2, 16)
    assert old.in_dim == 65 and new.in_dim == 142
    for t0, t1 in zip(old.step(s, a, xi), new.step(s, a, xi, torch.randn(2, 77))):
        torch.testing.assert_close(t0, t1, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError):
        new.step(s, a, xi)
    mdp = Go2VelocityMdp(terrain=TerrainHeightField.flat())
    obs = mdp.obs(s, a, xi, torch.zeros(2, dtype=torch.long))
    actor = Policy(42, 3, hist=5, num_joints=12, hidden=(32, 32), goal_mode="direct")
    assert actor.dec[0].in_features == 213
    h = obs[:, None].expand(2, 5, 42)
    goal = torch.zeros(2, 3)
    output = actor(h, goal)
    # Changes to privileged positions and world velocity never enter proprioception.
    changed = tuple(t.clone() for t in s)
    changed[0].add_(10)
    changed[2].add_(5)
    assert torch.equal(obs, mdp.obs(changed, a, xi, torch.zeros(2, dtype=torch.long)))
    assert torch.equal(output, actor(h, goal)) and output.abs().max() <= 1


def test_curriculum_progress_failure_recycling_and_fixed_eval():
    class RNG:
        def random(self, shape):
            return torch.ones(shape) * 0.9

    origins = np.zeros((5, 2, 3))
    origins[:, 0, 0] = np.arange(5) * 8
    c = TerrainCurriculum(3, origins, enabled=True)
    ids = torch.arange(3)
    c.reset(ids, RNG())
    c.steps[:] = 500
    c.episode_progress[:] = torch.tensor([4.0, 0.0, 1.0])
    c.commanded_progress[:] = 4.0
    c.episode_fall[2] = True
    c.reset(ids, RNG())
    assert c.mastery.tolist() == [1, 0, 0]  # stationary cannot promote
    c.steps[:] = 300
    c.episode_fall[:] = True
    c.reset(ids, RNG())
    assert c.mastery.tolist() == [0, 0, 0]
    c = TerrainCurriculum(3, origins, enabled=False, fixed_level=2)
    c.steps[:] = 500
    c.episode_progress[:] = 4.0
    c.reset(ids, RNG())
    assert c.terrain_level.tolist() == [2, 2, 2]


def test_h8_h32_query_predicted_pose_and_keep_blind_history():
    from types import SimpleNamespace

    from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner

    class Recorder:
        def __init__(self):
            self.positions = []

        def local_scan(self, p, r, **kw):
            self.positions.append(p.detach().clone())
            return p[..., 2, None].expand(*p.shape[:-1], 77) - 0.1 * p[..., 0, None]

    rec = Recorder()
    mdp = Go2VelocityMdp(terrain=rec)
    wm = WorldModel(hidden=(16,), num_joints=12, context_dim=16, terrain_dim=77)
    with torch.no_grad():
        wm.out_norm.mean[0] = 0.05
        wm.net[-1].weight.normal_(std=0.0001)
    l = object.__new__(SuperMimicLearner)
    l.cfg = SimpleNamespace(
        terrain_conditioning=True, terrain_sensor_grad=True, wm_window=8, goal_mode="direct"
    )
    l.mdp = mdp
    l.wm = wm
    l.pi = Policy(42, 3, hist=5, num_joints=12, hidden=(16,), goal_mode="direct")
    s = state()
    frames = torch.zeros(2, dtype=torch.long)
    a = torch.zeros(2, 12)
    xi = torch.zeros(2, 16)
    h = mdp.obs(s, a, xi, frames)[:, None].expand(2, 5, 42)
    g = torch.zeros(2, 3)
    for _ in range(32):
        s, a, h, frames, g = l._imag_step(s, h, a, frames, xi, xi, g)
    assert len(rec.positions) == 32 and h.shape == (2, 5, 42)
    assert (rec.positions[-1][:, 0] - rec.positions[0][:, 0]).min() > 1
    loss = s[0].sum() + s[4].sum()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in l.pi.parameters())
    rec.positions.clear()
    s0 = flatten_state(state())[:, None].expand(2, 8, 37)
    predicted = l._wm_rollout(s0, torch.zeros(2, 8, 12), torch.zeros(2, 8, 16))
    assert len(rec.positions) == 8
    predicted[0].sum().backward()


@pytest.mark.parametrize("heading", [0, 1, 2, 3])
def test_downhill_entry_direction_and_levels(heading):
    from unisupermimic.tasks.velocity.go2.terrain import terrain_profile

    for kind in ("slope_down", "stairs_down"):
        generator = terrain_profile(kind, levels=True, entry_quarter_turns=heading).generator
        sub = next(iter(generator.sub_terrains.values()))
        for level in range(5):
            result = sub.function((level + 0.5) / 5, np.random.default_rng(4))
            h = result.heightfield.physical_heights_xy()
            ix, iy = np.rint(result.origin[:2] / 0.05).astype(int)
            assert h[ix, iy] >= h[h.shape[0] // 2, h.shape[1] // 2]
            assert result.origin[2] == pytest.approx(h[ix, iy])
            if level == 0:
                assert np.ptp(h) == 0


def test_root_crossing_low_step_uses_local_height_without_spurious_fall():
    field = TerrainHeightField(
        torch.tensor([[0.0, 0.0, 0.08, 0.08], [0.0, 0.0, 0.08, 0.08]]), (0, 0, 0), (1.5, 1.0)
    )
    m = Go2VelocityMdp(terrain=field)
    s = state(3)
    s[0][:, 0] = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([0.3, 0.26, 0.22])
    torch.testing.assert_close(m.local_height(s), expected)
    assert not m.hard_termination(s).any()
    # At a fixed relative fall height, both sides of the step terminate equally.
    s[0][:, 2] = field.ground_height(s[0][:, :2]) + 0.16
    assert m.hard_termination(s).all()


def test_driver_contract_fails_closed_and_does_not_change_final_actor():
    from dataclasses import replace
    from types import SimpleNamespace

    from unisupermimic.rl.algos.supermimic.driver import DriverActionContract
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
    from unisupermimic.tasks.velocity.go2.contract import DEFAULT_Q, JOINTS

    l = object.__new__(SuperMimicLearner)
    l.cfg = SimpleNamespace(driver_enabled=True, goal_mode="direct")
    driver = SimpleNamespace(
        contract=DriverActionContract(tuple(JOINTS), tuple(DEFAULT_Q), 0.25, 0.02, 40.0, 1.0, 1.0)
    )
    l.set_collection_driver(driver)
    assert l.collection_driver is driver
    driver.contract = replace(driver.contract, control_dt=0.01)
    with pytest.raises(ValueError, match="contract mismatch"):
        l.set_collection_driver(driver)


def test_local_scan_yaw_gradient_matches_finite_difference():
    dtype = torch.float64
    field = TerrainHeightField(
        torch.linspace(-0.2, 0.2, 41, dtype=dtype).expand(41, 41), (0, 0, 0), (2, 2)
    )
    p = torch.tensor([[0.1, 0.0, 0.4]], dtype=dtype)
    angle = torch.tensor(0.4, dtype=dtype, requires_grad=True)

    def scan(theta):
        rotation = so3_exp(torch.stack((theta * 0, theta * 0, theta))[None])
        return field.local_scan(p, rotation).sum()

    grad = torch.autograd.grad(scan(angle), angle)[0]
    eps = 1e-5
    fd = (scan(angle.detach() + eps) - scan(angle.detach() - eps)) / (2 * eps)
    torch.testing.assert_close(grad, fd, atol=1e-8, rtol=1e-7)
    assert grad.abs() > 0.01


def test_curriculum_zero_command_holds_and_easy_recycle_cannot_promote_ceiling():
    class RNG:
        def random(self, shape):
            return torch.ones(shape) * 0.01

    c = TerrainCurriculum(
        1, np.zeros((5, 1, 3)), enabled=True, fixed_level=3, recycle_probability=1.0
    )
    ids = torch.tensor([0])
    c.steps[:] = 500
    c.reset(ids, RNG())
    assert c.mastery.item() == 3 and c.terrain_level.item() == 0
    c.steps[:] = 500
    c.episode_progress[:] = 4.0
    c.commanded_progress[:] = 4.0
    c.reset(ids, RNG())
    assert c.mastery.item() == 3  # successful flat revisit is not level-3 mastery


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_rough_curriculum_is_random_not_a_constant_platform(level):
    from unisupermimic.tasks.velocity.go2.terrain import terrain_profile

    sub = terrain_profile("rough", levels=True).generator.sub_terrains["rough"]
    generated = sub.function((level + 0.5) / 5, np.random.default_rng(11))
    h = generated.heightfield.physical_heights_xy()
    interior = h[25:-25, 25:-25]
    assert np.ptp(interior) == pytest.approx(0.01 * level)
    assert np.unique(interior).size >= 3
    assert h.min() >= 0 and h.max() <= 0.01 * level + 1e-8
