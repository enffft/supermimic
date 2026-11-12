"""CPU-only analytical checks; no environment or CUDA initialization."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unisupermimic.tasks.velocity.go2.command_sampler import fixed_domains
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2w.contract import DEFAULT_Q
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp
from unisupermimic.tasks.velocity.go2w.terrain import Go2WTerrainCurriculum, mixed_terrain
from unisupermimic.tasks.velocity.go2w.terrain_plane import terrain_plane_normal


def test_stage_migration_is_narrow_and_rejects_old_reward_protocol():
    from unisupermimic.tasks.velocity.go2w.checkpoint import validate_stage_migration

    old = {
        "schema": "go2w_velocity_supermimic_v2",
        "reward_protocol": "plane_v2",
        "commands": {
            "sampler": {"domain_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "terrain_level_count": 8}
        },
        "model": "same",
    }
    new = deepcopy(old)
    new["commands"]["sampler"]["domain_weights"] = [0.5, 0.2, 0.15, 0.15, 0.0, 0.0]
    validate_stage_migration(old, new)
    bad = deepcopy(new)
    bad["model"] = "different"
    with pytest.raises(ValueError):
        validate_stage_migration(old, bad)
    bad = deepcopy(old)
    bad["schema"] = "go2w_velocity_supermimic_v1"
    with pytest.raises(ValueError):
        validate_stage_migration(bad, new)
    bad = deepcopy(new)
    bad["commands"]["sampler"]["terrain_level_count"] = 5
    with pytest.raises(ValueError):
        validate_stage_migration(old, bad)


def test_g1_legacy_protocol_remains_encoded():
    from unisupermimic.rl.algos.supermimic.geometry import flatten_state, unflatten_state
    from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

    policy = Policy(
        MDP.obs_dim, MDP.goal_frame_dim, hist=5, num_joints=29, hidden=(8,), goal_mode="encoded"
    )
    assert hasattr(policy, "enc")
    wm = WorldModel(num_joints=29, context_dim=MDP.context_dim, hidden=(8,))
    assert wm.out_dim == 70
    s = (
        torch.zeros(1, 3),
        torch.eye(3)[None],
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.zeros(1, 29),
        torch.arange(29, dtype=torch.float32)[None],
    )
    value = flatten_state(s)
    assert value.shape == (1, 71)
    for a, b in zip(s, unflatten_state(value)):
        torch.testing.assert_close(a, b)


def test_wm_coverage_does_not_certify_actor_mastery_and_phase_switch():
    from unilab.managers.rng import ManagerRng

    from unisupermimic.tasks.velocity.go2w.diagnostics import begin_iteration

    c = Go2WTerrainCurriculum(
        128,
        np.zeros((8, 5, 3), dtype=np.float32),
        enabled=True,
        max_level=7,
        phase="wm",
        coverage_weights=(0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.03, 0.01),
    )
    domains = fixed_domains(128, (0.25, 0.2, 0.1, 0.1, 0.175, 0.175))
    c.fixed_domain = domains
    c.reset(torch.arange(128), ManagerRng.seeded(2026, device="cpu"))
    assert (c.terrain_level > 0).any() and (c.mastery == 0).all()
    assert (c.terrain_level[domains == 0] == 0).all()
    command = SimpleNamespace(
        curriculum=c,
        command_domain=domains,
        cfg=SimpleNamespace(automatic_wm_skill_transition=True),
    )
    env = SimpleNamespace(device="cpu", command_manager=SimpleNamespace(get_term=lambda _: command))
    learner = SimpleNamespace(
        iteration=0,
        cfg=SimpleNamespace(wm_only_iters=2),
        device="cpu",
        buffer=SimpleNamespace(),
        profile_ms={},
    )
    begin_iteration(env, learner)
    assert learner.buffer.start_scope_mask.shape == (6, 8)
    assert not learner.buffer.start_scope_mask[:, 1:].any()
    learner.iteration = 2
    begin_iteration(env, learner)
    assert c.phase == "skill" and (c.mastery == 0).all()
    c.mastery[domains == 4] = 2
    begin_iteration(env, learner)
    assert learner.buffer.start_scope_mask[4, :3].all()
    assert not learner.buffer.start_scope_mask[4, 3:].any()


def state(batch=1):
    return (
        torch.tensor([[0.0, 0.0, 0.42]]).expand(batch, -1).clone(),
        torch.eye(3).expand(batch, 3, 3).clone(),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.tensor(DEFAULT_Q).expand(batch, -1).clone(),
        torch.zeros(batch, 16),
    )


@pytest.mark.parametrize("slope", [0.0, 0.1, 0.2, 5.0])
def test_plane_and_alignment(slope):
    grid = torch.linspace(-2, 2, 17)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    field = TerrainHeightField(slope * x + 0.03 * y, (0.0, 0.0, 0.0), (2.0, 2.0))
    s = list(state())
    scan = field.local_scan(s[0], s[1])
    normal = terrain_plane_normal(scan, s[1])
    if slope < 1:
        expected = torch.tensor([[-slope, -0.03, 1.0]])
        expected /= expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(normal, expected, atol=2e-5, rtol=2e-5)
    assert torch.acos(normal[..., 2]) <= torch.tensor(35.0 * torch.pi / 180 + 1e-5)
    # World normal is invariant to scan yaw.
    yaw = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(
        terrain_plane_normal(field.local_scan(s[0], yaw), yaw), normal, atol=2e-5, rtol=2e-5
    )
    mdp = Go2WVelocityMdp(terrain=field)
    # Construct orthonormal body frame whose Z axis equals fitted normal.
    z = normal[0]
    yy = torch.linalg.cross(z, z.new_tensor([1.0, 0.0, 0.0]))
    yy /= yy.norm()
    xx = torch.linalg.cross(yy, z)
    aligned = list(s)
    aligned[1] = torch.stack((xx, yy, z), -1)[None].requires_grad_()
    a, goal = torch.zeros(1, 16), torch.zeros(1, 3)
    horizontal = mdp.terms(s, s, goal, a, a)["terrain_alignment"]
    reward = mdp.terms(s, aligned, goal, a, a)["terrain_alignment"]
    assert reward >= horizontal
    reward.sum().backward()
    assert torch.isfinite(aligned[1].grad).all()


def test_invalid_flat_and_stair_edge_normals():
    s = state()
    up = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(terrain_plane_normal(torch.ones(1, 77), s[1]), up)
    torch.testing.assert_close(terrain_plane_normal(torch.full((1, 77), float("nan")), s[1]), up)
    grid = torch.linspace(-2, 2, 81)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    field = TerrainHeightField((x / 0.4).floor().clamp(0, 3) * 0.21, (0.0, 0.0, 0.0), (2.0, 2.0))
    for px in (-0.4, -0.01, 0.01, 0.4, 0.8, 1.19, 1.21, 1.6):
        p = torch.tensor([[px, 0.0, 0.5]], requires_grad=True)
        normal = terrain_plane_normal(field.local_scan(p, s[1]), s[1])
        assert torch.isfinite(normal).all()
        assert normal[0, 2] >= np.cos(np.deg2rad(35)) - 1e-6
        normal.sum().backward()
        assert torch.isfinite(p.grad).all()


def test_domain_weights_time_major_and_reward_gradients():
    mdp = Go2WVelocityMdp(terrain=TerrainHeightField.flat())
    start = {"command_domain": torch.arange(6)}
    tags = mdp.imagination_reward_kwargs(start, 3)["command_domain"]
    torch.testing.assert_close(tags, torch.arange(6).repeat(3))
    s = list(state(18))
    s[2][:, 2] = 1
    s[2].requires_grad_()
    s[4] += 0.1
    a, goal = torch.zeros(18, 16, requires_grad=True), torch.zeros(18, 3)
    terms = mdp.terms(s, s, goal, a, a.detach(), command_domain=tags)
    torch.testing.assert_close(
        terms["base_lin_vel_z"], torch.tensor(mdp.reward_cfg.base_lin_vel_z).repeat(3)
    )
    torch.testing.assert_close(
        terms["leg_joint_deviation"],
        torch.tensor(mdp.reward_cfg.leg_joint_deviation).repeat(3) * 0.12,
    )
    reward = mdp.reward_transition(s, s, goal, a, a.detach(), command_domain=tags)
    assert reward.shape == (18,)
    reward.sum().backward()
    assert torch.isfinite(s[2].grad).all() and torch.isfinite(a.grad).all()


@pytest.mark.parametrize("n", [32, 128, 512, 1024, 2048, 4096])
def test_domain_quotas(n):
    weights = torch.tensor([0.25, 0.20, 0.10, 0.10, 0.175, 0.175], dtype=torch.float64)
    d = fixed_domains(n, weights)
    count = torch.bincount(d, minlength=6)
    assert count.sum() == n
    assert ((count.double() - n * weights).abs() < 1.01).all()
    ids = torch.arange(n)
    for k in range(1, 6):
        assert (ids[d == k] % 5 == k - 1).all()


def test_eight_level_generated_geometry_and_curriculum_restore():
    terrain = mixed_terrain()
    assert terrain.generator.num_rows == 8
    for name, cfg in terrain.generator.sub_terrains.items():
        for level in range(8):
            result = cfg.function((level + 0.5) / 8, np.random.default_rng(2026))
            heights = result.heightfield.physical_heights_xy()
            assert np.isfinite(heights).all()
            if level == 0:
                assert np.ptp(heights) == 0
            if "stairs" in name and level > 0:
                centerline = heights[:, heights.shape[1] // 2]
                steps = np.abs(np.diff(centerline))
                steps = steps[steps > 0.001]
                assert len(steps) > 0
                assert np.allclose(steps, 0.03 * level, atol=0.00501)
                entry = centerline[12]
                assert (centerline[len(centerline) // 2] < entry) == name.endswith("down")
    origins = np.zeros((8, 5, 3), dtype=np.float32)
    c = Go2WTerrainCurriculum(
        6,
        origins,
        enabled=True,
        max_level=7,
        phase="wm",
        coverage_weights=(0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.03, 0.01),
    )
    c.fixed_domain = torch.arange(6)
    c.phase = "skill"
    c.mastery.fill_(7)
    saved = deepcopy(c.state_dict())
    c.phase = "wm"
    c.load_state_dict(saved)
    assert c.phase == "skill" and (c.mastery == 7).all()
