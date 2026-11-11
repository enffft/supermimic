"""Probability, migration and command semantics tests for the opt-in sampler."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unilab.managers.rng import ManagerRng

from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.command_sampler import (
    TerrainAwareVelocitySampler,
    VelocitySamplerCfg,
    command_modes,
    fixed_domains,
    validate_distribution_migration,
)
from unisupermimic.tasks.velocity.go2.curriculum import TerrainCurriculum
from unisupermimic.tasks.velocity.go2.manager_terms import VelocityCommand, VelocityCommandCfg


def sample(n=100000, domain=0, level=2):
    s = TerrainAwareVelocitySampler(VelocitySamplerCfg(flat_speed_level=level))
    return s, s.sample(torch.full((n,), domain), torch.full((n,), 4))


def test_flat_sign_pure_axis_contract_and_proportions():
    s, (g, m, c, hold) = sample()
    assert (g.min(0).values < 0).all() and (g.max(0).values > 0).all()
    assert torch.equal(command_modes(g), m)
    assert (g[m == 3][:, [0, 2]] == 0).all() and (g[m == 4][:, :2] == 0).all()
    assert torch.allclose(
        torch.bincount(m).float() / len(m), torch.tensor(s.cfg.flat_mode_weights), atol=0.006
    )
    assert not c.any() and hold.min() >= 2 and hold.max() <= 5
    for mode, axis in ((3, 1), (4, 2)):
        positive = (g[m == mode, axis] > 0).float().mean()
        assert 0.48 < positive < 0.52
    assert (g[:, 1].abs() < 0.1).any() and (g[:, 1].abs() > 0.3).any()


@pytest.mark.parametrize("domain", [1, 2, 3, 4, 5])
def test_terrain_ranges_and_conservative_modes(domain):
    s, (g, m, c, hold) = sample(20000, domain)
    assert (g[:, 0] <= s.highs[domain, 0] + 1e-7).all()
    assert (g[:, 1:].abs() <= s.highs[domain, 1:] + 1e-7).all()
    if domain >= 2:
        assert (g[:, 0] >= 0).all()
    assert torch.equal(c, m == 1) and (hold > 10).all()
    assert (m == 3).any() and (m == 4).any()


def test_rng_state_and_speed_curriculum_independence():
    s, (g, *_) = sample(1000)
    saved = s.state_dict()
    d = torch.zeros(500, dtype=torch.long)
    expected = s.sample(d, d)
    other = TerrainAwareVelocitySampler(s.cfg)
    other.load_state_dict(saved)
    assert all(torch.equal(a, b) for a, b in zip(expected, other.sample(d, d), strict=True))
    other.set_speed_level(0)
    _, g0 = sample(1000, level=0)
    assert other.speed_level == 0
    c = TerrainCurriculum(5, np.zeros((5, 5, 3)))
    c.mastery.fill_(4)
    assert other.speed_level == 0


def test_fixed_domain_quotas_preserve_atlas_type():
    d = fixed_domains(2048, VelocitySamplerCfg().domain_weights)
    assert torch.bincount(d).tolist() == [717, 409, 205, 205, 256, 256]
    ids = torch.arange(2048)
    assert torch.equal(ids[d != 0] % 5, d[d != 0] - 1)
    assert (d == 0).sum() / 2048 == pytest.approx(0.35, abs=0.001)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_legacy_sampling_unchanged(stage):
    # Golden formula reproduces both RNG draws exactly, including Stage0 zeros.
    cfg = VelocityCommandCfg(stage=stage)
    term = SimpleNamespace(
        cfg=cfg, g=torch.zeros(5000, 3), num_envs=5000, sampler=None, _rng=ManagerRng.seeded(40)
    )
    VelocityCommand._resample_command(term, torch.ones(5000, dtype=torch.bool))
    rng = ManagerRng.seeded(40)
    lo, hi = (
        ((0.25, 0, 0), (0.6, 0, 0)),
        ((-0.2, -0.15, -0.4), (0.7, 0.15, 0.4)),
        ((-0.4, -0.3, -0.8), (1.0, 0.3, 0.8)),
    )[stage]
    expected = rng.uniform(torch.tensor(lo), torch.tensor(hi), (5000, 3))
    expected = torch.where((rng.random((5000,)) < 0.05)[:, None], 0.0, expected)
    assert torch.equal(term.g, expected)


def test_adaptive_mask_period_and_crossing_labels():
    n = 100
    s, _ = sample(1)
    term = SimpleNamespace(
        cfg=VelocityCommandCfg(sampling_mode="terrain_adaptive"),
        sampler=s,
        command_domain=torch.arange(n) % 6,
        command_mode=torch.zeros(n, dtype=torch.long),
        crossing_episode=torch.zeros(n, dtype=torch.bool),
        g=torch.zeros(n, 3),
        num_envs=n,
        time_left=torch.zeros(n),
        curriculum=None,
    )
    mask = torch.arange(n) % 2 == 0
    VelocityCommand._resample_command(term, mask)
    assert (term.g[~mask] == 0).all() and (term.time_left[~mask] == 0).all()
    assert (term.time_left[mask & (term.command_domain != 0)] > 10).all()
    assert not term.crossing_episode[term.command_domain == 0].any()


def test_general_episode_falls_do_not_demote_or_reset_crossing_streak():
    c = TerrainCurriculum(5, np.zeros((5, 5, 3)), enabled=True, fixed_level=2)
    c.fixed_domain = torch.arange(5) + 1
    c.crossing_episode.zero_()
    c.success_streak.fill_(2)
    for _ in range(3):
        c.steps.fill_(500)
        c.episode_fall.fill_(True)
        c.reset(torch.arange(5), ManagerRng.seeded(9))
    assert (c.mastery == 2).all() and (c.success_streak == 2).all()


def replay():
    b = ReplayBuffer(
        12,
        24,
        "cpu",
        num_joints=12,
        context_dim=16,
        goal_dim=3,
        terrain_metadata=True,
        command_metadata=True,
    )
    for t in range(12):
        d = {k: torch.zeros((24,) + v.shape[2:], dtype=v.dtype) for k, v in b.data.items()}
        d["g"] = torch.tensor(
            [[0, 0, 0], [0.4, 0, 0], [-0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.5], [0.3, 0.1, 0.2]]
        ).repeat(4, 1)
        d["command_mode"] = command_modes(d["g"])
        d["ep_step"].fill_(t)
        d["s"].fill_(t)
        b.add(d)
    return b


def test_replay_stratified_distribution_and_goal_history_alignment():
    b = replay()
    w = [0.05, 0.05, 0.2, 0.3, 0.3, 0.1]
    start, hist = b.sample_starts(20000, 5, w)
    assert torch.equal(command_modes(start["g"]), start["command_mode"])
    assert torch.equal(start["s"], hist["s"][:, -1])
    assert torch.allclose(
        torch.bincount(start["command_mode"]).float() / 20000, torch.tensor(w), atol=0.01
    )
    original = b.state_dict()
    other = replay()
    other.load_state_dict(original)
    assert other.ptr == b.ptr and other.size == b.size


def test_migration_keeps_physics_reward_and_network_contract_strict():
    old = {"commands": {"stage": 0}, "terminations": {}, "action": "same", "reward": {"a": 1}}
    new = {
        "commands": {"stage": 0, "sampling_mode": "terrain_adaptive"},
        "terminations": {},
        "action": "same",
        "reward": {"a": 1},
    }
    validate_distribution_migration(old, new)
    new["reward"]["a"] = 2
    with pytest.raises(ValueError, match="protected"):
        validate_distribution_migration(old, new)


def test_adaptive_physics_goal_snapshot_resume_and_h32_blind_contract(tmp_path):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
    from unisupermimic.tasks.velocity.go2.config import load_config
    from unisupermimic.tasks.velocity.go2.env import make_go2_env

    cfg = load_config(terrain=True)
    c = cfg.commands["velocity"]
    c.terrain_profile = "mixed"
    c.sampling_mode = "terrain_adaptive"
    c.terrain_curriculum = True
    c.terrain_phase = "skill"
    cfg.max_episode_seconds = 0.12
    c.sampler.flat_hold_range = (0.04, 0.06)
    env = make_go2_env(cfg, 32)
    env.init_state()
    lc = SuperMimicConfig(
        goal_mode="direct",
        goal_frames=1,
        hist=5,
        wm_hidden=(32,),
        policy_hidden=(32,),
        critic_hidden=(16,),
        buffer_steps=16,
        compile=False,
        amp="none",
        no_sync=False,
        wm_batch=4,
        pi_batch=4,
        pi_opt="adam",
        wm_opt="adam",
        wm_window=8,
        horizon=32,
        terrain_training=True,
        terrain_conditioning=True,
        checkpoint_replay=True,
        command_start_weights=(0.08, 0.25, 0.20, 0.20, 0.17, 0.10),
    )
    try:
        l = SuperMimicLearner(env, lc, tmp_path / "first")
        for _ in range(12):
            before = env.current_info()
            l.collect(1)
            row = (l.buffer.ptr - 1) % l.buffer.T
            assert torch.equal(l.buffer.data["g"][row], before["g"])
            assert torch.equal(l.buffer.data["ep_step"][row], before["ep_step"])
        cmd = env.command_manager.get_term("velocity")
        assert (l.buffer.data["command_domain"][:12] == cmd.command_domain).all()
        assert l.pi.dec[0].in_features == 213 and l.wm.terrain_dim == 77
        start, history = l.buffer.sample_starts(16, 5, lc.command_start_weights)
        s = unflatten_state(start["s"])
        g = start["g"].clone()
        original = g.clone()
        a = start["a_prev"]
        f = start["f"]
        xi = start["xi"]
        h = l.mdp.obs(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"])
        for _ in range(32):
            s, a, h, f, g = l._imag_step(s, h, a, f, xi, l.mdp.normalize_context(xi), g)
            assert torch.equal(g, original) and h.shape == (16, 5, 42)
        l.iteration = 57
        l.update_counts = dict(wm=11, actor=3, critic=17)
        cmd.sampler.set_speed_level(1)
        saved = l.save()
        expected_state = cmd.sampler.state_dict()
        resumed = SuperMimicLearner(env, lc, tmp_path / "resume")
        resumed.load(saved)
        assert resumed.iteration == 57 and resumed.update_counts == l.update_counts
        assert resumed.env.command_manager.get_term("velocity").sampler.speed_level == 1
        # New environment episodes intentionally consume fresh draws on resume;
        # standalone sampler restore is bit-exact (tested separately).
        assert resumed.buffer.data["done"][(resumed.buffer.ptr - 1) % resumed.buffer.T].all()
        assert resumed._checkpoint(True)["command_sampler"]["config"] == expected_state["config"]
        assert torch.equal(resumed.pi.dec[0].weight, l.pi.dec[0].weight)
    finally:
        env.close()
        env.backend.close()


def test_early_crossing_falls_demote_even_before_one_metre_commanded_progress():
    c = TerrainCurriculum(
        1, np.zeros((5, 5, 3)), enabled=True, fixed_level=2, recycle_probability=0.0
    )
    c.fixed_domain = torch.ones(1, dtype=torch.long)
    c.crossing_episode.fill_(True)
    for _ in range(2):
        c.steps.fill_(30)
        c.eligible_steps.fill_(30)
        c.commanded_progress.fill_(0.2)
        c.episode_fall.fill_(True)
        c.reset(torch.tensor([0]), ManagerRng.seeded(99))
    assert c.mastery.item() == 1


def test_all_direction_physics_guard_detects_new_failures_and_mixed_yaw_drift():
    from copy import deepcopy

    from tools.train_go2_omnidirectional import physical_regressions

    metric = dict(xy_rmse_m_s=0.03, yaw_rmse_rad_s=0.03, fall=0.0, crossing_passed=1.0)
    baseline = {
        k: deepcopy(metric) for k in ("lateral_left", "mixed", "stairs_up_L4_cmd0", "boundary")
    }
    baseline["boundary"]["fall"] = 1.0
    new = deepcopy(baseline)
    assert not physical_regressions(new, baseline)
    new["lateral_left"]["xy_rmse_m_s"] = 0.2
    new["mixed"]["yaw_rmse_rad_s"] = 0.4
    new["stairs_up_L4_cmd0"]["crossing_passed"] = 0.2
    new["stairs_up_L4_cmd0"]["fall"] = 0.4
    regressions = physical_regressions(new, baseline)
    assert {r["metric"] for r in regressions} == {
        "xy_rmse_m_s",
        "yaw_rmse_rad_s",
        "fall",
        "crossing_passed",
    }
