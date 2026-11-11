"""Opt-in stairs commands; protected task/network/reward contracts stay fixed."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from unisupermimic.tasks.velocity.go2.command_sampler import (
    TerrainAwareVelocitySampler,
    VelocitySamplerCfg,
    validate_distribution_migration,
)
from unisupermimic.tasks.velocity.go2.gait import Go2GaitCfg
from unisupermimic.tasks.velocity.go2.manager_terms import VelocityCommand, VelocityCommandCfg


def sampler(level=0):
    s = TerrainAwareVelocitySampler(VelocitySamplerCfg(stairs_omni=True))
    s.stairs_levels.fill_(level)
    return s


@pytest.mark.parametrize(
    "level,vy,wz", [(0, 0.08, 0.15), (1, 0.15, 0.3), (2, 0.22, 0.45), (3, 0.3, 0.6)]
)
def test_stairs_speed_ranges_modes_and_signs(level, vy, wz):
    s = sampler(level)
    n = 50000
    domain = torch.arange(n) % 2 + 4
    g, mode, cross, hold = s.sample(domain, torch.arange(n) % 5)
    assert g[:, 0].min() >= -0.3 and g[:, 0].max() <= 0.8
    assert g[:, 1].abs().max() <= vy + 1e-7 and g[:, 2].abs().max() <= wz + 1e-7
    assert g[:, 1].max() > vy * 0.98 and g[:, 1].min() < -vy * 0.98
    assert g[:, 2].max() > wz * 0.98 and g[:, 2].min() < -wz * 0.98
    for axis, pure in ((1, 3), (2, 4)):
        mask = mode == pure
        assert mask.float().mean().item() == pytest.approx(0.10, abs=0.008)
        assert (g[mask, axis] > 0).float().mean().item() == pytest.approx(0.5, abs=0.025)
        assert (g[mask][:, [j for j in range(3) if j != axis]] == 0).all()
    assert cross.float().mean().item() == pytest.approx(0.55, abs=0.01)
    assert (g[(g != 0).all(-1)].shape[0]) > n * 0.4
    assert (hold[cross] > 10).all()
    assert ((hold[~cross] >= 2) & (hold[~cross] <= 4)).all()
    assert (g[cross, 1].abs() <= 0.05).all() and (g[cross, 2].abs() <= 0.1).all()


def test_speed_levels_are_per_direction_and_terrain_but_not_mastery():
    s = sampler()
    s.set_stairs_speed_level(0, 2, 3)
    s.set_speed_level(1)
    assert s.stairs_levels.tolist() == [[0, 0, 3, 0, 0], [0, 0, 0, 0, 0]]
    d = torch.tensor([4, 4, 5]).repeat(5000)
    level = torch.tensor([2, 4, 2]).repeat(5000)
    g, *_ = s.sample(d, level)
    assert g[::3, 1].abs().max() > 0.29
    assert g[1::3, 1].abs().max() <= 0.08
    assert g[2::3, 1].abs().max() <= 0.08
    assert s.speed_level == 1


def test_command_migration_preserves_rng_and_explicit_speed_state():
    old = TerrainAwareVelocitySampler(VelocitySamplerCfg())
    old.set_speed_level(1)
    state = old.state_dict()
    new = sampler()
    with pytest.raises(ValueError, match="migration"):
        new.load_state_dict(state)
    new.load_state_dict(state, allow_distribution_change=True)
    assert new.speed_level == 1 and (new.stairs_levels == 0).all()
    new.set_stairs_speed_level(1, 4, 2)
    saved = new.state_dict()
    expected = new.sample(torch.tensor([0, 4, 5] * 100), torch.tensor([0, 2, 4] * 100))
    other = sampler()
    other.load_state_dict(saved)
    actual = other.sample(torch.tensor([0, 4, 5] * 100), torch.tensor([0, 2, 4] * 100))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_resampling_never_converts_maneuver_history_into_traversal():
    n = 500
    s = sampler()
    term = SimpleNamespace(
        cfg=VelocityCommandCfg(),
        sampler=s,
        command_domain=torch.full((n,), 4),
        curriculum=None,
        command_mode=torch.zeros(n, dtype=torch.long),
        crossing_episode=torch.zeros(n, dtype=torch.bool),
        time_left=torch.zeros(n),
        ep_step=torch.full((n,), 100),
        g=torch.zeros(n, 3),
    )
    VelocityCommand._resample_command(term, torch.ones(n, dtype=torch.bool))
    assert not term.crossing_episode.any()
    assert ((term.time_left >= 2) & (term.time_left <= 4)).all()
    term.ep_step.zero_()
    VelocityCommand._resample_command(term, torch.ones(n, dtype=torch.bool))
    assert term.crossing_episode.any()


def test_reward_and_all_protected_migration_fields_remain_strict():
    assert asdict(Go2GaitCfg()) == dict(
        feet_clearance=0.0,
        diagonal_gait=-0.15,
        feet_slip=-0.2,
        clearance_target=0.05,
        clearance_sigma=0.025,
    )
    old = dict(
        commands=dict(sampler={}),
        reward=dict(gait=asdict(Go2GaitCfg())),
        terminations={},
        action="same",
        terrain="same",
        actor="213",
        state="same",
        ctrl_dt=0.02,
    )
    new = deepcopy(old)
    new["commands"]["sampler"] = asdict(VelocitySamplerCfg(stairs_omni=True))
    validate_distribution_migration(old, new)
    for key in ("reward", "action", "terrain", "actor", "state", "ctrl_dt", "terminations"):
        bad = deepcopy(new)
        bad[key] = {"changed": True} if key == "terminations" else "changed"
        with pytest.raises(ValueError):
            validate_distribution_migration(old, bad)


def test_speed_admission_needs_real_evidence_for_each_direction_level():
    from unisupermimic.tasks.velocity.go2.speed_curriculum import stairs_speed_admission

    prefix = "stairs_up_L2_S1_"
    evaluation = {
        prefix + str(i): dict(
            fall=0, survival_s=4, xy_rmse_m_s=0.04, yaw_rmse_rad_s=0.04, outside_tile_fraction=0
        )
        for i in range(7)
    }
    diagnostics = dict(
        gradients=[
            dict(
                scenario=prefix + str(i),
                quantity="joint_velocity",
                epsilon=0.03,
                sign_agreement=0.9,
                normalized_error=0.3,
            )
            for i in range(7)
        ],
        prediction=[
            dict(
                scenario=prefix + str(i),
                horizon=32,
                position_m=0.01,
                velocity_m_s=0.02,
                joint_position_rad=0.03,
                attitude_rad=0.02,
            )
            for i in range(7)
        ],
    )
    assert stairs_speed_admission(evaluation, diagnostics, 0, 2, 1)["admitted"]
    assert not stairs_speed_admission(evaluation, diagnostics, 1, 2, 1)["admitted"]
    diagnostics["gradients"][0]["sign_agreement"] = 0.79
    assert not stairs_speed_admission(evaluation, diagnostics, 0, 2, 1)["admitted"]
    diagnostics["gradients"][0]["sign_agreement"] = 0.9
    evaluation[prefix + "0"]["fall"] = 0.2
    assert not stairs_speed_admission(evaluation, diagnostics, 0, 2, 1)["admitted"]


def test_new_command_only_real_resume_and_reward_contract(tmp_path):
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
    from unisupermimic.tasks.velocity.go2.config import load_config
    from unisupermimic.tasks.velocity.go2.env import make_go2_env

    cfg = load_config(terrain=True)
    c = cfg.commands["velocity"]
    c.terrain_profile = "mixed"
    c.terrain_phase = "skill"
    c.terrain_curriculum = True
    c.sampling_mode = "terrain_adaptive"
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
    )
    try:
        l = SuperMimicLearner(env, lc, tmp_path / "old")
        l.collect(12)
        saved = l.save()
        original_reward = deepcopy(env.supermimic_contract["reward"])
        oldreplay = l.buffer.data["g"].clone()
    finally:
        env.close()
        env.backend.close()
    c.sampler.stairs_omni = True
    c.sampler.stairs_maneuver_entry_offset = 1.6
    env = make_go2_env(cfg, 32)
    env.init_state()
    try:
        resumed = SuperMimicLearner(env, lc, tmp_path / "new")
        resumed.load(saved, allow_command_distribution_change=True)
        assert env.supermimic_contract["reward"] == original_reward
        assert torch.equal(resumed.buffer.data["g"], oldreplay)
        cmd = env.command_manager.get_term("velocity")
        for _ in range(15):
            info = env.current_info()
            resumed.collect(1)
            assert torch.equal(resumed.buffer.data["g"][(resumed.buffer.ptr - 1) % 16], info["g"])
        assert resumed.pi.dec[0].in_features == 213 and resumed.wm.terrain_dim == 77
        cmd.sampler.set_stairs_speed_level(1, 4, 2)
        out = resumed.save()
        again = SuperMimicLearner(env, lc, tmp_path / "resume")
        again.load(out)
        assert cmd.sampler.stairs_levels[1, 4] == 2
        assert again.buffer.data["done"][(again.buffer.ptr - 1) % 16].all()
    finally:
        env.close()
        env.backend.close()


def test_scope_filter_excludes_failed_wm_pairs_without_changing_goal_history():
    from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer

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
            row = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
            row["command_mode"] = torch.arange(24) % 6
            row["g"] = torch.tensor(
                [[0, 0, 0], [0.4, 0, 0], [-0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.5], [0.3, 0.1, 0.2]]
            ).repeat(4, 1)
            row["ep_step"].fill_(t)
            row["s"].fill_(t)
            b.add(row)
        return b

    b = replay()
    # The test buffer has six modes; reinterpret two as failed/allowed domains.
    b.data["command_domain"][:] = b.data["command_mode"]
    b.data["terrain_level"].fill_(2)
    b.start_scope_mask = torch.ones(6, 5, dtype=torch.bool)
    b.start_scope_mask[4, 2] = False
    for weights in (None, [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]):
        start, hist = b.sample_starts(1000, 5, weights)
        assert not (start["command_domain"] == 4).any()
        assert torch.equal(start["s"], hist["s"][:, -1])
        assert torch.equal(start["g"], b.data["g"][0][start["command_domain"]])
    values = b.state_dict()
    other = replay()
    other.load_state_dict(values)
    assert torch.equal(other.start_scope_mask, b.start_scope_mask)
