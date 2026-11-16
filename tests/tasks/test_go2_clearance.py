"""Clearance-only ablation: gradients, shared math, migration and protected interfaces."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from test_go2_velocity import state

from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.gait import (
    Go2GaitCfg,
    clearance_region_weight,
    clearance_shape,
    go2_gait_reward_terms,
)
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp
from unisupermimic.tasks.velocity.go2.reward_migration import (
    relabel_replay_clearance,
    validate_reward_migration,
)


@pytest.mark.parametrize("mode", ["gaussian", "soft_deficit"])
def test_low_height_gradient_finite_target_and_no_unbounded_lift(mode):
    cfg = Go2GaitCfg(
        feet_clearance=0.05, clearance_target=0.12, clearance_sigma=0.065, clearance_mode=mode
    )
    h = torch.tensor([0.0, 0.05, 0.08, 0.12, 0.16, 0.30], dtype=torch.float64, requires_grad=True)
    r = clearance_shape(h, cfg)
    grad = torch.autograd.grad(r.sum(), h)[0]
    assert torch.isfinite(grad).all() and (grad[:3] > 0).all()
    assert grad[3].abs() < 1e-10 and (grad[4:] <= 0).all()
    eps = 1e-6
    fd = (clearance_shape(h.detach() + eps, cfg) - clearance_shape(h.detach() - eps, cfg)) / (
        2 * eps
    )
    torch.testing.assert_close(grad, fd, atol=3e-5, rtol=1e-4)
    if mode == "soft_deficit":
        assert (r[3:] == 0).all()


def test_disabled_exact_standing_and_other_terms_unchanged():
    m = Go2VelocityMdp()
    s = state()
    src = m.source_state(s)
    src[:, 25:] = torch.randn(4, 12)
    g = torch.tensor([[0.4, 0, 0.0]]).expand(4, 3)
    old = go2_gait_reward_terms(src, g, Go2GaitCfg())
    for mode in ("gaussian", "soft_deficit"):
        off = go2_gait_reward_terms(src, g, Go2GaitCfg(clearance_mode=mode, clearance_target=0.12))
        for key in old:
            torch.testing.assert_close(old[key], off[key], atol=0, rtol=0)
        on = go2_gait_reward_terms(
            src, torch.zeros_like(g), Go2GaitCfg(feet_clearance=0.05, clearance_mode=mode)
        )
        assert (on["feet_clearance"] == 0).all()


def test_position_region_gradient_no_reset_id_and_mdp_to_preserves_cfg():
    boxes = torch.tensor([[0.0, -4.0, 8.0, 4.0]])
    xy = torch.tensor([[-1.0, 0.0], [0.05, 0.0], [1.0, 0.0], [9.0, 0.0]], requires_grad=True)
    w = clearance_region_weight(xy, boxes)
    torch.testing.assert_close(w, torch.tensor([0.0, 0.5, 1.0, 0.0]))
    w.sum().backward()
    assert torch.isfinite(xy.grad).all() and xy.grad[1, 0] > 0
    m = Go2VelocityMdp()
    m.params.cfg.gait = Go2GaitCfg(feet_clearance=0.05, clearance_region="stairs")
    m.clearance_boxes = boxes
    m.max_step_reward = 0.105
    other = m.to("cpu")
    assert asdict(other.params.cfg) == asdict(m.params.cfg)
    assert other.max_step_reward == m.max_step_reward
    torch.testing.assert_close(other.clearance_boxes, boxes)


def contract():
    return dict(
        reward=dict(gait=asdict(Go2GaitCfg()), linear_tracking=3.0),
        reward_manager=dict(velocity=dict(params={}, weight=1.0)),
        actor="213",
        terrain="same",
        commands={"same": 1},
    )


def test_named_migration_rejects_every_other_semantic_change():
    old = contract()
    new = deepcopy(old)
    new["reward"]["gait"]["feet_clearance"] = 0.05
    new["reward_manager"]["velocity"]["params"]["gait"] = {"feet_clearance": 0.05}
    assert len(validate_reward_migration(old, new)) == 1
    for key in ("actor", "terrain", "commands"):
        bad = deepcopy(new)
        bad[key] = "wrong"
        with pytest.raises(ValueError, match="Protected"):
            validate_reward_migration(old, bad)
    for key in ("feet_slip", "diagonal_gait"):
        bad = deepcopy(new)
        bad["reward"]["gait"][key] = 0
        with pytest.raises(ValueError):
            validate_reward_migration(old, bad)


def test_cached_reward_relabel_only_clearance_and_unknown_legacy_speed():
    b = ReplayBuffer(8, 2, "cpu", 12, 16, 3, terrain_metadata=True, command_metadata=True)
    m = Go2VelocityMdp()
    for _ in range(8):
        row = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
        row["s"] = row["s1"] = flatten_state(state(2))
        row["g"][:, 0] = 0.4
        row["r"][:] = 0.123
        row.pop("command_speed_level")
        b.add(row)
    assert (b.data["command_speed_level"] == -1).all()
    old = b.state_dict()
    old["data"] = dict(old["data"])
    old["data"].pop("command_speed_level")
    b.load_state_dict(old)
    assert (b.data["command_speed_level"] == -1).all()
    before = b.data["r"].clone()
    relabel_replay_clearance(b.state_dict(), Go2GaitCfg(), m)
    torch.testing.assert_close(before, b.data["r"], atol=0, rtol=0)
    b.data["terrain_level"][:, 0] = 6
    b.data["command_domain"][:, 0] = 4
    b.wm_rare_fraction = 0.75
    b.sample_windows(1000, 8)
    assert b.wm_level_counts[6] >= 750
    b.start_scope_mask = torch.ones(6, 8, dtype=torch.bool)
    b.start_scope_mask[:, 6] = False
    starts, _ = b.sample_starts(100, 5)
    assert (starts["terrain_level"] != 6).all()


def test_real_mujoco_and_imagined_reward_share_geometry_and_formula():
    # Actual compiled collision atlas and measured torques; no substitute plane.
    from tools.go2_omni_common import configuration, environment

    cfg = configuration("stairs_clearance_gaussian")
    env = environment(cfg, 32)
    env.init_state()
    try:
        m = env.supermimic_mdp
        assert m.params.cfg.gait.feet_clearance == 0.05
        assert m.max_step_reward == pytest.approx(0.105)
        env.reset(seed=1026)
        action = torch.zeros(32, 12)
        tr = env.step(action).info["supermimic"]
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

        s, s1 = unflatten_state(tr["s"]), unflatten_state(tr["s1"])
        # Reward manager and standalone/H32 MDP share the same math, with equal measurements.
        expected = m.reward_transition(
            s, s1, tr["g"], tr["a"], tr["a_prev"], measured_torque=env.velocity_measured_torque
        )
        torch.testing.assert_close(expected, tr["r"], atol=1e-7, rtol=1e-6)
        imagined = m.to("cpu")
        terms = imagined.terms(s, s1, tr["g"], tr["a"], tr["a_prev"])
        real = m.terms(
            s, s1, tr["g"], tr["a"], tr["a_prev"], measured_torque=env.velocity_measured_torque
        )
        torch.testing.assert_close(terms["feet_clearance"], real["feet_clearance"], atol=0, rtol=0)
        assert m.obs(s, tr["a_prev"], tr["xi"], tr["f"]).shape == (32, 42)
        assert m.terrain.local_scan(s[0], s[1]).shape == (32, 77)
        old_cfg = configuration("stairs_omni_level7")
        old_env = environment(old_cfg, 32)
        old_env.init_state()
        try:
            assert (
                len(validate_reward_migration(old_env.supermimic_contract, env.supermimic_contract))
                == 4
            )
        finally:
            old_env.close()
            old_env.backend.close()
    finally:
        env.close()
        env.backend.close()


def test_4cm_stair_spacing_and_independent_rough_slope_cap():
    import numpy as np

    from unisupermimic.tasks.velocity.go2.terrain import terrain_contract, terrain_profile

    terrain = terrain_profile("mixed", levels=True, max_level=5, stair_step_increment=0.04)
    sub = terrain.generator.sub_terrains
    assert sub["rough"].profile_max_level == sub["slope"].profile_max_level == 4
    assert sub["stairs"].profile_max_level == 5
    for level in range(1, 6):
        for name in ("stairs", "stairs_down"):
            generated = sub[name].function((level + 0.1) / 6, np.random.default_rng(42))
            h = generated.heightfield.physical_heights_xy()
            assert abs(h[28, 80] - h[12, 80]) == pytest.approx(0.04 * level, abs=0.0051)
    legacy = terrain_contract(terrain_profile("mixed", levels=True, max_level=7))
    assert all(
        "stair_step_increment" not in c for c in legacy["generator"]["sub_terrains"].values()
    )


def test_opt_in_wm_balancing_preserves_actor_admission():
    b = ReplayBuffer(8, 100, "cpu", 12, 16, 3, terrain_metadata=True, command_metadata=True)
    # 90% Level3 vs 10% Level5 in real data; WM rare sampler balances observed levels.
    for _ in range(8):
        row = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
        row["command_domain"][:] = 4
        row["terrain_level"][:90] = 3
        row["terrain_level"][90:] = 5
        b.add(row)
    b.wm_rare_fraction = 1.0
    b.wm_rare_min_level = 3
    b.wm_rare_balance_levels = True
    b.sample_windows(10000, 8)
    assert 0.45 < float(b.wm_level_counts[5] / 10000) < 0.55
    b.start_scope_mask = torch.ones(6, 6, dtype=torch.bool)
    b.start_scope_mask[4, 5] = False
    starts, _ = b.sample_starts(1000, 5)
    assert not (starts["terrain_level"] == 5).any()
    copy = ReplayBuffer(8, 100, "cpu", 12, 16, 3, terrain_metadata=True, command_metadata=True)
    copy.load_state_dict(b.state_dict())
    assert copy.wm_rare_balance_levels and copy.wm_rare_min_level == 3
