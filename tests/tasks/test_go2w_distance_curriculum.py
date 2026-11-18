"""Distance-only local advancement, real geometric crossings and domain sampling."""

import copy

import numpy as np
import pytest
import torch
from tests.tasks.test_go2w_mode_critic import state

from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT
from unisupermimic.tasks.velocity.go2w.curriculum import UnifiedCurriculum, UnifiedCurriculumCfg

WEIGHTS = [0.40, 0.10, 0.08, 0.08, 0.12, 0.10, 0.08, 0.04]


def make(n=128):
    c = UnifiedCurriculum(
        n,
        np.zeros((10, 7, 3), np.float32),
        UnifiedCurriculumCfg(review_fraction=0),
        torch.zeros(n, dtype=torch.long),
    )
    c.reset(torch.arange(n), None)
    return c


def ordinary(c, domain):
    return ((c.fixed_domain == domain) & (c.coverage_level < 0)).nonzero().flatten()


def move(c, goal, velocity, steps=250, angular=0.0, meta=None, fallen=False):
    n = len(c.steps)
    s = state(n)
    s[0][:, :2] = c.previous_xy
    goals = torch.tensor(goal, dtype=torch.float32).expand(n, 3)
    s[2][:] = torch.tensor(velocity)
    s[3][:, 2] = angular
    for _ in range(steps):
        s[0][:, :2] += s[2][:, :2] * 0.02
        c.observe(s, goals, torch.full((n,), fallen), traversal=meta)
    return s


@pytest.mark.parametrize(
    "goal,velocity,angular,skill",
    [
        ((0.3, 0, 0), (0.3, 0, 0), 0, 1),
        ((-0.3, 0, 0), (-0.3, 0, 0), 0, 2),
        ((0, 0.2, 0), (0, 0.2, 0), 0, 3),
        ((0, -0.2, 0), (0, -0.2, 0), 0, 4),
        ((0, 0, 0.35), (0, 0, 0), 0.35, 5),
        ((0, 0, -0.35), (0, 0, 0), -0.35, 6),
        ((0.3, 0.1, 0.2), (0.3, 0.1, 0), 0.2, 1),
    ],
)
def test_one_episode_promotes_own_signed_skill(goal, velocity, angular, skill):
    c = make()
    ids = ordinary(c, 0)[:2]
    move(c, goal, velocity, angular=angular)
    c.episode_ended[ids[0]] = True
    c.reset(ids, None)
    assert c.flat_levels[ids[0], skill] == 1
    assert not c.flat_levels[ids[1]].any()
    assert c.flat_levels[ids[0]].sum() == 1


def test_backwards_wrong_direction_holds_demotes_and_bounds():
    c = make()
    ids = ordinary(c, 0)[:1]
    c.flat_levels[ids, 2] = 2
    c.terrain_level[ids] = 2
    move(c, (-0.3, 0, 0), (0.3, 0, 0))
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.flat_levels[ids, 2] == 1
    c.terrain_level[ids] = 1
    move(c, (-0.3, 0, 0), (-0.13, 0, 0))
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.flat_levels[ids, 2] == 1  # ratio .433: modest failure holds
    c.flat_levels[ids, 2] = 9
    c.terrain_level[ids] = 9
    move(c, (-0.3, 0, 0), (-0.3, 0, 0))
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.flat_levels[ids, 2] == 9


def test_stop_and_manual_reset_and_command_changes():
    c = make()
    ids = ordinary(c, 0)[:1]
    move(c, (0, 0, 0), (0, 0, 0), steps=300)
    c.reset(ids, None)
    assert not c.flat_levels[ids].any()  # manual reset cannot certify survival
    move(c, (0, 0, 0), (0, 0, 0), steps=300)
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.flat_levels[ids, 0] == 1
    move(c, (0.3, 0, 0), (0.3, 0, 0), steps=100)
    move(c, (-0.3, 0, 0), (-0.3, 0, 0), steps=100)
    assert c.skill_expected[ids, 1, 0] == pytest.approx(0.6)
    assert c.skill_expected[ids, 2, 0] == pytest.approx(0.6)
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.flat_levels[ids, 1] == 1 and c.flat_levels[ids, 2] == 1


@pytest.mark.parametrize("domain", [1, 2, 3])
def test_rough_and_slopes_forward_need_no_yaw(domain):
    c = make()
    ids = ordinary(c, domain)[:2]
    move(c, (0.3, 0, 0), (0.3, 0, 0))
    c.episode_ended[ids[0]] = True
    c.reset(ids, None)
    assert c.mastery[ids[0]] == 1 and c.mastery[ids[1]] == 0
    assert c.promotions[domain] == 1


def test_slope_non_crossing_episode_neither_promotes_nor_demotes():
    c = make()
    ids = ordinary(c, 2)[:1]
    c.mastery[ids] = 3
    c.terrain_level[ids] = 3
    move(c, (0, 0.3, 0), (0, 0.3, 0))
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.mastery[ids] == 3


@pytest.mark.parametrize(
    "domain,face,margin", [(4, 0.8, 0.25), (5, 0.8, 0.15), (6, 2.0, 0.30), (7, 2.0, 0.20)]
)
def test_stair_geometry_no_strict_success_required_and_crossing_is_latched(domain, face, margin):
    c = make()
    ids = ordinary(c, domain)[:1]
    n = len(c.steps)
    meta = torch.zeros(n, 10)
    meta[:, 3] = face
    meta[:, 5] = 1
    meta[:, 2] = 0.36
    meta[:, 9] = 1
    s = state(n)
    s[0][:, 2] = 0.36
    s[2][:, 0] = 0.3
    g = torch.tensor([0.3, 0, 0]).expand(n, 3)
    for _ in range(25):
        c.observe(s, g, torch.zeros(n, dtype=torch.bool), traversal=meta)
    s[0][:, 0] = face + margin - 0.01
    c.observe(s, g, torch.zeros(n, dtype=torch.bool), traversal=meta)
    assert not c.curriculum_crossed_obstacle[ids]
    s[0][:, 0] = face + margin + 0.01
    c.observe(s, g, torch.zeros(n, dtype=torch.bool), traversal=meta)
    assert c.curriculum_crossed_obstacle[ids] and not c.traversal_success[ids]
    c.observe(s, g, torch.ones(n, dtype=torch.bool), traversal=meta)
    c.episode_ended[ids] = True
    c.reset(ids, None)
    assert c.mastery[ids] == 1


@pytest.mark.parametrize("fault", ["side", "before_fall", "height", "backward"])
def test_stair_false_crossings_rejected(fault):
    c = make()
    ids = ordinary(c, 4)[:1]
    n = len(c.steps)
    meta = torch.zeros(n, 10)
    meta[:, 3] = 0.8
    meta[:, 5] = 1
    meta[:, 2] = 0.58
    meta[:, 7] = 0.22
    meta[:, 9] = 1
    s = state(n)
    s[0][:, 2] = 0.58
    g = torch.tensor([0.3, 0, 0]).expand(n, 3).clone()
    if fault == "side":
        s[0][:, 1] = 1
    if fault == "before_fall":
        c.episode_fall.fill_(True)
    if fault == "height":
        s[0][:, 2] = 0.35
    if fault == "backward":
        g[:, 0] = -0.3
    s[0][:, 0] = 1.1
    c.observe(s, g, torch.zeros(n, dtype=torch.bool), traversal=meta)
    assert not c.curriculum_crossed_obstacle[ids]


def test_fixed_4096_quotas_include_coverage_and_do_not_depend_on_global_levels():
    c = make(4096)
    counts = torch.bincount(c.fixed_domain, minlength=8)
    assert counts.sum() == 4096
    assert (counts / 4096 - torch.tensor(c.cfg.domain_weights)).abs().max() < 1 / 4096
    assert (c.coverage_level >= 0).sum() == 512
    for domain in range(1, 8):
        assert ((c.fixed_domain == domain) & (c.coverage_level == 9)).any()
        assert ((c.fixed_domain == domain) & (c.coverage_level < 0) & (c.terrain_level == 0)).any()
    c.skill_levels.fill_(9)
    c.reset(torch.arange(4096), None)
    assert torch.equal(counts, torch.bincount(c.fixed_domain, minlength=8))
    assert not c.mastery.any()
    old = copy.deepcopy(c.state_dict())
    d = make(4096)
    d.load_state_dict(old)
    for key, value in old["tensors"].items():
        torch.testing.assert_close(getattr(d, key), value)


def replay():
    b = ReplayBuffer(
        16,
        128,
        "cpu",
        num_joints=16,
        context_dim=0,
        goal_dim=7,
        terrain_metadata=True,
        command_metadata=True,
        state_layout=LAYOUT,
        level_count=10,
        domain_count=8,
        extra_fields={"target_id": ((), torch.long)},
    )
    domain = torch.arange(128) % 8
    for t in range(16):
        x = {k: torch.zeros_like(v[0]) for k, v in b.data.items()}
        x["s"][:, 0] = t
        x["ep_step"].fill_(t)
        x["command_domain"] = domain
        x["command_mode"] = (torch.arange(128) // 8) % 6
        x["terrain_level"].fill_(9)
        x["g"][:, 3:] = torch.nn.functional.one_hot(
            domain.new_tensor([0, 1, 2, 2, 3, 3, 3, 3])[domain], 4
        )
        b.add(x)
    return b


def test_actor_domain_then_command_stratification_high_levels_and_missing_domains():
    torch.manual_seed(2026)
    b = replay()
    start, hist = b.sample_starts(2048, 5, [0.06, 0.20, 0.12, 0.26, 0.20, 0.16], WEIGHTS)
    counts = torch.bincount(start["command_domain"], minlength=8)
    assert (counts / 2048 - torch.tensor(WEIGHTS)).abs().max() < 1 / 2048
    assert (start["terrain_level"] == 9).all()
    assert (hist["s"][:, 1:, 0] - hist["s"][:, :-1, 0] >= 0).all()
    b.data["done"][:, b.data["command_domain"][0] == 7] = True
    start, _ = b.sample_starts(2048, 5, [1, 0, 0, 0, 0, 0], WEIGHTS)
    assert not (start["command_domain"] == 7).any()
    assert b.last_actor_sampling["missing_domains"][7]
    assert (start["command_domain"] != 0).float().mean() > 0.55


def test_starts_filter_before_budgets_and_boundary_padding():
    b = replay()
    b.data["done"][13] = True
    # A manually ended trajectory cannot contribute history before its boundary.
    start, hist = b.sample_starts(
        2048, 5, None, WEIGHTS, lambda f: torch.ones_like(f, dtype=torch.bool)
    )
    for i in range(len(start["s"])):
        t = int(start["s"][i, 0])
        values = hist["s"][i, :, 0]
        if t >= 14:
            assert (values >= 14).all()
    b.data["g"][:, :, 3:] = 0
    with pytest.raises(ValueError, match="No valid physical"):
        b.sample_starts(64, 5, None, WEIGHTS)


def test_migration_rejects_reward_control_and_algorithm_changes():
    from dataclasses import asdict

    from tests.tasks.test_go2w_world_model import contract

    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
    from unisupermimic.tasks.velocity.go2w.migration import (
        validate_algo_migration,
        validate_migration_contract,
    )

    old = contract() | {
        "curriculum_protocol": "ten_level_per_skill_hysteresis_review_and_coverage",
        "commands": {"curriculum": {"window": 64}, "actor_level_lookahead": 1},
    }
    new = copy.deepcopy(old)
    new["curriculum_protocol"] = "per_env_episode_distance_fixed_domain_v1"
    new["commands"] = {"curriculum": {"review_fraction": 0.2}}
    validate_migration_contract(old, new)
    for key in ("reward", "actions", "terrain_fingerprint"):
        bad = copy.deepcopy(new)
        bad[key] = "different"
        with pytest.raises(ValueError):
            validate_migration_contract(old, bad)
    cfg = asdict(SuperMimicConfig())
    assert validate_algo_migration(cfg, cfg | {"actor_domain_weights": WEIGHTS}) == [
        "actor_domain_weights"
    ]
    for key in ("horizon", "buffer_steps", "pi_batch"):
        with pytest.raises(ValueError, match="forbidden"):
            validate_algo_migration(cfg, cfg | {key: cfg[key] + 1})


@pytest.mark.slow
def test_physical_auto_reset_consumes_completed_episode_before_clear(monkeypatch):
    from unisupermimic.tasks.velocity.go2w.config import load_config
    from unisupermimic.tasks.velocity.go2w.env import make_go2w_env

    env = make_go2w_env(load_config("train"), 16)
    try:
        cur = env.command_manager.get_term("velocity").curriculum
        seen = []
        original = cur._mastery

        def capture(ids):
            ended = cur.episode_ended[ids]
            if ended.any():
                seen.append(cur.steps[ids][ended].clone())
            return original(ids)

        monkeypatch.setattr(cur, "_mastery", capture)
        env.reset()
        assert not seen
        for _ in range(501):
            result = env.step(torch.zeros(16, 16))
            assert torch.isfinite(result.reward).all()
        assert seen and all((steps > 0).all() for steps in seen)
        before = (cur.promotions + cur.demotions + cur.holds).clone()
        env.reset()
        torch.testing.assert_close(cur.promotions + cur.demotions + cur.holds, before)
    finally:
        env.close()
        env.backend.close()
