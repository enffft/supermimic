"""Ten-level mastery, contiguous history, traversal reward and H8 admission invariants."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.tasks.test_go2w_mode_critic import state
from tools.audit_go2w_warmstart import evaluate_direction_gate

from unisupermimic.rl.algos.supermimic.history import HistoricalWindowBank
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT
from unisupermimic.tasks.velocity.go2w.curriculum import UnifiedCurriculum, UnifiedCurriculumCfg
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp
from unisupermimic.tasks.velocity.go2w.reward_core import VelocityRewardCfg
from unisupermimic.tasks.velocity.go2w.traversal import reward_terms, success_proxy


def test_h8_admission_requires_h1_h8_and_sampled_contacts():
    rows = []
    for h in (1, 8, 16):
        for source, contact in [
            ("neutral_control", "no_wheel_contact"),
            ("actor_plus_collection_noise", "wheel_contact"),
        ]:
            rows.append(
                dict(
                    case=source,
                    epsilon=0.002,
                    horizon=h,
                    axes=["vx", "vy", "yaw"],
                    all_finite=True,
                    physical_norm=[1.0, 1.0, 1.0],
                    cosine=[1.0, 1.0, 1.0] if h < 16 else [-1.0, -1.0, -1.0],
                    action_source=source,
                    contact={"category": contact},
                )
            )
    expected = [(source, 0.002) for source in ("neutral_control", "actor_plus_collection_noise")]
    result = evaluate_direction_gate(rows, [], 8, expected)
    assert result["gradient_direction_gate_passed"] and len(result["diagnostic_warnings"]) == 6
    rows[2]["cosine"][1] = -0.1
    assert not evaluate_direction_gate(rows, [], 8, expected)["gradient_direction_gate_passed"]
    assert not evaluate_direction_gate(rows[:2], [], 8, expected)["gradient_direction_gate_passed"]


def curriculum(n=128):
    return UnifiedCurriculum(
        n,
        np.zeros((10, 7, 3), np.float32),
        UnifiedCurriculumCfg(review_fraction=0),
        torch.zeros(n, dtype=torch.long),
    )


def test_distance_curriculum_state_is_versioned_and_roundtrips_rng():
    c = curriculum()
    saved = copy.deepcopy(c.state_dict())
    d = curriculum()
    d.load_state_dict(saved)
    torch.testing.assert_close(c.rng.get_state(), d.rng.get_state())
    saved.pop("schema")
    with pytest.raises(ValueError, match="migration"):
        d.load_state_dict(saved)
    assert c.start_scope_mask().all()


def metadata(n):
    return torch.tensor([[2.0, 0.0, 0.42, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0]]).expand(n, -1).clone()


def test_progress_direction_once_budget_and_differentiable_success():
    prev = state(1)
    nxt = tuple(t.clone() for t in prev)
    nxt[0][:, 0] = 0.1
    meta = metadata(1)
    peak = torch.zeros(1)
    forward = reward_terms(prev, nxt, meta, peak, 0.99, 1.0, 2.0, 0.02)
    assert forward["stair_progress"] > 0
    backward = reward_terms(nxt, prev, meta, peak, 0.99, 1.0, 2.0, 0.02)
    assert backward["stair_progress"] < 0
    assert success_proxy(prev, meta) < 1e-5
    # Enter, leave, re-enter: reward budget cannot be repaid on a second visit.
    trajectory = state(4)
    trajectory[0][:, 0] = torch.tensor([2.0, 2.1, 0.5, 2.1])
    trajectory[0].requires_grad_()
    terms = reward_terms(
        state(4), trajectory, metadata(4), torch.zeros(4), 0.99, 1.0, 2.0, 0.02, horizon=4
    )
    assert terms["stair_success"][2:].sum() == 0
    assert 0 < float(terms["stair_success"].sum().detach() * 0.02) <= 2.0
    terms["stair_success"].sum().backward()
    assert torch.isfinite(trajectory[0].grad).all()


def test_shared_real_imagined_reward_per_term_parity_and_actor_privacy():
    cfg = VelocityRewardCfg(
        wheel_consistency=0.05, side_drift=0.1, stair_progress=1.0, stair_success=2.0
    )
    mdp = Go2WVelocityMdp(
        terrain=TerrainHeightField.flat(), reward=cfg, terrain_modes=True, unified=True
    )
    prior = state(8)
    nxt = state(8)
    nxt[0][:, 0] = torch.arange(8) * 0.2
    goals = torch.zeros(8, 7)
    goals[:, 0] = 0.3
    goals[:, 3] = 1
    domain = torch.arange(8)
    meta = metadata(8)
    meta[:4, 9] = 0
    action = torch.rand(8, 16) * 0.1
    expected = mdp.terms(
        prior,
        nxt,
        goals,
        action,
        action,
        command_domain=domain,
        traversal=meta,
        success_peak=torch.zeros(8),
    )
    start = dict(command_domain=domain, traversal=meta, success_peak=torch.zeros(8))
    actual = mdp.terms(prior, nxt, goals, action, action, **mdp.imagination_reward_kwargs(start, 1))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    obs = mdp.obs(nxt, action, torch.empty(8, 0), torch.zeros(8, dtype=torch.long))
    meta[:, :3] += 100
    torch.testing.assert_close(
        obs,
        mdp.obs(nxt, action, torch.empty(8, 0), torch.zeros(8, dtype=torch.long)),
        atol=0,
        rtol=0,
    )


def history_windows(n=32, k=8):
    return dict(
        s=torch.zeros(n, k, 41),
        s1=torch.ones(n, k, 41),
        a=torch.zeros(n, k, 16),
        done=torch.zeros(n, k, dtype=torch.bool),
        ep_step=torch.arange(k, dtype=torch.int32).expand(n, k).clone(),
        target_id=torch.arange(n)[:, None].expand(n, k).clone(),
        g=torch.zeros(n, k, 7),
        command_domain=torch.arange(n)[:, None].expand(n, k) % 8,
        terrain_level=torch.arange(n)[:, None].expand(n, k) % 10,
        command_mode=torch.arange(n)[:, None].expand(n, k) % 6,
        mask=torch.ones(n, k),
    )


def test_history_rejects_reset_target_crossing_and_roundtrips_rng():
    bank = HistoricalWindowBank(16, 8, protocol="strict")
    data = history_windows()
    data["done"][0, 3] = True
    data["target_id"][1, 4:] += 1
    data["ep_step"][2, 4:] += 1
    bank.add(data)
    assert bank.size == 16
    assert not bank.data["done"][:16, :-1].any()
    assert (bank.data["target_id"][:16] == bank.data["target_id"][:16, :1]).all()
    saved = copy.deepcopy(bank.state_dict())
    other = HistoricalWindowBank(16, 8, protocol="strict")
    other.load_state_dict(saved)
    expected = bank.state_dict()
    other.load_state_dict(copy.deepcopy(expected))
    a, b = bank.sample(20, "cpu"), other.sample(20, "cpu")
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    with pytest.raises(ValueError):
        HistoricalWindowBank(16, 8, protocol="wrong").load_state_dict(saved)


def test_history_actual_ratio_counts_only_final_batch_and_replay_1024():
    replay = ReplayBuffer(
        1024,
        4,
        "cpu",
        num_joints=16,
        context_dim=0,
        goal_dim=7,
        state_layout=LAYOUT,
        terrain_metadata=True,
        command_metadata=True,
        level_count=10,
        domain_count=8,
        extra_fields={"target_id": ((), torch.long)},
        history_capacity=16,
    )
    for t in range(32):
        transition = {key: torch.zeros_like(value[0]) for key, value in replay.data.items()}
        transition["ep_step"].fill_(t)
        transition["terrain_level"].fill_(9)
        replay.add(transition)
    assert replay.size == 32 and replay.history.size
    assert replay.wm_level_counts.sum() == 0
    replay.sample_windows(64, 8)
    assert replay.history_actual == [32, 32]
    assert replay.wm_level_counts.sum() == 64
    assert replay.sampling_counts["wm_modes"].sum() == 64 * 8
    restored = ReplayBuffer(
        1024,
        4,
        "cpu",
        num_joints=16,
        context_dim=0,
        goal_dim=7,
        state_layout=LAYOUT,
        terrain_metadata=True,
        command_metadata=True,
        level_count=10,
        domain_count=8,
        extra_fields={"target_id": ((), torch.long)},
        history_capacity=16,
    )
    restored.load_state_dict(replay.state_dict())
    assert restored.history_actual == [32, 32]
