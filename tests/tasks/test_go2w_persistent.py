"""Coverage/reset and isolated reward-change regressions, CPU only."""

import numpy as np
import pytest
import torch
from unilab.managers.rng import ManagerRng

from unisupermimic.tasks.velocity.go2.command_sampler import fixed_domains
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp
from unisupermimic.tasks.velocity.go2w.reward_core import VelocityRewardCfg
from unisupermimic.tasks.velocity.go2w.terrain import Go2WTerrainCurriculum


def curriculum(num_envs=4096):
    origins = np.arange(8 * 5 * 3, dtype=np.float32).reshape(8, 5, 3)
    c = Go2WTerrainCurriculum(
        num_envs, origins, enabled=True, max_level=7, phase="skill",
        coverage_weights=(.2, .25, .2, .15, .1, .06, .03, .01),
    )
    c.fixed_domain = fixed_domains(num_envs, (.25, .2, .1, .1, .175, .175))
    c.configure_persistent_coverage()
    return c


@pytest.mark.parametrize("num_envs", [128, 512, 4096])
def test_coverage_survives_resets_without_certifying_skill(num_envs):
    c = curriculum(num_envs)
    covered = c.coverage_level >= 1
    assert not (covered & (c.fixed_domain == 0)).any()
    rng = ManagerRng.seeded(2026, device="cpu")
    ids = torch.arange(num_envs)
    for _ in range(4):
        c.steps.fill_(500)
        c.episode_progress.fill_(2)
        c.commanded_progress.fill_(2)
        c.eligible_steps.fill_(500)
        origins = c.reset(ids, rng)
        torch.testing.assert_close(c.terrain_level[covered], c.coverage_level[covered])
        torch.testing.assert_close(origins[covered], c.origins[c.coverage_level[covered], c.terrain_type[covered]])
    assert (c.mastery[covered] == 0).all()
    assert (c.success_streak[covered] == 0).all()
    if num_envs == 4096:
        counts = torch.bincount(c.coverage_level[covered], minlength=8)
        assert counts.tolist() == [0, 96, 96, 96, 80, 64, 48, 32]
        assert covered.sum() == 512
        for domain in (4, 5):
            assert ((c.fixed_domain == domain) & (c.coverage_level >= 6)).sum() == 20
    # Exact restoration preserves roles and rejects different allocations.
    restored = curriculum(num_envs)
    restored.load_state_dict(c.state_dict())
    torch.testing.assert_close(restored.coverage_level, c.coverage_level)
    values = c.state_dict()
    values["tensors"]["coverage_level"][covered.nonzero()[0]] = 0
    with pytest.raises(ValueError, match="roles changed"):
        restored.load_state_dict(values)


def test_stop_gate_changes_only_stop_term_and_preserves_baseline():
    original = Go2WVelocityMdp(reward=VelocityRewardCfg())
    narrowed = Go2WVelocityMdp(reward=VelocityRewardCfg(stop_reward_command_sigma=.02))
    p = torch.tensor([[0., 0., .42]])
    state = (p, torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3), original.default_q[None], torch.zeros(1, 16))
    action = torch.zeros(1, 16)
    goal = torch.tensor([[0., .06, 0.]])
    a = original.terms(state, state, goal, action, action)
    b = narrowed.terms(state, state, goal, action, action)
    for key in a.keys() - {"stand_still"}:
        torch.testing.assert_close(a[key], b[key])
    assert a["stand_still"] > .1
    assert b["stand_still"] < .0001
    for mdp in (original, narrowed):
        assert mdp.terms(state, state, goal * 0, action, action)["stand_still"] == .5


def test_initialization_rejects_physics_change_and_used_learner(tmp_path):
    from copy import deepcopy
    from dataclasses import asdict
    from types import SimpleNamespace

    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
    from unisupermimic.tasks.velocity.go2w.checkpoint import initialize_experiment

    contract = {
        "schema": "go2w_velocity_supermimic_v2",
        "curriculum_protocol": "wm_coverage_then_skill_v2_episode_fixed_domain",
        "commands": {"reward": {"sigma_y": .25}}, "reward": {"sigma_y": .25},
        "control": {"leg_position_scale": .25},
    }
    cfg = SuperMimicConfig()
    pi, wm = torch.nn.Linear(2, 2), torch.nn.Linear(3, 3)
    source = {"task_contract": contract, "args": asdict(cfg), "iteration": 7000,
              "pi": pi.state_dict(), "wm": wm.state_dict(),
              "opt_wm": torch.optim.Adam(wm.parameters()).state_dict()}
    path = tmp_path / "source.pt"
    torch.save(source, path)
    target = deepcopy(contract)
    target["curriculum_protocol"] = "persistent_wm_eighth_domain_level_quotas"
    target["commands"]["persistent_wm_coverage"] = True
    for reward in (target["reward"], target["commands"]["reward"]):
        reward.update(sigma_y=.15, stop_reward_command_sigma=.02)
    learner = SimpleNamespace(
        cfg=cfg, env=SimpleNamespace(supermimic_contract=target),
        pi=torch.nn.Linear(2, 2), wm=torch.nn.Linear(3, 3), pi_avg=None,
        extra_state={}, iteration=0, buffer=SimpleNamespace(size=0), update_counts={"wm": 0},
    )
    learner.opt_wm = torch.optim.Adam(learner.wm.parameters())
    audit = initialize_experiment(learner, path)
    assert audit["source_iteration"] == 7000
    torch.testing.assert_close(learner.pi.weight, pi.weight)
    target["control"]["leg_position_scale"] = .5
    with pytest.raises(ValueError, match="protected task"):
        initialize_experiment(learner, path)
    learner.iteration = 1
    with pytest.raises(ValueError, match="untouched"):
        initialize_experiment(learner, path)


def test_full_joint_range_reaches_deployment_fold_posture_and_has_finite_gradients():
    from unisupermimic.tasks.velocity.go2w.contract import (
        DEFAULT_Q,
        EFFORT,
        MixedControlCfg,
        leg_position_target,
        mixed_torque,
        soft_limits,
    )

    default = torch.tensor(DEFAULT_Q)
    control = MixedControlCfg(full_leg_target_range=True, leg_limit_fraction=.95)
    limits = soft_limits(fraction=.95)
    a = torch.stack((-torch.ones(16), torch.zeros(16), torch.ones(16)))
    target = leg_position_target(a, control, default, limits)
    torch.testing.assert_close(target[0], limits[:, 0])
    torch.testing.assert_close(target[1], default)
    torch.testing.assert_close(target[2], limits[:, 1])
    # The confirmed rl_sar Go2W pre-getup target is (0,1.36,-2.65).
    folded = torch.tensor([0., 1.36, -2.65] * 4)
    assert ((folded >= limits[:, 0]) & (folded <= limits[:, 1])).all()
    spans = torch.where(folded >= default, limits[:, 1] - default, default - limits[:, 0])
    inverse = (folded - default) / spans
    torch.testing.assert_close(leg_position_target(inverse, control, default, limits), folded)
    action = torch.linspace(-.8, .8, 16, requires_grad=True)
    torque = mixed_torque(default, torch.zeros(16), action, control, default, limits)
    assert (torque.abs() <= torch.tensor(EFFORT)).all()
    leg_position_target(action, control, default, limits).square().sum().backward()
    assert torch.isfinite(action.grad).all()
    assert (action.grad[:12].abs() > 0).all()


def test_joint_range_materializes_consistently_in_owner_config():
    from unisupermimic.tasks.velocity.go2w.config import load_config

    cfg = load_config("joint_range")
    assert cfg.actions["mixed"].control.full_leg_target_range
    assert cfg.actions["mixed"].control.leg_limit_fraction == .95
    assert cfg.commands["velocity"].persistent_wm_coverage
