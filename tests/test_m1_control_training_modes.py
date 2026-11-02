from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch


def _phase_test_trainer(mode: str, *, freeze_prior: bool, safe: bool = True):
    from supermimic.motion_prior.trainer import M1AlternatingTrainer

    trainer = object.__new__(M1AlternatingTrainer)
    trainer.cfg = SimpleNamespace(
        training_mode=mode,
        freeze_prior=freeze_prior,
        wm_updates_per_event=1,
        prior_updates_per_event=1,
        actor_updates_per_event=1,
        wm_readiness_min_events=1,
        wm_readiness_max_nse_1=1.0,
        wm_readiness_max_nse_horizon=1.0,
        actor_pretrain_updates=0,
        prior_warmup_updates=0,
        readiness_window_steps=4,
        readiness_min_samples=1,
        readiness_min_survival=0.9,
        readiness_max_root_tracking_error=0.2,
        readiness_max_arm_tracking_error=0.25,
        readiness_max_wheel_tracking_error=0.75,
        readiness_max_action_saturation=0.1,
        controlled_joint_min_wm_events=1,
        controlled_joint_actor_pretrain_updates=0,
        controlled_joint_min_real_samples=1,
        controlled_joint_min_survival=0.9,
        controlled_joint_min_base_height_m=0.22,
        controlled_joint_max_base_tilt_rad=0.9,
        controlled_joint_max_action_saturation=0.2,
    )
    trainer.wm_trainer = SimpleNamespace(
        cfg=SimpleNamespace(batches_per_update=1, window=8),
        update=lambda _buffer: {"nse_1step": 0.05, "nse_kstep": 0.25},
    )
    trainer.buffer = object()
    trainer.wm_update_events = 0
    trainer.wm_optimizer_steps = 0
    trainer.prior_optimizer_steps = 0
    trainer.actor_optimizer_steps = 0
    trainer.stage = "wm_warmup"
    trainer._actor_ready_window = deque([{
        "survival": 1.0,
        "root_tracking_error": 0.1,
        "arm_tracking_error": 0.1,
        "wheel_tracking_error": 0.1,
        "action_saturation": 0.0,
        "base_height_m": 0.35 if safe else 0.1,
        "base_tilt_rad": 0.05 if safe else 1.2,
    }], maxlen=4)
    trainer._real_metrics = {}
    trainer.last_wm_metrics = {"nse_1step": 0.05, "nse_kstep": 0.25}
    trainer.last_actor_metrics = {}
    trainer.last_prior_metrics = {}
    trainer.last_wm_readiness = {}
    updates = {"actor": 0, "prior": 0}

    def update_actor():
        updates["actor"] += 1
        trainer.actor_optimizer_steps += 1
        return {"actor/test_update": 1.0}

    def update_prior():
        updates["prior"] += 1
        trainer.prior_optimizer_steps += 1
        return {"prior/test_update": 1.0}

    trainer._update_actor = update_actor
    trainer._update_prior = update_prior
    return trainer, updates


def test_m1_mixed_action_position_and_wheel_mapping():
    from supermimic.robots.spec import M1_DEFAULT, M1_JOINTS, get_robot_spec
    from supermimic.tasks.action_contract import mixed_m1_setpoints

    default = torch.tensor(M1_DEFAULT)
    scales = torch.tensor(get_robot_spec("m1_piper").action_scales)
    wheel = torch.tensor([n.endswith("FOOT_JOINT") for n in M1_JOINTS])
    limits = torch.tensor([[-3.0, 3.0]] * 22)
    action = torch.ones(2, 22) * 1.5
    action[1] = -1.0
    target = mixed_m1_setpoints(action, default, scales, wheel, limits)

    for idx, name in enumerate(M1_JOINTS):
        expected = scales[idx] if name.endswith("FOOT_JOINT") else default[idx] + scales[idx]
        torch.testing.assert_close(target[0, idx], expected)
        expected = -scales[idx] if name.endswith("FOOT_JOINT") else default[idx] - scales[idx]
        torch.testing.assert_close(target[1, idx], expected)
    piper2 = M1_JOINTS.index("piper_x_joint2")
    tight_limits = limits.clone()
    tight_limits[piper2] = torch.tensor([2.4, 2.6])
    clipped = mixed_m1_setpoints(action[:1], default, scales, wheel, tight_limits)
    torch.testing.assert_close(clipped[0, piper2], torch.tensor(2.6))


def test_m1_piper_reference_action_envelope_covers_audited_references():
    from supermimic.robots.spec import M1_DEFAULT, M1_JOINTS, get_robot_spec

    spec = get_robot_spec("m1_piper")
    scales = dict(zip(M1_JOINTS, spec.action_scales))
    defaults = dict(zip(M1_JOINTS, M1_DEFAULT))
    # L-task prior_mean references at the audited fixed reset/goal seed.
    assert scales["piper_x_joint2"] >= defaults["piper_x_joint2"] - 2.2284
    assert scales["piper_x_joint4"] >= -0.5385 - defaults["piper_x_joint4"]
    assert scales["piper_x_joint2"] == pytest.approx(0.35)
    assert scales["piper_x_joint4"] == pytest.approx(0.35)


def test_m1_warmup_exploration_is_bounded_and_matches_replay_action():
    from supermimic.tasks.action_contract import add_bounded_action_exploration

    policy_action = torch.tensor([[0.95, -0.95], [0.2, -0.2]])
    noise = torch.tensor([[0.2, -0.2], [0.1, 0.1]])
    selected = torch.tensor([True, False])
    executed = add_bounded_action_exploration(policy_action, noise, selected)
    torch.testing.assert_close(executed, torch.tensor([[1.0, -1.0], [0.2, -0.2]]))
    assert torch.isfinite(executed).all() and (executed.abs() <= 1.0).all()


def test_m1_actor_only_mode_never_updates_prior():
    trainer, updates = _phase_test_trainer("actor_only", freeze_prior=True)
    trainer.update_event(policy_allowed=True)
    assert trainer.stage == "actor_only_prior_frozen"
    assert updates == {"actor": 1, "prior": 0}


def test_m1_auto_joint_honors_manual_freeze_even_if_quality_gates_pass():
    trainer, updates = _phase_test_trainer("auto_joint", freeze_prior=True)
    trainer.update_event(policy_allowed=True)
    assert trainer.stage == "actor_only_prior_frozen"
    assert updates == {"actor": 1, "prior": 0}


def test_m1_controlled_joint_requires_explicit_unfreeze_and_real_safety():
    from supermimic.motion_prior.trainer import M1AlternatingTrainer

    with pytest.raises(ValueError, match="no-freeze-prior"):
        M1AlternatingTrainer(
            wm=None, wm_trainer=None, prior=None, actor=None,
            buffer=SimpleNamespace(task_type="M"), default_joint_pos=None,
            soft_limits=None, fk=None, tcp_index=0,
            cfg=SimpleNamespace(
                prior_horizon=10, actor_horizon=10, replanning_cadence=10,
                state_source="privileged", prior_mode="prior_mean",
                batch_size=1, micro_batch_size=1, training_mode="controlled_joint_experiment",
                freeze_prior=True,
            ),
            robot_joint_names=(), robot_model_sha256=None, command=None,
        )

    trainer, updates = _phase_test_trainer("controlled_joint_experiment", freeze_prior=False, safe=False)
    trainer.update_event(policy_allowed=True)
    assert trainer.stage == "controlled_joint_warmup"
    assert trainer.prior_optimizer_steps == 0
    assert trainer.last_wm_readiness["schedule/wm_ready"] == 1.0
    assert trainer._controlled_joint_readiness(policy_allowed=True)["schedule/controlled_joint_base_height_pass"] == 0.0

    trainer._actor_ready_window[-1]["base_height_m"] = 0.35
    trainer._actor_ready_window[-1]["base_tilt_rad"] = 0.05
    trainer.update_event(policy_allowed=True)
    assert trainer.stage == "controlled_joint_experiment"
    assert updates == {"actor": 2, "prior": 1}
