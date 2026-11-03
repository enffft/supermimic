"""M1 Action22 exploration and staged Prior gate regression tests."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch


def _gate_trainer(tmp_path=None, *, freeze_prior=False):
    from types import SimpleNamespace

    from supermimic.motion_prior.trainer import M1AlternatingTrainer
    from supermimic.robots.spec import get_robot_spec

    trainer = object.__new__(M1AlternatingTrainer)
    trainer.cfg = SimpleNamespace(
        prior_gate_evidence_path=tmp_path,
        freeze_prior=freeze_prior,
        readiness_min_samples=1,
        actor_pretrain_updates=0,
        readiness_min_survival=0.9,
        readiness_max_root_tracking_error=0.2,
        readiness_max_arm_tracking_error=0.25,
        readiness_max_wheel_tracking_error=0.75,
        readiness_max_action_saturation=0.1,
    )
    trainer.wm = torch.nn.Linear(2, 2)
    trainer.robot_spec = get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    trainer.command = SimpleNamespace(cfg=SimpleNamespace(reset_pose_profile="dataset_v3_3_l_high_carry"))
    trainer.prior = SimpleNamespace(source={"sha256": "prior-test-sha"})
    trainer._prior_calibration_wm = None
    trainer._prior_calibration_wm_sha256 = None
    trainer._prior_calibration_checkpoint = None
    trainer._prior_calibration_checkpoint_signature = None
    trainer._force_prior_frozen = False
    trainer.actor_optimizer_steps = 1
    trainer._actor_ready_window = deque([{
        "survival": 1.0,
        "root_tracking_error": 0.1,
        "arm_tracking_error": 0.1,
        "wheel_tracking_error": 0.1,
        "action_saturation": 0.0,
    }])
    return trainer


def test_m1_mixed_action_position_and_wheel_mapping():
    from supermimic.robots.spec import M1_JOINTS, get_robot_spec
    from supermimic.tasks.action_contract import mixed_m1_setpoints

    spec = get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    default = torch.tensor(spec.default_joint_pos)
    scales = torch.tensor(spec.action_scales)
    wheel = torch.tensor([n.endswith("FOOT_JOINT") for n in M1_JOINTS])
    limits = torch.tensor([[-3.0, 3.0]] * 22)
    bounds = torch.tensor(spec.position_target_bounds)
    action = torch.ones(2, 22)
    action[1] = -1.0
    target = mixed_m1_setpoints(action, default, scales, wheel, limits, bounds)

    for idx, name in enumerate(M1_JOINTS):
        if name.endswith("FOOT_JOINT"):
            expected = torch.tensor((scales[idx], -scales[idx]))
        else:
            expected = torch.tensor((bounds[idx, 1], bounds[idx, 0]))
        torch.testing.assert_close(target[:, idx], expected)
    position_mask = ~wheel
    assert torch.all(default[position_mask] >= bounds[position_mask, 0])
    assert torch.all(default[position_mask] <= bounds[position_mask, 1])


def test_m1_exploration_is_bounded_and_distinguishes_clean_and_executed_actions():
    from supermimic.motion_prior.exploration import M1BoundedGaussianExplorer
    from supermimic.tasks.action_contract import add_bounded_action_exploration

    explorer = M1BoundedGaussianExplorer(
        leg_std=0.02, arm_std=0.015, wheel_std=0.05, env_fraction=1.0,
    )
    action = torch.zeros(8, 22)
    noise, mask, scales = explorer.sample(action, iteration=0, total_iterations=100)
    assert mask.all() and scales.shape == (22,)
    executed = add_bounded_action_exploration(action, noise, mask)
    assert torch.equal(executed, noise.clamp(-1, 1))
    assert torch.allclose(scales[[3, 7, 11, 15]], torch.full((4,), 0.05))
    assert torch.allclose(scales[-6:], torch.full((6,), 0.015))
    assert torch.allclose(scales[[i for i in range(22) if i not in (3, 7, 11, 15) and i < 16]], torch.full((12,), 0.02))
    clean = torch.tensor([[0.95] * 22])
    replay_action = add_bounded_action_exploration(clean, torch.ones_like(clean) * 0.2, torch.tensor([True]))
    assert torch.equal(replay_action, torch.ones_like(replay_action))
    assert torch.all(clean == 0.95)  # raw Actor output remains available for observation/diagnostics


def test_legacy_m1_training_modes_fail_fast():
    from supermimic.motion_prior.trainer import M1AlternatingTrainer

    with pytest.raises(ValueError, match="single staged_action22_v1"):
        M1AlternatingTrainer(
            wm=None, wm_trainer=None, prior=None, actor=None,
            buffer=SimpleNamespace(task_type="L"), default_joint_pos=None,
            soft_limits=None, fk=None, tcp_index=0,
            cfg=SimpleNamespace(
                prior_horizon=10, actor_horizon=10, replanning_cadence=10,
                state_source="privileged", prior_mode="prior_mean",
                batch_size=1, micro_batch_size=1, training_mode="controlled_joint_experiment",
                prior_updates_per_event=0, actor_updates_per_event=1,
            ),
            robot_joint_names=(), robot_model_sha256=None, command=None,
        )


def test_prior_gate_fails_closed_without_heldout_physical_evidence():
    from supermimic.motion_prior.trainer import module_sha256

    trainer = _gate_trainer(freeze_prior=False)
    result = trainer._prior_gate()
    assert result["schedule/prior_gate_pass"] == 0.0
    assert result["schedule/prior_gate_evidence_present"] == 0.0
    assert result["schedule/prior_gate_actor_ready"] == 1.0
    assert module_sha256(trainer.wm)


def test_prior_gate_accepts_only_matching_complete_evidence(tmp_path):
    import json

    from supermimic.motion_prior.trainer import module_sha256
    from supermimic.tasks.action_contract import serialize_m1_action_contract

    evidence_path = tmp_path / "gate.json"
    trainer = _gate_trainer(evidence_path, freeze_prior=False)
    spec = trainer.robot_spec
    from supermimic.motion_prior.trainer import M1AlternatingTrainer
    from supermimic.tasks.action_contract import serialize_m1_action_contract

    wm_sha = module_sha256(trainer.wm)
    checkpoint_path = tmp_path / "wm_snapshot.pt"
    torch.save({
        "format": M1AlternatingTrainer.CHECKPOINT_FORMAT,
        "wm": trainer.wm.state_dict(),
        "world_model_sha256": wm_sha,
        "action_mapping": serialize_m1_action_contract(spec),
        "prior_source": {"sha256": "prior-test-sha"},
        "reset_contract": {
            "profile": spec.default_pose_profile,
            "task_conditional": False,
            "default_base_height_m": float(spec.default_base_height),
            "default_joint_pos": tuple(spec.default_joint_pos),
        },
    }, checkpoint_path)
    evidence_path.write_text(json.dumps({
        "audit_type": "m1_real_mujoco_wm_action_response_v1",
        "action_response_pass": True,
        "actor_task_pass": True,
        "real_task_safety_pass": True,
        "world_model_sha256": wm_sha,
        "checkpoint": str(checkpoint_path),
        "action_mapping": serialize_m1_action_contract(trainer.robot_spec),
        "heldout_state_count": 128,
    }))
    # A later live WM update must not invalidate calibration against this
    # immutable checkpoint snapshot.
    with torch.no_grad():
        trainer.wm.weight.add_(0.25)
    result = trainer._prior_gate()
    assert result["schedule/prior_gate_pass"] == 1.0
    assert result["schedule/prior_gate_wm_snapshot_matches"] == 1.0
    assert trainer._prior_calibration_wm_sha256 == wm_sha

    trainer.cfg.freeze_prior = True
    result = trainer._prior_gate()
    assert result["schedule/prior_gate_pass"] == 0.0
    assert result["schedule/prior_gate_freeze_config_clear"] == 0.0


def test_prior_validation_failure_restores_weights_and_optimizer_then_freezes():
    from copy import deepcopy

    from supermimic.motion_prior.trainer import M1AlternatingTrainer, module_sha256

    trainer = object.__new__(M1AlternatingTrainer)
    trainer.stage = "prior_calibration"
    trainer._last_validation_pass = None
    trainer.prior = torch.nn.Linear(3, 2)
    trainer.prior_optimizer = torch.optim.Adam(trainer.prior.parameters(), lr=1e-3)
    loss = trainer.prior(torch.ones(1, 3)).square().sum()
    loss.backward(); trainer.prior_optimizer.step(); trainer.prior_optimizer.zero_grad(set_to_none=True)
    weight_before = deepcopy(trainer.prior.state_dict())
    optimizer_before = deepcopy(trainer.prior_optimizer.state_dict())
    hash_before = module_sha256(trainer.prior)
    trainer.prior_optimizer_steps = 0
    trainer._prior_calibration_backup = {
        "prior": weight_before, "optimizer": optimizer_before,
        "optimizer_steps": 0, "prior_sha256": hash_before,
    }
    trainer._prior_calibration_status = "in_progress"
    trainer._prior_calibration_rolled_back_updates = 0
    trainer._force_prior_frozen = False
    trainer._prior_calibration_validation = None
    with torch.no_grad():
        for parameter in trainer.prior.parameters():
            parameter.add_(1.0)
    trainer.prior_optimizer_steps = 1

    trainer.set_validation_result(passed=False, metrics={"real_task_safety_pass": False})
    assert module_sha256(trainer.prior) == hash_before
    restored_optimizer = trainer.prior_optimizer.state_dict()
    assert restored_optimizer["param_groups"] == optimizer_before["param_groups"]
    assert restored_optimizer["state"].keys() == optimizer_before["state"].keys()
    for parameter_id, expected_state in optimizer_before["state"].items():
        for key, expected_value in expected_state.items():
            current_value = restored_optimizer["state"][parameter_id][key]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(current_value, expected_value, rtol=0, atol=0)
            else:
                assert current_value == expected_value
    assert trainer.prior_optimizer_steps == 1  # retain actual optimizer operation count
    assert trainer._prior_calibration_rolled_back_updates == 1
    assert trainer._prior_calibration_status == "rejected_rollback"
    assert trainer._force_prior_frozen
