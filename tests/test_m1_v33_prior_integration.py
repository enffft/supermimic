from __future__ import annotations

import copy
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


NEW_PRIOR = Path(
    "/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/"
    "cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt"
)
OLD_PRIOR = Path(
    "/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/"
    "cvae_v1_strict_fcl_2272/checkpoint_best_val.pt"
)
NEW_SHA256 = "05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c"


@pytest.mark.skipif(not NEW_PRIOR.is_file(), reason="local V3.3 prior artifact is unavailable")
def test_v33_prior_metadata_normalization_and_real_q6_qd6_inputs():
    from supermimic.motion_prior import load_m1_motion_prior
    from supermimic.motion_prior.features import state_history_to_cvae53

    prior = load_m1_motion_prior(NEW_PRIOR, device="cpu", expected_sha256=NEW_SHA256)
    assert prior.source["feature_version"] == "m1_cvae_motion_prior_v1_state53_goal8_future31"
    assert prior.source["condition_contract"] == "History[4,53]+Goal[8]; L=Velocity, M/LM=Reach; no future condition"
    assert prior.source["target_contract"] == "Future[10,31], 0.2 seconds at 0.02 seconds/frame"
    assert prior.source["model_config"] == {"latent_dim": 32, "hidden_dim": 256, "layers": 3}
    assert prior.source["dataset_hash"] == "95f9f0a39b05d5a0d736532d24f897b3b85b8cbf8f74977de620acc9b1a7146e"
    assert prior.source["split_hash"] == "b1bb36beb45b141a77f28837df376690abc3e1d6b24dd810d5ee07da40fd2624"
    assert prior.source["normalization_hash"] == "0dea5b4cac9a7297ad27e1e4d4192211a6f7af7cd138c73046b2ed7c80dd904f"
    assert prior.source["assurance_mode"] == "mixed_kinematic_and_sampled_fcl"
    assert prior.source["strict_fcl_only"] is False
    assert prior.source["step"] == 9600
    assert prior.input_compatibility == "legacy_v1"
    assert torch.all(prior.x_std[[30, 83, 136, 189]] > 0.36)
    assert torch.all(prior.x_std[[48, 101, 154, 207]] > 0.26)

    state = torch.zeros(1, 4, 57)
    state[..., 3] = 1.0
    state[:, :, 28] = torch.tensor((0.0, 0.1, 0.2, 0.3))
    state[:, :, 56] = torch.tensor((0.0, 0.05, 0.10, 0.15))
    goal = torch.tensor([[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    features = state_history_to_cvae53(state)
    assert torch.allclose(features[0, :, 30], state[0, :, 28])
    assert torch.allclose(features[0, :, 48], state[0, :, 56])
    with torch.inference_mode():
        output = prior.forward_features(features, goal)
        diag = prior.input_diagnostics(state, goal)
    assert output.shape == (1, 10, 31) and torch.isfinite(output).all()
    assert float(diag["prior/input/compat_projection_count"]) == 0.0
    assert float(diag["prior/input/q6_normalized"]) > 0.0
    assert float(diag["prior/input/qd6_normalized"]) > 0.0
    with pytest.raises(ValueError, match="tiny-std q6/qd6"):
        type(prior)(
            prior.x_mean, prior.x_std, prior.y_mean, prior.y_std,
            latent_dim=32, hidden_dim=256, layers=3,
            input_compatibility="train_constant_compat_v1",
        )


@pytest.mark.skipif(not NEW_PRIOR.is_file(), reason="local V3.3 prior artifact is unavailable")
def test_amp_and_supermimic_prior_mean_match_on_identical_state53_and_goal8():
    amp_root = Path("/home/yifeiy/AMP-LOCO-M1")
    if not (amp_root / "motion_prior/cvae_inference.py").is_file():
        pytest.skip("AMP original inference module is unavailable")
    sys.path.insert(0, str(amp_root))
    try:
        amp = importlib.import_module("motion_prior.cvae_inference")
        from supermimic.motion_prior import load_m1_motion_prior

        supermimic_prior = load_m1_motion_prior(
            NEW_PRIOR, device="cpu", expected_sha256=NEW_SHA256, input_compatibility="legacy_v1",
        )
        amp_prior = amp.load_cvae_checkpoint(NEW_PRIOR, device="cpu")
        history = torch.linspace(-0.25, 0.35, 4 * 53).reshape(1, 4, 53)
        goal = torch.tensor([[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        with torch.inference_mode():
            expected = amp_prior.predict(history, goal)
            actual = supermimic_prior.forward_features(history, goal)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    finally:
        sys.path.remove(str(amp_root))
        sys.modules.pop("motion_prior.cvae_inference", None)
        sys.modules.pop("motion_prior.cvae_model", None)


@pytest.mark.skipif(not OLD_PRIOR.is_file(), reason="legacy CVAE baseline artifact is unavailable")
def test_legacy_tiny_std_checkpoint_keeps_train_constant_reproducibility():
    from supermimic.motion_prior import load_m1_motion_prior

    prior = load_m1_motion_prior(OLD_PRIOR, device="cpu", input_compatibility="train_constant_compat_v1")
    legacy = copy.deepcopy(prior)
    legacy.input_compatibility = "legacy_v1"
    features = torch.randn(2, 4, 53)
    goal = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).expand(2, -1).clone()
    projected = features.clone()
    for frame in range(4):
        projected[:, frame, 30] = prior.x_mean[frame * 53 + 30]
        projected[:, frame, 48] = prior.x_mean[frame * 53 + 48]
    with torch.inference_mode():
        actual = prior.forward_features(features, goal)
        equivalent = legacy.forward_features(projected, goal)
    torch.testing.assert_close(actual, equivalent, rtol=1e-6, atol=1e-6)


def test_differentiable_rollout_separates_policy_observation_and_executed_wm_action(monkeypatch):
    from types import SimpleNamespace
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.motion_prior import rollout

    from supermimic.robots.spec import get_robot_spec

    seen_actions = []
    seen_actor_references = []
    seen_wm_actions = []

    def fake_observation(state, action, xi, default_joint_pos, wheel_indices):
        seen_actions.append(action.clone())
        return state.new_zeros((state.shape[0], 72))

    class Actor:
        cfg = SimpleNamespace(history=1)

        def __call__(self, proprio, goal, reference_action):
            seen_actor_references.append(reference_action.clone())
            return torch.full((len(proprio), 22), 1.2, device=proprio.device)

    class WM:
        spec = SimpleNamespace(wheel_indices=(3, 7, 11, 15), nj=22)

        def step(self, state, action, xi):
            seen_wm_actions.append(action.clone())
            return state

    monkeypatch.setattr(rollout, "analytic_observation", fake_observation)
    state = torch.zeros(1, 57)
    state[:, 3] = 1.0
    start = StartBatch(
        state=state, xi=torch.zeros(1, 26), frame=torch.zeros(1, dtype=torch.long),
        prev_action=torch.zeros(1, 22), hist_states=state[None],
        hist_actions=torch.zeros(1, 1, 22), task_type=torch.zeros(1, dtype=torch.long),
        velocity_command=torch.zeros(1, 3), fixed_world_target=torch.zeros(1, 3),
    )
    result = rollout.differentiable_m1_rollout(
        WM(), Actor(), start, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        torch.zeros(1, 10, 31), torch.zeros(22),
        robot_spec=get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3"),
        horizon=1,
    )
    assert len(seen_actions) == 2 and len(seen_wm_actions) == 2
    # Initial history carries a zero action. The next proprio observation
    # carries raw policy output, while the World Model saw its clipped value.
    torch.testing.assert_close(seen_actions[0], torch.zeros_like(seen_actions[0]))
    torch.testing.assert_close(seen_actions[-1], torch.full_like(seen_actions[-1], 1.2))
    torch.testing.assert_close(seen_wm_actions[0], torch.ones_like(seen_wm_actions[0]))
    torch.testing.assert_close(seen_wm_actions[1], result.reference_actions[:, 0])
    torch.testing.assert_close(seen_actor_references[0], result.reference_actions[:, 0])
    assert result.actor_states.shape == result.prior_direct_states.shape == (2, 1, 57)


def test_reference_inverse_action_round_trips_the_v3_setpoint_mapping():
    from supermimic.robots.spec import get_robot_spec
    from supermimic.tasks.action_contract import inverse_m1_setpoint_action, mixed_m1_setpoints

    spec = get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    default = torch.tensor(spec.default_joint_pos)
    scales = torch.tensor(spec.action_scales)
    wheel_mask = torch.tensor([name.endswith("FOOT_JOINT") for name in spec.joint_names])
    bounds = torch.tensor(spec.position_target_bounds)
    soft_limits = torch.tensor([[-4.0, 4.0]] * spec.num_joints)
    target = default.unsqueeze(0).repeat(3, 1)
    rate = torch.zeros(3, 4)
    for joint_id, name in enumerate(spec.joint_names):
        if name.endswith("FOOT_JOINT"):
            continue
        target[0, joint_id] = bounds[joint_id, 0]
        target[2, joint_id] = bounds[joint_id, 1]
    rate[0] = -5.0
    rate[2] = 5.0
    action = inverse_m1_setpoint_action(target, rate, robot_spec=spec)
    mapped = mixed_m1_setpoints(action, default, scales, wheel_mask, soft_limits, bounds)
    for row in range(3):
        position_mask = ~wheel_mask
        torch.testing.assert_close(mapped[row, position_mask], target[row, position_mask], rtol=0, atol=1e-6)
        torch.testing.assert_close(mapped[row, wheel_mask], rate[row], rtol=0, atol=1e-6)


def test_fixed_evaluation_defaults_to_saved_task_and_rejects_mismatch():
    import importlib.util

    script = Path(__file__).parents[1] / "scripts/evaluate_m1_prior_joint.py"
    spec = importlib.util.spec_from_file_location("evaluate_m1_prior_joint", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _resolve_saved_task_type = module._resolve_saved_task_type

    checkpoint = {
        "task_sampler_state": {"task_type": "L"},
        "config": {"m1_joint_training": {"task_type": "M"}},
    }
    assert _resolve_saved_task_type(checkpoint) == "L"
    assert _resolve_saved_task_type(checkpoint, "L") == "L"
    with pytest.raises(ValueError, match="differs from saved checkpoint task"):
        _resolve_saved_task_type(checkpoint, "M")
    assert _resolve_saved_task_type({"config": {"m1_joint_training": {"task_type": "M"}}}) == "M"


def _make_checkpoint_trainer(*, prior, robot_spec=None, cfg=None):
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.motion_prior.trainer import M1AlternatingTrainer
    from supermimic.policy.nets import M1ActionReferenceActorCfg, M1ActionReferenceActor
    from supermimic.robots.spec import get_robot_spec

    spec = robot_spec or get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    cfg = cfg or SimpleNamespace(
        prior_horizon=10, actor_horizon=10, replanning_cadence=10,
        state_source="privileged", prior_mode="prior_mean", training_mode="staged_action22_v1",
        freeze_prior=True, prior_updates_per_event=1, actor_updates_per_event=1,
        prior_gate_evidence_path=None,
        exploration_enabled=True, exploration_env_fraction=0.25,
        exploration_leg_std=0.02, exploration_arm_std=0.015,
        exploration_wheel_std=0.05, exploration_decay_end_fraction=0.75,
        batch_size=4, micro_batch_size=2, prior_lr=1e-5, actor_lr=2e-4,
        grad_clip_norm=1.0, loss=__import__("supermimic.motion_prior.losses", fromlist=["M1JointLossCfg"]).M1JointLossCfg(),
        readiness_window_steps=8, readiness_min_samples=4, readiness_min_survival=0.9,
        readiness_max_root_tracking_error=0.2, readiness_max_arm_tracking_error=0.25,
        readiness_max_wheel_tracking_error=0.75, readiness_max_action_saturation=0.1,
        actor_pretrain_updates=0,
    )
    wm = torch.nn.Linear(2, 2)
    wm_trainer = SimpleNamespace(
        opt=torch.optim.Adam(wm.parameters(), lr=1e-3),
        cfg=SimpleNamespace(batches_per_update=1, window=8),
    )
    replay = ReplayBuffer(2, 16, 57, 22, 1, "cpu", goal_dim=8, m1_joint_context=True)

    class Command:
        def state_dict(self):
            return {"task_type": "L", "vx_range": (-0.09, 0.145), "vx_stop_fraction": 0.15,
                    "vx_reverse_fraction": 0.35, "vy_range": (0.0, 0.0), "wz_range": (0.0, 0.0)}

    actor = M1ActionReferenceActor(M1ActionReferenceActorCfg(history=5))
    return M1AlternatingTrainer(
        wm=wm, wm_trainer=wm_trainer, prior=prior, actor=actor, buffer=replay,
        default_joint_pos=torch.tensor(spec.default_joint_pos),
        soft_limits=torch.tensor([[-3.0, 3.0]] * 22), fk=object(), tcp_index=0,
        cfg=cfg, robot_joint_names=spec.joint_names, robot_model_sha256=None,
        command=Command(), robot_spec=spec,
    )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.skipif(not NEW_PRIOR.is_file(), reason="local V3.3 prior artifact is unavailable")
def test_v3_trainer_save_load_restores_weights_optimizer_counters_and_contract(tmp_path):
    from supermimic.motion_prior import load_m1_motion_prior

    prior_a = load_m1_motion_prior(NEW_PRIOR, device="cpu", expected_sha256=NEW_SHA256)
    first = _make_checkpoint_trainer(prior=prior_a)
    for module, optimizer in (
        (first.wm, first.wm_trainer.opt), (first.prior, first.prior_optimizer),
        (first.actor, first.actor_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in module.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    first.wm_update_events = 3
    first.wm_optimizer_steps = 3
    first.prior_optimizer_steps = 1
    first.actor_optimizer_steps = 2
    first.stage = "actor_prior_frozen"
    config = {"action_scales": first.robot_spec.action_scales}
    saved = first.checkpoint(
        iteration=51, config_snapshot=config, command_state=first.command.state_dict(),
    )
    path = tmp_path / "action22_v4.pt"
    torch.save(saved, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)

    prior_b = load_m1_motion_prior(NEW_PRIOR, device="cpu", expected_sha256=NEW_SHA256)
    second = _make_checkpoint_trainer(prior=prior_b)
    iteration = second.load_checkpoint(restored, expected_source=prior_b.source)
    assert iteration == 51
    assert second.wm_update_events == 3
    assert second.wm_optimizer_steps == 3
    assert second.prior_optimizer_steps == 1
    assert second.actor_optimizer_steps == 2
    assert second.stage == "actor_prior_frozen"
    for key, module in (("wm", second.wm), ("motion_prior", second.prior), ("actor", second.actor)):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, restored[key][name], rtol=0, atol=0)
    _assert_nested_equal(second.wm_trainer.opt.state_dict(), restored["wm_optimizer"])
    _assert_nested_equal(second.prior_optimizer.state_dict(), restored["prior_optimizer"])
    _assert_nested_equal(second.actor_optimizer.state_dict(), restored["actor_optimizer"])
    assert restored["action_mapping"]["wheel_mapping"] == "clip(action,-1,1)*wheel_scale_rad_s"

    if torch.cuda.is_available():
        # The production M1 loader deserializes the full Checkpoint onto the
        # training device. That remaps the saved CPU RNG state to CUDA too.
        mapped_to_cuda = torch.load(path, map_location="cuda", weights_only=False)
        prior_c = load_m1_motion_prior(NEW_PRIOR, device="cpu", expected_sha256=NEW_SHA256)
        third = _make_checkpoint_trainer(prior=prior_c)
        assert third.load_checkpoint(mapped_to_cuda, expected_source=prior_c.source) == 51

    with pytest.raises(ValueError, match="Action22-reference checkpoint v4"):
        second.load_checkpoint({**restored, "format": "supermimic_m1_prior_actor_joint_v2", "format_version": 2}, expected_source=prior_b.source)

    changed_source = copy.deepcopy(restored)
    changed_source["prior_source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Motion Prior provenance mismatch"):
        second.load_checkpoint(changed_source, expected_source=prior_b.source)

    changed_wm_hash = copy.deepcopy(restored)
    changed_wm_hash["world_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="World Model state/hash mismatch"):
        second.load_checkpoint(changed_wm_hash, expected_source=prior_b.source)

    from supermimic.robots.spec import get_robot_spec
    carry = get_robot_spec("m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v3")
    wrong_pose = _make_checkpoint_trainer(prior=prior_b, robot_spec=carry)
    with pytest.raises(ValueError, match="Action22 checkpoint action_mapping mismatch"):
        wrong_pose.load_checkpoint(restored, expected_source=prior_b.source)

    bounds = list(second.robot_spec.position_target_bounds)
    bounds[second.robot_spec.joint_names.index("piper_x_joint1")] = (-0.45, 0.51)
    wrong_bounds = _make_checkpoint_trainer(
        prior=prior_b, robot_spec=replace(second.robot_spec, position_target_bounds=tuple(bounds)),
    )
    with pytest.raises(ValueError, match="Action22 checkpoint action_mapping mismatch"):
        wrong_bounds.load_checkpoint(restored, expected_source=prior_b.source)

    second.prior.input_compatibility = "train_constant_compat_v1"
    with pytest.raises(ValueError, match="Action22 checkpoint prior_input_compatibility mismatch"):
        second.load_checkpoint(restored, expected_source=prior_b.source)
