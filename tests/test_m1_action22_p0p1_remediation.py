from __future__ import annotations

import torch
import pytest


class _RootTcp:
    def __call__(self, qpos, qvel=None):
        del qvel
        return {"pos": qpos[:, None, :3]}


def _trajectory(xs: list[float], *, tilt: float = 0.0) -> torch.Tensor:
    states = torch.zeros(len(xs), 1, 57)
    states[:, 0, 2] = 0.47
    states[:, 0, 0] = torch.tensor(xs)
    states[:, 0, 3] = torch.cos(torch.tensor(tilt / 2))
    states[:, 0, 4] = torch.sin(torch.tensor(tilt / 2))
    return states


def _actor_terms(states: torch.Tensor, *, task: int, target: tuple[float, float, float], survival=None, valid=None):
    from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses

    initial = states[0]
    future = states[1:]
    actions = torch.zeros(future.shape[0], future.shape[1], 22, dtype=states.dtype, device=states.device)
    return actor_losses(
        future, future.clone(), actions, torch.zeros_like(actions[0]),
        soft_limits=torch.tensor([[-3.0, 3.0]] * 22, dtype=states.dtype, device=states.device),
        cfg=M1JointLossCfg(), task_type=torch.full((future.shape[1],), task, device=states.device),
        velocity_command=torch.zeros(future.shape[1], 3, device=states.device),
        fixed_world_target=torch.tensor([target], dtype=states.dtype, device=states.device),
        fk=_RootTcp(), reference_actions=torch.zeros_like(actions), initial_state=initial,
        valid_mask=valid, safety_mask=valid, survival_probability=survival,
    )


def test_m_task_process_terminal_braking_and_stable_unmet_goal():
    # Safe approach followed by a settled endpoint.
    settled = _trajectory([0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.20, 0.20, 0.20, 0.20])
    # Same near terminal TCP distance, but a final high-speed pass through it.
    pass_through = _trajectory([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18, 0.20])
    settled_terms = _actor_terms(settled, task=1, target=(0.20, 0.0, 0.47))
    pass_terms = _actor_terms(pass_through, task=1, target=(0.20, 0.0, 0.47))
    assert settled_terms["terminal_error_m_task"] < 1e-6
    assert pass_terms["terminal_error_m_task"] < 1e-6
    assert pass_terms["m_task_near_goal_speed_task"] > settled_terms["m_task_near_goal_speed_task"]
    assert pass_terms["task"] > settled_terms["task"]
    for key in ("m_tcp_distance_mean_m_task", "m_tcp_speed_mean_m_s_task", "m_task_process_task", "m_task_terminal_task"):
        assert torch.isfinite(settled_terms[key])

    # A nearby but moving endpoint costs more than the same endpoint at rest.
    moving_near = _trajectory([0.10] * 9 + [0.18, 0.19])
    stopped_near = _trajectory([0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.17, 0.18, 0.19, 0.19, 0.19])
    moving_terms = _actor_terms(moving_near, task=1, target=(0.20, 0.0, 0.47))
    stopped_terms = _actor_terms(stopped_near, task=1, target=(0.20, 0.0, 0.47))
    assert moving_terms["m_task_near_goal_speed_task"] > stopped_terms["m_task_near_goal_speed_task"]

    # Failure to reach remains a finite positive task cost without inventing a fall.
    stable_unmet = _trajectory([0.0] * 11)
    unmet_terms = _actor_terms(stable_unmet, task=1, target=(0.30, 0.0, 0.47))
    assert unmet_terms["task_M"] > 0
    assert unmet_terms["stability"] == 0
    assert torch.isfinite(unmet_terms["total"])


def test_m_task_penalizes_transient_tcp_progress_with_unstable_base():
    stable = _trajectory([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    tilted = _trajectory([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20], tilt=1.1)
    stable_terms = _actor_terms(stable, task=1, target=(0.20, 0.0, 0.47))
    tilted_terms = _actor_terms(tilted, task=1, target=(0.20, 0.0, 0.47))
    assert tilted_terms["stability"] > stable_terms["stability"]
    assert tilted_terms["total"] > stable_terms["total"]


def test_reference_smoothness_penalizes_current_to_first_future_velocity_jumps():
    from supermimic.motion_prior.losses import M1JointLossCfg, _reference_smoothness_terms
    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, get_robot_spec

    robot = get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3")
    initial = torch.zeros(1, 57)
    initial[:, 2] = 0.47
    initial[:, 3] = 1.0
    initial[:, 7:29] = torch.tensor(robot.default_joint_pos)
    # A static Future31 would jump from each of these nonzero measured rates
    # back to zero at the first planned step.
    initial[0, 29] = 1.0
    initial[0, 34] = 1.0
    initial[0, 35 + M1_JOINTS.index("piper_x_joint1")] = 2.0
    initial[0, 35 + M1_JOINTS.index("FAR_FOOT_JOINT")] = 10.0
    future = torch.zeros(1, 10, 31)
    future[..., 3] = 1.0
    future[..., 7] = 1.0
    for i, name in enumerate(M1_AMP_JOINTS):
        future[..., 9 + i] = initial[:, 7 + M1_JOINTS.index(name)].unsqueeze(1)
    terms = _reference_smoothness_terms(initial, future, M1JointLossCfg())
    assert terms["smooth_root_acceleration"] > 0
    assert terms["smooth_root_angular_acceleration"] > 0
    assert terms["smooth_joint_acceleration"] > 0
    assert terms["smooth_wheel_acceleration"] > 0
    assert all(torch.isfinite(value) for value in terms.values())


def test_fixed_horizon_failure_cost_ranks_survival_then_late_then_immediate():
    horizon = 10
    stationary = _trajectory([0.0] * (horizon + 1))
    full = torch.ones(horizon, 1)
    late = torch.tensor([1.0] * 3 + [0.0] * (horizon - 3)).unsqueeze(-1)
    immediate = torch.zeros(horizon, 1)
    valid_full = torch.ones_like(full, dtype=torch.bool)
    valid_late = late.bool()
    valid_immediate = immediate.bool()

    safe_m = _actor_terms(stationary, task=1, target=(0.30, 0.0, 0.47), survival=full, valid=valid_full)
    late_m = _actor_terms(stationary, task=1, target=(0.30, 0.0, 0.47), survival=late, valid=valid_late)
    immediate_m = _actor_terms(stationary, task=1, target=(0.30, 0.0, 0.47), survival=immediate, valid=valid_immediate)
    assert safe_m["total"] < late_m["total"] < immediate_m["total"]
    assert immediate_m["task"] > 0  # hard termination cannot erase the M target cost

    # Re-run the original 0.30 m target example: the old 3.3 -> 1.0 shortcut is gone.
    old_safe = _actor_terms(stationary, task=1, target=(0.30, 0.0, 0.47), survival=full, valid=valid_full)
    old_fail = _actor_terms(stationary, task=1, target=(0.30, 0.0, 0.47), survival=immediate, valid=valid_immediate)
    assert old_fail["total"] > old_safe["total"]

    # A near-goal failure also costs more than a safe, stopped near-goal state.
    near = _trajectory([0.0] * horizon + [0.0])
    near_safe = _actor_terms(near, task=1, target=(0.04, 0.0, 0.47), survival=full, valid=valid_full)
    near_fail = _actor_terms(near, task=1, target=(0.04, 0.0, 0.47), survival=immediate, valid=valid_immediate)
    assert near_safe["total"] < near_fail["total"]

    # L's large velocity error cannot be removed by the same hard mask.
    from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses
    states = stationary[1:].clone()
    states[..., 29] = -1.0
    l_fail = actor_losses(
        states, states.clone(), torch.zeros(horizon, 1, 22), torch.zeros(1, 22),
        soft_limits=torch.tensor([[-3.0, 3.0]] * 22), cfg=M1JointLossCfg(),
        task_type=torch.tensor([0]), velocity_command=torch.tensor([[0.3, 0.0, 0.0]]),
        fixed_world_target=torch.zeros(1, 3), fk=_RootTcp(), reference_actions=torch.zeros(horizon, 1, 22),
        initial_state=stationary[0], valid_mask=valid_immediate, safety_mask=valid_immediate,
        survival_probability=immediate,
    )
    assert l_fail["task_L"] > 0 and l_fail["failure_loss"] > 0


def test_failure_probability_extremes_keep_finite_action_gradients():
    from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses

    horizon = 10
    action = torch.zeros(horizon, 1, 22, requires_grad=True)
    future = torch.zeros(horizon, 1, 57)
    future[..., 2] = 0.47
    future[..., 3] = 1.0
    future[..., 29] = action[..., 0]
    initial = future[0].detach().clone()
    reference = future.detach().clone()
    logits = torch.full((horizon, 1), -12.0, requires_grad=True)
    survival = torch.cumprod(1.0 - logits.sigmoid(), dim=0)
    terms = actor_losses(
        future, reference, action, torch.zeros(1, 22), soft_limits=torch.tensor([[-3.0, 3.0]] * 22),
        cfg=M1JointLossCfg(), task_type=torch.tensor([0]),
        velocity_command=torch.tensor([[0.12, 0.0, 0.0]]), fixed_world_target=torch.zeros(1, 3),
        fk=_RootTcp(), reference_actions=torch.zeros_like(action), initial_state=initial,
        valid_mask=torch.ones(horizon, 1, dtype=torch.bool), safety_mask=torch.ones(horizon, 1, dtype=torch.bool),
        survival_probability=survival,
    )
    action_grad, = torch.autograd.grad(terms["task_L"], action, retain_graph=True)
    logit_grad, = torch.autograd.grad(terms["total"], logits)
    assert action_grad.abs().sum() > 0 and torch.isfinite(action_grad).all()
    assert logit_grad.abs().sum() > 0 and torch.isfinite(logit_grad).all()

    near_one_logits = torch.full((horizon, 1), 12.0, requires_grad=True)
    near_one_survival = torch.cumprod(1.0 - near_one_logits.sigmoid(), dim=0)
    high_failure = actor_losses(
        future, reference, action, torch.zeros(1, 22), soft_limits=torch.tensor([[-3.0, 3.0]] * 22),
        cfg=M1JointLossCfg(), task_type=torch.tensor([0]),
        velocity_command=torch.tensor([[0.12, 0.0, 0.0]]), fixed_world_target=torch.zeros(1, 3),
        fk=_RootTcp(), reference_actions=torch.zeros_like(action), initial_state=initial,
        valid_mask=torch.zeros(horizon, 1, dtype=torch.bool), safety_mask=torch.zeros(horizon, 1, dtype=torch.bool),
        survival_probability=near_one_survival,
    )
    assert torch.isfinite(high_failure["total"])
    assert high_failure["task_L"] > 0 and high_failure["failure_loss"] > 0


def test_bounded_actor_residual_groups_and_alpha_endpoints():
    import importlib.util
    from pathlib import Path

    from supermimic.policy.nets import M1ActionReferenceActor, M1ActionReferenceActorCfg
    from supermimic.robots.spec import M1_JOINTS, get_robot_spec
    from supermimic.tasks.action_contract import mixed_m1_setpoints

    spec = importlib.util.spec_from_file_location(
        "evaluate_m1_prior_joint", Path(__file__).resolve().parents[1] / "scripts/evaluate_m1_prior_joint.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _blend_actor_residual = module._blend_actor_residual

    cfg = M1ActionReferenceActorCfg(history=2, hidden=(16, 8))
    actor = M1ActionReferenceActor(cfg)
    proprio = torch.zeros(1, 2 * 72)
    goal = torch.zeros(1, 8)
    reference = torch.zeros(1, 22)
    with torch.no_grad():
        actor.residual[-1].bias.fill_(20.0)
    output = actor(proprio, goal, reference)
    robot_spec = get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    default = torch.tensor(robot_spec.default_joint_pos)
    scales = torch.tensor(robot_spec.action_scales)
    wheel_mask = torch.tensor([name in robot_spec.wheel_joint_names for name in M1_JOINTS])
    bounds = torch.tensor(robot_spec.position_target_bounds)
    soft_limits = torch.tensor([[-10.0, 10.0]] * 22)
    physical_residual = mixed_m1_setpoints(
        output, default, scales, wheel_mask, soft_limits, bounds,
    ) - mixed_m1_setpoints(
        torch.zeros_like(output), default, scales, wheel_mask, soft_limits, bounds,
    )
    for index, name in enumerate(M1_JOINTS):
        limit = cfg.wheel_delta_max_normalized if name.endswith("FOOT_JOINT") else (
            cfg.arm_delta_max_normalized if name.startswith("piper_x_joint") else cfg.leg_delta_max_normalized
        )
        assert abs(float(output[0, index].detach())) <= limit + 1e-6
        if name.endswith("FOOT_JOINT"):
            physical_bound = limit * robot_spec.action_scales[index]
        else:
            default_q = robot_spec.default_joint_pos[index]
            lower, upper = robot_spec.position_target_bounds[index]
            physical_bound = limit * max(default_q - lower, upper - default_q)
        assert abs(float(physical_residual[0, index].detach())) <= physical_bound + 1e-6
    actor_action = torch.full_like(reference, 0.7)
    torch.testing.assert_close(_blend_actor_residual(reference, actor_action, 0.0), reference)
    torch.testing.assert_close(_blend_actor_residual(reference, actor_action, 1.0), actor_action)


def test_wm_finite_difference_audit_uses_paired_directional_responses():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "audit_m1_wm_finite_difference",
        Path(__file__).resolve().parents[1] / "scripts/audit_m1_wm_finite_difference.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    real = torch.tensor([[0.02, 0.0], [-0.02, 0.0]])
    aligned = torch.tensor([[0.01, 0.0], [-0.01, 0.0]])
    good = module._paired_response(real, aligned, 0.02, 1e-5)
    bad = module._paired_response(real, aligned.flip(0), 0.02, 1e-5)
    quiet = module._paired_response(real * 1e-6, aligned * 1e-6, 0.02, 1e-5)
    assert good["direction_cosine"] > 0.99
    assert good["relative_response_error"] == pytest.approx(0.5)
    assert bad["direction_cosine"] < -0.99
    assert quiet["low_signal"]

    task_good = module._task_response([1.0, 2.0], [0.5, 1.5], 0.02)
    task_bad = module._task_response([1.0, 2.0], [1.5, 0.5], 0.02)
    assert task_good["direction_agreement"] is True
    assert task_bad["direction_agreement"] is False


def test_task_finite_difference_loss_is_computed_per_branch():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "audit_m1_wm_finite_difference_task_rows",
        Path(__file__).resolve().parents[1] / "scripts/audit_m1_wm_finite_difference.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    initial = torch.zeros(2, 57)
    initial[:, 2] = 0.47
    initial[:, 3] = 1.0
    l_states = initial.clone()
    l_states[0, 29] = 0.06
    l_states[1, 29] = -0.06
    command = torch.tensor([[0.12, 0.0, 0.0]] * 2)
    target = torch.zeros(2, 3)
    l_values = module._task_losses(initial, l_states, 0, command, target, _RootTcp(), 0)
    assert l_values.shape == (2,)
    assert l_values[0] != l_values[1]

    m_states = initial.clone()
    m_states[0, 0] = 0.04
    m_states[1, 0] = -0.04
    m_target = torch.tensor([[0.30, 0.0, 0.47]] * 2)
    m_values = module._task_losses(initial, m_states, 1, torch.zeros_like(command), m_target, _RootTcp(), 0)
    assert m_values.shape == (2,)
    assert m_values[0] < m_values[1]


def test_actor_gradient_diagnostics_are_weighted_and_do_not_write_optimizer_grads():
    from supermimic.motion_prior.trainer import actor_loss_gradient_diagnostics

    parameter = torch.nn.Parameter(torch.tensor(0.4))
    actions = (parameter * torch.arange(1.0, 5.0).reshape(4, 1, 1)).expand(-1, -1, 22)
    task_l = actions.square().mean()
    task_m = (actions - 0.2).square().mean()
    tracking = (actions - 0.1).square().mean()
    bc = actions.square().mean()
    safety = actions.abs().mean()
    failure = actions.sum() * 0.01
    expected_bc_grad, = torch.autograd.grad(0.05 * bc, actions, retain_graph=True)
    metrics = actor_loss_gradient_diagnostics({
        "task": task_l + task_m,
        "task_L": task_l,
        "task_M": task_m,
        "task_L_weighted": 0.5 * task_l,
        "task_M_weighted": 0.5 * task_m,
        "tracking": tracking,
        "bc": bc,
        "bc_weighted": 0.05 * bc,
        "safety": safety,
        "safety_weighted": safety,
        "failure_loss": failure,
        "failure_weighted": failure,
    }, actions, [parameter])
    assert parameter.grad is None
    assert metrics["grad_norm/task_L"] > 0
    assert metrics["grad_norm/task_M"] > 0
    assert metrics["action_grad_norm/BC"] == pytest.approx(float(expected_bc_grad.norm()))
    assert torch.isfinite(torch.tensor(metrics["gradient_cosine_parameter/L_vs_M"]))


def test_real_checkpoint_selector_keeps_full_optimizer_state_on_rollback(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "select_m1_action22_best_real_checkpoint",
        Path(__file__).resolve().parents[1] / "scripts/select_m1_action22_best_real_checkpoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    accepted_ckpt = tmp_path / "accepted.pt"
    candidate_ckpt = tmp_path / "candidate.pt"
    for path, step in ((accepted_ckpt, 7), (candidate_ckpt, 19)):
        torch.save({
            "format": module.EXPECTED_FORMAT, "format_version": module.EXPECTED_VERSION,
            "actor": {"weight": torch.tensor([step])}, "actor_optimizer": {"step": step},
            "wm": {}, "wm_optimizer": {}, "motion_prior": {}, "prior_optimizer": {},
            "robot_identity": "m1_piper", "robot_model_sha256": "robot-hash",
            "action_mapping": {"contract_version": "v3"}, "prior_schema_version": "v3.3",
            "prior_source": {"sha256": "prior-hash"}, "prior_mode": "prior_mean",
            "prior_input_compatibility": "legacy_v1", "actor_schema": {"version": "v2"},
            "actor_input_contract": "state-history-goal-action22-v1",
            "reset_contract": {"profile": "high-carry"},
            "reference_adapter_schema": "future31_to_interleaved_action22_v1",
        }, path)

    contract = {
        "task_type": "L", "controller": "actor", "actor_residual_alpha": 1.0,
        "seed": 42, "num_envs": 8, "steps": 100,
        "domain_randomization": False, "obs_noise": False, "action_noise": 0.0,
        "initial_condition_sha256": "same-state", "fixed_group_names": ["backward_id", "forward_id", "stop_id"],
    }
    def write_eval(path, checkpoint, *, rmse, falls):
        groups = {
            "stop_id": {"base_vx_rmse_m_s": rmse, "positive_command_negative_vx_fraction": 0.0,
                        "signed_forward_displacement_mean_m": 0.01, "fall_rate": falls,
                        "episode_count": 2, "base_tilt_mean_rad": 0.1, "base_height_mean_m": 0.47},
            "forward_id": {"base_vx_rmse_m_s": rmse, "positive_command_negative_vx_fraction": 0.0,
                           "signed_forward_displacement_mean_m": 0.2, "fall_rate": falls,
                           "episode_count": 2, "base_tilt_mean_rad": 0.1, "base_height_mean_m": 0.47},
            "backward_id": {"base_vx_rmse_m_s": rmse, "positive_command_negative_vx_fraction": 0.0,
                            "signed_forward_displacement_mean_m": -0.1, "fall_rate": falls,
                            "episode_count": 2, "base_tilt_mean_rad": 0.1, "base_height_mean_m": 0.47},
        }
        path.write_text(json.dumps({"checkpoint": str(checkpoint), "evaluation_contract": contract,
                                    "groups": groups}))

    accepted_eval = tmp_path / "accepted.json"
    candidate_eval = tmp_path / "candidate.json"
    write_eval(accepted_eval, accepted_ckpt, rmse=0.2, falls=0.0)
    write_eval(candidate_eval, candidate_ckpt, rmse=0.1, falls=0.5)
    result = module.select_best_real(
        task="L", accepted_checkpoint=accepted_ckpt, accepted_evaluation=accepted_eval,
        candidate_checkpoint=candidate_ckpt, candidate_evaluation=candidate_eval,
        output_dir=tmp_path / "selection",
    )
    assert result["status"] == "rejected_rollback"
    restored = torch.load(result["selected_resume_checkpoint"], map_location="cpu", weights_only=False)
    assert int(restored["actor"]["weight"][0]) == 7
    assert restored["actor_optimizer"]["step"] == 7
