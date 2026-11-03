"""CPU contract tests for Future31 -> Action22-reference M1 training."""

from types import SimpleNamespace

import pytest
import torch


def _spec():
    from supermimic.robots.spec import get_robot_spec

    return get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )


def _future(batch=1):
    from supermimic.robots.spec import M1_AMP_JOINTS

    spec = _spec()
    future = torch.zeros(batch, 10, 31)
    future[..., 3] = 1.0
    future[..., 7] = 1.0
    for i, name in enumerate(M1_AMP_JOINTS):
        future[..., 9 + i] = spec.default_joint_pos[spec.joint_names.index(name)]
    return future


def test_future31_named_mapping_round_trip_and_wheel_signs():
    from supermimic.motion_prior.reference import future31_to_m1_action22
    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS
    from supermimic.tasks.action_contract import mixed_m1_setpoints

    spec = _spec()
    future = _future(batch=2)
    for i, name in enumerate(M1_AMP_JOINTS):
        joint_id = spec.joint_names.index(name)
        lo, hi = spec.position_target_bounds[joint_id]
        default = spec.default_joint_pos[joint_id]
        future[0, :, 9 + i] = default + 0.1 * (hi - default)
        future[1, :, 9 + i] = default - 0.1 * (default - lo)
        assert lo <= future[1, 0, 9 + i] <= hi and lo <= future[0, 0, 9 + i] <= hi
    future[0, :, 27:31] = torch.tensor((-2.0, -0.5, 0.5, 2.0))
    future[1, :, 27:31] = torch.tensor((2.0, 0.5, -0.5, -2.0))
    action, raw, diagnostics = future31_to_m1_action22(
        future, robot_spec=spec, return_diagnostics=True,
    )
    assert action.shape == raw.shape == (2, 10, 22)
    assert action.dtype == future.dtype and action.device == future.device
    assert torch.isfinite(action).all() and (action.abs() <= 1).all()
    torch.testing.assert_close(action, raw, atol=1e-7, rtol=0)
    assert diagnostics["out_of_action_envelope_fraction"] == 0

    wheel_ids = [spec.joint_names.index(n) for n in spec.wheel_joint_names]
    for row in range(2):
        for wheel_slot, joint_id in enumerate(wheel_ids):
            scale = spec.action_scales[joint_id]
            torch.testing.assert_close(action[row, :, joint_id], future[row, :, 27 + wheel_slot] / scale)
    assert action[0, 0, wheel_ids[0]] < 0 and action[0, 0, wheel_ids[3]] > 0
    default = torch.tensor(spec.default_joint_pos).expand(2, 10, -1)
    scales = torch.tensor(spec.action_scales)
    wheel_mask = torch.tensor([name in spec.wheel_joint_names for name in M1_JOINTS])
    soft = torch.tensor([[-10.0, 10.0]] * 22)
    mapped = mixed_m1_setpoints(
        action, default, scales, wheel_mask, soft,
        torch.tensor(spec.position_target_bounds),
    )
    expected = default.clone()
    for i, name in enumerate(M1_AMP_JOINTS):
        expected[..., spec.joint_names.index(name)] = future[..., 9 + i]
    for i, joint_id in enumerate(wheel_ids):
        expected[..., joint_id] = future[..., 27 + i]
    torch.testing.assert_close(mapped, expected, atol=2e-6, rtol=0)


def test_out_of_envelope_action_is_bounded_but_keeps_preclip_gradient():
    from supermimic.motion_prior.reference import future31_to_m1_action22

    spec = _spec()
    future = _future()
    arm_joint = spec.joint_names.index("piper_x_joint2")
    amp_i = spec.amp_joint_names.index("piper_x_joint2")
    upper = spec.position_target_bounds[arm_joint][1]
    future[..., 9 + amp_i] = upper + 0.25
    future[..., 27] = 15.0
    future.requires_grad_(True)
    action, raw, diagnostics = future31_to_m1_action22(
        future, robot_spec=spec, return_diagnostics=True,
    )
    assert (raw.abs() > 1).any() and torch.all(action.abs() <= 1)
    assert diagnostics["position_envelope_violation_fraction"] > 0
    assert diagnostics["wheel_rate_violation_fraction"] > 0
    grad = torch.autograd.grad(diagnostics["out_of_action_envelope_penalty"], future)[0]
    assert torch.isfinite(grad).all() and grad[..., 9 + amp_i].abs().sum() > 0
    assert grad[..., 27].abs().sum() > 0


def test_one_future_joint_or_wheel_only_changes_its_named_action_slot():
    from supermimic.motion_prior.reference import future31_to_m1_action22
    from supermimic.robots.spec import M1_AMP_JOINTS

    spec = _spec()
    base = _future()
    action0 = future31_to_m1_action22(base, robot_spec=spec)
    for future_i, joint_name in ((9, M1_AMP_JOINTS[0]), (23, "piper_x_joint3"), (28, "FBL_FOOT_JOINT")):
        changed = base.clone()
        changed[..., future_i] += 0.03
        action1 = future31_to_m1_action22(changed, robot_spec=spec)
        delta = (action1 - action0).abs().amax(dim=(0, 1))
        expected = spec.joint_names.index(joint_name)
        affected = (delta > 1e-7).nonzero().flatten().tolist()
        assert affected == [expected]


def test_action_reference_actor_schema_shapes_residual_init_and_bounds():
    from supermimic.policy.nets import M1ActionReferenceActor, M1ActionReferenceActorCfg

    actor = M1ActionReferenceActor(M1ActionReferenceActorCfg(history=5, hidden=(32, 16)))
    proprio = torch.zeros(3, 5 * 72)
    goal = torch.zeros(3, 8)
    reference = torch.linspace(-0.98, 0.98, 22).repeat(3, 1)
    output = actor(proprio, goal, reference)
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=0)
    changed = reference.clone()
    changed[:, 4] += 0.1
    assert not torch.equal(actor(proprio, goal, changed), output)
    assert output.shape == (3, 22) and torch.isfinite(output).all()
    assert output.abs().max() <= 1
    with pytest.raises(ValueError, match="current_ref_action22 must have shape"):
        actor(proprio, goal, torch.zeros(3, 10, 31))
    with pytest.raises(ValueError, match="proprio_history must have shape"):
        actor(torch.zeros(3, 10 * 72), goal, reference)


def _start(history=2):
    from supermimic.data.replay_buffer import StartBatch

    spec = _spec()
    state = torch.zeros(1, 57)
    state[:, 2] = 0.47
    state[:, 3] = 1.0
    state[:, 7:29] = torch.tensor(spec.default_joint_pos)
    return StartBatch(
        state=state, xi=torch.zeros(1, 26), frame=torch.zeros(1, dtype=torch.long),
        prev_action=torch.zeros(1, 22), hist_states=state[None].expand(history, -1, -1).clone(),
        hist_actions=torch.zeros(history, 1, 22), task_type=torch.zeros(1, dtype=torch.long),
        velocity_command=torch.tensor([[0.2, 0.0, 0.0]]), fixed_world_target=torch.zeros(1, 3),
    )


class _WheelResponseWM(torch.nn.Module):
    def __init__(self, wheel_slot):
        super().__init__()
        self.spec = SimpleNamespace(wheel_indices=(3, 7, 11, 15), nj=22)
        self.wheel_slot = wheel_slot
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def step(self, state, action, xi):
        del xi
        self.seen.append(action.detach().clone())
        velocity = torch.zeros_like(state[:, :3])
        velocity[:, 0] = action[:, self.wheel_slot] * self.gain
        return torch.cat((
            state[:, :3] + velocity * 0.02, state[:, 3:7], state[:, 7:29],
            velocity, state[:, 32:35], state[:, 35:57],
        ), dim=-1)


def test_actor_and_direct_wm_paths_align_phase_and_retain_input_gradients(monkeypatch):
    from supermimic.motion_prior.reference import future31_to_m1_action22
    from supermimic.motion_prior.rollout import differentiable_m1_rollout
    from supermimic.policy.nets import M1ActionReferenceActor, M1ActionReferenceActorCfg

    spec = _spec()
    wheel_slot = spec.joint_names.index("FAR_FOOT_JOINT")
    actor = M1ActionReferenceActor(M1ActionReferenceActorCfg(history=2, hidden=(32, 16)))
    wm = _WheelResponseWM(wheel_slot)
    start = _start()
    future = _future()
    future[..., 27] = torch.linspace(0.2, 0.6, 10)
    future.requires_grad_(True)
    goal = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    for horizon in (3, 10):
        wm.seen.clear()
        rollout = differentiable_m1_rollout(
            wm, actor, start, goal, future, torch.tensor(spec.default_joint_pos),
            robot_spec=spec, horizon=horizon,
        )
        assert rollout.actor_states.shape == rollout.prior_direct_states.shape == (horizon + 1, 1, 57)
        assert rollout.actor_actions.shape == (horizon, 1, 22)
        assert rollout.reference_actions.shape == (1, horizon, 22)
        for k in range(horizon):
            torch.testing.assert_close(rollout.reference_actions[:, k], future31_to_m1_action22(future, robot_spec=spec)[:, k])
        # Each time step calls the same WM once for the Actor path and once for
        # the direct candidate path. The direct sequence is exactly U_ref[k].
        direct_actions = wm.seen[1::2]
        assert len(direct_actions) == horizon
        for k, action in enumerate(direct_actions):
            torch.testing.assert_close(action, rollout.reference_actions[:, k])
        grad = torch.autograd.grad(
            rollout.actor_states[-1, 0, 0] + rollout.prior_direct_states[-1, 0, 0],
            future, retain_graph=True,
        )[0]
        assert torch.isfinite(grad).all() and grad[..., 27].abs().sum() > 0


def test_actor_history_keeps_raw_action_while_wm_gets_clamped_execution():
    from supermimic.motion_prior.rollout import differentiable_m1_rollout

    spec = _spec()
    start = _start()
    future = _future()
    future[..., 27] = 0.2
    goal = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    wheel_slot = spec.joint_names.index("FAR_FOOT_JOINT")

    class RawActor:
        cfg = SimpleNamespace(history=2)
        inputs = []

        def __call__(self, proprio, current_goal, current_ref):
            del current_goal, current_ref
            self.inputs.append(proprio.clone())
            return torch.full((proprio.shape[0], 22), 1.2)

    actor = RawActor()
    wm = _WheelResponseWM(wheel_slot)
    rollout = differentiable_m1_rollout(
        wm, actor, start, goal, future, torch.tensor(spec.default_joint_pos),
        robot_spec=spec, horizon=2,
    )
    torch.testing.assert_close(rollout.actor_actions, torch.full((2, 1, 22), 1.2))
    torch.testing.assert_close(rollout.actor_executed_actions, torch.ones(2, 1, 22))
    torch.testing.assert_close(wm.seen[0], torch.ones(1, 22))
    torch.testing.assert_close(actor.inputs[1][..., -22:], torch.full((1, 22), 1.2))


def test_online_planner_uses_phase_action_and_replans_on_reset_or_goal_change():
    from supermimic.motion_prior.trainer import M1AlternatingTrainer

    spec = _spec()
    trainer = object.__new__(M1AlternatingTrainer)
    trainer.cfg = SimpleNamespace(prior_mode="prior_mean")
    trainer.robot_spec = spec

    class Prior:
        model = SimpleNamespace(latent_dim=32)

        def __init__(self):
            self.calls = 0

        def __call__(self, history, goal, mode):
            assert mode == "prior_mean"
            self.calls += 1
            future = _future(len(goal)).to(goal)
            future[..., 27:31] = goal[:, None, 0:1] + torch.arange(10, device=goal.device)[None, :, None] * 0.01
            return future

    class Actor:
        cfg = SimpleNamespace(history=5)

        def __init__(self):
            self.references = []

        def __call__(self, proprio, goal, reference):
            del goal
            assert proprio.shape == (len(reference), 5 * 72)
            self.references.append(reference.clone())
            return reference

    trainer.prior = Prior()
    trainer.actor = Actor()
    trainer._plan = None
    trainer._reference_actions = None
    trainer._reference_actions_unclipped = None
    trainer._plan_anchor_state = None
    trainer._plan_goal = None
    history = _start().state[:, None].expand(-1, 4, -1).clone()
    proprio = torch.zeros(1, 5 * 72)
    goal = torch.tensor([[0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    phase = torch.tensor([0])
    action0, _ = trainer.act(proprio, goal, history, phase)
    torch.testing.assert_close(action0, trainer._reference_actions[:, 0])
    assert trainer.prior.calls == 1

    phase[:] = 4
    trainer.act(proprio, goal, history, phase)
    torch.testing.assert_close(trainer.actor.references[-1], trainer._reference_actions[:, 4])
    assert trainer.prior.calls == 1

    changed_goal = goal.clone()
    changed_goal[:, 0] += 0.08
    phase[:] = 7
    trainer.act(proprio, changed_goal, history, phase)
    assert phase.item() == 0
    torch.testing.assert_close(trainer.actor.references[-1], trainer._reference_actions[:, 0])
    assert trainer.prior.calls == 2

    trainer.act(proprio, changed_goal, history, torch.tensor([0]))
    assert trainer.prior.calls == 3  # an episode reset always regenerates the plan


def test_expected_termination_failure_cost_orders_early_failures_and_has_finite_gradients():
    from supermimic.motion_prior.losses import expected_termination_failure_cost

    full_survival = torch.ones(10, 1)
    fail_first = torch.zeros(10, 1)
    fail_third = torch.tensor([1.0, 1.0, *([0.0] * 8)]).unsqueeze(-1)
    full_cost = expected_termination_failure_cost(full_survival, reference=full_survival)
    first_cost = expected_termination_failure_cost(fail_first, reference=fail_first)
    third_cost = expected_termination_failure_cost(fail_third, reference=fail_third)
    assert full_cost == 0 and first_cost == 1
    torch.testing.assert_close(third_cost, torch.tensor(0.8))
    assert first_cost > third_cost > full_cost

    fail_logits = torch.linspace(-2.0, 2.0, 10, requires_grad=True)[:, None]
    survival = torch.cumprod(1.0 - fail_logits.sigmoid(), dim=0)
    cost = expected_termination_failure_cost(survival, reference=survival)
    grad, = torch.autograd.grad(cost, fail_logits)
    assert torch.isfinite(cost) and torch.isfinite(grad).all()
    assert torch.all(grad > 0)  # lowering terminal probability lowers failure cost


def test_immediate_failure_is_costly_for_far_m_reach_and_large_l_velocity_error():
    from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses

    class RootAsTcp:
        def __call__(self, qpos, qvel):
            del qvel
            return {"pos": qpos[:, None, :3]}

    horizon = 10
    states = torch.zeros(horizon, 1, 57)
    states[..., 2] = 0.47
    states[..., 3] = 1.0
    actions = torch.zeros(horizon, 1, 22, requires_grad=True)
    reference = states.detach().clone()
    previous = torch.zeros(1, 22)
    soft_limits = torch.tensor([[-3.0, 3.0]] * 22)
    fail_now = torch.zeros(horizon, 1)
    first_state_safety = torch.zeros(horizon, 1)
    first_state_safety[0] = 1

    # L: a large positive speed command is missed, but termination cannot mask
    # the complete objective to zero.
    l_states = states.clone()
    l_states[..., 29] = -1.0
    l = actor_losses(
        l_states, reference, actions, previous, soft_limits=soft_limits,
        cfg=M1JointLossCfg(), task_type=torch.tensor([0]),
        velocity_command=torch.tensor([[0.3, 0.0, 0.0]]),
        fixed_world_target=torch.zeros(1, 3), fk=RootAsTcp(),
        reference_actions=torch.zeros_like(actions), initial_state=states[0],
        valid_mask=torch.zeros(horizon, 1), safety_mask=first_state_safety,
        survival_probability=fail_now,
    )
    assert l["task"] == 0  # hard post-termination task mask stays explicit
    assert l["failure_loss"] == 1 and l["total"] >= 1
    assert all(torch.isfinite(v).all() for v in l.values() if isinstance(v, torch.Tensor))

    # M: a far fixed target cannot become a zero-loss shortcut after an early
    # predicted fall; the terminal state remains in the dense safety mask.
    m = actor_losses(
        states, reference, actions, previous, soft_limits=soft_limits,
        cfg=M1JointLossCfg(), task_type=torch.tensor([1]),
        velocity_command=torch.zeros(1, 3),
        fixed_world_target=torch.tensor([[20.0, 0.0, 0.47]]), fk=RootAsTcp(),
        reference_actions=torch.zeros_like(actions), initial_state=states[0],
        valid_mask=torch.zeros(horizon, 1), safety_mask=first_state_safety,
        survival_probability=fail_now,
    )
    assert m["task"] == 0
    assert m["failure_loss"] == 1 and m["total"] >= 1


def test_task_loss_gradient_reaches_actions_with_soft_survival_and_hard_mask():
    from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses

    class RootAsTcp:
        def __call__(self, qpos, qvel):
            del qvel
            return {"pos": qpos[:, None, :3]}

    horizon = 10
    actions = torch.zeros(horizon, 1, 22, requires_grad=True)
    states = torch.zeros(horizon, 1, 57)
    states[..., 2] = 0.47
    states[..., 3] = 1.0
    states[..., 29] = actions[..., 0]
    survival = torch.ones(horizon, 1)
    terms = actor_losses(
        states, states.detach(), actions, torch.zeros(1, 22),
        soft_limits=torch.tensor([[-3.0, 3.0]] * 22), cfg=M1JointLossCfg(),
        task_type=torch.tensor([0]), velocity_command=torch.tensor([[0.2, 0.0, 0.0]]),
        fixed_world_target=torch.zeros(1, 3), fk=RootAsTcp(), reference_actions=actions.detach(),
        initial_state=states[0], valid_mask=torch.ones(horizon, 1),
        safety_mask=torch.ones(horizon, 1), survival_probability=survival,
    )
    grad, = torch.autograd.grad(terms["task"], actions)
    assert torch.isfinite(grad).all() and grad[..., 0].abs().sum() > 0


def test_rollout_uses_positive_termination_logit_as_failure_and_keeps_soft_gradient():
    from supermimic.motion_prior.rollout import differentiable_m1_rollout

    spec = _spec()
    start = _start()
    wheel_slot = spec.joint_names.index("FAR_FOOT_JOINT")

    class TerminationWM(_WheelResponseWM):
        def termination_logits(self, state):
            # In the production WM, BCE labels done=1 as a positive logit.
            return 10.0 * state[:, 29] - 1.5

    class EchoActor:
        cfg = SimpleNamespace(history=2)

        def __call__(self, proprio, goal, ref):
            del proprio, goal
            return ref

    future = _future()
    future[..., 27:31] = 1.0
    future.requires_grad_(True)
    goal = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    rollout = differentiable_m1_rollout(
        TerminationWM(wheel_slot), EchoActor(), start, goal, future,
        torch.tensor(spec.default_joint_pos), robot_spec=spec, horizon=3,
    )
    assert not rollout.prior_valid_mask[0].item()
    assert rollout.prior_safety_mask[0].item()
    assert rollout.prior_survival_probability[0, 0] < 0.5
    grad, = torch.autograd.grad(rollout.prior_survival_probability[-1].sum(), future)
    assert torch.isfinite(grad).all() and grad[..., 27:31].abs().sum() > 0


def test_fixed_balanced_env_slots_are_exact_and_survive_episode_resets():
    from supermimic.envs.mdp.commands import M1PriorJointCommand, fixed_balanced_task_labels

    labels = fixed_balanced_task_labels(2048, 0.5, "cpu")
    assert torch.equal(torch.bincount(labels, minlength=2), torch.tensor([1024, 1024]))
    cfg = SimpleNamespace(
        task_type="balanced", balanced_fixed_quota=True, l_task_fraction=0.5,
        vx_range=(-0.09, 0.145), vx_stop_fraction=0.15, vx_reverse_fraction=0.35,
        vy_range=(-0.05, 0.055), wz_range=(-0.17, 0.18),
    )
    command = SimpleNamespace(
        cfg=cfg, device=torch.device("cpu"), task_type=labels.clone(),
        velocity_command_body=torch.zeros(2048, 3),
        target_position_world=torch.zeros(2048, 3),
        _hold_seconds=torch.zeros(2048), _episode_success=torch.zeros(2048, dtype=torch.bool),
    )
    original = command.task_type.clone()
    M1PriorJointCommand._resample_command(command, torch.arange(2048))
    assert torch.equal(command.task_type, original)
    assert torch.equal(torch.bincount(command.task_type, minlength=2), torch.tensor([1024, 1024]))
    assert torch.equal(command.velocity_command_body[command.task_type == 1], torch.zeros(1024, 3))


def test_balanced_checkpoint_can_be_evaluated_for_either_fixed_task():
    from scripts.evaluate_m1_prior_joint import _resolve_saved_task_type

    checkpoint = {"task_sampler_state": {"task_type": "balanced"}}
    assert _resolve_saved_task_type(checkpoint, "L") == "L"
    assert _resolve_saved_task_type(checkpoint, "M") == "M"
    with pytest.raises(ValueError, match="requires task_type"):
        _resolve_saved_task_type(checkpoint)
    fixed_l = {"task_sampler_state": {"task_type": "L"}}
    with pytest.raises(ValueError, match="differs"):
        _resolve_saved_task_type(fixed_l, "M")
