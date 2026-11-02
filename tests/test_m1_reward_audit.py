"""Regression tests for M1 L/M reward and real-metric semantics."""

from __future__ import annotations

import torch


def _states(batch: int = 1, horizon: int = 10) -> torch.Tensor:
    from supermimic.robots.spec import M1_DEFAULT

    state = torch.zeros(batch, 57)
    state[:, 2] = 0.65
    state[:, 3] = 1.0
    state[:, 7:29] = torch.tensor(M1_DEFAULT)
    return state[None].expand(horizon + 1, -1, -1).clone()


class _RootTcpFK:
    """Small FK fixture: a fixed local TCP offset from the root."""

    def __call__(self, qpos, qvel=None):
        del qvel
        pos = qpos[:, :3] + qpos.new_tensor([0.30, 0.0, 0.25])
        return {"pos": pos[:, None]}


def _future_reference(initial: torch.Tensor) -> torch.Tensor:
    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS

    batch = initial.shape[0]
    reference = torch.zeros(batch, 10, 31)
    reference[..., 3] = 1.0
    reference[..., 7] = 1.0  # identity rotation6D columns: [1,0,0, 0,1,0]
    for i, name in enumerate(M1_AMP_JOINTS):
        reference[..., 9 + i] = initial[:, M1_JOINTS.index(name) + 7, None]
    return reference


def _planner_terms(states: torch.Tensor, task: int, command: torch.Tensor, target: torch.Tensor):
    from supermimic.motion_prior.losses import M1JointLossCfg, _task_planner_terms

    return _task_planner_terms(
        states, torch.full((states.shape[1],), task), command, target,
        _RootTcpFK(), 0, M1JointLossCfg(),
    )


def _planner_total(states: torch.Tensor, task: int, command: torch.Tensor, target: torch.Tensor):
    from supermimic.motion_prior.losses import M1JointLossCfg, planner_losses

    initial = states[0]
    reference = _future_reference(initial)
    return planner_losses(
        states, torch.zeros(10, states.shape[1], 22), initial,
        reference, reference.clone(), torch.full((states.shape[1],), task),
        command, target, fk=_RootTcpFK(), tcp_index=0,
        soft_limits=torch.tensor([[-10.0, 10.0]] * 22), cfg=M1JointLossCfg(),
    )


def test_m1_velocity_command_axes():
    from supermimic.motion_prior.velocity import velocity_command_errors

    state = _states(horizon=1)[1]
    cases = [
        (torch.tensor([[0.0, 0.0, 0.0]]), (0.0, 0.0, 0.0)),
        (torch.tensor([[0.3, 0.0, 0.0]]), (0.3, 0.0, 0.0)),
        (torch.tensor([[0.0, 0.2, 0.0]]), (0.0, 0.2, 0.0)),
        (torch.tensor([[0.0, 0.0, 0.3]]), (0.0, 0.0, 0.3)),
        (torch.tensor([[0.3, 0.0, 0.2]]), (0.3, 0.0, 0.2)),
    ]
    for command, desired in cases:
        state[:, 29:32] = torch.tensor(desired[:2] + (0.0,))
        state[:, 34] = desired[2]
        xy_error, _, yaw_error, _ = velocity_command_errors(state, command)
        torch.testing.assert_close(xy_error, torch.zeros_like(xy_error))
        torch.testing.assert_close(yaw_error, torch.zeros_like(yaw_error))


def test_m1_pure_yaw_no_vertical_tracking():
    from supermimic.motion_prior.velocity import velocity_command_errors

    command = torch.tensor([[0.0, 0.0, 0.3]])
    states = _states(horizon=1)
    states[1, 0, 34] = 0.3
    states[1, 0, 31] = 12.0
    first = _planner_terms(states, 0, command, torch.zeros(1, 3))
    xy_error_1, xy_sq_1, yaw_1, vz_1 = velocity_command_errors(states[1], command)
    states[1, 0, 31] = -17.0
    second = _planner_terms(states, 0, command, torch.zeros(1, 3))
    xy_error_2, xy_sq_2, yaw_2, vz_2 = velocity_command_errors(states[1], command)
    torch.testing.assert_close(first["task_l"], torch.tensor(0.0))
    torch.testing.assert_close(first["task_l"], second["task_l"])
    torch.testing.assert_close(first["task_l_linear_velocity"], torch.tensor(0.0))
    torch.testing.assert_close(first["task_l_yaw_rate"], torch.tensor(0.0))
    torch.testing.assert_close(xy_error_1, xy_error_2)
    torch.testing.assert_close(xy_sq_1, xy_sq_2)
    torch.testing.assert_close(yaw_1, yaw_2)
    assert vz_1.item() == 12.0 and vz_2.item() == -17.0


def test_m1_real_imag_velocity_metric_parity():
    from supermimic.motion_prior.losses import _task_planner_terms
    from supermimic.motion_prior.velocity import velocity_command_errors

    states = _states(horizon=2)
    states[:, 0, 29:32] = torch.tensor([0.32, -0.08, 0.7])
    states[:, 0, 34] = 0.19
    command = torch.tensor([[0.30, -0.10, 0.20]])
    xy_error, xy_sq_error, yaw_error, _ = velocity_command_errors(states[1:], command)
    terms = _task_planner_terms(
        states, torch.tensor([0]), command, torch.zeros(1, 3),
        _RootTcpFK(), 0,
        __import__("supermimic.motion_prior.losses", fromlist=["M1JointLossCfg"]).M1JointLossCfg(),
    )
    expected_linear = torch.nn.functional.smooth_l1_loss(
        xy_error, torch.zeros_like(xy_error), reduction="none",
    ).div(0.5).mean(-1).mean()
    expected_yaw = torch.nn.functional.smooth_l1_loss(
        yaw_error, torch.zeros_like(yaw_error), reduction="none",
    ).mean()
    torch.testing.assert_close(terms["task_l_linear_velocity"], expected_linear)
    torch.testing.assert_close(terms["task_l_yaw_rate"], expected_yaw)
    torch.testing.assert_close(xy_sq_error.mean().sqrt(), torch.tensor((0.02**2 + 0.02**2) ** 0.5))


def test_m1_stationary_vs_moving_task_loss():
    command = torch.tensor([[0.3, 0.0, 0.0]])
    target = torch.zeros(1, 3)
    stationary = _states()
    moving = _states()
    moving[:, 0, 0] = torch.arange(11) * 0.02 * 0.3
    moving[1:, 0, 29] = 0.3
    wrong_way = _states()
    wrong_way[:, 0, 0] = -torch.arange(11) * 0.02 * 0.3
    wrong_way[1:, 0, 29] = -0.3
    wheel_spin = _states()
    wheel_spin[1:, 0, 35:57] = 0.0
    from supermimic.robots.spec import M1_JOINTS

    for name in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT"):
        wheel_spin[1:, 0, 35 + M1_JOINTS.index(name)] = 5.0
    crouched = _states()
    crouched[:, 0, 2] = 0.20

    losses = {
        name: _planner_total(value, 0, command, target)
        for name, value in (
            ("stationary", stationary), ("moving", moving), ("wrong_way", wrong_way),
            ("wheel_spin", wheel_spin), ("crouched", crouched),
        )
    }
    assert losses["moving"]["task_l"] < losses["stationary"]["task_l"]
    assert losses["stationary"]["task_l"] < losses["wrong_way"]["task_l"]
    torch.testing.assert_close(losses["wheel_spin"]["task_l"], losses["stationary"]["task_l"])
    assert losses["crouched"]["feasibility"] > losses["stationary"]["feasibility"]
    assert losses["moving"]["total"] < losses["stationary"]["total"]
    assert losses["stationary"]["total"] < losses["wrong_way"]["total"]


def test_m1_l_task_local_gradient_axes_and_direction():
    command = torch.tensor([[0.3, 0.0, 0.3]])
    target = torch.zeros(1, 3)

    def grad_at(state: torch.Tensor) -> torch.Tensor:
        state = state.requires_grad_(True)
        loss = _planner_terms(state, 0, command, target)["task_l"]
        return torch.autograd.grad(loss, state)[0]

    stationary = _states(horizon=1)
    stationary_grad = grad_at(stationary)
    assert stationary_grad[1, 0, 29] < 0  # gradient descent increases vx toward command
    assert stationary_grad[1, 0, 34] < 0  # and increases wz toward command
    assert stationary_grad[1, 0, 31] == 0  # wz is never a vz target

    correct = _states(horizon=1)
    correct[1, 0, 29] = 0.3
    correct[1, 0, 34] = 0.3
    correct_grad = grad_at(correct)
    torch.testing.assert_close(correct_grad[1, 0, [29, 31, 34]], torch.zeros(3))

    wrong_way = _states(horizon=1)
    wrong_way[1, 0, 29] = -0.3
    wrong_grad = grad_at(wrong_way)
    assert wrong_grad[1, 0, 29] < 0  # still points toward positive vx

    wheel_spin = _states(horizon=1)
    from supermimic.robots.spec import M1_JOINTS

    wheel_index = 35 + M1_JOINTS.index("FAR_FOOT_JOINT")
    wheel_spin[1, 0, wheel_index] = 5.0
    wheel_grad = grad_at(wheel_spin)
    assert wheel_grad[1, 0, wheel_index] == 0  # wheel speed alone cannot satisfy base velocity

    crouched = _states(horizon=1)
    crouched[1, 0, 2] = 0.20
    crouched.requires_grad_(True)
    crouch_total = _planner_total(crouched, 0, command, target)["total"]
    crouch_grad = torch.autograd.grad(crouch_total, crouched)[0]
    # The soft height penalty should push height upward; state index 2 is z.
    assert crouch_grad is not None and crouch_grad[1, 0, 2] < 0


def test_m1_loss_component_scale():
    from supermimic.motion_prior.losses import M1JointLossCfg, _reference_smoothness_terms

    initial = _states(horizon=10)[0]
    reference = _future_reference(initial)
    reference[:, :, 0] = torch.tensor([0.0, 0.005, 0.0, 0.005, 0.0, 0.005, 0.0, 0.005, 0.0, 0.005])
    terms = _reference_smoothness_terms(initial, reference, M1JointLossCfg())
    assert all(torch.isfinite(value) for value in terms.values())
    assert all(value >= 0 for value in terms.values())
    unweighted = torch.stack(tuple(terms.values())).sum()
    sweeps = {weight: weight * unweighted for weight in (0.0, 0.0001, 0.001, 0.003, 0.01)}
    assert sweeps[0.0] == 0
    assert sweeps[0.0001] < sweeps[0.001] < sweeps[0.003] < sweeps[0.01]
    assert all(torch.isfinite(value) for value in sweeps.values())


def test_m1_reach_progress_objective():
    command = torch.zeros(1, 3)
    initial = _states()
    target = torch.tensor([[0.60, 0.0, 0.90]])
    stationary = initial.clone()
    approaching = initial.clone()
    approaching[:, 0, 0] = torch.arange(11) * 0.015
    approaching[1:, 0, 29] = 0.75
    stay = _planner_total(stationary, 1, command, target)
    progress = _planner_total(approaching, 1, command, target)
    assert progress["task_progress"] > 0
    assert progress["terminal_error"] < stay["terminal_error"]
    assert progress["task_m"] < stay["task_m"]
    assert progress["total"] < stay["total"]


def test_m1_reach_progress_forward_side_up_and_near_goals():
    command = torch.zeros(1, 3)
    start = _states(horizon=10)[0]
    tcp0 = start[0, :3] + torch.tensor([0.30, 0.0, 0.25])
    cases = {
        "forward": (tcp0 + torch.tensor([0.30, 0.0, 0.0]), torch.tensor([0.10, 0.0, 0.0])),
        "side": (tcp0 + torch.tensor([0.0, 0.30, 0.0]), torch.tensor([0.0, 0.10, 0.0])),
        "up": (tcp0 + torch.tensor([0.0, 0.0, 0.30]), torch.tensor([0.0, 0.0, 0.10])),
        "near": (tcp0 + torch.tensor([0.04, 0.0, 0.0]), torch.tensor([0.01, 0.0, 0.0])),
    }
    for target_name, (target, displacement) in cases.items():
        stationary = _states(horizon=10)
        approaching = _states(horizon=10)
        for frame in range(11):
            approaching[frame, 0, :3] = start[0, :3] + displacement * (frame / 10.0)
        stationary_loss = _planner_total(stationary, 1, command, target[None])
        approaching_loss = _planner_total(approaching, 1, command, target[None])
        assert approaching_loss["task_progress"] > 0, target_name
        assert approaching_loss["task_m"] < stationary_loss["task_m"], target_name

        probe = approaching.requires_grad_(True)
        terms = _planner_terms(probe, 1, command, target[None])
        gradient = torch.autograd.grad(terms["task_m"], probe)[0]
        direction = displacement / torch.linalg.vector_norm(displacement)
        assert torch.dot(gradient[-1, 0, :3], direction) < 0, target_name


def test_m1_goal8_recomputes_delta_to_fixed_world_target():
    from supermimic.motion_prior.features import task_goal8

    target = torch.tensor([[1.0, 0.2, 0.9]])
    task = torch.tensor([1])
    command = torch.zeros(1, 3)
    first = _states(horizon=0)[0]
    moved = first.clone()
    moved[:, 0] = 0.1
    first_goal = task_goal8(first, task, command, target)
    moved_goal = task_goal8(moved, task, command, target)
    torch.testing.assert_close(first_goal[0, 3:6], torch.tensor([1.0, 0.2, 0.25]))
    torch.testing.assert_close(moved_goal[0, 3:6], torch.tensor([0.9, 0.2, 0.25]))
    # The caller retains the same fixed world target; only the heading-relative
    # representation is recomputed from the current base pose.
    torch.testing.assert_close(target, torch.tensor([[1.0, 0.2, 0.9]]))


def test_m1_reach_success_requires_stable_base():
    import math
    from types import SimpleNamespace

    from supermimic.kinematics.rotations import quat_from_rotvec
    from supermimic.motion_prior.metrics import root_tilt_and_stability
    from supermimic.training.loop import Trainer

    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    heights = torch.tensor([0.65])
    tilt, stable = root_tilt_and_stability(
        upright, heights, minimum_height_m=0.20, maximum_tilt_rad=math.radians(70.0),
    )
    assert tilt.item() == 0.0 and stable.item()

    low_tilt, low_stable = root_tilt_and_stability(
        upright, torch.tensor([0.19]),
        minimum_height_m=0.20, maximum_tilt_rad=math.radians(70.0),
    )
    assert low_tilt.item() == 0.0 and not low_stable.item()

    fallen_quaternion = quat_from_rotvec(torch.tensor([[math.radians(80.0), 0.0, 0.0]]))
    high_tilt, high_tilt_stable = root_tilt_and_stability(
        fallen_quaternion, heights,
        minimum_height_m=0.20, maximum_tilt_rad=math.radians(70.0),
    )
    torch.testing.assert_close(high_tilt, torch.tensor([math.radians(80.0)]), atol=1e-6, rtol=0)
    assert not high_tilt_stable.item()

    class RootAsTcpFK:
        def __call__(self, qpos, qvel):
            del qvel
            return {"pos": qpos[:, None, :3]}

    trainer = object.__new__(Trainer)
    trainer.command = SimpleNamespace(cfg=SimpleNamespace(
        success_minimum_base_height_m=0.20,
        success_maximum_tilt_rad=math.radians(70.0),
        success_distance=0.08,
    ))
    trainer.m1_joint_trainer = SimpleNamespace(fk=RootAsTcpFK(), tcp_index=0)
    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
        soft_joint_pos_limits=torch.tensor([[[-3.0, 3.0]] * 22]),
    )))
    trainer.env = SimpleNamespace(scene=scene)
    for height, expected_success in ((0.65, 1.0), (0.19, 0.0)):
        state = _states(horizon=0)[0]
        state[0, 2] = height
        measured = trainer._measure_m1_joint_task_step(
            state, torch.tensor([1]), torch.zeros(1, 3), state[:, :3].clone(),
            torch.zeros(1, 3), torch.zeros(1),
        )
        assert measured["ee_distance"].item() == 0.0
        assert measured["ee_success"].item() == expected_success


def test_m1_task_masked_metrics():
    from supermimic.motion_prior.velocity import task_masked_mean

    values = torch.tensor([0.2, 0.0, 0.4, 0.0])
    l_mask = torch.tensor([True, False, True, False])
    m_mask = ~l_mask
    torch.testing.assert_close(task_masked_mean(values, l_mask), torch.tensor(0.3))
    torch.testing.assert_close(task_masked_mean(values, m_mask), torch.tensor(0.0))
    assert task_masked_mean(values, torch.zeros(4, dtype=torch.bool)) is None


def test_m1_wm_readiness_requires_configured_quality_limits():
    from supermimic.motion_prior.trainer import M1AlternatingTrainer

    trainer = object.__new__(M1AlternatingTrainer)
    trainer.cfg = type("Cfg", (), {
        "wm_readiness_min_events": 2,
        "wm_readiness_max_nse_1": None,
        "wm_readiness_max_nse_horizon": None,
    })()
    trainer.wm_update_events = 2
    trainer.wm_optimizer_steps = 2
    trainer.last_wm_metrics = {"nse_1step": 0.02, "nse_kstep": 0.20}
    diag = trainer._wm_readiness(policy_allowed=True)
    assert diag["schedule/wm_fit_ready"] == 1.0
    assert diag["schedule/wm_ready"] == 0.0
    assert diag["schedule/wm_quality_thresholds_configured"] == 0.0
    trainer.cfg.wm_readiness_max_nse_1 = 0.10
    trainer.cfg.wm_readiness_max_nse_horizon = 0.30
    diag = trainer._wm_readiness(policy_allowed=True)
    assert diag["schedule/wm_ready"] == 1.0
    trainer.cfg.wm_readiness_max_nse_horizon = 0.10
    assert trainer._wm_readiness(policy_allowed=True)["schedule/wm_ready"] == 0.0


def test_m1_log_formats_unavailable_metrics_without_nan():
    from supermimic.training.loop import _format_m1_aggregate, _format_m1_metric

    assert _format_m1_metric(float("nan")) == "n/a"
    assert _format_m1_metric(0.25, 2) == "0.25"
    assert _format_m1_aggregate({}, 0, "prior/task_loss") == "n/a"


def test_m1_prior_actor_gradients_after_reward_fix():
    from supermimic.motion_prior.losses import M1JointLossCfg, _task_planner_terms

    command = torch.tensor([[0.0, 0.0, 0.3]])
    states = _states(horizon=1).requires_grad_(True)
    states.data[1, 0, 34] = 0.1
    states.data[1, 0, 31] = 3.0
    terms = _task_planner_terms(
        states, torch.tensor([0]), command, torch.zeros(1, 3),
        _RootTcpFK(), 0, M1JointLossCfg(),
    )
    grad = torch.autograd.grad(terms["task_l"], states)[0]
    assert torch.isfinite(grad).all()
    assert grad[1, 0, 31].abs() == 0  # vertical velocity has no L task gradient
    assert grad[1, 0, 34].abs() > 0  # yaw command reaches yaw rate
