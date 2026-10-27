"""Pure checks for Go2 reward math, curriculum promotion, and training schedules."""

from __future__ import annotations

import math

import pytest
import torch

from supermimic.tasks.velocity_reward import (
    VelocityRewardCfg,
    VelocityRobotParams,
    velocity_reward_terms,
    velocity_reward_total,
)
from supermimic.training.schedule import actor_ramp_scale, scheduled_lr


def _params(n: int = 12) -> VelocityRobotParams:
    q0 = torch.linspace(-0.3, 0.3, n)
    return VelocityRobotParams(
        default_joint_pos=q0,
        soft_joint_limits=torch.stack([q0 - 1.0, q0 + 1.0], dim=-1),
        action_scales=torch.full((n,), 0.25),
        kp=torch.full((n,), 40.0),
        kd=torch.ones(n),
        effort_limits=torch.full((n,), 23.5),
        dt=0.02,
        cfg=VelocityRewardCfg(min_safe_height=0.22),
    )


def _states(batch: int = 4, n: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    s0 = torch.zeros(batch, 13 + 2 * n)
    s1 = torch.zeros_like(s0)
    s0[:, 2] = s1[:, 2] = 0.29
    s0[:, 3] = s1[:, 3] = 1.0
    s0[:, 7 : 7 + n] = s1[:, 7 : 7 + n] = _params(n).default_joint_pos
    s0[:, 13 + n : 13 + 2 * n] = torch.randn(batch, n) * 0.1
    s1[:, 13 + n : 13 + 2 * n] = s0[:, 13 + n : 13 + 2 * n] + torch.randn(batch, n) * 0.02
    s1[:, 10 + n + 2] = 0.1
    return s0, s1


def test_velocity_reward_terms_have_dt_once_and_differentiate_state_and_action():
    n, batch = 12, 4
    params = _params(n)
    s0, s1 = _states(batch, n)
    s0.requires_grad_()
    s1.requires_grad_()
    action = torch.full((batch, n), 0.1, requires_grad=True)
    previous_action = torch.zeros_like(action)
    goal = torch.tensor([0.3, 0.0, 0.1]).expand(batch, -1)

    terms = velocity_reward_terms(s0, s1, goal, action, previous_action, params)
    total = velocity_reward_total(terms)
    assert all(value.shape == (batch,) for value in terms.values())
    (-(total * params.dt).mean()).backward()
    assert s1.grad is not None and torch.isfinite(s1.grad).all() and s1.grad.abs().sum() > 0
    assert action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    # The core returns rates; only VelocityCommandTask / RewardManager adds control dt.
    torch.testing.assert_close(terms["track_lin_vel_xy"], torch.full((batch,), 3.0 * math.exp(-0.09 / 0.25)))


def test_real_measurements_use_the_same_reward_core_without_proxy_terms():
    n, batch = 12, 3
    params = _params(n)
    s0, s1 = _states(batch, n)
    action = torch.randn(batch, n) * 0.1
    prev_action = torch.randn(batch, n) * 0.1
    goal = torch.tensor([[0.2, 0.1, -0.2]]).expand(batch, -1)
    torque = torch.randn(batch, n) * 2
    acceleration = torch.randn(batch, n) * 5

    direct = velocity_reward_terms(
        s0, s1, goal, action, prev_action, params,
        measured_torque=torque, measured_joint_acceleration=acceleration,
    )
    from supermimic.imagination.tasks import VelocityCommandTask

    task = VelocityCommandTask(params=params, dt=params.dt)
    imagined_api = task.reward_terms(
        s0, s1, goal, action, prev_action,
        measured_torque=torque, measured_joint_acceleration=acceleration,
    )
    for name in direct:
        torch.testing.assert_close(imagined_api[name], direct[name] * params.dt)


def test_direction_reward_handles_forward_reverse_sideways_and_zero_commands():
    params = _params()
    state, _ = _states(batch=4)
    # State's world-frame base linear velocity begins at 19 for 12 joints.
    state[:, 19:22] = torch.tensor([[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.4, 0.0, 0.0]])
    goals = torch.tensor([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]])
    terms = velocity_reward_terms(state, state, goals, torch.zeros(4, 12), torch.zeros(4, 12), params)
    direction = terms["command_direction"]
    assert torch.all(direction[:3] > 0)
    assert direction[3] == 0
    assert torch.all(direction.abs() <= params.cfg.command_direction)


def test_analytic_go2_termination_is_near_zero_standing_and_differentiable_at_failure_margin():
    from supermimic.imagination.tasks import VelocityCommandTask

    task = VelocityCommandTask(fall_height=0.17, tilt_feature_threshold=1.0 - math.cos(math.radians(70.0)))
    state = torch.zeros(2, 37, requires_grad=True)
    state.data[:, 2] = torch.tensor([0.29, 0.16])
    state.data[:, 3] = 1.0
    p = task.termination_probability(state)
    assert float(p[0].detach()) < 1e-3
    assert float(p[1].detach()) > 0.5
    p.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_actor_ramp_counts_real_optimizer_updates_and_lr_mode_is_explicit():
    assert actor_ramp_scale(1, 1000) == pytest.approx(0.001)
    assert actor_ramp_scale(500, 1000) == pytest.approx(0.5)
    assert actor_ramp_scale(1000, 1000) == 1.0
    assert actor_ramp_scale(500, 0) == 1.0
    assert scheduled_lr("constant", 90_000, 80_000, 2e-4) == pytest.approx(2e-4)
    # max_hours changes runtime only; the configured 80k-step curve is identical.
    from supermimic.config import ScheduleCfg

    short = ScheduleCfg(total_iterations=80_000, max_hours=0.5)
    long = ScheduleCfg(total_iterations=80_000, max_hours=None)
    assert scheduled_lr("cosine", 1000, short.total_iterations, 2e-4) == pytest.approx(
        scheduled_lr("cosine", 1000, long.total_iterations, 2e-4)
    )


def test_critic_bootstrap_value_clip_is_optional_and_task_configurable():
    from supermimic.imagination.policy_trainer import _clip_bootstrap_values

    value = torch.tensor([-20.0, -0.5, 0.25, 30.0])
    unchanged, fraction = _clip_bootstrap_values(value, None)
    torch.testing.assert_close(unchanged, value)
    assert fraction == 0
    clipped, fraction = _clip_bootstrap_values(value, 5.0)
    torch.testing.assert_close(clipped, torch.tensor([-5.0, -0.5, 0.25, 5.0]))
    assert float(fraction) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="positive"):
        _clip_bootstrap_values(value, 0.0)


def test_multirobot_defaults_keep_m1_2048_go2_3072_and_g1_unchanged():
    from types import SimpleNamespace

    from supermimic.config import ScheduleCfg
    from supermimic.training.loop import Trainer

    assert ScheduleCfg().wm_only_iterations == 200
    assert ScheduleCfg().m1_wm_only_iterations == 200
    assert ScheduleCfg().g1_motion_wm_only_updates == 0
    assert ScheduleCfg().lr_schedule == "cosine"
    assert Trainer._resolved_num_envs("m1_piper") == 2048
    assert Trainer._resolved_num_envs("go2") == 3072
    assert Trainer._resolved_num_envs("g1") == 4096
    assert Trainer._resolved_num_envs("go2", 128) == 128

    trainer = object.__new__(Trainer)
    trainer.cfg = SimpleNamespace(
        env=SimpleNamespace(robot_name="g1", task_name="motion_tracking"),
        schedule=SimpleNamespace(g1_motion_wm_only_updates=200, m1_wm_only_iterations=80, wm_only_iterations=90),
    )
    trainer.is_m1_motion = False
    trainer.pretrained_wm = False
    assert trainer._effective_wm_only_target() == 200
    trainer.cfg.schedule.g1_motion_wm_only_updates = 0
    assert trainer._effective_wm_only_target() == 0


def test_velocity_command_ranges_and_two_window_curriculum_promotion():
    from supermimic.envs.mdp.commands import VelocityCommand, VelocityCommandCfg

    cfg = VelocityCommandCfg()
    command = object.__new__(VelocityCommand)
    command.cfg = cfg
    from types import SimpleNamespace

    command._env = SimpleNamespace(device=torch.device("cpu"))
    command.stage = 0
    command.eval_count = 0
    command.promotion_streak = 0
    command.promotion_count = 0
    command._eval_window = []
    command.last_eval_metrics = {}
    a = command._sample_a(10_000)
    moving = torch.linalg.vector_norm(a[:, :2], dim=-1) > cfg.moving_command_threshold
    assert float(a[moving, 0].min()) >= 0.15
    assert float(a[moving, 0].max()) <= 0.55
    assert float((~moving).float().mean()) == pytest.approx(0.15, abs=0.02)
    assert torch.all(a[:, 1:] == 0)

    def passing_window():
        command._eval_window = [
            {
                "vel_rmse": 0.05,
                "yaw_rmse": float("nan"),
                "static_speed": 0.01,
                "moving_steps": 200,
                "yaw_steps": 0,
                "static_steps": 200,
                "fall": 0.0,
                "length": 500,
                "height": 0.28,
                "direction": -1,
                "magnitude": -1,
            }
            for _ in range(cfg.promotion_window_episodes)
        ]
        command._evaluate_window()

    passing_window()
    assert command.stage == 0 and command.promotion_streak == 1
    passing_window()
    assert command.stage == 1 and command.promotion_count == 1
    saved = command.state_dict()
    restored = object.__new__(VelocityCommand)
    restored.stage = 0
    restored.load_state_dict(saved)
    assert restored.stage == 1 and restored.promotion_count == 1


def test_go2_fixed_play_command_survives_command_resampling():
    from supermimic.envs.mdp.commands import VelocityCommand, VelocityCommandCfg

    command = object.__new__(VelocityCommand)
    command.cfg = VelocityCommandCfg(fixed_command=(0.3, 0.0, 0.5))
    from types import SimpleNamespace

    command._env = SimpleNamespace(device=torch.device("cpu"))
    command._command = torch.zeros(4, 3)
    command.stage = 0
    command._sample_commands = lambda n, stage: pytest.fail("fixed playback must not resample")
    for _ in range(3):
        torch.testing.assert_close(command._commands_for_reset(4), torch.tensor([[0.3, 0.0, 0.5]]).expand(4, -1))
