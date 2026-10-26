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
    torch.testing.assert_close(terms["track_lin_vel_xy"], torch.full((batch,), 2.0 * math.exp(-0.09 / 0.25)))


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


def test_multirobot_defaults_keep_m1_2048_go2_3072_and_g1_unchanged():
    from supermimic.config import ScheduleCfg
    from supermimic.training.loop import Trainer

    assert ScheduleCfg().wm_only_iterations == 200
    assert ScheduleCfg().m1_wm_only_iterations == 200
    assert ScheduleCfg().lr_schedule == "cosine"
    assert Trainer._resolved_num_envs("m1_piper") == 2048
    assert Trainer._resolved_num_envs("go2") == 3072
    assert Trainer._resolved_num_envs("g1") == 4096
    assert Trainer._resolved_num_envs("go2", 128) == 128


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
