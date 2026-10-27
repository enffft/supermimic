from __future__ import annotations

import torch

from supermimic.imagination.tasks import ChaseBallTask
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.chase_reward import ChaseBallRewardCfg, chase_ball_reward_terms
from supermimic.tasks.world_point_reward import WorldPointRobotParams, tcp_position_world, world_point_reward_terms
from supermimic.tasks.velocity_reward import VelocityRobotParams


def _go2_params() -> VelocityRobotParams:
    n = 12
    return VelocityRobotParams(
        default_joint_pos=torch.zeros(n), soft_joint_limits=torch.tensor([[-3.0, 3.0]] * n),
        action_scales=torch.full((n,), 0.25), kp=torch.full((n,), 40.0), kd=torch.ones(n),
        effort_limits=torch.tensor([23.7, 23.7, 35.55] * 4), dt=0.02,
    )


def test_chase_torque_and_acceleration_regularizers_have_expected_sign_and_dt():
    p = _go2_params()
    s0 = torch.zeros(2, 37)
    s1 = s0.clone()
    s0[:, 2] = s1[:, 2] = 0.3
    s0[:, 3] = s1[:, 3] = 1.0
    action = torch.zeros(2, 12)
    ball = torch.tensor([[1.0, 0.0, 0.09, 0.0, 0.0, 0.0]]).expand(2, -1)
    terms_zero = chase_ball_reward_terms(s1, ball, action, action, p.default_joint_pos, ChaseBallRewardCfg(), state_prev=s0, robot_params=p)
    s1[:, 25:] = 0.02
    terms = chase_ball_reward_terms(s1, ball, action, action, p.default_joint_pos, ChaseBallRewardCfg(), state_prev=s0, robot_params=p)
    assert torch.all(terms_zero["joint_torque"] == 0)
    assert torch.all(terms["joint_acceleration"] < 0)
    task = ChaseBallTask(p.default_joint_pos, params=p, dt=0.02)
    scaled = task.reward_terms(s0, s1, ball, action, action)
    torch.testing.assert_close(scaled["joint_acceleration"], terms["joint_acceleration"] * 0.02)


def test_chase_regularizer_gradient_reaches_action_through_pd_proxy():
    p = _go2_params()
    s0 = torch.zeros(1, 37); s1 = s0.clone(); s0[:, 2] = s1[:, 2] = 0.3; s0[:, 3] = s1[:, 3] = 1.0
    action = torch.full((1, 12), 0.2, requires_grad=True)
    ball = torch.tensor([[1.0, 0.0, 0.09, 0.0, 0.0, 0.0]])
    terms = chase_ball_reward_terms(s1, ball, action, torch.zeros_like(action), p.default_joint_pos, ChaseBallRewardCfg(), state_prev=s0, robot_params=p)
    (-terms["joint_torque"]).sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def test_m1_leg_action_magnitude_excludes_wheels_and_piper():
    robot = get_robot_spec("m1_piper")
    fk = G1Kinematics(load_robot_spec_model("m1_piper"), device="cpu")
    tcp_idx = (fk.body_index("piper_x_link7"), fk.body_index("piper_x_link8"))
    q0 = torch.tensor(robot.default_joint_pos)
    params = WorldPointRobotParams(q0, torch.tensor([[-3.0, 3.0]] * 22), dt=0.02)
    s0 = torch.zeros(1, 57); s1 = s0.clone(); s0[:, 2] = s1[:, 2] = robot.default_base_height; s0[:, 3] = s1[:, 3] = 1.0
    s0[:, 7:29] = s1[:, 7:29] = q0
    target = tcp_position_world(s1, fk, tcp_idx)
    action = torch.zeros(1, 22); action[:, 3] = 0.8; action[:, 16] = 0.8; action[:, 0] = 0.8
    terms = world_point_reward_terms(s0, s1, target, action, torch.zeros_like(action), params, fk, tcp_idx)
    expected = params.leg_action_magnitude_weight * 0.8**2
    torch.testing.assert_close(terms["leg_action_magnitude"], torch.tensor([expected]))
