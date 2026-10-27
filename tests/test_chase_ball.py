from __future__ import annotations

import torch

from supermimic.envs.mdp.motion_library import RefBodies, RefState, full_body_goal_features
from supermimic.imagination.tasks import ChaseBallTask
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.tasks.chase_reward import (
    ChaseBallRewardCfg,
    chase_ball_reward_terms,
    chase_goal_yaw_frame,
)


def _go2_state(batch: int = 1) -> torch.Tensor:
    state = torch.zeros(batch, 37)
    state[:, 2] = 0.29
    state[:, 3] = 1.0
    return state


def test_chase_goal_uses_yaw_frame_for_ball_position_and_velocity():
    state = _go2_state()
    yaw = torch.tensor(torch.pi / 2)
    state[:, 3] = torch.cos(yaw / 2)
    state[:, 6] = torch.sin(yaw / 2)
    ball = torch.tensor([[0.0, 2.0, 0.09, 0.0, 1.0, 0.0]])
    goal = chase_goal_yaw_frame(state, ball)
    torch.testing.assert_close(goal[0], torch.tensor([2.0, 0.0, -0.20, 1.0, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_chase_approach_reward_is_signed_and_tag_shaping_is_finite():
    toward = _go2_state()
    away = toward.clone()
    toward[:, 19] = 0.3
    away[:, 19] = -0.3
    ball = torch.tensor([[1.0, 0.0, 0.09, 0.0, 0.0, 0.0]])
    cfg = ChaseBallRewardCfg()
    action = torch.zeros(1, 12)
    terms_toward = chase_ball_reward_terms(toward, ball, action, action, torch.zeros(12), cfg)
    terms_away = chase_ball_reward_terms(away, ball, action, action, torch.zeros(12), cfg)
    assert terms_toward["approach"].item() > 0
    assert terms_away["approach"].item() < 0
    assert terms_toward["near"].item() > 0
    assert torch.isfinite(torch.stack(tuple(terms_toward.values()))).all()
    tagged_ball = torch.tensor([[0.35, 0.0, 0.09, 0.0, 0.0, 0.0]])
    tag = chase_ball_reward_terms(toward, tagged_ball, action, action, torch.zeros(12), cfg)
    assert tag["tag_success"].item() > cfg.tag_weight / 2


def test_chase_task_predicts_ball_and_keeps_future_goal_differentiable():
    task = ChaseBallTask(default_joint_pos=torch.zeros(12), dt=0.02)
    state = _go2_state().requires_grad_()
    ball = torch.tensor([[1.0, 0.0, 0.09, 0.25, 0.0, 0.0]])
    next_ball = task.advance_goal(ball)
    torch.testing.assert_close(next_ball[:, 0], torch.tensor([1.005]))
    goal = task.actor_goal(state, next_ball)
    goal[:, 0].sum().backward()
    assert state.grad is not None and state.grad[0, 0].abs() > 0


def test_chase_reward_rate_matches_imagination_dt_and_tag_is_a_distinct_terminal_signal():
    task = ChaseBallTask(default_joint_pos=torch.zeros(12), dt=0.02)
    state = _go2_state()
    action = torch.zeros(1, 12)
    ball = torch.tensor([[1.0, 0.0, 0.09, 0.1, 0.0, 0.0]])
    ball_next = task.advance_goal(ball)
    raw = chase_ball_reward_terms(state, ball_next, action, action, torch.zeros(12), task.reward_cfg)
    scaled = task.reward_terms(state, state, ball, action, action)
    for name, value in raw.items():
        torch.testing.assert_close(scaled[name], value * task.dt)

    safe_near = torch.tensor([[0.20, 0.0, 0.09, 0.0, 0.0, 0.0]])
    p_safe_far = task.termination_probability(state, ball)
    p_safe_tag = task.termination_probability(state, safe_near)
    fallen = state.clone()
    fallen[:, 2] = task.fall_height - 0.05
    p_fall = task.termination_probability(fallen, ball)
    assert float(p_safe_far) < 0.01
    assert float(p_safe_tag) > 0.9
    assert float(p_fall) > 0.9
    assert torch.isfinite(task.termination_logits(state, safe_near)).all()


def test_direct_chase_actor_uses_six_dimensional_goal():
    policy = Policy(PolicyCfg(
        decoder_hidden=(32,), encoder_hidden=(32,), num_joints=12,
        history=10, goal_dim=6, goal_encoding="direct",
        goal_interface="chase_ball", goal_term_sizes=(3, 3), goal_term_scale=(1.0, 1.0),
        action_output="tanh",
    ))
    obs = torch.zeros(2, 420)
    goal0 = torch.zeros(2, 6)
    goal1 = goal0.clone()
    goal1[:, 0] = 1.0
    with torch.no_grad():
        a0 = policy(obs, goal0)
        a1 = policy(obs, goal1)
    assert a0.shape == (2, 12)
    assert torch.max(torch.abs(a1 - a0)) > 0


def test_chase_env_records_external_ball_goal_and_computes_real_reward():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.commands import ChaseBallCommand

    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="go2", task_name="chase_ball", num_envs=8,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=True, replay_capacity=256,
        chase_fixed_stage=0, seed=44,
    ), device="cuda:0")
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("chase_ball")
        assert isinstance(command, ChaseBallCommand)
        assert obs["policy"].shape == (8, 420)
        assert obs["goal"].shape == (8, 6)
        goal_at_collection = command.goal.clone()
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(8, 12, device=env.device))
        buffer = env.recorder_manager.get_term("replay").buffer
        assert torch.isfinite(reward).all()
        assert not (terminated | truncated).any()
        torch.testing.assert_close(buffer.goal[:, 0], goal_at_collection)
        assert torch.isfinite(obs["goal"]).all()
    finally:
        env.close()


def test_full_body_reference_schema_matches_268_features_per_frame():
    frames, bodies, joints = 2, 14, 29
    ref = RefState(
        torch.zeros(frames, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(frames, -1),
        torch.full((frames, joints), 0.2), torch.zeros(frames, 3),
        torch.zeros(frames, 3), torch.full((frames, joints), 0.1),
    )
    body_pos = torch.zeros(frames, bodies, 3)
    body_pos[..., 2] = 0.8
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    body_motion = torch.zeros_like(body_pos)
    body_states = RefBodies(body_pos, body_quat, body_motion, body_motion)
    goal = full_body_goal_features(ref, body_states, torch.zeros(joints))
    assert goal.shape == (frames, 268)
    torch.testing.assert_close(goal[:, :joints], torch.full((frames, joints), 2.0 / 3.0))
    torch.testing.assert_close(goal[:, joints:2*joints], torch.full((frames, joints), 0.02))
    # Root body height is absolute z, scaled by 0.3, as in Breezewrf's legacy schema.
    torch.testing.assert_close(goal[:, 2*joints + 2], torch.full((frames,), 0.8 / 0.3))
