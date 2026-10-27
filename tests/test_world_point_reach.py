from __future__ import annotations

import torch

from supermimic.imagination.tasks import WorldPointReachTask
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.world_point_reward import WorldPointRobotParams, tcp_position_world
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _task_and_state():
    robot = get_robot_spec("m1_piper")
    fk = G1Kinematics(load_robot_spec_model("m1_piper"), device="cpu")
    indices = (fk.body_index("piper_x_link7"), fk.body_index("piper_x_link8"))
    q0 = torch.tensor(robot.default_joint_pos)
    limits = torch.tensor([[-3.2, 3.2]] * robot.num_joints)
    params = WorldPointRobotParams(q0, limits, dt=robot.control_dt)
    task = WorldPointReachTask(params, fk, indices, dt=robot.control_dt)
    state = torch.zeros(1, robot.state_dim)
    state[:, 2] = robot.default_base_height
    state[:, 3] = 1.0
    state[:, 7:7 + robot.num_joints] = q0
    return robot, fk, indices, task, state


def test_world_target_remains_fixed_while_actor_goal_tracks_base_pose():
    robot, fk, indices, task, state = _task_and_state()
    tcp = tcp_position_world(state, fk, indices)
    target_world = tcp + torch.tensor([[0.25, -0.10, 0.05]])
    goal0 = task.actor_goal(state, target_world)
    moved = state.clone()
    moved[:, 0] += 0.10
    goal1 = task.actor_goal(moved, target_world)
    torch.testing.assert_close(goal1[:, 0], goal0[:, 0] - 0.10)
    torch.testing.assert_close(goal1[:, 1:], goal0[:, 1:])


def test_world_point_reset_samples_after_environment_origins_are_applied():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.commands import WorldPointCommand

    args = SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="world_point_reach",
        num_envs=32, action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=True, replay_capacity=64,
        seed=123,
    )
    env = make_env(args, device="cuda:0")
    try:
        env.reset()
        command = env.command_manager.get_term("world_point")
        assert isinstance(command, WorldPointCommand)
        assert env.scene.env_origins.abs().amax() > 2.0
        # The sampler's largest configured body-frame offset is <1.5m. A stale
        # pre-forward target would include the grid origin and grow with the grid.
        assert command.metrics["target_distance"].max() < 1.6
        command.target_position_world[0] += 10.0
        dt = torch.full((env.num_envs,), float(env.step_dt), device=env.device)
        dt[0] = 0.0  # mjlab marks automatically reset environments with zero dt.
        command.compute(dt)
        assert command.metrics["target_distance"][0] < 1.6
    finally:
        env.close()


def test_world_point_reward_gradient_crosses_world_model_fk_to_legs_wheels_and_arm():
    robot, fk, indices, task, state = _task_and_state()
    heads = AnalyticHeads(
        fk, robot.tracked_body_names, robot.anchor_body_name, robot.root_body_name,
        task.params.soft_joint_limits, task.params.default_joint_pos,
        default_base_height=robot.default_base_height,
    )
    wm = WorldModel(
        WorldModelCfg(
            hidden=(64,), term_hidden=(8,), num_joints=22, action_dim=22, xi_dim=26,
            state_spec=StateSpecCfg(num_joints=22, dt=robot.control_dt),
        ),
        heads,
        "cpu",
    )
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(mean=0.0, std=1e-3)
    tcp = tcp_position_world(state, fk, indices)
    target_world = tcp + torch.tensor([[0.22, 0.03, 0.04]])
    action = torch.zeros(1, 22, requires_grad=True)
    xi = torch.zeros(1, 26)
    s_next = wm.step(state, action, xi)
    terms = task.reward_terms(state, s_next, target_world, action, torch.zeros_like(action))
    (-terms["tcp_distance"] - terms["tcp_approach"] - terms["tcp_near"]).sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad[:, :16].abs().sum() > 0  # leg/wheel channels
    assert action.grad[:, 16:].abs().sum() > 0  # six Piper arm joints
    assert s_next.requires_grad
