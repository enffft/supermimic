"""Explicit single-environment MuJoCo checks; excluded from routine CPU tests."""

import mujoco
import pytest
import torch

from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import EFFORT, JOINTS
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.terrain_plane import terrain_plane_normal

pytestmark = pytest.mark.slow


def test_single_env_mapping_control_reset_and_collision_scan():
    cfg = load_config("terrain")
    cfg.commands["velocity"].spawn_height = 2.0
    env = make_go2w_env(cfg, 1)
    try:
        env.reset()
        model = env.get_playback_model(0)
        assert (model.nq, model.nv, model.nu) == (23, 22, 16)
        assert model.opt.timestep == 0.005
        c = env.command_manager.get_term("velocity")
        action = env.action_manager.get_term("mixed")
        for i, name in enumerate(JOINTS):
            motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert model.actuator_trnid[motor, 0] == joint
            assert model.actuator_ctrlrange[motor, 1] == pytest.approx(EFFORT[i])
        # Isolate actuator sign from rolling contacts: suspend the robot with
        # zero qvel, then apply positive/negative wheel commands for one ctrl dt.
        for sign in (1.0, -1.0):
            env.reset()
            result = env.step(torch.cat((torch.zeros(1, 12), torch.full((1, 4), sign * 0.1)), -1))
            assert (c.state()[5][:, 12:] * sign > 0).all()
            assert torch.isfinite(result.reward).all()
            assert (env.velocity_measured_torque.abs() <= torch.tensor(EFFORT) + 1e-4).all()
        # Pure-controller test against the backend's named actuator controls.
        action.process_actions(torch.ones(1, 16))
        action.apply_actions()
        assert (action.tensor_target.abs() <= torch.tensor(EFFORT)).all()
        env.reset()
        assert (env.current_info()["a_prev"] == 0).all()
        assert (env.current_info()["ep_step"] == 0).all()
        field = c.mdp.terrain
        origins = c.curriculum.origins
        # The compiled collision field and Torch query share exact geometry.
        # Interior slope scanner windows exclude entry/center plateaus.
        for level, slope in ((3, 0.10), (7, 0.20)):
            for terrain_type, direction in ((1, 1), (2, -1)):
                p = origins[level, terrain_type].clone()[None]
                p[:, 0] += 1.1
                p[:, 2] = field.ground_height(p[:, :2]) + 0.42
                normal = terrain_plane_normal(
                    field.local_scan(p, torch.eye(3)[None]), torch.eye(3)[None]
                )
                assert normal[0, 0] * direction < 0
                assert -normal[0, 0] / normal[0, 2] == pytest.approx(direction * slope, abs=0.015)
        for level, height in ((1, 0.03), (4, 0.12), (7, 0.21)):
            for terrain_type, direction in ((3, 1), (4, -1)):
                entry = origins[level, terrain_type][None, :2]
                x = torch.arange(0.1, 2.5, 0.05)
                points = entry[:, None, :] + torch.stack((x, torch.zeros_like(x)), -1)
                h = field.ground_height(points)[0]
                increments = h[1:] - h[:-1]
                # Hfield cell boundaries can split one vertical rise between
                # two samples; verify total plateau-to-plateau change instead.
                low = field.ground_height(entry + entry.new_tensor([[0.55, 0]]))
                high = field.ground_height(entry + entry.new_tensor([[0.85, 0]]))
                assert (high - low).item() == pytest.approx(direction * height, abs=0.006)
                assert torch.isfinite(increments).all()
    finally:
        env.close()
        env.backend.close()
