from __future__ import annotations

import torch

from supermimic.world_model.normalization import RunningNorm
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.kinematics.g1_fk import G1Kinematics


def test_running_norm_is_explicit_fixed_and_differentiable():
    norm = RunningNorm(3, min_std=0.1)
    x = torch.tensor([[1.0, 4.0, 7.0], [3.0, 8.0, 11.0]])
    norm.update(x)
    assert int(norm.count) == 2
    x_grad = x.clone().requires_grad_()
    y = norm.normalize(x_grad)
    torch.testing.assert_close(norm.denormalize(y), x)
    y.square().sum().backward()
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()
    before = int(norm.count)
    norm.normalize(torch.zeros(1, 3))
    assert int(norm.count) == before


def test_state_spec_delta_target_inverts_fixed_scale_integration():
    spec = StateSpec(StateSpecCfg(num_joints=12, dt=0.02), "cpu")
    s = torch.zeros(4, 37)
    s[:, 2] = 0.29
    s[:, 3] = 1.0
    s[:, 7:19] = torch.randn(4, 12) * 0.2
    s[:, 19:22] = torch.randn(4, 3) * 0.1
    s[:, 22:25] = torch.randn(4, 3) * 0.1
    s[:, 25:37] = torch.randn(4, 12) * 0.2
    delta = torch.randn(4, 36) * 0.2
    s_next = spec.integrate(s, delta)
    recovered = spec.delta_target(s, s_next)
    torch.testing.assert_close(recovered, delta, atol=2e-5, rtol=2e-5)


def test_old_fixed_world_model_has_no_normalizer_state_and_enabled_model_has_stats():
    robot = get_robot_spec("m1_piper")
    model = load_robot_spec_model(robot.name)
    fk = G1Kinematics(model)
    heads = AnalyticHeads(
        fk, robot.tracked_body_names, robot.anchor_body_name, robot.root_body_name,
        torch.tensor([[-3.0, 3.0]] * robot.num_joints),
        torch.tensor(robot.default_joint_pos), default_base_height=robot.default_base_height,
    )
    base = dict(
        hidden=(32,), term_hidden=(8,), num_joints=22, action_dim=22, xi_dim=26,
        state_spec=StateSpecCfg(num_joints=22),
    )
    fixed = WorldModel(WorldModelCfg(**base), heads, "cpu")
    assert not any(k.startswith("normalizer.") for k in fixed.state_dict())
    normalized = WorldModel(WorldModelCfg(**base, running_normalization=True), heads, "cpu")
    assert any(k.startswith("normalizer.input.") for k in normalized.state_dict())
