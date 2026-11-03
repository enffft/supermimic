from __future__ import annotations

import pytest
import torch


def _v3():
    from supermimic.robots.spec import M1_JOINTS, get_robot_spec

    spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v3")
    return (
        spec,
        torch.tensor(spec.default_joint_pos),
        torch.tensor(spec.action_scales),
        torch.tensor(
            [n.endswith("FOOT_JOINT") for n in M1_JOINTS],
            dtype=torch.bool,
        ),
        torch.tensor(spec.position_target_bounds),
    )


def test_m1_v3_action_endpoints_default_and_wheel_velocity_mapping():
    from supermimic.tasks.action_contract import mixed_m1_setpoints

    spec, default, scale, wheel, bounds = _v3()
    limits = torch.tensor([[-4.0, 4.0]] * 22)
    actions = torch.stack((torch.full((22,), -1.0), torch.zeros(22), torch.ones(22)))
    mapped = mixed_m1_setpoints(actions, default, scale, wheel, limits, bounds)
    for index, name in enumerate(spec.joint_names):
        if name.endswith("FOOT_JOINT"):
            torch.testing.assert_close(mapped[0, index], torch.tensor(-5.0))
            torch.testing.assert_close(mapped[1, index], torch.tensor(0.0))
            torch.testing.assert_close(mapped[2, index], torch.tensor(5.0))
        else:
            torch.testing.assert_close(mapped[0, index], bounds[index, 0])
            torch.testing.assert_close(mapped[1, index], default[index])
            torch.testing.assert_close(mapped[2, index], bounds[index, 1])
            assert (
                bounds[index, 0]
                <= mapped[:, index].min()
                <= mapped[:, index].max()
                <= bounds[index, 1]
            )


def test_m1_v3_piecewise_mapping_has_finite_gradient_and_positive_side_at_zero():
    from supermimic.tasks.action_contract import mixed_m1_setpoints

    spec, default, scale, wheel, bounds = _v3()
    action = torch.zeros(1, 22, requires_grad=True)
    limits = torch.tensor([[-4.0, 4.0]] * 22)
    output = mixed_m1_setpoints(action, default, scale, wheel, limits, bounds)
    piper2 = spec.joint_names.index("piper_x_joint2")
    gradient = torch.autograd.grad(output[0, piper2], action)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, piper2] == pytest.approx(float(bounds[piper2, 1] - default[piper2]))


def test_m1_v2_mapping_and_non_m1_action_helpers_remain_unchanged():
    from supermimic.robots.spec import M1_DEFAULT, M1_JOINTS, get_robot_spec
    from supermimic.tasks.action_contract import mixed_m1_setpoints, position_setpoint

    spec = get_robot_spec("m1_piper")
    default = torch.tensor(M1_DEFAULT)
    scale = torch.tensor(spec.action_scales)
    wheel = torch.tensor([n.endswith("FOOT_JOINT") for n in M1_JOINTS])
    limits = torch.tensor([[-4.0, 4.0]] * 22)
    action = torch.linspace(-1, 1, 22).unsqueeze(0)
    result = mixed_m1_setpoints(action, default, scale, wheel, limits)
    expected = torch.where(wheel, action[0] * scale, default + action[0] * scale)
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(
        position_setpoint(torch.tensor([[1.0]]), torch.tensor([0.5]), torch.tensor([0.25])),
        torch.tensor([[0.75]]),
    )


def test_old_m1_joint_checkpoint_is_rejected_for_contract_v3():
    from supermimic.motion_prior.trainer import M1AlternatingTrainer

    trainer = object.__new__(M1AlternatingTrainer)
    with pytest.raises(ValueError, match="Action22-reference checkpoint v4"):
        trainer.load_checkpoint(
            {"format": "supermimic_m1_prior_actor_joint_v2", "format_version": 2},
            expected_source={},
        )
