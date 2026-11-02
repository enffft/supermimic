from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

AMP_ROOT = Path("/home/yifeiy/AMP-LOCO-M1")
CHECKPOINT = AMP_ROOT / "logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt"
EXAMPLE = AMP_ROOT / "logs/motion_prior/cvae_v1_strict_fcl_2272/inference_examples/L_forward.npz"


@pytest.fixture(scope="module")
def prior_pair():
    from supermimic.motion_prior import load_m1_motion_prior

    return (
        load_m1_motion_prior(CHECKPOINT, device="cpu", input_compatibility="legacy_v1"),
        load_m1_motion_prior(
            CHECKPOINT, device="cpu", input_compatibility="train_constant_compat_v1"
        ),
    )


def test_amp_L_forward_sample_regression_and_unchanged_buffers(prior_pair):
    legacy, compat = prior_pair
    sample = np.load(EXAMPLE)
    features = torch.from_numpy(sample["state_history"]).float()[None]
    goal = torch.from_numpy(sample["task_goal"]).float()[None]
    expected = torch.from_numpy(sample["reference_mean"]).float()
    legacy_state = {key: value.clone() for key, value in legacy.state_dict().items()}
    compat_state = {key: value.clone() for key, value in compat.state_dict().items()}
    input_before = features.clone()
    with torch.inference_mode():
        old_output = legacy.forward_features(features, goal)[0]
        compatible_output = compat.forward_features(features, goal)[0]
    torch.testing.assert_close(old_output, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(compatible_output, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(features, input_before, atol=0.0, rtol=0.0)
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, legacy_state[key], atol=0.0, rtol=0.0)
    for key, value in compat.state_dict().items():
        torch.testing.assert_close(value, compat_state[key], atol=0.0, rtol=0.0)
    assert compat.source["sha256"] == legacy.source["sha256"]
    assert compat.source["input_compatibility"] == "train_constant_compat_v1"


@pytest.mark.parametrize("state_index", [30, 48])
def test_q6_qd6_tiny_std_perturbation_is_projected_only_for_prior_input(prior_pair, state_index):
    legacy, compat = prior_pair
    sample = np.load(EXAMPLE)
    base = torch.from_numpy(sample["state_history"]).float()[None]
    goal = torch.tensor([[-0.3, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)
    changed = base.clone()
    changed[:, :, state_index] = 1.0e-4
    with torch.inference_mode():
        old = legacy.forward_features(changed, goal)
        projected = compat.forward_features(changed, goal)
        baseline = compat.forward_features(base, goal)
    condition = torch.cat((changed.flatten(1), goal), dim=-1)
    old_z = (condition - legacy.x_mean) / legacy.x_std
    assert old_z.abs().max() >= 99.0
    torch.testing.assert_close(projected, baseline, atol=0.0, rtol=0.0)
    assert torch.isfinite(old).all() and torch.isfinite(projected).all()
    assert torch.equal(
        changed[:, :, state_index], torch.full_like(changed[:, :, state_index], 1.0e-4)
    )


def test_compatibility_mode_is_explicit_and_does_not_mutate_normalization():
    from supermimic.motion_prior import load_m1_motion_prior

    compat = load_m1_motion_prior(
        CHECKPOINT, device="cpu", input_compatibility="train_constant_compat_v1"
    )
    with pytest.raises(ValueError, match="input_compatibility"):
        load_m1_motion_prior(CHECKPOINT, device="cpu", input_compatibility="silent_std_fix")
    assert compat.input_compatibility == "train_constant_compat_v1"
    assert (
        compat.source["sha256"]
        == "1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d"
    )
    assert float(compat.x_std[30]) == pytest.approx(1e-6)


def test_latest_heading_relative_yaw_feature_is_written_as_analytic_zero():
    from supermimic.kinematics.rotations import quat_from_rotvec
    from supermimic.motion_prior.features import state_history_to_cvae53
    from supermimic.robots.spec import M1_CARRY_DEFAULT

    state = torch.zeros(1, 4, 57)
    state[..., 2] = 0.3892858
    state[..., 3:7] = quat_from_rotvec(torch.tensor([[0.17, -0.21, 0.43]])).expand(1, 4, 4)
    state[..., 7:29] = torch.tensor(M1_CARRY_DEFAULT)
    features = state_history_to_cvae53(state)
    assert torch.equal(features[:, -1, 2], torch.zeros(1))
