from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from supermimic.motion_prior import load_m1_motion_prior

CHECKPOINT = Path(
    "/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt"
)
EXAMPLE = Path(
    "/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_strict_fcl_2272/inference_examples/L_forward.npz"
)


def test_same_l_history_goal_changes_output_but_root_displacement_response_is_small():
    prior = load_m1_motion_prior(
        CHECKPOINT, device="cpu", input_compatibility="train_constant_compat_v1"
    )
    sample = np.load(EXAMPLE)
    features = torch.from_numpy(sample["state_history"]).float()[None]
    rows = {}
    with torch.inference_mode():
        for vx in (-0.3, -0.2, 0.0, 0.2, 0.3):
            goal = torch.tensor([[vx, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)
            rows[vx] = prior.forward_features(features, goal)[0]
    response = float((rows[0.3][-1, 0] - rows[-0.3][-1, 0]).abs())
    assert all(torch.isfinite(value).all() for value in rows.values())
    assert response > 0.0
    # This test deliberately documents, rather than hides, the weak response
    # measured on the frozen source example; it is not a locomotion success gate.
    assert response < 0.01


def test_action_contract_v3_resume_does_not_silently_accept_pose_change():
    from supermimic.robots.spec import get_robot_spec

    carry = get_robot_spec("m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v3")
    with np.testing.assert_raises_regex(ValueError, "outside v3 target bounds"):
        get_robot_spec("m1_piper", m1_pose_profile="legacy_manipulation", m1_action_contract="v3")
    assert carry.action_contract_version == "m1_mixed_action_v3"
