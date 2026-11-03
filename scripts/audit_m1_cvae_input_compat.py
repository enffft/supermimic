#!/usr/bin/env python3
"""CPU-only regression and perturbation audit for the frozen AMP M1 CVAE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.cvae import _Q6_CONDITION_INDICES, _QD6_CONDITION_INDICES

DEFAULT_AMP = Path("/home/yifeiy/AMP-LOCO-M1")
DEFAULT_CHECKPOINT = (
    DEFAULT_AMP / "logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt"
)
DEFAULT_EXAMPLE = (
    DEFAULT_AMP / "logs/motion_prior/cvae_v1_strict_fcl_2272/inference_examples/L_forward.npz"
)


def summarize(reference: np.ndarray) -> dict[str, object]:
    return {
        "root_dx_terminal_0p2s_m": float(reference[-1, 0]),
        "root_vx_displacement_proxy_m_s": float(reference[-1, 0] / 0.2),
        "wheel_rate_mean_rad_s": reference[:, 27:31].mean(axis=0).tolist(),
        "wheel_rate_first_rad_s": reference[0, 27:31].tolist(),
        "leg_joint_terminal_rad": reference[-1, 9:21].tolist(),
        "piper_joint_terminal_rad": reference[-1, 21:27].tolist(),
        "finite": bool(np.isfinite(reference).all()),
    }


def prior_result(prior, history: np.ndarray, goal: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        value = prior.forward_features(
            torch.as_tensor(history[None], dtype=torch.float32),
            torch.as_tensor(goal[None], dtype=torch.float32),
            mode="prior_mean",
        )
    return value[0].cpu().numpy()


def normalized_peak(prior, history: np.ndarray, goal: np.ndarray, compatibility: bool) -> float:
    condition = np.concatenate((history.reshape(-1), goal))[None]
    mean = prior.x_mean.cpu().numpy()
    std = prior.x_std.cpu().numpy()
    if compatibility:
        indices = np.asarray((*_Q6_CONDITION_INDICES, *_QD6_CONDITION_INDICES), dtype=np.int64)
        condition[:, indices] = mean[indices]
    return float(np.abs((condition - mean) / std).max())


def run(args: argparse.Namespace) -> dict[str, object]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only CVAE audit must not initialize CUDA")
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.amp_root))
    from motion_prior.cvae_inference import load_cvae_checkpoint

    example = np.load(args.example)
    base_history = np.asarray(example["state_history"], dtype=np.float32)
    base_goal = np.asarray(example["task_goal"], dtype=np.float32)
    expected = np.asarray(example["reference_mean"], dtype=np.float32)
    before = base_history.copy()
    amp = load_cvae_checkpoint(args.checkpoint, device="cpu")
    amp_output = amp.predict(base_history, base_goal, mode="prior_mean")[0].numpy()
    legacy = load_m1_motion_prior(
        args.checkpoint, device="cpu", expected_sha256=None, input_compatibility="legacy_v1"
    )
    compat = load_m1_motion_prior(
        args.checkpoint,
        device="cpu",
        expected_sha256=None,
        input_compatibility="train_constant_compat_v1",
    )
    legacy_output = prior_result(legacy, base_history, base_goal)
    compat_output = prior_result(compat, base_history, base_goal)

    goals = {}
    for vx in (-0.3, -0.2, 0.0, 0.2, 0.3):
        goal = base_goal.copy()
        goal[0] = vx
        goals[f"{vx:+.1f}"] = {
            "legacy_v1": summarize(prior_result(legacy, base_history, goal)),
            "train_constant_compat_v1": summarize(prior_result(compat, base_history, goal)),
        }
    for mode in ("legacy_v1", "train_constant_compat_v1"):
        output_key = "legacy_v1" if mode == "legacy_v1" else "train_constant_compat_v1"
        zero = np.asarray(goals["+0.0"][output_key]["root_dx_terminal_0p2s_m"])
        for row in goals.values():
            row[output_key]["root_dx_difference_from_zero_m"] = float(
                row[output_key]["root_dx_terminal_0p2s_m"] - zero
            )

    perturbations = {
        "q6_rad": (-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01),
        "qd6_rad_s": (-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01),
    }
    scans: dict[str, object] = {}
    for feature, values in perturbations.items():
        rows = []
        for value in values:
            history = base_history.copy()
            state_index = 30 if feature == "q6_rad" else 48
            history[:, state_index] = value
            rows.append(
                {
                    "value": value,
                    "legacy_v1": {
                        **summarize(
                            prior_result(
                                legacy,
                                history,
                                np.asarray([0.3, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
                            )
                        ),
                        "max_abs_zscore": normalized_peak(
                            legacy,
                            history,
                            np.asarray([0.3, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
                            False,
                        ),
                    },
                    "train_constant_compat_v1": {
                        **summarize(
                            prior_result(
                                compat,
                                history,
                                np.asarray([0.3, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
                            )
                        ),
                        "max_abs_zscore": normalized_peak(
                            compat,
                            history,
                            np.asarray([0.3, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
                            True,
                        ),
                    },
                }
            )
        scans[feature] = rows

    pose_ablation = {}
    for pose_name, arm_pose in (
        ("L_bank_carry_history", (0.0, 0.5, -0.4, 0.3, 0.0, 0.0)),
        ("legacy_manipulation_reset", (0.0, 2.55, -1.5, -0.8, 0.0, 0.0)),
        ("dataset_carry_reset", (0.0, 0.5, -0.4, 0.3, 0.0, 0.0)),
    ):
        history = base_history.copy()
        history[:, 25:31] = np.asarray(arm_pose, dtype=np.float32)
        history[:, 43:49] = 0.0
        pose_ablation[pose_name] = {}
        for vx in (-0.3, 0.0, 0.3):
            goal = np.asarray([vx, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
            pose_ablation[pose_name][f"vx={vx:+.1f}"] = {
                "legacy_v1": summarize(prior_result(legacy, history, goal)),
                "train_constant_compat_v1": summarize(prior_result(compat, history, goal)),
            }

    goal_response = {
        policy: max(
            abs(float(goals[f"{vx:+.1f}"][policy]["root_dx_difference_from_zero_m"]))
            for vx in (-0.3, -0.2, 0.2, 0.3)
        )
        for policy in ("legacy_v1", "train_constant_compat_v1")
    }
    goal_threshold_m = 0.01
    goal_status = {
        policy: (
            "PRETRAINED_GOAL_CONDITIONING_LIMITATION"
            if response <= goal_threshold_m
            else "measurable_response_on_tested_history"
        )
        for policy, response in goal_response.items()
    }
    return {
        "schema": "m1_cvae_input_compat_audit_v1",
        "device": "cpu",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": legacy.source["sha256"],
        "source_weights_unchanged": True,
        "runtime_input_policy": {
            "legacy_v1": "No input projection; exactly the old normalization path.",
            "train_constant_compat_v1": "Project only the four history q6 and four qd6 state features to checkpoint x_mean.",
            "projected_state53_indices": sorted([30, 83, 136, 189, 48, 101, 154, 207]),
            "normalization_buffers_modified": False,
            "raw_state_history_modified": False,
        },
        "normalization": {
            "q6_std": float(legacy.x_std[_Q6_CONDITION_INDICES[-1]].item()),
            "qd6_std": float(legacy.x_std[_QD6_CONDITION_INDICES[-1]].item()),
            "heading_relative_latest_row1_col0_index_161_std": float(legacy.x_std[161].item()),
            "heading_relative_latest_row1_col0_is_analytic_zero": True,
        },
        "amp_L_forward_regression": {
            "example": str(args.example.resolve()),
            "goal_vx_m_s": float(base_goal[0]),
            "source_inference": summarize(amp_output),
            "source_example_reference": summarize(expected),
            "source_vs_saved_example_max_abs": float(np.max(np.abs(amp_output - expected))),
            "supermimic_legacy_vs_source_max_abs": float(
                np.max(np.abs(legacy_output - amp_output))
            ),
            "compat_vs_source_max_abs_at_training_constant_q6": float(
                np.max(np.abs(compat_output - amp_output))
            ),
            "input_history_unchanged": bool(np.array_equal(base_history, before)),
        },
        "same_history_goal_sweep": {
            "rows": goals,
            "max_root_dx_change_vs_zero_m": goal_response,
            "goal_response_diagnostic_threshold_m": goal_threshold_m,
            "goal_conditioning_status": goal_status,
        },
        "q6_qd6_perturbation_scans": scans,
        "pose_alignment_ablation": pose_ablation,
        "limitations": [
            "The compatibility projection restores only the old model input to the bank mean; it does not teach q6/qd6-conditioned motion.",
            "Goal sensitivity is a property of this frozen pretrained checkpoint and its data, not repaired by input compatibility.",
            "CPU inference checks numerical contracts only; no physical dynamics, collision, or deployment safety validation was performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-root", type=Path, default=DEFAULT_AMP)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--example", type=Path, default=DEFAULT_EXAMPLE)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/artifacts/m1_cvae_input_compat_audit.json")
    )
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "source_max_abs_regression": result["amp_L_forward_regression"][
                    "source_vs_saved_example_max_abs"
                ],
                "legacy_vs_amp_max_abs": result["amp_L_forward_regression"][
                    "supermimic_legacy_vs_source_max_abs"
                ],
                "goal_response_dx": result["same_history_goal_sweep"][
                    "max_root_dx_change_vs_zero_m"
                ],
                "perturb_max_zscore_legacy_q6_plus_1e-4": next(
                    row["legacy_v1"]["max_abs_zscore"]
                    for row in result["q6_qd6_perturbation_scans"]["q6_rad"]
                    if row["value"] == 0.0001
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
