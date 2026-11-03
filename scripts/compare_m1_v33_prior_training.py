"""Compare the initial and fine-tuned V3.3 Prior on identical CPU inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_m1_v33_cpu_prior import (
    GOALS,
    _assess_reference,
    _contract_bounds,
    _load_state_history,
    _static_history,
    _urdf_limits,
)
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import state_history_to_cvae53
from supermimic.robots.spec import M1_CARRY_DEFAULT, get_robot_spec


def compare(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "supermimic_m1_prior_actor_joint_v3":
        raise ValueError("expected a v3 M1 joint-training Checkpoint")
    source = checkpoint["prior_source"]
    compatibility = checkpoint["prior_input_compatibility"]
    initial = load_m1_motion_prior(
        source["path"], device="cpu", expected_sha256=source["sha256"],
        input_compatibility=compatibility,
    )
    trained = load_m1_motion_prior(
        source["path"], device="cpu", expected_sha256=source["sha256"],
        input_compatibility=compatibility,
    )
    initial.load_state_dict(checkpoint["prior_initial"], strict=True)
    trained.load_state_dict(checkpoint["motion_prior"], strict=True)

    real_history, _, _ = _load_state_history()
    spec = get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    )
    static_high = _static_history(0.47, spec.default_joint_pos)
    static_low = _static_history(0.3892858, M1_CARRY_DEFAULT)
    histories = {
        "static_high_carry": static_high,
        "real_l_middle_history": real_history,
        "static_low_carry_context": static_low,
    }
    _, bounds = _contract_bounds("v33_v3")
    urdf = _urdf_limits()
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "iteration": int(checkpoint["iteration"]),
        "prior_optimizer_steps": int(checkpoint["prior_optimizer_steps"]),
        "source_prior_sha256": source["sha256"],
        "input_compatibility": compatibility,
        "reference_not_dynamically_validated": True,
        "goal_vx_commands_m_s": list(GOALS),
        "comparisons": {},
    }
    for history_name, history in histories.items():
        features = state_history_to_cvae53(history)
        rows = []
        for vx in GOALS:
            goal = torch.tensor([[vx, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
            pair = {"command_vx_m_s": vx}
            for label, prior in (("initial", initial), ("fine_tuned", trained)):
                with torch.inference_mode():
                    future = prior.forward_features(features, goal)
                pair[label] = _assess_reference(future, history[0, -1], bounds, urdf, vx)
            rows.append(pair)
        report["comparisons"][history_name] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare(args.checkpoint)
    encoded = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
