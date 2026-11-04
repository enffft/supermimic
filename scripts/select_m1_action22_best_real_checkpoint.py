#!/usr/bin/env python3
"""Select or roll back a full V5 Actor checkpoint using paired real evaluations.

Run this after one finite training phase. The selected checkpoint includes its
matching optimizer states, so a rejected phase resumes from the prior accepted
Actor *and* optimizer state. Inputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch


EXPECTED_FORMAT = "supermimic_m1_action22_reference_v2"
EXPECTED_VERSION = 5
EVAL_CONTRACT_KEYS = (
    "task_type", "controller", "actor_residual_alpha", "seed", "num_envs",
    "steps", "domain_randomization", "obs_noise", "action_noise",
    "initial_condition_sha256", "fixed_group_names",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v5_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != EXPECTED_FORMAT or checkpoint.get("format_version") != EXPECTED_VERSION:
        raise ValueError(f"{path} is not an Action22 bounded-residual V5 checkpoint")
    required = ("actor", "actor_optimizer", "wm", "wm_optimizer", "motion_prior", "prior_optimizer")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"{path} cannot be safely resumed; missing full training state: {missing}")
    return checkpoint


def _resume_contract(checkpoint: dict) -> dict:
    keys = (
        "robot_identity", "robot_model_sha256", "action_mapping", "prior_schema_version",
        "prior_source", "prior_mode", "prior_input_compatibility", "actor_schema",
        "actor_input_contract", "reset_contract", "reference_adapter_schema",
    )
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise ValueError(f"V5 checkpoint is missing selection contract fields: {missing}")
    return json.loads(json.dumps({key: checkpoint[key] for key in keys}, sort_keys=True))


def _load_evaluation(path: Path, *, checkpoint_path: Path, task: str) -> tuple[dict, dict]:
    evaluation = json.loads(path.read_text())
    contract = evaluation.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{path} lacks a machine-readable fixed evaluation_contract")
    normalized = {key: contract.get(key, evaluation.get(key)) for key in EVAL_CONTRACT_KEYS}
    if normalized["task_type"] != task or normalized["controller"] != "actor":
        raise ValueError(f"{path} must evaluate task {task} with controller=actor")
    if float(normalized["actor_residual_alpha"] or -1) != 1.0:
        raise ValueError("best-real selection requires the unblended Actor (alpha=1)")
    if normalized["domain_randomization"] is not False or normalized["obs_noise"] is not False:
        raise ValueError("best-real selection requires DR and observation noise disabled")
    if float(normalized["action_noise"] or 0) != 0.0:
        raise ValueError("best-real selection requires action noise disabled")
    if not normalized["initial_condition_sha256"] or not normalized["fixed_group_names"]:
        raise ValueError("best-real selection requires a hashed initial condition and fixed group list")
    if not isinstance(normalized["seed"], int) or int(normalized["num_envs"] or 0) < 1 or int(normalized["steps"] or 0) < 1:
        raise ValueError("evaluation contract must fix seed, num_envs and steps")
    saved_path = evaluation.get("checkpoint")
    if saved_path is not None and Path(saved_path).expanduser().resolve() != checkpoint_path.resolve():
        raise ValueError(f"evaluation {path} names a different checkpoint: {saved_path}")
    groups = evaluation.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"evaluation {path} has no task result groups")
    return evaluation, normalized


def _metrics(evaluation: dict, task: str) -> dict:
    groups = evaluation["groups"]
    required = ("stop_id", "forward_id", "backward_id") if task == "L" else (
        "forward", "side", "up", "combined",
    )
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"{task} evaluation is missing fixed groups: {missing}")
    selected = [groups[name] for name in required]
    fall_rates = [float(group.get("fall_rate", 0.0)) for group in selected]
    tilt = [float(group.get("base_tilt_mean_rad", 0.0)) for group in selected]
    height = [float(group.get("base_height_mean_m", 0.0)) for group in selected]
    if task == "L":
        errors = [float(groups[name]["base_vx_rmse_m_s"]) for name in required]
        wrong_direction = float(groups["forward_id"].get("positive_command_negative_vx_fraction", 0.0))
        stop_drift = abs(float(groups["stop_id"].get("signed_forward_displacement_mean_m", 0.0)))
        score = sum(errors) / len(errors) + 0.25 * wrong_direction + 0.25 * stop_drift
        hold = None
    else:
        end_distance = [float(group["end_window_ee_distance_m"]) for group in selected]
        hold_rates = [float(group.get("stable_hold_success_rate", 0.0)) for group in selected]
        score = sum(end_distance) / len(end_distance) - 0.25 * sum(hold_rates) / len(hold_rates)
        hold = sum(hold_rates) / len(hold_rates)
        wrong_direction = None
        stop_drift = None
    return {
        "score_lower_is_better": score,
        "max_observed_fall_rate": max(fall_rates),
        "mean_base_tilt_rad": sum(tilt) / len(tilt),
        "min_base_height_m": min(height),
        "mean_stable_hold_rate": hold,
        "positive_command_negative_vx_fraction": wrong_direction,
        "stop_abs_displacement_m": stop_drift,
        "groups": list(required),
        "episode_counts": {name: int(groups[name].get("episode_count", 0)) for name in required},
        "fall_rates": {name: float(groups[name].get("fall_rate", 0.0)) for name in required},
    }


def _safety_not_worse(candidate: dict, accepted: dict, *, tolerance: float) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["max_observed_fall_rate"] > accepted["max_observed_fall_rate"] + tolerance:
        reasons.append("fall_rate_regressed")
    if candidate["mean_base_tilt_rad"] > accepted["mean_base_tilt_rad"] + 0.10:
        reasons.append("base_tilt_regressed")
    if candidate["min_base_height_m"] < accepted["min_base_height_m"] - 0.03:
        reasons.append("base_height_regressed")
    old_wrong = accepted.get("positive_command_negative_vx_fraction")
    new_wrong = candidate.get("positive_command_negative_vx_fraction")
    if old_wrong is not None and new_wrong is not None and new_wrong > old_wrong + 0.02:
        reasons.append("forward_direction_regressed")
    old_hold, new_hold = accepted.get("mean_stable_hold_rate"), candidate.get("mean_stable_hold_rate")
    if old_hold is not None and new_hold is not None and new_hold < old_hold - tolerance:
        reasons.append("stable_hold_regressed")
    return not reasons, reasons


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selected checkpoint: {destination}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        shutil.copystat(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def select_best_real(*, task: str, accepted_checkpoint: Path, accepted_evaluation: Path,
                     candidate_checkpoint: Path, candidate_evaluation: Path,
                     output_dir: Path, score_tolerance: float = 1.0e-4,
                     safety_tolerance: float = 1.0e-6) -> dict:
    if task not in ("L", "M"):
        raise ValueError("task must be L or M")
    accepted_checkpoint = accepted_checkpoint.expanduser().resolve()
    candidate_checkpoint = candidate_checkpoint.expanduser().resolve()
    if not accepted_checkpoint.is_file() or not candidate_checkpoint.is_file():
        raise FileNotFoundError("accepted and candidate V5 checkpoints must both exist")
    # Keep only small contract metadata while validating large optimizer states.
    accepted_data = _load_v5_checkpoint(accepted_checkpoint)
    accepted_contract_saved = _resume_contract(accepted_data)
    del accepted_data
    candidate_data = _load_v5_checkpoint(candidate_checkpoint)
    candidate_contract_saved = _resume_contract(candidate_data)
    del candidate_data
    if accepted_contract_saved != candidate_contract_saved:
        raise ValueError("real checkpoint selection cannot compare different robot, prior, reset or action contracts")
    accepted_eval, accepted_contract = _load_evaluation(
        accepted_evaluation, checkpoint_path=accepted_checkpoint, task=task,
    )
    candidate_eval, candidate_contract = _load_evaluation(
        candidate_evaluation, checkpoint_path=candidate_checkpoint, task=task,
    )
    if accepted_contract != candidate_contract:
        raise ValueError(f"evaluation conditions differ: {accepted_contract} != {candidate_contract}")
    accepted_metrics = _metrics(accepted_eval, task)
    candidate_metrics = _metrics(candidate_eval, task)
    safety_ok, safety_reasons = _safety_not_worse(
        candidate_metrics, accepted_metrics, tolerance=safety_tolerance,
    )
    score_improved = candidate_metrics["score_lower_is_better"] < accepted_metrics["score_lower_is_better"] - score_tolerance
    chosen = candidate_checkpoint if safety_ok and score_improved else accepted_checkpoint
    status = "accepted" if chosen == candidate_checkpoint else "rejected_rollback"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "ckpt_best_real.pt"
    _atomic_copy(chosen, best_path)
    manifest = {
        "selector": "m1_action22_best_real_v1",
        "task_type": task,
        "status": status,
        "decision_reasons": safety_reasons if not safety_ok else ([] if score_improved else ["real_task_score_not_improved"]),
        "evaluation_contract": accepted_contract,
        "accepted_checkpoint": str(accepted_checkpoint),
        "accepted_checkpoint_sha256": _sha256(accepted_checkpoint),
        "accepted_metrics": accepted_metrics,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "candidate_metrics": candidate_metrics,
        "selected_resume_checkpoint": str(best_path),
        "selected_checkpoint_sha256": _sha256(best_path),
        "resume_restores_actor_and_optimizer": True,
    }
    (output_dir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-type", choices=("L", "M"), required=True)
    parser.add_argument("--accepted-checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()
    result = select_best_real(
        task=args.task_type, accepted_checkpoint=args.accepted_checkpoint,
        accepted_evaluation=args.accepted_evaluation,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_evaluation=args.candidate_evaluation,
        output_dir=args.output_dir, score_tolerance=args.score_tolerance,
    )
    print(json.dumps({key: result[key] for key in ("status", "selected_resume_checkpoint", "decision_reasons")}, indent=2))


if __name__ == "__main__":
    main()
