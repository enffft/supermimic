#!/usr/bin/env python3
"""Compare V4 Motion Prior Action22 responses in MuJoCo and its saved World Model.

This is a physical validation tool, not a training command. It collects
deterministic held-out reset states and compares prior-direct references plus
small legal wheel/leg perturbations from the same state and simulator context.
No action-response threshold is considered passed unless the caller supplies
calibrated thresholds. Actor task/safety review remains a separate requirement
for the Prior-unfreeze gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.trainer import module_sha256
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.reference import future31_to_m1_action22
from supermimic.robots.spec import M1_JOINTS
from supermimic.tasks.action_contract import m1_robot_spec_from_checkpoint
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _build_wm(checkpoint: dict, device: str):
    saved = dict(checkpoint["config"]["world_model"])
    state_spec = dict(saved.pop("state_spec"))
    if "wheel_indices" in state_spec:
        state_spec["wheel_indices"] = tuple(state_spec["wheel_indices"])
    wm = WorldModel(
        WorldModelCfg(**saved, state_spec=StateSpecCfg(**state_spec)),
        heads=object(), device=device,
    )
    wm.load_state_dict(checkpoint["wm"], strict=True)
    wm.eval()
    for parameter in wm.parameters():
        parameter.requires_grad_(False)
    return wm


def _action_cases(spec, reference_actions: dict[str, torch.Tensor], device: str):
    cases = dict(reference_actions)
    zero = torch.zeros(1, 10, 22, device=device)
    cases["wheel_stop"] = zero.clone()
    for label, sign in (("wheel_positive", 1.0), ("wheel_negative", -1.0)):
        action = zero.clone()
        for name in spec.wheel_joint_names:
            joint_id = spec.joint_names.index(name)
            scale = spec.action_scales[joint_id]
            action[..., joint_id] = sign * min(1.0, 1.0 / scale)
        cases[label] = action
    for label, sign in (("front_hip_positive", 1.0), ("front_hip_negative", -1.0)):
        action = zero.clone()
        joint_id = spec.joint_names.index("FAR_HIP_JOINT")
        action[..., joint_id] = sign * 0.05
        cases[label] = action
    return cases


def audit(
    checkpoint_path: Path,
    *,
    state_count: int,
    seed: int,
    device: str,
    out: Path | None,
    min_direction_agreement: float | None,
    max_relative_response_error: float | None,
) -> dict:
    if state_count < 1:
        raise ValueError("state_count must be positive")
    if min_direction_agreement is not None and not 0.0 <= min_direction_agreement <= 1.0:
        raise ValueError("min_direction_agreement must lie in [0,1]")
    if max_relative_response_error is not None and max_relative_response_error < 0:
        raise ValueError("max_relative_response_error must be nonnegative")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "supermimic_m1_action22_reference_v1":
        raise ValueError("real action-response audit requires an Action22-reference v4 checkpoint")
    spec = m1_robot_spec_from_checkpoint(checkpoint)
    device_obj = torch.device(device)
    wm = _build_wm(checkpoint, device)
    computed_wm_sha256 = module_sha256(wm)
    if checkpoint.get("world_model_sha256") != computed_wm_sha256:
        raise ValueError(
            "loaded World Model weights do not match the checkpoint's saved world_model_sha256"
        )
    prior_source = checkpoint["prior_source"]
    prior = load_m1_motion_prior(
        prior_source["path"], device=device, expected_sha256=prior_source["sha256"],
        input_compatibility=checkpoint["prior_input_compatibility"],
    )
    prior.load_state_dict(checkpoint["motion_prior"], strict=True)
    prior.eval()

    sampler = checkpoint.get("task_sampler_state", {})
    vx_range = tuple(sampler.get("vx_range", (-0.09, 0.145)))
    forward_command = min(0.12, float(vx_range[1]))
    backward_command = max(-0.08, float(vx_range[0]))

    # Eight paired trials share the same state/xi at every held-out seed:
    # stop, forward/backward Prior references, ± wheel response, and ± leg.
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        m1_pose_profile={
            "m1_pose_dataset_v3_3_l_high_carry_v1": "dataset_v3_3_l_high_carry",
            "m1_pose_dataset_carry_v1": "dataset_carry",
            "m1_pose_legacy_manipulation_v1": "legacy_manipulation",
        }[spec.default_pose_profile],
        m1_action_contract="v3", num_envs=1, history_length=5,
        episode_length_s=10.0, action_noise_std=0.0,
        obs_noise=False, reset_noise=True, domain_randomization=True,
        record_transitions=False, replay_capacity=16, seed=seed,
        m1_joint_task_type="L", m1_joint_vx_range=vx_range,
        m1_joint_vy_range=(0.0, 0.0), m1_joint_wz_range=(0.0, 0.0),
    ), device=device)
    state_io = StateIO(env)
    context = SimContext(env, spec.foot_geom_regex, entity_name="robot")
    command = env.command_manager.get_term("m1_prior_joint")
    action_cases = _action_cases(spec, {}, device)
    steps_at = (1, 5, 10)
    rows: dict[str, dict[str, list[float]]] = {}
    paired_start_errors: list[float] = []
    paired_xi_errors: list[float] = []
    fall_counts: dict[str, int] = {}
    termination_samples = 0

    try:
        for state_i in range(state_count):
            sample_seed = seed + state_i
            state_by_case: dict[str, torch.Tensor] = {}
            xi_by_case: dict[str, torch.Tensor] = {}
            for scenario, vx in (("prior_stop", 0.0), ("prior_forward", forward_command), ("prior_backward", backward_command),
                                 ("wheel_stop", 0.0), ("wheel_positive", 0.0), ("wheel_negative", 0.0),
                                 ("front_hip_positive", 0.0), ("front_hip_negative", 0.0)):
                env.reset(seed=sample_seed)
                state = state_io.read()
                xi = context.read()
                state_by_case[scenario] = state.clone()
                xi_by_case[scenario] = xi.clone()
                command.task_type[:] = 0
                command.velocity_command_body[:] = 0.0
                command.velocity_command_body[:, 0] = vx
                command.target_position_world[:] = 0.0

                if scenario.startswith("prior_"):
                    history = state[:, None].expand(-1, 4, -1).clone()
                    task = torch.zeros(1, dtype=torch.long, device=device_obj)
                    goal = task_goal8(state, task, command.velocity_command_body, torch.zeros_like(command.target_position_world))
                    with torch.inference_mode():
                        future = prior(history, goal, mode="prior_mean")
                        sequence = future31_to_m1_action22(future, robot_spec=spec)
                    action_cases[scenario] = sequence
                else:
                    sequence = action_cases[scenario]

                if state_i == 0:
                    rows[scenario] = {
                        key: [] for key in (
                            "mujoco_dx_m", "wm_dx_m", "mujoco_body_vx_m_s", "wm_body_vx_m_s",
                            "mujoco_body_vy_m_s", "wm_body_vy_m_s", "mujoco_yaw_rate_rad_s",
                            "wm_yaw_rate_rad_s", "wheel_reference_mean_rad_s",
                            "mujoco_wheel_rate_mean_rad_s", "wm_wheel_rate_mean_rad_s",
                        )
                    }
                    fall_counts[scenario] = 0
                s0, xi0 = state, xi
                wm_state = state.clone()
                initial_orientation = state[:, 3:7].clone()
                final_real, final_wm = state, wm_state
                completed_horizons: set[int] = set()
                wheel_ids = torch.tensor(
                    [M1_JOINTS.index(name) for name in spec.wheel_joint_names], device=device_obj,
                )
                for step in range(10):
                    action = sequence[:, step].to(device_obj)
                    # The explicit action has no hidden action noise; this is
                    # exactly the normalized tensor passed to both models.
                    executed = action.clamp(-1.0, 1.0)
                    with torch.inference_mode():
                        wm_state = wm.step(wm_state, executed, xi0)
                        _, _, terminated, truncated, _ = env.step(executed)
                        final_real = state_io.read()
                    final_wm = wm_state
                    if bool((terminated | truncated).any()):
                        fall_counts[scenario] += 1
                        termination_samples += 1
                        break
                    if step + 1 in steps_at:
                        actual_dx = quat_apply_inverse(initial_orientation, final_real[:, :3] - s0[:, :3])[:, 0]
                        predicted_dx = quat_apply_inverse(initial_orientation, final_wm[:, :3] - s0[:, :3])[:, 0]
                        actual_body_v = quat_apply_inverse(final_real[:, 3:7], final_real[:, 29:32])
                        predicted_body_v = quat_apply_inverse(final_wm[:, 3:7], final_wm[:, 29:32])
                        for metric, value in (
                            ("mujoco_dx_m", actual_dx), ("wm_dx_m", predicted_dx),
                            ("mujoco_body_vx_m_s", actual_body_v[:, 0]), ("wm_body_vx_m_s", predicted_body_v[:, 0]),
                            ("mujoco_body_vy_m_s", actual_body_v[:, 1]), ("wm_body_vy_m_s", predicted_body_v[:, 1]),
                            ("mujoco_yaw_rate_rad_s", final_real[:, 34]), ("wm_yaw_rate_rad_s", final_wm[:, 34]),
                            ("wheel_reference_mean_rad_s", sequence[:, :step + 1, :].new_zeros(1)),
                            ("mujoco_wheel_rate_mean_rad_s", final_real[:, 35:57].index_select(-1, wheel_ids).mean(-1)),
                            ("wm_wheel_rate_mean_rad_s", final_wm[:, 35:57].index_select(-1, wheel_ids).mean(-1)),
                        ):
                            if metric == "wheel_reference_mean_rad_s":
                                future_rates = future[..., 27:31] if scenario.startswith("prior_") else torch.zeros(1, 10, 4, device=device_obj)
                                value = future_rates[:, :step + 1].mean((1, 2))
                            rows[scenario][metric].append(float(value.mean().detach().cpu()))
                        completed_horizons.add(step + 1)
                for missing_horizon in steps_at:
                    if missing_horizon not in completed_horizons:
                        for metric in rows[scenario]:
                            rows[scenario][metric].append(float("nan"))

            canonical = state_by_case["prior_stop"]
            canonical_xi = xi_by_case["prior_stop"]
            for scenario, state in state_by_case.items():
                paired_start_errors.append(float((state - canonical).abs().amax()))
                paired_xi_errors.append(float((xi_by_case[scenario] - canonical_xi).abs().amax()))

        # Compare the paired positive/negative wheel perturbation slopes.
        actual_delta = torch.tensor(rows["wheel_positive"]["mujoco_dx_m"]) - torch.tensor(rows["wheel_negative"]["mujoco_dx_m"])
        wm_delta = torch.tensor(rows["wheel_positive"]["wm_dx_m"]) - torch.tensor(rows["wheel_negative"]["wm_dx_m"])
        valid_delta = actual_delta.abs() > 1e-5
        direction_agreement = (
            float(((actual_delta * wm_delta) > 0).float()[valid_delta].mean()) if valid_delta.any() else None
        )
        relative_error = (
            float(((wm_delta - actual_delta).abs() / actual_delta.abs().clamp_min(1e-5))[valid_delta].mean())
            if valid_delta.any() else None
        )
        starts_match = max(paired_start_errors, default=float("inf")) <= 1e-5
        xi_match = max(paired_xi_errors, default=float("inf")) <= 1e-5
        thresholds_configured = min_direction_agreement is not None and max_relative_response_error is not None
        direction_pass = thresholds_configured and direction_agreement is not None and direction_agreement >= min_direction_agreement
        magnitude_pass = thresholds_configured and relative_error is not None and relative_error <= max_relative_response_error
        action_response_pass = bool(
            starts_match and xi_match and direction_pass and magnitude_pass and termination_samples == 0
        )
        result = {
            "audit_type": "m1_real_mujoco_wm_action_response_v1",
            "checkpoint": str(checkpoint_path),
            "iteration": int(checkpoint["iteration"]),
            "prior_source_sha256": prior_source["sha256"],
            "prior_input_compatibility": checkpoint["prior_input_compatibility"],
            "action_mapping": checkpoint["action_mapping"],
            "world_model_sha256": checkpoint.get("world_model_sha256"),
            "device": device, "seed": seed, "heldout_state_count": state_count,
            "prior_reference_commands_vx_m_s": {
                "stop": 0.0, "forward": forward_command, "backward": backward_command,
                "saved_training_support": list(vx_range),
            },
            "paired_initial_state_max_abs_error": max(paired_start_errors, default=None),
            "paired_xi_max_abs_error": max(paired_xi_errors, default=None),
            "paired_initial_states_match": starts_match, "paired_xi_match": xi_match,
            "wheel_positive_negative_root_dx_direction_agreement": direction_agreement,
            "wheel_positive_negative_root_dx_relative_response_error": relative_error,
            "thresholds": {
                "minimum_direction_agreement": min_direction_agreement,
                "maximum_relative_response_error": max_relative_response_error,
            },
            "action_response_pass": action_response_pass,
            "action_response_failure_reasons": [
                reason for passed, reason in (
                    (starts_match, "paired_initial_state_mismatch"),
                    (xi_match, "paired_xi_mismatch"),
                    (thresholds_configured, "response_thresholds_not_configured"),
                    (direction_pass, "wheel_perturbation_direction_response_failed"),
                    (magnitude_pass, "wheel_perturbation_response_magnitude_failed"),
                    (termination_samples == 0, "fall_or_termination_during_action_trials"),
                ) if not passed
            ],
            "actor_task_pass": False,
            "real_task_safety_pass": False,
            "actor_task_safety_review_required": True,
            "fall_counts": fall_counts,
            "step_metrics_at": list(steps_at),
            "scenarios": rows,
        }
        if out:
            out = out.expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-direction-agreement", type=float, default=None)
    parser.add_argument("--max-relative-response-error", type=float, default=None)
    args = parser.parse_args()
    result = audit(
        args.checkpoint, state_count=args.state_count, seed=args.seed,
        device=args.device, out=args.out,
        min_direction_agreement=args.min_direction_agreement,
        max_relative_response_error=args.max_relative_response_error,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
