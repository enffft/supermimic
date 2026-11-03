"""CPU-only V3.3 Motion Prior comparison on matched M1 State57 histories."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import state_history_to_cvae53
from supermimic.motion_prior.reference import decode_future31
from supermimic.robots.spec import (
    M1_AMP_JOINTS, M1_CARRY_DEFAULT, M1_JOINTS,
    get_robot_spec,
)


ROOT = Path("/home/yifeiy/AMP-LOCO-M1")
BANK = ROOT / "dataset/m1_kinematic_motion_bank_v3_3_variable_l"
NEW_PRIOR = ROOT / "logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt"
OLD_PRIOR = ROOT / "logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt"
NEW_SHA256 = "05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c"
OLD_SHA256 = "1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d"
GOALS = (-0.30, -0.15, -0.08, 0.0, 0.12, 0.15, 0.30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state_history() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = json.loads((BANK / "manifest.json").read_text())
    rows = [row for row in manifest["motions"] if row.get("motion_category") == "L"]
    if len(rows) != 999:
        raise ValueError(f"expected 999 V3.3 L motions, found {len(rows)}")
    row = next(row for row in rows if row["motion_id"] == "l_000000_forward")
    motion_path = BANK / "motions" / f"{row['motion_id']}.txt"
    sidecar_path = motion_path.with_suffix(".npz")
    payload = json.loads(motion_path.read_text())
    frames = np.asarray(payload["Frames"], dtype=np.float32)
    with np.load(sidecar_path, allow_pickle=False) as archive:
        wheel_rates = np.asarray(archive["wheel_angular_velocity_rad_s"], dtype=np.float32)
        wheel_angles = np.asarray(archive["wheel_angle_rad"], dtype=np.float32)
        goal = np.asarray(archive["base_reference_velocity_body"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 79:
        raise ValueError(f"invalid V3.3 motion shape: {frames.shape}")
    robot = get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3")
    states = np.zeros((len(frames), 57), dtype=np.float32)
    states[:, :3] = frames[:, :3]
    states[:, 3:7] = frames[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> StateIO wxyz
    states[:, 7:29] = np.asarray(robot.default_joint_pos, dtype=np.float32)
    states[:, 29:32] = Rotation.from_quat(frames[:, 3:7]).apply(frames[:, 40:43]).astype(np.float32)
    states[:, 32:35] = frames[:, 43:46]
    states[:, 35:57] = 0.0
    robot_joint_index = {name: i for i, name in enumerate(M1_JOINTS)}
    for amp_index, name in enumerate(M1_AMP_JOINTS):
        index = robot_joint_index[name]
        states[:, 7 + index] = frames[:, 7 + amp_index]
        states[:, 35 + index] = frames[:, 46 + amp_index]
    wheel_names = ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")
    for wheel_i, name in enumerate(wheel_names):
        index = robot_joint_index[name]
        states[:, 7 + index] = wheel_angles[:, wheel_i]
        states[:, 35 + index] = wheel_rates[:, wheel_i]
    middle = len(frames) // 2
    real_history = torch.from_numpy(states[middle - 3:middle + 1].copy())[None]
    real_goal = torch.from_numpy(goal[middle].copy())
    if real_history.shape != (1, 4, 57):
        raise RuntimeError("failed to construct a four-frame source history")
    return real_history, real_goal, torch.from_numpy(states[0].copy())


def _static_history(base_height: float, default_joint_pos: tuple[float, ...]) -> torch.Tensor:
    state = torch.zeros(57, dtype=torch.float32)
    state[2] = float(base_height)
    state[3] = 1.0
    state[7:29] = torch.tensor(default_joint_pos)
    return state.view(1, 1, 57).expand(1, 4, 57).clone()


def _urdf_limits() -> dict[str, tuple[float, float, float]]:
    path = ROOT / "resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf"
    limits = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        entry = joint.find("limit")
        if entry is None:
            continue
        name = joint.get("name")
        limits[name] = (
            float(entry.get("lower", "-inf")),
            float(entry.get("upper", "inf")),
            float(entry.get("velocity", "inf")),
        )
    return limits


def _contract_bounds(label: str):
    if label == "old_v2":
        spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v2")
        bounds = tuple(
            (q - scale, q + scale) if not name.endswith("FOOT_JOINT") else (0.0, 0.0)
            for name, q, scale in zip(spec.joint_names, spec.default_joint_pos, spec.action_scales)
        )
        return spec, bounds
    if label == "ab_v3":
        spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v3")
        return spec, spec.position_target_bounds
    spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3")
    return spec, spec.position_target_bounds


def _assess_reference(future, anchor, bounds, urdf, command_vx: float) -> dict:
    state = anchor[None]
    with torch.inference_mode():
        predicted = decode_future31(state, future)
    piper_sdk = {
        "piper_x_joint1": (-2.6179, 2.6179, 3.0),
        "piper_x_joint2": (0.0, 3.14, 3.0),
        "piper_x_joint3": (-2.967, 0.0, 3.0),
        "piper_x_joint4": (-1.745, 1.745, 3.0),
        "piper_x_joint5": (-1.22, 1.22, 3.0),
        "piper_x_joint6": (-2.09439, 2.09439, 3.0),
    }
    counts = {key: 0 for key in (
        "out_of_action_envelope", "urdf_limit_violation", "sdk_limit_violation",
        "velocity_limit_violation", "command_forward_reference_not_positive",
    )}
    total_position = 0
    total_velocity = 0
    position_ids = [M1_JOINTS.index(name) for name in M1_AMP_JOINTS]
    for joint_id, name in zip(position_ids, M1_AMP_JOINTS):
        q = predicted[0, :, 7 + joint_id]
        lo, hi = bounds[joint_id]
        counts["out_of_action_envelope"] += int(((q < lo) | (q > hi)).sum())
        hard_lo, hard_hi, velocity_limit = urdf.get(name, (-float("inf"), float("inf"), float("inf")))
        counts["urdf_limit_violation"] += int(((q < hard_lo) | (q > hard_hi)).sum())
        if name in piper_sdk:
            sdk_lo, sdk_hi, sdk_speed = piper_sdk[name]
            counts["sdk_limit_violation"] += int(((q < sdk_lo) | (q > sdk_hi)).sum())
            q_velocity = torch.diff(torch.cat((anchor[7 + joint_id:8 + joint_id], q))) / 0.02
            counts["velocity_limit_violation"] += int((q_velocity.abs() > min(velocity_limit, sdk_speed)).sum())
        else:
            q_velocity = torch.diff(torch.cat((anchor[7 + joint_id:8 + joint_id], q))) / 0.02
            counts["velocity_limit_violation"] += int((q_velocity.abs() > velocity_limit).sum())
        total_position += len(q)
        total_velocity += 10
    wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
    wheel_reference = future[0, :, 27:31]
    for wheel_id, name in zip(wheel_ids, ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")):
        slot = ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT").index(name)
        wheel = wheel_reference[:, slot]
        counts["out_of_action_envelope"] += int((wheel.abs() > 5.0).sum())
        wheel_limit = urdf.get(name, (-float("inf"), float("inf"), 110.0))[2]
        counts["velocity_limit_violation"] += int((wheel.abs() > wheel_limit).sum())
        total_velocity += len(wheel)
    root_delta_heading = future[0, -1, :3]
    counts["command_forward_reference_not_positive"] = int(command_vx > 0.0 and root_delta_heading[0] <= 0.0)
    return {
        "predicted_root_delta_0p2s_heading_local_m": root_delta_heading.cpu().tolist(),
        "predicted_approx_root_velocity_heading_local_m_s": (root_delta_heading / 0.2).cpu().tolist(),
        "predicted_approx_root_velocity_world_m_s": predicted[0, -1, 29:32].cpu().tolist(),
        "wheel_reference_rates_rad_s_by_joint": wheel_reference.mean(0).cpu().tolist(),
        "wheel_reference_final_rates_rad_s_by_joint": wheel_reference[-1].cpu().tolist(),
        "leg_reference_mean_rad_by_joint": {
            name: float(predicted[0, :, 7 + M1_JOINTS.index(name)].mean()) for name in M1_AMP_JOINTS[:12]
        },
        "piper_reference_mean_rad_by_joint": {
            name: float(predicted[0, :, 7 + M1_JOINTS.index(name)].mean()) for name in M1_AMP_JOINTS[12:]
        },
        "joint_velocity_max_abs_rad_s": float(predicted[0, :, 35:57].abs().max()),
        "violation_counts_per_10_step_prediction": counts,
        "violation_denominators": {"position_joint_frames": total_position, "velocity_joint_frames": total_velocity},
        "reference_not_dynamically_validated": True,
    }


def run() -> dict:
    torch.set_num_threads(1)
    real_history, real_goal, initial = _load_state_history()
    high = _static_history(0.47, get_robot_spec(
        "m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
    ).default_joint_pos)
    low = _static_history(0.3892858, M1_CARRY_DEFAULT)
    histories = {"V3.3 L real middle-history": real_history, "V3.3 L static high": high, "old static low carry": low}
    goal_batch = []
    for vx in GOALS:
        goal_batch.append(torch.tensor([[vx, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32))
    models = {
        "old_cvae+old_v2_contract": (OLD_PRIOR, OLD_SHA256, "train_constant_compat_v1", "old_v2"),
        "old_cvae+ab_v3_contract": (OLD_PRIOR, OLD_SHA256, "train_constant_compat_v1", "ab_v3"),
        "new_v33_cvae+v33_l_v3_contract": (NEW_PRIOR, NEW_SHA256, "legacy_v1", "v33_v3"),
    }
    urdf = _urdf_limits()
    report: dict[str, object] = {
        "dataset": str(BANK),
        "new_prior_path": str(NEW_PRIOR), "new_prior_sha256": _sha256(NEW_PRIOR),
        "old_prior_path": str(OLD_PRIOR), "old_prior_sha256": _sha256(OLD_PRIOR),
        "source_history_motion_id": "l_000000_forward, frames around 50% duration",
        "source_history_native_goal_vx_vy_wz": real_goal.tolist(),
        "goal_vx_commands_m_s": list(GOALS),
        "goal_support": {
            "v3_3_l_effective_vx_p1_p99": [-0.089499, 0.144130],
            "v3_3_l_effective_vy_p1_p99": [-0.050272, 0.056770],
            "v3_3_l_effective_wz_p1_p99": [-0.171703, 0.181637],
            "vx_in_support": {str(v): -0.089499 <= v <= 0.144130 for v in GOALS},
        },
        "histories": {},
        "comparisons": {},
    }
    for history_name, history in histories.items():
        report["histories"][history_name] = {
            "root_height_m": float(history[0, -1, 2]),
            "initial_leg_and_arm_joint_positions_rad": {
                name: float(history[0, -1, 7 + M1_JOINTS.index(name)]) for name in M1_AMP_JOINTS
            },
        }
        for model_name, (path, sha, compatibility, contract_name) in models.items():
            prior = load_m1_motion_prior(path, device="cpu", expected_sha256=sha, input_compatibility=compatibility)
            contract, bounds = _contract_bounds(contract_name)
            rows = []
            for command_i, vx in enumerate(GOALS):
                goal = goal_batch[command_i]
                with torch.inference_mode():
                    feature = state_history_to_cvae53(history)
                    output = prior.forward_features(feature, goal)
                    diagnostics = prior.input_diagnostics(history, goal)
                assessment = _assess_reference(output, history[0, -1], bounds, urdf, vx)
                assessment["input_normalized_max_abs_zscore"] = float(diagnostics["prior/input/max_abs_zscore"])
                assessment["input_normalized_p99_abs_zscore"] = float(diagnostics["prior/input/p99_abs_zscore"])
                assessment["input_q6_zscore_max_abs"] = float(diagnostics["prior/input/q6_normalized"])
                assessment["input_qd6_zscore_max_abs"] = float(diagnostics["prior/input/qd6_normalized"])
                assessment["input_compat_projection_count"] = float(diagnostics["prior/input/compat_projection_count"])
                assessment["input_compatibility"] = compatibility
                assessment["action_contract"] = contract.action_contract_version
                assessment["pose_profile"] = contract.default_pose_profile
                assessment["command_vx_m_s"] = vx
                rows.append(assessment)
            report["comparisons"][f"{history_name} / {model_name}"] = rows
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
