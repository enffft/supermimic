"""CPU audit of raw V3.3 M1 trajectories, task labels, pose groups, and limits."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, get_robot_spec


BANK = Path("/home/yifeiy/AMP-LOCO-M1/dataset/m1_kinematic_motion_bank_v3_3_variable_l")
WHEEL_NAMES = ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")
WHEEL_RATE_ENVELOPE = 5.0
PIPER_SDK_LIMITS = {
    "piper_x_joint1": (-2.6179, 2.6179, 3.0),
    "piper_x_joint2": (0.0, 3.14, 3.0),
    "piper_x_joint3": (-2.967, 0.0, 3.0),
    "piper_x_joint4": (-1.745, 1.745, 3.0),
    "piper_x_joint5": (-1.22, 1.22, 3.0),
    "piper_x_joint6": (-2.09439, 2.09439, 3.0),
}


def _limits():
    urdf = Path("/home/yifeiy/AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf")
    out = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        node = joint.find("limit")
        if node is None:
            continue
        out[joint.get("name")] = (
            float(node.get("lower", "-inf")),
            float(node.get("upper", "inf")),
            float(node.get("velocity", "inf")),
        )
    return out


def _stats(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(arr):
        return {"p1": None, "p50": None, "p99": None, "min": None, "max": None}
    return {
        "p1": float(np.quantile(arr, 0.01)), "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)), "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def audit() -> dict:
    manifest = json.loads((BANK / "manifest.json").read_text())
    rows = manifest["motions"]
    if len(rows) != 3272:
        raise ValueError(f"expected 3272 motion rows, found {len(rows)}")
    if manifest.get("frame_dim") != 79:
        raise ValueError("manifest frame dimension differs from the M1 AMP 79-D contract")
    spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3")
    urdf = _limits()
    states = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    violations = defaultdict(Counter)
    goals = defaultdict(lambda: defaultdict(list))
    total_frames = Counter()
    joint_idx = {name: i for i, name in enumerate(M1_JOINTS)}
    piper_names = tuple(f"piper_x_joint{i}" for i in range(1, 7))

    for row in rows:
        task = str(row.get("motion_category") or row.get("task_type") or "unknown")
        assurance = str(row.get("collision_status", "unknown"))
        counts[f"task/{task}"] += 1
        counts[f"task_assurance/{task}/{assurance}"] += 1
        counts[f"task_validation_level/{task}/{row.get('validation_level', 'unknown')}"] += 1
        motion_id = str(row["motion_id"])
        path = BANK / "motions" / f"{motion_id}.txt"
        payload = json.loads(path.read_text())
        frame = np.asarray(payload["Frames"], dtype=np.float64)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as side:
            wheel_rate = np.asarray(side["wheel_angular_velocity_rad_s"], dtype=np.float64)
            wheel_angle = np.asarray(side["wheel_angle_rad"], dtype=np.float64)
            if task == "L":
                if "base_reference_velocity_body" in side:
                    goal = np.asarray(side["base_reference_velocity_body"], dtype=np.float64)
                elif "command_vx_vy_yaw" in side:
                    goal = np.asarray(side["command_vx_vy_yaw"], dtype=np.float64)
                else:
                    goal = np.repeat(np.asarray(row["command_vx_vy_yaw"], dtype=np.float64)[None], len(frame), axis=0)
            else:
                goal = None
            root_world = np.asarray(
                side["base_linear_velocity_world_m_s"] if "base_linear_velocity_world_m_s" in side
                else Rotation.from_quat(frame[:, 3:7]).apply(frame[:, 40:43]), dtype=np.float64,
            )
            root_rpy = np.asarray(
                side["base_roll_pitch_yaw_rad"] if "base_roll_pitch_yaw_rad" in side
                else Rotation.from_quat(frame[:, 3:7]).as_euler("xyz"), dtype=np.float64,
            )
        if frame.ndim != 2 or frame.shape[1] != 79 or len(frame) < 2:
            raise ValueError(f"{motion_id} has invalid Frames shape {frame.shape}")
        pose_fields = (row.get("standing_pose_profile"), row.get("standing_pose_profile_version"), row.get("start_arm_pose"))
        if all(pose_fields):
            pose_group = "/".join(str(value) for value in pose_fields)
        else:
            pose_group = f"root_z_{frame[0, 2]:.2f}m"
        counts[f"task_pose_group/{task}/{pose_group}"] += 1
        total_frames[task] += len(frame)

        # The dataset stores the 18 non-wheel positions/velocities in named AMP order.
        q = np.zeros((len(frame), len(M1_JOINTS)), dtype=np.float64)
        qd = np.zeros_like(q)
        for i, name in enumerate(M1_AMP_JOINTS):
            j = joint_idx[name]
            q[:, j] = frame[:, 7 + i]
            qd[:, j] = frame[:, 46 + i]
        for i, name in enumerate(WHEEL_NAMES):
            j = joint_idx[name]
            q[:, j] = wheel_angle[:, i]
            qd[:, j] = wheel_rate[:, i]

        initial = frame[0]
        states[task][f"{pose_group}/root_z_m"].append(initial[2])
        states[task][f"{pose_group}/roll_rad"].append(root_rpy[0, 0])
        states[task][f"{pose_group}/pitch_rad"].append(root_rpy[0, 1])
        states[task][f"{pose_group}/yaw_rad"].append(root_rpy[0, 2])
        states[task][f"{pose_group}/root_velocity_world_m_s"].append(root_world[0])
        for name, index in joint_idx.items():
            states[task][f"{pose_group}/q/{name}"].append(q[0, index])
            states[task][f"{pose_group}/qd/{name}"].append(qd[0, index])
        if goal is not None:
            for axis, name in enumerate(("vx", "vy", "wz")):
                goals[task][name].append(goal[:, axis])

        source_pos_violation = np.zeros_like(q, dtype=bool)
        source_sdk_violation = np.zeros_like(q, dtype=bool)
        source_velocity_violation = np.zeros_like(q, dtype=bool)
        source_envelope_violation = np.zeros_like(q, dtype=bool)
        for name, index in joint_idx.items():
            if name.endswith("FOOT_JOINT"):
                source_envelope_violation[:, index] = np.abs(qd[:, index]) > WHEEL_RATE_ENVELOPE
            else:
                lower, upper = spec.position_target_bounds[index]
                source_envelope_violation[:, index] = (q[:, index] < lower) | (q[:, index] > upper)
            hard_lo, hard_hi, hard_speed = urdf.get(name, (-np.inf, np.inf, np.inf))
            source_pos_violation[:, index] = (q[:, index] < hard_lo) | (q[:, index] > hard_hi)
            sdk = PIPER_SDK_LIMITS.get(name)
            if sdk is not None:
                sdk_lo, sdk_hi, sdk_speed = sdk
                source_sdk_violation[:, index] = (q[:, index] < sdk_lo) | (q[:, index] > sdk_hi)
                source_velocity_violation[:, index] = np.abs(qd[:, index]) > min(hard_speed, sdk_speed)
            else:
                source_velocity_violation[:, index] = np.abs(qd[:, index]) > hard_speed
        violations[task]["out_of_action_envelope_joint_frames"] += int(source_envelope_violation.sum())
        violations[task]["urdf_limit_joint_frames"] += int(source_pos_violation.sum())
        violations[task]["sdk_limit_joint_frames"] += int(source_sdk_violation.sum())
        violations[task]["velocity_limit_joint_frames"] += int(source_velocity_violation.sum())
        violations[task]["all_joint_frames"] += int(q.size)

    initial_by_group = {}
    for task, fields in states.items():
        initial_by_group[task] = {}
        for key, values in fields.items():
            array = np.asarray(values, dtype=np.float64)
            if key.endswith("root_velocity_world_m_s"):
                initial_by_group[task][key] = {axis: _stats(array[:, i]) for i, axis in enumerate(("x", "y", "z"))}
            else:
                initial_by_group[task][key] = _stats(array)
    goal_stats = {
        task: {name: _stats(np.concatenate(values)) for name, values in axes.items()}
        for task, axes in goals.items()
    }
    safety = {}
    for task, row in violations.items():
        denominator = max(int(row["all_joint_frames"]), 1)
        safety[task] = {
            key: int(value) for key, value in row.items()
        }
        for key in ("out_of_action_envelope_joint_frames", "urdf_limit_joint_frames",
                    "sdk_limit_joint_frames", "velocity_limit_joint_frames"):
            safety[task][key + "_fraction"] = float(row[key] / denominator)
        safety[task]["reference_not_dynamically_validated"] = not bool(manifest.get("physx_validated", False))
    return {
        "dataset": str(BANK), "dataset_hash_from_prior": "95f9f0a39b05d5a0d736532d24f897b3b85b8cbf8f74977de620acc9b1a7146e",
        "manifest_schema": manifest.get("schema_version"), "physx_validated": manifest.get("physx_validated"),
        "counts": dict(counts), "frames_by_task": dict(total_frames),
        "initial_state_p1_p50_p99_min_max_by_task_and_pose_group": initial_by_group,
        "effective_goal_p1_p50_p99_min_max_by_task": goal_stats,
        "source_motion_bank_safety_classification_by_task": safety,
        "classification_notes": {
            "out_of_action_envelope": "V3.3 v3 position setpoint bounds or wheel |rate| > 5 rad/s",
            "urdf_limit_violation": "hard position or velocity limits from the selected M1 URDF",
            "sdk_limit_violation": "Piper SDK position limits only; SDK qd cap is included in velocity_limit_violation",
            "reference_not_dynamically_validated": "dataset manifest physx_validated is false; quick/FCL checks are collision assurance only",
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
