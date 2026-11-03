#!/usr/bin/env python3
"""CPU-only systematic audit of AMP M1 references against Action Contract v2/v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from supermimic.robots.spec import (
    M1_JOINTS,
    M1_POSITION_TARGET_BOUNDS_V3_BY_NAME,
    get_robot_spec,
)

DEFAULT_CACHE = Path("/tmp/m1_cvae_window_cache_v2/cvae_v1_strict_fcl_2272")
DEFAULT_URDF = Path("/home/yifeiy/AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf")
DEFAULT_SDK = Path("/home/yifeiy/m1_piper_rl_grasp/src/rl_sar/piper/piper_backend.py")
TASKS = ("L", "M", "LM")
POSITION_NAMES = tuple(n for n in M1_JOINTS if not n.endswith("FOOT_JOINT"))
AMP_POSITION_NAMES = (
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
    *(f"piper_x_joint{i}" for i in range(1, 7)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sdk_limits(path: Path) -> list[list[float]]:
    text = path.read_text()
    match = re.search(
        r"_PIPER_JOINT_LIMITS_MILLIDEG:\s*List\[Tuple\[float, float\]\]\s*=\s*\[(.*?)\n\]",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"could not parse Piper SDK software limits from {path}")
    rows = re.findall(r"\(([^,]+),\s*([^)]+)\)", match.group(1))

    def millidegrees(expression: str) -> float:
        values = [float(part.strip()) for part in expression.split("*")]
        return math.prod(values)

    parsed = []
    for lower, upper in rows:
        parsed.append(
            [math.radians(millidegrees(lower) / 1000.0), math.radians(millidegrees(upper) / 1000.0)]
        )
    if len(parsed) != 6:
        raise ValueError(f"expected six Piper SDK limits, parsed {len(parsed)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument(
        "--stride",
        type=int,
        default=35,
        help="systematic row sample stride; 35 gives roughly 6k windows/task",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/artifacts/m1_action_contract_v3_audit.json")
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")

    cache_audit = json.loads((args.cache / "window_audit.json").read_text())
    motion_ids = cache_audit["motion_ids_by_index"]
    meta = np.load(args.cache / "meta.npy", mmap_mode="r")
    future = np.load(args.cache / "y.npy", mmap_mode="r")
    history = np.load(args.cache / "x.npy", mmap_mode="r")
    samples: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    wheels: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    episode_start_arm: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    sampled_windows = {task: 0 for task in TASKS}
    for row in range(len(meta)):
        motion_index, start_frame = int(meta[row, 0]), int(meta[row, 1])
        task = motion_ids[motion_index].split("_")[0].upper()
        if task not in samples:
            continue
        if row % args.stride == 0:
            samples[task].append(np.asarray(future[row, :, 9:27]))
            wheels[task].append(np.asarray(future[row, :, 27:31]))
            sampled_windows[task] += 1
        if start_frame == 3:
            episode_start_arm[task].append(np.asarray(history[row, 172:190])[12:18])

    urdf_root = ET.parse(args.urdf).getroot()
    urdf_limits = {
        joint.attrib["name"]: [
            float(joint.find("limit").attrib["lower"]),
            float(joint.find("limit").attrib["upper"]),
        ]
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("type") == "revolute" and joint.find("limit") is not None
    }
    sdk = sdk_limits(args.sdk)
    hard: dict[str, list[float]] = {}
    soft: dict[str, list[float]] = {}
    for name in POSITION_NAMES:
        lo, hi = urdf_limits[name]
        if name.startswith("piper_x_joint"):
            sdk_lo, sdk_hi = sdk[int(name[-1]) - 1]
            lo, hi = max(lo, sdk_lo), min(hi, sdk_hi)
        hard[name] = [lo, hi]
        center = 0.5 * (lo + hi)
        half = 0.45 * (hi - lo)  # matches mjlab 0.9 soft-limit contraction
        soft[name] = [center - half, center + half]

    v2 = get_robot_spec("m1_piper")
    legacy = dict(zip(M1_JOINTS, v2.default_joint_pos))
    legacy_scale = dict(zip(M1_JOINTS, v2.action_scales))
    v2_bounds = {
        name: [
            max(soft[name][0], legacy[name] - legacy_scale[name]),
            min(soft[name][1], legacy[name] + legacy_scale[name]),
        ]
        for name in POSITION_NAMES
    }
    new_bounds = {name: list(M1_POSITION_TARGET_BOUNDS_V3_BY_NAME[name]) for name in POSITION_NAMES}

    result: dict[str, object] = {
        "schema": "m1_action_contract_v3_audit_v1",
        "sources": {
            "cache": str(args.cache.resolve()),
            "cache_window_counts": cache_audit["windows_by_task"],
            "systematic_stride": args.stride,
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": sha256(args.urdf),
            "piper_sdk_limits_source": str(args.sdk.resolve()),
            "sdk_sha256": sha256(args.sdk),
            "sdk_limits_rad": sdk,
        },
        "pose_profiles": {
            task: {
                "sampled_trajectory_start_count": len(episode_start_arm[task]),
                "p5_p50_p95_piper_q_rad": np.quantile(
                    np.asarray(episode_start_arm[task]), [0.05, 0.50, 0.95], axis=0
                ).tolist()
                if episode_start_arm[task]
                else None,
            }
            for task in TASKS
        },
        "joint_order_future_position": list(AMP_POSITION_NAMES),
        "limits": {
            name: {
                "urdf_rad": urdf_limits[name],
                "sdk_rad": sdk[int(name[-1]) - 1] if name.startswith("piper_x_joint") else None,
                "hardware_intersection_rad": hard[name],
                "mujoco_soft_rad": soft[name],
            }
            for name in POSITION_NAMES
        },
        "position_bounds": {
            "v2_legacy_manipulation": v2_bounds,
            "v3_dataset_carry": new_bounds,
        },
        "tasks": {},
    }
    for task in TASKS:
        q = np.concatenate(samples[task], axis=0)
        wheel = np.concatenate(wheels[task], axis=0)
        p = {name: q[:, AMP_POSITION_NAMES.index(name)] for name in POSITION_NAMES}
        inside_v2 = np.stack(
            [(p[name] >= v2_bounds[name][0]) & (p[name] <= v2_bounds[name][1])
             for name in POSITION_NAMES],
            axis=-1,
        )
        inside_v3 = np.stack(
            [(p[name] >= new_bounds[name][0]) & (p[name] <= new_bounds[name][1])
             for name in POSITION_NAMES],
            axis=-1,
        )
        q_stats = {}
        carry = dict(
            zip(
                M1_JOINTS,
                get_robot_spec(
                    "m1_piper",
                    m1_pose_profile="dataset_carry",
                    m1_action_contract="v3",
                ).default_joint_pos,
            )
        )
        for name, values in p.items():
            sdk_interval = sdk[int(name[-1]) - 1] if name.startswith("piper_x_joint") else None
            lo, hi = new_bounds[name]
            q0 = carry[name]
            normalized = np.where(
                values >= q0, (values - q0) / max(hi - q0, 1e-8), (values - q0) / max(q0 - lo, 1e-8)
            )
            q_stats[name] = {
                "min_p1_p5_p50_p95_p99_max_rad": [
                    float(values.min()),
                    *np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99]).tolist(),
                    float(values.max()),
                ],
                "v2_outside_fraction": float(
                    np.mean((values < v2_bounds[name][0]) | (values > v2_bounds[name][1]))
                ),
                "v3_outside_fraction": float(
                    np.mean((values < new_bounds[name][0]) | (values > new_bounds[name][1]))
                ),
                "v3_near_action_saturation_fraction": float(np.mean(np.abs(normalized) >= 0.98)),
                "urdf_hard_violation_fraction": float(
                    np.mean((values < urdf_limits[name][0]) | (values > urdf_limits[name][1]))
                ),
                "sdk_violation_fraction": float(
                    np.mean((values < sdk_interval[0]) | (values > sdk_interval[1]))
                )
                if sdk_interval
                else 0.0,
                "mujoco_soft_violation_fraction": float(
                    np.mean((values < soft[name][0]) | (values > soft[name][1]))
                ),
            }
        result["tasks"][task] = {
            "sampled_windows": sampled_windows[task],
            "sampled_future_frames": len(q),
            "position_joint_stats": q_stats,
            "wheel_rate_abs_p50_p90_p99_rad_s": np.quantile(
                np.abs(wheel), [0.50, 0.90, 0.99], axis=0
            ).tolist(),
            "wheel_outside_v2_pm5_fraction": float(np.mean(np.abs(wheel) > 5.0)),
            "wheel_outside_v3_pm5_fraction": float(np.mean(np.abs(wheel) > 5.0)),
            "v2_coordinate_coverage_fraction": float(np.mean(inside_v2)),
            "v3_coordinate_coverage_fraction": float(np.mean(inside_v3)),
            "v2_full_pose_coverage_fraction": float(np.mean(np.all(inside_v2, axis=-1))),
            "v3_full_pose_coverage_fraction": float(np.mean(np.all(inside_v3, axis=-1))),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "sampled_windows": sampled_windows,
                "new_bounds": new_bounds,
                "coverage": {
                    task: result["tasks"][task]["v3_full_pose_coverage_fraction"]
                    for task in TASKS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
