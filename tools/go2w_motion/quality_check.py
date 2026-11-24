"""Rebuild Go2W geometry and velocities from saved joint states and audit every frame."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

import mujoco
import numpy as np
import yaml

from . import DIAGONALS, FIELDS, LEGS, MODEL, ROLLING, STANCE, SWING, TRAIN
from .kinematics import UP, Robot, support_geometry
from .trajectories import (
    command_trace,
    derivative,
    heading_velocity,
    integrate,
    rotations,
    trapezoid_integral,
)


def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def load_clip(path):
    with np.load(path, allow_pickle=False) as clip:
        return dict(clip)


def check_clip(clip, nominal, robot, config, gait="crawl"):
    hard, review = [], []
    dt = config["dt"]
    frames = round(config["motion_duration"]/dt) + 1
    sample_time = np.arange(frames) * dt
    for name, (shape, units, _) in FIELDS.items():
        value = clip.get(name)
        expected = tuple(frames if i == "T" else i for i in shape)
        if value is None or value.shape != expected:
            hard.append(f"shape:{name}")
        elif ((units == "bool" and value.dtype.kind != "b")
              or (units == "enum" and value.dtype.kind not in "iu")
              or (units not in ("bool", "enum") and value.dtype.kind not in "fiu")):
            hard.append(f"dtype:{name}")
        elif not np.isfinite(value).all():
            hard.append(f"nonfinite:{name}")
    if hard:
        return {"quality_status": "rejected", "failure_reasons": hard}

    thresholds = config["quality"]
    state_atol = thresholds["state_atol"]

    def require(condition, reason, severity=hard):
        if not condition:
            severity.append(reason)

    def close(name, rebuilt, atol=state_atol):
        require(np.allclose(clip[name], rebuilt, atol=atol, rtol=0), f"inconsistent:{name}")

    close("time", sample_time)
    close("command_nominal", nominal)
    expected = command_trace(sample_time, nominal, config["rest_duration"], config["ramp_duration"])
    close("command", expected)
    pos, quat = clip["base_pos"], clip["base_quat"]
    quaternion_ok = np.allclose(np.linalg.norm(quat, axis=1), 1, atol=state_atol, rtol=0)
    require(quaternion_ok, "quaternion_norm")
    require(np.all(np.sum(quat[:-1]*quat[1:], axis=1) >= 0), "quaternion_sign")
    require(np.max(np.abs(quat[:, 1:3])) < state_atol, "base_not_yaw_only")
    if not quaternion_ok:
        return {"quality_status": "rejected", "failure_reasons": hard}

    yaw = np.unwrap(2*np.arctan2(quat[:, 3], quat[:, 0]))
    base_lin = derivative(pos, dt)
    angular = np.column_stack((np.zeros((frames, 2)), derivative(yaw, dt)))
    close("base_lin_vel", base_lin)
    close("base_ang_vel", angular)
    nominal_pos, nominal_yaw = integrate(expected, dt, robot.stand_height)
    close("base_pos_nominal", nominal_pos)
    require(np.max(np.abs(nominal_yaw-yaw)) < state_atol, "yaw_tracking")
    require(np.max(np.abs(np.diff(pos[:, 2]))) <= thresholds["base_height_step"], "base_height_step")
    require(np.max(np.abs(np.diff(yaw))) <= thresholds["base_angle_step"], "base_angle_step")

    q, wheel_q, wheel_qd = clip["leg_q"], clip["wheel_q"], clip["wheel_qd"]
    qd = derivative(q, dt)
    close("leg_qd", qd)
    close("wheel_q", trapezoid_integral(wheel_qd, dt))
    hard_margin = np.min(np.minimum(q - robot.hard[:, 0], robot.hard[:, 1] - q))
    soft_margin = np.min(np.minimum(q - robot.soft[:, 0], robot.soft[:, 1] - q))
    action_margin = np.min(np.minimum(q - robot.action_limits[:, 0], robot.action_limits[:, 1] - q))
    max_qd = np.abs(qd).max()
    max_qdd = np.abs(derivative(qd, dt)).max()
    max_jerk = np.abs(derivative(derivative(qd, dt), dt)).max()
    require(hard_margin >= -state_atol, "hard_joint_limit")
    require(soft_margin >= -state_atol, "soft_joint_limit", review)
    require(action_margin >= -state_atol, "action_position_range", review)
    require(hard_margin >= thresholds["near_joint_margin"], "near_joint_limit", review)
    require(max_qd <= thresholds["joint_velocity_reject"], "severe_joint_velocity")
    require(max_qd <= thresholds["joint_velocity"], "joint_velocity", review)
    require(max_qdd <= thresholds["joint_acceleration"], "joint_acceleration", review)
    require(max_jerk <= thresholds["joint_jerk"], "joint_jerk", review)
    require(np.max(np.abs(np.diff(q, axis=0))) <= thresholds["joint_step_reject"], "joint_jump")
    require(np.max(np.abs(wheel_qd)) <= robot.wheel_rate_limit + state_atol, "wheel_rate_limit")

    phase = clip["motion_phase"]
    contact = phase != SWING
    require(np.isin(phase, [STANCE, ROLLING, SWING]).all(), "invalid_contact_mode")
    min_contacts = 2 if gait == "diagonal" else 3
    count = contact.sum(axis=1)
    require(count.min() >= min_contacts, "insufficient_support_wheels")
    require(contact[[0, -1]].all(), "incomplete_swing")
    close("contact_mask", contact)
    if count.min() < min_contacts:
        return {"quality_status": "rejected", "failure_reasons": hard}
    diagonal = count == 2
    if gait == "diagonal":
        boundaries = np.flatnonzero(np.diff(np.r_[False, diagonal, False]))
        pairs = [tuple(np.flatnonzero(~contact[start])) for start in boundaries[::2]]
        require(all(pair in DIAGONALS for pair in pairs), "non_diagonal_pair")
        require(all(a != b for a, b in zip(pairs, pairs[1:])), "diagonal_pairs_do_not_alternate")
        require(all(np.all(contact[start:end] == contact[start])
                    for start, end in boundaries.reshape(-1, 2)), "unsynchronized_diagonal_pair")
        require(np.all((count == 2) | (count == 4)), "unsynchronized_diagonal_pair")
        require(diagonal.any() or not np.any(np.asarray(nominal)[1:]), "missing_diagonal_steps", review)
        if diagonal.any():
            review.append("dynamic_diagonal_support")

    center = np.empty((frames, 4, 3))
    points = np.empty_like(center)
    com = np.empty_like(pos)
    margins = np.empty(frames)
    rolling, lateral, workspace = (np.empty((frames, 4)) for _ in range(3))
    planar_contact_rate = np.empty((frames, 4))
    collisions = Counter()
    worst_collision = 0.
    collision_frames = 0
    d, m = robot.data, robot.model
    jac = np.zeros((3, m.nv))
    for frame in range(frames):
        robot.set_pose(pos[frame], quat[frame], q[frame], wheel_q[frame])
        rebuilt_center, rebuilt_points, axes, _, rebuilt_com = robot.geometry()
        center[frame], points[frame], com[frame] = rebuilt_center, rebuilt_points, rebuilt_com
        _, margins[frame], _ = support_geometry(rebuilt_points[contact[frame]], rebuilt_com)
        hits = robot.collisions(thresholds["collision_penetration"])
        if hits:
            collisions.update(hits.keys())
            worst_collision = min(worst_collision, *hits.values())
            collision_frames += 1

        d.qvel[robot.root_qvel] = np.r_[base_lin[frame], angular[frame]]
        d.qvel[robot.qvel] = np.r_[qd[frame], wheel_qd[frame]]
        for wheel, body in enumerate(robot.wheel_bodies):
            mujoco.mj_jac(m, d, jac, None, rebuilt_center[wheel], body)
            dofs = robot.qvel[wheel*3:wheel*3+3]
            workspace[frame, wheel] = np.linalg.svd(jac[:, dofs], compute_uv=False)[-1]
            mujoco.mj_jac(m, d, jac, None, rebuilt_points[wheel], body)
            velocity = jac @ d.qvel
            planar_contact_rate[frame, wheel] = np.linalg.norm(velocity[:2])
            direction = np.cross(axes[wheel], UP)
            direction /= np.linalg.norm(direction)
            rolling[frame, wheel] = velocity @ direction
            lateral[frame, wheel] = velocity @ np.cross(UP, direction)

    error = np.linalg.norm(center - clip["wheel_center_target"], axis=-1)
    fk_error = np.linalg.norm(center - clip["wheel_center_fk"], axis=-1).max()
    failed = clip["ik_failed"] | (error > thresholds["ik_error"])
    require(not failed.any(), "ik_workspace_or_limit")
    require(fk_error <= thresholds["fk_error"], "wheel_fk")
    close("ik_error", error)
    close("contact_point", points)
    close("com_pos", com)
    close("support_margin", margins)
    # Generation differentiates observed centers. Audit differentiates generalized coordinates
    # and evaluates the full point Jacobian, independently of the rolling-rate solve.
    require(workspace.min() >= thresholds["workspace_singular_value"], "near_workspace_singularity", review)
    require(not collisions, "geometry_collision")
    gap = points[..., 2]
    require(gap.min() >= -thresholds["ground_penetration_reject"], "wheel_ground_penetration")
    require(np.max(np.abs(gap[contact]), initial=0) <= thresholds["contact_height"], "support_ground_gap", review)
    close("wheel_rate_valid", contact & ~clip["ik_failed"] & (np.abs(gap) <= thresholds["contact_height"]))
    require(np.max(planar_contact_rate[phase == STANCE], initial=0) <= thresholds["stance_speed"],
            "stance_contact_motion", review)
    center_vel = derivative(center, dt)
    center_accel = derivative(center_vel, dt)
    gap_vel = derivative(gap, dt)
    gap_accel = derivative(gap_vel, dt)
    switches = np.diff(contact, axis=0)
    transition = np.vstack((switches, np.zeros_like(contact[:1]))) | np.vstack((np.zeros_like(contact[:1]), switches))
    start, wheel = np.where(contact[:-1] & ~contact[1:])
    end, wheel_end = np.where(~contact[:-1] & contact[1:])
    if len(start) == len(end):
        for w in range(4):
            for first, last in zip(start[wheel == w]+1, end[wheel_end == w]+1):
                require(gap[first:last, w].max() >= thresholds["swing_min_peak_height"],
                        "swing_peak_clearance", review)
    require(np.min(gap[~contact], initial=0) >= -state_atol, "swing_ground_touch", review)
    support_transition = transition & contact
    require(np.max(np.abs(gap_vel[support_transition]), initial=0) <=
            thresholds["transition_vertical_speed"], "transition_vertical_speed", review)
    require(np.max(np.abs(gap_accel[support_transition]), initial=0) <=
            thresholds["transition_acceleration"], "transition_vertical_acceleration", review)
    require(np.max(np.linalg.norm(center_accel[..., :2][transition], axis=-1), initial=0) <=
            thresholds["transition_horizontal_acceleration"], "transition_horizontal_acceleration", review)

    rolling_rms = rms(rolling[contact])
    lateral_rms = rms(lateral[contact])
    require(rolling_rms <= thresholds["rolling_rms"], "rolling_residual", review)
    require(lateral_rms <= thresholds["lateral_rms"], "lateral_residual", review)
    require(np.max(np.abs(lateral[contact]), initial=0) <= thresholds["lateral_peak"], "lateral_peak", review)
    polygon_margin = margins[~diagonal].min()
    line_distance = -margins[diagonal].min() if diagonal.any() else 0.
    require(polygon_margin >= thresholds["support_min"], "com_support_margin", review)
    require(line_distance <= thresholds.get("diagonal_com_distance", 0.08), "diagonal_com_distance", review)
    correction = np.linalg.norm(pos-nominal_pos, axis=1).max()
    require(correction <= thresholds["correction_max"] + state_atol, "base_correction", review)

    ref = heading_velocity(base_lin, angular, rotations(yaw))
    plateau = (sample_time >= config["rest_duration"] + config["ramp_duration"] - state_atol) & (
        sample_time <= sample_time[-1] - config["rest_duration"] - config["ramp_duration"] + state_atol)
    plateau_error = rms(ref[plateau] - np.asarray(nominal), axis=0)
    mean = ref.mean(axis=0)
    plateau_mean = ref[plateau].mean(axis=0)
    nominal_ref = heading_velocity(derivative(nominal_pos, dt), angular, rotations(nominal_yaw))
    nominal_rmse = rms(nominal_ref[plateau] - np.asarray(nominal), axis=0)
    require(np.all(plateau_error <= thresholds["plateau_rmse"]), "plateau_tracking", review)
    require(np.all(np.abs(mean[np.asarray(nominal) == 0]) <= np.array(thresholds["drift_mean"])[np.asarray(nominal) == 0]),
            "uncommanded_axis_drift", review)
    require(np.max(np.abs(ref[[0, -1]])) <= thresholds["terminal_speed"], "terminal_base_motion", review)
    status = "rejected" if hard else "candidate" if review else "qualified"

    metrics = dict(
        max_ik_error=error.max(), max_fk_error=fk_error, min_joint_margin=hard_margin,
        min_soft_joint_margin=soft_margin, min_action_margin=action_margin, max_joint_velocity=max_qd,
        max_joint_acceleration=max_qdd, max_joint_jerk=max_jerk, max_wheel_velocity=np.abs(wheel_qd).max(),
        min_workspace_singular_value=workspace.min(), ik_failure_frames=int(failed.any(axis=1).sum()),
        first_ik_failure_frame=int(np.argmax(failed.any(axis=1))) if failed.any() else None,
        rolling_residual_rms=rolling_rms, rolling_residual_peak=np.max(np.abs(rolling[contact]), initial=0),
        lateral_residual_rms=lateral_rms, lateral_residual_peak=np.max(np.abs(lateral[contact]), initial=0),
        min_support_margin=polygon_margin, max_base_correction=correction,
        max_diagonal_com_distance=line_distance, diagonal_frames=int(diagonal.sum()),
        collision_count=sum(collisions.values()), collision_frames=collision_frames,
        worst_collision_distance=worst_collision, collision_pairs=dict(collisions),
        min_wheel_ground_gap=gap.min(), swing_frames=int((~contact).any(axis=1).sum()),
    )
    for axis, name in enumerate(("vx", "vy", "wz")):
        metrics[f"{name}_ref_mean"] = mean[axis]
        metrics[f"{name}_plateau_mean"] = plateau_mean[axis]
        metrics[f"{name}_plateau_rmse"] = plateau_error[axis]
        metrics[f"{name}_nominal_plateau_rmse"] = nominal_rmse[axis]
    return {**metrics, "quality_status": status, "failure_reasons": list(dict.fromkeys(hard + review))}


def mirror_checks(rows, clips, config):
    by_command = {tuple(row["nominal_command"]): row["motion_id"] for row in rows}
    mirror = np.array([1, -1, -1])
    leg_order = [1, 0, 3, 2]
    joint_sign = np.array([-1, 1, 1])
    checks = []
    for key, motion_id in by_command.items():
        opposite = by_command.get(tuple(np.asarray(key) * mirror))
        if not opposite or motion_id >= opposite or motion_id not in clips or opposite not in clips:
            continue
        a, b = clips[motion_id], clips[opposite]
        position = rms(a["base_pos"]*np.array([1, -1, 1]) - b["base_pos"])
        joint = rms(a["leg_q"].reshape(-1, 4, 3)[:, leg_order] * joint_sign - b["leg_q"].reshape(-1, 4, 3))
        velocity = rms(a["base_lin_vel"]*np.array([1, -1, 1]) - b["base_lin_vel"], axis=0)
        yaw_rate = rms(a["base_ang_vel"][:, 2] + b["base_ang_vel"][:, 2])
        within = (position <= config["quality"]["mirror_position_rmse"]
                  and joint <= config["quality"]["mirror_joint_rmse"])
        checks.append(dict(pair=[motion_id, opposite], base_position_rmse=position,
                           world_linear_velocity_rmse=velocity.tolist(), yaw_rate_rmse=yaw_rate,
                           joint_rmse=joint, within_review_thresholds=bool(within)))
    return checks


def audit_dataset(dataset, robot=None):
    started = perf_counter()
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    config = yaml.safe_load((dataset / "generation_config.yaml").read_text())
    robot = Robot() if robot is None else robot
    for name, expected in {
        "format": "go2w_kinematic_reference", "validation_level": "kinematic_only", "physics_validated": False,
        "leg_joint_names": list(robot.joint_names[:12]), "wheel_joint_names": list(robot.joint_names[12:]),
        "actuator_names": list(robot.actuator_names),
        "wheel_order": list(LEGS), "quaternion_order": "wxyz", "dt": config["dt"],
        "motion_phase": {"STANCE": STANCE, "ROLLING": ROLLING, "SWING": SWING},
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "control_config_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "wheel_mesh_sha256": hashlib.sha256((MODEL.parent / "assets/wheel.stl").read_bytes()).hexdigest(),
    }.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Manifest {name} disagrees with the current schema or robot")

    gait = manifest.get("gait", "crawl")
    if gait not in ("crawl", "diagonal"):
        raise ValueError(f"Unknown gait {gait}")
    rows, clips = [], {}
    for motion in manifest["motions"]:
        filename = (dataset / motion["file_path"]).resolve()
        if not filename.is_relative_to(dataset.resolve()):
            raise ValueError(f"Motion path escaped the dataset: {filename}")
        nominal = np.asarray(motion["nominal_command"])
        try:
            clip = load_clip(filename)
            check = check_clip(clip, nominal, robot, config, gait=gait)
            if "max_fk_error" in check:
                clips[motion["motion_id"]] = clip
        except (OSError, ValueError, BadZipFile) as error:
            check = {"quality_status": "rejected", "failure_reasons": [f"unreadable_clip:{error}"]}
        rows.append(dict(motion_id=motion["motion_id"], category=motion["motion_category"],
                         nominal_command=motion["nominal_command"], vx_cmd=nominal[0], vy_cmd=nominal[1],
                         wz_cmd=nominal[2], generation_time=motion["generation_time"], **check))

    report = dataset / "reports"
    report.mkdir(exist_ok=True)
    categories = {}
    for group in dict.fromkeys(row["category"] for row in rows):
        items = [row for row in rows if row["category"] == group]
        categories[group] = {
            "requested": len(items),
            "saved": sum((dataset / m["file_path"]).is_file() for m in manifest["motions"] if m["motion_category"] == group),
            **{s: sum(row["quality_status"] == s for row in items) for s in ("qualified", "candidate", "rejected")},
            "qualified_commands": [row["nominal_command"] for row in items if row["quality_status"] == "qualified"],
            "generation_seconds": sum(row["generation_time"] for row in items),
            "failure_reasons": dict(Counter(reason for row in items for reason in row["failure_reasons"])),
            "max_metrics": {key: max(row[key] for row in items if key in row) for key in (
                "max_ik_error", "max_fk_error", "max_joint_velocity", "max_wheel_velocity", "lateral_residual_rms",
                "rolling_residual_rms", "vx_plateau_rmse", "vy_plateau_rmse", "wz_plateau_rmse",
            ) if any(key in row for row in items)},
            "min_support_margin": min((row["min_support_margin"] for row in items if "min_support_margin" in row), default=None),
            "collision_count": sum(row.get("collision_count", 0) for row in items),
            "max_diagonal_com_distance": max((row.get("max_diagonal_com_distance", 0.) for row in items), default=0.),
            "diagonal_frames": sum(row.get("diagonal_frames", 0) for row in items),
        }
    mirrors = mirror_checks(rows, clips, config)
    summary = dict(
        validation_level="kinematic_only", physics_validated=False,
        generated_at=manifest["generated_at"], generator_commit=manifest["generator_commit"],
        gait=gait, categories=categories, generation_seconds=manifest["generation_seconds"],
        frame_count=sum(len(clip["time"]) for clip in clips.values()),
        space_bytes=sum(p.stat().st_size for p in (dataset / "motions").rglob("*.npz")),
        audited_frames=sum(len(clip["time"]) for clip in clips.values()),
        audit_seconds=perf_counter() - started,
        geometry_coverage=dict(self_pairs=len(robot.self_pairs), nonwheel_ground_pairs=len(robot.ground_geoms),
                               wheel_ground_meshes=4, excluded="same link and direct parent-child"),
        mirror_checks=mirrors,
        manifest_disagreements=[row["motion_id"] for row, motion in zip(rows, manifest["motions"])
                                if row["quality_status"] != motion["quality_status"]
                                or row["failure_reasons"] != motion["failure_reasons"]],
    )
    (report / "quality_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    required = ("motion_id category vx_cmd vy_cmd wz_cmd vx_ref_mean vy_ref_mean wz_ref_mean "
                "vx_plateau_rmse vy_plateau_rmse wz_plateau_rmse max_ik_error max_fk_error "
                "min_joint_margin max_joint_velocity max_wheel_velocity rolling_residual_rms "
                "lateral_residual_rms min_support_margin collision_count generation_time quality_status").split()
    fieldnames = list(dict.fromkeys(required + [
        key for row in rows for key in row if key not in ("failure_reasons", "nominal_command")
    ]))
    with (report / "quality_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fieldnames, "failure_reason"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fieldnames if key in row},
                             "failure_reason": " | ".join(row["failure_reasons"])})
    write_report(report / "generation_report.md", summary, rows)
    return summary


def write_report(path, summary, rows):
    duration = summary["generation_seconds"]
    lines = [
        "# Go2W flat kinematic generation report", "",
        f"Generated {summary['generated_at']} from commit `{summary['generator_commit']}`.",
        f"Generated {summary['frame_count']:,} audited frames in {duration:.2f} s "
        f"({summary['frame_count']/max(duration, 1e-6):.1f} frames/s). "
        f"NPZ size {summary['space_bytes']/2**20:.2f} MiB. Audit {summary['audit_seconds']:.2f} s.",
        f"Gait `{summary['gait']}`. Validation level `kinematic_only`. Physics validated `false`.", "",
        "| Category | Requested | Saved | Qualified | Candidate | Rejected | Generation (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stats in summary["categories"].items():
        lines.append(f"| {group} | {stats['requested']} | {stats['saved']} | {stats['qualified']} | "
                     f"{stats['candidate']} | {stats['rejected']} | {stats['generation_seconds']:.2f} |")
    coverage = summary["geometry_coverage"]
    lines.extend(["", f"Geometry coverage includes {coverage['self_pairs']} nonadjacent robot pairs, "
                  f"{coverage['nonwheel_ground_pairs']} nonwheel ground pairs, and all four wheel meshes per sample. "
                  "Same-link and direct parent-child pairs are excluded. Continuous-time collision checking is outside this audit.",
                  "", "## Commands requiring review", "",
                  "| Motion | Requested (vx, vy, wz) | Status | Findings |",
                  "|---|---|---|---|"])
    for row in rows:
        if row["quality_status"] != "qualified":
            reasons = ", ".join(row["failure_reasons"])
            lines.append(f"| {row['motion_id']} | {row['nominal_command']} | {row['quality_status']} | {reasons} |")
    (path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    summary = audit_dataset(args.dataset)
    for category, counts in summary["categories"].items():
        print(category, {key: counts[key] for key in ("qualified", "candidate", "rejected")})
    print(f"Audited {summary['audited_frames']} frames in {summary['audit_seconds']:.2f} s")
    if summary["manifest_disagreements"]:
        print("Manifest disagreements:", ", ".join(summary["manifest_disagreements"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
