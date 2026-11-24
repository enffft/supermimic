"""Generate deterministic flat-ground Go2W kinematic reference clips."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import yaml

from . import CONFIG, FIELDS, LEGS, MODEL, ROOT, TRAIN, STANCE, ROLLING, SWING
from .contact_planner import plan, support_correction
from .kinematics import Robot, UP, support_geometry, wheel_rates
from .quality_check import check_clip, audit_dataset
from .trajectories import command_trace, derivative, integrate, rotations, trapezoid_integral, yaw_quat


MIRROR = np.array([1, -1, -1])
PILOT_COMMANDS = (
    ("forward_backward", (0.30, 0, 0)),
    ("forward_backward", (-0.20, 0, 0)),
    ("lateral", (0, 0.10, 0)),
    ("lateral", (0, -0.10, 0)),
    ("yaw", (0, 0, 0.30)),
    ("yaw", (0, 0, -0.30)),
    ("forward_yaw", (0.30, 0, 0.30)),
    ("forward_lateral", (0.30, 0.10, 0)),
    ("lateral_yaw", (0, 0.10, 0.30)),
)


def commands(config):
    for category, rows in config["commands"].items():
        if category == "forward_backward":
            for vx in rows:
                direction = "forward" if vx > 0 else "backward"
                yield f"{direction}_vx_{round(abs(vx)*100):03d}", category, (vx, 0., 0.)
        elif category in ("lateral", "yaw"):
            axis = 1 if category == "lateral" else 2
            name = "vy" if axis == 1 else "wz"
            for value in rows:
                direction = "left" if value > 0 else "right"
                command = [0., 0., 0.]
                command[axis] = value
                yield f"{category}_{direction}_{name}_{round(abs(value)*100):03d}", category, tuple(command)
        else:
            for index, row in enumerate(rows + (np.asarray(rows) * MIRROR).tolist(), 1):
                yield f"{category}_{index:03d}", category, tuple(row)


def geometry(robot, base, quat, leg_q, wheel_q):
    return tuple(np.stack(items) for items in zip(*(
        pose_geometry(robot, p, r, q, wheel)
        for p, r, q, wheel in zip(base, quat, leg_q, wheel_q)
    )))


def pose_geometry(robot, pos, quat, leg_q, wheel_q):
    robot.set_pose(pos, quat, leg_q, wheel_q)
    return robot.geometry()


def synthesize(robot, nominal, config, parameters, trace=None):
    dt = config["dt"]
    time = np.arange(round(config["motion_duration"]/dt) + 1) * dt
    if trace is None:
        trace = command_trace(time, nominal, config["rest_duration"], config["ramp_duration"])
    if trace.shape != (len(time), 3) or not np.isfinite(trace).all():
        raise ValueError("Expected a finite per-frame heading velocity trace of shape (T, 3)")
    base_nominal, yaw = integrate(trace, dt, robot.stand_height)
    base = base_nominal.copy()
    rot, quat = rotations(yaw), yaw_quat(yaw)
    planner = config["planner"]
    xy, lift, phase = plan(robot, base_nominal, rot, trace, dt, planner, parameters)
    contact = phase != SWING
    still_wheel = np.zeros((len(time), 4))
    correcting = np.any(~contact)
    passes = planner["correction_passes"] if correcting else 0
    for attempt in range(passes + 1):
        xy_local = (xy - base[:, None, :2]) @ rot[:, :2, :2]
        local, q, valid = robot.targets(xy_local, lift, base[:, 2],
                                        compensate_camber=correcting)
        if not valid.all():
            # Keep the last reachable pose for review and expose every failed target.
            previous = robot.stand_q.reshape(4, 3)
            for frame, supported in enumerate(valid):
                q[frame, ~supported] = previous[~supported]
                previous = q[frame]
        leg_q = q.reshape(-1, 12)
        target = local @ rot.transpose(0, 2, 1) + base[:, None, :]
        center, points, axis, leg_axes, com = geometry(robot, base, quat, leg_q, still_wheel)
        if attempt < passes:
            correction = base - base_nominal + support_correction(
                points, com, phase, dt, parameters[4], planner["support_goal"])
            magnitude = np.linalg.norm(correction, axis=1, keepdims=True)
            base = base_nominal + correction * np.minimum(
                1., planner["correction_limit"] / np.maximum(magnitude, 1e-12))

    base_linear = derivative(base, dt)
    base_angular = np.zeros_like(base)
    base_angular[:, 2] = derivative(yaw, dt)
    leg_qd = derivative(leg_q, dt)
    center_vel = derivative(center, dt)
    carrier = base_angular[:, None, :] + np.sum(leg_axes * leg_qd.reshape(-1, 4, 3, 1), axis=2)
    rates = wheel_rates(center_vel, carrier, axis, points - center)
    # Airborne spin is illustrative, with an interpolation between support rates.
    for wheel in range(4):
        anchors = np.flatnonzero(contact[:, wheel])
        rates[:, wheel] = np.interp(time, time[anchors], rates[anchors, wheel])
    moving = np.abs(rates) * robot.wheel_radius > planner["stance_speed"]
    if correcting:
        phase[contact & moving] = ROLLING
    wheel_q = trapezoid_integral(rates, dt)
    center, points, axis, _, com = geometry(robot, base, quat, leg_q, wheel_q)
    arm = points - center
    rolling_dir = np.cross(axis, UP)
    rolling_dir /= np.linalg.norm(rolling_dir, axis=-1, keepdims=True)
    contact_vel = center_vel + np.cross(carrier + rates[..., None] * axis, arm)
    rolling = np.sum(contact_vel * rolling_dir, axis=-1)
    lateral = np.sum(contact_vel * np.cross(UP, rolling_dir), axis=-1)
    margins = np.array([support_geometry(p[mask], c)[1] for p, mask, c in zip(points, contact, com)])
    return dict(
        time=time, command=trace, command_nominal=np.asarray(nominal),
        base_pos_nominal=base_nominal, base_pos=base, base_quat=quat,
        base_lin_vel=base_linear, base_ang_vel=base_angular, com_pos=com,
        leg_q=leg_q, leg_qd=leg_qd, wheel_q=wheel_q, wheel_qd=rates,
        wheel_center_target=target, wheel_center_fk=center, contact_point=points,
        contact_mask=contact, wheel_rate_valid=contact & valid & (np.abs(points[..., 2]) <= config["quality"]["contact_height"]),
        motion_phase=phase, ik_failed=~valid, ik_error=np.linalg.norm(target-center, axis=-1),
        rolling_residual=rolling, lateral_residual=lateral, support_margin=margins,
    )


def candidate_score(row):
    return (
        {"qualified": 0, "candidate": 1, "rejected": 2}[row["quality_status"]],
        row["collision_count"], row["ik_failure_frames"],
        len(row["failure_reasons"]),
        max(0., -row["min_support_margin"]) + row["lateral_residual_rms"] + row["vx_plateau_rmse"] + row["vy_plateau_rmse"],
    )


def generate_dataset(output, config_path=CONFIG, category=None, pilot=False):
    config = yaml.safe_load(Path(config_path).read_text())
    jobs = list(commands(config))
    if pilot:
        jobs = [
            next((job for job in jobs if job[1:] == (group, command)),
                 (f"pilot_{group}", group, command))
            for group, command in PILOT_COMMANDS
        ]
    if category:
        jobs = [job for job in jobs if job[1] == category]
        if not jobs and category != "standing":
            raise ValueError(f"No commands for {category}")
    if category in (None, "standing"):
        jobs.append(("standing", "standing", (0., 0., 0.)))
    if len({job[0] for job in jobs}) != len(jobs) or len({job[2] for job in jobs}) != len(jobs):
        raise ValueError("Commands or motion IDs are duplicated")
    if not (config["dt"] > 0 and config["ramp_duration"] > 0
            and np.isclose(round(config["motion_duration"] / config["dt"]) * config["dt"], config["motion_duration"])
            and 2*(config["rest_duration"] + config["ramp_duration"]) < config["motion_duration"]):
        raise ValueError("Expected an integral frame grid and a nonempty constant-speed interval")

    candidates = config["planner"]["swing_candidates"]
    if not candidates or any(len(row) != 5 or row[0] < 4*config["dt"] or min(row) <= 0 for row in candidates):
        raise ValueError("Expected nonempty, positive five-column swing candidates on the frame grid")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Motion bank output already exists: {output}")
    robot = Robot()
    output.mkdir(parents=True, exist_ok=False)
    (output / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "format": "go2w_kinematic_reference", "validation_level": "kinematic_only", "physics_validated": False,
        "generated_at": datetime.now().astimezone().isoformat(), "generator_commit": commit,
        "seed": config["seed"], "dt": config["dt"], "quaternion_order": "wxyz",
        "leg_joint_names": robot.joint_names[:12], "wheel_joint_names": robot.joint_names[12:],
        "actuator_names": robot.actuator_names, "wheel_order": LEGS,
        "model": str(MODEL.relative_to(ROOT)), "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "wheel_mesh_sha256": hashlib.sha256((MODEL.parent / "assets/wheel.stl").read_bytes()).hexdigest(),
        "control_config_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "motion_phase": {"STANCE": STANCE, "ROLLING": ROLLING, "SWING": SWING},
        "fields": {name: {"shape": shape, "units": units, "description": description}
                   for name, (shape, units, description) in FIELDS.items()},
        "motions": [],
    }
    started = perf_counter()
    for motion_id, group, nominal in jobs:
        clip_started = perf_counter()
        parameters = candidates if any(nominal[1:]) else candidates[:1]
        best = None
        trials = []
        for parameter in parameters:
            clip = synthesize(robot, nominal, config, parameter)
            assessment = check_clip(clip, nominal, robot, config)
            trials.append({"parameters": parameter, "quality_status": assessment["quality_status"],
                           "failure_reasons": assessment["failure_reasons"]})
            score = candidate_score(assessment)
            if best is None or score < best[0]:
                best = score, clip, assessment, parameter
            if assessment["quality_status"] == "qualified":
                break
        _, clip, assessment, selected = best
        path = Path("motions") / group / f"{motion_id}.npz"
        (output / path.parent).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / path, **clip)
        manifest["motions"].append({
            "motion_id": motion_id, "motion_category": group, "nominal_command": nominal,
            "duration": float(clip["time"][-1]), "frame_count": len(clip["time"]),
            "file_path": path.as_posix(), "quality_status": assessment["quality_status"],
            "failure_reasons": assessment["failure_reasons"], "generator_commit": commit,
            "seed": config["seed"], "generation_time": perf_counter() - clip_started,
            "planning_parameters": selected, "attempts": trials,
        })
        print(f"{motion_id:27s} {assessment['quality_status']:9s} {' '.join(assessment['failure_reasons'])}", flush=True)

    manifest["generation_seconds"] = perf_counter() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return audit_dataset(output, robot=robot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--category", choices=[*yaml.safe_load(CONFIG.read_text())["commands"], "standing"])
    args = parser.parse_args()
    try:
        generate_dataset(args.output_dir, args.config, args.category, args.pilot)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
