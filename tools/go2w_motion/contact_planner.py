"""Plan rolling support and single-wheel or diagonal swings on a flat plane."""

import numpy as np

from . import DIAGONALS, ROLLING, STANCE, SWING
from .kinematics import support_geometry
from .trajectories import derivative, smootherstep, swing, swing_bump


def plan(robot, base, rotation, command, dt, config, candidate, gait="crawl"):
    count = len(base)
    xy = base[:, None, :2] + robot.stand_xy @ rotation[:, :2, :2].transpose(0, 2, 1)
    lift = np.zeros((count, 4))
    phase = np.full((count, 4), ROLLING if np.any(command) else STANCE, dtype=np.int8)
    if not np.any(command[:, 1:]):
        return xy, lift, phase

    duration, height, preview, settle_time, _ = candidate
    steps = round(duration / dt)
    duration = steps * dt
    lead = round(preview / dt)
    settle = round(settle_time / dt)
    longitudinal = config["longitudinal_recenter"]
    heading = rotation[:, :2, 0]
    lateral = rotation[:, :2, 1]
    nominal = robot.stand_xy
    ideal = xy.copy()
    command_rate = derivative(command, dt)
    blend = np.linalg.norm(command, axis=1) / config["rest_blend_speed"]
    gain = longitudinal * smootherstep(blend)
    gain_rate = longitudinal * 30 * blend.clip(0, 1)**2 * (1-blend.clip(0, 1))**2 * derivative(blend, dt)
    limits = robot.action_limits.reshape(4, 3, 2)

    def rolling(frame, point, leg):
        rel = point - base[frame, :2]
        x, y = rel @ heading[frame], rel @ lateral[frame]
        vx, vy, wz = command[frame]
        vx_rate, _, wz_rate = command_rate[frame]
        # Longitudinal tracking with the wheel axis fixed by the hip chain.
        # Stop recentering at rest so the clip terminates at rest as well.
        error = nominal[leg, 0] - x
        speed = vx - wz*y + gain[frame]*error
        x_rate = speed - vx + wz*y
        y_rate = -vy - wz*x
        speed_rate = vx_rate - wz_rate*y - wz*y_rate + gain_rate[frame]*error - gain[frame]*x_rate
        return heading[frame]*speed, heading[frame]*speed_rate + lateral[frame]*(speed*wz)

    active_until = np.flatnonzero(np.any(command != 0, axis=1))[-1]
    first_twist = command[np.flatnonzero(np.any(command[:, 1:] != 0, axis=1))[0], 1:]
    order = np.array([0, 1, 2, 3] if first_twist[np.flatnonzero(first_twist)[0]] > 0 else [1, 0, 3, 2])
    free_at = swing_end = 0
    swinging = ()
    previous_pair = None
    progress = np.linspace(0, 1, steps + 1)
    for frame in range(count - 1):
        if frame >= free_at and frame + steps <= active_until:
            end = frame + steps
            future = min(end + lead, count - 1)
            urgency = np.abs((ideal[end] - xy[frame]) @ lateral[end])
            groups = []
            if urgency.max() >= config["step_trigger"]:
                if gait == "diagonal":
                    pairs = DIAGONALS if order[0] == 0 else DIAGONALS[::-1]
                    groups = sorted(
                        (pair for pair in pairs if pair != previous_pair),
                        key=lambda pair: -round(float(urgency[list(pair)].max()), 6))
                else:
                    # Preview the triangle after removing each possible swing wheel.
                    mid = (frame + end) // 2
                    contacts = xy[frame].copy()
                    for leg in range(4):
                        velocity, _ = rolling(frame, xy[frame, leg], leg)
                        contacts[leg] += velocity * ((mid - frame) * dt)
                    contacts += (0.055 * np.sign(nominal[:, 1, None])) * lateral[mid]
                    com = base[mid, :2] + robot.stand_com[:2] @ rotation[mid, :2, :2].T
                    margins = [support_geometry(contacts[np.arange(4) != leg], com)[1] for leg in range(4)]
                    merit = urgency + config["support_priority"] * np.minimum(margins, 0.)
                    rank = order[np.argsort(-np.round(merit[order], 6), kind="stable")]
                    minimum = max(config["step_trigger"], urgency.max() * 0.55)
                    groups = [(leg,) for leg in rank if urgency[leg] >= minimum]
            for group in groups:
                legs = list(group)
                future_y = (ideal[future, legs] - ideal[end, legs]) @ lateral[end]
                lead_y = np.clip(future_y, -config["max_landing_lead"], config["max_landing_lead"])
                landing = ideal[end, legs] + lead_y[:, None] * lateral[end]
                next_xy = nominal.copy()
                next_xy[legs] = (landing - base[end, :2]) @ rotation[end, :2, :2]
                _, landing_q, valid = robot.targets(
                    next_xy[None], np.zeros((1, 4)), robot.stand_height, compensate_camber=True)
                if not valid[0, legs].all():
                    continue
                if np.any(landing_q[0, legs] < limits[legs, :, 0]) or np.any(landing_q[0, legs] > limits[legs, :, 1]):
                    continue
                for leg, point in zip(group, landing):
                    v0, a0 = rolling(frame, xy[frame, leg], leg)
                    v1, a1 = rolling(end, point, leg)
                    xy[frame:end + 1, leg] = swing(xy[frame, leg], point, v0, v1, a0, a1, duration, progress)
                    lift[frame:end + 1, leg] = height * swing_bump(progress)
                    phase[frame + 1:end, leg] = SWING
                swinging, free_at, swing_end = group, end + settle, end
                previous_pair = group
                break
        for leg in range(4):
            if leg in swinging and frame < swing_end:
                continue
            v0, _ = rolling(frame, xy[frame, leg], leg)
            v1, _ = rolling(frame + 1, xy[frame, leg] + dt*v0, leg)
            xy[frame + 1, leg] = xy[frame, leg] + 0.5*dt*(v0 + v1)
            if phase[frame, leg] != SWING and np.linalg.norm(v0) < config["stance_speed"]:
                phase[frame, leg] = STANCE
    phase[-1] = STANCE if not np.any(command[-1]) else phase[-2]
    return xy, lift, phase


def support_correction(points, com, phase, dt, blend, support_goal):
    """Smooth the required inward CoM offsets before feeding them back to IK."""
    correction = np.zeros_like(com)
    width = max(1, round(blend / dt))
    for leg in range(4):
        active = phase[:, leg] == SWING
        boundaries = np.flatnonzero(np.diff(np.r_[False, active, False]))
        for start, end in boundaries.reshape(-1, 2):
            margins, normals = [], []
            for frame in range(start, end):
                _, margin, normal = support_geometry(points[frame, phase[frame] != SWING], com[frame])
                margins.append(margin)
                normals.append(normal)
            deficit = np.maximum(0., support_goal - np.asarray(margins))
            worst = deficit.argmax()
            if deficit[worst] == 0:
                continue
            idx = np.arange(max(0, start - width), min(len(com), end + width))
            envelope = smootherstep(np.minimum((idx-start+width)/width, (end-1+width-idx)/width))
            correction[idx, :2] += envelope[:, None] * deficit[worst] * normals[worst]
    return correction
