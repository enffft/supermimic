"""Heading commands, SE(2) integration and differentiable swing profiles."""

import numpy as np


def smootherstep(u):
    u = np.asarray(u).clip(0, 1)
    return u**3 * (10 + u * (-15 + 6 * u))


def swing_bump(u):
    return 64 * np.asarray(u)**3 * (1 - np.asarray(u))**3


def command_trace(time, nominal, rest, ramp):
    envelope = smootherstep((np.minimum(time, time[-1] - time) - rest) / ramp)
    return envelope[:, None] * nominal


def rotations(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack((c, -s, z, s, c, z, z, z, o), axis=-1).reshape(yaw.shape + (3, 3))


def integrate(command, dt, height):
    """Midpoint twist with exact constant-twist SE(2) exponential per interval."""
    mid = (command[:-1] + command[1:]) / 2
    turn = mid[:, 2] * dt
    yaw = np.r_[0., np.cumsum(turn)]
    heading_mid = yaw[:-1] + turn / 2
    heading_delta = mid.copy()
    heading_delta[:, 2] = 0
    delta = (rotations(heading_mid) @ heading_delta[..., None])[..., 0]
    position = np.vstack((np.zeros(3), np.cumsum(delta * (dt * np.sinc(turn / (2 * np.pi)))[:, None], 0)))
    position[:, 2] = height
    return position, yaw


def derivative(values, dt):
    return np.gradient(values, dt, axis=0, edge_order=2)


def trapezoid_integral(values, dt):
    steps = (values[1:] + values[:-1]) * (0.5 * dt)
    return np.concatenate((np.zeros_like(values[:1]), np.cumsum(steps, axis=0)))


def yaw_quat(yaw):
    q = np.zeros((len(yaw), 4))
    q[:, 0], q[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    return q


def heading_velocity(linear_world, angular_world, rotation):
    return np.column_stack((np.einsum("tji,tj->ti", rotation, linear_world)[:, :2],
                            angular_world[:, 2]))


def swing(start, end, v_start, v_end, a_start, a_end, duration, u):
    """Quintic Hermite interpolation with position, velocity and acceleration at both ends."""
    u = np.asarray(u)[..., None]
    c0 = np.asarray(start)
    c1 = duration * v_start
    c2 = 0.5 * duration**2 * a_start
    dp = end - c0 - c1 - c2
    dv = duration * v_end - c1 - 2*c2
    da = duration**2 * a_end - 2*c2
    c3 = 10*dp - 4*dv + 0.5*da
    c4 = -15*dp + 7*dv - da
    c5 = 6*dp - 3*dv + 0.5*da
    return c0 + u*(c1 + u*(c2 + u*(c3 + u*(c4 + u*c5))))
