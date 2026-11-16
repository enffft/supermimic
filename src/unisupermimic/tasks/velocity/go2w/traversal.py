"""Privileged static targets, differentiable wheel geometry and one-event success."""

import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec


def wheel_centers(state):
    """Exact link-center FK for the audited MJCF leg chain (no contact inputs)."""
    p, r, _, _, q, _ = state
    q = q.reshape(*q.shape[:-1], 4, 3)
    hip, thigh, calf = q.unbind(-1)
    # Ry(thigh) * ([0,0,-.213] + Ry(calf)*[0,0,-.2264]).
    x = -0.213 * thigh.sin() - 0.2264 * (thigh + calf).sin()
    z = -0.213 * thigh.cos() - 0.2264 * (thigh + calf).cos()
    y = p.new_tensor([0.0955, -0.0955, 0.0955, -0.0955]).expand_as(z)
    local = torch.stack((x, hip.cos() * y - hip.sin() * z, hip.sin() * y + hip.cos() * z), -1)
    local = local + p.new_tensor(
        [[0.1934, 0.0465, 0], [0.1934, -0.0465, 0], [-0.1934, 0.0465, 0], [-0.1934, -0.0465, 0]]
    )
    return p[..., None, :] + torch.einsum("...ij,...kj->...ki", r, local)


def reset_targets(command, ids):
    origin = command._origins[ids]
    domain = command.command_domain[ids]
    delta = torch.where(domain >= 6, 2.8, 1.8)
    target = origin.clone()
    target[:, 0] += delta
    target[:, 2] = command.mdp.terrain.ground_height(target[:, :2]) + 0.36
    start = origin.clone()
    start[:, 2] = command.mdp.terrain.ground_height(start[:, :2]) + command.cfg.spawn_height
    meta = command.traversal[ids].clone()
    meta[:, :3] = target
    meta[:, 3:5] = origin[:, :2] + origin.new_tensor([0.8, 0])
    # Multi-stair pass requires all four wheels beyond the final riser.
    meta[domain >= 6, 3] += 1.2
    meta[:, 5:7] = origin.new_tensor([1.0, 0.0])
    meta[:, 7] = target[:, 2] - 0.36
    meta[:, 8] = (target - start).norm(dim=-1).clamp_min(0.1)
    meta[:, 9] = (domain >= 4).float()
    command.traversal[ids] = meta
    command.target_id[ids] = (command.command_counter[ids] + 1) * command.num_envs + ids
    command.success_peak[ids] = 0
    command.success_hold[ids] = 0
    command.success_event[ids] = False


def success_proxy(state, meta):
    wheels = wheel_centers(state)
    beyond = ((wheels[..., :2] - meta[..., None, 3:5]) * meta[..., None, 5:7]).sum(-1)
    clearance = torch.sigmoid((beyond.amin(-1) - 0.05) / 0.05)
    lateral = (state[0][..., 1] - meta[..., 1]).abs()
    upright = torch.sigmoid((state[1][..., 2, 2] - 0.8) / 0.05)
    height = torch.exp(-((state[0][..., 2] - meta[..., 2]) / 0.12).square())
    return meta[..., 9] * clearance * upright * height * torch.exp(-(lateral / 0.5).square())


def reward_terms(
    previous, state, meta, peak, gamma, progress_weight, success_weight, dt, horizon=1
):
    # Shift potential to zero at spawn: standing at spawn yields zero shaping.
    phi0 = 1 - (previous[0] - meta[..., :3]).norm(dim=-1) / meta[..., 8]
    phi1 = 1 - (state[0] - meta[..., :3]).norm(dim=-1) / meta[..., 8]
    progress = meta[..., 9] * progress_weight * (gamma * phi1 - phi0)
    proxy = success_proxy(state, meta)
    values = proxy.reshape(horizon, -1)
    initial = peak.reshape(horizon, -1)[0]
    maxima = torch.maximum(values.cummax(0).values, initial[None])
    preceding = torch.cat((initial[None], maxima[:-1]), 0)
    increment = (maxima - preceding).clamp_min(0).reshape_as(proxy)
    # Caller multiplies every rate by dt, so potentials are converted to rates.
    return dict(stair_progress=progress / dt, stair_success=success_weight * increment / dt)


def observe_success(command, state):
    meta = command.traversal
    wheels = wheel_centers(state)
    field = command.mdp.terrain
    beyond = ((wheels[..., :2] - meta[:, None, 3:5]) * meta[:, None, 5:7]).sum(-1).amin(-1) > 0.05
    support = (
        wheels[..., 2] - field.ground_height(wheels[..., :2]) - command.mdp.reward_cfg.wheel_radius
    ).abs().amax(-1) < 0.10
    valid = (
        (meta[:, 9] > 0)
        & beyond
        & support
        & (state[1][:, 2, 2] > 0.8)
        & (~command.mdp.hard_termination(state))
    )
    valid &= (state[0][:, 2] - meta[:, 7] > 0.22) & ((state[0][:, 1] - meta[:, 1]).abs() < 0.5)
    command.success_hold.copy_(torch.where(valid, command.success_hold + 1, 0))
    command.success_event.copy_(command.success_hold >= 10)
    command.curriculum.traversal_success |= command.success_event
    command.success_peak.copy_(torch.maximum(command.success_peak, success_proxy(state, meta)))
