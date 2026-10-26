"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep, VelocityImagStep
from supermimic.imagination.tasks import MotionTrackingTask, VelocityCommandTask
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_relative_obs, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, S)
    actions: torch.Tensor  # (H, n, A)
    rewards: torch.Tensor  # (H, n), r_{t+1}
    dhat: torch.Tensor  # (H+1, n), predicted termination probability; d_0=0
    survival: torch.Tensor  # (H+1, n), c_t
    goals: torch.Tensor  # (H+1, n, G)
    e_root: torch.Tensor  # (H+1, n, 4); zeros for reference-free tasks
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict[str, float]


def _observation_noise(obs: torch.Tensor, num_joints: int) -> torch.Tensor:
    half = obs.new_tensor([0.01] * num_joints + [0.5] * num_joints + [0.2] * 3 + [0.05] * 3 + [0.0] * num_joints)
    return obs + (torch.rand_like(obs) * 2 - 1) * half


def imagine(
    wm: WorldModel,
    policy: Policy,
    task: MotionTrackingTask | VelocityCommandTask | MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | VelocityImagStep | None = None,
    anchor_obs: bool = False,
) -> ImagRollout:
    """Roll H steps, calling the Actor at each newly predicted observation.

    ``MotionLibrary`` is accepted for compatibility with the existing G1 benchmark and
    is wrapped as a ``MotionTrackingTask``. Velocity tasks keep the sampled replay Goal
    fixed for all H steps and do not synthesize a reference frame.
    """
    if isinstance(task, MotionLibrary):
        task = MotionTrackingTask(task, future_frames=future_frames, anchor_obs=anchor_obs)
    is_motion = isinstance(task, MotionTrackingTask)
    device = start.state.device
    n = start.state.shape[0]
    q0 = task.library.default_joint_pos if is_motion else task.default_joint_pos
    if step_fn is None:
        if is_motion:
            step_fn = ImagStep(wm, q0)
        else:
            step_fn = VelocityImagStep(wm, task, q0)
    if anchor_obs and not is_motion:
        raise ValueError("anchor-relative motion observations are unavailable for velocity-command tasks")

    nj = wm.spec.nj
    def noisy(o):
        return _observation_noise(o, nj) if obs_noise else o

    hist = [
        noisy(
            analytic_observation(
                start.hist_states[i], start.hist_actions[i], start.xi, q0, wm.spec.wheel_indices
            )
        )
        for i in range(start.hist_states.shape[0])
    ]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)
    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    ai = wm.heads.anchor_index

    if is_motion:
        lib = task.library
        last = lib.last_frame_of(frame)
        frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])
        refs_all = reference_at(lib, frames_all.reshape(-1))

        def ref_at(t: int) -> ReferenceFrame:
            sl = slice(t * n, (t + 1) * n)
            return ReferenceFrame(
                refs_all.body_pos_w[sl], refs_all.body_quat_w[sl], refs_all.body_lin_vel_w[sl],
                refs_all.body_ang_vel_w[sl], refs_all.root_pos_w[sl], refs_all.root_quat_w[sl],
            )

        def motion_goal(frame_t, ref_t, anchor_pose):
            g = lib.goal_window(frame_t, task.future_frames)
            if not task.anchor_obs:
                return g
            rel = anchor_relative_obs(
                anchor_pose[..., :3], anchor_pose[..., 3:7],
                ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :],
            )
            if obs_noise:
                half = rel.new_tensor([0.25] * 3 + [0.05] * 6)
                rel = rel + (torch.rand_like(rel) * 2 - 1) * half
            return torch.cat([g, rel], dim=-1)

        ref = ref_at(0)
        if task.anchor_obs:
            body0 = wm.heads.robot_bodies(s)
            anchor_pose = torch.cat([body0["pos"][..., ai, :], body0["quat"][..., ai, :]], dim=-1)
        else:
            anchor_pose = None
        goal = motion_goal(frame, ref, anchor_pose)
        e_root = wm.heads.root_error_features(s, ref)
        reward_names = REWARD_TERM_NAMES
    else:
        if start.goal is None:
            raise ValueError("velocity-command imagination requires the Goal snapshot stored in ReplayBuffer")
        goal = start.goal
        if goal.shape[-1] != task.goal_dim:
            raise ValueError(f"velocity Goal has {goal.shape[-1]} features; task expects {task.goal_dim}")
        frames_all = frame[None, :].expand(horizon + 1, n)
        ref = None
        anchor_pose = None
        e_root = torch.zeros(n, 4, device=device, dtype=s.dtype)
        reward_names = task.reward_term_names

    goals = [goal]
    e_roots = [e_root]
    term_sum = torch.zeros(len(reward_names), device=device)
    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0), num_joints=nj)
        # The policy is re-evaluated from the latest predicted state/history every step.
        a = policy(obs_flat, goals[-1])
        if is_motion:
            assert ref is not None
            s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, ref)
            frame = frames_all[t + 1]
            ref = ref_at(t + 1)
            goal_next = motion_goal(frame, ref, anchor_pose)
            e_root = wm.heads.root_error_features(s_next, ref)
        else:
            s_next, r, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, goal)
            term_vec = r.unsqueeze(0)
            goal_next = goal
            e_root = torch.zeros(n, 4, device=device, dtype=s.dtype)

        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)
        hist = hist[1:] + [noisy(obs_clean)]
        a_prev, s = a, s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(goal_next)
        e_roots.append(e_root)
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states), actions=torch.stack(actions), rewards=torch.stack(rewards),
        dhat=torch.stack(dhats), survival=torch.stack(survivals), goals=torch.stack(goals),
        e_root=torch.stack(e_roots), frames=torch.stack(frames),
        reward_terms={k: float(v / horizon) for k, v in zip(reward_names, term_sum.tolist())},
    )
