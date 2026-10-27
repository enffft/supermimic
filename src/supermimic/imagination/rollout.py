"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep, VelocityImagStep
from supermimic.imagination.tasks import ChaseBallTask, MotionTrackingTask, VelocityCommandTask, WorldPointReachTask
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_orientation_relative_obs, anchor_relative_obs, flatten_history_term_major
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
    term_features: torch.Tensor  # (H+1, n, 2), raw height/tilt feature pair
    root_height: torch.Tensor  # (H+1, n), absolute world-frame root height
    tilt: torch.Tensor  # (H+1, n), second termination feature
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict[str, float]


def _observation_noise(obs: torch.Tensor, num_joints: int) -> torch.Tensor:
    half = obs.new_tensor([0.01] * num_joints + [0.5] * num_joints + [0.2] * 3 + [0.05] * 3 + [0.0] * num_joints)
    return obs + (torch.rand_like(obs) * 2 - 1) * half


def imagine(
    wm: WorldModel,
    policy: Policy,
    task: MotionTrackingTask | VelocityCommandTask | ChaseBallTask | WorldPointReachTask | MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | VelocityImagStep | None = None,
    anchor_obs: bool = False,
    anchor_orientation_obs: bool = False,
) -> ImagRollout:
    """Roll H steps, calling the Actor at each newly predicted observation.

    ``MotionLibrary`` is accepted for compatibility with the existing G1 benchmark and
    is wrapped as a ``MotionTrackingTask``. Reference-free tasks keep the sampled task
    Goal fixed; world-point Actor goals are re-expressed against each predicted base pose.
    """
    if isinstance(task, MotionLibrary):
        task = MotionTrackingTask(task, future_frames=future_frames, anchor_obs=anchor_obs, anchor_orientation_obs=anchor_orientation_obs)
    is_motion = isinstance(task, MotionTrackingTask)
    is_world_point = isinstance(task, WorldPointReachTask)
    is_chase_ball = isinstance(task, ChaseBallTask)
    device = start.state.device
    n = start.state.shape[0]
    q0 = task.library.default_joint_pos if is_motion else task.default_joint_pos
    if step_fn is None:
        if is_motion:
            step_fn = ImagStep(wm, q0)
        else:
            step_fn = VelocityImagStep(wm, task, q0)
    anchor_obs = bool(anchor_obs or (is_motion and task.anchor_obs))
    anchor_orientation_obs = bool(anchor_orientation_obs or (is_motion and task.anchor_orientation_obs))
    if anchor_obs and anchor_orientation_obs:
        raise ValueError("choose position+orientation Anchor or orientation-only Anchor")
    if (anchor_obs or anchor_orientation_obs) and not is_motion:
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
    term_features, root_heights, tilts = [], [], []
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
            if not (task.anchor_obs or task.anchor_orientation_obs):
                return g
            if task.anchor_orientation_obs:
                rel = anchor_orientation_relative_obs(anchor_pose[..., 3:7], ref_t.body_quat_w[..., ai, :])
                half = rel.new_tensor([0.05] * 6)
            else:
                rel = anchor_relative_obs(
                    anchor_pose[..., :3], anchor_pose[..., 3:7],
                    ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :],
                )
                half = rel.new_tensor([0.25] * 3 + [0.05] * 6)
            if obs_noise:
                rel = rel + (torch.rand_like(rel) * 2 - 1) * half
            return torch.cat([g, rel], dim=-1)

        ref = ref_at(0)
        if task.anchor_obs or task.anchor_orientation_obs:
            body0 = wm.heads.robot_bodies(s)
            anchor_pose = torch.cat([body0["pos"][..., ai, :], body0["quat"][..., ai, :]], dim=-1)
        else:
            anchor_pose = None
        goal = motion_goal(frame, ref, anchor_pose)
        e_root = wm.heads.root_error_features(s, ref)
        reward_names = REWARD_TERM_NAMES
    else:
        if start.goal is None:
            raise ValueError("reference-free imagination requires the task Goal snapshot stored in ReplayBuffer")
        task_goal = start.goal
        goal = task.actor_goal(s, task_goal) if (is_world_point or is_chase_ball) else task_goal
        if goal.shape[-1] != task.goal_dim:
            raise ValueError(f"task Goal has {goal.shape[-1]} features; task expects {task.goal_dim}")
        frames_all = frame[None, :].expand(horizon + 1, n)
        ref = None
        anchor_pose = None
        e_root = torch.zeros(n, 4, device=device, dtype=s.dtype)
        reward_names = task.reward_term_names

    with torch.no_grad():
        initial_features = (
            wm.heads.termination_features(s[..., 0:3], s[..., 3:7], ref)
            if is_motion
            else task.termination_features(s) if hasattr(task, "termination_features")
            else wm.heads.free_termination_features(s[..., 0:3], s[..., 3:7], wm.cfg.termination_base_height)
        )
    term_features.append(initial_features)
    root_heights.append(s[..., 2])
    tilts.append(initial_features[..., 1])

    goals = [goal]
    e_roots = [e_root]
    term_sum = torch.zeros(len(reward_names), device=device)
    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0), num_joints=nj)
        # The policy is re-evaluated from the latest predicted state/history every step.
        a = policy(obs_flat, goals[-1])
        if is_motion:
            assert ref is not None
            transition_ref = ref
            s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, transition_ref)
            frame = frames_all[t + 1]
            ref = ref_at(t + 1)
            goal_next = motion_goal(frame, ref, anchor_pose)
            e_root = wm.heads.root_error_features(s_next, ref)
        else:
            s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, task_goal)
            if is_chase_ball:
                task_goal = task.advance_goal(task_goal)
                goal_next = task.actor_goal(s_next, task_goal)
            elif is_world_point:
                goal_next = task.actor_goal(s_next, task_goal)
            else:
                goal_next = task_goal
            e_root = torch.zeros(n, 4, device=device, dtype=s.dtype)

        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        with torch.no_grad():
            if is_motion:
                features = wm.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], transition_ref)
            elif hasattr(task, "termination_features"):
                features = task.termination_features(s_next)
            else:
                features = wm.heads.free_termination_features(
                    s_next[..., 0:3], s_next[..., 3:7], wm.cfg.termination_base_height
                )
        term_features.append(features)
        root_heights.append(s_next[..., 2].detach())
        tilts.append(features[..., 1])
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
        term_features=torch.stack(term_features),
        root_height=torch.stack(root_heights),
        tilt=torch.stack(tilts),
        reward_terms={k: float(v / horizon) for k, v in zip(reward_names, term_sum.tolist())},
    )
