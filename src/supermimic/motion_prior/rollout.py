"""10-step Prior-conditioned Actor BPTT through the frozen physical World Model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.reference import decode_future31
from supermimic.policy.nets import M1PriorConditionedActor
from supermimic.tasks.action_contract import clamp_executed_action
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel


@dataclass
class M1JointRollout:
    states: torch.Tensor       # [H+1,B,57]
    actions: torch.Tensor      # [H,B,22]
    reference: torch.Tensor    # [B,10,31], single reference plan for all phases
    reference_states: torch.Tensor  # [B,10,57]


def differentiable_m1_rollout(
    wm: WorldModel,
    actor: M1PriorConditionedActor,
    start: StartBatch,
    task_goal: torch.Tensor,
    reference: torch.Tensor,
    default_joint_pos: torch.Tensor,
    horizon: int = 10,
) -> M1JointRollout:
    """Roll Actor actions through WM while preserving dLoss/dReference paths."""
    if horizon < 1 or horizon > 10:
        raise ValueError("M1 joint BPTT horizon must be in [1,10]")
    if reference.shape != (start.state.shape[0], 10, 31):
        raise ValueError("reference must be [B,10,31]")
    if task_goal.shape != (start.state.shape[0], 8):
        raise ValueError("initial task_goal must be [B,8]")
    if start.hist_states.ndim != 3 or start.hist_states.shape[-1] != 57:
        raise ValueError("StartBatch history must be [history,B,57]")
    if start.hist_states.shape[1] != start.state.shape[0]:
        raise ValueError("history and start-state batch sizes do not match")
    if start.task_type is None or start.velocity_command is None or start.fixed_world_target is None:
        raise ValueError("M1 joint rollout requires named task context in replay")
    if start.hist_states.shape[0] != actor.cfg.history:
        raise ValueError(f"Actor needs {actor.cfg.history} history frames, got {start.hist_states.shape[0]}")

    wheel_indices = wm.spec.wheel_indices
    observations = [
        analytic_observation(start.hist_states[t], start.hist_actions[t], start.xi,
                             default_joint_pos, wheel_indices)
        for t in range(start.hist_states.shape[0])
    ]
    hist = torch.stack(observations, dim=0)
    s = start.state
    states = [s]
    actions = []
    for phase in range(horizon):
        obs_flat = flatten_history_term_major(hist, num_joints=wm.spec.nj)
        goal_now = task_goal if phase == 0 else task_goal8(
            s, start.task_type, start.velocity_command, start.fixed_world_target,
        )
        a = actor(obs_flat, goal_now, reference, torch.full(
            (len(s),), phase, dtype=s.dtype, device=s.device,
        ))
        a_exec = clamp_executed_action(a, 1.0)
        s_next = wm.step(s, a_exec, start.xi)
        obs_next = analytic_observation(s_next, a, start.xi, default_joint_pos, wheel_indices)
        hist = torch.cat((hist[1:], obs_next[None]), dim=0)
        states.append(s_next)
        actions.append(a)
        s = s_next
    state_seq = torch.stack(states, dim=0)
    action_seq = torch.stack(actions, dim=0)
    ref_states = decode_future31(start.state, reference)
    return M1JointRollout(state_seq, action_seq, reference, ref_states)
