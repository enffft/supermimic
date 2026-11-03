"""Aligned M1 reference, Actor and direct-Prior rollouts through one World Model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.reference import decode_future31, future31_to_m1_action22
from supermimic.policy.nets import M1ActionReferenceActor
from supermimic.tasks.action_contract import clamp_executed_action
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel


@dataclass
class M1Action22Rollout:
    """Both differentiable control paths and their aligned reference sequence.

    State paths are time-major ``[H+1,B,57]`` and action paths are
    ``[H,B,22]``. Reference tensors are batch-major ``[B,10,...]`` so the
    single conversion boundary remains visible to callers.
    """

    reference_future: torch.Tensor
    reference_actions: torch.Tensor
    reference_actions_unclipped: torch.Tensor
    reference_states: torch.Tensor
    actor_actions: torch.Tensor
    actor_executed_actions: torch.Tensor
    actor_states: torch.Tensor
    prior_direct_states: torch.Tensor
    actor_valid_mask: torch.Tensor
    prior_valid_mask: torch.Tensor
    action_diagnostics: dict[str, torch.Tensor]


def differentiable_m1_rollout(
    wm: WorldModel,
    actor: M1ActionReferenceActor,
    start: StartBatch,
    task_goal: torch.Tensor,
    reference: torch.Tensor,
    default_joint_pos: torch.Tensor,
    *,
    robot_spec,
    horizon: int = 10,
) -> M1Action22Rollout:
    """Roll the current reference through Actor→WM and directly through WM.

    Both paths start at the same real replay state and use the same fixed
    per-episode ``xi`` and normalized executed-action clamp. ``Future31[k]``
    supplies the action for interval ``k -> k+1`` (no one-frame shift).
    """
    if horizon < 1 or horizon > 10:
        raise ValueError("M1 Action22 BPTT horizon must be in [1,10]")
    batch = start.state.shape[0]
    if reference.shape != (batch, 10, 31):
        raise ValueError("reference must be [B,10,31]")
    if task_goal.shape != (batch, 8):
        raise ValueError("initial task_goal must be [B,8]")
    if start.state.shape != (batch, 57) or start.xi.shape[0] != batch:
        raise ValueError("M1 rollout requires replay State57 and xi rows")
    if start.hist_states.ndim != 3 or start.hist_states.shape[-1] != 57:
        raise ValueError("StartBatch history must be [history,B,57]")
    if start.hist_states.shape[1] != batch or start.hist_actions.shape[:2] != start.hist_states.shape[:2]:
        raise ValueError("history, previous actions and start-state batches do not match")
    if start.task_type is None or start.velocity_command is None or start.fixed_world_target is None:
        raise ValueError("M1 Action22 rollout requires named L/M task context in replay")
    if start.hist_states.shape[0] != actor.cfg.history:
        raise ValueError(f"Actor needs {actor.cfg.history} history frames, got {start.hist_states.shape[0]}")

    u_ref, u_ref_raw, action_diagnostics = future31_to_m1_action22(
        reference, robot_spec=robot_spec, return_diagnostics=True,
    )
    reference_states_batch = decode_future31(start.state, reference)

    wheel_indices = tuple(wm.spec.wheel_indices)
    obs_history = torch.stack([
        analytic_observation(
            start.hist_states[t], start.hist_actions[t], start.xi,
            default_joint_pos, wheel_indices,
        )
        for t in range(start.hist_states.shape[0])
    ], dim=0)

    actor_state = start.state
    direct_state = start.state
    actor_states = [actor_state]
    direct_states = [direct_state]
    actor_actions: list[torch.Tensor] = []
    actor_executed_actions: list[torch.Tensor] = []
    actor_valid_rows: list[torch.Tensor] = []
    prior_valid_rows: list[torch.Tensor] = []
    actor_alive = torch.ones(batch, dtype=torch.bool, device=start.state.device)
    prior_alive = torch.ones(batch, dtype=torch.bool, device=start.state.device)
    for k in range(horizon):
        obs_flat = flatten_history_term_major(obs_history, num_joints=wm.spec.nj)
        goal_k = task_goal8(
            actor_state, start.task_type, start.velocity_command,
            start.fixed_world_target,
        )
        u_k = u_ref[:, k, :]
        raw_action = actor(obs_flat, goal_k, u_k)
        executed = clamp_executed_action(raw_action)
        actor_next = wm.step(actor_state, executed, start.xi)
        direct_executed = clamp_executed_action(u_k)
        direct_next = wm.step(direct_state, direct_executed, start.xi)

        # Observation history follows the real environment contract: the
        # previous-action observation stores the policy output; WM state
        # transition uses the final bounded action.
        obs_next = analytic_observation(
            actor_next, raw_action, start.xi, default_joint_pos, wheel_indices,
        )
        obs_history = torch.cat((obs_history[1:], obs_next.unsqueeze(0)), dim=0)
        if hasattr(wm, "termination_logits"):
            actor_survived = torch.sigmoid(wm.termination_logits(actor_next.detach())).reshape(-1) < 0.5
            direct_survived = torch.sigmoid(wm.termination_logits(direct_next.detach())).reshape(-1) < 0.5
            actor_alive = actor_alive & actor_survived
            prior_alive = prior_alive & direct_survived
        actor_valid_rows.append(actor_alive)
        prior_valid_rows.append(prior_alive)
        actor_actions.append(raw_action)
        actor_executed_actions.append(executed)
        actor_state, direct_state = actor_next, direct_next
        actor_states.append(actor_state)
        direct_states.append(direct_state)

    return M1Action22Rollout(
        reference_future=reference,
        reference_actions=u_ref[:, :horizon],
        reference_actions_unclipped=u_ref_raw[:, :horizon],
        reference_states=reference_states_batch[:, :horizon],
        actor_actions=torch.stack(actor_actions, dim=0),
        actor_executed_actions=torch.stack(actor_executed_actions, dim=0),
        actor_states=torch.stack(actor_states, dim=0),
        prior_direct_states=torch.stack(direct_states, dim=0),
        actor_valid_mask=torch.stack(actor_valid_rows, dim=0),
        prior_valid_mask=torch.stack(prior_valid_rows, dim=0),
        action_diagnostics=action_diagnostics,
    )
