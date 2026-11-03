"""Staged Action22-reference, Actor and World Model training for M1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses, planner_losses
from supermimic.motion_prior.reference import decode_future31
from supermimic.motion_prior.rollout import differentiable_m1_rollout
from supermimic.policy.nets import M1ActionReferenceActor
from supermimic.robots.spec import M1_JOINTS, RobotSpec, get_robot_spec
from supermimic.tasks.action_contract import mixed_m1_setpoints, serialize_m1_action_contract
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


@contextmanager
def frozen_parameters(*modules):
    """Freeze parameters without detaching inputs or blocking BPTT gradients."""
    previous = [[p.requires_grad for p in module.parameters()] for module in modules]
    try:
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        yield
    finally:
        for module, flags in zip(modules, previous):
            for parameter, required in zip(module.parameters(), flags):
                parameter.requires_grad_(required)


def module_sha256(module: torch.nn.Module) -> str:
    """Deterministic digest for a model state used by fail-closed evidence gates."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class M1AlternatingTrainer:
    """Own the v4 Actor/Prior/WM optimizers and their phase/gate boundaries."""

    FORMAT_VERSION = 4
    CHECKPOINT_FORMAT = "supermimic_m1_action22_reference_v1"
    ACTOR_SCHEMA = "m1_action_reference_actor_v1"
    REFERENCE_SCHEMA = "future31_to_interleaved_action22_v1"

    def __init__(self, *, wm: WorldModel, wm_trainer: WorldModelTrainer, prior,
                 actor: M1ActionReferenceActor, buffer: ReplayBuffer,
                 default_joint_pos: torch.Tensor, soft_limits: torch.Tensor,
                 fk, tcp_index: int, cfg, robot_joint_names: tuple[str, ...],
                 robot_model_sha256: str | None, command,
                 robot_spec: RobotSpec | None = None):
        if buffer.task_type is None:
            raise ValueError("M1 Action22 trainer requires named L/M Replay context")
        if (int(cfg.prior_horizon), int(cfg.actor_horizon), int(cfg.replanning_cadence)) != (10, 10, 10):
            raise ValueError("Action22 v1 requires Future31 horizon and replanning cadence of 10")
        if cfg.state_source != "privileged":
            raise NotImplementedError("estimated State57 input has no validated M1 estimator connected")
        if cfg.training_mode != "staged_action22_v1":
            raise ValueError("M1 V1 uses the single staged_action22_v1 training state machine")
        if int(cfg.prior_updates_per_event) > int(cfg.actor_updates_per_event):
            raise ValueError("Prior updates cannot outnumber Actor updates")
        if cfg.batch_size <= 0 or cfg.micro_batch_size <= 0 or cfg.batch_size % cfg.micro_batch_size:
            raise ValueError("M1 batch_size must be divisible by positive micro_batch_size")
        self.wm, self.wm_trainer = wm, wm_trainer
        self.prior, self.actor, self.buffer = prior, actor, buffer
        self.default_joint_pos, self.soft_limits = default_joint_pos, soft_limits
        self.fk, self.tcp_index, self.cfg = fk, int(tcp_index), cfg
        self.loss_cfg: M1JointLossCfg = cfg.loss
        self.robot_joint_names = tuple(robot_joint_names)
        self.robot_model_sha256 = robot_model_sha256
        self.robot_spec = robot_spec or get_robot_spec(
            "m1_piper", m1_pose_profile="dataset_carry", m1_action_contract="v3",
        )
        if self.robot_spec.action_contract_version != "m1_mixed_action_v3":
            raise ValueError("Action22-reference training requires Action Contract v3")
        self.command = command
        self.prior_initial = deepcopy(prior).eval()
        self.prior_initial.requires_grad_(False)
        self.prior_initial.source = dict(getattr(prior, "source", {}))
        self.prior_optimizer = torch.optim.Adam(self.prior.parameters(), lr=cfg.prior_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.wm_update_events = 0
        self.wm_optimizer_steps = 0
        self.prior_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.stage = "wm_warmup"
        self._actor_ready_window: deque[dict[str, float]] = deque(maxlen=int(cfg.readiness_window_steps))
        self._real_metrics: dict[str, float] = {}
        self._plan: torch.Tensor | None = None
        self._reference_actions: torch.Tensor | None = None
        self._reference_actions_unclipped: torch.Tensor | None = None
        self._plan_anchor_state: torch.Tensor | None = None
        self._plan_goal: torch.Tensor | None = None
        self.last_prior_metrics: dict[str, float] = {}
        self.last_actor_metrics: dict[str, float] = {}
        self.last_wm_metrics: dict[str, float] = {}
        self.last_wm_readiness: dict[str, float] = {}
        self._last_gate_diagnostics: dict[str, float] = {}
        self._prior_calibration_wm: WorldModel | None = None
        self._prior_calibration_wm_sha256: str | None = None
        self._prior_calibration_checkpoint: str | None = None
        self._prior_calibration_checkpoint_signature: tuple[str, int, int] | None = None
        self._prior_calibration_backup: dict | None = None
        self._prior_calibration_status = "not_started"
        self._prior_calibration_rolled_back_updates = 0
        self._prior_calibration_validation: dict | None = None
        self._force_prior_frozen = False

    def set_lr(self, *, prior_lr: float, actor_lr: float) -> None:
        for group in self.prior_optimizer.param_groups:
            group["lr"] = float(prior_lr)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(actor_lr)

    @property
    def actor_ready(self) -> bool:
        return bool(self.actor_readiness_diagnostics["schedule/actor_ready"])

    @property
    def actor_readiness_diagnostics(self) -> dict[str, float]:
        count = len(self._actor_ready_window)
        enough_samples = count >= int(self.cfg.readiness_min_samples)
        means = ({key: sum(row[key] for row in self._actor_ready_window) / count
                  for key in self._actor_ready_window[0]} if count else {})
        checks = {
            "samples": enough_samples,
            "pretrain": self.actor_optimizer_steps >= int(self.cfg.actor_pretrain_updates),
            "survival": enough_samples and means.get("survival", 0.0) >= self.cfg.readiness_min_survival,
            "root_tracking": enough_samples and means.get("root_tracking_error", float("inf")) <= self.cfg.readiness_max_root_tracking_error,
            "arm_tracking": enough_samples and means.get("arm_tracking_error", float("inf")) <= self.cfg.readiness_max_arm_tracking_error,
            "wheel_tracking": enough_samples and means.get("wheel_tracking_error", float("inf")) <= self.cfg.readiness_max_wheel_tracking_error,
            "action_saturation": enough_samples and means.get("action_saturation", float("inf")) <= self.cfg.readiness_max_action_saturation,
        }
        result = {"schedule/actor_ready": float(all(checks.values())), "schedule/actor_readiness_samples": float(count)}
        result.update({f"schedule/actor_readiness_{key}_pass": float(value) for key, value in checks.items()})
        result.update({f"schedule/actor_readiness_mean_{key}": float(value) for key, value in means.items()})
        return result

    @staticmethod
    def _prior_state_history(start) -> torch.Tensor:
        return start.hist_states[-4:].permute(1, 0, 2).contiguous()

    def _sample_prior(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.cfg.prior_mode == "prior_mean":
            return self.prior(history, goal, mode="prior_mean")
        if self.cfg.prior_mode != "prior_sample":
            raise ValueError(f"unsupported Prior mode {self.cfg.prior_mode!r}")
        eps = torch.randn((len(goal), self.prior.model.latent_dim), dtype=goal.dtype, device=goal.device)
        return self.prior(history, goal, mode="prior_sample", latent_eps=eps)

    @staticmethod
    def _material_goal_change(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        task_change = (previous[:, 6:8] - current[:, 6:8]).abs().amax(-1) > 0.5
        is_l = current[:, 6] > 0.5
        velocity_change = (previous[:, :3] - current[:, :3]).abs().amax(-1) > 0.025
        # M's relative target changes naturally with base motion. Replan only
        # for a material world-target displacement, not every moving step.
        ee_change = (previous[:, 3:6] - current[:, 3:6]).abs().amax(-1) > 0.08
        return task_change | torch.where(is_l, velocity_change, ee_change)

    @torch.no_grad()
    def act(self, proprio_history: torch.Tensor, goal: torch.Tensor,
            state_history: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replan on cadence/reset/material Goal switch and use only current U_ref[k]."""
        batch = len(goal)
        if state_history.shape != (batch, 4, 57):
            raise ValueError("online M1 planner requires State57 history [B,4,57]")
        if proprio_history.shape != (batch, self.actor.cfg.history * 72):
            raise ValueError("online Actor proprioception does not match its saved history schema")
        phase = phase.reshape(batch).to(device=goal.device, dtype=torch.long)
        if self._plan is None or self._plan.shape[0] != batch:
            self._plan = torch.zeros(batch, 10, 31, dtype=goal.dtype, device=goal.device)
            self._reference_actions = torch.zeros(batch, 10, 22, dtype=goal.dtype, device=goal.device)
            self._reference_actions_unclipped = torch.zeros_like(self._reference_actions)
            self._plan_anchor_state = torch.zeros(batch, 57, dtype=goal.dtype, device=goal.device)
            self._plan_goal = torch.full_like(goal, float("nan"))
        changed = torch.zeros(batch, dtype=torch.bool, device=goal.device)
        finite_old = torch.isfinite(self._plan_goal).all(-1)
        changed[finite_old] = self._material_goal_change(self._plan_goal[finite_old], goal[finite_old])
        replan = (phase == 0) | changed | ~finite_old
        phase[replan] = 0  # reset phase on a material command/task switch
        indices = replan.nonzero(as_tuple=False).flatten()
        if indices.numel():
            future = self._sample_prior(state_history[indices], goal[indices])
            from supermimic.motion_prior.reference import future31_to_m1_action22

            actions, raw, _ = future31_to_m1_action22(
                future, robot_spec=self.robot_spec, return_diagnostics=True,
            )
            self._plan[indices] = future
            self._reference_actions[indices] = actions
            self._reference_actions_unclipped[indices] = raw
            self._plan_anchor_state[indices] = state_history[indices, -1]
            self._plan_goal[indices] = goal[indices]
        rows = torch.arange(batch, device=goal.device)
        current_reference = self._reference_actions[rows, phase]
        action = self.actor(proprio_history, goal, current_reference)
        return action, self._plan

    @torch.no_grad()
    def record_real_step(self, next_state: torch.Tensor, action: torch.Tensor,
                         phase: torch.Tensor, terminated: torch.Tensor, *,
                         executed_action: torch.Tensor | None = None,
                         physical_setpoint: torch.Tensor | None = None) -> dict[str, float]:
        if self._plan is None or self._plan_anchor_state is None:
            raise RuntimeError("real-step metrics require an active Future31 plan")
        phase = phase.reshape(-1).long().clamp(0, 9)
        ref_state = decode_future31(self._plan_anchor_state, self._plan)
        rows = torch.arange(len(phase), device=phase.device)
        target = ref_state[rows, phase]
        arm_ids = [M1_JOINTS.index(f"piper_x_joint{i}") for i in range(1, 7)]
        leg_ids = [M1_JOINTS.index(name) for name in self.robot_spec.amp_joint_names[:12]]
        wheel_ids = [M1_JOINTS.index(n) for n in self.robot_spec.wheel_joint_names]
        root_error = torch.linalg.vector_norm(next_state[:, :3] - target[:, :3], dim=-1)
        arm_error = (next_state[:, 7:29][:, arm_ids] - target[:, 7:29][:, arm_ids]).square().mean(-1).sqrt()
        leg_error = (next_state[:, 7:29][:, leg_ids] - target[:, 7:29][:, leg_ids]).square().mean(-1).sqrt()
        wheel_error = (next_state[:, 35:57][:, wheel_ids] - target[:, 35:57][:, wheel_ids]).square().mean(-1).sqrt()
        from supermimic.kinematics.rotations import quat_apply_inverse

        gravity = torch.zeros_like(next_state[:, :3]); gravity[:, 2] = -1.0
        gravity_body = quat_apply_inverse(next_state[:, 3:7], gravity)
        base_tilt = torch.acos((-gravity_body[:, 2]).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        executed = action if executed_action is None else executed_action
        if physical_setpoint is None:
            wheel_mask = torch.tensor(
                [name in self.robot_spec.wheel_joint_names for name in self.robot_joint_names],
                dtype=torch.bool, device=action.device,
            )
            physical_setpoint = mixed_m1_setpoints(
                executed, self.default_joint_pos.to(action), action.new_tensor(self.robot_spec.action_scales),
                wheel_mask, self.soft_limits.to(action),
                action.new_tensor(self.robot_spec.position_target_bounds),
            )
        position_mask = torch.tensor(
            [name not in self.robot_spec.wheel_joint_names for name in self.robot_joint_names],
            dtype=torch.bool, device=action.device,
        )
        soft_limits = self.soft_limits.to(physical_setpoint)
        at_soft_limit = ((physical_setpoint <= soft_limits[:, 0] + 1e-5)
                         | (physical_setpoint >= soft_limits[:, 1] - 1e-5)) & position_mask
        mapping_bounds = action.new_tensor(self.robot_spec.position_target_bounds)
        at_action_limit = ((physical_setpoint <= mapping_bounds[:, 0] + 1e-5)
                           | (physical_setpoint >= mapping_bounds[:, 1] - 1e-5)) & position_mask
        metrics = {
            "survival": float((~terminated.bool()).float().mean()),
            "root_tracking_error": float(root_error.mean()),
            "leg_tracking_error": float(leg_error.mean()),
            "arm_tracking_error": float(arm_error.mean()),
            "wheel_tracking_error": float(wheel_error.mean()),
            "action_saturation": float((action.abs() >= 0.98).float().mean()),
            "executed_action_saturation": float((executed.abs() >= 0.98).float().mean()),
            "joint_target_soft_limit_saturation": float(at_soft_limit.float().sum() / position_mask.sum().clamp_min(1) / len(action)),
            "joint_target_action_envelope_saturation": float(at_action_limit.float().sum() / position_mask.sum().clamp_min(1) / len(action)),
            "base_height_m": float(next_state[:, 2].mean()),
            "base_tilt_rad": float(base_tilt.mean()),
        }
        for name, joint_id in zip(self.robot_spec.wheel_joint_names, wheel_ids):
            metrics[f"wheel_reference_rate/{name}"] = float(target[:, 35 + joint_id].mean())
            metrics[f"wheel_actual_rate/{name}"] = float(next_state[:, 35 + joint_id].mean())
            metrics[f"wheel_rate_error/{name}"] = float((next_state[:, 35 + joint_id] - target[:, 35 + joint_id]).square().mean().sqrt())
        for axis, axis_name in enumerate(("x", "y", "z")):
            metrics[f"prior/reference_root_delta_{axis}_m"] = float(self._plan[:, -1, axis].mean())
        self._actor_ready_window.append(metrics)
        self._real_metrics = {f"real/{key}": value for key, value in metrics.items()}
        return self._real_metrics

    def _one_micro_rollout(self, micro: int, *, planner: bool):
        start = self.buffer.sample_starts(
            micro, self.actor.cfg.history,
            task_balanced=bool(getattr(self.cfg, "balanced_replay_sampling", False)),
        )
        goal = task_goal8(start.state, start.task_type, start.velocity_command, start.fixed_world_target)
        state_history = self._prior_state_history(start)
        if planner:
            self.last_prior_input_diagnostics = self.prior.input_diagnostics(state_history, goal)
            reference = self._sample_prior(state_history, goal)
            with torch.no_grad():
                anchor = self.prior_initial(state_history, goal, mode="prior_mean")
        else:
            with torch.no_grad():
                reference = self._sample_prior(state_history, goal)
            reference = reference.detach()
            anchor = None
        rollout_wm = self._prior_calibration_wm if planner and self._prior_calibration_wm is not None else self.wm
        rollout = differentiable_m1_rollout(
            rollout_wm, self.actor, start, goal, reference, self.default_joint_pos,
            robot_spec=self.robot_spec,
            horizon=int(self.cfg.prior_horizon if planner else self.cfg.actor_horizon),
        )
        if planner:
            terms = planner_losses(
                rollout.prior_direct_states,
                rollout.reference_actions.transpose(0, 1),
                start.state, reference, anchor,
                start.task_type, start.velocity_command, start.fixed_world_target,
                fk=self.fk, tcp_index=self.tcp_index, soft_limits=self.soft_limits,
                cfg=self.loss_cfg,
                actor_states=rollout.actor_states,
                raw_reference_actions=rollout.reference_actions_unclipped,
                action_diagnostics=rollout.action_diagnostics,
                direct_valid_mask=rollout.prior_valid_mask,
                actor_valid_mask=rollout.actor_valid_mask,
                direct_safety_mask=rollout.prior_safety_mask,
                actor_safety_mask=rollout.actor_safety_mask,
                direct_survival_probability=rollout.prior_survival_probability,
                actor_survival_probability=rollout.actor_survival_probability,
            )
        else:
            terms = actor_losses(
                rollout.actor_states[1:], rollout.reference_states.transpose(0, 1),
                rollout.actor_actions, start.prev_action,
                soft_limits=self.soft_limits, cfg=self.loss_cfg,
                task_type=start.task_type, velocity_command=start.velocity_command,
                fixed_world_target=start.fixed_world_target, fk=self.fk,
                tcp_index=self.tcp_index,
                reference_actions=rollout.reference_actions.transpose(0, 1),
                initial_state=start.state,
                valid_mask=rollout.actor_valid_mask,
                safety_mask=rollout.actor_safety_mask,
                survival_probability=rollout.actor_survival_probability,
            )
        if start.task_type is not None:
            is_l = (start.task_type == 0).to(start.state.dtype)
            is_m = (start.task_type == 1).to(start.state.dtype)
            terms["replay/task_l_count"] = is_l.sum().detach()
            terms["replay/task_m_count"] = is_m.sum().detach()
            terms["imag/actor_valid_fraction_l"] = (
                rollout.actor_valid_mask.to(start.state.dtype) * is_l[None]
            ).sum() / (rollout.actor_valid_mask.shape[0] * is_l.sum()).clamp_min(1.0)
            terms["imag/actor_valid_fraction_m"] = (
                rollout.actor_valid_mask.to(start.state.dtype) * is_m[None]
            ).sum() / (rollout.actor_valid_mask.shape[0] * is_m.sum()).clamp_min(1.0)
            terms["imag/prior_valid_fraction_l"] = (
                rollout.prior_valid_mask.to(start.state.dtype) * is_l[None]
            ).sum() / (rollout.prior_valid_mask.shape[0] * is_l.sum()).clamp_min(1.0)
            terms["imag/prior_valid_fraction_m"] = (
                rollout.prior_valid_mask.to(start.state.dtype) * is_m[None]
            ).sum() / (rollout.prior_valid_mask.shape[0] * is_m.sum()).clamp_min(1.0)
            terms["replay/task_l_fraction"] = is_l.mean().detach()
        return terms, rollout

    def _update_prior(self) -> dict[str, float]:
        if self._prior_calibration_backup is None:
            self._prior_calibration_backup = {
                "prior": deepcopy(self.prior.state_dict()),
                "optimizer": deepcopy(self.prior_optimizer.state_dict()),
                "optimizer_steps": int(self.prior_optimizer_steps),
                "prior_sha256": module_sha256(self.prior),
            }
            self._prior_calibration_status = "snapshot_saved"
        n_micro = int(self.cfg.batch_size) // int(self.cfg.micro_batch_size)
        self.prior_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.wm_trainer.opt.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        try:
            planner_wm = self._prior_calibration_wm or self.wm
            with frozen_parameters(self.actor, planner_wm):
                for _ in range(n_micro):
                    terms, _ = self._one_micro_rollout(int(self.cfg.micro_batch_size), planner=True)
                    if not torch.isfinite(terms["total"]):
                        raise FloatingPointError("non-finite M1 Planner loss")
                    (terms["total"] / n_micro).backward()
                    for name, value in terms.items():
                        if value.ndim == 0:
                            divisor = 1 if name in ("replay/task_l_count", "replay/task_m_count") else n_micro
                            sums[name] = sums.get(name, 0.0) + float(value.detach()) / divisor
                prior_parameters = [p for p in self.prior.parameters() if p.requires_grad and p.grad is not None]
                if not prior_parameters:
                    raise RuntimeError("Planner loss has no gradient path to Motion Prior parameters")
                probe = prior_parameters[0]
                before = probe.detach().clone()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    prior_parameters, self.cfg.grad_clip_norm, error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                    raise RuntimeError(f"invalid Motion Prior gradient norm {float(grad_norm)}")
                if any(p.grad is not None for p in self.actor.parameters()) or any(p.grad is not None for p in self.wm.parameters()):
                    raise RuntimeError("Prior update violated Actor/WM optimizer ownership")
                self.prior_optimizer.step()
                if torch.equal(before, probe.detach()):
                    raise RuntimeError("Motion Prior optimizer step did not change parameters")
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.wm_trainer.opt.zero_grad(set_to_none=True)
            self.prior_optimizer.zero_grad(set_to_none=True)
        self.prior_optimizer_steps += 1
        self._prior_calibration_status = "in_progress"
        sums["grad_norm"] = float(grad_norm)
        sums["reference_shift"] = self._reference_shift()
        self.last_prior_metrics = {f"prior/{key}": value for key, value in sums.items()}
        self.last_prior_metrics.update({key: float(value.detach()) for key, value in getattr(self, "last_prior_input_diagnostics", {}).items()})
        self.last_prior_metrics["imag/planner_direct_task_return"] = -sums.get("task_direct", float("nan"))
        self.last_prior_metrics.update(self._goal_sensitivity())
        return self.last_prior_metrics

    def _update_actor(self) -> dict[str, float]:
        n_micro = int(self.cfg.batch_size) // int(self.cfg.micro_batch_size)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.prior_optimizer.zero_grad(set_to_none=True)
        self.wm_trainer.opt.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        try:
            with frozen_parameters(self.prior, self.wm):
                for _ in range(n_micro):
                    terms, _ = self._one_micro_rollout(int(self.cfg.micro_batch_size), planner=False)
                    if not torch.isfinite(terms["total"]):
                        raise FloatingPointError("non-finite M1 Actor loss")
                    (terms["total"] / n_micro).backward()
                    for name, value in terms.items():
                        if value.ndim == 0:
                            divisor = 1 if name in ("replay/task_l_count", "replay/task_m_count") else n_micro
                            sums[name] = sums.get(name, 0.0) + float(value.detach()) / divisor
                actor_parameters = list(self.actor.parameters())
                if any(p.grad is None for p in actor_parameters):
                    raise RuntimeError("Actor task/tracking loss did not reach every Actor parameter")
                before = [parameter.detach().clone() for parameter in actor_parameters]
                grad_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, self.cfg.grad_clip_norm, error_if_nonfinite=True)
                if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                    raise RuntimeError(f"invalid Actor gradient norm {float(grad_norm)}")
                if any(p.grad is not None for p in self.prior.parameters()) or any(p.grad is not None for p in self.wm.parameters()):
                    raise RuntimeError("Actor update violated Prior/WM optimizer ownership")
                self.actor_optimizer.step()
                if all(torch.equal(old, parameter.detach()) for old, parameter in zip(before, actor_parameters)):
                    raise RuntimeError("Actor optimizer step did not change parameters")
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.prior_optimizer.zero_grad(set_to_none=True)
            self.wm_trainer.opt.zero_grad(set_to_none=True)
        self.actor_optimizer_steps += 1
        sums["grad_norm"] = float(grad_norm)
        self.last_actor_metrics = {f"actor/{key}": value for key, value in sums.items()}
        self.last_actor_metrics["imag/actor_task_return"] = -sums.get("task", float("nan"))
        return self.last_actor_metrics

    def _reference_shift(self) -> float:
        if self.buffer.size == 0:
            return float("nan")
        with torch.no_grad():
            start = self.buffer.sample_starts(1, self.actor.cfg.history)
            goal = task_goal8(start.state, start.task_type, start.velocity_command, start.fixed_world_target)
            history = self._prior_state_history(start)
            current = self.prior(history, goal, mode="prior_mean")
            initial = self.prior_initial(history, goal, mode="prior_mean")
            scales = current.new_tensor([0.5] * 3 + [1.0] * 6 + [2.0] * 12 + [2.0] * 6 + [10.0] * 4)
            return float(((current - initial) / scales).square().mean().sqrt())

    @torch.no_grad()
    def _goal_sensitivity(self) -> dict[str, float]:
        names = (
            "prior/L/vx_goal_sensitivity", "prior/L/vy_goal_sensitivity",
            "prior/L/wz_goal_sensitivity", "prior/M/ee_goal_sensitivity",
            "prior/task_switch_sensitivity",
        )
        result = {name: float("nan") for name in names}
        if self.buffer.size == 0:
            return result
        start = self.buffer.sample_starts(min(32, self.buffer.E), self.actor.cfg.history)
        history_all = self._prior_state_history(start)
        scales = history_all.new_tensor([0.5] * 3 + [1.0] * 6 + [2.0] * 12 + [2.0] * 6 + [10.0] * 4)
        for task, goal_start, deltas, output_keys in (
            (0, 0, (0.03, 0.02, 0.04), names[:3]),
            (1, 3, (0.05, 0.05, 0.05), (names[3],) * 3),
        ):
            selected = (start.task_type == task).nonzero(as_tuple=False).flatten()
            if not selected.numel():
                continue
            state = start.state.index_select(0, selected)
            task_ids = start.task_type.index_select(0, selected)
            velocity = start.velocity_command.index_select(0, selected)
            target = start.fixed_world_target.index_select(0, selected)
            history = history_all.index_select(0, selected)
            goal = task_goal8(state, task_ids, velocity, target)
            baseline = self.prior(history, goal, mode="prior_mean")
            alternatives = goal.repeat(3, 1)
            for axis, delta in enumerate(deltas):
                alternatives[axis * len(selected):(axis + 1) * len(selected), goal_start + axis] += delta
            pred = self.prior(history.repeat(3, 1, 1), alternatives, mode="prior_mean").reshape(3, len(selected), 10, 31)
            difference = ((pred - baseline[None]) / scales).square().mean((1, 2, 3)).sqrt()
            if task == 0:
                for key, value in zip(output_keys, difference):
                    result[key] = float(value)
            else:
                result[names[3]] = float(difference.square().mean().sqrt())
        task_ids = start.task_type
        goal = task_goal8(start.state, task_ids, start.velocity_command, start.fixed_world_target)
        alternate_task = 1 - task_ids
        alternate = task_goal8(start.state, alternate_task, start.velocity_command, start.fixed_world_target)
        a = self.prior(history_all, goal, mode="prior_mean")
        b = self.prior(history_all, alternate, mode="prior_mean")
        result[names[4]] = float(((a - b) / scales).square().mean().sqrt())
        return result

    def _load_prior_calibration_wm(self, evidence: dict) -> str | None:
        """Load the exact WM checkpoint audited by MuJoCo and freeze that snapshot."""
        checkpoint_path = evidence.get("checkpoint")
        if not checkpoint_path:
            return None
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            return None
        identity = str(path)
        stat = path.stat()
        signature = (identity, int(stat.st_mtime_ns), int(stat.st_size))
        if self._prior_calibration_checkpoint_signature != signature:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
                return None
            if checkpoint.get("action_mapping") != serialize_m1_action_contract(self.robot_spec):
                return None
            if checkpoint.get("prior_source", {}).get("sha256") != self.prior.source.get("sha256"):
                return None
            expected_reset = {
                "profile": self.robot_spec.default_pose_profile,
                "task_conditional": False,
                "default_base_height_m": float(self.robot_spec.default_base_height),
                "default_joint_pos": tuple(self.robot_spec.default_joint_pos),
            }
            if checkpoint.get("reset_contract", expected_reset) != expected_reset:
                return None
            snapshot = deepcopy(self.wm).eval()
            snapshot.load_state_dict(checkpoint["wm"], strict=True)
            snapshot.requires_grad_(False)
            snapshot_sha = module_sha256(snapshot)
            if checkpoint.get("world_model_sha256") != snapshot_sha:
                return None
            self._prior_calibration_wm = snapshot
            self._prior_calibration_wm_sha256 = snapshot_sha
            self._prior_calibration_checkpoint = identity
            self._prior_calibration_checkpoint_signature = signature
        return self._prior_calibration_wm_sha256

    def _prior_gate(self) -> dict[str, float]:
        evidence_path = self.cfg.prior_gate_evidence_path
        checks = {"evidence_present": False, "action_response": False, "actor_task": False,
                  "real_task_safety": False, "wm_snapshot_matches": False, "action_mapping_matches": False,
                  "heldout_coverage": False}
        details: dict[str, float] = {"schedule/prior_gate_pass": 0.0}
        evidence: dict = {}
        if evidence_path:
            path = Path(evidence_path).expanduser()
            if path.is_file():
                try:
                    evidence = json.loads(path.read_text())
                    checks["evidence_present"] = evidence.get("audit_type") == "m1_real_mujoco_wm_action_response_v1"
                    checks["action_response"] = evidence.get("action_response_pass") is True
                    checks["actor_task"] = evidence.get("actor_task_pass") is True
                    checks["real_task_safety"] = evidence.get("real_task_safety_pass") is True
                    snapshot_sha = self._load_prior_calibration_wm(evidence)
                    checks["wm_snapshot_matches"] = (
                        snapshot_sha is not None
                        and evidence.get("world_model_sha256") == snapshot_sha
                    )
                    saved_mapping = evidence.get("action_mapping")
                    expected_mapping = serialize_m1_action_contract(self.robot_spec)
                    # Evidence is JSON, so tuples in the in-memory Action
                    # Contract are represented as lists on disk.
                    checks["action_mapping_matches"] = (
                        isinstance(saved_mapping, dict)
                        and json.loads(json.dumps(saved_mapping, sort_keys=True))
                        == json.loads(json.dumps(expected_mapping, sort_keys=True))
                    )
                    checks["heldout_coverage"] = int(evidence.get("heldout_state_count", 0)) >= 128
                except (OSError, ValueError, TypeError):
                    pass
        actor_ready = self.actor_ready
        force_frozen = bool(getattr(self, "_force_prior_frozen", False))
        all_checks = all(checks.values()) and actor_ready and not self.cfg.freeze_prior and not force_frozen
        details.update({f"schedule/prior_gate_{key}": float(value) for key, value in checks.items()})
        details["schedule/prior_gate_actor_ready"] = float(actor_ready)
        details["schedule/prior_gate_freeze_config_clear"] = float(not self.cfg.freeze_prior)
        details["schedule/prior_gate_force_frozen"] = float(force_frozen)
        details["schedule/prior_calibration_wm_sha256_present"] = float(
            getattr(self, "_prior_calibration_wm_sha256", None) is not None
        )
        details["schedule/prior_gate_pass"] = float(all_checks)
        if evidence:
            details["schedule/prior_gate_heldout_state_count"] = float(evidence.get("heldout_state_count", 0))
        return details

    def update_event(self, *, policy_allowed: bool) -> dict[str, float]:
        """One state-machine event: real Replay WM fit, then frozen- or gated-Prior learning."""
        output: dict[str, float] = {}
        prior_steps_before = self.prior_optimizer_steps
        for _ in range(int(self.cfg.wm_updates_per_event)):
            self.last_wm_metrics = dict(self.wm_trainer.update(
                self.buffer,
                task_balanced=bool(getattr(self.cfg, "balanced_replay_sampling", False)),
            ))
            self.wm_update_events += 1
            self.wm_optimizer_steps += int(self.wm_trainer.cfg.batches_per_update)
        for key, value in self.last_wm_metrics.items():
            output[f"wm/{key}"] = float(value)
        finite_wm = all(np.isfinite(value) for value in self.last_wm_metrics.values())
        output.update({
            "schedule/wm_update_events": float(self.wm_update_events),
            "schedule/wm_optimizer_steps": float(self.wm_optimizer_steps),
            "schedule/wm_replay_metrics_finite": float(finite_wm),
        })
        if not policy_allowed:
            self.stage = "wm_warmup"
        else:
            gate = self._prior_gate()
            self._last_gate_diagnostics = gate
            output.update(gate)
            allow_prior = (
                gate["schedule/prior_gate_pass"] > 0.5
                and int(self.cfg.prior_updates_per_event) > 0
            )
            if allow_prior:
                self.stage = "prior_calibration"
                for _ in range(int(self.cfg.prior_updates_per_event)):
                    output.update(self._update_prior())
                self.stage = "actor_adaptation"
            else:
                self.stage = "actor_prior_frozen"
            for _ in range(int(self.cfg.actor_updates_per_event)):
                output.update(self._update_actor())
        output.update(self._real_metrics)
        output.update({
            "schedule/prior_optimizer_steps": float(self.prior_optimizer_steps),
            "schedule/actor_optimizer_steps": float(self.actor_optimizer_steps),
            "schedule/prior_unfrozen": float(self.prior_optimizer_steps > prior_steps_before),
            "schedule/training_phase": float({
                "wm_warmup": 0, "actor_prior_frozen": 1, "prior_calibration": 2,
                "actor_adaptation": 3, "validation": 4,
            }[self.stage]),
            "prior_state_source/privileged": 1.0,
        })
        output.update(self.actor_readiness_diagnostics)
        return output

    def set_validation_result(self, *, passed: bool, metrics: dict | None = None) -> None:
        """Accept calibration only with comparison evidence; otherwise restore prior+optimizer."""
        self.stage = "validation"
        self._last_validation_pass = bool(passed)
        if self._prior_calibration_backup is None:
            self._prior_calibration_status = "no_calibration_to_validate"
            self._prior_calibration_validation = {"passed": bool(passed), "metrics": metrics or {}}
            return
        if passed and not metrics:
            raise ValueError("accepting a Prior calibration requires L/M Actor/Prior Direct and safety comparison metrics")
        self._prior_calibration_validation = {"passed": bool(passed), "metrics": metrics or {}}
        if passed:
            self._prior_calibration_status = "accepted"
            self._prior_calibration_backup = None
            return
        backup = self._prior_calibration_backup
        self.prior.load_state_dict(backup["prior"], strict=True)
        self.prior_optimizer.load_state_dict(backup["optimizer"])
        self._prior_calibration_rolled_back_updates += max(
            0, int(self.prior_optimizer_steps) - int(backup["optimizer_steps"]),
        )
        restored_sha = module_sha256(self.prior)
        if restored_sha != backup["prior_sha256"]:
            raise RuntimeError("Prior calibration rollback did not restore the saved valid weights")
        self._prior_calibration_status = "rejected_rollback"
        self._force_prior_frozen = True
        self._prior_calibration_backup = None

    def checkpoint(self, *, iteration: int, config_snapshot: dict, command_state: dict,
                   wm_only_updates: int = 0) -> dict:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "format_version": self.FORMAT_VERSION,
            "iteration": int(iteration),
            "wm_update_events": self.wm_update_events,
            "wm_optimizer_steps": self.wm_optimizer_steps,
            "prior_optimizer_steps": self.prior_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "training_phase": self.stage,
            "training_mode": self.cfg.training_mode,
            "wm_only_updates": int(wm_only_updates),
            "actor_readiness_window": list(self._actor_ready_window),
            "wm": self.wm.state_dict(),
            "motion_prior": self.prior.state_dict(),
            "prior_initial": self.prior_initial.state_dict(),
            "actor": self.actor.state_dict(),
            "wm_optimizer": self.wm_trainer.opt.state_dict(),
            "prior_optimizer": self.prior_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "task_sampler_state": command_state,
            "reset_contract": {
                "profile": self.robot_spec.default_pose_profile,
                "task_conditional": False,
                "default_base_height_m": float(self.robot_spec.default_base_height),
                "default_joint_pos": tuple(self.robot_spec.default_joint_pos),
            },
            "balanced_replay_sampling": bool(getattr(self.cfg, "balanced_replay_sampling", False)),
            "rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "robot_identity": "m1_piper",
            "robot_joint_names": self.robot_joint_names,
            "robot_model_sha256": self.robot_model_sha256,
            "prior_schema_version": self.prior.feature_version,
            "prior_source": dict(self.prior.source),
            "prior_mode": self.cfg.prior_mode,
            "prior_input_compatibility": self.prior.input_compatibility,
            "actor_schema": asdict(self.actor.cfg),
            "actor_input_contract": self.actor.input_contract,
            "reference_adapter_schema": self.REFERENCE_SCHEMA,
            "action_mapping": serialize_m1_action_contract(self.robot_spec),
            "goal_mapping": "Goal8[vx,vy,wz,ee_dx_heading,ee_dy_heading,ee_dz_heading,valid_L,valid_M]",
            "tcp_convention": "piper_x_link7",
            "reference_horizon": 10,
            "replanning_cadence": 10,
            "prior_state_source": self.cfg.state_source,
            "loss_coefficients": asdict(self.loss_cfg),
            "world_model_sha256": module_sha256(self.wm),
            "prior_gate_evidence_path": str(self.cfg.prior_gate_evidence_path) if self.cfg.prior_gate_evidence_path else None,
            "prior_gate_diagnostics": dict(self._last_gate_diagnostics),
            "prior_calibration": {
                "status": self._prior_calibration_status,
                "wm_snapshot_sha256": self._prior_calibration_wm_sha256,
                "wm_snapshot_checkpoint": self._prior_calibration_checkpoint,
                "backup": deepcopy(self._prior_calibration_backup),
                "validation": deepcopy(self._prior_calibration_validation),
                "force_prior_frozen": self._force_prior_frozen,
                "rolled_back_optimizer_steps": self._prior_calibration_rolled_back_updates,
                "current_prior_sha256": module_sha256(self.prior),
            },
            "exploration": {
                "enabled": self.cfg.exploration_enabled,
                "env_fraction": self.cfg.exploration_env_fraction,
                "leg_std": self.cfg.exploration_leg_std,
                "arm_std": self.cfg.exploration_arm_std,
                "wheel_std": self.cfg.exploration_wheel_std,
                "decay_end_fraction": self.cfg.exploration_decay_end_fraction,
            },
            "config": config_snapshot,
        }

    def load_checkpoint(self, checkpoint: dict, *, expected_source: dict) -> int:
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT or checkpoint.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("M1 resume requires Action22-reference checkpoint v4; Future31 Actor v2/v3 is input-incompatible")
        checks = {
            "robot_identity": "m1_piper",
            "robot_joint_names": self.robot_joint_names,
            "robot_model_sha256": self.robot_model_sha256,
            "prior_schema_version": self.prior.feature_version,
            "prior_mode": self.cfg.prior_mode,
            "prior_input_compatibility": self.prior.input_compatibility,
            "reference_adapter_schema": self.REFERENCE_SCHEMA,
            "actor_schema": asdict(self.actor.cfg),
            "actor_input_contract": self.actor.input_contract,
            "action_mapping": serialize_m1_action_contract(self.robot_spec),
            "loss_coefficients": asdict(self.loss_cfg),
            "training_mode": self.cfg.training_mode,
        }
        for key, expected in checks.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Action22 checkpoint {key} mismatch: saved={checkpoint.get(key)!r}, current={expected!r}")
        provenance_keys = (
            "path", "sha256", "step", "feature_version", "condition_contract", "model_config",
            "dataset_hash", "split_hash", "normalization_hash", "target_contract",
            "assurance_mode", "strict_fcl_only", "input_compatibility",
        )
        saved_source = checkpoint.get("prior_source", {})
        if any(saved_source.get(key) != expected_source.get(key) for key in provenance_keys):
            raise ValueError("Action22 resume Motion Prior provenance mismatch")
        if checkpoint.get("task_sampler_state") != self.command.state_dict():
            raise ValueError("Action22 resume task sampler/command range mismatch")
        expected_reset = {
            "profile": self.robot_spec.default_pose_profile,
            "task_conditional": False,
            "default_base_height_m": float(self.robot_spec.default_base_height),
            "default_joint_pos": tuple(self.robot_spec.default_joint_pos),
        }
        if checkpoint.get("reset_contract", expected_reset) != expected_reset:
            raise ValueError("Action22 resume reset pose contract mismatch")
        if checkpoint.get("balanced_replay_sampling", False) != bool(getattr(self.cfg, "balanced_replay_sampling", False)):
            raise ValueError("Action22 resume Replay task-balancing contract mismatch")
        for key, module in (("wm", self.wm), ("motion_prior", self.prior),
                            ("prior_initial", self.prior_initial), ("actor", self.actor)):
            if key not in checkpoint:
                raise ValueError(f"Action22 checkpoint is missing {key!r}")
            module.load_state_dict(checkpoint[key], strict=True)
        if checkpoint.get("world_model_sha256") != module_sha256(self.wm):
            raise ValueError("Action22 resume World Model state/hash mismatch")
        self.wm_trainer.opt.load_state_dict(checkpoint["wm_optimizer"])
        self.prior_optimizer.load_state_dict(checkpoint["prior_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        for name in ("wm_update_events", "wm_optimizer_steps", "prior_optimizer_steps", "actor_optimizer_steps"):
            setattr(self, name, int(checkpoint[name]))
        self.stage = str(checkpoint["training_phase"])
        self._actor_ready_window.clear()
        self._actor_ready_window.extend(checkpoint.get("actor_readiness_window", []))
        self._last_gate_diagnostics = dict(checkpoint.get("prior_gate_diagnostics", {}))
        calibration = checkpoint.get("prior_calibration", {})
        self._prior_calibration_status = str(calibration.get("status", "not_started"))
        self._prior_calibration_wm_sha256 = calibration.get("wm_snapshot_sha256")
        self._prior_calibration_checkpoint = calibration.get("wm_snapshot_checkpoint")
        self._prior_calibration_checkpoint_signature = None
        self._prior_calibration_backup = deepcopy(calibration.get("backup"))
        self._prior_calibration_validation = deepcopy(calibration.get("validation"))
        self._force_prior_frozen = bool(calibration.get("force_prior_frozen", False))
        self._prior_calibration_rolled_back_updates = int(calibration.get("rolled_back_optimizer_steps", 0))
        rng = checkpoint.get("rng_state", {})
        if "python" in rng: random.setstate(rng["python"])
        if "numpy" in rng: np.random.set_state(rng["numpy"])
        if "torch_cpu" in rng: torch.set_rng_state(rng["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8))
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all([state.detach().to(device="cpu", dtype=torch.uint8) for state in rng["torch_cuda"]])
        return int(checkpoint["iteration"])
