"""Alternating WM → Motion Prior → Tracking Actor optimizer for M1 v1."""

from __future__ import annotations

import random
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.losses import M1JointLossCfg, actor_losses, planner_losses
from supermimic.motion_prior.reference import decode_future31
from supermimic.motion_prior.rollout import differentiable_m1_rollout
from supermimic.policy.nets import M1PriorConditionedActor
from supermimic.robots.spec import M1_JOINTS
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


@contextmanager
def frozen_parameters(*modules):
    """Freeze parameter gradients without detaching module inputs; always restore flags."""
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


class M1AlternatingTrainer:
    """Owns the separate Prior/Actor optimizers and their gradient boundaries."""

    FORMAT_VERSION = 2

    def __init__(self, *, wm: WorldModel, wm_trainer: WorldModelTrainer,
                 prior, actor: M1PriorConditionedActor, buffer: ReplayBuffer,
                 default_joint_pos: torch.Tensor, soft_limits: torch.Tensor,
                 fk, tcp_index: int, cfg, robot_joint_names: tuple[str, ...],
                 robot_model_sha256: str | None, command):
        if buffer.task_type is None:
            raise ValueError("M1 joint trainer requires the named L/M Replay context")
        if int(cfg.prior_horizon) != 10 or int(cfg.actor_horizon) != 10 or int(cfg.replanning_cadence) != 10:
            raise ValueError("v1 supports only aligned 10-step references and 10-step replanning")
        if cfg.state_source != "privileged":
            raise NotImplementedError(
                "prior_state_source='estimated' is reserved but no validated M1 velocity estimator is connected; "
                "the planner will not silently use MuJoCo truth"
            )
        if cfg.prior_mode not in ("prior_mean", "prior_sample"):
            raise ValueError("unsupported Motion Prior inference mode")
        if cfg.batch_size <= 0 or cfg.micro_batch_size <= 0 or cfg.batch_size % cfg.micro_batch_size:
            raise ValueError("M1 joint batch_size must be divisible by positive micro_batch_size")
        self.wm, self.wm_trainer = wm, wm_trainer
        self.prior, self.actor, self.buffer = prior, actor, buffer
        self.default_joint_pos = default_joint_pos
        self.soft_limits = soft_limits
        self.fk, self.tcp_index = fk, int(tcp_index)
        self.cfg = cfg
        self.loss_cfg: M1JointLossCfg = cfg.loss
        self.robot_joint_names = tuple(robot_joint_names)
        self.robot_model_sha256 = robot_model_sha256
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
        self._plan_anchor_state: torch.Tensor | None = None
        self._plan_has_sampled_latent: torch.Tensor | None = None
        self.last_prior_metrics: dict[str, float] = {}
        self.last_actor_metrics: dict[str, float] = {}

    def set_lr(self, *, prior_lr: float, actor_lr: float) -> None:
        for group in self.prior_optimizer.param_groups:
            group["lr"] = float(prior_lr)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(actor_lr)

    @property
    def actor_ready(self) -> bool:
        cfg = self.cfg
        if len(self._actor_ready_window) < int(cfg.readiness_min_samples):
            return False
        if self.actor_optimizer_steps < int(cfg.actor_pretrain_updates):
            return False
        means = {key: sum(row[key] for row in self._actor_ready_window) / len(self._actor_ready_window)
                 for key in self._actor_ready_window[0]}
        return (
            means["survival"] >= cfg.readiness_min_survival
            and means["root_tracking_error"] <= cfg.readiness_max_root_tracking_error
            and means["arm_tracking_error"] <= cfg.readiness_max_arm_tracking_error
            and means["wheel_tracking_error"] <= cfg.readiness_max_wheel_tracking_error
            and means["action_saturation"] <= cfg.readiness_max_action_saturation
        )

    @staticmethod
    def _prior_state_history(start) -> torch.Tensor:
        # Replay keeps an oldest→newest sequence [H,B,57]; CVAE uses exactly the last four.
        return start.hist_states[-4:].permute(1, 0, 2).contiguous()

    def _sample_prior(self, state_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.cfg.prior_mode == "prior_mean":
            return self.prior(state_history, goal, mode="prior_mean")
        eps = torch.randn((len(goal), self.prior.model.latent_dim), dtype=goal.dtype, device=goal.device)
        return self.prior(state_history, goal, mode="prior_sample", latent_eps=eps)

    @torch.no_grad()
    def act(self, proprio_history: torch.Tensor, goal: torch.Tensor,
            state_history: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current bounded action; replan only at each 10-step boundary."""
        batch = len(goal)
        if state_history.shape != (batch, 4, 57):
            raise ValueError("online planner requires privileged/estimated state history [B,4,57]")
        phase = phase.reshape(batch).to(device=goal.device, dtype=torch.long)
        if self._plan is None or self._plan.shape[0] != batch:
            self._plan = torch.zeros(batch, 10, 31, dtype=goal.dtype, device=goal.device)
            self._plan_anchor_state = torch.zeros(batch, 57, dtype=goal.dtype, device=goal.device)
        replan = (phase == 0).nonzero(as_tuple=False).flatten()
        if replan.numel():
            self._plan[replan] = self._sample_prior(state_history[replan], goal[replan])
            self._plan_anchor_state[replan] = state_history[replan, -1]
        action = self.actor(proprio_history, goal, self._plan, phase.to(goal.dtype))
        return action, self._plan

    @torch.no_grad()
    def record_real_step(self, next_state: torch.Tensor, action: torch.Tensor,
                         phase: torch.Tensor, terminated: torch.Tensor) -> dict[str, float]:
        if self._plan is None or self._plan_anchor_state is None:
            raise RuntimeError("real-step metrics require an active reference plan")
        phase = phase.reshape(-1).long().clamp(0, 9)
        ref_state = decode_future31(self._plan_anchor_state, self._plan)
        rows = torch.arange(len(phase), device=phase.device)
        target = ref_state[rows, phase]
        root_error = torch.linalg.vector_norm(next_state[:, :3] - target[:, :3], dim=-1)
        arm_ids = [M1_JOINTS.index(f"piper_x_joint{i}") for i in range(1, 7)]
        wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
        arm_error = (next_state[:, 7:29][:, arm_ids] - target[:, 7:29][:, arm_ids]).square().mean(-1).sqrt()
        wheel_error = (next_state[:, 35:57][:, wheel_ids] - target[:, 35:57][:, wheel_ids]).square().mean(-1).sqrt()
        metrics = {
            "survival": float((~terminated.bool()).float().mean()),
            "root_tracking_error": float(root_error.mean()),
            "arm_tracking_error": float(arm_error.mean()),
            "wheel_tracking_error": float(wheel_error.mean()),
            "action_saturation": float((action.abs() >= 0.98).float().mean()),
        }
        self._actor_ready_window.append(metrics)
        self._real_metrics = {
            f"real/{key}": value for key, value in metrics.items()
        }
        return self._real_metrics

    def _one_micro_rollout(self, micro: int, *, planner: bool) -> tuple[dict[str, torch.Tensor], object]:
        start = self.buffer.sample_starts(micro, self.actor.cfg.history)
        goal = task_goal8(start.state, start.task_type, start.velocity_command, start.fixed_world_target)
        state_history = self._prior_state_history(start)
        reference = self._sample_prior(state_history, goal)
        if planner:
            with torch.no_grad():
                anchor = self.prior_initial(state_history, goal, mode="prior_mean")
            rollout = differentiable_m1_rollout(
                self.wm, self.actor, start, goal, reference,
                self.default_joint_pos, int(self.cfg.prior_horizon),
            )
            terms = planner_losses(
                rollout.states, rollout.actions, start.state, reference, anchor,
                start.task_type, start.velocity_command, start.fixed_world_target,
                fk=self.fk, tcp_index=self.tcp_index, soft_limits=self.soft_limits,
                cfg=self.loss_cfg,
            )
        else:
            reference = reference.detach()
            ref_states = decode_future31(start.state, reference).transpose(0, 1)
            rollout = differentiable_m1_rollout(
                self.wm, self.actor, start, goal, reference,
                self.default_joint_pos, int(self.cfg.actor_horizon),
            )
            terms = actor_losses(
                rollout.states[1:], ref_states, rollout.actions, start.prev_action,
                soft_limits=self.soft_limits, cfg=self.loss_cfg,
            )
        return terms, rollout

    def _update_prior(self) -> dict[str, float]:
        cfg = self.cfg
        logical = int(cfg.batch_size)
        micro = int(cfg.micro_batch_size)
        n_micro = logical // micro
        self.prior_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.wm_trainer.opt.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        # Frozen parameters do not block gradients from Action→WM→Actor input→Reference.
        try:
            with frozen_parameters(self.actor, self.wm):
                for _ in range(n_micro):
                    terms, rollout = self._one_micro_rollout(micro, planner=True)
                    if not torch.isfinite(terms["total"]):
                        raise FloatingPointError("non-finite M1 Planner loss")
                    (terms["total"] / n_micro).backward()
                    for name, value in terms.items():
                        if value.ndim == 0:
                            sums[name] = sums.get(name, 0.0) + float(value.detach()) / n_micro
                    if hasattr(self.wm, "termination_logits"):
                        survival = 1.0 - torch.sigmoid(self.wm.termination_logits(rollout.states[1:].detach())).mean()
                        sums["imag_survival"] = sums.get("imag_survival", 0.0) + float(survival) / n_micro
                prior_parameters = [p for p in self.prior.parameters() if p.requires_grad]
                if any(p.grad is None for p in prior_parameters):
                    # Some CVAE submodules (Posterior) are inference-only and intentionally unused.
                    prior_parameters = [p for p in prior_parameters if p.grad is not None]
                if not prior_parameters:
                    raise RuntimeError("Planner loss has no gradient path to Motion Prior parameters")
                probe = prior_parameters[0]
                probe_before = probe.detach().clone()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    prior_parameters, cfg.grad_clip_norm, error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                    raise RuntimeError(f"invalid Motion Prior gradient norm {float(grad_norm)}")
                if any(p.grad is not None for p in self.actor.parameters()):
                    raise RuntimeError("Actor received parameter gradients during Planner update")
                if any(p.grad is not None for p in self.wm.parameters()):
                    raise RuntimeError("World Model received parameter gradients during Planner update")
                self.prior_optimizer.step()
                if torch.equal(probe_before, probe.detach()):
                    raise RuntimeError("Motion Prior optimizer step did not change its parameters")
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.wm_trainer.opt.zero_grad(set_to_none=True)
            self.prior_optimizer.zero_grad(set_to_none=True)
        self.prior_optimizer_steps += 1
        sums["grad_norm"] = float(grad_norm)
        sums["reference_shift"] = self._reference_shift()
        metric_names = {
            "task": "task_loss", "task_l": "task_l_loss", "task_m": "task_m_loss",
            "task_progress": "predicted_ee_progress", "terminal_error": "terminal_error",
            "base_drift": "base_drift", "energy": "energy_loss",
            "control_effort_surrogate": "control_effort_surrogate",
            "feasibility": "feasibility_loss", "smoothness": "smooth_loss",
            "anchor": "anchor_loss", "supervised": "supervised_loss",
            "grad_norm": "grad_norm", "reference_shift": "reference_shift",
            "imag_survival": "imag_survival",
        }
        self.last_prior_metrics = {
            f"prior/{metric_names.get(k, k)}": v for k, v in sums.items()
        }
        self.last_prior_metrics["imag/planner_return"] = -sums.get("task", float("nan"))
        if "imag_survival" in sums:
            self.last_prior_metrics["imag/survival"] = sums["imag_survival"]
        self.last_prior_metrics["prior/goal_sensitivity"] = self._goal_sensitivity()
        return self.last_prior_metrics

    def _update_actor(self) -> dict[str, float]:
        cfg = self.cfg
        logical = int(cfg.batch_size)
        micro = int(cfg.micro_batch_size)
        n_micro = logical // micro
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.prior_optimizer.zero_grad(set_to_none=True)
        self.wm_trainer.opt.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        try:
            with frozen_parameters(self.prior, self.wm):
                for _ in range(n_micro):
                    terms, rollout = self._one_micro_rollout(micro, planner=False)
                    if not torch.isfinite(terms["total"]):
                        raise FloatingPointError("non-finite M1 Actor tracking loss")
                    (terms["total"] / n_micro).backward()
                    for name, value in terms.items():
                        if value.ndim == 0:
                            sums[name] = sums.get(name, 0.0) + float(value.detach()) / n_micro
                    if hasattr(self.wm, "termination_logits"):
                        survival = 1.0 - torch.sigmoid(self.wm.termination_logits(rollout.states[1:].detach())).mean()
                        sums["imag_survival"] = sums.get("imag_survival", 0.0) + float(survival) / n_micro
                actor_parameters = list(self.actor.parameters())
                if any(p.grad is None for p in actor_parameters):
                    raise RuntimeError("Actor tracking loss did not reach every Actor parameter")
                probe = actor_parameters[0]
                probe_before = probe.detach().clone()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, cfg.grad_clip_norm, error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                    raise RuntimeError(f"invalid Actor gradient norm {float(grad_norm)}")
                if any(p.grad is not None for p in self.prior.parameters()):
                    raise RuntimeError("Motion Prior received parameter gradients during Actor update")
                if any(p.grad is not None for p in self.wm.parameters()):
                    raise RuntimeError("World Model received parameter gradients during Actor update")
                self.actor_optimizer.step()
                if torch.equal(probe_before, probe.detach()):
                    raise RuntimeError("Actor optimizer step did not change its parameters")
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.prior_optimizer.zero_grad(set_to_none=True)
            self.wm_trainer.opt.zero_grad(set_to_none=True)
        self.actor_optimizer_steps += 1
        sums["grad_norm"] = float(grad_norm)
        metric_names = {"tracking": "tracking_error", "imag_survival": "imag_survival"}
        self.last_actor_metrics = {
            f"actor/{metric_names.get(k, k)}": v for k, v in sums.items()
        }
        self.last_actor_metrics["imag/tracking_return"] = -sums.get("tracking", float("nan"))
        if "imag_survival" in sums:
            self.last_actor_metrics["imag/survival"] = sums["imag_survival"]
        return self.last_actor_metrics

    def _reference_shift(self) -> float:
        if self.buffer.size == 0:
            return float("nan")
        with torch.no_grad():
            start = self.buffer.sample_starts(1, self.actor.cfg.history)
            goal = task_goal8(start.state, start.task_type, start.velocity_command, start.fixed_world_target)
            history = self._prior_state_history(start)
            current = self.prior(history, goal)
            initial = self.prior_initial(history, goal)
            scales = current.new_tensor([0.5] * 3 + [1.0] * 6 + [2.0] * 12 + [2.0] * 6 + [10.0] * 4)
            return float(((current - initial) / scales).square().mean().sqrt())

    @torch.no_grad()
    def _goal_sensitivity(self) -> float:
        if self.buffer.size == 0:
            return float("nan")
        start = self.buffer.sample_starts(min(4, self.buffer.E), self.actor.cfg.history)
        alternative_task = 1 - start.task_type
        alternative_goal = task_goal8(
            start.state, alternative_task, start.velocity_command, start.fixed_world_target,
        )
        state_history = self._prior_state_history(start)
        base = self.prior(state_history, task_goal8(
            start.state, start.task_type, start.velocity_command, start.fixed_world_target,
        ))
        alternative = self.prior(state_history, alternative_goal)
        scales = base.new_tensor([0.5] * 3 + [1.0] * 6 + [2.0] * 12 + [2.0] * 6 + [10.0] * 4)
        return float(((base - alternative) / scales).square().mean().sqrt())

    def update_event(self, *, policy_allowed: bool) -> dict[str, float]:
        """One event: real replay WM updates, then (when ready) Prior→Actor."""
        output: dict[str, float] = {}
        for _ in range(int(self.cfg.wm_updates_per_event)):
            metrics = self.wm_trainer.update(self.buffer)
            self.wm_update_events += 1
            self.wm_optimizer_steps += int(self.wm_trainer.cfg.batches_per_update)
            for key, value in metrics.items():
                alias = {"nse_1step": "nse_1", "nse_kstep": f"nse_{self.wm_trainer.cfg.window}"}.get(key, key)
                output[f"wm/{alias}"] = value
        ready = self.actor_ready or (
            not bool(self.cfg.freeze_prior)
            and self.actor_optimizer_steps >= int(self.cfg.actor_pretrain_updates)
        )
        if not policy_allowed:
            self.stage = "wm_warmup"
        elif not ready:
            self.stage = "actor_tracking_warmup"
            for _ in range(int(self.cfg.actor_updates_per_event)):
                output.update(self._update_actor())
        elif self.actor_optimizer_steps < int(self.cfg.actor_pretrain_updates) + int(self.cfg.prior_warmup_updates):
            self.stage = "prior_frozen_tracking_warmup"
            for _ in range(int(self.cfg.actor_updates_per_event)):
                output.update(self._update_actor())
        elif int(self.cfg.prior_updates_per_event) > 0:
            self.stage = "joint_alternating"
            for _ in range(int(self.cfg.prior_updates_per_event)):
                output.update(self._update_prior())
            for _ in range(int(self.cfg.actor_updates_per_event)):
                output.update(self._update_actor())
        else:
            self.stage = "actor_only_prior_frozen"
            for _ in range(int(self.cfg.actor_updates_per_event)):
                output.update(self._update_actor())
        output.update(self._real_metrics)
        phase_code = {"wm_warmup": 0, "actor_tracking_warmup": 1,
                      "prior_frozen_tracking_warmup": 2, "joint_alternating": 3,
                      "actor_only_prior_frozen": 4}[self.stage]
        output.update({
            "schedule/wm_update_events": float(self.wm_update_events),
            "schedule/wm_optimizer_steps": float(self.wm_optimizer_steps),
            "schedule/prior_optimizer_steps": float(self.prior_optimizer_steps),
            "schedule/actor_optimizer_steps": float(self.actor_optimizer_steps),
            "schedule/phase": float(phase_code),
            "prior/actor_ready": float(ready),
            "prior_state_source/privileged": 1.0,
        })
        return output

    def checkpoint(self, *, iteration: int, config_snapshot: dict, command_state: dict,
                   wm_only_updates: int = 0) -> dict:
        return {
            "format": "supermimic_m1_prior_actor_joint_v2",
            "format_version": self.FORMAT_VERSION,
            "iteration": int(iteration),
            "wm_update_events": self.wm_update_events,
            "wm_optimizer_steps": self.wm_optimizer_steps,
            "prior_optimizer_steps": self.prior_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "training_phase": self.stage,
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
            "actor_schema": asdict(self.actor.cfg),
            "action_mapping": {
                "joint_names": self.robot_joint_names,
                "wheel_joint_names": ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT"),
                "position_control_count": 18, "wheel_velocity_control_count": 4,
            },
            "goal_mapping": "Goal8[vx,vy,wz,ee_dx_heading,ee_dy_heading,ee_dz_heading,velocity_valid,ee_valid]",
            "tcp_convention": "piper_x_link7 for CVAE Future, Planner FK, and m1_prior_joint real target/metrics",
            "reference_horizon": 10,
            "replanning_cadence": int(self.cfg.replanning_cadence),
            "prior_state_source": self.cfg.state_source,
            "loss_coefficients": asdict(self.loss_cfg),
            "config": config_snapshot,
        }

    def load_checkpoint(self, checkpoint: dict, *, expected_source: dict) -> int:
        if checkpoint.get("format") != "supermimic_m1_prior_actor_joint_v2" or checkpoint.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("joint checkpoint format/version mismatch")
        checks = {
            "robot_identity": "m1_piper",
            "robot_joint_names": self.robot_joint_names,
            "robot_model_sha256": self.robot_model_sha256,
            "prior_schema_version": self.prior.feature_version,
            "reference_horizon": 10,
            "replanning_cadence": int(self.cfg.replanning_cadence),
            "prior_state_source": self.cfg.state_source,
            "prior_mode": self.cfg.prior_mode,
            "actor_schema": asdict(self.actor.cfg),
        }
        for key, expected in checks.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"joint resume {key} mismatch: saved={checkpoint.get(key)!r}, current={expected!r}")
        saved_source = checkpoint.get("prior_source", {})
        if saved_source.get("sha256") != expected_source.get("sha256"):
            raise ValueError("joint resume Motion Prior source checkpoint/hash mismatch")
        if checkpoint.get("loss_coefficients") != asdict(self.loss_cfg):
            raise ValueError("joint resume loss coefficient/schema mismatch")
        if checkpoint.get("action_mapping", {}).get("joint_names") != self.robot_joint_names:
            raise ValueError("joint resume M1 mixed-action joint mapping mismatch")
        current_task_state = self.command.state_dict()
        if checkpoint.get("task_sampler_state") != current_task_state:
            raise ValueError("joint resume L/M task sampler configuration mismatch")
        for key, module in (("wm", self.wm), ("motion_prior", self.prior),
                            ("prior_initial", self.prior_initial), ("actor", self.actor)):
            if key not in checkpoint:
                raise ValueError(f"joint resume checkpoint is missing {key!r}")
            module.load_state_dict(checkpoint[key], strict=True)
        self.wm_trainer.opt.load_state_dict(checkpoint["wm_optimizer"])
        self.prior_optimizer.load_state_dict(checkpoint["prior_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.wm_update_events = int(checkpoint["wm_update_events"])
        self.wm_optimizer_steps = int(checkpoint["wm_optimizer_steps"])
        self.prior_optimizer_steps = int(checkpoint["prior_optimizer_steps"])
        self.actor_optimizer_steps = int(checkpoint["actor_optimizer_steps"])
        self.stage = str(checkpoint["training_phase"])
        self._actor_ready_window.clear()
        self._actor_ready_window.extend(checkpoint.get("actor_readiness_window", []))
        rng = checkpoint.get("rng_state", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        return int(checkpoint["iteration"])
