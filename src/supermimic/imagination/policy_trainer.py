"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.imagination.tasks import ChaseBallTask, MotionTrackingTask, VelocityCommandTask, WorldPointReachTask
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


def _clip_bootstrap_values(values: torch.Tensor, value_clip: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally bound bootstrap values and report the fraction affected."""
    if value_clip is None:
        return values, values.new_zeros(())
    if value_clip <= 0:
        raise ValueError("critic_value_clip must be positive or None")
    return values.clamp(-value_clip, value_clip), (values.abs() > value_clip).float().mean()


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    # Optional GPU micro-batch. Several imagined rollouts are accumulated into one
    # logical batch so large reference batch sizes do not require a large GPU.
    micro_batch_size: int | None = None
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    # Optional symmetric bound for critic bootstrap values. Keep unset by default;
    # reward scales differ across Motion, Velocity and World Point tasks.
    critic_value_clip: float | None = None
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"
    anchor_orientation_obs: bool = False


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, task: MotionTrackingTask | VelocityCommandTask | ChaseBallTask | WorldPointReachTask | MotionLibrary, cfg: PolicyTrainerCfg):
        if isinstance(task, MotionLibrary):
            task = MotionTrackingTask(task, future_frames=cfg.future_frames, anchor_obs=cfg.anchor_obs, anchor_orientation_obs=cfg.anchor_orientation_obs)
        self.wm, self.policy, self.critic, self.task, self.cfg = wm, policy, critic, task, cfg
        if cfg.critic_value_clip is not None and cfg.critic_value_clip <= 0:
            raise ValueError("critic_value_clip must be positive or None")
        self.last_imag_diagnostics: dict[str, list[float] | float] = {}
        self.lib = task.library if isinstance(task, MotionTrackingTask) else None
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import ImagStep, VelocityImagStep

        if isinstance(task, MotionTrackingTask):
            self.step_fn = ImagStep(wm, task.library.default_joint_pos, compile_mode=cfg.compile)
        else:
            if task.default_joint_pos is None:
                raise ValueError("VelocityCommandTask needs the robot default_joint_pos for analytic observations")
            self.step_fn = VelocityImagStep(wm, task, task.default_joint_pos, compile_mode=cfg.compile)

    def set_lr(self, lr: float, critic_lr: float | None = None) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr if critic_lr is None else critic_lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        micro = cfg.micro_batch_size or cfg.batch_size
        if micro <= 0 or cfg.batch_size <= 0 or cfg.batch_size % micro:
            raise ValueError(
                "policy batch_size must be a positive multiple of micro_batch_size; "
                f"got batch_size={cfg.batch_size}, micro_batch_size={micro}"
            )
        n_micro = cfg.batch_size // micro

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        objective_sum = 0.0
        imag_return_sum = 0.0
        imag_reward_sum = 0.0
        imag_survival_sum = 0.0
        imag_dhat_sum = 0.0
        action_abs_sum = 0.0
        reward_metric_sum: dict[str, float] = {}
        critic_payload: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        raw_value_sum = 0.0
        target_sum = 0.0
        value_clip_sum = 0.0
        try:
            for _ in range(n_micro):
                start = buffer.sample_starts(micro, cfg.history)
                xi_norm = self.wm.spec.normalize_xi(start.xi)
                ro = imagine(
                    self.wm, self.policy, self.task, start, cfg.horizon, cfg.future_frames,
                    obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs,
                    anchor_orientation_obs=cfg.anchor_orientation_obs,
                )
                with torch.no_grad():
                    if hasattr(self.task, "termination_probability"):
                        initial_term_prob = float(self.task.termination_probability(start.state).mean())
                    else:
                        initial_term_prob = float(torch.sigmoid(self.wm.term_head(ro.term_features[0])).mean())
                    self.last_imag_diagnostics = {
                        "real_replay_state_termination_probability": initial_term_prob,
                        "termination_probability": [float(x) for x in ro.dhat[1:].mean(dim=1)],
                        "survival": [float(x) for x in ro.survival[1:].mean(dim=1)],
                        "root_height_m": [float(x) for x in ro.root_height[1:].mean(dim=1)],
                        "tilt_feature": [float(x) for x in ro.tilt[1:].mean(dim=1)],
                    }
                feat, goals, e_root = self._critic_inputs(ro)
                if cfg.use_critic:
                    v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])
                    if cfg.critic_value_clip is not None:
                        v_H, _ = _clip_bootstrap_values(v_H, cfg.critic_value_clip)
                else:
                    v_H = None
                J = policy_objective(ro, v_H, cfg.gamma)
                # Average micro-batch objectives so the logical update has the same
                # scale as one full batch.
                (-J / cfg.horizon / n_micro).backward()

                objective_sum += float(J.detach())
                imag_return_sum += float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach())
                imag_reward_sum += float(ro.rewards.mean().detach())
                imag_survival_sum += float(ro.survival[-1].mean().detach())
                imag_dhat_sum += float(ro.dhat[1:].mean().detach())
                action_abs_sum += float(ro.actions.abs().mean().detach())
                for k, v in ro.reward_terms.items():
                    reward_metric_sum[k] = reward_metric_sum.get(k, 0.0) + v

                if cfg.use_critic:
                    # Retain only detached critic data after the Actor graph has been
                    # backpropagated; this avoids retaining four BPTT graphs.
                    with torch.no_grad():
                        feat_d, goals_d, e_root_d = feat.detach(), goals.detach(), e_root.detach()
                        H1, n, _ = feat_d.shape
                        xi_rep = xi_norm[None].expand(H1, n, -1)
                        raw_values = self.critic(
                            feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1),
                            xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1),
                        ).reshape(H1, n)
                        values, value_clip_fraction = _clip_bootstrap_values(raw_values, cfg.critic_value_clip)
                        targets = gae_targets(
                            ro.rewards.detach(), ro.dhat[1:].detach(), values,
                            cfg.gamma, cfg.gae_lambda,
                        )
                        critic_payload.append((
                            feat_d[:-1].reshape(-1, feat_d.shape[-1]),
                            goals_d[:-1].reshape(-1, goals_d.shape[-1]),
                            xi_rep[:-1].reshape(-1, xi_rep.shape[-1]),
                            e_root_d[:-1].reshape(-1, 4),
                            targets.reshape(-1),
                        ))
                        raw_value_sum += float(raw_values.mean())
                        target_sum += float(targets.mean())
                        value_clip_sum += float(value_clip_fraction)

            allreduce_grads(self.policy)
            gn = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True
            )
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
            self.opt.step()
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": objective_sum / n_micro,
            "imag_return_undisc": imag_return_sum / n_micro,
            "imag_reward_mean": imag_reward_sum / n_micro,
            "imag_survival_H": imag_survival_sum / n_micro,
            "imag_dhat_mean": imag_dhat_sum / n_micro,
            "policy_grad_norm": float(gn),
            "action_abs_mean": action_abs_sum / n_micro,
        }
        for k, v in reward_metric_sum.items():
            metrics[f"imag_r/{k}"] = v / n_micro

        # ---- critic: fit to GAE targets of the logical rollout batch -----------------------
        if cfg.use_critic:
            if not critic_payload:
                raise RuntimeError("critic is enabled but no policy micro-batch payload was collected")
            X_feat = torch.cat([p[0] for p in critic_payload], dim=0)
            X_goal = torch.cat([p[1] for p in critic_payload], dim=0)
            X_xi = torch.cat([p[2] for p in critic_payload], dim=0)
            X_er = torch.cat([p[3] for p in critic_payload], dim=0)
            Y = torch.cat([p[4] for p in critic_payload], dim=0)
            N = Y.shape[0]
            if N < cfg.critic_minibatches or N % cfg.critic_minibatches:
                raise ValueError(
                    f"critic logical batch ({N}) must be divisible by critic_minibatches ({cfg.critic_minibatches})"
                )
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), cfg.critic_grad_clip, error_if_nonfinite=True
                    )
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = raw_value_sum / n_micro
            metrics["critic_target_mean"] = target_sum / n_micro
            metrics["critic_value_clip_fraction"] = value_clip_sum / n_micro
        return metrics
