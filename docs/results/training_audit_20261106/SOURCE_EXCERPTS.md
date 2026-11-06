# 审计时主机源码节选

这些是实际执行/检查的工作区文本，SHA256 见 provenance.json。行号对应主机工作区，不假定等同于报告提交的训练源码。

## src/supermimic/imagination/objective.py:1–35

```python
"""Policy objective (Eq. 2) and critic GAE targets (Eq. 3)."""

from __future__ import annotations

import torch

from supermimic.imagination.rollout import ImagRollout


def policy_objective(ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H ] (SHAC-style; V_H keeps the gradient w.r.t. s_H).

    terminal_values: (n,) critic estimate at s_H, or None for the critic-free variant.
    """
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + (gamma**H) * ro.survival[-1] * terminal_values
    return J.mean()


@torch.no_grad()
def gae_targets(rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out
```

## src/supermimic/imagination/policy_trainer.py:156–207

```python
                    terminal = self.task.terminal_reward(ro.states, start.goal)
                    J = J + cfg.gamma ** cfg.horizon * terminal.mean()
                    reward_metric_sum["terminal_distance"] = reward_metric_sum.get("terminal_distance", 0.) + float(terminal.mean().detach())
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
```

## src/supermimic/imagination/policy_trainer.py:232–255

```python
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
```

## src/supermimic/imagination/tasks.py:310–330

```python
                               self.reward_cfg.success_distance_m,
                               self.params.fall_height, self.params.tilt_feature_threshold)

    def terminal_reward(self, states, context):
        """Horizon endpoint, in reward units once (no dt), active only for M.

        A completed current goal has no residual endpoint cost. The detached
        arrival mask truncates this goal without imagining random new targets.
        Physical survival does not erase a negative distance cost.
        """
        from supermimic.tasks.world_point_reward import tcp_position_world
        delta = tcp_position_world(states[-1], self.fk, self.tcp_body_indices) - context[..., 4:7]
        distance = torch.sqrt(delta.square().sum(-1) + 1.e-8) - 1.e-4
        active = context[..., 0] >= .5
        if self.resample_on_success:
            with torch.no_grad():
                predicted = states[1:].detach()
                alive = (self.termination_features(predicted) <= 0).all(-1).to(torch.int64).cumprod(0).bool()
                completed = (self.goal_reached(predicted, context) & alive).any(0)
                active = active & ~completed
        return -self.terminal_distance_weight * active.to(distance.dtype) * distance
```

## src/supermimic/tasks/m1_reward.py:138–168

```python
    gravity_b = quat_apply_inverse(s_next[..., 3:7], gravity)
    q, qd = s_next[..., 7:29], s_next[..., 35:57]
    qdd = (qd - s_prev[..., 35:57]) / dt
    position_mask = q.new_ones(22); position_mask[[3, 7, 11, 15]] = 0.0
    leg_ids = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    limit_error = (F.relu(soft_limits[..., 0] - q).square() + F.relu(q - soft_limits[..., 1]).square()) * position_mask
    return {
        "l_linear": is_l * cfg.linear_tracking * torch.exp(-xy_sq / cfg.linear_sigma**2),
        "l_yaw": is_l * cfg.yaw_tracking * torch.exp(-yaw_error.square() / cfg.yaw_sigma**2),
        "m_reach_coarse": is_m * cfg.reach_coarse * torch.exp(-d_next.square() / cfg.reach_coarse_sigma**2),
        "m_reach_fine": is_m * cfg.reach_fine * near,
        "m_progress": is_m * cfg.reach_progress * torch.tanh((d_prev - d_next) / (dt * 0.5)),
        "m_hold": is_m * cfg.reach_hold * near * torch.exp(-tcp_speed_sq / cfg.hold_speed_sigma**2),
        "upright": cfg.upright * torch.exp(-(1.0 + gravity_b[..., 2]) / 0.1),
        "height": cfg.height * torch.exp(-((s_next[..., 2] - base_height) / 0.10).square()),
        "vertical_velocity": cfg.vertical_velocity * vz.square(),
        "roll_pitch_rate": cfg.roll_pitch_rate * s_next[..., 32:34].square().sum(-1),
        "leg_pose": cfg.leg_pose * (q[..., leg_ids] - default_joint_pos[..., leg_ids]).square().sum(-1),
        "l_arm_carry": is_l * cfg.arm_carry * (q[..., 16:22] - default_joint_pos[..., 16:22]).square().sum(-1),
        "action_rate": cfg.action_rate * (action - previous_action).square().mean(-1),
        "joint_limit": cfg.joint_limit * limit_error.sum(-1),
        "joint_velocity": cfg.joint_velocity * qd.square().mean(-1),
        "joint_acceleration": cfg.joint_acceleration * qdd.square().mean(-1),
        "control_effort": cfg.control_effort * action.square().mean(-1),
        "m_base_speed": is_m * cfg.mani_base_speed * (xy_error.square().sum(-1) + s_next[..., 34].square()),
        "m_distance_time": -is_m * cfg.distance_time_weight * d_next / (d_next + cfg.distance_time_scale_m),
        "m_success": (cfg.success_bonus / dt) * m1_goal_reached(
            s_next, context, fk, tcp_body_indices, cfg.success_distance_m, fall_height, tilt_feature_threshold).to(s_next.dtype),
    }


```

## src/supermimic/tasks/go2_gait.py:29–64

```python
def go2_gait_reward_terms(
    state: torch.Tensor, goal: torch.Tensor, cfg: Go2GaitCfg,
) -> dict[str, torch.Tensor]:
    """Encourage swing clearance, diagonal support, and stationary stance feet.

    No phase clock or contact history is required: the WM predicts the same
    joint/root state used by real rewards. Contact is a smooth geometric proxy,
    NOT a measured contact force. Ground is z=0 in the current velocity task.
    """
    zero = state[..., 0] * 0.0
    if cfg.feet_clearance == cfg.diagonal_gait == cfg.feet_slip == 0.0:
        return dict.fromkeys(GO2_GAIT_REWARD_NAMES, zero)
    pos, vel = go2_feet_world(state)
    height = pos[..., 2] - GO2_FOOT_RADIUS
    speed_sq = vel[..., :2].square().sum(-1)
    # Includes pure yaw commands (0.25 m is a representative foot lever arm).
    command_speed = (
        goal[..., :2].square().sum(-1) + (0.25 * goal[..., 2]).square() + 1e-8
    ).sqrt() - 1e-4
    moving = ((command_speed - 0.03) / 0.12).clamp(0.0, 1.0)
    swing_speed = torch.tanh((torch.sqrt(speed_sq + 1e-6) - 0.001) / 0.5)
    clearance = torch.exp(-((height - cfg.clearance_target) / cfg.clearance_sigma).square())
    swing = torch.sigmoid((height - 0.025) / 0.008)
    # FL/RR and FR/RL move together; both diagonals airborne or grounded cost
    # more than two-foot support. This is a soft coordination bias, not a timer.
    diagonal_error = 0.5 * (
        (swing[..., 0] - swing[..., 3]).square()
        + (swing[..., 1] - swing[..., 2]).square()
    )
    support_error = ((swing.sum(-1) - 2.0) / 2.0).square()
    stance = torch.sigmoid((0.015 - height) / 0.005)
    return {
        "feet_clearance": cfg.feet_clearance * moving * (clearance * swing_speed).mean(-1),
        "diagonal_gait": cfg.diagonal_gait * moving * (diagonal_error + support_error),
        "feet_slip": cfg.feet_slip * (stance * speed_sq).mean(-1),
    }
```

## src/supermimic/policy/nets.py:119–136

```python
        elif self.goal_encoding == "encoded":
            input_dim = cfg.latent_dim + obs_dim
        else:
            raise ValueError(f"unknown goal encoding {self.goal_encoding!r}")
        self.mlp = MLP(input_dim, cfg.decoder_hidden, act_dim, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z_or_goal: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        obs_scaled = obs_hist * self.scale
        if self.goal_encoding == "direct":
            # Direct velocity mode follows [scaled 420-D proprioception, scaled 3-D goal].
            action = self.mlp(torch.cat([obs_scaled, z_or_goal * self.goal_scale], dim=-1))
            return torch.tanh(action) if self.action_output == "tanh" else action
        return self.mlp(torch.cat([z_or_goal, obs_scaled], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""
```
