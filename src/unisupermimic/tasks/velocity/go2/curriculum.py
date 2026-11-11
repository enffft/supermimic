"""Separate privileged WM data coverage from measured actor terrain mastery.

All statistics are metadata, never actor observations or dynamics context.
A promotion certifies an entry-region traversal, not an entire eight-metre tile.
"""

import math

import torch


class TerrainCurriculum:
    def __init__(
        self,
        num_envs,
        origins,
        *,
        max_level=4,
        enabled=False,
        fixed_level=0,
        recycle_probability=0.15,
        device="cpu",
        phase="skill",
        coverage_weights=(0.2, 0.4, 0.3, 0.1, 0.0),
        min_episodes=3,
        target_progress=1.2,
        flat_probability=0.15,
        probe_probability=0.0,
    ):
        self.origins = torch.tensor(origins.copy(), device=device, dtype=torch.float32)
        if not 0 <= fixed_level <= max_level < self.origins.shape[0]:
            raise ValueError("Terrain level is outside generated atlas")
        if phase not in ("skill", "wm", "fixed"):
            raise ValueError("phase must be skill, wm or fixed")
        if any(not 0 <= x <= 1 for x in (recycle_probability, flat_probability, probe_probability)):
            raise ValueError("Invalid curriculum probability")
        weights = torch.tensor(coverage_weights, device=device, dtype=torch.float32)
        if len(weights) != self.origins.shape[0] or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Invalid WM level coverage weights")
        weights[max_level + 1 :] = 0
        self.coverage_cdf = weights.cumsum(0) / weights.sum()
        self.enabled, self.max_level, self.phase = enabled, max_level, phase
        self.recycle_probability, self.flat_probability = recycle_probability, flat_probability
        self.probe_probability = probe_probability
        self.min_episodes, self.target_progress = min_episodes, target_progress
        self.mastery = torch.full((num_envs,), fixed_level, device=device, dtype=torch.long)
        self.terrain_level = self.mastery.clone()
        self.terrain_type = torch.arange(num_envs, device=device) % self.origins.shape[1]
        self.fixed_domain = None
        self.crossing_episode = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.last_decision = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.probe = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.probe_offset = torch.zeros(num_envs, device=device)
        self.success_streak = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.failure_streak = torch.zeros_like(self.success_streak)
        for name in (
            "episode_progress",
            "commanded_progress",
            "velocity_tracking_error",
            "yaw_tracking_error",
            "steps",
            "eligible_steps",
            "max_lateral",
        ):
            setattr(self, name, torch.zeros(num_envs, device=device))
        self.episode_fall = torch.zeros(num_envs, device=device, dtype=torch.bool)
        shape = (self.origins.shape[1], self.origins.shape[0])
        self.statistics = {
            name: torch.zeros(shape, device=device, dtype=torch.float64)
            for name in (
                "episodes",
                "crossing_episodes",
                "crossing_falls",
                "success",
                "falls",
                "timeouts",
                "xy_sq_sum",
                "yaw_sq_sum",
                "steps",
                "progress",
                "commanded_progress",
                "promotions",
                "demotions",
                "probes",
            )
        }

    def observe(self, state, goal, fallen, heading):
        p, rotation, v, w, *_ = state
        direction = p.new_tensor((math.cos(heading), math.sin(heading)))
        lateral = p.new_tensor((-math.sin(heading), math.cos(heading)))
        displacement = p[..., :2] - self.start_xy
        self.episode_progress.copy_((displacement * direction).sum(-1))
        self.max_lateral.copy_(
            torch.maximum(self.max_lateral, (displacement * lateral).sum(-1).abs())
        )
        local_v = (rotation.transpose(-1, -2) @ v[..., None]).squeeze(-1)
        local_w = (rotation.transpose(-1, -2) @ w[..., None]).squeeze(-1)
        self.velocity_tracking_error.add_((local_v[..., :2] - goal[..., :2]).square().sum(-1))
        self.yaw_tracking_error.add_((local_w[..., 2] - goal[..., 2]).square())
        commanded_world = (rotation[..., :2, :2] @ goal[..., :2, None]).squeeze(-1)
        forward = (commanded_world * direction).sum(-1)
        self.commanded_progress.add_(forward.clamp_min(0) * 0.02)
        eligible = (
            (goal[:, 0] >= 0.15)
            & (goal[:, 1].abs() <= 0.1)
            & (goal[:, 2].abs() <= 0.2)
            & (forward >= 0.15)
        )
        self.eligible_steps.add_(eligible)
        self.episode_fall.logical_or_(fallen)
        self.steps.add_(1)

    def reset(self, ids, rng):
        count = self.steps[ids]
        completed = (count >= 499) | self.episode_fall[ids]
        ratio = self.episode_progress[ids] / self.commanded_progress[ids].clamp_min(1e-6)
        crossing = (
            self.crossing_episode[ids]
            & (self.eligible_steps[ids] >= 0.9 * count)
            & (self.commanded_progress[ids] >= 1.0)
        )
        success = (
            completed
            & crossing
            & ~self.probe[ids]
            & ~self.episode_fall[ids]
            & (ratio >= 0.7)
            & (self.episode_progress[ids] >= self.target_progress)
            & (self.max_lateral[ids] <= 0.6)
            & ((self.velocity_tracking_error[ids] / count.clamp_min(1)).sqrt() < 0.35)
            & ((self.yaw_tracking_error[ids] / count.clamp_min(1)).sqrt() < 0.4)
        )
        stalled = completed & crossing & (ratio < 0.2)
        failure = self.episode_fall[ids] | stalled
        if self.fixed_domain is not None:
            failure &= self.crossing_episode[ids]
        # Manual reset/probes are counted separately and can never certify skill.
        recorded = completed & (count > 0)
        indices = self.terrain_type[ids] * self.origins.shape[0] + self.terrain_level[ids]
        metrics = dict(
            episodes=recorded,
            crossing_episodes=recorded & self.crossing_episode[ids],
            crossing_falls=recorded & self.crossing_episode[ids] & self.episode_fall[ids],
            success=success,
            falls=recorded & self.episode_fall[ids],
            timeouts=recorded & ~self.episode_fall[ids],
            xy_sq_sum=self.velocity_tracking_error[ids] * recorded,
            yaw_sq_sum=self.yaw_tracking_error[ids] * recorded,
            steps=count * recorded,
            progress=self.episode_progress[ids] * recorded,
            commanded_progress=self.commanded_progress[ids] * recorded,
            probes=recorded & self.probe[ids],
        )
        for name, values in metrics.items():
            self.statistics[name].view(-1).scatter_add_(0, indices, values.double())
        decision = torch.zeros_like(ids)
        if self.enabled and self.phase == "skill":
            at_ceiling = (
                (self.terrain_level[ids] == self.mastery[ids]) & ~self.probe[ids] & recorded
            )
            if self.fixed_domain is not None:
                at_ceiling &= self.crossing_episode[ids]
            self.success_streak[ids] = torch.where(
                at_ceiling,
                torch.where(success, self.success_streak[ids] + 1, 0),
                self.success_streak[ids],
            )
            self.failure_streak[ids] = torch.where(
                at_ceiling,
                torch.where(failure, self.failure_streak[ids] + 1, 0),
                self.failure_streak[ids],
            )
            promote = (
                at_ceiling
                & (self.success_streak[ids] >= self.min_episodes)
                & (self.mastery[ids] < self.max_level)
            )
            demote = at_ceiling & (self.failure_streak[ids] >= 2) & (self.mastery[ids] > 0)
            decision = promote.long() - demote.long()
            self.mastery[ids] += decision
            self.success_streak[ids] = torch.where(decision != 0, 0, self.success_streak[ids])
            self.failure_streak[ids] = torch.where(decision != 0, 0, self.failure_streak[ids])
            for name, values in (("promotions", promote), ("demotions", demote)):
                self.statistics[name].view(-1).scatter_add_(0, indices, values.double())
            recycle = rng.random((ids.numel(),)) < self.recycle_probability
            lower = (rng.random((ids.numel(),)) * self.mastery[ids].clamp_min(1)).long()
            selected = torch.where(recycle, lower, self.mastery[ids])
            flat = rng.random((ids.numel(),)) < self.flat_probability
            self.terrain_level[ids] = torch.where(flat, 0, selected)
        elif self.enabled and self.phase == "wm":
            draws = rng.random((ids.numel(),))
            self.terrain_level[ids] = torch.searchsorted(self.coverage_cdf, draws).clamp_max(
                self.max_level
            )
        if self.fixed_domain is not None:
            self.terrain_level[ids] = torch.where(
                self.fixed_domain[ids] == 0, 0, self.terrain_level[ids]
            )
        self.probe[ids] = (self.enabled and self.phase == "wm") & (
            rng.random((ids.numel(),)) < self.probe_probability
        )
        self.probe_offset[ids] = torch.where(
            self.probe[ids], 0.8 + 1.4 * rng.random((ids.numel(),)), 0.0
        )
        self.last_decision[ids] = decision
        for name in (
            "episode_progress",
            "commanded_progress",
            "velocity_tracking_error",
            "yaw_tracking_error",
            "steps",
            "eligible_steps",
            "max_lateral",
            "episode_fall",
        ):
            getattr(self, name)[ids] = 0
        selected_origins = self.origins[self.terrain_level[ids], self.terrain_type[ids]].clone()
        if self.fixed_domain is not None:
            # Atlas entry is .6m from the X edge, center is +3.4m.
            selected_origins[:, 0] += (self.fixed_domain[ids] == 0) * 3.4
        self.start_xy[ids] = selected_origins[:, :2]
        return selected_origins

    def state_dict(self):
        return {
            "phase": self.phase,
            "tensors": {k: v.clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)},
            "statistics": {k: v.clone() for k, v in self.statistics.items()},
        }

    def load_state_dict(self, values):
        if values["phase"] != self.phase:
            raise ValueError(
                "Explicit phase transition required when resuming a different curriculum"
            )
        for name, tensor in values["tensors"].items():
            current = getattr(self, name)
            if current.shape != tensor.shape:
                raise ValueError(f"Curriculum shape changed: {name}")
            current.copy_(tensor)
        for name, tensor in values["statistics"].items():
            self.statistics[name].copy_(tensor)

    def summary(self):
        out = {k: v.cpu().tolist() for k, v in self.statistics.items()}
        episodes = self.statistics["episodes"].clamp_min(1)
        steps = self.statistics["steps"].clamp_min(1)
        for name in ("success", "falls", "timeouts"):
            out[name + "_rate"] = (self.statistics[name] / episodes).cpu().tolist()
        crossing_episodes = self.statistics["crossing_episodes"].clamp_min(1)
        out["crossing_success_rate"] = (
            (self.statistics["success"] / crossing_episodes).cpu().tolist()
        )
        out["crossing_fall_rate"] = (
            (self.statistics["crossing_falls"] / crossing_episodes).cpu().tolist()
        )
        out["xy_rmse_m_s"] = (self.statistics["xy_sq_sum"] / steps).sqrt().cpu().tolist()
        out["yaw_rmse_rad_s"] = (self.statistics["yaw_sq_sum"] / steps).sqrt().cpu().tolist()
        out["mastery_histogram"] = torch.bincount(self.mastery, minlength=5).cpu().tolist()
        out["level_histogram"] = torch.bincount(self.terrain_level, minlength=5).cpu().tolist()
        if self.fixed_domain is not None:
            terrain_envs = self.fixed_domain != 0
            out["terrain_domain_mastery_histogram"] = (
                torch.bincount(self.mastery[terrain_envs], minlength=5).cpu().tolist()
            )
            out["fixed_flat_envs"] = int((~terrain_envs).sum())
        out["phase"] = self.phase
        return out
