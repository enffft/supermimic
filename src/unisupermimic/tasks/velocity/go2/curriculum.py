"""Episode-performance terrain selection, independent from velocity reward.

Progress is signed displacement along the ENTRY direction. Standing still and
pure turning cannot promote. Fixed evaluation disables all adaptation/recycling.
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
    ):
        self.origins = torch.tensor(origins.copy(), device=device, dtype=torch.float32)
        if not 0 <= fixed_level <= max_level < self.origins.shape[0]:
            raise ValueError("Terrain level is outside generated atlas")
        if not 0 <= recycle_probability <= 1:
            raise ValueError("Invalid curriculum recycle probability")
        self.enabled, self.max_level = enabled, max_level
        self.recycle_probability = recycle_probability
        self.mastery = torch.full((num_envs,), fixed_level, device=device, dtype=torch.long)
        self.terrain_level = self.mastery.clone()
        self.terrain_type = torch.arange(num_envs, device=device) % self.origins.shape[1]
        self.episode_progress = torch.zeros(num_envs, device=device)
        self.commanded_progress = torch.zeros(num_envs, device=device)
        self.episode_fall = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.velocity_tracking_error = torch.zeros(num_envs, device=device)
        self.steps = torch.zeros(num_envs, device=device)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.last_decision = torch.zeros(num_envs, device=device, dtype=torch.long)

    def observe(self, state, goal, fallen, heading):
        p, rotation, v, *_ = state
        direction = p.new_tensor((math.cos(heading), math.sin(heading)))
        self.episode_progress.copy_(((p[..., :2] - self.start_xy) * direction).sum(-1))
        local_v = (rotation.transpose(-1, -2) @ v[..., None]).squeeze(-1)
        self.velocity_tracking_error.add_((local_v[..., :2] - goal[..., :2]).norm(dim=-1))
        commanded_world = (rotation[..., :2, :2] @ goal[..., :2, None]).squeeze(-1)
        self.commanded_progress.add_((commanded_world * direction).sum(-1).clamp_min(0) * 0.02)
        self.episode_fall.logical_or_(fallen)
        self.steps.add_(1)

    def reset(self, ids, rng):
        count = self.steps[ids]
        error = self.velocity_tracking_error[ids] / count.clamp_min(1)
        # Cross the approach border and substantial ascending/descending region.
        success = (
            (self.episode_progress[ids] >= 3.5)
            & (self.commanded_progress[ids] >= 1.0)
            & ~self.episode_fall[ids]
            & (error < 0.35)
        )
        failure = self.episode_fall[ids] | (
            (count >= 250)
            & (self.commanded_progress[ids] >= 1.0)
            & (self.episode_progress[ids] < 0.4)
        )
        # Recycled easy episodes maintain practice, never certify an unseen level.
        success = success & (self.terrain_level[ids] >= self.mastery[ids])
        decision = success.long() - (failure & ~success).long()
        decision = torch.where(count > 0, decision, 0)
        if self.enabled:
            self.mastery[ids] = (self.mastery[ids] + decision).clamp(0, self.max_level)
            # Revisit lower levels without losing the mastered ceiling.
            recycle = rng.random((ids.numel(),)) < self.recycle_probability
            lower = (rng.random((ids.numel(),)) * self.mastery[ids].clamp_min(1)).long()
            self.terrain_level[ids] = torch.where(recycle, lower, self.mastery[ids])
        self.last_decision[ids] = decision if self.enabled else 0
        self.episode_progress[ids] = 0
        self.commanded_progress[ids] = 0
        self.episode_fall[ids] = False
        self.velocity_tracking_error[ids] = 0
        self.steps[ids] = 0
        selected = self.origins[self.terrain_level[ids], self.terrain_type[ids]]
        self.start_xy[ids] = selected[:, :2]
        return selected
