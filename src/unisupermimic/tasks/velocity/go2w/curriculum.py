"""Ten-level, per-skill mastery with bounded coverage roles and mixed review."""

from dataclasses import dataclass

import torch

LEVEL_COUNT = 10
DOMAINS = (
    "flat",
    "rough",
    "slope_up",
    "slope_down",
    "single_step_up",
    "single_step_down",
    "multi_stairs_up",
    "multi_stairs_down",
)
MODE_MAP = (0, 1, 2, 2, 3, 3, 3, 3)
SINGLE_HEIGHTS = (0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.20, 0.22)
STAIR_HEIGHTS = (0.03, 0.03, 0.03, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
MIX = (
    (1, 0, 0, 0),
    (0.85, 0.15, 0, 0),
    (0.75, 0.15, 0.10, 0),
    (0.70, 0.15, 0.10, 0.05),
    (0.65, 0.15, 0.10, 0.10),
    (0.60, 0.15, 0.15, 0.10),
    (0.60, 0.15, 0.15, 0.10),
    (0.55, 0.15, 0.20, 0.10),
    (0.55, 0.15, 0.20, 0.10),
    (0.50, 0.15, 0.20, 0.15),
)


@dataclass
class UnifiedCurriculumCfg:
    level_count: int = LEVEL_COUNT
    window: int = 64
    hysteresis: int = 2
    coverage_fraction: float = 0.125
    review_fraction: float = 0.15
    seed: int = 2026

    def validate(self):
        if self.level_count != LEVEL_COUNT or self.window < 16 or self.hysteresis < 1:
            raise ValueError("Unified Go2W requires ten levels and valid mastery windows")
        if not 0 <= self.coverage_fraction <= 0.25 or not 0 <= self.review_fraction <= 0.5:
            raise ValueError("Invalid curriculum mixture")


class UnifiedSampler:
    """Commands are balanced by signed axis; stair commands remain forward."""

    def __init__(self, cfg, device):
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.speed_level = 0
        self.stairs_levels = torch.zeros(2, LEVEL_COUNT, dtype=torch.long, device=device)

    def sample(self, domain, level):
        from unisupermimic.tasks.velocity.go2.command_sampler import command_modes

        n = len(domain)
        rand = torch.rand(n, 6, generator=self.rng, device=self.device)
        signed_axis = (rand[:, 0] * 7).long()  # stop + six signed single axes
        axis = ((signed_axis - 1).clamp_min(0) // 2).clamp_max(2)
        sign = torch.where(signed_axis % 2 == 1, 1.0, -1.0)
        magnitude = 0.10 + 0.15 * rand[:, 1]
        magnitude = torch.where(axis == 0, 0.15 + 0.45 * rand[:, 1], magnitude)
        magnitude = torch.where(axis == 2, 0.20 + 0.15 * rand[:, 1], magnitude)
        goal = torch.zeros(n, 3, device=self.device)
        goal.scatter_(1, axis[:, None], (magnitude * sign)[:, None])
        goal[signed_axis == 0] = 0
        # Increasing mixed commands never removes signed single-axis review.
        mixed = (rand[:, 2] < (0.03 * level).clamp(max=0.25)) & (domain < 4)
        goal[mixed] = (rand[mixed, 3:6] * 2 - 1) * goal.new_tensor([0.5, 0.20, 0.35])
        nonflat = (domain > 0) & (domain < 4)
        goal[nonflat] *= 0.7
        stair = domain >= 4
        goal[stair] = torch.stack(
            (
                0.20 + 0.15 * rand[stair, 3],
                (rand[stair, 4] - 0.5) * 0.04,
                (rand[stair, 5] - 0.5) * 0.12,
            ),
            -1,
        )
        mode = command_modes(goal)
        return goal, mode, stair, torch.full((n,), 10.02, device=self.device)

    def state_dict(self):
        return dict(rng=self.rng.get_state())

    def load_state_dict(self, values, **kwargs):
        self.rng.set_state(values["rng"].cpu())


class UnifiedCurriculum:
    """Vectorized per-env reset, per signed skill window statistics, WM isolation."""

    level_count = LEVEL_COUNT
    num_domains = len(DOMAINS)

    def __init__(self, num_envs, origins, cfg, command_domain, device="cpu"):
        cfg.validate()
        self.cfg = cfg
        self.origins = torch.as_tensor(origins.copy(), device=device, dtype=torch.float32)
        if self.origins.shape[:2] != (LEVEL_COUNT, 7):
            raise ValueError("Unified terrain atlas must be 10 x 7")
        self.rng = torch.Generator(device=torch.device(device)).manual_seed(cfg.seed)
        self.enabled, self.phase = True, "skill"
        self.fixed_domain = command_domain
        self.skill_levels = torch.zeros(14, dtype=torch.long, device=device)
        self.pass_windows = torch.zeros_like(self.skill_levels)
        self.fail_windows = torch.zeros_like(self.skill_levels)
        self.window_counts = torch.zeros(14, device=device)
        self.window_success = torch.zeros(14, device=device)
        self.window_falls = torch.zeros(14, device=device)
        self.mastery = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.terrain_level = self.mastery.clone()
        self.terrain_type = self.mastery.clone()
        self.skill = self.mastery.clone()
        self.coverage_level = torch.full_like(self.mastery, -1)
        coverage = round(num_envs * cfg.coverage_fraction)
        ids = torch.arange(coverage, device=device)
        self.coverage_level[ids] = ids % LEVEL_COUNT
        command_domain[ids] = 1 + (ids // LEVEL_COUNT) % 7
        self.probe = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.probe_offset = torch.zeros(num_envs, device=device)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.crossing_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.steps = torch.zeros(num_envs, device=device)
        self.error = torch.zeros(num_envs, device=device)
        self.progress = torch.zeros(num_envs, device=device)
        self.yaw_progress = torch.zeros(num_envs, device=device)
        self.tilt = torch.zeros(num_envs, device=device)
        self.episode_fall = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.traversal_success = torch.zeros_like(self.episode_fall)
        self.statistics = {
            name: torch.zeros(7, LEVEL_COUNT, device=device, dtype=torch.float64)
            for name in (
                "real_transitions",
                "real_valid_transitions",
                "real_falls",
                "episodes",
                "success",
                "falls",
            )
        }
        self.promotions = torch.zeros(14, dtype=torch.long, device=device)
        self.demotions = torch.zeros_like(self.promotions)

    def observe(self, state, goal, fallen, heading):
        from unisupermimic.rl.algos.supermimic.geometry import matvec

        v = matvec(state[1].transpose(-1, -2), state[2])
        w = matvec(state[1].transpose(-1, -2), state[3])
        actual = torch.cat((v[:, :2], w[:, 2:3]), -1)
        self.error += (actual - goal[:, :3]).square().sum(-1)
        direction = goal[:, :2] / goal[:, :2].norm(dim=-1, keepdim=True).clamp_min(0.01)
        self.progress += (v[:, :2] * direction).sum(-1) * 0.02
        self.yaw_progress += w[:, 2] * goal[:, 2].sign() * 0.02
        self.tilt += (1 - state[1][:, 2, 2]).clamp_min(0)
        self.steps += 1
        self.episode_fall |= fallen
        axis = goal[:, :3].abs().argmax(-1)
        signed = 1 + axis * 2 + (goal.gather(1, axis[:, None])[:, 0] < 0).long()
        signed[goal[:, :3].norm(dim=-1) < 0.05] = 0
        # Six flat directions, rough, two slopes, and four separate stair branches.
        self.skill.copy_(
            torch.where(self.fixed_domain == 0, signed, (self.fixed_domain + 6).clamp(max=13))
        )

    def _mastery(self, ids):
        valid = (
            (self.steps[ids] >= 499) | self.episode_fall[ids] | self.traversal_success[ids]
        ) & (self.coverage_level[ids] < 0)
        ids = ids[valid]
        if not len(ids):
            return
        skills = self.skill[ids]
        effective = (self.error[ids] / self.steps[ids].clamp_min(1)).sqrt() < 0.15
        effective &= (self.skill[ids] == 0) | torch.where(
            self.skill[ids] >= 5, self.yaw_progress[ids] > 0.3, self.progress[ids] > 0.3
        )
        slope = (self.fixed_domain[ids] == 2) | (self.fixed_domain[ids] == 3)
        effective &= ~slope | (
            (self.progress[ids] > 1.0) & (self.tilt[ids] / self.steps[ids].clamp_min(1) < 0.2)
        )
        effective = torch.where(self.fixed_domain[ids] >= 4, self.traversal_success[ids], effective)
        success = effective & ~self.episode_fall[ids] & (self.steps[ids] >= 25)
        for dest, values in (
            (self.window_counts, torch.ones_like(self.steps[ids])),
            (self.window_success, success.float()),
            (self.window_falls, self.episode_fall[ids].float()),
        ):
            dest.scatter_add_(0, skills, values)
        ready = self.window_counts >= self.cfg.window
        good = (
            ready
            & (self.window_success / self.window_counts.clamp_min(1) >= 0.70)
            & (self.window_falls / self.window_counts.clamp_min(1) < 0.15)
        )
        bad = ready & (
            (self.window_success / self.window_counts.clamp_min(1) < 0.35)
            | (self.window_falls / self.window_counts.clamp_min(1) > 0.35)
        )
        self.pass_windows = torch.where(
            ready, torch.where(good, self.pass_windows + 1, 0), self.pass_windows
        )
        self.fail_windows = torch.where(
            ready, torch.where(bad, self.fail_windows + 1, 0), self.fail_windows
        )
        up = (self.pass_windows >= self.cfg.hysteresis) & (self.skill_levels < LEVEL_COUNT - 1)
        down = (self.fail_windows >= self.cfg.hysteresis) & (self.skill_levels > 0)
        self.skill_levels += up.long() - down.long()
        self.promotions += up.long()
        self.demotions += down.long()
        self.pass_windows[up | down] = 0
        self.fail_windows[up | down] = 0
        self.window_counts[ready] = 0
        self.window_success[ready] = 0
        self.window_falls[ready] = 0

    def reset(self, ids, rng):
        self._mastery(ids)
        covered = self.coverage_level[ids] >= 0
        if self.enabled:
            # Terrain branches get bounded probes from level 1; no global lock on lateral skill.
            current = self.skill_levels[1:7].max()
            mix = torch.tensor(MIX, device=ids.device)[current]
            weights = torch.stack(
                (
                    mix[0],
                    mix[1] / 3,
                    mix[1] / 3,
                    mix[1] / 3,
                    mix[2] / 2,
                    mix[2] / 2,
                    mix[3] / 2,
                    mix[3] / 2,
                )
            )
            domain = torch.multinomial(weights.expand(len(ids), -1), 1, generator=self.rng)[:, 0]
            self.fixed_domain[ids] = torch.where(covered, self.fixed_domain[ids], domain)
            domain = self.fixed_domain[ids]
            ceiling = torch.where(
                domain == 0,
                self.skill_levels[:7].max(),
                self.skill_levels[(domain + 6).clamp(max=13)],
            )
            level = ceiling.clone()
            review = (
                torch.rand(len(ids), device=ids.device, generator=self.rng)
                < self.cfg.review_fraction
            )
            lower = (
                torch.rand(len(ids), device=ids.device, generator=self.rng) * (ceiling + 1)
            ).long()
            level = torch.where(review, lower, level)
            self.terrain_level[ids] = torch.where(covered, self.coverage_level[ids], level)
            self.mastery[ids] = ceiling
            self.terrain_type[ids] = (domain - 1).clamp_min(0)
        self.steps[ids] = 0
        self.error[ids] = 0
        self.progress[ids] = 0
        self.yaw_progress[ids] = 0
        self.tilt[ids] = 0
        self.episode_fall[ids] = False
        self.traversal_success[ids] = False
        origins = self.origins[self.terrain_level[ids], self.terrain_type[ids]].clone()
        # A separate level-1 rough tile is perfectly flat; center gives room for all directions.
        flat = self.fixed_domain[ids] == 0
        origins[flat] = self.origins[0, 0]
        origins[flat, 0] += 3.4
        self.start_xy[ids] = origins[:, :2]
        return origins

    def start_scope_mask(self):
        ceilings = torch.cat((self.skill_levels[:7].max()[None], self.skill_levels[7:14]))
        return torch.arange(LEVEL_COUNT, device=self.mastery.device)[None] <= ceilings[:, None]

    def state_dict(self):
        return dict(
            tensors={k: v.clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)},
            statistics=self.statistics,
            rng=self.rng.get_state(),
            phase=self.phase,
        )

    def load_state_dict(self, values):
        for k, v in values["tensors"].items():
            if getattr(self, k).shape != v.shape:
                raise ValueError(f"Curriculum shape changed: {k}")
            getattr(self, k).copy_(v)
        self.statistics = {k: v.to(self.mastery.device) for k, v in values["statistics"].items()}
        self.rng.set_state(values["rng"].cpu())

    def summary(self):
        scopes = self.fixed_domain * LEVEL_COUNT + self.terrain_level
        return dict(
            skill_levels=(self.skill_levels + 1).cpu().tolist(),
            domain_level=torch.bincount(scopes, minlength=8 * LEVEL_COUNT)
            .reshape(8, LEVEL_COUNT)
            .cpu()
            .tolist(),
            coverage_environments=int((self.coverage_level >= 0).sum()),
            promotions=self.promotions.cpu().tolist(),
            demotions=self.demotions.cpu().tolist(),
        )
