"""Per-environment episode-distance curriculum and fixed physical domain quotas."""

from dataclasses import asdict, dataclass

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
CURRICULUM_SCHEMA = "go2w_per_env_distance_v1"


def quotas(n, weights):
    weights = torch.as_tensor(weights, dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid domain weights")
    expected = n * weights / weights.sum()
    result = expected.floor().long()
    result[
        torch.argsort(expected - result, descending=True, stable=True)[: n - int(result.sum())]
    ] += 1
    return result


@dataclass
class UnifiedCurriculumCfg:
    level_count: int = 10
    coverage_fraction: float = 0.125
    review_fraction: float = 0.20
    seed: int = 2026
    domain_weights: tuple[float, ...] = (0.40, 0.12, 0.08, 0.08, 0.12, 0.08, 0.07, 0.05)
    min_steps: int = 25
    min_translation: float = 0.4
    min_rotation: float = 0.4
    flat_promote: float = 0.65
    flat_demote: float = 0.25
    terrain_promote: float = 0.60
    terrain_demote: float = 0.20
    stop_seconds: float = 5.0
    stop_linear_rms: float = 0.15
    stop_angular_rms: float = 0.35
    lane_half_width: float = 0.5
    landing_height_tolerance: float = 0.16
    crossing_margins: tuple[float, ...] = (0.25, 0.15, 0.30, 0.20)

    def validate(self):
        if self.level_count != 10 or len(self.domain_weights) != 8:
            raise ValueError("Ten levels/eight domains required")
        quotas(4096, self.domain_weights)
        if not 0 <= self.coverage_fraction <= 0.25 or not 0 <= self.review_fraction <= 0.5:
            raise ValueError("Invalid coverage/review fractions")
        if (
            not 0 <= self.flat_demote < self.flat_promote <= 1
            or not 0 <= self.terrain_demote < self.terrain_promote <= 1
        ):
            raise ValueError("Invalid progress thresholds")
        if len(self.crossing_margins) != 4 or min(self.crossing_margins) <= 0 or self.min_steps < 1:
            raise ValueError("Invalid crossing protocol")


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
        forward = domain >= 2
        goal[forward] = torch.stack(
            (
                0.20 + 0.15 * rand[forward, 3],
                (rand[forward, 4] - 0.5) * 0.04,
                (rand[forward, 5] - 0.5) * 0.12,
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
    """Local levels advance once per genuine episode; summaries never unlock data."""

    level_count = 10
    num_domains = 8

    def __init__(self, num_envs, origins, cfg, command_domain, device="cpu"):
        cfg.validate()
        self.cfg = cfg
        self.origins = torch.as_tensor(origins.copy(), device=device, dtype=torch.float32)
        if self.origins.shape[:2] != (10, 7):
            raise ValueError("Expected 10x7 atlas")
        self.rng = torch.Generator(device=torch.device(device)).manual_seed(cfg.seed)
        self.enabled, self.phase = True, "skill"
        self.fixed_domain = command_domain
        self.coverage_level = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.assign_domains(cfg.domain_weights, cfg.coverage_fraction)
        self.mastery = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.terrain_level = self.mastery.clone()
        self.terrain_type = (self.fixed_domain - 1).clamp_min(0)
        self.flat_levels = torch.zeros(num_envs, 7, dtype=torch.long, device=device)
        self.skill = self.mastery.clone()
        self.skill_levels = torch.zeros(14, dtype=torch.long, device=device)  # diagnostic only
        self.probe = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.probe_offset = torch.zeros(num_envs, device=device)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.previous_xy = self.start_xy.clone()
        self.crossing_episode = self.probe.clone()
        for name in (
            "steps",
            "expected",
            "progress",
            "yaw_progress",
            "expected_yaw",
            "error",
            "tilt",
            "max_safe_progress",
        ):
            setattr(self, name, torch.zeros(num_envs, device=device))
        for name in (
            "episode_fall",
            "episode_ended",
            "traversal_success",
            "curriculum_crossed_obstacle",
        ):
            setattr(self, name, self.probe.clone())
        self.forward_only = torch.ones_like(self.probe)
        self.in_channel = torch.ones_like(self.probe)
        self.skill_episode_level = torch.full((num_envs, 7), -1, dtype=torch.long, device=device)
        self.skill_expected = torch.zeros(num_envs, 7, 2, device=device)
        self.skill_progress = torch.zeros_like(self.skill_expected)
        self.skill_energy = torch.zeros_like(self.skill_expected)
        self.skill_steps = torch.zeros(num_envs, 7, device=device)
        self.promotions = torch.zeros(8, dtype=torch.long, device=device)
        self.demotions = torch.zeros_like(self.promotions)
        self.holds = torch.zeros_like(self.promotions)
        self.distance_sum = torch.zeros(8, device=device)
        self.expected_sum = torch.zeros(8, device=device)
        self.statistics = {
            name: torch.zeros(7, 10, device=device, dtype=torch.float64)
            for name in (
                "real_transitions",
                "real_valid_transitions",
                "real_falls",
                "episodes",
                "success",
                "falls",
            )
        }

    def assign_domains(self, weights, coverage_fraction):
        """Apply new fixed quotas on the same atlas at an explicit reset boundary."""
        count = len(self.fixed_domain)
        total = quotas(count, weights).tolist()
        eligible = count - total[0]
        n_coverage = min(eligible, round(count * coverage_fraction))
        covered = quotas(n_coverage, (0, *weights[1:])).tolist() if n_coverage else [0] * 8
        self.coverage_level.fill_(-1)
        offset = 0
        for domain, (n, nc) in enumerate(zip(total, covered)):
            self.fixed_domain[offset : offset + n] = domain
            self.coverage_level[offset : offset + min(n, nc)] = (
                torch.arange(min(n, nc), device=self.fixed_domain.device) % 10
            )
            offset += n

    def reassign_domains(self, weights, coverage_fraction):
        before = self.fixed_domain.clone()
        self.assign_domains(weights, coverage_fraction)
        changed = before != self.fixed_domain
        self.mastery.masked_fill_(changed, 0)
        self.terrain_level.masked_fill_(changed, 0)
        self.flat_levels.masked_fill_(changed[:, None], 0)
        self.terrain_type.copy_((self.fixed_domain - 1).clamp_min(0))
        self.episode_ended.fill_(False)
        self.clear_episodes(torch.arange(len(changed), device=changed.device))

    def observe(self, state, goal, fallen, heading=0.0, traversal=None):
        from unisupermimic.rl.algos.supermimic.geometry import matvec

        v = matvec(state[1].transpose(-1, -2), state[2])
        w = matvec(state[1].transpose(-1, -2), state[3])
        command = goal[:, :3]
        speed = command[:, :2].norm(dim=-1)
        actual = (v[:, :2] * (command[:, :2] / speed[:, None].clamp_min(1e-6))).sum(-1)
        self.expected += speed * 0.02
        self.progress += actual * 0.02
        self.expected_yaw += command[:, 2].abs() * 0.02
        self.yaw_progress += w[:, 2] * command[:, 2].sign() * 0.02
        self.error += (torch.cat((v[:, :2], w[:, 2:3]), -1) - command).square().sum(-1)
        self.tilt += (1 - state[1][:, 2, 2]).clamp_min(0)
        self.steps += 1
        prior_fall = self.episode_fall.clone()
        self.episode_fall |= fallen
        axis = command.abs().argmax(-1)
        signed = 1 + axis * 2 + (command.gather(1, axis[:, None])[:, 0] < 0).long()
        signed[command.norm(dim=-1) < 0.05] = 0
        self.skill.copy_(torch.where(self.fixed_domain == 0, signed, self.fixed_domain + 6))
        ids = torch.arange(len(goal), device=goal.device)
        self.skill_expected[ids, signed] += torch.stack((speed, command[:, 2].abs()), -1) * 0.02
        self.skill_progress[ids, signed] += (
            torch.stack((actual, w[:, 2] * command[:, 2].sign()), -1) * 0.02
        )
        self.skill_energy[ids, signed] += torch.stack(
            (v[:, :2].square().sum(-1), w[:, 2].square()), -1
        )
        self.skill_steps[ids, signed] += 1
        self.skill_episode_level[ids, signed] = torch.maximum(
            self.skill_episode_level[ids, signed], self.terrain_level
        )
        slope = (self.fixed_domain == 2) | (self.fixed_domain == 3)
        forward = (
            (command[:, 0] > 0.08) & (command[:, 1].abs() < 0.08) & (command[:, 2].abs() < 0.15)
        )
        self.forward_only &= ~slope | forward
        self.in_channel &= (state[0][:, 1] - self.start_xy[:, 1]).abs() < self.cfg.lane_half_width
        delta = state[0][:, :2] - self.previous_xy
        self.progress += torch.where(slope, delta[:, 0] - actual * 0.02, 0)
        self.expected += torch.where(slope, (command[:, 0].clamp_min(0) - speed) * 0.02, 0)
        self.previous_xy.copy_(state[0][:, :2])
        if traversal is not None:
            along = ((state[0][:, :2] - self.start_xy) * traversal[:, 5:7]).sum(-1)
            safe = (
                self.in_channel
                & forward
                & (~prior_fall)
                & (~fallen)
                & ((state[0][:, 1] - self.start_xy[:, 1]).abs() < self.cfg.lane_half_width)
                & (state[1][:, 2, 2] > 0.65)
            )
            self.max_safe_progress.copy_(
                torch.maximum(self.max_safe_progress, torch.where(safe, along, 0))
            )
            domain = self.fixed_domain
            margin = state[0].new_tensor(self.cfg.crossing_margins)[(domain - 4).clamp(0, 3)]
            beyond = ((state[0][:, :2] - traversal[:, 3:5]) * traversal[:, 5:7]).sum(-1) >= margin
            height = (
                (state[0][:, 2] - traversal[:, 2]).abs() <= self.cfg.landing_height_tolerance
            ) & (state[0][:, 2] - traversal[:, 7] >= 0.22)
            self.curriculum_crossed_obstacle |= (domain >= 4) & safe & beyond & height

    def _mastery(self, ids, *, track_flat=True):
        valid = (
            self.episode_ended[ids]
            & (self.steps[ids] >= self.cfg.min_steps)
            & (self.coverage_level[ids] < 0)
        )
        ids = ids[valid & ((self.fixed_domain[ids] != 0) | track_flat)]
        if not len(ids):
            return
        domain = self.fixed_domain[ids]
        ex = self.expected[ids]
        pr = self.progress[ids]
        up = (
            (ex >= self.cfg.min_translation)
            & (pr >= self.cfg.terrain_promote * ex)
            & ~self.episode_fall[ids]
        )
        down = (ex >= self.cfg.min_translation) & (pr < self.cfg.terrain_demote * ex)
        eligible = torch.where(
            (domain == 2) | (domain == 3),
            self.forward_only[ids] & self.in_channel[ids],
            torch.ones_like(up),
        )
        frontier = self.terrain_level[ids] >= self.mastery[ids]
        up = (
            torch.where(domain >= 4, self.curriculum_crossed_obstacle[ids], up)
            & eligible
            & frontier
        )
        down = (
            torch.where(
                domain >= 4,
                self.max_safe_progress[ids] < 0.20 * torch.where(domain >= 6, 2.0, 0.8),
                down,
            )
            & eligible
            & frontier
            & ~up
        )
        expected, progress = self.skill_expected[ids], self.skill_progress[ids]
        components = expected >= expected.new_tensor(
            [self.cfg.min_translation, self.cfg.min_rotation]
        )
        enough = components.any(-1) & (self.skill_steps[ids] >= self.cfg.min_steps)
        fu = (
            (
                (
                    progress
                    >= expected
                    * expected.new_tensor([self.cfg.flat_promote, self.cfg.terrain_promote])
                )
                | ~components
            ).all(-1)
            & enough
            & ~self.episode_fall[ids, None]
        )
        fd = (
            (
                (
                    progress
                    < expected
                    * expected.new_tensor([self.cfg.flat_demote, self.cfg.terrain_demote])
                )
                & components
            ).any(-1)
            & enough
            & ~fu
        )
        ss = self.skill_steps[ids, 0]
        rms = (self.skill_energy[ids, 0] / ss[:, None].clamp_min(1)).sqrt()
        fu[:, 0] = (
            (ss * 0.02 >= self.cfg.stop_seconds)
            & ~self.episode_fall[ids]
            & (rms[:, 0] < self.cfg.stop_linear_rms)
            & (rms[:, 1] < self.cfg.stop_angular_rms)
        )
        fd[:, 0] = (ss * 0.02 >= self.cfg.stop_seconds) & (
            (rms[:, 0] > 2 * self.cfg.stop_linear_rms) | (rms[:, 1] > 2 * self.cfg.stop_angular_rms)
        )
        flat = domain == 0
        old_flat = self.flat_levels[ids].clone()
        at_frontier = self.skill_episode_level[ids] >= old_flat
        new_flat = (old_flat + (fu & at_frontier).long() - (fd & at_frontier).long()).clamp(0, 9)
        self.flat_levels[ids] = torch.where(flat[:, None], new_flat, old_flat)
        delta = new_flat - old_flat
        up = torch.where(flat, (delta > 0).any(-1), up)
        down = torch.where(flat, (delta < 0).any(-1) & ~up, down)
        previous = self.mastery[ids].clone()
        new_level = (previous + up.long() - down.long()).clamp(0, 9)
        self.mastery[ids] = torch.where(flat, self.flat_levels[ids].amax(-1), new_level)
        up &= torch.where(flat, (delta > 0).any(-1), new_level > previous)
        down &= torch.where(flat, (delta < 0).any(-1), new_level < previous)
        for tensor, value in [
            (self.promotions, up.long()),
            (self.demotions, down.long()),
            (self.holds, (~(up | down)).long()),
            (self.distance_sum, pr),
            (self.expected_sum, ex),
        ]:
            tensor.scatter_add_(0, domain, value)
        # Consume once even if called before reset again.
        self.episode_ended[ids] = False

    def clear_episodes(self, ids):
        for name in (
            "steps",
            "expected",
            "progress",
            "yaw_progress",
            "expected_yaw",
            "error",
            "tilt",
            "episode_fall",
            "episode_ended",
            "traversal_success",
            "curriculum_crossed_obstacle",
            "max_safe_progress",
            "skill_expected",
            "skill_progress",
            "skill_energy",
            "skill_steps",
        ):
            getattr(self, name)[ids] = 0
        self.forward_only[ids] = True
        self.in_channel[ids] = True
        self.skill_episode_level[ids] = -1

    def reset(self, ids, rng, *, track_flat=True):
        if self.enabled:
            self._mastery(ids, track_flat=track_flat)
            ceiling = self.mastery[ids]
            covered = self.coverage_level[ids] >= 0
            review = (
                torch.rand(len(ids), device=ids.device, generator=self.rng)
                < self.cfg.review_fraction
            )
            lower = (
                torch.rand(len(ids), device=ids.device, generator=self.rng) * (ceiling + 1)
            ).long()
            self.terrain_level[ids] = torch.where(
                covered, self.coverage_level[ids], torch.where(review, lower, ceiling)
            )
            self.terrain_type[ids] = (self.fixed_domain[ids] - 1).clamp_min(0)
        self.clear_episodes(ids)
        origins = self.origins[self.terrain_level[ids], self.terrain_type[ids]].clone()
        flat = self.fixed_domain[ids] == 0
        origins[flat] = self.origins[0, 0]
        origins[flat, 0] += 3.4
        self.start_xy[ids] = origins[:, :2]
        self.previous_xy[ids] = origins[:, :2]
        return origins

    def start_scope_mask(self):
        return torch.ones(8, 10, dtype=torch.bool, device=self.mastery.device)

    def state_dict(self):
        return dict(
            schema=CURRICULUM_SCHEMA,
            configuration=asdict(self.cfg),
            tensors={k: v.clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)},
            statistics=self.statistics,
            rng=self.rng.get_state(),
            phase=self.phase,
        )

    def load_state_dict(self, values):
        if values.get("schema") != CURRICULUM_SCHEMA or values.get("configuration") != asdict(
            self.cfg
        ):
            raise ValueError("Incompatible curriculum: explicit distance migration required")
        keys = {k for k, v in vars(self).items() if isinstance(v, torch.Tensor)}
        if values["tensors"].keys() != keys:
            raise ValueError("Curriculum tensor schema differs")
        for k, v in values["tensors"].items():
            if v.shape != getattr(self, k).shape or v.dtype != getattr(self, k).dtype:
                raise ValueError(f"Curriculum tensor differs: {k}")
            getattr(self, k).copy_(v)
        self.statistics = {k: v.to(self.mastery.device) for k, v in values["statistics"].items()}
        self.rng.set_state(values["rng"].cpu())
        self.phase = values["phase"]

    def summary(self):
        flat = self.fixed_domain == 0
        if flat.any():
            self.skill_levels[:7] = self.flat_levels[flat].amax(0)
        for domain in range(1, 8):
            ids = (self.fixed_domain == domain) & (self.coverage_level < 0)
            self.skill_levels[domain + 6] = self.mastery[ids].amax() if ids.any() else 0
        scopes = self.fixed_domain * 10 + self.terrain_level
        return dict(
            schema=CURRICULUM_SCHEMA,
            skill_levels=(self.skill_levels + 1).cpu().tolist(),
            domain_counts=torch.bincount(self.fixed_domain, minlength=8).cpu().tolist(),
            domain_level=torch.bincount(scopes, minlength=80).reshape(8, 10).cpu().tolist(),
            coverage_environments=int((self.coverage_level >= 0).sum()),
            promotions=self.promotions.cpu().tolist(),
            demotions=self.demotions.cpu().tolist(),
            holds=self.holds.cpu().tolist(),
            distance_sum=self.distance_sum.cpu().tolist(),
            expected_sum=self.expected_sum.cpu().tolist(),
        )
