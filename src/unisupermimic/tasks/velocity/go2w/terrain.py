"""Reuse the Uni atlas/approach geometry with Go2W eight-level difficulty mappings."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace

from unisupermimic.tasks.velocity.go2.curriculum import TerrainCurriculum
from unisupermimic.tasks.velocity.go2.terrain import ApproachTerrainCfg, terrain_profile


@dataclass(kw_only=True)
class Go2WApproachTerrainCfg(ApproachTerrainCfg):
    generator_version: str = "go2w_eight_levels_v2"

    def function(self, difficulty, rng):
        level = min(self.atlas_max_level, max(0, int(difficulty * (self.atlas_max_level + 1))))
        if level == 0:
            return super().function(difficulty, rng)
        cfg = deepcopy(self.delegate)
        fraction = (level - 1) / max(1, self.atlas_max_level - 1)
        if hasattr(cfg, "noise_range"):
            cfg.noise_range = (0.0, 0.01 + 0.02 * fraction)
            cfg.noise_step = 0.005
        elif hasattr(cfg, "slope_range"):
            cfg.slope_range = (0.05 + 0.15 * fraction,) * 2
        elif hasattr(cfg, "step_height_range"):
            cfg.step_height_range = (0.03 * level,) * 2
            cfg.step_width = 0.40
        return ApproachTerrainCfg.function(
            replace(self, delegate=cfg, discrete_levels=False), difficulty, rng
        )


def mixed_terrain(seed=2026):
    terrain = terrain_profile("mixed", seed=seed, levels=True, max_level=7)
    terrain.generator.sub_terrains = {
        name: Go2WApproachTerrainCfg(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})
        for name, cfg in terrain.generator.sub_terrains.items()
    }
    return terrain


def terrain_contract(cfg):
    """Serialize only effective Go2W generator fields, independent of Go2 tuning."""
    from dataclasses import asdict

    values = asdict(cfg)
    for sub in values["generator"]["sub_terrains"].values():
        # The Go2W override always uses .03*level, never the optional Go2 knob.
        sub.pop("stair_step_increment", None)
    return values


class Go2WTerrainCurriculum(TerrainCurriculum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch

        for name in ("transitions", "valid_transitions", "falls"):
            self.statistics["real_" + name] = torch.zeros_like(self.statistics["episodes"])

    def configure_persistent_coverage(self):
        """Fixed roles inside existing physical domains, scaled from 4096 envs.

        Role and level are checkpointed tensors. Only reset changes the physical
        location; coverage episodes never promote or demote skill mastery.
        """
        import torch

        if self.fixed_domain is None:
            raise ValueError("Persistent coverage requires fixed domains")
        self.coverage_level = torch.full_like(self.mastery, -1)
        weights = torch.tensor([24, 24, 24, 20, 16, 12, 8], device=self.mastery.device)
        for domain, quota in ((1, 128), (2, 64), (3, 64), (4, 128), (5, 128)):
            ids = (self.fixed_domain == domain).nonzero().flatten()
            total = round(len(self.mastery) * quota / 4096)
            if total > len(ids):
                raise ValueError("Coverage quota exceeds physical domain capacity")
            exact = weights.float() * total / weights.sum()
            counts = exact.long()
            remainder = total - int(counts.sum())
            counts[torch.argsort(exact - counts, descending=True, stable=True)[:remainder]] += 1
            levels = torch.arange(1, 8, device=ids.device).repeat_interleave(counts)
            self.coverage_level[ids[:total]] = levels

    def reset(self, ids, rng):
        if not hasattr(self, "coverage_level"):
            return super().reset(ids, rng)
        covered = ids[self.coverage_level[ids] >= 1]
        crossing = self.crossing_episode[covered].clone()
        # The parent qualifies skill updates using crossing_episode. Coverage
        # data remain in real_* statistics, independently of certified success.
        self.crossing_episode[covered] = False
        try:
            origins = super().reset(ids, rng)
        finally:
            self.crossing_episode[covered] = crossing
        selected = self.coverage_level[ids] >= 1
        if self.enabled and self.phase in ("wm", "skill"):
            self.terrain_level[covered] = self.coverage_level[covered]
            self.probe[covered] = False
            self.probe_offset[covered] = 0
            origins[selected] = self.origins[
                self.terrain_level[covered], self.terrain_type[covered]
            ]
            self.start_xy[covered] = origins[selected, :2]
        return origins

    def summary(self):
        out = super().summary()
        if hasattr(self, "coverage_level"):
            import torch

            selected = self.coverage_level >= 1
            scopes = self.fixed_domain[selected] * 8 + self.coverage_level[selected]
            out["persistent_coverage_domain_level"] = (
                torch.bincount(scopes, minlength=48).reshape(6, 8).cpu().tolist()
            )
        return out

    def load_state_dict(self, values):
        # Learner.load already validates the task contract, with command-only
        # migration requiring allow_command_distribution_change=True. Preserve
        # the newly configured quotas instead of restoring old A/B/C assignments.
        domains = self.fixed_domain.clone()
        if hasattr(self, "coverage_level"):
            import torch

            saved = values["tensors"].get("coverage_level")
            if saved is None or not torch.equal(saved.to(self.coverage_level), self.coverage_level):
                raise ValueError("Persistent coverage roles changed on resume")
        if values["phase"] not in ("wm", "skill"):
            raise ValueError("Unsupported Go2W curriculum phase")
        self.phase = values["phase"]
        super().load_state_dict(values)
        changed = self.fixed_domain != domains
        self.fixed_domain.copy_(domains)
        for name in ("mastery", "terrain_level", "success_streak", "failure_streak"):
            getattr(self, name)[changed] = 0


@dataclass(kw_only=True)
class TraversalTerrainCfg(ApproachTerrainCfg):
    """One approach, one signed obstacle direction and an extended landing platform."""

    kind: str = "single_step_up"
    generator_version: str = "go2w_unified_traversal"

    def function(self, difficulty, rng):
        import numpy as np
        from unilab.terrains.heightfield_terrains import _make_terrain_output

        from .curriculum import LEVEL_COUNT, SINGLE_HEIGHTS, STAIR_HEIGHTS

        level = min(LEVEL_COUNT - 1, max(0, int(difficulty * LEVEL_COUNT)))
        nx, ny = (int(s / self.horizontal_scale) for s in self.size)
        x = np.arange(nx) * self.horizontal_scale
        height = SINGLE_HEIGHTS[level] if self.kind.startswith("single") else STAIR_HEIGHTS[level]
        if self.kind == "rough":
            profile = rng.uniform(-0.005 * (level + 1), 0.005 * (level + 1), size=(nx, ny))
            profile[:24] = 0
            if level == 0:
                profile[:] = 0
        elif self.kind.startswith("slope"):
            profile = np.clip(x - 1.4, 0, 3.0) * (0.03 + 0.015 * level)
            profile = np.broadcast_to(profile[:, None], (nx, ny)).copy()
        else:
            steps = (
                (x >= 1.4).astype(float)
                if self.kind.startswith("single")
                else np.clip(np.floor((x - 1.4) / 0.4) + 1, 0, 4)
            )
            profile = np.broadcast_to((steps * height)[:, None], (nx, ny)).copy()
        if self.kind.endswith("down"):
            profile = profile.max() - profile
        # Stair collision uses exact vertical box faces; no triangulated hfield ramp.
        if self.kind.startswith(("single", "multi")):
            profile[:] = 0
        noise = np.rint(profile / self.vertical_scale).astype(np.int16)
        return _make_terrain_output(
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=1.0,
            origin=np.array([0.6, self.size[1] / 2, float(profile[12, ny // 2])]),
            flat_patch_sampling=None,
            rng=rng,
        )


def unified_terrain(seed=2026):
    from unilab.terrains import TerrainGeneratorCfg
    from unilab.terrains.heightfield_terrains import HfFlatTerrainCfg
    from unisim.scene import TerrainSceneCfg

    from .curriculum import DOMAINS, LEVEL_COUNT

    return TerrainSceneCfg(
        generator=TerrainGeneratorCfg(
            seed=seed,
            size=(8.0, 8.0),
            num_rows=LEVEL_COUNT,
            num_cols=7,
            curriculum=True,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            border_width=2.0,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                name: TraversalTerrainCfg(
                    kind=name,
                    delegate=HfFlatTerrainCfg(),
                    size=(8.0, 8.0),
                    proportion=1.0,
                    atlas_max_level=9,
                    profile_max_level=9,
                    require_low_border=False,
                )
                for name in DOMAINS[1:]
            },
        ),
        hfield_name="go2_heightfield",
        geom_name="go2_terrain",
    )
