"""Reuse the Uni atlas/approach geometry with conservative Go2W level ranges."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace

from unisupermimic.tasks.velocity.go2.terrain import ApproachTerrainCfg, terrain_profile
from unisupermimic.tasks.velocity.go2.curriculum import TerrainCurriculum


@dataclass(kw_only=True)
class Go2WApproachTerrainCfg(ApproachTerrainCfg):
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
            cfg.step_height_range = (0.02 + 0.03 * fraction,) * 2
            cfg.step_width = 0.40
        return ApproachTerrainCfg.function(
            replace(self, delegate=cfg, discrete_levels=False), difficulty, rng)


def mixed_terrain(seed=2026):
    terrain = terrain_profile("mixed", seed=seed, levels=True, max_level=4)
    terrain.generator.sub_terrains = {
        name: Go2WApproachTerrainCfg(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})
        for name, cfg in terrain.generator.sub_terrains.items()
    }
    return terrain


class Go2WTerrainCurriculum(TerrainCurriculum):
    def load_state_dict(self, values):
        # Learner.load already validates the task contract, with command-only
        # migration requiring allow_command_distribution_change=True. Preserve
        # the newly configured quotas instead of restoring old A/B/C assignments.
        domains = self.fixed_domain.clone()
        super().load_state_dict(values)
        changed = self.fixed_domain != domains
        self.fixed_domain.copy_(domains)
        for name in ("mastery", "terrain_level", "success_streak", "failure_streak"):
            getattr(self, name)[changed] = 0
