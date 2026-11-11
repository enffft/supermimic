"""Go2-only terrain profiles. Upstream presets are copied, never mutated."""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np
from unilab.terrains import SubTerrainCfg, TerrainGeneratorCfg
from unilab.terrains.config import ROUGH_TERRAINS_CFG, STAIRS_TERRAINS_CFG
from unisim.scene import TerrainSceneCfg

from .contract import MODEL


@dataclass(kw_only=True)
class ApproachTerrainCfg(SubTerrainCfg):
    """Delegate generation, rotate actual heights, and choose a checked low border.

    Quarter turns rotate heightfield geometry in world XY. Robot heading is an
    independent reset parameter; pyramid profiles themselves have fourfold symmetry.
    """

    delegate: SubTerrainCfg
    quarter_turns: int = 0
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    discrete_levels: bool = False
    require_low_border: bool = True
    entry_quarter_turns: int = 0
    atlas_max_level: int = 4
    profile_max_level: int = 4

    def function(self, difficulty, rng):
        cfg = deepcopy(self.delegate)
        cfg.size = self.size
        cfg.horizontal_scale = self.horizontal_scale
        cfg.vertical_scale = self.vertical_scale
        if self.discrete_levels:
            from unilab.terrains.heightfield_terrains import HfFlatTerrainCfg

            level = min(
                self.profile_max_level, max(0, int(difficulty * (self.atlas_max_level + 1)))
            )
            if level == 0:
                cfg = HfFlatTerrainCfg(
                    size=self.size,
                    horizontal_scale=self.horizontal_scale,
                    vertical_scale=self.vertical_scale,
                )
            elif hasattr(cfg, "noise_range"):
                cfg.noise_range = (0.0, 0.01 * level)
                cfg.noise_step = 0.005
            elif hasattr(cfg, "slope_range"):
                cfg.slope_range = (0.05 * level, 0.05 * level)
            elif hasattr(cfg, "step_height_range"):
                cfg.step_height_range = (0.02 * level, 0.02 * level)
        output = cfg.function(difficulty, rng)
        hf = output.heightfield
        hf.noise = np.rot90(hf.noise, self.quarter_turns).copy()
        heights = hf.physical_heights_xy()
        # Entire 0.8 x 0.5 m footprint stays in a 1.2 m wide approach border.
        entry = np.array((0.60, self.size[1] / 2))
        for _ in range(self.entry_quarter_turns % 4):
            entry = np.array((self.size[0] - entry[1], entry[0]))
        cell = np.rint(entry / self.horizontal_scale).astype(int)
        footprint = (0.40, 0.25) if self.entry_quarter_turns % 2 == 0 else (0.25, 0.40)
        hx, hy = np.ceil(np.array(footprint) / self.horizontal_scale).astype(int)
        if (
            cell[0] - hx < 0
            or cell[0] + hx >= heights.shape[0]
            or cell[1] - hy < 0
            or cell[1] + hy >= heights.shape[1]
        ):
            raise ValueError("Go2 footprint is outside the terrain tile")
        patch = heights[cell[0] - hx : cell[0] + hx + 1, cell[1] - hy : cell[1] + hy + 1]
        if np.ptp(patch) > 0.005 + 1e-8 or (
            self.require_low_border and patch.max() > heights.min() + 0.005 + 1e-8
        ):
            raise ValueError(
                "Go2 approach spawn is not a flat low border; refusing center fallback"
            )
        output.origin = np.array(
            (cell[0] * self.horizontal_scale, cell[1] * self.horizontal_scale, patch.max())
        )
        output.flat_patches = None
        return output


def terrain_profile(
    name, *, seed=2026, quarter_turns=0, levels=False, entry_quarter_turns=0, max_level=4
):
    if not 0 <= max_level <= 7:
        raise ValueError("Audited terrain level range is 0..7")
    if name == "flat":
        # Preserve the exact existing flat XML, including its plane and contact settings.
        return None
    if name not in ("rough", "slope", "slope_down", "stairs", "stairs_down", "mixed"):
        raise ValueError(f"unknown Go2 terrain profile {name!r}")
    if isinstance(quarter_turns, bool) or quarter_turns not in (0, 1, 2, 3):
        raise ValueError("terrain quarter_turns must be 0, 1, 2, or 3")
    rough = deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["random_rough"])
    rough.noise_range = (0.01, 0.03)
    rough.border_width = 1.2
    slope = deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_slope"])
    slope.slope_range = (0.05, 0.20)
    slope.border_width = 1.2
    stairs = deepcopy(STAIRS_TERRAINS_CFG.sub_terrains["easy_stairs"])
    stairs.step_height_range = (0.02, 0.05)
    stairs.step_width = 0.4
    stairs.border_width = 1.2
    slope_down = deepcopy(slope)
    slope_down.inverted = True
    stairs_down = deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"])
    stairs_down.step_height_range = stairs.step_height_range
    stairs_down.step_width = stairs.step_width
    stairs_down.border_width = stairs.border_width
    entries = {
        "rough": rough,
        "slope": slope,
        "slope_down": slope_down,
        "stairs": stairs,
        "stairs_down": stairs_down,
    }
    selected = (
        (entries if levels else {k: entries[k] for k in ("rough", "slope", "stairs")})
        if name == "mixed"
        else {name: entries[name]}
    )
    subs = {
        key: ApproachTerrainCfg(
            delegate=value,
            quarter_turns=quarter_turns,
            size=(8.0, 8.0),
            proportion=1.0,
            discrete_levels=levels,
            require_low_border=not key.endswith("_down"),
            entry_quarter_turns=entry_quarter_turns,
            atlas_max_level=max_level,
            profile_max_level=max_level if "stairs" in key else min(max_level, 4),
        )
        for key, value in selected.items()
    }
    return TerrainSceneCfg(
        generator=TerrainGeneratorCfg(
            seed=seed,
            size=(8.0, 8.0),
            num_rows=max_level + 1 if levels else 1,
            num_cols=len(subs),
            curriculum=True,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            border_width=2.0,
            difficulty_range=(0.0, 1.0) if levels else (0.0, 0.3),
            sub_terrains=subs,
        ),
        hfield_name="go2_heightfield",
        geom_name="go2_terrain",
    )


def terrain_contract(cfg):
    """Keep the original Level0..4 serialization/fingerprint unchanged."""
    values = asdict(cfg)
    for sub in values["generator"]["sub_terrains"].values():
        for key in ("atlas_max_level", "profile_max_level"):
            if sub.get(key) == 4:
                sub.pop(key)
    return values


def apply_profile(cfg, name, *, backend="mujoco", seed=2026, quarter_turns=0, heading=0.0):
    if not math.isfinite(heading):
        raise ValueError("spawn heading must be finite radians")
    profile = terrain_profile(name, seed=seed, quarter_turns=quarter_turns)
    if profile is not None and backend != "mujoco":
        raise NotImplementedError(
            "Generated terrain is supported only by the audited MuJoCo backend; mjwarp rejects heightfields"
        )
    cfg.scene.terrain = profile
    cfg.scene.model_file = str(MODEL if profile is None else MODEL.with_name("terrain_robot.xml"))
    cfg.commands["velocity"].spawn_heading = heading
    return cfg


def bind_perception(env, offsets, *, alignment="yaw"):
    """Separate extension interface; no height samples are appended to Actor5x42."""
    if env.backend.backend_type != "mujoco":
        raise NotImplementedError("mjwarp height scanner is unsupported by UniSim 2.7.1")
    if env.cfg.scene.terrain is None:
        raise ValueError("height scanner requires a generated terrain profile")
    return env.backend.bind_height_scanner(
        hfield_geom_id=env.backend.get_geom_id("go2_terrain"),
        offsets=np.asarray(offsets, dtype=np.float64),
        frame_body_id=env.backend.get_body_id("robot/base"),
        alignment=alignment,
        output="height",
    )


def observation_interface(env, *, mode="blind", scanner=None):
    """Named extensible interfaces; perceptive features never alter the blind Actor."""
    if mode not in ("blind", "perceptive"):
        raise ValueError("Go2 observation mode must be blind or perceptive")
    command = env.command_manager.get_term("velocity")
    values = {
        "proprioception": command.mdp.obs(command.state(), command.a_prev, command.xi, command.f),
        "velocity_goal": command.g.clone(),
    }
    if mode == "perceptive":
        if scanner is None:
            raise ValueError("perceptive interface requires an explicitly bound height scanner")
        values["height_world"] = scanner.scan()
    return values
