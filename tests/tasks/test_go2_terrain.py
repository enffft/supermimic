"""Terrain ownership and unsupported-backend boundaries."""

from copy import deepcopy

import numpy as np
import pytest
from unilab.terrains.config import ROUGH_TERRAINS_CFG, STAIRS_TERRAINS_CFG

from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.contract import MODEL
from unisupermimic.tasks.velocity.go2.terrain import apply_profile, terrain_profile


def test_profile_roundtrip_restores_exact_flat_model_and_presets():
    rough = deepcopy(ROUGH_TERRAINS_CFG)
    stairs = deepcopy(STAIRS_TERRAINS_CFG)
    cfg = load_config()
    apply_profile(cfg, "rough")
    assert cfg.scene.model_file.endswith("terrain_robot.xml")
    apply_profile(cfg, "flat")
    assert cfg.scene.terrain is None
    assert cfg.scene.model_file == str(MODEL)
    assert rough == ROUGH_TERRAINS_CFG
    assert stairs == STAIRS_TERRAINS_CFG


@pytest.mark.parametrize("profile", ["rough", "slope", "stairs", "mixed"])
@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_generated_approach_is_checked_low_border(profile, rotation):
    cfg = terrain_profile(profile, quarter_turns=rotation).generator
    for sub in cfg.sub_terrains.values():
        output = sub.function(0.15, np.random.default_rng(2026))
        assert output.origin[0] == pytest.approx(0.6)
        assert output.origin[2] <= output.heightfield.physical_heights_xy().min() + 0.00500001
        assert output.flat_patches is None


def test_mjwarp_terrain_rejected_and_invalid_heading():
    with pytest.raises(NotImplementedError, match="MuJoCo"):
        apply_profile(load_config(), "stairs", backend="mjwarp")
    with pytest.raises(ValueError, match="finite"):
        apply_profile(load_config(), "flat", heading=float("nan"))
