"""Stairs-only ceiling extension; legacy profiles and blind protocols preserved."""

import numpy as np
import torch
from unilab.managers.rng import ManagerRng

from unisupermimic.tasks.velocity.go2.command_sampler import (
    TerrainAwareVelocitySampler,
    VelocitySamplerCfg,
)
from unisupermimic.tasks.velocity.go2.curriculum import TerrainCurriculum
from unisupermimic.tasks.velocity.go2.terrain import terrain_contract, terrain_profile


def test_only_stairs_extend_nominal_heights_and_rows():
    old = terrain_profile("mixed", levels=True)
    new = terrain_profile("mixed", levels=True, max_level=7)
    assert old.generator.num_rows == 5 and new.generator.num_rows == 8
    for key, sub in new.generator.sub_terrains.items():
        assert sub.profile_max_level == (7 if "stairs" in key else 4)
        assert sub.atlas_max_level == 7
    legacy = terrain_contract(old)
    assert all(
        "atlas_max_level" not in v and "profile_max_level" not in v
        for v in legacy["generator"]["sub_terrains"].values()
    )


def test_terrain_mastery_caps_are_independent_by_type():
    c = TerrainCurriculum(
        5,
        np.zeros((8, 5, 3)),
        max_level=7,
        fixed_level=4,
        enabled=True,
        flat_probability=0,
        recycle_probability=0,
        coverage_weights=(0.2, 0.4, 0.3, 0.1, 0, 0, 0, 0),
        type_max_levels=[4, 4, 4, 7, 7],
    )
    c.fixed_domain = torch.arange(5) + 1
    for _ in range(3):
        c.steps.fill_(500)
        c.eligible_steps.fill_(500)
        c.episode_progress.fill_(2.5)
        c.commanded_progress.fill_(2.5)
        c.reset(torch.arange(5), ManagerRng.seeded(10))
    assert c.mastery.tolist() == [4, 4, 4, 5, 5]
    assert len(c.summary()["mastery_histogram"]) == 8


def test_speed_state_expands_without_promoting_new_levels():
    old = TerrainAwareVelocitySampler(VelocitySamplerCfg(stairs_omni=True))
    old.set_stairs_speed_level(0, 4, 2)
    new = TerrainAwareVelocitySampler(VelocitySamplerCfg(stairs_omni=True, terrain_level_count=8))
    new.load_state_dict(old.state_dict(), allow_distribution_change=True)
    assert new.stairs_levels[0, 4] == 2 and (new.stairs_levels[:, 5:] == 0).all()
    g, *_ = new.sample(torch.full((1000,), 4), torch.full((1000,), 7))
    assert g[:, 1].abs().max() <= 0.08 and g[:, 2].abs().max() <= 0.15
