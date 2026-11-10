"""Complete ten-second synthetic episodes; no physics-success claims."""

import numpy as np
import torch

from unisupermimic.tasks.velocity.go2.curriculum import TerrainCurriculum


class RNG:
    def random(self, shape):
        return torch.full(shape, 0.9)


def episode(c, *, speed=0.25, actual=None, lateral=0.0, yaw=0.0, fall=False):
    n = len(c.mastery)
    v = torch.zeros(n, 3)
    v[:, 0] = speed if actual is None else actual
    goal = torch.tensor([speed, lateral, yaw]).expand(n, 3)
    w = torch.zeros(n, 3)
    w[:, 2] = yaw
    p = torch.zeros(n, 3)
    rotation = torch.eye(3).expand(n, 3, 3)
    for i in range(500):
        p[:, 0] = c.start_xy[:, 0] + v[:, 0] * (i + 1) * 0.02
        p[:, 1] = c.start_xy[:, 1]
        c.observe((p, rotation, v, w), goal, torch.full((n,), fall and i == 499), 0.0)
    c.reset(torch.arange(n), RNG())


def test_low_speed_crossing_promotes_with_rolling_evidence_and_fall_demotes():
    c = TerrainCurriculum(2, np.zeros((5, 2, 3)), enabled=True)
    for _ in range(3):
        episode(c)
    assert c.mastery.tolist() == [1, 1]
    assert c.statistics["promotions"].sum() == 2
    for _ in range(3):
        episode(c)
    assert c.mastery.tolist() == [2, 2]
    episode(c, fall=True)
    assert c.mastery.tolist() == [2, 2]
    episode(c, fall=True)
    assert c.mastery.tolist() == [1, 1]


def test_zero_rotation_stall_timeout_and_bypass_do_not_promote():
    for kw in (
        dict(speed=0.0),
        dict(speed=0.0, yaw=0.5),
        dict(actual=0.0),
        dict(actual=0.1),
        dict(lateral=0.3),
    ):
        c = TerrainCurriculum(1, np.zeros((5, 1, 3)), enabled=True)
        for _ in range(4):
            episode(c, **kw)
        assert c.mastery.item() == 0


def test_wm_coverage_is_not_skill_and_probe_is_not_success():
    c = TerrainCurriculum(4, np.zeros((5, 2, 3)), enabled=True, phase="wm", probe_probability=1.0)
    c.reset(torch.arange(4), RNG())
    assert c.terrain_level.min() > 0 and c.probe.all()
    for _ in range(4):
        episode(c)
    assert not c.mastery.any() and not c.statistics["success"].any()
    assert c.statistics["episodes"].sum() == 16
    assert c.statistics["probes"].sum() == 16


def test_stats_fixed_eval_and_serialization_are_independent():
    c = TerrainCurriculum(2, np.zeros((5, 2, 3)), enabled=True)
    episode(c)
    assert c.statistics["episodes"][:, 0].tolist() == [1.0, 1.0]
    assert c.statistics["episodes"][:, 1:].sum() == 0
    state = c.state_dict()
    other = TerrainCurriculum(2, np.zeros((5, 2, 3)), enabled=True)
    other.load_state_dict(state)
    assert other.summary() == c.summary()
    fixed = TerrainCurriculum(2, np.zeros((5, 2, 3)), phase="fixed", fixed_level=2)
    for _ in range(4):
        episode(fixed)
    assert fixed.mastery.tolist() == [2, 2] and not fixed.last_decision.any()
    assert other.summary() == c.summary()
