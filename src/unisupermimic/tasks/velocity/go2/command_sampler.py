"""Privileged collector metadata only. Goals are body-frame (m/s,m/s,rad/s).

No environment loop, learned encoder, or actor input change. Speed level is an
independent, explicit evaluation-controlled setting, never terrain mastery.
"""

from dataclasses import dataclass

import torch

DOMAINS = ("flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down")
MODES = ("standing", "forward", "backward", "lateral", "turning", "mixed")


@dataclass
class VelocitySamplerCfg:
    domain_weights: tuple = (0.35, 0.20, 0.10, 0.10, 0.125, 0.125)
    flat_mode_weights: tuple = (0.08, 0.20, 0.18, 0.18, 0.16, 0.20)
    flat_speed_level: int = 0
    flat_hold_range: tuple = (2.0, 5.0)
    terrain_hold_seconds: float = 10.02
    seed: int = 2026
    # Each entry is ((vx_lo,vx_hi),(vy_lo,vy_hi),(wz_lo,wz_hi)).
    flat_velocity_levels: tuple = (
        ((-0.25, 0.60), (-0.15, 0.15), (-0.35, 0.35)),
        ((-0.40, 0.80), (-0.30, 0.30), (-0.65, 0.65)),
        ((-0.50, 1.0), (-0.40, 0.40), (-0.90, 0.90)),
    )
    terrain_velocity_ranges: tuple = (
        ((-0.20, 0.70), (-0.15, 0.15), (-0.35, 0.35)),
        ((0.10, 0.70), (-0.10, 0.10), (-0.25, 0.25)),
        ((0.10, 0.70), (-0.10, 0.10), (-0.25, 0.25)),
        ((0.15, 0.60), (-0.06, 0.06), (-0.15, 0.15)),
        ((0.15, 0.60), (-0.06, 0.06), (-0.15, 0.15)),
    )


def command_modes(goal):
    """Exact pure modes; values are commands, never measured robot motion."""
    x, y, z = goal.unbind(-1)
    result = torch.full_like(x, 5, dtype=torch.long)
    result = torch.where((y == 0) & (z == 0) & (x > 0), 1, result)
    result = torch.where((y == 0) & (z == 0) & (x < 0), 2, result)
    result = torch.where((x == 0) & (z == 0) & (y != 0), 3, result)
    result = torch.where((x == 0) & (y == 0) & (z != 0), 4, result)
    return torch.where((goal == 0).all(-1), 0, result)


def fixed_domains(n, weights, device="cpu"):
    """Exact fixed env quotas; keep legacy atlas type for every terrain env.

    The legacy atlas assigns type=env_id%5. Allocate within each type, leaving
    unused envs flat. This preserves the existing per-type mastery on migration.
    """
    p = torch.as_tensor(weights, dtype=torch.float64)
    if (
        p.shape != (6,)
        or not torch.isfinite(p).all()
        or (p < 0).any()
        or not torch.isclose(p.sum(), p.new_tensor(1.0))
    ):
        raise ValueError("Six finite nonnegative domain weights must sum to one")
    count = (p * n).floor().long()
    count[torch.argsort(p * n - count, descending=True)[: n - int(count.sum())]] += 1
    ids = torch.arange(n, device=device)
    domain = torch.zeros(n, dtype=torch.long, device=device)
    # Five domain-wide vector operations, never a loop over environments.
    for t in range(5):
        selected = ids[ids % 5 == t]
        if count[t + 1] > len(selected):
            raise ValueError("Terrain domain quota exceeds legacy atlas type capacity")
        domain[selected[: int(count[t + 1])]] = t + 1
    return domain


class TerrainAwareVelocitySampler:
    def __init__(self, cfg, device="cpu"):
        from unilab.managers.rng import ManagerRng

        self.cfg = cfg
        self.rng = ManagerRng.seeded(cfg.seed, device=device)
        self.speed_level = cfg.flat_speed_level
        self.set_speed_level(self.speed_level)
        w = torch.tensor(
            [
                cfg.flat_mode_weights,
                (0.05, 0.55, 0.15, 0.075, 0.075, 0.10),
                (0.05, 0.75, 0.0, 0.075, 0.075, 0.05),
                (0.05, 0.75, 0.0, 0.075, 0.075, 0.05),
                (0.05, 0.80, 0.0, 0.075, 0.075, 0.0),
                (0.05, 0.80, 0.0, 0.075, 0.075, 0.0),
            ],
            device=device,
        )
        if (
            not torch.isfinite(w).all()
            or (w < 0).any()
            or not torch.allclose(w.sum(-1), torch.ones(6, device=device))
        ):
            raise ValueError("Invalid command mode probabilities")
        if (
            not 0 < cfg.flat_hold_range[0] <= cfg.flat_hold_range[1]
            or cfg.terrain_hold_seconds <= 0
        ):
            raise ValueError("Invalid command hold times")
        self.cdf = w.cumsum(-1)
        self.flat_ranges = w.new_tensor(cfg.flat_velocity_levels)
        terrain_ranges = w.new_tensor(cfg.terrain_velocity_ranges)
        if self.flat_ranges.shape != (3, 3, 2) or terrain_ranges.shape != (5, 3, 2):
            raise ValueError("Expected three flat levels and five terrain velocity ranges")
        ranges = torch.cat((self.flat_ranges[-1:], terrain_ranges), 0)
        if not torch.isfinite(ranges).all() or (ranges[..., 0] > ranges[..., 1]).any():
            raise ValueError("Invalid velocity range endpoints")
        self.lows, self.highs = ranges[..., 0], ranges[..., 1]

    def set_speed_level(self, level):
        if level not in (0, 1, 2):
            raise ValueError("Independent flat speed level must be 0,1,2")
        self.speed_level = int(level)

    def sample(self, terrain_domain, terrain_level, rng=None):
        rng = self.rng if rng is None else rng
        d = terrain_domain
        n = len(d)
        mode = (rng.random((n, 1)) > self.cdf[d]).sum(-1).clamp_max(5)
        lo, hi = self.lows[d].clone(), self.highs[d].clone()
        flat_lo, flat_hi = self.flat_ranges[self.speed_level].unbind(-1)
        lo = torch.where((d == 0)[:, None], flat_lo, lo)
        hi = torch.where((d == 0)[:, None], flat_hi, hi)
        # High stairs retain pure-forward coverage but smaller general corrections.
        correction = torch.where((d >= 4) & (terrain_level >= 3), 0.65, 1.0)
        lo[:, 1:] *= correction[:, None]
        hi[:, 1:] *= correction[:, None]
        unit = rng.random((n, 3))
        mixed = lo + unit * (hi - lo)
        # Equal-width speed bins include low and medium magnitudes with equal mass.
        magnitude = rng.random((n, 3))
        sign = torch.where(rng.random((n, 3)) < 0.5, -1.0, 1.0)
        forward = torch.zeros_like(mixed)
        minimum = torch.where(d == 0, 0.15, lo[:, 0].clamp_min(0.10))
        forward[:, 0] = minimum + magnitude[:, 0] * (hi[:, 0] - minimum)
        backward = torch.zeros_like(mixed)
        backward[:, 0] = -(0.10 + magnitude[:, 0] * ((-lo[:, 0]).clamp_min(0.10) - 0.10))
        lateral = torch.zeros_like(mixed)
        # Pure lateral excludes exact zero; low nonzero commands remain represented.
        lateral[:, 1] = sign[:, 1] * (0.01 + magnitude[:, 1] * (hi[:, 1] - 0.01))
        turning = torch.zeros_like(mixed)
        turning[:, 2] = sign[:, 2] * (0.02 + magnitude[:, 2] * (hi[:, 2] - 0.02))
        candidates = torch.stack(
            (torch.zeros_like(mixed), forward, backward, lateral, turning, mixed), 1
        )
        goal = candidates[torch.arange(n, device=d.device), mode]
        crossing = (d != 0) & (mode == 1)
        hold = self.cfg.flat_hold_range[0] + rng.random((n,)) * (
            self.cfg.flat_hold_range[1] - self.cfg.flat_hold_range[0]
        )
        hold = torch.where(d == 0, hold, self.cfg.terrain_hold_seconds)
        return goal, mode, crossing, hold

    def state_dict(self):
        from dataclasses import asdict

        return dict(config=asdict(self.cfg), speed_level=self.speed_level, rng=self.rng.get_state())

    def load_state_dict(self, state):
        # List/tuple conversion through YAML must not change equality semantics.
        import json
        from dataclasses import asdict

        previous = asdict(VelocitySamplerCfg()) | state["config"]
        if json.dumps(previous, sort_keys=True) != json.dumps(asdict(self.cfg), sort_keys=True):
            raise ValueError("Sampler config changed; explicit distribution migration required")
        self.set_speed_level(state["speed_level"])
        self.rng.set_state(state["rng"].cpu())


def validate_distribution_migration(old, new):
    """Permit only explicitly requested command/domain distribution changes.

    Physics, reward, state, network and backend gates remain strict. The sole new
    truncation resets flat-domain robots before reaching non-flat atlas rows.
    """
    import copy

    if old is None or new is None:
        raise ValueError("Missing task contract")
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    allowed = (
        "sampling_mode",
        "sampler",
        "resampling_time_range",
        "stage",
        "terrain_flat_probability",
    )
    for key in allowed:
        a["commands"].pop(key, None)
        b["commands"].pop(key, None)
    boundary = b["terminations"].pop("flat_domain_boundary", None)
    if boundary is not None and (
        boundary["func"] != "unisupermimic.tasks.velocity.go2.manager_terms.flat_domain_boundary"
        or not boundary["time_out"]
    ):
        raise ValueError("Unexpected flat-domain termination semantics")
    a["terminations"].pop("flat_domain_boundary", None)
    if a != b:
        changed = [key for key in a if a[key] != b.get(key)]
        raise ValueError(f"Command migration changed protected task semantics: {changed}")
