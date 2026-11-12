"""Strict, explicit A/B/C command-distribution migration for the v2 task."""

from copy import deepcopy


def validate_stage_migration(old, new):
    if old is None or new is None or old.get("schema") != "go2w_velocity_supermimic_v2":
        raise ValueError("Go2W v2 requires a new experiment; older task semantics cannot migrate")
    a, b = deepcopy(old), deepcopy(new)
    stages = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.50, 0.20, 0.15, 0.15, 0.0, 0.0),
        (0.25, 0.20, 0.10, 0.10, 0.175, 0.175),
    )
    source = tuple(a["commands"]["sampler"]["domain_weights"])
    target = tuple(b["commands"]["sampler"]["domain_weights"])
    if (
        source not in stages
        or target not in stages
        or stages.index(target) - stages.index(source) not in (0, 1)
    ):
        raise ValueError("Only same-stage resume or A -> B -> C distribution migration is audited")
    # Stage A/B/C may change only these audited sampler fields. Physics,
    # rewards, terrain, coverage schedule, state and controller remain strict.
    for contract in (a, b):
        sampler = contract["commands"]["sampler"]
        for key in ("domain_weights", "flat_velocity_levels", "flat_speed_level"):
            sampler.pop(key, None)
    if a != b:
        raise ValueError("Stage migration changed protected Go2W task semantics")
