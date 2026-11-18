"""Explicit, auditable migration from window mastery to distance curriculum."""

import copy
from dataclasses import asdict

import torch

from .experiment import write_json
from .world_model import dynamics_contract


def validate_migration_contract(old, new):
    if old.get("curriculum_protocol") != "ten_level_per_skill_hysteresis_review_and_coverage":
        raise ValueError("Expected the recorded old window curriculum")
    if new.get("curriculum_protocol") != "per_env_episode_distance_fixed_domain_v1":
        raise ValueError("Expected the distance curriculum")
    if dynamics_contract(old) != dynamics_contract(new):
        raise ValueError("Migration cannot change dynamics/robot/control")
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    for item in (a, b):
        item.pop("curriculum_protocol")
        item["commands"].pop("curriculum", None)
        item["commands"].pop("actor_level_lookahead", None)
    if a != b:
        raise ValueError(
            "Migration permits only curriculum/distribution semantics; reward or other task protocol differs"
        )


def validate_algo_migration(old, new, *, smoke=False):
    old = dict(old)
    old.setdefault("actor_domain_weights", None)
    allowed = {"actor_domain_weights"}
    if smoke:
        allowed |= {"iterations", "log_every", "log_backend"}
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if changed - allowed:
        raise ValueError(f"Migration algorithm change forbidden: {sorted(changed - allowed)}")
    return sorted(changed)


def migrate_checkpoint(learner, source, output, source_identity, *, smoke=False):
    """Preserve all networks/normalizers/optimizers and physical WM history.

    Formal runs retain recent ring data with original metadata. Smaller smoke
    batches explicitly refill only the recent ring, whose N dimension changed.
    """
    saved = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    required = {"wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v", "replay", "curriculum"}
    if not required <= saved.keys():
        raise ValueError("Migration needs a complete training checkpoint")
    validate_migration_contract(saved["task_contract"], learner.env.supermimic_contract)
    changed = validate_algo_migration(saved["args"], asdict(learner.cfg), smoke=smoke)
    if saved["replay"]["history"]["protocol"] != learner.buffer.history.protocol:
        raise ValueError("Historical dynamics protocol differs")
    migrated = dict(saved)
    migrated["args"] = asdict(learner.cfg)
    migrated["task_contract"] = learner.env.supermimic_contract
    command = learner.env.command_manager.get_term("velocity")
    migrated["curriculum"] = command.curriculum.state_dict()
    migrated["command_sampler"] = command.sampler.state_dict()
    same_layout = (saved["replay"]["N"], saved["replay"]["T"]) == (
        learner.buffer.N,
        learner.buffer.T,
    )
    if not same_layout and not smoke:
        raise ValueError("Formal migration cannot reduce environment/buffer dimensions")
    replay = dict(saved["replay"]) if same_layout else learner.buffer.state_dict()
    replay["history"] = saved["replay"]["history"]
    replay["history_actual"] = [0, 0]
    replay["start_scope_mask"] = command.curriculum.start_scope_mask()
    replay["wm_scope_counts"] = torch.zeros_like(learner.buffer.wm_scope_counts)
    replay["sampling_counts"] = {
        k: torch.zeros_like(v) for k, v in learner.buffer.sampling_counts.items()
    }
    replay["wm_level_counts"] = torch.zeros_like(learner.buffer.wm_level_counts)
    migrated["replay"] = replay
    # Monotone target counters remain unique against retained old trajectories.
    if not same_layout:
        migrated["command_counter"] = command.command_counter.clone()
    extra = dict(saved.get("extra_state", {}))
    for key in (
        "actual_policy_start_modes",
        "actual_actor_start_modes",
        "actual_actor_start_scopes",
        "actor_starts_domain_terrain_speed",
        "best_physical_score",
    ):
        extra.pop(key, None)
    record = dict(
        migration="window_to_per_env_distance_v1",
        parent_checkpoint=str(source.resolve()),
        parent_source_identity=saved.get("extra_state", {}).get("formal_source_sha256"),
        current_source_identity=source_identity,
        algorithm_fields_changed=changed,
        preserved=[
            "Actor",
            "Critic",
            "World Model and all normalizers",
            "three optimizers",
            "value normalization",
            "iteration/time/update counters",
            "historical WM bank and RNG",
        ],
        recent_replay="retained physical data; reset boundary closed"
        if same_layout
        else "explicitly empty/refilled for smaller smoke N",
        reset=[
            "per-env curriculum levels and episode statistics",
            "fixed domain/coverage roles",
            "command sampler RNG",
            "sampling diagnostics",
            "old best-selection score",
        ],
        semantics="new sampling experiment, not step-equivalent; no reward/control/dynamics change",
    )
    extra["sampling_curriculum_migration"] = record
    extra["formal_source_sha256"] = source_identity
    migrated["extra_state"] = extra
    path = output / "migration_initial.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(migrated, temporary)
    temporary.replace(path)
    write_json(output / "curriculum_migration.json", record)
    return path, same_layout
