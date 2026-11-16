"""Strict, explicit A/B/C command-distribution migration for the v2 task."""

from copy import deepcopy


def initialize_wm_only(learner, path, *, expected_file_sha256, expected_tensor_sha256):
    """Load an audited Go2W dynamics asset into an untouched new learner.

    Retain every normalizer buffer, but never restore any optimizer or update
    counter. Validate the entire transfer before changing the destination WM.
    The input asset is opened read-only and never published back by this API.
    """
    import hashlib
    import json
    from dataclasses import asdict
    from pathlib import Path

    import torch

    from .world_model import dynamics_contract, tensor_digest

    optimizers = (learner.opt_wm, learner.opt_pi, learner.opt_v)
    if (learner.iteration or learner.sim_steps or learner.buffer.size
            or any(learner.update_counts.values())
            or any(optimizer.state for optimizer in optimizers)
            or learner.initialization["wm"] != "fresh"
            or learner.initialization["actor"] != "fresh"
            or learner.initialization["critic"] != "fresh"):
        raise ValueError("WM-only initialization requires fresh Actor/Critic, optimizers and replay")
    if not isinstance(learner.opt_wm, torch.optim.Adam):
        raise ValueError("WM-only warm start requires a fresh Adam optimizer")
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    file_sha = digest.hexdigest()
    if file_sha != expected_file_sha256:
        raise ValueError("WM source file SHA256 differs from audited asset")
    asset = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if asset.get("format") != "go2w_continual_dynamics_asset":
        raise ValueError("WM-only initialization requires an exported Go2W dynamics asset")
    expected_contract = dynamics_contract(learner.env.supermimic_contract)
    if asset.get("dynamics_contract") != expected_contract:
        differences = [key for key in expected_contract
                       if asset.get("dynamics_contract", {}).get(key) != expected_contract[key]]
        raise ValueError(f"WM physical protocol differs: {differences}")
    architecture = asset.get("architecture", {})
    required = {"wm_hidden", "wm_structured", "terrain_conditioning", "terrain_sensor_grad"}
    if architecture.keys() != required:
        raise ValueError("WM architecture metadata is incomplete or unknown")
    args = asdict(learner.cfg)
    for key, value in architecture.items():
        current = args[key]
        if isinstance(value, (list, tuple)):
            value, current = tuple(value), tuple(current)
        if value != current:
            raise ValueError(f"WM architecture differs: {key}")
    source = asset["wm"]
    target = learner.wm.state_dict()
    # Explicit key checks precede load_state_dict: the legacy WM loader supports
    # normalizer migration, which must never occur in this exact warm start.
    if source.keys() != target.keys():
        raise ValueError("WM state dictionary keys differ; migration is forbidden")
    for name, tensor in source.items():
        if tensor.shape != target[name].shape or tensor.dtype != target[name].dtype:
            raise ValueError(f"WM tensor layout differs: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"WM tensor is not finite: {name}")
        if name.endswith((".var", ".count")) and (tensor < 0).any():
            raise ValueError(f"WM normalization statistic is negative: {name}")
    for prefix in ("in_norm", "out_norm", "inc_norm"):
        if not all(f"{prefix}.{suffix}" in source for suffix in ("mean", "var", "count")):
            raise ValueError(f"WM normalizer is incomplete: {prefix}")
    source_sha = tensor_digest(source)
    if source_sha != expected_tensor_sha256 or source_sha != asset.get("tensor_sha256"):
        raise ValueError("WM tensor SHA256 differs from audited asset")
    before = {"actor": tensor_digest(learner.pi.state_dict()),
              "critic": tensor_digest(learner.critic.state_dict())}
    if learner.pi_avg is not None:
        before["actor_ema"] = tensor_digest(learner.pi_avg.state_dict())
    learner.wm.load_state_dict(source, strict=True)
    learner.wm.requires_grad_(True)
    learner.wm.train()
    if tensor_digest(learner.wm.state_dict()) != source_sha:
        raise RuntimeError("WM transfer changed source tensors")
    after = {"actor": tensor_digest(learner.pi.state_dict()),
             "critic": tensor_digest(learner.critic.state_dict())}
    if learner.pi_avg is not None:
        after["actor_ema"] = tensor_digest(learner.pi_avg.state_dict())
    if before != after or any(optimizer.state for optimizer in optimizers):
        raise RuntimeError("WM-only initialization changed policy/value or optimizer state")
    record = dict(
        path=str(path), file_sha256=file_sha, tensor_sha256=source_sha,
        source_checkpoint=asset.get("source_checkpoint"),
        source_iteration=asset.get("source_iteration"),
        source_wm_updates=asset.get("wm_updates"), run_wm_updates=0,
        restored=["WM_network", "input_normalizer", "output_normalizer", "increment_normalizer"],
        optimizer_initialization="fresh Adam; empty state; step zero",
        normalizer_counts={key: float(source[key]) for key in source if key.endswith(".count")},
        actor_critic_before=before, actor_critic_after=after,
        replay_size=learner.buffer.size, iteration=learner.iteration,
        update_counts=dict(learner.update_counts),
        frozen=False, old_replay_loaded=False,
    )
    learner.extra_state["wm_only_initialization"] = record
    learner.initialization["wm"] = record
    (learner.log_dir / "wm_initialization.json").write_text(json.dumps(record, indent=2))
    (learner.log_dir / "initialization.json").write_text(json.dumps(learner.initialization, indent=2))
    return record


def initialize_experiment(learner, path):
    """Initialize actor and continue dynamics in a different reward experiment.

    Critic, actor optimizer, replay and mastery remain fresh; the WM optimizer,
    normalizers and WM update counter continue. Protected dynamics must match.
    """
    from dataclasses import asdict

    import torch

    if learner.iteration or learner.buffer.size or any(learner.update_counts.values()):
        raise ValueError("Weight initialization requires an untouched new learner")
    source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    old, new = deepcopy(source["task_contract"]), deepcopy(learner.env.supermimic_contract)
    original_old, original_new = deepcopy(old), deepcopy(new)
    for contract in (old, new):
        if contract["schema"] != "go2w_velocity_supermimic_v2":
            raise ValueError("Experiment initialization requires Go2W v2")
        if contract["curriculum_protocol"] not in (
            "wm_coverage_then_skill_v2_episode_fixed_domain",
            "persistent_wm_eighth_domain_level_quotas",
        ):
            raise ValueError("Unknown curriculum protocol")
        contract.pop("curriculum_protocol")
        contract["commands"].pop("persistent_wm_coverage", None)
        for reward in (contract["reward"], contract["commands"]["reward"]):
            if reward["sigma_y"] not in (0.25, 0.15):
                raise ValueError("Unaudited lateral reward width")
            if reward.get("stop_reward_command_sigma", 0.05) not in (0.05, 0.02):
                raise ValueError("Unaudited stop gate")
            reward.pop("stop_reward_command_sigma", None)
            reward.pop("sigma_y")
    if old != new:
        raise ValueError("Experiment initialization changed protected task semantics")
    current_args = asdict(learner.cfg)
    for key in (
        "hist", "goal_mode", "goal_frames", "policy_hidden", "wm_hidden", "wm_structured",
        "terrain_training", "terrain_conditioning", "terrain_sensor_grad",
        "critic_terrain_conditioning", "horizon", "wm_window", "action_noise",
    ):
        old_value, new_value = source["args"][key], current_args[key]
        if isinstance(old_value, (list, tuple)):
            old_value, new_value = tuple(old_value), tuple(new_value)
        if old_value != new_value:
            raise ValueError(f"Experiment initialization changed protected learner field: {key}")
    learner.pi.load_state_dict(source["pi"])
    learner.wm.load_state_dict(source["wm"])
    if "opt_wm" not in source:
        raise ValueError("Continual dynamics initialization requires the source WM optimizer")
    learner.opt_wm.load_state_dict(source["opt_wm"])
    learner.update_counts["wm"] = source.get("update_counts", {}).get("wm", 0)
    if learner.pi_avg is not None:
        learner.pi_avg.module.load_state_dict(source["pi"])
    audit = {
        "source": str(path), "source_iteration": source["iteration"],
        "transferred": ["actor", "world_model", "their_normalizers", "WM_optimizer", "WM_update_count"],
        "fresh": ["critic", "actor_and_critic_optimizers", "replay", "mastery", "RNG", "iteration"],
        "source_contract": original_old, "target_contract": original_new,
        "mode": "new_experiment_weight_initialization_not_resume",
    }
    learner.extra_state["experiment_initialization"] = audit
    return audit


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
