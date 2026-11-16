"""Strict, explicit A/B/C command-distribution migration for the v2 task."""

from copy import deepcopy


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
