"""Fresh SuperMimic comparison run using the same reference config as PPO."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import time

import torch

from supermimic.config import TrainConfig

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt"


def apply_config(target, values):
    """Restore dataclass config, retaining defaults for newly added unrelated fields."""
    for f in fields(target):
        if f.name not in values:
            continue
        old, value = getattr(target, f.name), values[f.name]
        if is_dataclass(old):
            apply_config(old, value)
        else:
            if isinstance(old, Path) and value is not None:
                value = Path(value)
            elif isinstance(old, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(target, f.name, value)
    return target


def fresh_config(reference_config, log_dir, num_envs=3072, max_hours=2.5, iterations=320000):
    cfg = apply_config(TrainConfig(), reference_config)
    cfg.resume = cfg.init_wm_from = None
    cfg.seed = 2026
    cfg.run_name, cfg.log_root = log_dir.name, log_dir.parent
    cfg.env.num_envs = num_envs
    cfg.env.velocity_curriculum_initial_stage = 0
    # Env noise is off in this comparison; do not silently retain the old
    # default that injects observation noise only into imagined Actor inputs.
    cfg.policy_trainer.obs_noise = cfg.env.obs_noise
    cfg.schedule.max_hours = max_hours
    cfg.schedule.total_iterations = iterations
    cfg.schedule.stop_at_iteration = None
    cfg.schedule.checkpoint_interval_minutes = 5.0
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--num-envs", type=int, default=3072)
    p.add_argument("--max-hours", type=float, default=2.5)
    p.add_argument("--iterations", type=int, default=320000)
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs/go2_comparison/supermimic_2p5h_seed2026")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    reference = torch.load(a.reference, map_location="cpu", weights_only=False, mmap=True)["config"]
    out = a.log_dir.resolve()
    cfg = fresh_config(reference, out, a.num_envs, a.max_hours, a.iterations)
    if a.resume is not None:
        saved = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
        cfg = apply_config(TrainConfig(), saved["config"])
        cfg.log_root, cfg.run_name = out.parent, out.name
        cfg.resume, cfg.init_wm_from = a.resume.resolve(), None
        cfg.schedule.max_hours = max(0., a.max_hours - float(saved.get("training_wall_seconds", 0.)) / 3600)
        cfg.schedule.total_iterations = a.iterations
        del saved
    # Bound initialization memory on this 16 GiB host. Eager execution keeps
    # the same algorithm; the archived failure ended during compiler setup.
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    if a.smoke:
        cfg.env.replay_capacity = 64
        cfg.env.velocity_wm_batch_size = cfg.env.velocity_actor_batch_size = 4
        cfg.schedule.warmup_iterations = 6
        cfg.schedule.wm_only_iterations = 1
        cfg.schedule.actor_ramp_iterations = 1
        cfg.schedule.log_interval = 2
        cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = 4
        cfg.wm_trainer.micro_batch_size = cfg.policy_trainer.micro_batch_size = 4
        cfg.wm_trainer.batches_per_update = 1
    from dataclasses import asdict
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_config.json").write_text(json.dumps({
        "algorithm": "SuperMimic", "fresh_initialization": True,
        "reference_config_only": str(a.reference.resolve()), "config": asdict(cfg),
        "primary_budget": "equal wall clock; simulation/model training/compilation/checkpoint IO included; initialization excluded",
    }, indent=2, default=str) + "\n")
    if a.plan_only:
        print(json.dumps({"resume": cfg.resume, "init_wm_from": cfg.init_wm_from,
                          "envs": cfg.env.num_envs, "max_hours": cfg.schedule.max_hours}))
        return
    from supermimic.training.loop import Trainer
    trainer = Trainer(cfg)
    started = time.monotonic()
    try:
        trainer.run()
    finally:
        trainer.env.close()
    (out / "progress.json").write_text(json.dumps({
        "completed_env_steps_per_env": trainer.iteration,
        "actor_updates": trainer.actor_updates, "wm_update_events": trainer.num_updates,
        "horizon": cfg.policy_trainer.horizon,
        "real_transitions": trainer.iteration * a.num_envs,
        "run_wall_seconds": time.monotonic() - started,
        "training_wall_seconds": trainer._training_elapsed_before + time.time() - trainer._training_start_time if hasattr(trainer, "_training_elapsed_before") else time.time() - trainer._training_start_time,
        "checkpoint": str(out / "ckpt_latest.pt"),
    }, indent=2) + "\n")
    (out / "completion.json").write_text((out / "progress.json").read_text())


if __name__ == "__main__":
    main()
