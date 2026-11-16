"""Named, clearance-only migration preserving source networks and optimizer state."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from unisupermimic.tasks.velocity.go2.gait import Go2GaitCfg
from unisupermimic.tasks.velocity.go2.reward_migration import (
    MIGRATION,
    relabel_replay_clearance,
    validate_reward_migration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_omni_common import configuration, environment


def migrate(checkpoint, output, task):
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = configuration(task)
    env = environment(cfg, 32)
    env.init_state()
    try:
        changes = validate_reward_migration(saved["task_contract"], env.supermimic_contract)
        # Copy only changed tensors; never mutate file-backed source storage.
        saved = dict(saved)
        saved["replay"] = dict(saved["replay"])
        saved["replay"]["data"] = dict(saved["replay"]["data"])
        saved["replay"]["data"]["r"] = saved["replay"]["data"]["r"].clone()
        labels = relabel_replay_clearance(
            saved["replay"],
            Go2GaitCfg(**saved["task_contract"]["reward"]["gait"]),
            env.supermimic_mdp,
        )
        old = saved["task_contract"]
        saved["task_contract"] = env.supermimic_contract
        saved["extra_state"] = copy.deepcopy(saved["extra_state"])
        manifest = dict(
            migration=MIGRATION,
            source=str(checkpoint.resolve()),
            source_sha256=hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
            changes=changes,
            replay=labels,
            inherited=[
                "pi",
                "wm",
                "critic",
                "opt_pi",
                "opt_wm",
                "opt_v",
                "value_normalization",
                "curricula",
                "sampler",
                "rng",
                "counts",
            ],
            critic="preserved; 100 critic-only updates required before Actor; raw-return EMA adapts during warmup",
            normalization="max_step_reward is reporting only; new positive Gaussian upper bound .105 vs .104 per step",
            old_reward=old["reward"],
            new_reward=saved["task_contract"]["reward"],
        )
        saved["extra_state"]["clearance_reward_migration"] = manifest
        torch.save(saved, output / "migrated.pt")
        manifest["sha256"] = hashlib.file_digest(
            (output / "migrated.pt").open("rb"), "sha256"
        ).hexdigest()
        (output / "migration.json").write_text(json.dumps(manifest, indent=2))
        OmegaConf.save(cfg, output / "config.yaml")
        print(json.dumps(manifest), flush=True)
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--task",
        choices=["stairs_clearance_gaussian", "stairs_clearance_soft_deficit"],
        required=True,
    )
    a = p.parse_args()
    migrate(a.checkpoint, a.output, a.task)
