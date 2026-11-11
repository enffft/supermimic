"""Explicit 5-to-8-row atlas migration; preserve networks/optimizers, refill Replay.

Old world coordinates are invalid on the expanded map. Never relabel old
transitions as new terrain or silently claim a step-equivalent resume.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_omni_common import configuration, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = torch.load(a.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    oldcfg = configuration("stairs_omni_continuation")
    newcfg = configuration("stairs_omni_level7")
    oldenv = environment(oldcfg, 32)
    oldenv.init_state()
    newenv = environment(newcfg, 32)
    newenv.init_state()
    try:
        old, new = oldenv.supermimic_contract, newenv.supermimic_contract
        if old != source["task_contract"]:
            raise ValueError("Source does not match the checked Level4 atlas/config")
        protected = {
            k: v for k, v in old.items() if k not in ("commands", "terrain", "terrain_fingerprint")
        }
        if protected != {
            k: v for k, v in new.items() if k not in ("commands", "terrain", "terrain_fingerprint")
        }:
            raise ValueError("Protected semantics changed")
        oc, nc = copy.deepcopy(old["commands"]), copy.deepcopy(new["commands"])
        for c in (oc, nc):
            c.pop("terrain_max_level")
            c.pop("terrain_coverage_weights")
            c["sampler"].pop("terrain_level_count", None)
        if oc != nc:
            raise ValueError("Unexpected command migration")
        oldterm = oldenv.command_manager.get_term("velocity")
        newterm = newenv.command_manager.get_term("velocity")
        axis = torch.linspace(-0.5, 7.2, 155)
        x, y = torch.meshgrid(axis, torch.linspace(-3.8, 3.8, 153), indexing="ij")
        offset = torch.stack((x, y), -1)
        errors = []
        for level in range(5):
            for t in range(5):
                o = oldterm.curriculum.origins[level, t, :2]
                n = newterm.curriculum.origins[level, t, :2]
                h0 = oldterm.mdp.terrain.ground_height(o + offset)
                h1 = newterm.mdp.terrain.ground_height(n + offset)
                # Compare identical local collision surface including absolute Z.
                errors.append(float((h0 - h1).abs().max()))
        print("legacy_tile_errors", errors, flush=True)
        # UniSim exports PNG; MuJoCo's compiled hfield is quantized and its
        # pixel spacing depends on atlas extent. We explicitly compare tread
        # plateaus separately from riser interpolation bands. This is a NEW
        # collision atlas, not a claim of byte-identical old terrain.
        plateau_errors = []
        points = torch.tensor(
            [[-0.1, 0], [0.3, 0], [0.8, 0], [1.2, 0], [1.6, 0], [2.0, 0], [2.4, 0], [3.0, 0]]
        )
        for level in range(5):
            for t in (3, 4):
                o = oldterm.curriculum.origins[level, t, :2]
                n = newterm.curriculum.origins[level, t, :2]
                plateau_errors.append(
                    float(
                        (
                            oldterm.mdp.terrain.ground_height(o + points)
                            - newterm.mdp.terrain.ground_height(n + points)
                        )
                        .abs()
                        .max()
                    )
                )
        if max(plateau_errors) > 0.005:
            raise ValueError(
                f"Legacy stair tread heights changed beyond 5mm: {max(plateau_errors)}"
            )
        nlevels = newterm.sampler.stairs_levels.shape[1]
        migrated = {
            k: v for k, v in source.items() if k not in ("replay", "curriculum", "command_sampler")
        }
        migrated["task_contract"] = new
        cur = newterm.curriculum
        saved = source["curriculum"]
        field = copy.deepcopy(saved)
        field["tensors"]["origins"] = cur.origins.cpu()
        field["tensors"]["coverage_cdf"] = cur.coverage_cdf.cpu()
        field["tensors"]["type_max_levels"] = cur.type_max_levels.cpu()
        for key, value in field["statistics"].items():
            out = torch.zeros(value.shape[0], nlevels, dtype=value.dtype)
            out[:, : value.shape[1]] = value
            field["statistics"][key] = out
        migrated["curriculum"] = field
        newterm.sampler.load_state_dict(source["command_sampler"], allow_distribution_change=True)
        migrated["command_sampler"] = newterm.sampler.state_dict()
        migrated["extra_state"] = copy.deepcopy(source["extra_state"])
        extra = migrated["extra_state"]
        for key in (
            "stairs_warm_admission",
            "stairs_schedule",
            "omni_coverage",
            "stairs_command_coverage",
            "actual_actor_start_modes",
            "actual_policy_start_modes",
            "actual_actor_start_scopes",
        ):
            extra.pop(key, None)
        extra["stairs_run_origin"] = {
            k: source[k] for k in ("iteration", "sim_steps", "update_counts", "wall_time")
        }
        extra["stairs_run_origin"]["path"] = str(a.checkpoint.resolve())
        extra["atlas_migration"] = dict(
            source=str(a.checkpoint.resolve()),
            source_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
            old_fingerprint=old["terrain_fingerprint"],
            new_fingerprint=new["terrain_fingerprint"],
            old_rows=5,
            new_rows=8,
            stairs_ceiling_cm=14,
            other_types_max_level=4,
            legacy_collision_height_max_error_m=max(errors),
            legacy_stair_tread_max_error_m=max(plateau_errors),
            collision_change="expanded PNG atlas changes compiled quantization/interpolation; old map preserved, Replay discarded",
            replay="intentionally empty: changed atlas coordinates",
            weights_optimizers_unchanged=True,
            source_counters={
                k: source[k] for k in ("iteration", "sim_steps", "update_counts", "wall_time")
            },
        )
        # Generate a layout-compatible empty buffer without allocating a second
        # 2048-env simulator. Shapes/dtypes follow the source, not guessed fields.
        migrated["replay"] = {
            k: v for k, v in source["replay"].items() if k not in ("data", "start_scope_mask")
        }
        migrated["replay"].update(
            ptr=0,
            size=0,
            data={k: torch.zeros_like(v) for k, v in source["replay"]["data"].items()},
            start_scope_mask=None,
        )
        destination = a.output / "migrated.pt"
        torch.save(migrated, destination)
        manifest = extra["atlas_migration"] | dict(
            checkpoint=str(destination.resolve()),
            sha256=hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
        )
        (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        OmegaConf.save(newcfg, a.output / "config.yaml")
        print(json.dumps(manifest), flush=True)
    finally:
        oldenv.close()
        oldenv.backend.close()
        newenv.close()
        newenv.backend.close()


if __name__ == "__main__":
    main()
