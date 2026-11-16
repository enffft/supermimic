"""Explicit user-authorized 4cm/Level5 atlas migration, keeping A2/network/PD fixed."""

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_omni_common import configuration, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = configuration("stairs_clearance_20cm")
    env = environment(cfg, 32)
    env.init_state()
    try:
        old, new = source["task_contract"], env.supermimic_contract

        def protected(c):
            return {
                k: v
                for k, v in c.items()
                if k not in ("commands", "terrain", "terrain_fingerprint")
            }

        if protected(old) != protected(new):
            raise ValueError("Non-terrain semantics changed, including A2 reward")
        oc, nc = copy.deepcopy(old["commands"]), copy.deepcopy(new["commands"])
        for c in (oc, nc):
            for key in ("terrain_max_level", "terrain_stair_increment", "terrain_coverage_weights"):
                c.pop(key, None)
            c["sampler"].pop("terrain_level_count", None)
        if oc != nc:
            raise ValueError("Unexpected sampler/control difference")
        command = env.command_manager.get_term("velocity")
        cur = command.curriculum
        field = command.mdp.terrain
        quantization_tolerance = max(
            0.005, float(env.get_playback_model(0).hfield_size[0, 2]) / 255 + 1e-6
        )
        heights = []
        for l in range(1, 6):
            for t in (3, 4):
                entry = cur.origins[l, t, :2]
                actual = float(
                    (
                        field.ground_height(entry + entry.new_tensor([0.8, 0]))
                        - field.ground_height(entry)
                    ).abs()
                )
                if abs(actual - 0.04 * l) > quantization_tolerance:
                    raise ValueError(f"Compiled first stair height L{l} type{t}: {actual}")
                heights.append(
                    dict(level=l, type=t, target_m=0.04 * l, compiled_first_riser_m=actual)
                )
        saved = {
            k: v
            for k, v in source.items()
            if k not in ("replay", "curriculum", "command_sampler", "extra_state")
        }
        saved["task_contract"] = new
        saved["extra_state"] = copy.deepcopy(source["extra_state"])
        oldcur = source["curriculum"]
        ct = copy.deepcopy(oldcur["tensors"])
        for key in ("origins", "coverage_cdf", "type_max_levels"):
            ct[key] = getattr(cur, key).cpu().clone()
        stairs = ct["terrain_type"] >= 3
        ct["mastery"] = torch.where(stairs, ct["mastery"] // 2, ct["mastery"].clamp_max(4))
        ct["terrain_level"] = ct["mastery"].clone()
        for key in (
            "episode_progress",
            "commanded_progress",
            "velocity_tracking_error",
            "yaw_tracking_error",
            "steps",
            "eligible_steps",
            "max_lateral",
            "episode_fall",
            "probe",
            "probe_offset",
            "last_decision",
            "success_streak",
            "failure_streak",
            "start_xy",
        ):
            ct[key].zero_()
        saved["curriculum"] = dict(
            phase=cur.phase,
            tensors=ct,
            statistics={k: torch.zeros_like(v) for k, v in cur.statistics.items()},
        )
        sampler = copy.deepcopy(source["command_sampler"])
        sampler["config"] = asdict(command.sampler.cfg)
        speeds = torch.zeros(2, 6, dtype=torch.long)
        for l in range(1, 4):
            speeds[:, l] = sampler["stairs_levels"][:, 2 * l]
        sampler["stairs_levels"] = speeds
        saved["command_sampler"] = sampler
        replay = {
            k: v for k, v in source["replay"].items() if k not in ("data", "start_scope_mask")
        }
        mask = torch.ones(6, 6, dtype=torch.bool)
        mask[4:, 1:] = False
        replay.update(
            ptr=0,
            size=0,
            data={k: torch.zeros_like(v) for k, v in source["replay"]["data"].items()},
            start_scope_mask=mask,
            wm_level_counts=torch.zeros(8, dtype=torch.long),
        )
        if "command_speed_level" in replay["data"]:
            replay["data"]["command_speed_level"].fill_(-1)
        saved["replay"] = replay
        for key in (
            "stairs_run_origin",
            "stairs_source",
            "stairs_schedule",
            "stairs_warm_admission",
            "clearance_scope_admission",
            "actual_actor_start_scopes",
            "actor_starts_domain_terrain_speed",
            "stairs_command_coverage",
            "omni_coverage",
        ):
            saved["extra_state"].pop(key, None)
        manifest = dict(
            migration="go2_stairs_4cm_level5_atlas_v1",
            source=str(a.checkpoint.resolve()),
            source_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
            old_fingerprint=old["terrain_fingerprint"],
            new_fingerprint=new["terrain_fingerprint"],
            source_counters={
                k: source[k] for k in ("iteration", "sim_steps", "update_counts", "wall_time")
            },
            stair_heights_m=[0.04, 0.08, 0.12, 0.16, 0.20],
            rough_slope_max_level=4,
            mastery="stairs floor(old_level / 2), rough/slopes unchanged; reset streaks and new-map statistics; previous statistics archived",
            replay="intentionally empty: atlas world coordinates and collision interpolation changed; refill with real physics, no cross-reset windows",
            actor_admission="all new nonflat stairs blocked pending fresh seven-probe/H32 validation",
            preserved="Actor, WM, Critic, optimizers, value normalization, accumulated counters, flat speed state, sampler RNG",
            reward_equal=old["reward"] == new["reward"],
            compiled_heights=heights,
            compiled_height_tolerance_m=quantization_tolerance,
        )
        saved["extra_state"]["terrain_20cm_migration"] = manifest
        saved["extra_state"]["previous_atlas_statistics"] = source["curriculum"]["statistics"]
        torch.save(saved, a.output / "migrated.pt")
        manifest["output_sha256"] = hashlib.file_digest(
            (a.output / "migrated.pt").open("rb"), "sha256"
        ).hexdigest()
        (a.output / "migration.json").write_text(json.dumps(manifest, indent=2))
        OmegaConf.save(cfg, a.output / "config.yaml")
        print(json.dumps(manifest), flush=True)
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
