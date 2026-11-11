"""Export measured continuation counters and paired physics tables; never infer success from loss."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("results/stairs_omni_continuation_v2"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument(
        "--source", type=Path, default=Path("results/omnidirectional_terrain/stageB/last.pt")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    source = torch.load(a.source, map_location="cpu", weights_only=True, mmap=True)
    old = json.loads(a.baseline.read_text())
    new = json.loads(a.evaluation.read_text())
    summary = dict(
        checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
        source=str(a.source.resolve()),
        source_sha256=hashlib.file_digest(a.source.open("rb"), "sha256").hexdigest(),
        iteration=ckpt["iteration"],
        source_iteration=source["iteration"],
        cumulative_real_transitions=ckpt["sim_steps"],
        new_training_transitions=ckpt["sim_steps"] - source["sim_steps"],
        cumulative_updates=ckpt["update_counts"],
        new_attempted_updates={
            k: ckpt["update_counts"][k] - source["update_counts"][k]
            for k in source["update_counts"]
        },
        rollback_note="Counters include executed then rolled-back Actor optimizer updates. Selected Actor lineage must be read separately.",
        learner_wall_delta_s=ckpt["wall_time"] - source["wall_time"],
        wall_note="Learner-accounted wall time differs from process wall including external diagnostics; do not equate to two hours.",
        initial_actor_max_diff=max(
            float((ckpt["pi"][k] - v).abs().max()) for k, v in source["pi"].items()
        ),
        protected_contract_unchanged={
            k: source["task_contract"][k] == ckpt["task_contract"][k]
            for k in (
                "actor",
                "joints",
                "context",
                "action",
                "reward",
                "reward_manager",
                "terminations",
                "actions",
                "events",
                "model_sha256",
                "sim_dt",
                "ctrl_dt",
            )
        },
        stairs_speed_levels=ckpt["command_sampler"]["stairs_levels"].tolist()
        if "stairs_levels" in ckpt["command_sampler"]
        else None,
        flat_speed_level=ckpt["command_sampler"].get("speed_level"),
        episode_counts=len(new["records"]),
        evaluation_seeds=new["seeds"],
        same_evaluation_seeds=old["seeds"] == new["seeds"],
        same_evaluation_durations=old["seconds"] == new["seconds"]
        and old.get("maneuver_seconds") == new.get("maneuver_seconds"),
        same_safe_spawn=old.get("safe_spawn") == new.get("safe_spawn"),
    )
    if not all(summary["protected_contract_unchanged"].values()):
        raise ValueError("Protected contract changed")
    summary["phase_summaries"] = {}
    for phase in ("training", "controlled", "level7_training", "extra40_training"):
        path = a.run / phase / "summary.json"
        if path.exists():
            summary["phase_summaries"][phase] = json.loads(path.read_text())
    (a.output / "measured_summary.json").write_text(json.dumps(summary, indent=2))
    metrics = [
        "actual_vx_m_s",
        "actual_vy_m_s",
        "actual_wz_rad_s",
        "xy_rmse_m_s",
        "yaw_rmse_rad_s",
        "fall",
        "survival_s",
        "crossing_passed",
        "central_platform_reached",
        "max_terrain_change_m",
        "saturation",
        "geometric_slip_m_s",
    ]
    with (a.output / "paired_evaluation.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=["scenario"]
            + [f"{tag}_{k}" for tag in ("source", "candidate") for k in metrics],
        )
        w.writeheader()
        for name, val in new["aggregate"].items():
            if name not in old["aggregate"]:
                continue
            w.writerow(
                dict(
                    scenario=name,
                    **{
                        f"{tag}_{k}": vals["aggregate"][name].get(k)
                        for tag, vals in [("source", old), ("candidate", new)]
                        for k in metrics
                    },
                )
            )
    (a.output / "candidate_aggregate.json").write_text(
        json.dumps({k: v for k, v in new.items() if k != "records"}, indent=2)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "phase_summaries"}, indent=2))


if __name__ == "__main__":
    main()
