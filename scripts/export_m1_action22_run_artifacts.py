#!/usr/bin/env python3
"""Export compact, reproducible artifacts for an M1 Action22 training run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EVAL_METRICS = (
    "fall_rate",
    "base_vx_rmse_m_s",
    "base_vy_rmse_m_s",
    "base_velocity_xy_vector_rmse_m_s",
    "base_velocity_xy_per_axis_rmse_m_s",
    "yaw_rate_rmse_rad_s",
    "signed_forward_displacement_mean_m",
    "positive_command_negative_vx_fraction",
    "prior_reference_root_dx_0p2s_mean_m",
    "prior_reference_wrong_direction_fraction",
    "prior_reference_wheel_wrong_direction_fraction",
    "tracking_wheel_rmse_rad_s",
    "action_saturation_fraction",
    "mean_episode_duration_s",
    "base_height_mean_m",
    "base_tilt_mean_rad",
    "mean_ee_distance_m",
    "mean_ee_progress_m",
    "end_window_ee_distance_m",
    "stable_hold_success_rate",
    "base_displacement_mean_m",
    "command_in_saved_training_range",
    "ood_test",
    "episode_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def flatten_eval(path: Path, label: str) -> list[dict[str, Any]]:
    document = read_json(path)
    rows = []
    for group, values in document["groups"].items():
        row = {
            "evaluation": label,
            "iteration": document.get("iteration"),
            "task": document.get("task_type"),
            "controller": document.get("controller"),
            "group": group,
        }
        row.update({key: values.get(key) for key in EVAL_METRICS})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_scalars(run_dir: Path, output: Path) -> tuple[list[dict[str, Any]], list[str]]:
    accumulator = EventAccumulator(
        str(run_dir / "tb"), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    rows: list[dict[str, Any]] = []
    for tag in tags:
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "step": event.step,
                    "wall_time_unix": event.wall_time,
                    "tag": tag,
                    "value": event.value,
                }
            )
    rows.sort(key=lambda row: (row["step"], row["tag"]))
    write_csv(output / "training_scalars.csv", rows)
    return rows, tags


def make_plot(run_dir: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accumulator = EventAccumulator(
        str(run_dir / "tb"), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    panels = [
        ("wm/nse_1step", "wm/nse_kstep", "WM replay NSE (1 and 8 step)"),
        ("actor/total", "actor/task", "Actor total and task loss"),
        ("real/L/vx_rmse", "real/L/yaw_rate_error", "Task L velocity errors"),
        ("real/M/ee_distance", "real/M/ee_progress", "Task M EE distance and progress"),
        ("real/L_fall_rate", "real/M_fall_rate", "L and M episode fall rates"),
        ("schedule/wm_optimizer_steps", "schedule/actor_optimizer_steps", "Optimizer updates"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, (tag_a, tag_b, title) in zip(axes.flat, panels):
        for tag in (tag_a, tag_b):
            if tag not in tags:
                continue
            events = accumulator.Scalars(tag)
            axis.plot(
                [event.step for event in events],
                [event.value for event in events],
                label=tag.rsplit("/", 1)[-1],
                linewidth=1.2,
            )
        axis.set_title(title)
        axis.set_xlabel("training iteration")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("M1 Action22 L/M 50:50, 2048 environments", fontsize=14)
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def build_summary(run_dir: Path, output: Path, tags: list[str]) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    final_eval = run_dir / "evaluation_corrected"
    snapshot_eval = run_dir / "evaluation_snapshots"
    eval_rows: list[dict[str, Any]] = []
    eval_paths: list[dict[str, str]] = []
    for path in sorted(final_eval.glob("*.json")):
        eval_rows.extend(flatten_eval(path, f"final_{path.stem}"))
        eval_paths.append({"label": f"final_{path.stem}", "path": path.name})
        shutil.copy2(path, output / f"evaluation_final_{path.stem}.json")
    for path in sorted(snapshot_eval.glob("*/*.json")):
        label = f"{path.parent.name}_{path.stem}"
        eval_rows.extend(flatten_eval(path, label))
        eval_paths.append({"label": label, "path": f"{path.parent.name}/{path.name}"})
        (output / "evaluation_snapshots").mkdir(exist_ok=True)
        shutil.copy2(path, output / "evaluation_snapshots" / f"{label}.json")
    write_csv(output / "evaluation_metrics.csv", eval_rows)

    monitor = run_dir / "gpu_monitor.csv"
    gpu_summary: dict[str, Any] = {"samples": 0}
    if monitor.exists():
        with monitor.open(newline="") as stream:
            records = list(csv.DictReader(stream, skipinitialspace=True))
        if records:
            gpu_summary["samples"] = len(records)
            aliases = {
                "memory.used [MiB]": "memory_used_mib",
                "utilization.gpu [%]": "gpu_util_percent",
                "power.draw [W]": "power_w",
                "temperature.gpu": "temperature_c",
            }
            for column, name in aliases.items():
                values = [float(record[column].split()[0]) for record in records]
                gpu_summary[name] = {
                    "min": min(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                }
        normalized_monitor = monitor.read_text().replace("\r\n", "\n")
        (output / "gpu_monitor.csv").write_text(normalized_monitor)
    for name, output_name in (
        ("console.log", "training_console.log"),
        ("host_memory_monitor.log", "host_memory_monitor.log"),
    ):
        source = run_dir / name
        if source.exists():
            normalized_log = "\n".join(
                line.rstrip(" \t") for line in source.read_text().splitlines()
            ) + "\n"
            (output / output_name).write_text(normalized_log)

    checkpoints = []
    for path in sorted(run_dir.glob("ckpt_*.pt")):
        checkpoints.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "not_committed": True,
            }
        )
    prior = read_json(final_eval / "L_actor.json")["prior_source"]
    latest = read_json(final_eval / "L_actor.json")
    important_scalars = {}
    scalar_path = output / "training_scalars.csv"
    with scalar_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["tag"] in {
                "schedule/wm_optimizer_steps",
                "schedule/actor_optimizer_steps",
                "schedule/prior_optimizer_steps",
                "wm/nse_1step",
                "wm/nse_kstep",
                "actor/grad_norm",
                "real/L_sample_count",
                "real/M_sample_count",
                "env/task_l_count",
                "env/task_m_count",
                "real/L_fall_rate",
                "real/M_fall_rate",
                "schedule/actor_ready",
                "schedule/prior_gate_pass",
            }:
                important_scalars[row["tag"]] = {
                    "step": int(row["step"]),
                    "value": float(row["value"]),
                }

    summary = {
        "run_name": config["run_name"],
        "git_head_at_run": "f7858a6cf327e668de84341681292beeb061293e",
        "git_worktree_at_run": "dirty; training used the M1 working-tree implementation on top of git_head_at_run",
        "post_start_source_changes": [
            "metric aggregation denominator and positive-command direction metric were corrected after the running Python process started; the process did not hot-reload these changes",
        ],
        "task": "balanced L/M fixed quota",
        "num_envs": config["env"]["num_envs"],
        "iterations": 4000,
        "elapsed_training_hours": 1.0,
        "prior_updates": 0,
        "wm_optimizer_steps": 1584,
        "actor_optimizer_steps": 1384,
        "wm_only_updates": config["schedule"]["m1_wm_only_iterations"],
        "actual_task_quota_per_iteration": {"L": 1024, "M": 1024},
        "replay_batch_quota": {"L": 16, "M": 16},
        "prior": {
            "path": prior["path"],
            "sha256": prior["sha256"],
            "feature_version": prior["feature_version"],
            "input_compatibility": latest["prior_input_compatibility"],
            "projected_condition_indices": prior["projected_condition_indices"],
            "normalization_hash": prior["normalization_hash"],
            "strict_fcl_only": prior["strict_fcl_only"],
        },
        "reset_contract": latest["reset_contract"],
        "action_mapping": latest["action_mapping"],
        "fixed_evaluations": {
            "seed": latest["seed"],
            "num_envs": latest["num_envs"],
            "steps": latest["steps"],
            "domain_randomization": latest["domain_randomization"],
            "reports": eval_paths,
        },
        "gpu_monitor": gpu_summary,
        "snapshots": checkpoints,
        "selected_final_scalars": important_scalars,
        "tensorboard_scalar_tags": len(tags),
        "source_log_dir": str(run_dir),
        "note": "Checkpoint binaries are deliberately excluded from this results artifact; hashes and sizes are recorded.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("logs/m1_prior_actor/m1_action22_v1_LM50_2048_1h_20261103"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/m1_action22_v1_LM50_2048_1h_20261103"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, tags = export_scalars(args.run_dir, args.output_dir)
    make_plot(args.run_dir, args.output_dir)
    summary = build_summary(args.run_dir, args.output_dir, tags)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "scalar_tags": summary["tensorboard_scalar_tags"],
                "checkpoint_count": len(summary["snapshots"]),
                "gpu_samples": summary["gpu_monitor"]["samples"],
            }
        )
    )


if __name__ == "__main__":
    main()
