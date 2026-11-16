"""Standalone plots of training and independent Go2W physics assessments."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((args.run / "training/train_log.csv").open()))
    minutes = np.array([float(row["wall_h"]) * 60 for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for axis, key, label in zip(axes.flat,
                               ("wm_loss", "wm_err8", "sim_rew", "imag_survival"),
                               ("WM training loss", "WM normalized H8 training error",
                                "Normalized physical collection reward", "Imagined survival")):
        axis.plot(minutes, [float(row[key]) for row in rows], linewidth=1.3)
        axis.set(xlabel="Training minutes (excludes initial collection)", ylabel=label)
        axis.grid(alpha=.25)
        if key in ("wm_loss", "wm_err8"):
            axis.set_yscale("log")
    figure.suptitle("Go2W v3 training diagnostics; skill claims require independent assessment")
    figure.savefig(args.output / "training.png", dpi=180)
    plt.close(figure)

    initial = json.loads((args.run / "evaluation/bootstrap.json").read_text())
    final_path = args.run / "evaluation/final.json"
    if not final_path.exists():
        candidates = [path for path in (args.run / "evaluation").glob("iteration_*.json")
                      if path.stem.removeprefix("iteration_").isdigit()]
        final_path = sorted(candidates)[-1]
    final = json.loads(final_path.read_text())
    names = ("Initial", f"Iteration {final['iteration']}")
    colors = ("#7a8793", "#007f86")
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, cases, component, label in (
        (axes[0, 0], ("forward_030", "forward_060", "backward_015"), 0, "Forward velocity (m/s)"),
        (axes[0, 1], ("left_006", "right_006", "left_015", "right_015"), 1, "Lateral velocity (m/s)"),
        (axes[1, 0], ("yaw_left_020", "yaw_right_020", "yaw_left_035", "yaw_right_035"), 2, "Yaw rate (rad/s)"),
    ):
        positions = np.arange(len(cases))
        for index, stage in enumerate((initial, final)):
            selected = [next(row for row in stage["rows"] if row["name"] == name) for name in cases]
            values = [row["mean_actual_vx_vy_yaw"][component]
                      if row["mean_actual_vx_vy_yaw"] is not None else np.nan for row in selected]
            axis.bar(positions + (index - .5) * .3, values, width=.3, label=names[index], color=colors[index])
        requested = [next(row for row in initial["rows"] if row["name"] == name)["goal"][component] for name in cases]
        axis.scatter(positions, requested, marker="_", s=240, color="black", label="Command", zorder=4)
        axis.set(xticks=positions, xticklabels=cases, ylabel=label)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.axhline(0, color="black", linewidth=.6)
        axis.grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axis = axes[1, 1]
    positions = np.arange(1, 8)
    for index, stage in enumerate((initial, final)):
        selected = [next(row for row in stage["rows"] if row["domain"] == 4 and row["level"] == level)
                    for level in positions]
        axis.bar(positions + (index - .5) * .3,
                 [100 * row["climb_successes"] / row["replicas"] for row in selected],
                 width=.3, label=names[index], color=colors[index])
    axis.set(xticks=positions, xticklabels=[str(level * 3) for level in positions],
             xlabel="Stair riser (cm)", ylabel="Independent ascent successes (%)", ylim=(0, 100))
    axis.grid(axis="y", alpha=.25)

    axis = axes[2, 0]
    domains = ("Flat", "Rough", "Slope up", "Slope down", "Stairs up", "Stairs down")
    for index, stage in enumerate((initial, final)):
        fractions = []
        for domain in range(6):
            selected = [row for row in stage["rows"] if row["domain"] == domain]
            fractions.append(100 * sum(row["falls"] for row in selected) / sum(row["replicas"] for row in selected))
        axis.bar(np.arange(6) + (index - .5) * .3, fractions, width=.3, color=colors[index])
    axis.set(xticks=range(6), xticklabels=domains, ylabel="Falls in first episode (%)", ylim=(0, 100))
    axis.tick_params(axis="x", rotation=20, labelsize=8)
    axis.grid(axis="y", alpha=.25)
    axes[2, 1].axis("off")
    details = []
    for name, stage in zip(names, (initial, final)):
        details.append(f"{name}: {sum(row['falls'] for row in stage['rows'])}/"
                       f"{sum(row['replicas'] for row in stage['rows'])} falls")
    axes[2, 1].text(0, .9, "\n".join(details) + "\n\nVelocity averages use live samples after 1 second.\n"
                    "Averages before a fall do not certify stable tracking.\n"
                    "Each case: 3 reset seeds x 4 replicas, up to 10 seconds.\n"
                    "One training seed and one deterministic terrain atlas.\n"
                    "Ascent requires survival, progress and ground gain.", va="top", fontsize=10)
    figure.suptitle("Go2W v3 independent physics comparison")
    figure.savefig(args.output / "physics_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
