"""Plot measured directional A/B, training telemetry and fixed heldout diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    root = a.run
    out = root / "figures"
    out.mkdir(exist_ok=True)
    base = json.loads((root / "baseline_verified/evaluation.json").read_text())["aggregate"]
    new = json.loads((root / "stageB_evaluation/evaluation.json").read_text())["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, names, key in (
        (
            axes[0],
            ["forward_medium", "backward_medium", "lateral_left", "lateral_right"],
            "xy_rmse_m_s",
        ),
        (axes[1], ["turn_left", "turn_right"], "yaw_rmse_rad_s"),
    ):
        x = np.arange(len(names))
        ax.bar(x - 0.18, [base[n][key] for n in names], width=0.36, label="previous B")
        ax.bar(
            x + 0.18, [new[n][key] for n in names], width=0.36, label="omnidirectional adaptation"
        )
        ax.set_xticks(x, names, rotation=25)
        ax.set_ylabel(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "directional_comparison.png")
    plt.close(fig)
    names = [
        f"{t}_L{level}_cmd0"
        for t in ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down")
        for level in range(1, 5)
    ]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    x = np.arange(len(names))
    for ax, key in zip(axes, ("xy_rmse_m_s", "fall"), strict=True):
        ax.plot(x, [base[n][key] for n in names], "o-", label="previous B")
        ax.plot(x, [new[n][key] for n in names], "o-", label="adaptation")
        ax.set_xticks(x, names, rotation=60)
        ax.set_ylabel(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "terrain_comparison.png")
    plt.close(fig)
    rows = list(csv.DictReader((root / "stageB/train_log.csv").open()))
    x = [int(r["iteration"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, keys in zip(
        axes.flat,
        (("wm_loss", "wm_err8"), ("sim_rew", "imag_rew"), ("v_loss",), ("pi_gradnorm",)),
        strict=True,
    ):
        for k in keys:
            ax.plot(x, [float(r[k]) for r in rows], label=k)
        ax.set_xlabel("absolute outer iteration")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)
    prog = [json.loads(s) for s in (root / "stageB/progress.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r["new_iterations"] for r in prog], [r["saturation"] for r in prog])
    axes[0].set_ylabel("fixed-input action saturation")
    axes[1].plot([r["new_iterations"] for r in prog], [r["actor_drift_rmse"] for r in prog])
    axes[1].set_ylabel("fixed-input action drift RMSE")
    fig.tight_layout()
    fig.savefig(out / "actor_outputs.png")
    plt.close(fig)
    c = json.loads((root / "stageB/coverage.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 4))
    modes = c["modes"]
    real = np.array(c["transitions"]).sum(0)
    wm = np.array(c["wm_sample_modes"])
    pi = np.array(c["policy_start_modes"])
    x = np.arange(6)
    for offset, label, val in (
        (-0.25, "real transitions", real),
        (0, "WM valid frames", wm),
        (0.25, "actual imagination starts", pi),
    ):
        ax.bar(x + offset, val / max(val.sum(), 1), width=0.25, label=label)
    ax.set_xticks(x, modes)
    ax.set_ylabel("measured fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "replay_training_coverage.png")
    plt.close(fig)
    paths = [root / "wm_baseline/diagnostics.json", root / "wm_stageB/diagnostics.json"]
    if all(p.exists() for p in paths):
        values = [json.loads(p.read_text()) for p in paths]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, key in zip(
            axes.flat,
            (
                "joint_position_rad",
                "joint_velocity_rad_s",
                "position_m",
                "velocity_m_s",
                "attitude_rad",
                "foot_position_m",
            ),
            strict=True,
        ):
            for label, v in zip(("previous B", "adaptation"), values, strict=True):
                ax.plot(
                    [1, 8, 16, 32],
                    [
                        np.mean([r[key] for r in v["prediction"] if r["horizon"] == h])
                        for h in (1, 8, 16, 32)
                    ],
                    "o-",
                    label=label,
                )
            ax.set_title(key)
            ax.set_xlabel("horizon")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "heldout_prediction.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4))
        for label, v in zip(("previous B", "adaptation"), values, strict=True):
            g = [
                r
                for r in v["gradients"]
                if r["quantity"] == "joint_velocity" and r["epsilon"] == 0.03
            ]
            ax.plot(np.arange(len(g)), [r["sign_agreement"] for r in g], "o-", label=label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("joint velocity Jacobian sign agreement")
        ax.set_xticks(np.arange(len(g)), [r["scenario"] for r in g], rotation=70)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "action_gradients.png")
        plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
