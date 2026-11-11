"""Plot measured source/selected/final physics and paired same-trajectory WM diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("results/stairs_omni_continuation_v2"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    tags = ["source", "selected", "final"]
    paths = ["extra40_baseline", "selected_evaluation", "extra40_final_evaluation"]
    physics = [
        json.loads((a.run / path / "evaluation.json").read_text())["aggregate"] for path in paths
    ]
    keys = [f"stairs_{d}_L{l}_cmd0" for d in ("up", "down") for l in (4, 5, 6, 7)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, metric in zip(
        axes, ["crossing_passed", "central_platform_reached", "fall"], strict=True
    ):
        for i, (tag, data) in enumerate(zip(tags, physics, strict=True)):
            ax.bar(
                np.arange(len(keys)) + (i - 1) * 0.25,
                [data[k][metric] for k in keys],
                width=0.25,
                label=tag,
            )
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(
        range(len(keys)), [k.replace("stairs_", "").replace("_cmd0", "") for k in keys]
    )
    fig.tight_layout()
    fig.savefig(a.output / "stairs_ability_comparison.png")
    plt.close(fig)
    diagnostics = [
        json.loads((a.run / path / "diagnostics.json").read_text())
        for path in [
            "extra40_fresh_wm_source",
            "selected_wm_diagnostics",
            "extra40_fresh_wm_candidate",
        ]
    ]
    fields = [
        "joint_position_rad",
        "joint_velocity_rad_s",
        "position_m",
        "velocity_m_s",
        "attitude_rad",
        "foot_position_m",
    ]
    units = [
        "joint position (rad)",
        "joint velocity (rad/s)",
        "base position (m)",
        "base velocity (m/s)",
        "attitude (rad)",
        "foot position (m)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, field, label in zip(axes.flat, fields, units, strict=True):
        for tag, data in zip(tags, diagnostics, strict=True):
            y = [
                np.mean([x[field] for x in data["prediction"] if x["horizon"] == h])
                for h in (1, 8, 16, 32)
            ]
            ax.plot([1, 8, 16, 32], y, "o-", label=tag)
        ax.set_xlabel("horizon")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(a.output / "paired_wm_prediction.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, fn in [("mean", np.mean), ("minimum", np.min)]:
        vals = [
            fn(
                [
                    g["sign_agreement"]
                    for g in d["gradients"]
                    if g["quantity"] == "joint_velocity"
                    and g["epsilon"] == 0.03
                    and g["sign_agreement"] is not None
                ]
            )
            for d in diagnostics
        ]
        ax.plot(tags, vals, "o-", label=metric)
    ax.axhline(0.8, color="k", ls="--", label="per-probe threshold")
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("qd action-gradient sign agreement")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.output / "paired_wm_gradient_sign.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
