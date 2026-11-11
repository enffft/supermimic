"""Plot measured continuation curves and export paired physics CSV; no imputed runs."""

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
    p.add_argument("--run", type=Path, default=Path("results/stairs_omni_continuation_v2"))
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument(
        "--segments",
        nargs="+",
        default=["training", "controlled", "level7_training", "extra40_training"],
    )
    p.add_argument("--monitor-segment", default="controlled")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for segment in a.segments:
        path = a.run / segment / "train_log.csv"
        if path.exists():
            with path.open() as f:
                rows.extend(csv.DictReader(f))
    if rows:
        x = [int(r["iteration"]) for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, key in zip(
            axes.flatten(),
            ("wm_loss", "wm_err8", "sim_rew", "imag_rew", "pi_gradnorm", "v_loss"),
            strict=True,
        ):
            y = [float(r[key]) if r[key] not in ("None", "") else np.nan for r in rows]
            ax.plot(x, y)
            ax.set_title(key)
            ax.set_xlabel("absolute iteration")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(a.output / "training_curves.png")
        plt.close(fig)
    curves = []
    for path in sorted((a.run / "training").glob("wm_warm_*/diagnostics.json")):
        try:
            it = int(path.parent.name.split("_")[2])
        except ValueError:
            continue
        d = json.loads(path.read_text())
        g = [
            v["sign_agreement"]
            for v in d["gradients"]
            if v["quantity"] == "joint_velocity"
            and v["epsilon"] == 0.03
            and v["sign_agreement"] is not None
        ]
        curves.append((it, min(g), np.mean(g), d))
    curves.sort(key=lambda r: r[0])
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r[0] for r in curves], [r[1] for r in curves], label="worst probe")
        axes[0].plot([r[0] for r in curves], [r[2] for r in curves], label="mean probe")
        axes[0].axhline(0.8, color="k", linestyle="--", label="per-pair admission")
        axes[0].set_ylabel("qd action gradient sign agreement")
        axes[0].legend()
        for h in (1, 8, 16, 32):
            axes[1].plot(
                [r[0] for r in curves],
                [
                    np.mean([v["velocity_m_s"] for v in r[3]["prediction"] if v["horizon"] == h])
                    for r in curves
                ],
                label=f"H{h}",
            )
        axes[1].set_ylabel("held-out base velocity RMSE (m/s)")
        axes[1].legend()
        for ax in axes:
            ax.set_xlabel("absolute iteration")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(a.output / "wm_holdout_and_gradients.png")
        plt.close(fig)
    prog = a.run / a.monitor_segment / "progress.jsonl"
    if prog.exists():
        rs = [json.loads(s) for s in prog.read_text().splitlines()]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, key in zip(
            axes, ("actor_drift_rmse", "saturation", "flat_speed_level"), strict=True
        ):
            ax.plot([r["iteration"] for r in rs], [r[key] for r in rs])
            ax.set_title(key)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(a.output / "actor_monitor.png")
        plt.close(fig)
    coverage = a.run / a.monitor_segment / "stairs_coverage.json"
    if coverage.exists():
        d = json.loads(coverage.read_text())
        count = np.array(d["count"]).sum((0, 1))
        triple = np.array(d["triples"]).sum((0, 1))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(4) - 0.15, count, width=0.3, label="all stairs")
        ax.bar(np.arange(4) + 0.15, triple, width=0.3, label="3 nonzero goals")
        ax.set_xticks(range(4), ["S0", "S1", "S2", "S3"])
        ax.set_ylabel("real transitions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.output / "speed_coverage.png")
        plt.close(fig)
    if a.evaluation:
        old = json.loads((a.baseline or a.run / "baseline/evaluation.json").read_text())[
            "aggregate"
        ]
        new = json.loads(a.evaluation.read_text())["aggregate"]
        fields = (
            "actual_vx_m_s",
            "actual_vy_m_s",
            "actual_wz_rad_s",
            "xy_rmse_m_s",
            "yaw_rmse_rad_s",
            "fall",
            "survival_s",
            "crossing_passed",
            "saturation",
            "geometric_slip_m_s",
            "outside_tile_fraction",
        )
        with (a.output / "fixed_evaluation.csv").open("w") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["scenario"] + [f"{tag}_{k}" for tag in ("old", "new") for k in fields],
            )
            w.writeheader()
            for name, r in new.items():
                if name not in old:
                    continue
                w.writerow(
                    dict(
                        scenario=name,
                        **{
                            f"{tag}_{k}": values[name][k]
                            for tag, values in (("old", old), ("new", new))
                            for k in fields
                        },
                    )
                )
        keys = [f"stairs_{d}_L4_omni{i}" for d in ("up", "down") for i in range(10)]
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        for ax, key in zip(axes, ("xy_rmse_m_s", "yaw_rmse_rad_s", "fall"), strict=True):
            ax.plot([old[k][key] for k in keys], "o-", label="source")
            ax.plot([new[k][key] for k in keys], "o-", label="continuation")
            ax.set_ylabel(key)
            ax.legend()
            ax.grid(alpha=0.2)
        axes[-1].set_xticks(range(len(keys)), [k.replace("stairs_", "") for k in keys], rotation=90)
        fig.tight_layout()
        fig.savefig(a.output / "level4_maneuvers.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
