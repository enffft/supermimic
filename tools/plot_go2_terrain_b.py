"""Build report figures from recorded runs; never launches training."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    out = r / "figures"
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader((r / "warmstart/train_log.csv").open()))
    it = np.array([int(x["iteration"]) for x in rows])
    labels = ["short128", "000500", "001000", "001500", "002000"]
    updates = [128, 2000, 4000, 6000, 8000]
    preds = [read(r / f"validation_{x}/prediction.json") for x in labels]
    accepts = [read(r / f"validation_{x}/acceptance.json") for x in labels]

    def rms(data, metric, h=32):
        q = [x for x in data if x["horizon"] == h and x["level"] in (1, 2) and x["samples"]]
        return np.sqrt(sum(x["samples"] * x[metric] ** 2 for x in q) / sum(x["samples"] for x in q))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(it, [float(x["wm_loss"]) for x in rows])
    axs[0].set(xlabel="Outer iteration", ylabel="H8 training normalized loss", yscale="log")
    axs[1].plot(updates, [rms(x, "joint_velocity_rad_s", 8) ** 2 for x in preds], marker="o")
    axs[1].set(
        xlabel="WM optimizer updates",
        ylabel="Held-out H8 qdot MSE (rad/s)^2",
        title="Fixed held-out trajectories; same starts",
    )
    fig.tight_layout()
    fig.savefig(out / "wm_loss.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [
        ("joint_position_rad", "Joint position (rad)"),
        ("joint_velocity_rad_s", "Joint velocity (rad/s)"),
        ("position_m", "Base position (m)"),
        ("velocity_m_s", "Base velocity (m/s)"),
        ("attitude_rad", "Attitude (rad)"),
        ("foot_position_m", "Foot position (m)"),
    ]
    for ax, (metric, title) in zip(axs.flat, metrics):
        for horizon in (1, 8, 16, 32):
            ax.plot(
                updates, [rms(x, metric, horizon) for x in preds], marker=".", label=f"H{horizon}"
            )
        ax.set(xlabel="WM updates", ylabel="RMSE", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prediction_horizons.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for field, name in [
        ("mean_probe_sign_agreement", "Mean"),
        ("minimum_probe_sign_agreement", "Worst probe"),
    ]:
        axs[0].plot(updates, [x[field] for x in accepts], marker="o", label=name)
    axs[0].axhline(0.8, color="r", ls="--", label="Per-probe acceptance")
    axs[0].set(xlabel="WM updates", ylabel="qdot Jacobian sign agreement")
    axs[0].legend()
    axs[1].plot(updates, [x["mean_probe_normalized_error"] for x in accepts], marker="o")
    axs[1].set(xlabel="WM updates", ylabel="Mean normalized Jacobian error")
    fig.tight_layout()
    fig.savefig(out / "action_gradients.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(it, [float(x["sim_rew"]) for x in rows])
    axs[0].set(
        xlabel="Outer iteration",
        ylabel="Real reward / max step reward",
        title="Frozen actor; changing data distribution",
    )
    axs[1].text(
        0.5,
        0.5,
        "Actor and Critic were never unfrozen.\nNo training imagined-return series exists.\nCSV zero placeholders are not measurements.",
        ha="center",
        va="center",
    )
    axs[1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "real_reward.png", dpi=150)
    plt.close(fig)
    checkpoints = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    distributions = []
    saturation = []
    for n in checkpoints:
        ck = torch.load(
            r / f"warmstart/ckpt_{n:06}.pt", map_location="cpu", weights_only=True, mmap=True
        )
        levels = ck["curriculum"]["tensors"]["terrain_level"]
        distributions.append(torch.bincount(levels, minlength=5).numpy() / len(levels))
        ap = ck["replay"]["data"]["a_prev"][::16, ::4]
        saturation.append(float((ap.abs() > 0.98).float().mean()))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].stackplot(checkpoints, np.array(distributions).T, labels=[f"L{i}" for i in range(5)])
    axs[0].set(
        xlabel="WM-only outer iteration",
        ylabel="Environment fraction",
        title="WM data coverage; NOT skill mastery",
    )
    axs[0].legend(loc="upper right")
    history = read(r / "skill_curriculum_physics/curriculum_history.json")
    for level in range(5):
        axs[1].plot(
            np.arange(1, 11), [x["mastery_histogram"][level] for x in history], label=f"L{level}"
        )
    axs[1].set(
        xlabel="10s episode blocks (separate evaluation)",
        ylabel="Environment count",
        title="Old actor, real physics, no optimizer",
    )
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(out / "terrain_levels.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(checkpoints, saturation, marker="o")
    ax.set(
        xlabel="WM-only outer iteration",
        ylabel="Sampled previous clean action |a|>0.98",
        title="Actor weights unchanged; replay-state distribution varies",
    )
    fig.tight_layout()
    fig.savefig(out / "actor_saturation.png", dpi=150)
    plt.close(fig)
    fixed = read(r / "fixed_evaluation_verified/fixed_evaluation.json")
    types = ["rough", "slope_up", "slope_down", "stairs_up", "stairs_down"]
    aggregate = []
    for kind in ["flat", *types]:
        for level in [0] if kind == "flat" else range(1, 5):
            q = [x for x in fixed if x["level"] == level and (level == 0 or x["type"] == kind)]
            row = dict(
                type=kind,
                level=level,
                episodes=len(q),
                passed=sum(x["passed"] for x in q),
                falls=sum(x["fall"] for x in q),
            )
            for key in [
                "xy_rmse_m_s",
                "yaw_rmse_rad_s",
                "actual_progress_m",
                "commanded_progress_m",
                "progress_ratio",
                "mean_local_height_m",
                "action_saturation_fraction",
                "max_lateral_m",
                "actual_body_vx_m_s",
                "measured_nonfoot_contact_sample_fraction",
                "proxy_foot_slip_m_s",
            ]:
                row[key] = float(np.mean([x[key] for x in q]))
            aggregate.append(row)
    (r / "fixed_evaluation_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field in zip(axs, ["passed", "falls", "progress_ratio"]):
        arr = np.array(
            [
                [
                    next(x for x in aggregate if x["type"] == kind and x["level"] == lev)[field]
                    / (3 if field != "progress_ratio" else 1)
                    for lev in range(1, 5)
                ]
                for kind in types
            ]
        )
        im = ax.imshow(arr, vmin=0, vmax=1, cmap="viridis")
        fig.colorbar(im, ax=ax)
        for y in range(5):
            for x in range(4):
                ax.text(
                    x,
                    y,
                    f"{arr[y, x]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if arr[y, x] < 0.5 else "black",
                )
        ax.set(
            xticks=range(4),
            xticklabels=[1, 2, 3, 4],
            yticks=range(5),
            yticklabels=types,
            xlabel="Fixed level",
            title=field + " (3 seeds)",
        )
    fig.tight_layout()
    fig.savefig(out / "fixed_results.png", dpi=150)
    plt.close(fig)
    (r / "prediction_summary.json").write_text(
        json.dumps(
            [
                dict(updates=n, **{key: rms(x, key) for key, _ in metrics})
                for n, x in zip(updates, preds)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
