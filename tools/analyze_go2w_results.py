"""Summarize corrected real-physics reassessment and checkpoint comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unisupermimic.tasks.velocity.go2w.curriculum import DOMAINS, SINGLE_HEIGHTS, STAIR_HEIGHTS
from unisupermimic.tasks.velocity.go2w.experiment import write_json


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=True)

    def read(path):
        return json.loads(path.read_text())

    reassessment = run / "reassessment"
    final = read(reassessment / "final_11yaw.json")
    training = read(run / "training_summary.json")
    checkpoints = []
    for p in sorted(reassessment.glob("*corrected.json")):
        d = read(p)
        checkpoints.append(
            dict(iteration=d["trained_iterations"], label=p.stem, by_domain=d["by_domain"])
        )
    checkpoints.append(
        dict(
            iteration=final["trained_iterations"],
            label="final",
            by_domain=read(reassessment / "final_physical.json")["by_domain"],
        )
    )
    checkpoints.sort(key=lambda v: v["iteration"])
    axes = []
    flat_order = ("stop", "forward", "backward", "left", "right", "yaw_left", "yaw_right")
    for name in flat_order:
        rows = [v for v in final["rows"] if v["name"] == "flat_" + name]
        axes.append(
            dict(
                name=name,
                command=rows[0]["goal"],
                episodes=len(rows),
                passed=sum(v["passed"] for v in rows),
                fall_count=sum(v["fall"] for v in rows),
                mean_actual=np.mean([v["mean_actual_vx_vy_yaw"] for v in rows], axis=0).tolist(),
                mean_rmse=np.mean([v["rmse_vx_vy_yaw"] for v in rows], axis=0).tolist(),
                mean_duration=np.mean([v["duration_seconds"] for v in rows]).item(),
            )
        )
    stairs = []
    for domain in range(4, 8):
        for level in range(10):
            rows = [v for v in final["rows"] if v["domain"] == domain and v["level"] == level]
            stairs.append(
                dict(
                    domain=DOMAINS[domain],
                    level=level + 1,
                    height_cm=round((SINGLE_HEIGHTS if domain < 6 else STAIR_HEIGHTS)[level] * 100),
                    episodes=len(rows),
                    passed=sum(v["passed"] for v in rows),
                    falls=sum(v["fall"] for v in rows),
                )
            )
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=True, mmap=True)
    scopes = saved["extra_state"]["actual_actor_start_scopes"]
    starts = dict(
        by_domain=scopes.sum(-1).tolist(),
        percent=(scopes.sum(-1) / scopes.sum() * 100).tolist(),
        domain_level=scopes.tolist(),
        final_skill_levels=(saved["curriculum"]["tensors"]["skill_levels"] + 1).tolist(),
    )
    del saved
    report = dict(
        training=training,
        evaluation_protocol=final["evaluation_protocol"],
        scenarios=len(final["rows"]),
        yaw_values_rad=list(final["yaw_map"].values()),
        uncertainty="11 deterministic initial yaw conditions; not independent stochastic trials or a confidence bound",
        by_domain=final["by_domain"],
        flat_axes=axes,
        stairs=stairs,
        checkpoint_comparison=checkpoints,
        actor_starts=starts,
        validation=read(reassessment / "evaluation_validation.json"),
    )
    other = reassessment / "iteration_010183_11yaw.json"
    if other.exists():
        report["earlier_same_matrix"] = read(other)["by_domain"]
    write_json(output / "summary.json", report)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    names = [
        "flat",
        "rough",
        "slope up",
        "slope down",
        "step up",
        "step down",
        "stairs up",
        "stairs down",
    ]
    rates = [final["by_domain"][d]["success_rate"] * 100 for d in DOMAINS]
    falls = [final["by_domain"][d]["fall_rate"] * 100 for d in DOMAINS]
    x = np.arange(8)
    ax[0, 0].bar(x - 0.18, rates, width=0.36, label="Pass")
    ax[0, 0].bar(x + 0.18, falls, width=0.36, label="Fall")
    ax[0, 0].set_xticks(x, names, rotation=25)
    ax[0, 0].set_ylim(0, 105)
    ax[0, 0].set_ylabel("% of fixed conditions")
    ax[0, 0].legend()
    ax[0, 0].set_title("Final policy: 847 physical episodes, full t=0 clocks")
    for i, axis in enumerate(["vx", "vy", "yaw"]):
        ax[0, 1].bar(
            np.arange(7) + (i - 1) * 0.25, [r["mean_rmse"][i] for r in axes], width=0.25, label=axis
        )
    ax[0, 1].set_xticks(np.arange(7), flat_order, rotation=25)
    ax[0, 1].legend()
    ax[0, 1].set_title("Mean per-episode tracking RMSE (m/s, m/s, rad/s)")
    image = np.array(
        [
            [v["passed"] / v["episodes"] * 100 for v in stairs if v["domain"] == DOMAINS[d]]
            for d in range(4, 8)
        ]
    )
    im = ax[1, 0].imshow(image, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax[1, 0].set_yticks(range(4), names[4:])
    ax[1, 0].set_xticks(range(10), range(1, 11))
    for row in range(4):
        for col in range(10):
            ax[1, 0].text(
                col,
                row,
                f"{image[row, col]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    ax[1, 0].set_xlabel("Level (single 3..22cm; stairs 3..15cm riser)")
    ax[1, 0].set_title("Physical traversal pass %")
    fig.colorbar(im, ax=ax[1, 0])
    for domain in ("flat", "rough", "slope_up", "single_step_down"):
        ax[1, 1].plot(
            [v["iteration"] for v in checkpoints],
            [v["by_domain"][domain]["success_rate"] * 100 for v in checkpoints],
            marker="o",
            label=domain,
        )
    ax[1, 1].set_title("Intermediate checkpoints: same 3-yaw matrix (231 episodes)")
    ax[1, 1].set_xlabel("Training iteration")
    ax[1, 1].set_ylabel("Pass %")
    ax[1, 1].legend()
    fig.savefig(output / "physical_reassessment.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(dict(episodes=len(final["rows"]), flat=rates[0], step_down=rates[5]), indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
