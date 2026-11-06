"""Render figures from the committed audit JSON; no simulator or policy loads."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE.parents[1] / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def main():
    names = ["new_cold_2p5h", "old_chain_2p95h", "old_actor_2p96h", "old_final_chain_4p55h"]
    labels = ["Cold\n2.50h", "Old chain\n2.95h", "Old actor 2.96h\n(chain 3.21h)", "Old chain\n4.55h"]
    rows = [json.loads((HERE / f"go2_{n}.json").read_text())["real_mujoco"] for n in names]
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Go2: saved training lineage and matched nominal re-evaluation", fontsize=17, weight="bold")
    for cmd, color in [(0, "#2878b5"), (1, "#ed7d31")]:
        axes[0, 0].plot(x, [r[cmd]["mean_vx"] for r in rows], "o-", color=color,
                         label=f"command vx={rows[0][cmd]['command_body_xy_yaw'][0]:.1f}")
        axes[0, 0].axhline(rows[0][cmd]["command_body_xy_yaw"][0], color=color, ls="--", alpha=.5)
        axes[0, 1].plot(x, [100 * r[cmd]["diagonal_support_fraction"] for r in rows], "o-", color=color,
                         label=f"vx={rows[0][cmd]['command_body_xy_yaw'][0]:.1f}")
    axes[0, 0].set(title="Similar speed can hide very different gait", ylabel="Mean body vx (m/s)")
    axes[0, 1].set(title="Clear diagonal support appears later", ylabel="Geometric diagonal support (%)", ylim=(-2, 60))
    for cmd, label in enumerate(["vx=0.3", "vx=0.5", "vy=0.3", "yaw=0.4"]):
        axes[1, 0].plot(x, [r[cmd]["stance_slip_rms_m_s"] for r in rows], "o-", label=label)
    axes[1, 0].set(title="Slip improves substantially after the ~3h snapshots", ylabel="Stance slip RMS (m/s)")
    height = np.array([r[0]["mean_episode_peak_foot_clearance_m"] for r in rows]) * 100
    im = axes[1, 1].imshow(height, cmap="YlGnBu", vmin=0, vmax=7, aspect="auto")
    axes[1, 1].set(title="vx=0.3: asymmetric lift versus four-foot lift", xticks=range(4), xticklabels=["FL", "FR", "RL", "RR"],
                    yticks=range(4), yticklabels=[s.replace("\n", " ") for s in labels])
    for i in range(4):
        for j in range(4):
            axes[1, 1].text(j, i, f"{height[i,j]:.2f}", ha="center", va="center", color="white" if height[i,j] > 4 else "black")
    fig.colorbar(im, ax=axes[1, 1], label="Mean episode peak clearance (cm)")
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks(x, labels); ax.legend(fontsize=9); ax.grid(axis="y", alpha=.2)
    fig.supxlabel("16 nominal envs / 16 ended episodes per command / 10s / seed2026; no noise or DR. FK gait proxies; one training seed per lineage.", fontsize=9)
    fig.savefig(FIG / "GO2_TRAINING_HISTORY_MATCHED_20261106.png", dpi=170)
    plt.close(fig)

    snapshots = json.loads((HERE / "m1_fixed_input_snapshots.json").read_text())["snapshots"]
    scalars = json.loads((HERE / "m1_selected_training_scalars.json").read_text())
    clocks = {r["step"]: r["value"] for r in scalars["time/hours"]}
    gradients = json.loads((HERE / "m1_gradient_components.json").read_text())["snapshots"]
    feedback = json.loads((HERE / "m1_action_history_feedback.json").read_text())["results"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("M1: frozen-checkpoint saturation and local gradient diagnostics", fontsize=17, weight="bold")
    axes[0, 0].plot([r["hours"] for r in snapshots], [r["fixed_input_mean_abs_action"] for r in snapshots], label="fixed nominal input, zero action history")
    a = scalars["policy/action_abs_mean"]
    axes[0, 0].plot([clocks[r["step"]] for r in a], [r["value"] for r in a], label="training imagined rollout")
    axes[0, 0].axhline(.95, ls="--", color="gray")
    axes[0, 0].set(title="Input/history matters: not a constant +/-1 network", xlabel="SM training hours", ylabel="Mean |action|", ylim=(-.02, 1.03))
    chosen = {"ckpt_wall_060m.pt": "60 min", "ckpt_wall_090m.pt": "90 min", "ckpt_wall_100m.pt": "100 min", "ckpt_latest.pt": "final"}
    for r in feedback:
        if r["checkpoint"] in chosen:
            axes[0, 1].plot([s["step"] for s in r["rows"]], [s["abs_action_mean"] for s in r["rows"]], "o-", label=chosen[r["checkpoint"]])
    axes[0, 1].set(title="Artificial test: previous-action feedback alone amplifies", xlabel="Calls; physical input held fixed", ylabel="Mean |action|", ylim=(-.02, 1.03))
    times = [r["hours"] for r in gradients]
    for key, label in [("dense", "process reward"), ("critic", "critic bootstrap"), ("terminal", "endpoint distance")]:
        axes[1, 0].semilogy(times, [r["components"][key]["parameter_gradient_norm"] for r in gradients], "o-", label=label)
        axes[1, 1].plot(times, [r["components"][key]["cosine_with_total"] for r in gradients], "o-", label=label)
    axes[1, 0].set(title="Local gradient size: endpoint is not dominant here", xlabel="Snapshot training hours", ylabel="Actor parameter gradient L2 (J/H)")
    axes[1, 1].axhline(0, color="gray", ls="--")
    axes[1, 1].set(title="Components can oppose each other", xlabel="Snapshot training hours", ylabel="Cosine with total gradient", ylim=(-1.1, 1.1))
    for ax in axes.flatten():
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.supxlabel("Gradient probe: 4 controlled nominal starts, H32, frozen WM/Actor/Critic, no optimizer steps. Not the historical Replay batches or causal attribution.", fontsize=9)
    fig.savefig(FIG / "M1_SATURATION_GRADIENT_DIAGNOSTICS_20261106.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
