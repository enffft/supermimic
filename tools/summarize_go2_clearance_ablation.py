"""Produce paired physical, prediction, coverage plots and a fail-closed longrun decision."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unisupermimic.tasks.velocity.go2.speed_curriculum import imagination_scope_admission


def summarize(root):
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    variants = {"A0": "baseline", "A1": "gaussian", "A2": "soft_deficit"}
    source_path = root.parent / "stairs_omni_continuation_v2/best.pt"
    source = torch.load(source_path, weights_only=True, map_location="cpu", mmap=True)
    saved_eval = json.loads(
        (
            root.parent / "stairs_omni_continuation_v2/selected_evaluation/evaluation.json"
        ).read_text()
    )
    assert (
        saved_eval["checkpoint_sha256"]
        == hashlib.file_digest(source_path.open("rb"), "sha256").hexdigest()
    )
    physics = {"Source": saved_eval["aggregate"]}
    metrics = {}
    gates = {}
    diagnostics = {}
    for tag, name in variants.items():
        run = root / name / "trial"
        values = torch.load(run / "last.pt", weights_only=True, map_location="cpu", mmap=True)
        for key in (
            "actor",
            "joints",
            "action",
            "model_sha256",
            "state",
            "terrain_fingerprint",
            "sim_dt",
            "ctrl_dt",
            "commands",
            "terminations",
        ):
            assert values["task_contract"][key] == source["task_contract"][key], key
        evaluation = json.loads((root / "evaluation" / tag / "evaluation.json").read_text())
        for key in ("seeds", "seconds", "maneuver_seconds", "safe_spawn", "initial_perturbation"):
            assert evaluation[key] == saved_eval[key], key
        physics[tag] = evaluation["aggregate"]
        metrics[tag] = json.loads((run / "summary.json").read_text())
        metrics[tag]["checkpoint_sha256"] = hashlib.file_digest(
            (run / "last.pt").open("rb"), "sha256"
        ).hexdigest()
        diags = {
            speed: json.loads(
                (root / "diagnostics" / f"final_{tag}_S{speed}" / "diagnostics.json").read_text()
            )
            for speed in (0, 1)
        }
        diagnostics[tag] = diags[0]
        mask, details = imagination_scope_admission(
            diags, values["command_sampler"]["stairs_levels"].tolist()
        )
        gates[tag] = dict(mask=mask.tolist(), details=details)
    diagnostics["Source"] = json.loads(
        (root / "diagnostics/source_S0/diagnostics.json").read_text()
    )
    stairkeys = [f"stairs_{d}_L{l}_cmd0" for d in ("up", "down") for l in (4, 5, 6, 7)]
    tags = ["Source", "A0", "A1", "A2"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    for ax, key in zip(axs, ("crossing_passed", "central_platform_reached", "fall")):
        for i, tag in enumerate(tags):
            ax.bar(
                np.arange(8) + (i - 1.5) * 0.2,
                [physics[tag][k][key] for k in stairkeys],
                width=0.2,
                label=tag,
            )
        ax.set_ylabel(key)
        ax.set_ylim(0, 1.1)
        ax.legend(ncol=4)
        ax.grid(axis="y", alpha=0.2)
    axs[-1].set_xticks(range(8), [k.replace("stairs_", "").replace("_cmd0", "") for k in stairkeys])
    fig.tight_layout()
    fig.savefig(figs / "physical_stairs.png")
    plt.close(fig)
    flatkeys = [
        "forward_medium",
        "backward_medium",
        "lateral_left",
        "lateral_right",
        "turn_left",
        "turn_right",
        "mixed",
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axs, ("xy_rmse_m_s", "yaw_rmse_rad_s")):
        for tag in tags:
            ax.plot(flatkeys, [physics[tag][k][key] for k in flatkeys], "o-", label=tag)
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "flat_retention.png")
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(
        axs.flat,
        (
            "position_m",
            "velocity_m_s",
            "attitude_rad",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "foot_position_m",
        ),
    ):
        for tag in tags:
            ds = diagnostics[tag]["prediction"]
            ax.plot(
                [1, 8, 16, 32],
                [np.mean([r[key] for r in ds if r["horizon"] == h]) for h in (1, 8, 16, 32)],
                "o-",
                label=tag,
            )
        ax.set(xlabel="horizon", ylabel=key)
        ax.grid(alpha=0.2)
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figs / "heldout_prediction.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for tag in tags:
        ds = diagnostics[tag]["gradients"]
        vals = []
        for l in (4, 5, 6, 7):
            gs = [
                g["sign_agreement"]
                for g in ds
                if g["scenario"].startswith(f"stairs_up_L{l}_S0_")
                and g["quantity"] == "joint_velocity"
                and g["epsilon"] == 0.03
            ]
            vals.append(min(gs))
        ax.plot([4, 5, 6, 7], vals, "o-", label=tag)
    ax.axhline(0.8, c="k", ls="--", label="each probe threshold")
    ax.set(xlabel="stairs up level", ylabel="minimum qd sign agreement")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs / "contact_gradient_gate.png")
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for tag, name in variants.items():
        records = list(csv.DictReader((root / name / "trial/train_log.csv").open()))
        steps = np.array([int(r["iteration"]) - 30807 for r in records])
        axs[0, 0].plot(steps, [float(r["wm_loss"]) for r in records], label=tag)
        normalizer = 5.25 if tag == "A1" else 5.2
        axs[0, 1].plot(
            steps, [float(r["sim_rew"]) * normalizer for r in records], label=f"{tag} real"
        )
        valid = np.array([int(r["iteration"]) > 30907 for r in records])
        axs[0, 1].plot(
            steps[valid],
            np.array([float(r["imag_rew"]) * normalizer for r in records])[valid],
            ls="--",
            label=f"{tag} imagined",
        )
        guard = [
            json.loads(x) for x in (root / name / "trial/guards.jsonl").read_text().splitlines()
        ]
        axs[1, 0].plot(
            [r["new_iteration"] for r in guard], [r["saturation"] for r in guard], label=tag
        )
        axs[1, 1].plot(
            [r["new_iteration"] for r in guard], [r["actor_rmse"] for r in guard], label=tag
        )
    for ax, label in zip(
        axs.flat,
        (
            "WM H8 training loss",
            "raw reward rate; imagined survival-weighted",
            "fixed replay action saturation",
            "action RMSE vs source",
        ),
    ):
        ax.set(xlabel="new outer iteration", ylabel=label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs / "training_curves.png")
    plt.close(fig)
    feet = {
        tag: json.loads((root / "diagnostics" / folder / "corrected_analysis.json").read_text())
        for tag, folder in [
            ("Source", "baseline_feet"),
            ("A0", "A0_feet"),
            ("A1", "gaussian_feet"),
            ("A2", "A2_feet"),
        ]
    }
    native = {}
    for tag, ds in feet.items():
        native[tag] = {}
        for case in (
            "flat",
            "stairs_up_L4",
            "stairs_up_L5",
            "stairs_up_L6",
            "stairs_up_L7",
            "stairs_down_L5",
        ):
            rs = [r for r in ds["summaries"] if r["scenario"] == case]
            native[tag][case] = {
                k: float(np.mean([r[k] for r in rs if r[k] is not None]))
                for k in (
                    "peak_mean_m",
                    "peak_p50_m",
                    "peak_p90_m",
                    "confirmed_riser_foot_steps",
                    "actual_base_head_contact_steps",
                    "action_saturation",
                    "torque_limit_fraction",
                )
            }
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, tag in enumerate(tags):
        axs[0].bar(
            np.arange(4) + (i - 1.5) * 0.2,
            [native[tag][f"stairs_up_L{l}"]["peak_mean_m"] for l in (4, 5, 6, 7)],
            width=0.2,
            label=tag,
        )
        axs[1].bar(
            np.arange(4) + (i - 1.5) * 0.2,
            [native[tag][f"stairs_up_L{l}"]["confirmed_riser_foot_steps"] for l in (4, 5, 6, 7)],
            width=0.2,
            label=tag,
        )
    for ax in axs:
        ax.set_xticks(range(4), ["8cm", "10cm", "12cm", "14cm"])
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    axs[0].set_ylabel("native diagnostic mean swing peak (m)")
    axs[1].set_ylabel("confirmed riser foot-control-steps / episode")
    fig.tight_layout()
    fig.savefig(figs / "native_swing_contacts.png")
    plt.close(fig)
    # Fail closed: no 90-minute run on a reward-only improvement or blocked high WM.
    decisions = {}
    for tag in ("A1", "A2"):
        skills = physics[tag]
        base = physics["Source"]
        retained = all(
            skills[f"stairs_up_L{l}_cmd0"]["crossing_passed"]
            >= base[f"stairs_up_L{l}_cmd0"]["crossing_passed"]
            for l in (4, 5)
        )
        improvement = any(
            skills[f"stairs_up_L{l}_cmd0"]["crossing_passed"]
            > base[f"stairs_up_L{l}_cmd0"]["crossing_passed"]
            for l in (6, 7)
        )
        wm_ok = all(gates[tag]["mask"][4][l] for l in (6, 7))
        decisions[tag] = dict(
            retained_8_10cm_up=retained,
            measured_12_14cm_crossing_improved=improvement,
            wm_up_L6_L7_admitted=wm_ok,
            longrun_allowed=bool(retained and improvement and wm_ok),
        )
    result = dict(
        source_sha256=hashlib.file_digest(source_path.open("rb"), "sha256").hexdigest(),
        physics=physics,
        training=metrics,
        final_wm_admission=gates,
        native_diagnostics=native,
        longrun_decision=decisions,
        selected_checkpoint=str(source_path),
        new_best=False,
        formal_trials_total_transitions=sum(m["new_transitions"] for m in metrics.values()),
        formal_trials_updates={
            k: sum(m["new_updates"][k] for m in metrics.values()) for k in ("wm", "actor", "critic")
        },
        summed_trial_wall_s=sum(m["wall_s"] for m in metrics.values()),
        notes=[
            "native diagnostic not identical to long-horizon UniLab trajectories",
            "one earlier aborted A0 setup run had 100 iterations/400 WM/0 Actor and is excluded from paired formal trials; log retained",
            "same interaction budget, source, seed, dynamics, exploration; independent evaluations overlapped some runs, wall speed is not a scaling benchmark",
        ],
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "formal_trials_total_transitions",
                    "formal_trials_updates",
                    "summed_trial_wall_s",
                    "longrun_decision",
                    "selected_checkpoint",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/stairs_clearance_B4"))
    summarize(p.parse_args().root)
