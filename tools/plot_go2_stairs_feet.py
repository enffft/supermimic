"""Reclassify actual solver contacts, summarize contact-delimited swings and plot.

Native MjData rollouts are diagnostic evidence, not substitutes for UniLab fixed
policy evaluation. Riser contacts require BOTH a side normal and a real edge.
"""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state


def analyze(folder):
    saved = torch.load(folder / "trajectories.pt", weights_only=True)
    old = json.loads((folder / "diagnostics.json").read_text())
    names = old["geom_names"]
    cases = saved["cases"]
    index = {x[0]: i for i, x in enumerate(cases)}
    traces = {t["seed"]: t for t in saved["traces"]}
    contacts = {}
    for seed, t in traces.items():
        shape = t["data"]["contact"].shape
        contacts[seed] = dict(
            riser=torch.zeros(shape, dtype=torch.bool),
            body=torch.zeros(shape[:2], dtype=torch.bool),
            nonfoot=torch.zeros(shape[:2], dtype=torch.bool),
        )
    with gzip.open(folder / "contacts.jsonl.gz", "rt") as f:
        for line in f:
            c = json.loads(line)
            if np.linalg.norm(c["force_contact"][:3]) <= 2:
                continue
            seed, i, k = c["seed"], index[c["scenario"]], c["step"]
            j = c["foot"]
            if j >= 0:
                x = c["position"][0] - float(traces[seed]["entries"][i, 0])
                near = any(
                    abs(x - e[0]) < 0.06 and abs(e[2] - e[1]) > 0.03
                    for e in traces[seed]["edges"][i]
                )
                contacts[seed]["riser"][k, i, j] |= abs(c["normal"][2]) < 0.7 and near
            else:
                contacts[seed]["nonfoot"][k, i] = True
                pair = [(names[g] or "") for g in (c["geom1"], c["geom2"])]
                contacts[seed]["body"][k, i] |= any("base" in n or "head" in n for n in pair)
    summaries, peaks, edge_passes = [], [], []
    first_risers = []
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    for seed, tr in traces.items():
        d = tr["data"]
        feet, vel = d["feet"], d["foot_velocity"]
        feet_local = feet.clone()
        feet_local[..., :2] -= tr["entries"][None, :, None, :]
        for i, (name, _, level) in enumerate(cases):
            active = d["active"][:, i]
            stop = int(active.sum())
            rising = [e for e in tr["edges"][i] if e[2] - e[1] > 0.03]
            if rising:
                edge = rising[0]
                x = feet_local[:stop, i, :2, 0]
                base_x = unflatten_state(d["s1"][:stop, i])[0][:, 0] - tr["entries"][i, 0]
                approach = (
                    (torch.arange(stop)[:, None] > 25)
                    & (x > edge[0] - 0.12)
                    & (x < edge[0] + 0.06)
                    & (base_x[:, None] < edge[0])
                )
                margin = feet[:stop, i, :2, 2] - 0.022 - edge[2]
                first_risers.append(
                    dict(
                        scenario=name,
                        seed=seed,
                        max_front_margin_m=float(margin[approach].max())
                        if approach.any()
                        else None,
                        above_next_tread_fraction=float((margin[approach] > 0.005).float().mean())
                        if approach.any()
                        else None,
                        approach_samples=int(approach.sum()),
                    )
                )
            pk = []
            for j in range(4):
                airborne = (~d["contact"][:stop, i, j]).numpy()
                transitions = np.diff(np.r_[False, airborne, False].astype(int))
                for a, b in zip(
                    np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
                ):
                    if a < 25 or b >= stop or b - a < 2:
                        continue
                    # Lift relative to last support surface, not ground that jumps at a riser.
                    ground = float(feet[a - 1, i, j, 2] - 0.022 - d["clearance"][a - 1, i, j])
                    h = float(feet[a:b, i, j, 2].max() - 0.022 - ground)
                    pk.append(h)
                    peaks.append(
                        dict(
                            scenario=name,
                            seed=seed,
                            foot=j,
                            peak_m=h,
                            duration_s=(b - a) * 0.02,
                            landing_vz=float(vel[b, i, j, 2]),
                        )
                    )
                for edge_x, z0, z1 in tr["edges"][i]:
                    if abs(z1 - z0) < 0.03:
                        continue
                    x = feet_local[:stop, i, j, 0]
                    ix = ((x[:-1] < edge_x) & (x[1:] >= edge_x)).nonzero().flatten() + 1
                    for k in ix.tolist():
                        if k < 25:
                            continue
                        bottom = float(feet[k, i, j, 2] - 0.022)
                        edge_passes.append(
                            dict(
                                scenario=name,
                                seed=seed,
                                foot=j,
                                edge_x=edge_x,
                                time=k * 0.02,
                                margin_m=bottom - max(z0, z1),
                                clear=bottom >= max(z0, z1) + 0.005,
                            )
                        )
            pos = unflatten_state(d["s1"][:stop, i])[0]
            progress = float(pos[-1, 0] - unflatten_state(d["s"][0, i])[0][0])
            yaw_rmse = float((unflatten_state(d["s1"][:stop, i])[3][:, 2]).square().mean().sqrt())
            summaries.append(
                dict(
                    scenario=name,
                    seed=seed,
                    level=level,
                    progress_m=progress,
                    peak_mean_m=float(np.mean(pk)) if pk else None,
                    peak_p50_m=float(np.median(pk)) if pk else None,
                    peak_p90_m=float(np.quantile(pk, 0.9)) if pk else None,
                    confirmed_riser_foot_steps=int(contacts[seed]["riser"][:stop, i].sum()),
                    actual_base_head_contact_steps=int(contacts[seed]["body"][:stop, i].sum()),
                    other_leg_contact_steps=int(contacts[seed]["nonfoot"][:stop, i].sum()),
                    action_saturation=float((d["a"][:stop, i].abs() > 0.98).float().mean()),
                    torque_limit_fraction=float(
                        (
                            d["torque"][:stop, i].abs()
                            >= torch.tensor([23.7, 23.7, 45.43] * 4) * 0.98
                        )
                        .float()
                        .mean()
                    ),
                    yaw_rmse_rad_s=yaw_rmse,
                )
            )
            if seed != min(traces):
                continue
            time = np.arange(stop) * 0.02
            fig, axs = plt.subplots(2, 2, figsize=(13, 8))
            for j, leg in enumerate(("FL", "FR", "RL", "RR")):
                axs[0, 0].plot(time, d["clearance"][:stop, i, j], label=leg)
                axs[0, 1].plot(time, feet_local[:stop, i, j, 0], label=leg)
                axs[1, 0].plot(feet_local[:stop, i, j, 0], feet[:stop, i, j, 2] - 0.022, label=leg)
                axs[1, 1].plot(feet_local[:stop, i, j, 0], feet_local[:stop, i, j, 1], label=leg)
            axs[1, 0].plot(tr["section_x"], tr["section_z"][i], c="black", label="compiled ground")
            axs[0, 0].set(xlabel="time (s)", ylabel="sphere bottom / local ground (m)")
            axs[0, 1].set(xlabel="time (s)", ylabel="foot entry-relative x (m)")
            axs[1, 0].set(
                xlabel="entry-relative x (m)",
                ylabel="world bottom z (m)",
                xlim=(-0.4, max(1.5, float(feet_local[:stop, i, :, 0].max()) + 0.2)),
            )
            axs[1, 1].set(xlabel="entry-relative x (m)", ylabel="entry-relative y (m)")
            for ax in axs.flatten():
                ax.legend()
                ax.grid(alpha=0.2)
            fig.suptitle(f"{name} / seed {seed} / native solver diagnostic")
            fig.tight_layout()
            fig.savefig(out / f"{name}_trajectory.png")
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        [[p["peak_m"] for p in peaks if p["scenario"] == name] for name, _, _ in cases],
        tick_labels=[x[0] for x in cases],
        showfliers=False,
    )
    ax.tick_params(axis="x", rotation=30)
    ax.set(ylabel="swing peak above takeoff support ground (m)")
    fig.tight_layout()
    fig.savefig(out / "swing_peaks.png")
    plt.close(fig)
    result = dict(
        method="native solver; contact-delimited complete swings after .5s; force>2N; riser normal plus compiled edge; base/head distinct from legs",
        summaries=summaries,
        swing_peaks=peaks,
        edge_passes=edge_passes,
        first_riser_approaches=first_risers,
        calibration={
            k: old[k]
            for k in (
                "unilab_one_step_max_error",
                "foot_fk_position_max_error_m",
                "foot_fk_velocity_max_error_m_s",
            )
        },
    )
    (folder / "corrected_analysis.json").write_text(json.dumps(result, indent=2))
    (folder / "first_riser_analysis.json").write_text(json.dumps(first_risers, indent=2))
    print(
        json.dumps(
            {
                name: {
                    k: float(
                        np.mean(
                            [s[k] for s in summaries if s["scenario"] == name and s[k] is not None]
                        )
                    )
                    for k in (
                        "progress_m",
                        "peak_mean_m",
                        "confirmed_riser_foot_steps",
                        "actual_base_head_contact_steps",
                        "action_saturation",
                        "torque_limit_fraction",
                    )
                }
                for name, _, _ in cases
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    analyze(p.parse_args().folder)
