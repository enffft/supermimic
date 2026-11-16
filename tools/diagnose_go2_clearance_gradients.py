"""Paired A0/A1/A2 rewards/action gradients on identical recorded states and H32 WM."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2.gait import Go2GaitCfg, clearance_shape
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(a.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    traces = torch.load(a.trajectories, weights_only=True, map_location="cpu")
    mdp = Go2VelocityMdp("cuda", TerrainHeightField(**traces["field"]).to("cuda"))
    entries = traces["traces"][0]["entries"][1:]
    mdp.clearance_boxes = torch.cat(
        (entries - entries.new_tensor([0.6, 4]), entries + entries.new_tensor([7.4, 4])), -1
    ).cuda()
    wm = WorldModel(
        hidden=tuple(saved["args"]["wm_hidden"]), num_joints=12, context_dim=16, terrain_dim=77
    ).cuda()
    wm.load_state_dict(saved["wm"])
    wm.requires_grad_(False)
    del saved
    rows = []
    labels = []
    for tr in traces["traces"]:
        d = tr["data"]
        for i, (name, *_) in enumerate(traces["cases"]):
            for t in (40, 80, 140, 210, 500):
                if not d["done"][t : t + 32, i].any():
                    rows.append(
                        {
                            k: d[k][t : t + 32, i]
                            for k in ("s", "s1", "a", "a_prev", "xi", "g", "torque")
                        }
                    )
                    labels.append(name)
    b = {k: torch.stack([x[k] for x in rows]).cuda() for k in rows[0]}
    variants = {
        "A0": Go2GaitCfg(),
        "A1": Go2GaitCfg(
            feet_clearance=0.05,
            clearance_target=0.12,
            clearance_sigma=0.065,
            clearance_region="stairs",
        ),
        "A2": Go2GaitCfg(
            feet_clearance=0.05,
            clearance_target=0.12,
            clearance_sigma=0.065,
            clearance_mode="soft_deficit",
            clearance_region="stairs",
        ),
    }
    results = []
    gradients = {}
    for region in ("all", "stairs"):
        for mode, cfg in variants.items():
            mdp.params.cfg.gait = replace(cfg, clearance_region=region)
            with torch.no_grad():
                real = mdp.terms(
                    unflatten_state(b["s"]),
                    unflatten_state(b["s1"]),
                    b["g"],
                    b["a"],
                    b["a_prev"],
                    measured_torque=b["torque"],
                )
            actions = b["a"].clone().requires_grad_()
            state = unflatten_state(b["s"][:, 0])
            prev = b["a_prev"][:, 0]
            clears = []
            tracks = []
            totals = []
            for t in range(32):
                old = state
                state = wm.step(
                    state,
                    actions[:, t],
                    mdp.normalize_context(b["xi"][:, t]),
                    mdp.terrain.local_scan(state[0], state[1]),
                )
                terms = mdp.terms(old, state, b["g"][:, t], actions[:, t], prev)
                clears.append(terms["feet_clearance"])
                tracks.append(terms["track_lin_vel_xy"] + terms["track_yaw_rate"])
                totals.append(sum(terms.values()))
                prev = actions[:, t]
            clear = torch.stack(clears, 1)
            tracking = torch.stack(tracks, 1)
            total = torch.stack(totals, 1)
            cg = torch.autograd.grad(clear.sum() * 0.02, actions, retain_graph=True)[0]
            tg = torch.autograd.grad(tracking.sum() * 0.02, actions, retain_graph=True)[0]
            fg = torch.autograd.grad(total.sum() * 0.02, actions)[0]
            for name in sorted(set(labels)):
                mask = torch.tensor([x == name for x in labels], device="cuda")
                c, t = cg[mask].flatten(1), tg[mask].flatten(1)
                cosine = F.cosine_similarity(c, t, dim=-1)
                realclear = real["feet_clearance"][mask]
                denom = sum(real.values())[mask].abs().mean().clamp_min(1e-8)
                results.append(
                    dict(
                        mode=mode,
                        region=region,
                        scenario=name,
                        samples=int(mask.sum()),
                        real_clearance_rate_mean=float(realclear.mean()),
                        real_clearance_rate_max=float(realclear.max()),
                        imagined_clearance_rate_mean=float(clear[mask].mean()),
                        imagined_clearance_rate_max=float(clear[mask].max()),
                        real_clearance_abs_fraction=float(realclear.abs().mean() / denom),
                        action_clearance_gradient_norm=float(c.norm(dim=-1).mean()),
                        tracking_gradient_norm=float(t.norm(dim=-1).mean()),
                        total_gradient_norm=float(fg[mask].flatten(1).norm(dim=-1).mean()),
                        clearance_tracking_cosine=float(cosine.mean()),
                        finite=bool(torch.isfinite(fg[mask]).all()),
                    )
                )
            gradients[f"{mode}_{region}"] = cg.detach().cpu()
    heights = torch.linspace(-0.02, 0.25, 400, requires_grad=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    scalar = []
    for name, cfg in variants.items():
        value = cfg.feet_clearance * clearance_shape(heights, cfg)
        grad = torch.autograd.grad(value.sum(), heights)[0]
        axs[0].plot(heights.detach(), value.detach(), label=name)
        axs[1].plot(heights.detach(), grad, label=name)
        scalar.append(dict(mode=name, max_abs_height_grad=float(grad.abs().max())))
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_xlabel("local bottom clearance (m)")
    axs[0].set_ylabel("shape rate (before swing/moving gates)")
    axs[1].set_ylabel("d rate / d height")
    fig.tight_layout()
    fig.savefig(a.output / "height_gradients.png")
    plt.close(fig)
    (a.output / "gradients.json").write_text(
        json.dumps(
            dict(
                records=results,
                scalar=scalar,
                trajectories=str(a.trajectories),
                same_states_and_actions=True,
                wm="source WM, unchanged",
                dt=0.02,
                gate_note="XY foot speed is a proxy; sliding supports can activate; slip penalty unchanged",
            ),
            indent=2,
        )
    )
    torch.save(dict(gradients=gradients, labels=labels), a.output / "action_gradients.pt")
    assert all(x["finite"] for x in results)
    print(json.dumps(dict(samples=len(rows), all_finite=True, records=len(results))), flush=True)


if __name__ == "__main__":
    main()
