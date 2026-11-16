"""Audit native solver diagnostic rollouts; contact-foot speed includes wall impacts."""

import argparse
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.tasks.velocity.go2.gait import Go2GaitCfg
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp

p = argparse.ArgumentParser(description="Actual solver contact-foot speed and reward-hacking audit")
p.add_argument("--root", type=Path, default=Path("results/stairs_clearance_B4"))
r = p.parse_args().root
out = {}
for tag, folder in [
    ("Source", "baseline_feet"),
    ("A0", "A0_feet"),
    ("A1", "gaussian_feet"),
    ("A2", "A2_feet"),
]:
    x = torch.load(r / "diagnostics" / folder / "trajectories.pt", weights_only=True)
    m = Go2VelocityMdp(terrain=TerrainHeightField(**x["field"]))
    if tag in ["A1", "A2"]:
        m.params.cfg.gait = Go2GaitCfg(
            feet_clearance=0.05,
            clearance_target=0.12,
            clearance_sigma=0.065,
            clearance_mode="gaussian" if tag == "A1" else "soft_deficit",
            clearance_region="stairs",
        )
    entries = x["traces"][0]["entries"][1:]
    m.clearance_boxes = torch.cat(
        (entries - entries.new_tensor([0.6, 4.0]), entries + entries.new_tensor([7.4, 4.0])), -1
    )
    caseout = {}
    for i, (name, *_) in enumerate(x["cases"]):
        values = []
        for tr in x["traces"]:
            d = tr["data"]
            mask = d["active"][:, i]
            contact = d["contact"][mask, i]
            slip = d["foot_velocity"][mask, i, :, :2].norm(dim=-1)
            s, s1 = unflatten_state(d["s"][mask, i]), unflatten_state(d["s1"][mask, i])
            terms = m.terms(
                s,
                s1,
                d["g"][mask, i],
                d["a"][mask, i],
                d["a_prev"][mask, i],
                measured_torque=d["torque"][mask, i],
            )
            cl = terms["feet_clearance"]
            total = sum(terms.values())
            values.append(
                dict(
                    contact_slip_m_s=float(slip[contact].mean()) if contact.any() else 0,
                    clearance_rate_mean=float(cl.mean()),
                    clearance_rate_max=float(cl.max()),
                    clearance_abs_fraction=float(
                        cl.abs().mean() / total.abs().mean().clamp_min(1e-6)
                    ),
                    abs_vertical_velocity_p90=float(s1[2][:, 2].abs().quantile(0.9)),
                    joint_limit_fraction=float(
                        (
                            (s1[4] < m.params.soft_joint_limits[:, 0])
                            | (s1[4] > m.params.soft_joint_limits[:, 1])
                        )
                        .float()
                        .mean()
                    ),
                )
            )
        caseout[name] = {k: sum(v[k] for v in values) / len(values) for k in values[0]}
    out[tag] = caseout
(r / "diagnostics/reward_hacking.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v["stairs_up_L7"] for k, v in out.items()}, indent=2))
