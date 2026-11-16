"""Replay tail coverage; contact proxies are explicitly distinct from force diagnostics."""

import argparse
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp


def audit(checkpoint, output):
    s = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    b = s["replay"]
    d = b["data"]
    rows = (torch.arange(b["size"]) + b["ptr"] - b["size"]) % b["T"]
    out = []
    for direction in (4, 5):
        for level in range(1, 8):
            mask = (d["command_domain"][rows] == direction) & (d["terrain_level"][rows] == level)
            counts = int(mask.sum())
            episodes = int(((d["ep_step"][rows] == 0) & mask).sum())
            entry = dict(
                domain=direction,
                level=level,
                transitions=counts,
                observed_episode_starts=episodes,
                falls=int((d["term"][rows] & mask).sum()),
                probes=int((d["terrain_probe"][rows] & mask).sum()),
                valid_window_note="windows start at this scope; subsequent done masks prevent crossing resets",
            )
            for k in (8, 32):
                valid = mask[: len(rows) - k + 1].clone()
                for offset in range(k - 1):
                    valid &= ~d["done"][rows[offset : len(rows) - k + 1 + offset]]
                entry[f"valid_H{k}_windows"] = int(valid.sum())
                entry[f"valid_H{k}_fraction"] = float(
                    valid.sum() / mask[: len(rows) - k + 1].sum().clamp_min(1)
                )
            if counts:
                ids = mask.nonzero()
                r, e = rows[ids[:, 0]], ids[:, 1]
                goals = d["g"][r, e]
                actions = d["a"][r, e]
                entry["mixed_three_nonzero"] = int((goals != 0).all(-1).sum())
                entry["action_abs_max"] = float(actions.abs().max())
                entry["action_saturation"] = float((actions.abs() > 0.98).float().mean())
                entry["command_abs_max"] = goals.abs().amax(0).tolist()
            out.append(entry)
    starts = s["extra_state"].get("actor_starts_domain_terrain_speed")
    result = dict(
        checkpoint=str(checkpoint),
        replay_rows=b["size"],
        replay_capacity=b["T"] * b["N"],
        scopes=out,
        actor_starts_domain_terrain_speed=None if starts is None else starts.tolist(),
        speed_axis=["S0", "S1", "S2", "S3", "historical_unknown"],
        wm_sampled_start_levels=b.get("wm_level_counts", torch.zeros(8)).tolist(),
        actual_contacts="training Replay does not store solver forces; actual force/contact events are in diagnostics/baseline_feet/contacts.jsonl.gz",
        snapshot_not_cumulative=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.checkpoint, a.output)
