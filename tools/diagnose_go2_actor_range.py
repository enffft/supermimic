"""Blind inference from saved proprioception history; no environment/map required."""

import argparse
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.checkpoint import initialize_actor
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = torch.load(args.source / "replay_snapshot.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.source / "last.pt", map_location="cpu", weights_only=True)
    n = saved["data"]["s"].shape[1]
    count = len(saved["data"]["s"])
    rb = ReplayBuffer(count, n, "cpu", num_joints=12, context_dim=16, goal_dim=3)
    rb.data = saved["data"]
    rb.ptr = saved["ptr"]
    rb.size = saved["size"]
    torch.manual_seed(909)
    starts, hist = rb.sample_starts(256, 5)
    m = Go2VelocityMdp()
    obs = m.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"])
    baseline = Policy(42, 3, hist=5, num_joints=12, goal_mode="direct")
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "migration/source_reference/checkpoint_manifest.json"
        ).read_text()
    )
    initialize_actor(baseline, manifest["path"], saved["task_contract"])
    adapted = Policy(42, 3, hist=5, num_joints=12, goal_mode="direct")
    adapted.load_state_dict(checkpoint["pi"])
    with torch.no_grad():
        a0 = baseline(obs, starts["g"])
        a1 = adapted(obs, starts["g"])
    result = {
        "sample_count": 256,
        "actor_input_dim": 213,
        "environment_created": False,
        "terrain_map_loaded": False,
        "baseline_saturation_fraction": float((a0.abs() > 0.98).float().mean()),
        "adapted_saturation_fraction": float((a1.abs() > 0.98).float().mean()),
        "baseline_abs_max": float(a0.abs().max()),
        "adapted_abs_max": float(a1.abs().max()),
        "action_change_rmse": float((a1 - a0).square().mean().sqrt()),
        "bounded_finite": bool(torch.isfinite(a1).all() and a1.abs().max() <= 1),
        "note": "same replay histories and goals; tanh threshold .98; inference uses no map or simulated environment",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
