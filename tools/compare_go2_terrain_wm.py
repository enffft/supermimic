"""Paired A/B on identical chronological real replay, using shared wm_update().

No actor update, no simulator rollout in fitting, no new optimizer/training math.
The only network difference is 77 appended first-layer inputs, initialized zero.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_go2_blind_terrain import prediction_diagnostics
from validate_go2_migration import dump

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.updates <= 300:
        raise ValueError("bounded WM-only diagnostic")
    args.output.mkdir(parents=True, exist_ok=False)
    inv = json.loads((args.source / "invocation.json").read_text())
    cfg = load_config(terrain=True)
    cfg.commands["velocity"].terrain_profile = inv["profile"]
    cfg.commands["velocity"].terrain_fixed_level = inv["level"]
    env = make_go2_env(cfg, inv["envs"])
    env.init_state()
    try:
        snapshot = torch.load(
            args.source / "replay_snapshot.pt", map_location="cpu", weights_only=True
        )
        from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField

        # Re-hash compiled geometry with the source's recorded generator metadata.
        # This accepts added default config fields only when the EXACT collision
        # data, transforms, seed and origins still reproduce the source digest.
        if cfg.scene.terrain is not None:
            field = TerrainHeightField.from_mujoco(
                env.get_playback_model(0),
                metadata={
                    "seed": cfg.commands["velocity"].terrain_seed,
                    "origins": env.backend.get_terrain_spawn_data().terrain_origins.tolist(),
                    "generator": snapshot["task_contract"]["terrain"],
                },
            )
            if field.fingerprint != snapshot["task_contract"].get("terrain_fingerprint"):
                raise ValueError("Replay compiled terrain geometry/metadata fingerprint differs")
            env.command_manager.get_term("velocity").mdp.terrain = field
        count = len(snapshot["data"]["s"])
        size = snapshot.get("size", count)
        # Early smoke snapshots predate ptr metadata. Derive from recorded actual
        # simulation calls, verify s1 -> s continuity before accepting reconstruction.
        if "ptr" in snapshot:
            ptr = snapshot["ptr"]
            provenance = "explicit ring metadata"
        else:
            report = json.loads((args.source / "training.json").read_text())
            ptr = report["timings"]["simulation"]["count"] % count
            provenance = (
                "legacy ptr reconstructed from exact collection-call counter; continuity checked"
            )
        rows = (torch.arange(size) + ptr - size) % count
        data = {k: v[rows] for k, v in snapshot["data"].items()}
        # Explicit collection resets must close the preceding replay window.
        manual_boundaries = (data["ep_step"][1:] == 0) & ~data["done"][:-1]
        corrected_boundaries = int(manual_boundaries.sum())
        data["done"][:-1].logical_or_(data["ep_step"][1:] == 0)
        diff = (data["s1"][:-1] - data["s"][1:]).abs().amax(-1)
        live = ~data["done"][:-1]
        assert float(diff[live].max()) < 1e-5, "Replay chronology or reset alignment mismatch"
        train_count = size * 3 // 4
        learners = []
        for label, conditioning in [("A", False), ("B", True)]:
            folder = args.output / label
            folder.mkdir()
            lc = SuperMimicConfig(
                goal_mode="direct",
                goal_frames=1,
                hist=5,
                terrain_training=True,
                terrain_conditioning=conditioning,
                compile=False,
                amp="none",
                no_sync=False,
                wm_opt="adam",
                pi_opt="adam",
                wm_batch=args.batch,
                pi_batch=args.batch,
                buffer_steps=count,
                wm_window=8,
                horizon=32,
                seed=2026,
                profile=True,
            )
            l = SuperMimicLearner(env, lc, folder, device="cuda", backend="mujoco")
            for i in range(train_count):
                l.buffer.add({k: v[i].to(l.device) for k, v in data.items()})
            states = unflatten_state(data["s"][:train_count].to(l.device))
            targets = unflatten_state(data["s1"][:train_count].to(l.device))
            l.wm.in_norm.update(
                l.wm.inputs(
                    states,
                    data["a"][:train_count].to(l.device),
                    l.mdp.normalize_context(data["xi"][:train_count].to(l.device)),
                    l._terrain_scan(states),
                )
            )
            l.wm.out_norm.update(l.wm.targets(states, targets))
            l.wm.inc_norm.update(l.wm.increments(states, targets))
            learners.append(l)
        a, b = learners
        b.wm.initialize_from_flat(a.wm)
        # Appended scan stats are from the SAME training transitions, not held-out data.
        states = unflatten_state(data["s"][:train_count].to(b.device))
        scan = b._terrain_scan(states).flatten(0, 1)
        with torch.no_grad():
            b.wm.in_norm.mean[-77:] = scan.mean(0)
            b.wm.in_norm.var[-77:] = scan.var(0, unbiased=False)
        b.pi.load_state_dict(a.pi.state_dict())
        b.critic.load_state_dict(a.critic.state_dict())
        held = ReplayBuffer(count, inv["envs"], "cuda", num_joints=12, context_dim=16, goal_dim=3)
        for i in range(train_count, size):
            held.add({k: v[i].to("cuda") for k, v in data.items()})
        reports = []
        for label, l in zip(("A", "B"), learners):
            curve = []
            start = time.perf_counter()
            for update in range(args.updates):
                torch.manual_seed(3000 + update)  # identical windows and env ids
                loss, err = l.wm_update()
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("nonfinite WM")
                curve.append({"update": update + 1, "loss": float(loss), "err8": float(err)})
            elapsed = time.perf_counter() - start
            train = l.buffer
            l.buffer = held
            torch.manual_seed(4200)
            predictions = prediction_diagnostics(l, args.output / label)
            l.buffer = train
            l.save(args.output / label / "wm_only.pt")
            dump(args.output / label / "curve.json", curve)
            reports.append(
                {
                    "mode": label,
                    "conditioning": l.cfg.terrain_conditioning,
                    "predictions": predictions,
                    "updates": args.updates,
                    "elapsed_s": elapsed,
                    "final_loss": curve[-1]["loss"],
                    "wm_parameters": sum(t.numel() for t in l.wm.parameters()),
                    "mean_wm_update_ms": sum(l.profile_ms["wm_fit"]) / len(l.profile_ms["wm_fit"]),
                }
            )
        dump(
            args.output / "comparison.json",
            {
                "source": str(args.source),
                "replay_sha256": hashlib.sha256(
                    (args.source / "replay_snapshot.pt").read_bytes()
                ).hexdigest(),
                "seed": 2026,
                "train_transitions": train_count * inv["envs"],
                "heldout_transitions": (size - train_count) * inv["envs"],
                "split": "chronological 75/25, done-masked windows",
                "ring_metadata": provenance,
                "nonreset_state_alignment_max": float(diff[live].max()),
                "legacy_manual_reset_boundaries_closed": corrected_boundaries,
                "wm_widths": [2048, 2048, 2048],
                "batch": args.batch,
                "initialization": "paired original parameters and normalization; B appended columns zero; independent Adam",
                "actor_updated": False,
                "reports": reports,
                "limitations": [
                    "one seed and bounded data; no statistical performance claim",
                    "WM-only test does not establish improved real Actor locomotion",
                ],
            },
        )
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
