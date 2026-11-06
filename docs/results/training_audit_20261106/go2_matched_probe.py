"""Bounded re-evaluation of saved policies; no training or parameter updates.

Run from the repository root with PYTHONPATH=src in the supermimic Conda env.
Uses the existing repository evaluator; all cases run sequentially.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_go2_velocity import evaluate_command

OUT = Path(__file__).parent
BASE = ROOT / "logs/phase3_training/go2_velocity_deployable_v2_20261102"
GAIT = ROOT / "logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103"
WM_SOURCE = ROOT / "logs/go2_velocity_phase1_3072_b1024_20261027"
NEW = ROOT / "logs/go2_comparison/supermimic_2p5h_seed2026"
CASES = {
    "new_cold_2p5h": NEW / "ckpt_latest.pt",
    "old_chain_2p95h": GAIT / "ckpt_0022000.pt",
    "old_actor_2p96h": GAIT / "ckpt_0024000.pt",
    "old_final_chain_4p55h": GAIT / "ckpt_0034000.pt",
}
COMMANDS = ((.3, 0., 0.), (.5, 0., 0.), (0., .3, 0.), (0., 0., .4))


def clock(directory, iteration=None):
    reader = EventAccumulator(str(directory / "tb"), size_guidance={"scalars": 0}).Reload()
    rows = reader.Scalars("time/hours")
    row = rows[-1] if iteration is None else next(x for x in rows if x.step == iteration)
    return {"iteration": row.step, "hours": row.value, "wall_timestamp": row.wall_time}


def main():
    torch.set_num_threads(1)
    lineage = {"wm_source": clock(WM_SOURCE), "base_actor": clock(BASE),
               "gait_stop": clock(GAIT),
               "gait_saved": {str(i): clock(GAIT, i) for i in (22000, 24000, 34000)}}
    (OUT / "go2_lineage_clocks.json").write_text(json.dumps(lineage, indent=2) + "\n")
    for label, path in CASES.items():
        output = OUT / f"go2_{label}.json"
        if output.exists():
            print("SKIP completed", label, flush=True)
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        metadata = {k: saved.get(k) for k in ("iteration", "actor_updates", "num_updates",
                    "init_wm_from", "init_wm_source_iteration", "pretrained_wm",
                    "robot_model_sha256", "actuator_model_version")}
        config = saved["config"]
        metadata.update(seed=config["seed"], resume=config.get("resume"),
                        gait=config["env"].get("go2_gait"),
                        imagination_obs_noise=config["policy_trainer"]["obs_noise"])
        del saved
        partial = OUT / f"go2_{label}_partial.json"
        rows = json.loads(partial.read_text()) if partial.exists() else []
        for command in COMMANDS:
            if any(row["command_body_xy_yaw"] == list(command) for row in rows):
                continue
            print("EVALUATE", label, command, flush=True)
            row = evaluate_command(path, command, device="cuda:0", num_envs=16, episodes=16)
            rows.append(row)
            partial.write_text(json.dumps(rows, indent=2) + "\n")
            print("RESULT", label, json.dumps(row), flush=True)
        result = {"checkpoint": str(path.relative_to(ROOT)),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metadata": metadata,
                  "method": "existing evaluate_command; seed2026; 16 nominal envs, 16 ended episodes per command, 10s; sequential; no noise/DR; no updates",
                  "limitations": "Nominal deterministic initial states, not independent perturbation robustness trials; gait uses FK height proxies; one training seed per lineage",
                  "real_mujoco": rows}
        output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
