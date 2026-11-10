"""Run fixed physical and WM-gradient evaluations after a training process exits."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    status_path = run / "automatic_evaluation_status.json"
    write_json(status_path, {"status": "waiting_for_training", "pid": args.pid})
    try:
        process = psutil.Process(args.pid)
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            time.sleep(2)
    except psutil.NoSuchProcess:
        pass
    failure = (run / "failure.json").exists()
    checkpoint = run / ("interrupted_rollback.pt" if failure else "last.pt")
    if not checkpoint.exists():
        write_json(
            status_path, {"status": "failed", "reason": f"No usable checkpoint: {checkpoint}"}
        )
        return 1
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    jobs = [
        (
            "policy",
            [
                sys.executable,
                str(ROOT / "tools/evaluate_go2_terrain_policy.py"),
                "--mode",
                "fixed",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(run / "final_policy_evaluation"),
            ],
        ),
        (
            "world_model",
            [
                sys.executable,
                str(ROOT / "tools/evaluate_go2_terrain_b.py"),
                "--mode",
                "evaluate",
                "--dataset",
                str(ROOT / "results/longrun_B/holdout"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(run / "final_wm_evaluation"),
            ],
        ),
    ]
    results = {}
    for name, command in jobs:
        write_json(
            status_path,
            {
                "status": "evaluating",
                "current": name,
                "checkpoint": str(checkpoint),
                "results": results,
            },
        )
        with (run / f"final_{name}_evaluation.log").open("w") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
                results[name] = {"returncode": completed.returncode}
            except subprocess.TimeoutExpired:
                results[name] = {"returncode": None, "error": "evaluation timeout after 600 s"}
    outcome = {
        "status": "complete"
        if all(r["returncode"] == 0 for r in results.values())
        else "evaluation_failed",
        "training_failure": failure,
        "checkpoint": str(checkpoint),
        "jobs": results,
    }
    for name, relative in (
        ("training", "summary.json"),
        ("physics", "final_policy_evaluation/summary.json"),
        ("wm_gate", "final_wm_evaluation/acceptance.json"),
    ):
        path = run / relative
        if path.exists():
            outcome[name] = json.loads(path.read_text())
    records = run / "final_policy_evaluation/fixed_evaluation.json"
    if records.exists():
        values = json.loads(records.read_text())
        groups = {}
        for label, rows in (
            ("flat", [v for v in values if v["level"] == 0]),
            ("level1", [v for v in values if v["level"] == 1]),
        ):
            groups[label] = {
                "episodes": len(rows),
                "fall_rate": sum(v["fall"] for v in rows) / len(rows),
                "xy_rmse_m_s": sum(v["xy_rmse_m_s"] for v in rows) / len(rows),
                "complete_crossings": sum(v["passed"] for v in rows),
            }
        outcome["policy_summary"] = groups
    write_json(run / "automatic_evaluation_summary.json", outcome)
    write_json(status_path, outcome)
    return int(outcome["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
