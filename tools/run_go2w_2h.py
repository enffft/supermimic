"""Run a bounded Go2W experiment, then evaluate even if the UI disconnects."""

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / "source_snapshot"
    environment = os.environ | {"PYTHONPATH": str(source / "src"), "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    record = {"pid": os.getpid(), "started_at": dt.datetime.now().astimezone().isoformat(),
              "training_budget_seconds": args.seconds, "wm_asset": str(args.wm.resolve())}

    def status(state, **fields):
        record.update(state=state, **fields)
        temporary = run / "supervisor_status.json.tmp"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(run / "supervisor_status.json")

    def execute(name, command, training=False):
        with (run / f"{name}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, *map(str, command)], cwd=source,
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
            status(name, child_pid=process.pid, command=list(map(str, command)))
            result = process.wait()
        if result:
            status("failed", failed_stage=name, exit_code=result)
            raise RuntimeError(f"{name} exited {result}; see {run / (name + '.log')}")
        if training:
            record["training_finished_at"] = dt.datetime.now().astimezone().isoformat()

    try:
        execute("training", [source / "tools/train_go2w_experiment.py",
                              "--profile", "omni_v3", "--fresh", "--wm-only", "250",
                              "--actor-from", run / "bootstrap/actor.pt",
                              "--wm-from", args.wm, "--output", run / "training",
                              "--envs", "4096", "--batch", "2048",
                              "--seconds", str(args.seconds), "--iterations", "100000"], training=True)
        checkpoint = run / "training/last.pt"
        execute("final_policy_evaluation", [source / "tools/evaluate_go2w_experiment.py",
                                            "--profile", "omni_v3", "--checkpoint", checkpoint,
                                            "--output", run / "evaluation/final.json",
                                            "--save-windows", run / "evaluation/final_holdout.pt"])
        datasets = {
            "common": run.parent / "go2w_joint_range_20261114/evaluation/common_holdout.pt",
            "bootstrap": run / "evaluation/bootstrap_holdout.pt",
        }
        for name, path in datasets.items():
            execute(f"wm_{name}_evaluation", [source / "tools/evaluate_go2w_wm_holdout.py",
                                              "--profile", "omni_v3", "--checkpoint", checkpoint,
                                              "--dataset", path, "--output", run / f"evaluation/wm_{name}.json"])
        comparisons = []
        for stage in ("bootstrap", "final"):
            result = json.loads((run / f"evaluation/{stage}.json").read_text())
            for row in result["rows"]:
                observed = row["mean_actual_vx_vy_yaw"] or [None] * 3
                errors = row["RMSE_vx_vy_yaw"] or [None] * 3
                comparisons.append(dict(stage=stage, name=row["name"], domain=row["domain"],
                                        level=row["level"], falls=row["falls"], trials=row["replicas"],
                                        actual_vx=observed[0], actual_vy=observed[1], actual_yaw=observed[2],
                                        RMSE_vx=errors[0], RMSE_vy=errors[1], RMSE_yaw=errors[2],
                                        entry_successes=row["entry_successes"], climb_successes=row["climb_successes"],
                                        mean_progress_m=sum(row["progress_m"]) / row["replicas"]))
        with (run / "evaluation/comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        status("complete", finished_at=dt.datetime.now().astimezone().isoformat(), child_pid=None)
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
