"""Detached, bounded A2/4cm/Level5 continuation; evaluate once after training."""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/stairs_clearance_20cm_A2"))
    a = p.parse_args()
    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    output = root / "longrun_90min"
    cmd = [
        sys.executable,
        str(repo / "tools/train_go2_clearance_ablation.py"),
        "--checkpoint",
        str(root / "migration_final/migrated.pt"),
        "--output",
        str(output),
        "--task",
        "stairs_clearance_20cm",
        "--iterations",
        "200000",
        "--seconds",
        "5400",
        "--wm-adaptation",
        "200",
        "--critic-warmup",
        "100",
        "--actor-lr-scale",
        "0.01",
        "--probe-every",
        "200",
        "--scope-refresh",
        "1000",
        "--high-probe-min-level",
        "3",
        "--wm-balance-high-levels",
        "--holdout-root",
        str(root / "holdouts"),
        "--guard-baseline",
        str(root / "baseline_quick/evaluation.json"),
        "--physics-guard-seconds",
        "1200",
    ]
    status = dict(
        state="starting",
        supervisor_pid=os.getpid(),
        started_at=datetime.datetime.now().astimezone().isoformat(),
        budget_seconds=5400,
        command=cmd,
        output=str(output),
        meaning="90-minute wall budget includes collection, WM, Actor, Critic, sparse scope/physics checks; post-run evaluation separate",
    )

    def save():
        temp = root / "run_status.json.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(root / "run_status.json")

    save()
    started = time.monotonic()
    with (root / "training.log").open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        status.update(state="training", training_pid=proc.pid)
        save()
        try:
            rc = proc.wait(timeout=5550)
        except subprocess.TimeoutExpired:
            status["watchdog"] = "training exceeded budget+150s; request interrupt checkpoint"
            os.killpg(proc.pid, signal.SIGINT)
            try:
                rc = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                rc = proc.wait(timeout=30)
        status.update(training_returncode=rc, training_wall_s=time.monotonic() - started)
        save()
    summary = output / "summary.json"
    if summary.exists():
        status["training_summary"] = json.loads(summary.read_text())
    if rc != 0:
        status["state"] = "training_failed"
        save()
        return
    status["state"] = "final_evaluation"
    save()
    for name, checkpoint in [
        ("source_20cm", root / "migration_final/migrated.pt"),
        ("candidate_20cm", output / "last.pt"),
    ]:
        with (root / f"evaluation_{name}.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools/evaluate_go2_omnidirectional.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(root / "evaluation" / name),
                    "--stairs-omni",
                    "--max-level",
                    "5",
                    "--stair-step-increment",
                    ".04",
                    "--seconds",
                    "20",
                ],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        if result.returncode:
            status.update(state="evaluation_failed", failed_evaluation=name)
            save()
            return
    base = json.loads((root / "evaluation/source_20cm/evaluation.json").read_text())["aggregate"]
    cand = json.loads((root / "evaluation/candidate_20cm/evaluation.json").read_text())["aggregate"]
    comparison = {k: {"source": base[k], "candidate": cand[k]} for k in base}
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2))
    status.update(
        state="complete",
        completed_at=datetime.datetime.now().astimezone().isoformat(),
        candidate=str(output / "last.pt"),
        original_best_untouched=True,
    )
    save()
    report = repo / "docs/go2_stairs_20cm_A2_90min.md"
    with report.open("a") as f:
        f.write(
            "\n\n## 自动收尾结果\n\n运行状态：完成。完整训练统计见 `results/stairs_clearance_20cm_A2/run_status.json`，同条件新旧策略对照见 `comparison.json`。\n\n"
        )
        f.write(
            "```json\n"
            + json.dumps(status.get("training_summary", {}), ensure_ascii=False, indent=2)
            + "\n```\n"
        )
        f.write(
            "\n| 楼梯 | 新地图来源通过率 | 续训候选通过率 | 来源跌倒率 | 候选跌倒率 |\n|---|---:|---:|---:|---:|\n"
        )
        for direction in ("up", "down"):
            for level in range(1, 6):
                k = f"stairs_{direction}_L{level}_cmd0"
                x, y = base[k], cand[k]
                f.write(
                    f"| {direction} L{level} ({level * 4}cm) | {x['crossing_passed']:.0%} | {y['crossing_passed']:.0%} | {x['fall']:.0%} | {y['fall']:.0%} |\n"
                )


if __name__ == "__main__":
    main()
