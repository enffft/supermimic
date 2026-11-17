"""Wait for an authorized continuation, evaluate the final Actor and publish evidence."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from unisupermimic.tasks.velocity.go2w.experiment import ROOT, WM_FILE_SHA, WM_PATH, write_json


def read(path):
    return json.loads(path.read_text())


def execute(arguments, log):
    with log.open("w") as stream:
        subprocess.run(arguments, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)


def finalize(run, parent, pid):
    while not (run / "training_summary.json").exists() and not (run / "failure.json").exists():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            write_json(run / "failure.json", {"error": "Training process exited without summary"})
            break
        time.sleep(30)
    target = ROOT / "docs/validation/go2w_unified_h8_continuation"
    target.mkdir(parents=True, exist_ok=True)
    report = ROOT / "docs/experiments/go2w_h8_continuation_3p5h.md"
    summary = (
        read(run / "training_summary.json") if (run / "training_summary.json").exists() else {}
    )
    complete = summary.get("formal_budget_complete") is True
    comparison = None
    if complete:
        assessment = run / "reassessment"
        assessment.mkdir(exist_ok=True)
        execute(
            [
                sys.executable,
                str(ROOT / "tools/evaluate_go2w.py"),
                "--checkpoint",
                str(run / "evaluation/final.pt"),
                "--seeds",
                *map(str, range(2026, 2037)),
                "--output",
                str(assessment / "final_11yaw.json"),
                "--traces",
                str(assessment / "final_traces.pt"),
            ],
            assessment / "evaluation.log",
        )
        physical = read(assessment / "final_11yaw.json")
        baseline = read(parent / "reassessment/final_11yaw.json")
        rows = physical["rows"]
        validation = {
            "episodes": len(rows),
            "same_protocol": physical["evaluation_protocol"] == baseline["evaluation_protocol"],
            "same_conditions": [(r["name"], r["seed"], r["initial_yaw_rad"]) for r in rows]
            == [(r["name"], r["seed"], r["initial_yaw_rad"]) for r in baseline["rows"]],
            "all_timeouts_500_steps": all(
                r["steps"] == 500 for r in rows if r["terminal_reason"] == "time_out"
            ),
            "all_end_reasons_known": all(r["terminal_reason"] != "unknown" for r in rows),
        }
        write_json(assessment / "evaluation_validation.json", validation)
        if len(rows) != 847 or not all(v for k, v in validation.items() if k != "episodes"):
            raise RuntimeError(f"Continuation evaluation protocol mismatch: {validation}")
        comparison = {"before": baseline["by_domain"], "after": physical["by_domain"]}
        write_json(target / "comparison.json", comparison)
        shutil.copy2(assessment / "final_11yaw.json", target / "final_11yaw.json")
        shutil.copy2(
            assessment / "evaluation_validation.json", target / "evaluation_validation.json"
        )
        execute(
            [
                sys.executable,
                str(ROOT / "tools/render_go2w_evaluation.py"),
                "--traces",
                str(assessment / "final_traces.pt"),
                "--output",
                str(assessment / "videos"),
            ],
            assessment / "render.log",
        )
        for video in ("flat.mp4", "stairs.mp4"):
            shutil.copy2(assessment / "videos" / video, target / video)
    original_sha = hashlib.sha256(WM_PATH.read_bytes()).hexdigest()
    if original_sha != WM_FILE_SHA:
        raise RuntimeError("Original World Model asset hash changed")
    write_json(target / "source_asset_preserved.json", {"sha256": original_sha, "preserved": True})
    for name in (
        "training_summary.json",
        "resume_initialization.json",
        "budget.json",
        "launch.json",
        "failure.json",
        "progress.json",
        "provenance.json",
        "effective_algo.json",
        "evaluation/initial_physical.json",
        "evaluation/initial_wm.json",
        "evaluation/final_physical.json",
        "evaluation/final_wm.json",
    ):
        source = run / name
        if source.exists():
            shutil.copy2(source, target / name.replace("/", "_"))
    checkpoint = __import__("torch").load(
        run / "last.pt", map_location="cpu", weights_only=True, mmap=True
    )
    levels = (checkpoint["curriculum"]["tensors"]["skill_levels"] + 1).tolist()
    write_json(target / "final_curriculum_levels.json", levels)
    del checkpoint
    lines = [
        "# Go2W H8 额外3.5小时续训结果",
        "",
        f"状态：{'已完成额外12600秒有效训练循环' if complete else '未完成，见failure.json和progress.json'}。",
        f"父检查点：`{parent / 'last.pt'}`；续训目录：`{run}`。",
        "恢复已有Actor、Critic、WM、三套优化器、Replay、历史池和课程；环境重采样，非逐步完全复现。",
        "H8、wm_window8、4096env、1024buffer、WM/Actor batch2048、奖励及采样设置保持不变。",
        "方向检查保留诊断，依照用户明确授权继续训练。原始WM资产SHA未变。",
        "",
        f"本次有效循环：{summary.get('session_loop_seconds', 0):.3f}秒；累计：{summary.get('wall_time', 0):.3f}秒。",
        f"本次iteration：{summary.get('session_iterations')}；新增真实转移：{summary.get('session_transitions')}。",
        f"新增更新：{summary.get('session_updates')}；GPU峰值：{summary.get('peak_GPU_MiB')}MiB。",
        f"最终14项课程等级：{levels}。",
        "",
    ]
    if comparison:
        lines += [
            "评测为冻结Actor的真实物理847条固定条件，每条最多10秒；11个确定性初始yaw并非独立随机实验。",
            "",
            "|任务|续训前通过率|续训后通过率|续训后跌倒率|",
            "|---|---:|---:|---:|",
        ]
        for domain, value in comparison["after"].items():
            lines.append(
                f"|{domain}|{comparison['before'][domain]['success_rate']:.1%}|{value['success_rate']:.1%}|{value['fall_rate']:.1%}|"
            )
        lines += [
            "",
            "分轴速度、RMSE和每级台阶明细见[完整物理评测](../validation/go2w_unified_h8_continuation/final_11yaw.json)。",
            "[平地视频](../validation/go2w_unified_h8_continuation/flat.mp4) · [台阶视频](../validation/go2w_unified_h8_continuation/stairs.mp4)。",
            "训练时间完成与运动能力达标分开判断；通过率来自物理评测，不以loss或想象奖励替代。",
        ]
    if (run / "failure.json").exists():
        lines += ["", f"实际异常：`{read(run / 'failure.json')}`。"]
    report.write_text("\n".join(lines) + "\n")
    execute(["git", "add", "--", str(report), str(target)], run / "final_git_add.log")
    execute(
        [
            "git",
            "commit",
            "--only",
            "-m",
            "Report Go2W H8 continuation and physical comparison",
            "--",
            str(report),
            str(target),
        ],
        run / "final_git_commit.log",
    )
    execute(
        ["git", "push", "user", "HEAD:experiment/go2w-unified-curriculum"],
        run / "final_git_push.log",
    )
    write_json(
        run / "finalization_complete.json",
        {"training_budget_complete": complete, "report": str(report), "pushed": True},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    try:
        finalize(args.run, args.parent, args.pid)
    except BaseException as exc:
        write_json(args.run / "finalization_failure.json", {"error": repr(exc)})
        raise
