"""Finish the authorized distance-curriculum run with physical/reward evidence."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from unisupermimic.tasks.velocity.go2w.curriculum import DOMAINS, SINGLE_HEIGHTS, STAIR_HEIGHTS
from unisupermimic.tasks.velocity.go2w.experiment import ROOT, WM_FILE_SHA, WM_PATH, write_json


def read(path):
    return json.loads(path.read_text())


def execute(argv, log):
    with log.open("w") as stream:
        subprocess.run(argv, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)


def finalize(run, parent, pid):
    while not any((run / n).exists() for n in ("training_summary.json", "failure.json")):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            write_json(run / "failure.json", {"error": "Training exited without a summary"})
            break
        time.sleep(30)
    target = ROOT / "docs/validation/go2w_distance_20261118_results"
    target.mkdir(exist_ok=True, parents=True)
    report = ROOT / "docs/experiments/go2w_distance_20261118_results.md"
    summary = (
        read(run / "training_summary.json") if (run / "training_summary.json").exists() else {}
    )
    complete = summary.get("formal_budget_complete") is True
    lines = [
        "# Go2W 距离课程续训：实际结果",
        "",
        f"训练状态：{'完成额外12600秒有效训练' if complete else '未完成，见失败证据'}。",
        f"运行目录：`{run}`；父断点：`{parent / 'last.pt'}`。",
        "这是显式课程/采样迁移：保留三网络、归一化、三优化器、Replay和历史池，重置课程及采样统计。不是逐步等价恢复。",
        "H8/window8、4096env、1024Replay、2048batch、单Actor及21项reward不变。",
        "[实现及验证报告](go2w_distance_curriculum_20261118.md) · [reward公式与限制](go2w_h8_reward_analysis.md)",
        "",
        f"新增有效循环 {summary.get('session_loop_seconds', 0):.3f}s；新增轮数 {summary.get('session_iterations')}；新增转移 {summary.get('session_transitions')}。",
        f"实际更新 {summary.get('session_updates')}；峰值GPU {summary.get('peak_GPU_MiB')}MiB。",
        "",
    ]
    if complete:
        assessment = run / "reassessment"
        assessment.mkdir(exist_ok=True)
        physical_path = assessment / "final_11yaw_rewards.json"
        execute(
            [
                sys.executable,
                str(ROOT / "tools/evaluate_go2w.py"),
                "--checkpoint",
                str(run / "evaluation/final.pt"),
                "--seeds",
                *map(str, range(2026, 2037)),
                "--rewards",
                "--traces",
                str(assessment / "final_traces.pt"),
                "--output",
                str(physical_path),
            ],
            assessment / "evaluation.log",
        )
        current = read(physical_path)
        before = read(parent / "reassessment/final_rewards.json")
        rows = current["rows"]
        validation = dict(
            episodes=len(rows),
            same_protocol=current["evaluation_protocol"] == before["evaluation_protocol"],
            same_conditions=[(r["name"], r["seed"], r["initial_yaw_rad"]) for r in rows]
            == [(r["name"], r["seed"], r["initial_yaw_rad"]) for r in before["rows"]],
            correct_timeouts=all(
                r["steps"] == 500 for r in rows if r["terminal_reason"] == "time_out"
            ),
            known_endings=all("unknown" not in r["terminal_reason"] for r in rows),
            reward_parity=current["reward_analysis"]["recomputed_reward_max_abs_error"] <= 1e-6,
        )
        write_json(assessment / "evaluation_validation.json", validation)
        if len(rows) != 847 or not all(v for k, v in validation.items() if k != "episodes"):
            raise RuntimeError(f"Physical evaluation evidence invalid: {validation}")
        shutil.copy2(physical_path, target / physical_path.name)
        shutil.copy2(
            assessment / "evaluation_validation.json", target / "evaluation_validation.json"
        )
        lines += [
            "冻结Actor、真实物理847条固定条件，每条首次episode最多10秒；11个固定yaw不代表11次独立训练。",
            "课程基座越线与严格四轮/10帧物理成功分别统计。训练完成不代表能力通过。",
            "",
            "|任务|迁移前成功率|续训后成功率|续训后跌倒率|",
            "|---|---:|---:|---:|",
        ]
        for name in DOMAINS:
            a, b = before["by_domain"][name], current["by_domain"][name]
            lines.append(
                f"|{name}|{a['success_rate']:.1%}|{b['success_rate']:.1%}|{b['fall_rate']:.1%}|"
            )
        lines += [
            "",
            "## 平地分轴跟踪",
            "",
            "|命令|成功/条件|平均vx,vy,wz|平均RMSE|",
            "|---|---:|---|---|",
        ]
        for name in dict.fromkeys(r["name"] for r in rows if r["domain"] == 0):
            group = [r for r in rows if r["name"] == name]

            def mean(key):
                return [round(sum(r[key][i] for r in group) / len(group), 4) for i in range(3)]

            lines.append(
                f"|{name}|{sum(r['passed'] for r in group)}/{len(group)}|{mean('mean_actual_vx_vy_yaw')}|{mean('rmse_vx_vy_yaw')}|"
            )
        lines += [
            "",
            "## 各级严格越障",
            "",
            "|任务|Level|高度cm|成功/条件|跌倒/条件|",
            "|---|---:|---:|---:|---:|",
        ]
        for d in range(4, 8):
            for level in range(10):
                group = [r for r in rows if r["domain"] == d and r["level"] == level]
                height = (SINGLE_HEIGHTS if d < 6 else STAIR_HEIGHTS)[level] * 100
                lines.append(
                    f"|{DOMAINS[d]}|{level + 1}|{height:g}|{sum(r['passed'] for r in group)}/{len(group)}|{sum(r['fall'] for r in group)}/{len(group)}|"
                )
        lines += [
            "",
            "## 逐项物理reward",
            "",
            "加权rate，未乘dt=.02；按各域首次episode存活转移池化。短时跌倒可能有较高rate，须结合上表时长和成功率判断。",
            "没有改动奖励权重；公式和功能见上述reward报告。完整每episode贡献及各域数据见JSON。",
            "",
            "|reward|之前flat|之后flat|之前单级上|之后单级上|之后单级下|之后连续上|之后连续下|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        old, new = before["reward_analysis"]["by_domain"], current["reward_analysis"]["by_domain"]
        for key in new["flat"]["pooled_rate_terms"]:
            values = [
                old["flat"],
                new["flat"],
                old["single_step_up"],
                new["single_step_up"],
                new["single_step_down"],
                new["multi_stairs_up"],
                new["multi_stairs_down"],
            ]
            lines.append(
                f"|{key}|" + "|".join(f"{v['pooled_rate_terms'][key]:.6g}" for v in values) + "|"
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
        for name in ("flat.mp4", "stairs.mp4"):
            shutil.copy2(assessment / "videos" / name, target / name)
        lines += [
            "",
            "[完整数据](../validation/go2w_distance_20261118_results/final_11yaw_rewards.json) · [平地视频](../validation/go2w_distance_20261118_results/flat.mp4) · [台阶视频](../validation/go2w_distance_20261118_results/stairs.mp4)",
        ]
    original = hashlib.sha256(WM_PATH.read_bytes()).hexdigest()
    if original != WM_FILE_SHA:
        raise RuntimeError("Original WM asset changed")
    write_json(target / "source_asset_preserved.json", dict(sha256=original, preserved=True))
    for name in (
        "training_summary.json",
        "failure.json",
        "progress.json",
        "resume_initialization.json",
        "curriculum_migration.json",
        "budget.json",
        "launch.json",
        "provenance.json",
        "effective_algo.json",
    ):
        if (run / name).exists():
            shutil.copy2(run / name, target / name)
    with (run / "go2w_diagnostics.jsonl").open() as stream:
        last = None
        for line in stream:
            last = json.loads(line)
    write_json(
        target / "final_sampling_curriculum.json",
        {
            k: last[k]
            for k in (
                "iteration",
                "curriculum",
                "actor_domain_sampling",
                "actor_domain_level_cumulative",
                "wm_recent_history_domain_level_cumulative",
            )
        }
        if last
        else {},
    )
    if (run / "failure.json").exists():
        lines += ["", f"实际异常：`{read(run / 'failure.json')}`。"]
    lines += [
        "",
        "播放最终模型：",
        "",
        "```bash",
        f"cd {ROOT}",
        f"PYTHONPATH=src {sys.executable} tools/play_go2w.py --checkpoint {run / 'evaluation/final.pt'} --demo",
        "```",
    ]
    report.write_text("\n".join(lines) + "\n")
    execute(["git", "add", "--", str(report), str(target)], run / "final_git_add.log")
    execute(
        [
            "git",
            "commit",
            "--only",
            "-m",
            "Report distance curriculum continuation and physical rewards",
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
        dict(training_budget_complete=complete, pushed=True, report=str(report)),
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
