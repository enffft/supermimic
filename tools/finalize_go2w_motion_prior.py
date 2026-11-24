"""Archive a completed Go2W motion-prior run's physical, WM-update and video evidence."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from tools.evaluate_go2w_motion_prior import extend_physical_holdout
from tools.evaluate_go2w_wm import evaluate as evaluate_wm
from unisupermimic.tasks.velocity.go2w.experiment import ROOT, write_json

BENCHMARKS = (
    "go2w_prior_benchmark_2048",
    "go2w_prior_benchmark_4096_no_prior",
    "go2w_prior_benchmark_4096",
    "go2w_prior_benchmark_4096_profile",
)


def load(path):
    return json.loads(path.read_text())


def record_benchmarks(root):
    records = []
    for folder in BENCHMARKS:
        path = root / folder
        if not (path / "training_summary.json").exists():
            continue
        result, initial, progress = (
            load(path / f)
            for f in ("training_summary.json", "initialization.json", "progress.json")
        )
        n = sum(initial["domains"])
        record = {
            "run": folder,
            "num_envs": n,
            "iterations": result["iteration"],
            "seconds_per_iteration": result["training_loop_seconds"] / result["iteration"],
            "real_transitions_per_second": (result["real_transitions"] - 64 * n)
            / result["training_loop_seconds"],
            "gpu_peak_MiB": result["gpu_peak_MiB"],
            "profile_ms_per_call": progress.get("profile_ms_per_call"),
        }
        records.append(record)
    return records


def evaluate_actor(checkpoint, output, traces, holdout=None):
    args = [
        sys.executable,
        "-m",
        "tools.evaluate_go2w",
        "--protocol",
        "motion_prior",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--switches",
        "--traces",
        str(traces),
    ]
    if holdout is not None:
        args += ["--holdout", str(holdout)]
    subprocess.run(args, cwd=ROOT, check=True)


def finalize(run, *, early_stop=False):
    run = run.resolve()
    summary = load(run / "training_summary.json")
    evaluation = run / "evaluation"
    artifact = evaluation / "final.pt"
    if not summary["completed_budget"] and not early_stop:
        raise RuntimeError(
            "Training stopped early. Use --early-stop to finalize the saved checkpoint."
        )
    best = run / "best.pt"
    dataset = evaluation / "holdout.pt"
    final_report = evaluation / "final.json"
    final_traces = evaluation / "final_traces.pt"
    if early_stop and not artifact.exists():
        full = torch.load(run / "last.pt", weights_only=True, map_location="cpu", mmap=True)
        if full["iteration"] != summary["iteration"]:
            raise ValueError("The final recovery checkpoint disagrees with the training summary")
        keys = (
            "pi",
            "wm",
            "args",
            "iteration",
            "wall_time",
            "sim_steps",
            "task_contract",
            "motion_prior",
        )
        temporary = artifact.with_suffix(".tmp")
        torch.save({key: full[key] for key in keys}, temporary)
        temporary.replace(artifact)
        del full
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    if early_stop and not all(path.exists() for path in (final_report, final_traces, dataset)):
        evaluate_actor(artifact, final_report, final_traces, dataset)
    if early_stop:
        metrics = load(final_report)
        metrics.update(
            iteration=summary["iteration"],
            training_seconds=summary["training_loop_seconds"],
            real_transitions=summary["real_transitions"],
            stage_after_evaluation=summary["stage"],
        )
        write_json(final_report, metrics)
        selected_path = run / "best_selection.json"
        score = (
            metrics["flat_pass_rate"] - 0.05 * metrics["fall_rate"] - 0.01 * sum(metrics["rmse"])
        )
        if not selected_path.exists() or score > load(selected_path)["score"]:
            temporary = best.with_suffix(".tmp")
            shutil.copyfile(artifact, temporary)
            temporary.replace(best)
            write_json(
                selected_path, {"checkpoint": str(artifact.relative_to(ROOT)), "score": score}
            )
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    saved = torch.load(dataset, map_location="cpu", weights_only=True)
    if not any(row.get("source") == "reference_pd" for row in saved["cases"]):
        extend_physical_holdout(dataset, artifact)
    del saved
    for name, model in (("final", artifact), ("best", best)):
        if not (evaluation / f"{name}_wm.json").exists():
            evaluate_wm(model, dataset, evaluation / f"{name}_wm.json")
    if not (evaluation / "best_physical.json").exists():
        evaluate_actor(best, evaluation / "best_physical.json", evaluation / "best_traces.pt")
    for tag in ("final", "best"):
        output = evaluation / f"{tag}_videos"
        if not all((output / f"{name}.mp4").exists() for name in ("flat", "coupled")):
            child_env = {**os.environ, "MUJOCO_GL": "egl"}
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.render_go2w_evaluation",
                    "--traces",
                    str(evaluation / f"{tag}_traces.pt"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=child_env,
                check=True,
            )
    physical = [
        load(p)
        for p in (
            evaluation / "initial.json",
            *sorted(evaluation.glob("iteration_*.json")),
            evaluation / "final.json",
        )
    ]
    final, selected = physical[-1], load(evaluation / "best_physical.json")
    benchmark = record_benchmarks(run.parent)
    write_json(run / "benchmark_summary.json", benchmark)
    progress = load(run / "progress.json")
    base = load(run / "provenance.json")["git_head"]
    wm = load(evaluation / "final_wm.json")
    pools = progress["wm_pool"]
    logged = list(csv.DictReader((run / "train_log.csv").open()))
    lines = [
        "# Go2W Motion Prior 训练报告",
        "",
        f"训练源码为 `{base}`，由 `174c6c11` 开始开发。训练阶段为 `flat_prior`，"
        f"实际训练循环 **{summary['training_loop_seconds']:.1f} s**，进程在 {summary['process_seconds']:.1f} s "
        f"内完成 {summary['iteration']:,} 次迭代和 {summary['real_transitions']:,} 条物理 transition。",
        "4096 个训练环境，UniLab CPU 批量 MuJoCo 与 CUDA Learner。从随机权重开始，单组正式训练。",
        "应用户要求在迭代边界提前停止，评测基于最终落盘 checkpoint。"
        if early_stop
        else "训练达到预定循环时间。",
        "",
        "## 固定真实 MuJoCo 评测",
        "",
        "| 累计训练时间 | 迭代 | Transition | 通过率 | vx RMSE | vy RMSE | wz RMSE | 跌倒率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in physical:
        vx, vy, wz = record["rmse"]
        lines.append(
            f"| {record['training_seconds'] / 3600:.3f} h | {record['iteration']} | {record['real_transitions']:,} "
            f"| {record['flat_pass_rate']:.1%} | {vx:.3f} | {vy:.3f} | {wz:.3f} | {record['fall_rate']:.1%} |"
        )
    lines += [
        "",
        "每次固定评测使用普通站姿、Actor 自主动作和三个固定初始 yaw。线速度 RMSE 以 m/s 计，角速度以 rad/s 计。",
        f"最终命令切换通过率为 {final['switch_pass_rate']:.1%}，最佳 checkpoint 重测平地通过率为 {selected['flat_pass_rate']:.1%}。",
        f"`ready_for_next_stage = {str(summary['stage']['ready_for_next_stage']).lower()}`。",
        "",
        "| 运动方向 | 初始 | 最终 | 最佳 checkpoint |",
        "|---|---:|---:|---:|",
    ]
    for group, rate in final["pass_rates"].items():
        lines.append(
            f"| {group} | {physical[0]['pass_rates'][group]:.1%} | {rate:.1%} | {selected['pass_rates'][group]:.1%} |"
        )
    lines += [
        "",
        "## 采集和训练",
        "",
        "本轮源占比统计使用 float32 累加，普通 Actor 计数达到 2²⁴ 后饱和。"
        "因此以计数饱和之前的日志和并发配额核验约 70%／20%／10%，最终 JSON 中的累计占比失真。"
        if summary["collection"]["source_fractions"][0] < 0.70
        and summary["real_transitions"] > 2**24
        else f"实测采集占比为 {summary['collection']['source_fractions']}，顺序为普通 Actor、RSI Actor、参考 PD。",
        f"保护池、普通历史池和近期池的累计采样数见 `progress.json`。近期、普通历史、保护顺序的比例为 "
        f"{[round(x / max(1, sum(pools)), 4) for x in pools]}。",
        f"训练全程峰值 GPU 分配为 {summary['gpu_peak_MiB']:.0f} MiB。训练最后一次日志记录的归一化 WM H8 误差为 "
        f"{logged[-1]['wm_err8'] if logged else '缺失'}。",
        "",
        "| 短测 | 每迭代秒数 | 真实 steps/s | 峰值 MiB |",
        "|---|---:|---:|---:|",
    ]
    for r in benchmark:
        lines.append(
            f"| {r['run']} | {r['seconds_per_iteration']:.3f} | {r['real_transitions_per_second']:.0f} | {r['gpu_peak_MiB']:.0f} |"
        )
    lines += [
        "",
        "## 独立物理 World Model 预测",
        "",
        "| 数据来源 | 类别 | Horizon | 存活 / 样本 | 线速度 RMSE | 角速度 RMSE | 腿位置 RMSE | 腿速度 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in wm["rows"]:
        metrics = row["rmse"]
        values = (
            ("—",) * 4
            if metrics is None
            else tuple(
                f"{metrics[key]:.4f}"
                for key in (
                    "linear_velocity_m_s",
                    "angular_velocity_rad_s",
                    "leg_position_rad",
                    "leg_velocity_rad_s",
                )
            )
        )
        lines.append(
            f"| {row['source']} | {row['family']} | {row['horizon']} | {row['valid']}/{row['total']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## 结果文件",
        "",
        "- `best.pt` 保存推理权重。 `last.pt` 保存优化器、Replay、历史池、阶段及课程等恢复信息。",
        "- `evaluation/final.json` 和 `best_physical.json` 保存每个速度及命令切换的真实评测。",
        "- `evaluation/final_wm.json` 和 `best_wm.json` 保存 H1/H8 独立物理留出测试。",
        "- `evaluation/final_videos/` 和 `best_videos/` 中的画面取自真实 Actor rollout。",
        "",
        "一组正式实验可描述从零学习曲线。对 Motion Prior 或去掉默认姿态奖励的因果贡献，需要匹配对照实验。",
    ]
    (run / "training_report.md").write_text("\n".join(lines) + "\n")
    write_json(
        run / "finalization.json",
        {
            "physical_evaluations": len(physical),
            "ready_for_next_stage": summary["stage"]["ready_for_next_stage"],
            "best_flat_pass_rate": selected["flat_pass_rate"],
            "final_flat_pass_rate": final["flat_pass_rate"],
            "final_fall_rate": final["fall_rate"],
            "final_pass_rates": final["pass_rates"],
            "videos": {
                tag: [str(p.relative_to(run)) for p in (evaluation / f"{tag}_videos").glob("*.mp4")]
                for tag in ("best", "final")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--early-stop", action="store_true", help="Finalize after an authorized graceful stop"
    )
    args = parser.parse_args()
    finalize(args.run, early_stop=args.early_stop)
