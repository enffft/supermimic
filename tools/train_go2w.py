"""Strict H8 SuperMimic runner. Smoke runs never authorize formal training by themselves."""

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.checkpoint import initialize_wm_only
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import (
    BUDGET_SECONDS,
    ROOT,
    WM_FILE_SHA,
    WM_PATH,
    WM_TENSOR_SHA,
    configuration,
    snapshot,
    source_identity,
    write_json,
)


def validate_admission(path, identity, *, accept_direction_diagnostic=False):
    admission = json.loads(path.read_text())
    if admission.get("source_sha256") != identity:
        raise ValueError("Admission source snapshot differs")
    required = (
        "gradient_direction",
        "unit_regression",
        "reward_parity",
        "terrain_collision",
        "wheel_geometry",
        "initialization",
        "4096_smoke",
        "checkpoint_resume",
        "physical_evaluator",
    )
    for name in required:
        item = admission.get("checks", {}).get(name, {})
        evidence = Path(item.get("path", ""))
        direction_override = name == "gradient_direction" and accept_direction_diagnostic
        if (item.get("passed") is not True and not direction_override) or not evidence.is_file():
            raise ValueError(f"Gate not passed: {name}")
        import hashlib

        if hashlib.sha256(evidence.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError(f"Gate evidence changed: {name}")
    return admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50, help="Smoke only")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--accept-direction-diagnostic",
        action="store_true",
        help="Explicit user authorization: retain failed H8 directions as warnings, not admission veto",
    )
    args = parser.parse_args()
    owner, cfg = configuration()
    if args.envs < 1 or args.iterations < 1:
        parser.error("Positive smoke dimensions required")
    identity, _ = source_identity()
    if args.mode == "formal":
        if args.envs != 4096 or args.admission is None:
            parser.error("Formal requires 4096 envs and complete admission")
        validate_admission(
            args.admission, identity, accept_direction_diagnostic=args.accept_direction_diagnostic
        )
    args.output.mkdir(parents=True, exist_ok=False)
    cfg.iterations = 2**31 - 1 if args.mode == "formal" else args.iterations
    cfg.log_backend = "none"
    cfg.log_every = 1 if args.mode == "smoke" else 20
    cfg.ckpt_every = 2000
    cfg.numbered_ckpt_every = 10000
    for name, value in asdict(cfg).items():
        OmegaConf.update(owner, "algo." + name, value, force_add=True)
    owner.algo.num_envs = args.envs
    # No model/batch/buffer reductions in smoke; only env count and iteration limit differ.
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    snapshot(args.output, owner, cfg)
    write_json(
        args.output / "launch.json",
        dict(
            mode=args.mode,
            direction_diagnostic_accepted=args.accept_direction_diagnostic,
            direction_decision="User: 不要一味用gate阻挠，如果代码写好可以直接开始训练"
            if args.accept_direction_diagnostic
            else None,
            envs=args.envs,
            pid=os.getpid(),
            budget_seconds=BUDGET_SECONDS if args.mode == "formal" else None,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    env = learner = None
    last_checkpoint = time.monotonic()
    start = time.monotonic()
    stop = "iteration_budget"
    try:
        env = make_go2w_env(load_config("train"), args.envs)
        env.reset()
        learner = SuperMimicLearner(env, cfg, args.output, device="cuda:0", backend="mujoco")
        if args.resume:
            restored = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
            if restored.get("extra_state", {}).get("formal_source_sha256") != identity:
                raise ValueError("Resume source changed")
            if restored["args"] != asdict(cfg):
                raise ValueError("Resume effective algorithm differs")
            del restored
            learner.load(args.resume)
        else:
            audit = initialize_wm_only(
                learner,
                WM_PATH,
                expected_file_sha256=WM_FILE_SHA,
                expected_tensor_sha256=WM_TENSOR_SHA,
            )
            write_json(args.output / "wm_initialization.json", audit)
        learner.extra_state["formal_source_sha256"] = identity
        learner.extra_state["direction_diagnostic_accepted"] = args.accept_direction_diagnostic
        write_json(
            args.output / "admission.json",
            json.loads(args.admission.read_text()) if args.admission else {"mode": "smoke"},
        )
        write_json(args.output / "task_contract.json", env.supermimic_contract)
        write_json(
            args.output / "replay_manifest.json",
            dict(
                fields={
                    k: dict(
                        shape=list(v.shape), dtype=str(v.dtype), bytes=v.numel() * v.element_size()
                    )
                    for k, v in learner.buffer.data.items()
                },
                total_bytes=sum(v.numel() * v.element_size() for v in learner.buffer.data.values()),
                historical_capacity=cfg.history_capacity,
                history_fraction=cfg.history_fraction,
                historical_source="current run contiguous physical windows only",
            ),
        )

        def assess(label):
            evaluation = args.output / "evaluation"
            evaluation.mkdir(exist_ok=True)
            policy = learner.save(evaluation / f"{label}.pt", include_optimizers=False)

            def run(tool, arguments):
                with (evaluation / f"{label}_{tool}.log").open("w") as stream:
                    subprocess.run(
                        [sys.executable, str(ROOT / "tools" / f"{tool}.py"), *map(str, arguments)],
                        check=True,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        cwd=ROOT,
                    )

            run(
                "evaluate_go2w",
                ["--checkpoint", policy, "--output", evaluation / f"{label}_physical.json"],
            )
            run(
                "evaluate_go2w_wm",
                [
                    "--model",
                    policy,
                    "--holdout",
                    evaluation / "holdout.pt",
                    "--output",
                    evaluation / f"{label}_wm.json",
                ],
            )
            physical = json.loads((evaluation / f"{label}_physical.json").read_text())["by_domain"]
            score = (
                sum(
                    physical[d]["success_rate"]
                    for d in (
                        "flat",
                        "single_step_up",
                        "single_step_down",
                        "multi_stairs_up",
                        "multi_stairs_down",
                    )
                )
                / 5
            )
            prior = learner.extra_state.get("best_physical_score", 0.0)
            if score > prior and physical["flat"]["success_rate"] >= 0.7:
                import shutil

                shutil.copyfile(policy, args.output / "best.pt")
                learner.extra_state["best_physical_score"] = score
                write_json(
                    args.output / "best_selection.json",
                    dict(
                        score=score,
                        source=str(policy),
                        criterion="mean flat/four stair physical success; flat >= 70%; strictly improves",
                    ),
                )

        if args.mode == "formal" and args.resume is None:
            assess("initial")
        last_assessment = learner.wall_time

        def request_stop(signum, frame):
            nonlocal stop
            stop = f"signal_{signum}"
            learner.stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        def observe(current):
            nonlocal stop, last_checkpoint, last_assessment
            if args.mode == "formal" and current.wall_time >= BUDGET_SECONDS:
                stop = "effective_training_budget"
                current.stop_requested = True
            status = dict(
                iteration=current.iteration,
                loop_seconds=current.wall_time,
                updates=dict(current.update_counts),
                transitions=current.sim_steps,
                elapsed_seconds=time.monotonic() - start,
                rss_MiB=psutil.Process().memory_info().rss / 2**20,
                available_RAM_MiB=psutil.virtual_memory().available / 2**20,
                gpu_allocated_MiB=torch.cuda.memory_allocated() / 2**20,
                gpu_reserved_MiB=torch.cuda.memory_reserved() / 2**20,
                gpu_peak_MiB=torch.cuda.max_memory_allocated() / 2**20,
                history_size=current.buffer.history.size,
                wm_actual_recent_history=current.buffer.history_actual,
                recent_error8=current.extra_state.get("wm_recent_error8"),
                recent_loss=current.extra_state.get("wm_recent_loss"),
                historical_loss=current.extra_state.get("wm_historical_loss"),
                historical_error8=current.extra_state.get("wm_historical_error8"),
            )
            write_json(args.output / "progress.json", status)
            with (args.output / "system_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
            if args.mode == "formal" and current.wall_time - last_assessment >= 1800:
                current.save()
                assess(f"iteration_{current.iteration:06d}")
                last_assessment = current.wall_time
            if time.monotonic() - last_checkpoint >= 600:
                current.save()
                last_checkpoint = time.monotonic()

        learner.on_iteration = observe
        summary = learner.train()
        learner.save()
        if args.mode == "formal":
            assess("final")
        summary.update(
            mode=args.mode,
            stop_reason=stop,
            updates=dict(learner.update_counts),
            peak_CPU_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            peak_GPU_MiB=torch.cuda.max_memory_allocated() / 2**20,
            formal_budget_complete=args.mode == "formal" and learner.wall_time >= BUDGET_SECONDS,
            ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        write_json(args.output / "training_summary.json", summary)
    except BaseException as exc:
        write_json(
            args.output / "failure.json",
            dict(
                error=repr(exc),
                elapsed_seconds=time.monotonic() - start,
                iteration=getattr(learner, "iteration", 0),
                loop_seconds=getattr(learner, "wall_time", 0),
                peak_CPU_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                peak_GPU_MiB=torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0,
            ),
        )
        # last.pt remains atomic and recoverable. Do not serialize a partly applied update.
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
