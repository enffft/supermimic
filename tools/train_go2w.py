"""Strict H8 SuperMimic runner. Smoke runs never authorize formal training by themselves."""

import argparse
import hashlib
import json
import math
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


def loop_budget_target(previous_seconds, additional_seconds=None):
    if additional_seconds is None:
        return float(BUDGET_SECONDS)
    if not math.isfinite(additional_seconds) or additional_seconds <= 0:
        raise ValueError("Additional training budget must be finite and positive")
    if not math.isfinite(previous_seconds) or previous_seconds < 0:
        raise ValueError("Invalid restored effective loop time")
    return previous_seconds + additional_seconds


def validate_resume_manifest(
    parent, recorded_identity, current_manifest, *, curriculum_migration=False
):
    manifest = json.loads((parent / "source_manifest.json").read_text())
    if manifest["sha256"] != recorded_identity:
        raise ValueError("Resume checkpoint/source manifest identity differs")
    old = manifest["files"]
    changed = sorted(
        path
        for path in old.keys() | current_manifest.keys()
        if old.get(path) != current_manifest.get(path)
    )
    # Evaluation/runner/test changes can be snapshotted separately. Exact task,
    # learner, replay, assets, configuration and dependency files remain mandatory.
    authorized = (
        {
            "src/unisupermimic/tasks/velocity/go2w/curriculum.py",
            "src/unisupermimic/tasks/velocity/go2w/manager_terms.py",
            "src/unisupermimic/tasks/velocity/go2w/env.py",
            "src/unisupermimic/tasks/velocity/go2w/diagnostics.py",
            "src/unisupermimic/tasks/velocity/go2w/migration.py",
            "src/unisupermimic/rl/algos/supermimic/replay.py",
            "src/unisupermimic/rl/algos/supermimic/learner.py",
            "src/unisupermimic/conf/supermimic/task/go2w_velocity/train.yaml",
        }
        if curriculum_migration
        else set()
    )
    forbidden = [
        path
        for path in changed
        if not path.startswith(("tools/", "tests/")) and path not in authorized
    ]
    if forbidden:
        raise ValueError(f"Resume training source changed: {forbidden}")
    return changed


def nested_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50, help="Smoke only")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--migrate-curriculum",
        action="store_true",
        help="Explicit old-window to new distance/Actor-domain sampling experiment",
    )
    parser.add_argument(
        "--additional-seconds",
        type=float,
        help="Resume only: add this many actual training-loop seconds to restored time",
    )
    parser.add_argument(
        "--accept-direction-diagnostic",
        action="store_true",
        help="Explicit user authorization: retain failed H8 directions as warnings, not admission veto",
    )
    parser.add_argument(
        "--profile", choices=("continuation", "motion_prior"), default="continuation"
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seconds", type=float, help="Motion prior training-loop budget")
    parser.add_argument("--max-stage", choices=("flat_prior", "flat_autonomous", "terrain"))
    parser.add_argument("--prior-weight", type=float)
    parser.add_argument("--smoke-warmup", type=int, default=8)
    parser.add_argument(
        "--no-prior", action="store_true", help="Smoke only: disable the actor prior objective"
    )
    parser.add_argument("--gradient-diagnostics", action="store_true")
    parser.add_argument("--profile-sections", action="store_true")
    args = parser.parse_args()
    if args.profile == "motion_prior":
        if args.fresh == bool(args.resume):
            parser.error("Specify exactly one of --fresh and --resume")
        if (
            args.admission
            or args.migrate_curriculum
            or args.additional_seconds
            or args.accept_direction_diagnostic
        ):
            parser.error("Continuation options do not apply to the motion prior profile")
        if args.envs < 1 or args.iterations < 1 or args.smoke_warmup < 0:
            parser.error("Invalid smoke dimensions")
        if args.prior_weight is not None and (
            not math.isfinite(args.prior_weight) or args.prior_weight < 0
        ):
            parser.error("Prior weight must be finite and nonnegative")
        if args.mode == "formal":
            if (
                args.envs != 4096
                or args.seconds is None
                or not math.isfinite(args.seconds)
                or args.seconds <= 0
            ):
                parser.error(
                    "Formal motion prior runs require 4096 environments and a positive --seconds budget"
                )
            if args.no_prior or args.gradient_diagnostics or args.profile_sections:
                parser.error("Use the ablation and frequent profiling options in smoke mode")
        from tools.train_go2w_motion_prior import run

        return run(args)
    if args.fresh or args.seconds is not None or args.max_stage is not None:
        parser.error("Fresh-run budget and stage options require --profile motion_prior")
    owner, cfg = configuration()
    if args.envs < 1 or args.iterations < 1:
        parser.error("Positive smoke dimensions required")
    if args.additional_seconds is not None:
        if args.resume is None or args.mode != "formal":
            parser.error("--additional-seconds requires formal --resume")
        loop_budget_target(0.0, args.additional_seconds)
    if args.migrate_curriculum and not args.resume:
        parser.error("--migrate-curriculum requires a complete --resume checkpoint")
    identity, manifest = source_identity()
    if args.mode == "formal":
        if args.envs != 4096 or args.admission is None:
            parser.error("Formal requires 4096 envs and complete admission")
        validate_admission(
            args.admission, identity, accept_direction_diagnostic=args.accept_direction_diagnostic
        )
    args.output.mkdir(parents=True, exist_ok=False)
    cfg.iterations = 2**31 - 1 if args.mode == "formal" else args.iterations
    if args.mode == "smoke" and args.resume:
        header = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        cfg.iterations = header["iteration"] + args.iterations
        del header
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
            budget_seconds=(
                args.additional_seconds if args.additional_seconds is not None else BUDGET_SECONDS
            )
            if args.mode == "formal"
            else None,
            resume_from=str(args.resume.resolve()) if args.resume else None,
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
            parent_identity = restored.get("extra_state", {}).get("formal_source_sha256")
            changed_tools = validate_resume_manifest(
                args.resume.resolve().parent,
                parent_identity,
                manifest,
                curriculum_migration=args.migrate_curriculum,
            )
            replay_retained = True
            if args.migrate_curriculum:
                from unisupermimic.tasks.velocity.go2w.migration import migrate_checkpoint

                migration_path, replay_retained = migrate_checkpoint(
                    learner, args.resume, args.output, identity, smoke=args.mode == "smoke"
                )
                learner.load(migration_path)
            else:
                if restored["args"] != asdict(cfg):
                    raise ValueError("Resume effective algorithm differs")
                learner.load(args.resume)
            unchanged = {
                "actor": nested_digest(learner.pi.state_dict())
                == nested_digest(restored.get("pi_live", restored["pi"])),
                "critic": nested_digest(learner.critic.state_dict())
                == nested_digest(restored["critic"]),
                "wm": nested_digest(learner.wm.state_dict()) == nested_digest(restored["wm"]),
                "wm_optimizer": nested_digest(learner.opt_wm.state_dict())
                == nested_digest(restored["opt_wm"]),
                "actor_optimizer": nested_digest(learner.opt_pi.state_dict())
                == nested_digest(restored["opt_pi"]),
                "critic_optimizer": nested_digest(learner.opt_v.state_dict())
                == nested_digest(restored["opt_v"]),
                "replay_size": learner.buffer.size
                == (restored["replay"]["size"] if replay_retained else 0),
                "history_size": learner.buffer.history.size
                == restored["replay"]["history"]["size"],
                "history_data_and_rng": nested_digest(learner.buffer.history.state_dict())
                == nested_digest(restored["replay"]["history"]),
                "value_normalization": nested_digest(
                    dict(mean=learner._v_mean, std=learner._v_std, initialized=learner._v_init)
                )
                == nested_digest(restored["value_normalization"]),
                "iteration_time_and_transitions": (
                    learner.iteration,
                    learner.wall_time,
                    learner.sim_steps,
                    learner.step_accumulator,
                )
                == (
                    restored["iteration"],
                    restored["wall_time"],
                    restored["sim_steps"],
                    restored["step_accumulator"],
                ),
                "updates": learner.update_counts == restored["update_counts"],
                "curriculum_levels": (
                    not bool(env.command_manager.get_term("velocity").curriculum.mastery.any())
                )
                if args.migrate_curriculum
                else torch.equal(
                    env.command_manager.get_term("velocity").curriculum.skill_levels.cpu(),
                    restored["curriculum"]["tensors"]["skill_levels"],
                ),
            }
            if not all(unchanged.values()):
                raise RuntimeError(f"Resume restoration mismatch: {unchanged}")
            write_json(
                args.output / "resume_initialization.json",
                dict(
                    source_checkpoint=str(args.resume.resolve()),
                    source_identity=parent_identity,
                    current_identity=identity,
                    allowed_tool_test_changes=changed_tools,
                    unchanged_training_source=not args.migrate_curriculum,
                    explicit_curriculum_migration=args.migrate_curriculum,
                    restoration_checks=unchanged,
                    parent_iteration=learner.iteration,
                    parent_loop_seconds=learner.wall_time,
                    parent_updates=dict(learner.update_counts),
                    parent_transitions=learner.sim_steps,
                    replay_size=learner.buffer.size,
                    history_size=learner.buffer.history.size,
                    semantics=learner.resume_mode,
                ),
            )
            del restored
        else:
            audit = initialize_wm_only(
                learner,
                WM_PATH,
                expected_file_sha256=WM_FILE_SHA,
                expected_tensor_sha256=WM_TENSOR_SHA,
            )
            write_json(args.output / "wm_initialization.json", audit)
        base_loop_seconds = learner.wall_time
        base_updates = dict(learner.update_counts)
        base_transitions = learner.sim_steps
        base_iteration = learner.iteration
        target_loop_seconds = (
            loop_budget_target(base_loop_seconds, args.additional_seconds)
            if args.mode == "formal"
            else base_loop_seconds
        )
        write_json(
            args.output / "budget.json",
            dict(
                base_loop_seconds=base_loop_seconds,
                target_loop_seconds=target_loop_seconds,
                additional_seconds=args.additional_seconds,
                requested_session_seconds=target_loop_seconds - base_loop_seconds,
            ),
        )
        learner.extra_state["continuation_parent"] = (
            str(args.resume.resolve()) if args.resume else None
        )
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

        if args.mode == "formal":
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
            if args.mode == "formal" and current.wall_time >= target_loop_seconds:
                stop = "effective_training_budget"
                current.stop_requested = True
            status = dict(
                iteration=current.iteration,
                loop_seconds=current.wall_time,
                session_loop_seconds=current.wall_time - base_loop_seconds,
                target_loop_seconds=target_loop_seconds,
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
            formal_budget_complete=args.mode == "formal"
            and learner.wall_time >= target_loop_seconds,
            session_loop_seconds=learner.wall_time - base_loop_seconds,
            requested_session_seconds=target_loop_seconds - base_loop_seconds,
            parent_loop_seconds=base_loop_seconds,
            session_iterations=learner.iteration - base_iteration,
            session_transitions=learner.sim_steps - base_transitions,
            session_updates={
                key: learner.update_counts[key] - base_updates[key] for key in base_updates
            },
            resume_from=str(args.resume.resolve()) if args.resume else None,
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
