"""Lifecycle adapter for the fresh or same-profile Go2W Motion Prior experiment."""

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, fields

import psutil
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import ROOT, snapshot, write_json
from unisupermimic.tasks.velocity.go2w.motion_prior import (
    MotionBank,
    ReferenceCollection,
    apply_stage,
)
from unisupermimic.tasks.velocity.go2w.training_stages import TrainingStage


def owner_config():
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        return compose("config", overrides=["task=go2w_velocity/motion_prior"])


def detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: detached(item) for key, item in value.items()}
    return value


def run(args):
    owner = owner_config()
    ns = {field.name for field in fields(SuperMimicConfig)}
    raw = OmegaConf.to_container(owner.algo, resolve=True)
    cfg = SuperMimicConfig(**{key: value for key, value in raw.items() if key in ns})
    stage_cfg = OmegaConf.to_container(owner.stage, resolve=True)
    prior_cfg = OmegaConf.to_container(owner.motion_prior, resolve=True)
    if args.max_stage:
        stage_cfg["max_stage"] = args.max_stage
    if args.prior_weight is not None:
        prior_cfg["lambda_prior"] = args.prior_weight
    if args.no_prior:
        prior_cfg["enabled"] = False
    cfg.profile = args.profile_sections
    cfg.iterations = args.iterations if args.mode == "smoke" else 2**31 - 1
    if args.mode == "smoke":
        cfg.wm_only_iters = args.smoke_warmup
    cfg.log_backend = "none"
    cfg.log_every = (1 if args.gradient_diagnostics else 10) if args.mode == "smoke" else 50
    cfg.ckpt_every = 2**31 - 1
    cfg.numbered_ckpt_every = 2**31 - 1
    if cfg.init_wm_from or cfg.init_actor_from or raw["resume"]:
        raise ValueError("Injected pretrained model weights are forbidden in this profile")
    if cfg.horizon != prior_cfg["horizon"]:
        raise ValueError("Prior and imagination horizons differ")
    stage = TrainingStage(
        stage_cfg["initial"], stage_cfg["max_stage"], stage_cfg["auto_transition"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("The Go2W training learner requires CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    owner.stage = stage_cfg
    owner.motion_prior = prior_cfg
    for key, value in asdict(cfg).items():
        OmegaConf.update(owner, f"algo.{key}", value, force_add=True)
    owner.algo.num_envs = args.envs
    source_identity = snapshot(args.output, owner, cfg)
    process_start = time.monotonic()
    write_json(
        args.output / "launch.json",
        {
            "mode": args.mode,
            "profile": "motion_prior",
            "envs": args.envs,
            "fresh": args.resume is None,
            "resume": str(args.resume) if args.resume else None,
            "source_sha256": source_identity,
            "budget_seconds": args.seconds,
            "pid": os.getpid(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    env = learner = None
    try:
        env = make_go2w_env(load_config("motion_prior"), args.envs)
        bank = MotionBank(ROOT / prior_cfg["bank"], "cuda:0", prior_cfg)
        env.reference_collection = collection = ReferenceCollection(
            bank
            if env.device == bank.nominal.device
            else MotionBank(ROOT / prior_cfg["bank"], env.device, prior_cfg),
            env,
            prior_cfg,
            stage,
        )
        env.command_manager.get_term("velocity").reference_collection = collection
        env.reset()
        learner = SuperMimicLearner(env, cfg, args.output, device="cuda:0", backend="mujoco")
        learner.motion_prior = bank if prior_cfg["enabled"] else None
        learner.motion_prior_diagnostics = args.gradient_diagnostics
        learner.buffer.protected_success_fraction = prior_cfg["protected_success_fraction"]
        learner.buffer.history.protection_enabled = True
        if args.resume:
            learner.load(args.resume)
            if args.mode == "smoke":
                cfg.iterations += learner.iteration
        apply_stage(learner, stage, prior_cfg)
        term = env.command_manager.get_term("velocity")
        write_json(args.output / "task_contract.json", env.supermimic_contract)
        write_json(
            args.output / "initialization.json",
            {
                "wm_actor_critic_initialized_from_seed": args.resume is None,
                "iteration": learner.iteration,
                "replay_rows": learner.buffer.size,
                "history_windows": learner.buffer.history.size,
                "bank_identity": bank.identity,
                "qualified": sum(row["quality_status"] == "qualified" for row in bank.rows),
                "dynamic_candidate": sum(row["quality_status"] == "candidate" for row in bank.rows),
                "physics_class": f"{type(env.backend).__module__}.{type(env.backend).__name__}",
                "environment_device": str(env.device),
                "learner_device": str(learner.device),
                "torch": str(torch.__version__),
                "native_threads": getattr(env.backend, "_n_threads", None),
                "domains": torch.bincount(term.command_domain, minlength=8).tolist(),
            },
        )
        base_seconds = learner.wall_time
        target = base_seconds + args.seconds if args.seconds is not None else None
        process = psutil.Process()
        process.cpu_percent()
        assess_period = 1800.0
        last_assessment = base_seconds
        last_recovery = base_seconds
        best_score = float(learner.extra_state.get("flat_best_score", -1.0))
        evaluation = args.output / "evaluation"
        evaluation.mkdir()
        stopped_by_signal = False
        warmup_wall = None

        def gpu_utilization():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                capture_output=True,
                check=False,
            )
            return (
                result.stdout.splitlines()[0].strip()
                if result.returncode == 0 and result.stdout
                else None
            )

        def assess(label):
            nonlocal best_score
            checkpoint = learner.save(evaluation / f"{label}.pt", include_optimizers=False)
            path = evaluation / f"{label}.json"
            command = [
                sys.executable,
                "-m",
                "tools.evaluate_go2w",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(path),
                "--protocol",
                "motion_prior",
                "--seeds",
                "2026",
                "2027",
                "2028",
            ]
            if label == "final":
                command += [
                    "--traces",
                    str(evaluation / "final_traces.pt"),
                    "--holdout",
                    str(evaluation / "holdout.pt"),
                ]
            if stage.name == "flat_autonomous" or label == "final":
                command.append("--switches")
            if stage.name == "terrain":
                command.append("--terrain-review")
            with (evaluation / f"{label}.log").open("w") as stream:
                subprocess.run(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
            metrics = json.loads(path.read_text())
            old_review = stage.terrain_review
            transitioned = stage.assess(metrics, stage_cfg | prior_cfg)
            if transitioned or stage.terrain_review != old_review:
                apply_stage(learner, stage, prior_cfg, reset=True)
            metrics["iteration"] = learner.iteration
            metrics["training_seconds"] = learner.wall_time
            metrics["real_transitions"] = learner.sim_steps
            metrics["stage_after_evaluation"] = stage.state_dict()
            write_json(path, metrics)
            score = (
                metrics["flat_pass_rate"]
                - 0.05 * metrics["fall_rate"]
                - 0.01 * sum(metrics["rmse"])
            )
            if score > best_score:
                best_score = score
                learner.extra_state["flat_best_score"] = score
                temp = args.output / "best.pt.tmp"
                shutil.copyfile(checkpoint, temp)
                temp.replace(args.output / "best.pt")
                write_json(
                    args.output / "best_selection.json",
                    {"checkpoint": str(checkpoint), "score": score},
                )
            return metrics

        def stop(signum, frame):
            nonlocal stopped_by_signal
            stopped_by_signal = True
            learner.stop_requested = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        def observe(current):
            nonlocal last_assessment, last_recovery, warmup_wall
            seconds = current.wall_time
            if warmup_wall is None:
                warmup_wall = time.monotonic() - process_start - (seconds - base_seconds)
            stage.training_seconds = seconds
            apply_stage(current, stage, prior_cfg)
            if target is not None and seconds >= target:
                current.stop_requested = True
            due = (
                current.iteration % current.cfg.log_every == 0
                or current.iteration <= 3
                or current.stop_requested
            )
            if due:
                recent, old = current.buffer.history_actual
                protected = current.buffer.protected_actual
                domain = torch.bincount(term.command_domain, minlength=8)
                if stage.name != "terrain" and int(domain[1:].sum()):
                    raise RuntimeError("Flat stage acquired a nonflat environment")
                status = dict(
                    iteration=current.iteration,
                    training_seconds=seconds,
                    real_transitions=current.sim_steps,
                    stage=stage.state_dict(),
                    updates=current.update_counts,
                    domains=domain.tolist(),
                    collection=collection.metrics(),
                    prior=detached(current.extra_state.get("motion_prior_metrics")),
                    history_size=current.buffer.history.size,
                    wm_pool=[recent, old - protected, protected],
                    wm_scopes=detached(current.buffer.wm_scope_counts),
                    actor_scopes=detached(current.extra_state.get("actual_actor_start_scopes")),
                    actor_sampling=detached(current.buffer.last_actor_sampling),
                    curriculum_levels=torch.bincount(
                        term.curriculum.terrain_level, minlength=10
                    ).tolist(),
                    wm_h1_state_rmse=detached(current.extra_state.get("wm_h1_state_rmse")),
                    gpu_peak_MiB=torch.cuda.max_memory_allocated() / 2**20,
                    gpu_allocated_MiB=torch.cuda.memory_allocated() / 2**20,
                    gpu_reserved_MiB=torch.cuda.memory_reserved() / 2**20,
                    cpu_percent=process.cpu_percent(),
                    gpu_utilization=gpu_utilization()
                    if current.iteration % 200 == 0 or current.stop_requested
                    else None,
                )
                if current.profile_ms:
                    status["profile_ms_per_call"] = {
                        key: sum(values) / len(values) for key, values in current.profile_ms.items()
                    }
                write_json(args.output / "progress.json", status)
                with (args.output / "system_metrics.jsonl").open("a") as log:
                    log.write(json.dumps(status) + "\n")
            # No evaluation or checkpoint elapsed time is counted in learner.wall_time.
            if args.mode == "formal" and not current.stop_requested:
                if seconds - last_assessment >= assess_period:
                    assess(f"iteration_{current.iteration:06d}")
                    last_assessment = seconds
                if seconds - last_recovery >= 1800.0:
                    current.save()
                    last_recovery = seconds

        learner.on_iteration = observe
        if args.mode == "formal":
            assess("initial")
        summary = learner.train()
        peak = torch.cuda.max_memory_allocated() / 2**20
        if args.mode == "formal" and not stopped_by_signal:
            assess("final")
            learner.save()
        summary.update(
            completed_budget=target is not None and learner.wall_time >= target,
            training_loop_seconds=learner.wall_time - base_seconds,
            process_seconds=time.monotonic() - process_start,
            setup_and_warmup_seconds=warmup_wall,
            iteration=learner.iteration,
            real_transitions=learner.sim_steps,
            updates=learner.update_counts,
            stage=stage.state_dict(),
            gpu_peak_MiB=peak,
            rss_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            collection=collection.metrics(),
        )
        write_json(args.output / "training_summary.json", summary)
    except BaseException as error:
        write_json(
            args.output / "failure.json",
            {
                "error": repr(error),
                "iteration": getattr(learner, "iteration", None),
                "elapsed_seconds": time.monotonic() - process_start,
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
