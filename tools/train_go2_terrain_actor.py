"""Three-hour guarded continuation of the audited experiment B WM warmstart.

Uses SuperMimicLearner.train() for every optimizer update. Actor checkpoints and
physics evaluations are separate from the source WM warmstart directory.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_go2_terrain_b import Coverage, digest
from validate_go2_migration import ResourceMonitor, dump

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/longrun_B/warmstart/ckpt_002000.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--from-wm", type=Path, default=SOURCE)
    p.add_argument("--resume", type=Path)
    p.add_argument("--seconds", type=float, default=10800.0)
    p.add_argument("--iterations", type=int, default=50000)
    p.add_argument("--critic-warmup", type=int, default=300)
    p.add_argument("--ramp", type=int, default=500)
    p.add_argument("--eval-period", type=int, default=1000)
    args = p.parse_args()
    if args.seconds <= 0 or args.eval_period < 100:
        raise ValueError("Require a positive duration and evaluation interval >=100")
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        raise ValueError("Resume requires an existing run directory")
    out = args.output
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config", overrides=["task=go2_velocity/longrun_B"])
    OmegaConf.set_struct(cfg.algo, False)
    cfg.algo.iterations = args.iterations
    cfg.algo.wm_only_iters = 2000
    cfg.algo.critic_warmup_iters = args.critic_warmup
    cfg.algo.actor_ramp_iters = args.ramp
    cfg.algo.ckpt_every = 250
    cfg.algo.numbered_ckpt_every = 1000
    cfg.algo.log_every = 25
    if args.resume:
        cfg.env.commands.velocity.terrain_phase = "skill"
    from unilab.base import registry

    from unisupermimic.tasks.velocity.go2 import registry as go2_registry

    registry.ensure_registries()
    env_cfg = registry.materialize_env_config(go2_registry.TASK_NAME)
    apply_cfg_overrides(
        env_cfg,
        EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override(),
    )
    monitor = ResourceMonitor()
    monitor.thread.start()
    process = psutil.Process()
    process.cpu_percent()
    started = time.monotonic()
    env, learner = None, None
    stop_reason = "iteration_limit"
    checkpoint = args.resume or args.from_wm
    source_sha = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    dump(
        out / ("invocation_resume.json" if args.resume else "invocation.json"),
        dict(
            argv=sys.argv,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            worktree_changes=subprocess.check_output(["git", "status", "--short"], text=True),
            source_checkpoint=str(checkpoint.resolve()),
            source_sha256=source_sha,
            explicit_user_actor_authorization=True,
            prior_gradient_gate_passed=False,
            plan="2048 envs; H8, H32; critic warmup then ramped actor; skill curriculum",
            duration_s=args.seconds,
            actor_lr_scale=0.3,
            physics_guard="two consecutive severe failures",
        ),
    )
    try:
        env = make_go2_env(env_cfg, 2048)
        env.init_state()
        learner = SuperMimicLearner(
            env, build_learner_config(cfg), out, device="cuda", backend="mujoco"
        )
        learner.load(checkpoint)
        if not args.resume:
            if learner.iteration != 2000 or learner.update_counts["actor"] != 0:
                raise ValueError(
                    "Stage transition requires the validated 2000-iteration frozen-actor checkpoint"
                )
            c = env.command_manager.get_term("velocity")
            c.curriculum.phase = "skill"
            c.cfg.terrain_phase = "skill"
            c.curriculum.mastery.zero_()
            c.curriculum.success_streak.zero_()
            c.curriculum.failure_streak.zero_()
            for value in c.curriculum.statistics.values():
                value.zero_()
            c.curriculum.steps.zero_()
            c.curriculum.episode_fall.zero_()
            learner.buffer.mark_reset_boundary(
                torch.ones(env.num_envs, device=learner.device, dtype=torch.bool)
            )
            env.reset()
            learner.obs_hist.zero_()
            cfg.env.commands.velocity.terrain_phase = "skill"
            learner.resume_mode += (
                "; explicit_wm_to_skill_transition; mastery_and_skill_stats_reset"
            )
        OmegaConf.save(cfg, out / ("config_resume.yaml" if args.resume else "config.yaml"))
        coverage = Coverage(learner)
        # Fixed blind inputs isolate parameter drift from changing replay distributions.
        start, history = learner.buffer.sample_starts(1024, learner.cfg.hist)
        fixed_hist = learner.mdp.obs(
            unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]
        ).detach()
        fixed_goal = start["g"].detach()
        with torch.no_grad():
            reference_action = learner.pi(fixed_hist, fixed_goal).detach().clone()
        baseline_saturation = float((reference_action.abs() > 0.98).float().mean())
        stable = dict(
            pi=copy.deepcopy(learner.pi.state_dict()),
            optimizer=copy.deepcopy(learner.opt_pi.state_dict()),
            iteration=learner.iteration,
        )
        stable_path = out / "stable_actor.pt"
        torch.save(stable, stable_path)
        strikes = 0
        next_eval = 50
        last_evaluated = -1
        torch.cuda.reset_peak_memory_stats()

        def callback(l):
            nonlocal strikes, next_eval, last_evaluated, stop_reason, stable
            coverage.observe()
            elapsed = time.monotonic() - started
            actor_updates = l.update_counts["actor"]
            if l.iteration % 25 == 0:
                with torch.no_grad():
                    current = l.pi(fixed_hist, fixed_goal)
                    drift = float((current - reference_action).square().mean().sqrt())
                    saturation = float((current.abs() > 0.98).float().mean())
                row = dict(
                    iteration=l.iteration,
                    actor_updates=actor_updates,
                    wm_updates=l.update_counts["wm"],
                    critic_updates=l.update_counts["critic"],
                    real_transitions=l.sim_steps,
                    run_wall_s=elapsed,
                    actor_lr=l.opt_pi.param_groups[0]["lr"],
                    actor_drift_rmse=drift,
                    actor_saturation=saturation,
                    baseline_saturation=baseline_saturation,
                    train_reward=l._mean(l.stats["r"]),
                    rss_mib=process.memory_info().rss / 2**20,
                    host_available_mib=psutil.virtual_memory().available / 2**20,
                    cpu_percent=process.cpu_percent(),
                    gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    gpu_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                )
                with (out / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                dump(out / "coverage.json", coverage.summary())
                dump(
                    out / "curriculum.json",
                    env.command_manager.get_term("velocity").curriculum.summary(),
                )
                print(json.dumps(row), flush=True)
                if not torch.isfinite(current).all() or saturation > 0.60 or drift > 1.0:
                    stop_reason = "catastrophic_actor_output"
                    l.stop_requested = True
            if (
                actor_updates >= next_eval
                and actor_updates != last_evaluated
                and not l.stop_requested
            ):
                eval_out = out / f"physics_eval_{l.iteration:06d}"
                eval_checkpoint = l.save(out / "evaluation_policy.pt", include_optimizers=False)
                command = [
                    sys.executable,
                    str(ROOT / "tools/evaluate_go2_terrain_policy.py"),
                    "--mode",
                    "fixed",
                    "--checkpoint",
                    str(eval_checkpoint),
                    "--output",
                    str(eval_out),
                ]
                print(
                    json.dumps(
                        dict(
                            event="physical_evaluation",
                            iteration=l.iteration,
                            actor_updates=actor_updates,
                        )
                    ),
                    flush=True,
                )
                with (out / f"physics_eval_{l.iteration:06d}.log").open("w") as f:
                    done = subprocess.run(
                        command,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=ROOT,
                        timeout=300,
                        check=False,
                    )
                if done.returncode:
                    raise RuntimeError(
                        f"Physics safety evaluation failed: {eval_out}, status {done.returncode}"
                    )
                episodes = json.loads((eval_out / "fixed_evaluation.json").read_text())
                flat = [x for x in episodes if x["level"] == 0]
                easy = [x for x in episodes if x["level"] == 1]

                def avg(data, key):
                    return sum(x[key] for x in data) / len(data)

                metrics = dict(
                    iteration=l.iteration,
                    actor_updates=actor_updates,
                    flat_xy_rmse=avg(flat, "xy_rmse_m_s"),
                    flat_fall_rate=avg(flat, "fall"),
                    level1_xy_rmse=avg(easy, "xy_rmse_m_s"),
                    level1_fall_rate=avg(easy, "fall"),
                    evaluation_path=str(eval_out),
                )
                severe = (
                    metrics["flat_xy_rmse"] > 0.13
                    or metrics["flat_fall_rate"] >= 0.20
                    or metrics["level1_xy_rmse"] > 0.20
                    or metrics["level1_fall_rate"] >= 0.20
                )
                strikes = strikes + 1 if severe else 0
                metrics.update(severe=severe, consecutive_strikes=strikes)
                with (out / "physics_guard.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(dict(event="physics_guard", **metrics)), flush=True)
                if not severe:
                    stable = dict(
                        pi=copy.deepcopy(l.pi.state_dict()),
                        optimizer=copy.deepcopy(l.opt_pi.state_dict()),
                        iteration=l.iteration,
                    )
                    torch.save(stable, stable_path)
                elif strikes >= 2:
                    stop_reason = "two_consecutive_physical_regressions"
                    l.stop_requested = True
                last_evaluated = actor_updates
                next_eval = actor_updates + (150 if actor_updates < 200 else args.eval_period)
            if (
                process.memory_info().rss > 6.5 * 2**30
                or psutil.virtual_memory().available < 600 * 2**20
            ):
                stop_reason = "host_memory_guard"
                l.stop_requested = True
            if shutil.disk_usage(out).free < 30 * 2**30:
                stop_reason = "disk_space_guard"
                l.stop_requested = True
            if elapsed >= args.seconds:
                stop_reason = "wall_time_budget"
                l.stop_requested = True
            if l.stop_requested and stop_reason != "wall_time_budget":
                l.pi.load_state_dict(stable["pi"])
                l.opt_pi.load_state_dict(stable["optimizer"])
                dump(
                    out / "rollback.json",
                    dict(
                        reason=stop_reason,
                        actor_restored_from_iteration=stable["iteration"],
                        wm_and_replay_preserved=True,
                        iteration=l.iteration,
                    ),
                )

        learner.on_iteration = callback
        result = learner.train()
        timing = {
            name: dict(
                count=len(v), mean_ms=sum(v) / len(v), median_ms=float(torch.tensor(v).median())
            )
            for name, v in learner.profile_ms.items()
            if v
        }
        dump(
            out / "summary.json",
            dict(
                result=result,
                stop_reason=stop_reason,
                requested_wall_s=args.seconds,
                process_wall_s=time.monotonic() - started,
                updates=learner.update_counts,
                source_iteration=2000,
                source_transitions=10371072,
                policy_digest=digest(learner.pi),
                profile=timing,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                checkpoint=str((out / "last.pt").resolve()),
            ),
        )
    except BaseException as exc:
        dump(
            out / "failure.json",
            dict(type=type(exc).__name__, error=str(exc), elapsed_s=time.monotonic() - started),
        )
        if learner is not None:
            if "stable" in locals():
                learner.pi.load_state_dict(stable["pi"])
                learner.opt_pi.load_state_dict(stable["optimizer"])
            learner.save(out / "interrupted_rollback.pt")
        raise
    finally:
        monitor.finish(out)
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
