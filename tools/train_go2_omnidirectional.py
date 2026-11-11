"""Bounded continuation with all-direction independent physical rollback guard.

All WM/H32/critic updates remain in SuperMimicLearner.train(). This orchestration
never replaces the actor, invents teacher transitions or changes the reward.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_go2_command_coverage import CommandCoverage
from go2_omni_common import ROOT, configuration, environment
from validate_go2_migration import ResourceMonitor, dump


def physical_regressions(evaluation, baseline):
    """Compare every measured goal; inherited failures are not new regressions."""
    regressions = []
    for name, value in evaluation.items():
        old = baseline[name]
        for key in ("xy_rmse_m_s", "yaw_rmse_rad_s"):
            limit = max(old[key] * 1.7, old[key] + (0.12 if key.startswith("yaw") else 0.06))
            if value[key] > limit:
                regressions.append(dict(scenario=name, metric=key, value=value[key], limit=limit))
        if value["fall"] >= 0.2 and value["fall"] >= old["fall"] + 0.2:
            regressions.append(
                dict(scenario=name, metric="fall", value=value["fall"], baseline=old["fall"])
            )
        if (
            name.endswith("_cmd0")
            and name.startswith("stairs_")
            and old["crossing_passed"] >= 0.8
            and value["crossing_passed"] < old["crossing_passed"] - 0.4
        ):
            regressions.append(
                dict(
                    scenario=name,
                    metric="crossing_passed",
                    value=value["crossing_passed"],
                    baseline=old["crossing_passed"],
                )
            )
    return regressions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument("--wm-adaptation", type=int, default=100)
    p.add_argument("--eval-period", type=int, default=500)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.iterations <= 0 or a.seconds <= 0 or a.wm_adaptation < 0 or a.eval_period < 100:
        raise ValueError("Invalid bounded budget")
    a.output.mkdir(parents=True, exist_ok=a.resume)
    metadata = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = configuration()
    if a.resume:
        saved = metadata["extra_state"]["omni_stage_schedule"]
        cfg.algo.wm_only_iters = saved["wm_only_iters"]
        cfg.algo.critic_warmup_iters = saved["critic_warmup_iters"]
    else:
        cfg.algo.wm_only_iters = int(metadata["iteration"]) + a.wm_adaptation
    cfg.algo.iterations = int(metadata["iteration"]) + a.iterations
    cfg.algo.ckpt_every = 250
    cfg.algo.log_every = 25
    source = dict(
        iteration=metadata["iteration"],
        sim_steps=metadata["sim_steps"],
        wall_time=metadata["wall_time"],
        updates=metadata["update_counts"].copy(),
    )
    del metadata
    OmegaConf.save(cfg, a.output / "config.yaml")
    started = time.monotonic()
    monitor = ResourceMonitor()
    monitor.thread.start()
    env = None
    l = None
    reason = "iteration_budget"
    proc = psutil.Process()
    proc.cpu_percent()
    try:
        env = environment(cfg)
        env.init_state()
        l = SuperMimicLearner(
            env, build_learner_config(cfg), a.output, device="cuda", backend="mujoco"
        )
        l.load(a.checkpoint)
        l.extra_state["omni_stage_schedule"] = dict(
            wm_only_iters=cfg.algo.wm_only_iters, critic_warmup_iters=cfg.algo.critic_warmup_iters
        )
        coverage = CommandCoverage(l)
        start, hist = l.buffer.sample_starts(512, l.cfg.hist)
        fixed = l.mdp.obs(
            unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"]
        ).detach()
        goal = start["g"]
        with torch.no_grad():
            reference = l.pi(fixed, goal).detach().clone()
        stable = dict(
            pi=copy.deepcopy(l.pi.state_dict()),
            opt=copy.deepcopy(l.opt_pi.state_dict()),
            iteration=l.iteration,
        )
        baseline = json.loads(a.baseline.read_text())["aggregate"]
        strikes = 0
        last_eval = 0
        start_actor = l.update_counts["actor"]
        dump(
            a.output / "invocation.json",
            dict(
                argv=sys.argv,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                worktree=subprocess.check_output(["git", "status", "--short"], text=True),
                source=str(a.checkpoint.resolve()),
                source_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
                source_counters=source,
                environment_count=2048,
                resume_semantics=l.resume_mode,
            ),
        )

        (a.output / "source.patch").write_text(
            subprocess.check_output(["git", "diff", "HEAD"], text=True)
        )
        dump(
            a.output / "source_hashes.json",
            {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in (
                    "src/unisupermimic/tasks/velocity/go2",
                    "src/unisupermimic/rl/algos/supermimic",
                    "tools",
                )
                for p in (ROOT / folder).glob("*.py")
            },
        )

        def callback(learner):
            nonlocal reason, strikes, last_eval, stable
            coverage.observe()
            elapsed = time.monotonic() - started
            delta = learner.iteration - source["iteration"]
            if delta % 25 == 0:
                with torch.no_grad():
                    pred = learner.pi(fixed, goal)
                    drift = float((pred - reference).square().mean().sqrt())
                    sat = float((pred.abs() > 0.98).float().mean())
                row = dict(
                    iteration=learner.iteration,
                    new_iterations=delta,
                    new_transitions=learner.sim_steps - source["sim_steps"],
                    updates={
                        k: learner.update_counts[k] - source["updates"][k]
                        for k in source["updates"]
                    },
                    wall_s=elapsed,
                    actor_drift_rmse=drift,
                    saturation=sat,
                    actor_lr=learner.opt_pi.param_groups[0]["lr"],
                    rss_mib=proc.memory_info().rss / 2**20,
                    cpu_percent=proc.cpu_percent(),
                    gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    gpu_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                )
                with (a.output / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                dump(a.output / "coverage.json", coverage.summary())
                if not torch.isfinite(pred).all() or sat > 0.6 or drift > 0.8:
                    reason = "actor_output_guard"
                    learner.stop_requested = True
            if (
                delta >= last_eval + a.eval_period
                and learner.update_counts["actor"] > start_actor
                and not learner.stop_requested
            ):
                path = a.output / f"eval_{learner.iteration:06d}"
                policy = learner.save(a.output / "evaluation_policy.pt", include_optimizers=False)
                command = [
                    sys.executable,
                    str(ROOT / "tools/evaluate_go2_omnidirectional.py"),
                    "--checkpoint",
                    str(policy),
                    "--output",
                    str(path),
                    "--seconds",
                    "20",
                    "--seeds",
                    "2426",
                    "2427",
                    "2428",
                    "2429",
                    "2430",
                ]
                with (a.output / f"eval_{learner.iteration:06d}.log").open("w") as f:
                    subprocess.run(
                        command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300
                    )
                evaluation = json.loads((path / "evaluation.json").read_text())["aggregate"]
                regressions = physical_regressions(evaluation, baseline)
                strikes = strikes + 1 if regressions else 0
                event = dict(
                    iteration=learner.iteration,
                    new_iterations=delta,
                    regressions=regressions,
                    strikes=strikes,
                    evaluation=str(path),
                )
                with (a.output / "physics_guard.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                if not regressions:
                    stable = dict(
                        pi=copy.deepcopy(learner.pi.state_dict()),
                        opt=copy.deepcopy(learner.opt_pi.state_dict()),
                        iteration=learner.iteration,
                    )
                    torch.save(stable, a.output / "stable_actor.pt")
                elif strikes >= 2:
                    reason = "physical_regression"
                    learner.stop_requested = True
                last_eval = delta
            if (
                proc.memory_info().rss > 6.5 * 2**30
                or psutil.virtual_memory().available < 600 * 2**20
            ):
                reason = "host_memory_guard"
                learner.stop_requested = True
            if shutil.disk_usage(a.output).free < 20 * 2**30:
                reason = "disk_guard"
                learner.stop_requested = True
            if elapsed >= a.seconds:
                reason = "wall_budget"
                learner.stop_requested = True
            if learner.stop_requested and reason not in ("wall_budget", "iteration_budget"):
                learner.save(a.output / "pre_rollback.pt")
                learner.pi.load_state_dict(stable["pi"])
                learner.opt_pi.load_state_dict(stable["opt"])
                dump(
                    a.output / "rollback.json",
                    dict(
                        reason=reason,
                        stable_iteration=stable["iteration"],
                        wm_replay_preserved=True,
                    ),
                )

        l.on_iteration = callback
        result = l.train()
        coverage.save(a.output)
        dump(
            a.output / "curriculum.json",
            env.command_manager.get_term("velocity").curriculum.summary(),
        )
        dump(
            a.output / "summary.json",
            dict(
                result=result,
                stop_reason=reason,
                source=source,
                new_transitions=l.sim_steps - source["sim_steps"],
                new_updates={
                    k: l.update_counts[k] - source["updates"][k] for k in source["updates"]
                },
                wall_s=time.monotonic() - started,
                profile={
                    k: dict(count=len(v), mean_ms=sum(v) / len(v))
                    for k, v in l.profile_ms.items()
                    if v
                },
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                checkpoint=str((a.output / "last.pt").resolve()),
            ),
        )
    except BaseException as exc:
        dump(a.output / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
        if l is not None:
            l.save(a.output / "interrupted.pt")
        raise
    finally:
        monitor.finish(a.output)
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
