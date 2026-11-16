"""Bounded 2048-env clearance trial on the one existing SuperMimic train loop.

No new BPTT/update implementation. High-level real probes enrich WM windows;
Actor scopes require the unchanged seven-probe admission after adaptation.
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
from unisupermimic.tasks.velocity.go2.speed_curriculum import imagination_scope_admission

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_go2_command_coverage import CommandCoverage
from go2_omni_common import ROOT, configuration, environment
from train_go2_omnidirectional import physical_regressions
from train_go2_stairs_omni import StairsCoverage, run_tool
from validate_go2_migration import ResourceMonitor, dump


def high_stair_probe(l, steps=96, min_level=5):
    """Same sampler/Actor/noise; reset stair envs across L5/6/7, no skill credit.

    Resets are physical, with existing safe-spawn check, not fabricated model
    states. Frozen skill decisions avoid interpreting collection probes as mastery.
    """
    c = l.env.command_manager.get_term("velocity")
    cur = c.curriculum
    ids = (c.command_domain >= 4).nonzero().flatten()
    boundary = torch.zeros(l.env.num_envs, dtype=torch.bool, device=l.device)
    boundary[ids.to(l.device)] = True
    before = cur.mastery.clone()
    enabled = cur.enabled
    cur.enabled = False
    try:
        cur.terrain_level[ids] = min_level + torch.arange(len(ids), device=ids.device) % (
            c.cfg.terrain_max_level - min_level + 1
        )
        l.buffer.mark_reset_boundary(boundary)
        l.env.reset(ids)
        cur.probe[ids] = True
        c.crossing_episode[ids] = False
        cur.crossing_episode[ids] = False
        l.obs_hist[ids.to(l.device)] = 0
        l.collect(steps)
        assert torch.equal(before, cur.mastery), "Probe altered skill mastery"
        l.buffer.mark_reset_boundary(boundary)
        # Reset to inherited mastered terrain, never leave probe states as successes.
        cur.steps[ids] = 0
        cur.episode_fall[ids] = False
        cur.terrain_level[ids] = cur.mastery[ids]
        l.env.reset(ids)
        l.obs_hist[ids.to(l.device)] = 0
    finally:
        cur.enabled = enabled


def assess(l, path, out, holdout_root=None):
    c = l.env.command_manager.get_term("velocity")
    out.mkdir(parents=True, exist_ok=False)
    diagnostics = {}
    for speed in sorted(set(c.sampler.stairs_levels.flatten().tolist())):
        dest = out / f"S{speed}"
        dataset = (
            (holdout_root / f"S{speed}")
            if holdout_root
            else ROOT / f"results/stairs_omni_continuation_v2/extra40_training/holdout_S{speed}"
        )
        if not (dataset / "heldout.pt").exists():
            run_tool(
                "diagnose_go2_omni_wm.py",
                [
                    "--mode",
                    "collect",
                    "--checkpoint",
                    path,
                    "--output",
                    dataset,
                    "--stairs-speed",
                    speed,
                    "--max-level",
                    c.cfg.terrain_max_level,
                    "--stair-step-increment",
                    c.cfg.terrain_stair_increment,
                ],
                out / f"collect_S{speed}.log",
                timeout=480,
            )
        run_tool(
            "diagnose_go2_omni_wm.py",
            [
                "--mode",
                "evaluate",
                "--checkpoint",
                path,
                "--dataset",
                dataset,
                "--output",
                dest,
            ],
            out / f"S{speed}.log",
            timeout=480,
        )
        diagnostics[speed] = json.loads((dest / "diagnostics.json").read_text())
    mask, details = imagination_scope_admission(diagnostics, c.sampler.stairs_levels.tolist())
    l.buffer.start_scope_mask = mask.to(l.device)
    dump(
        out / "admission.json",
        dict(
            mask=mask.tolist(),
            details=details,
            threshold="unchanged: seven probes, each qd sign >=.80, norm error <1, H32 limits",
        ),
    )
    l.extra_state["clearance_scope_admission"] = dict(mask=mask, details=details)
    return details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--task",
        choices=[
            "stairs_omni_level7",
            "stairs_clearance_gaussian",
            "stairs_clearance_soft_deficit",
            "stairs_clearance_20cm",
        ],
        required=True,
    )
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--wm-adaptation", type=int, default=100)
    p.add_argument("--critic-warmup", type=int, default=100)
    p.add_argument("--actor-lr-scale", type=float, default=0.03)
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--high-probe-min-level", type=int, default=5)
    p.add_argument("--holdout-root", type=Path)
    p.add_argument("--probe-every", type=int, default=0)
    p.add_argument("--scope-refresh", type=int, default=0)
    p.add_argument("--wm-balance-high-levels", action="store_true")
    p.add_argument("--physics-guard-seconds", type=float, default=1200)
    p.add_argument("--guard-baseline", type=Path)
    a = p.parse_args()
    if not 0 < a.seconds <= 5400 or a.iterations < 1 or not 0 < a.actor_lr_scale <= 0.1:
        raise ValueError("Invalid bounded trial budget")
    a.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(a.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    origin = {k: saved[k] for k in ("iteration", "sim_steps", "update_counts", "wall_time")}
    cfg = configuration(a.task)
    cfg.algo.wm_only_iters = origin["iteration"] + a.wm_adaptation
    cfg.algo.critic_warmup_iters = a.critic_warmup
    cfg.algo.iterations = origin["iteration"] + a.iterations
    cfg.algo.actor_lr_scale = a.actor_lr_scale
    cfg.algo.ckpt_every = 250
    cfg.algo.numbered_ckpt_every = 1000000
    del saved
    OmegaConf.save(cfg, a.output / "config.yaml")
    start = time.monotonic()
    env = l = None
    reason = "iteration_budget"
    monitor = ResourceMonitor()
    monitor.thread.start()
    try:
        env = environment(cfg, 2048)
        env.init_state()
        l = SuperMimicLearner(
            env, build_learner_config(cfg), a.output, device="cuda", backend="mujoco"
        )
        l.load(a.checkpoint)  # Strict load after explicit migration; no bypass.
        c = env.command_manager.get_term("velocity")
        for key in (
            "actual_actor_start_scopes",
            "actor_starts_domain_terrain_speed",
            "actual_actor_start_modes",
            "actual_policy_start_modes",
            "omni_coverage",
            "stairs_command_coverage",
        ):
            l.extra_state.pop(key, None)
        l.extra_state["clearance_trial_origin"] = origin
        l.buffer.wm_rare_fraction = 0.35
        l.buffer.wm_rare_min_level = a.high_probe_min_level
        l.buffer.wm_rare_balance_levels = a.wm_balance_high_levels
        l.buffer.wm_level_counts.zero_()
        coverage = CommandCoverage(l)
        stairs = StairsCoverage(l)
        # Independent run-local metadata; historic speed labels are UNKNOWN (-1).
        if not a.no_probes:
            high_stair_probe(l, min_level=a.high_probe_min_level)
            coverage.observe()
            stairs.observe()
        starts, hist = l.buffer.sample_starts(512, 5)
        obs = l.mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"])
        goal = starts["g"]
        with torch.no_grad():
            ref = l.pi(obs, goal).clone()
        source_actor = copy.deepcopy(l.pi.state_dict())
        source_opt = copy.deepcopy(l.opt_pi.state_dict())
        metrics = []
        extra_steps = l.sim_steps - origin["sim_steps"]
        dump(
            a.output / "invocation.json",
            dict(
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                argv=sys.argv,
                source_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
                origin=origin,
                num_envs=2048,
                resume_mode=l.resume_mode,
                wm_rare_fraction=0.35,
                wm_rare_balance_levels=a.wm_balance_high_levels,
                periodic_probe_every=a.probe_every,
                scope_refresh_every=a.scope_refresh,
                speed_metadata="old replay explicitly unknown; new labels recorded before action",
                probe_steps=extra_steps,
                gates="unchanged; reassess before any Actor optimization; no high-level blanket admission",
            ),
        )

        last_physics = 0.0
        physics_strikes = 0
        baseline_physics = (
            json.loads(a.guard_baseline.read_text())["aggregate"] if a.guard_baseline else None
        )

        def callback(learner):
            nonlocal reason, last_physics, physics_strikes
            delta = l.iteration - origin["iteration"]
            coverage.observe()
            stairs.observe()
            if delta == a.wm_adaptation:
                warm = a.output / "warm.pt"
                l.save(warm)
                assess(l, warm, a.output / "wm_warm", a.holdout_root)
            if a.scope_refresh and delta % a.scope_refresh == 0:
                gate_checkpoint = a.output / "scope_latest.pt"
                l.save(gate_checkpoint)
                assess(l, gate_checkpoint, a.output / f"wm_refresh_{delta}", a.holdout_root)
            if not a.no_probes and (
                delta in (200, 400) or (a.probe_every and delta % a.probe_every == 0)
            ):
                high_stair_probe(l, min_level=a.high_probe_min_level)
                coverage.observe()
                stairs.observe()
            if delta % 25 == 0:
                with torch.no_grad():
                    action = l.pi(obs, goal)
                    row = dict(
                        iteration=l.iteration,
                        new_iteration=delta,
                        wall_s=time.monotonic() - start,
                        actor_rmse=float((action - ref).square().mean().sqrt()),
                        saturation=float((action.abs() > 0.98).float().mean()),
                        gpu_allocated=torch.cuda.memory_allocated(),
                        gpu_reserved=torch.cuda.memory_reserved(),
                        rss=psutil.Process().memory_info().rss,
                        new_transitions=l.sim_steps - origin["sim_steps"],
                        new_updates={
                            k: l.update_counts[k] - origin["update_counts"][k]
                            for k in l.update_counts
                        },
                    )
                metrics.append(row)
                with (a.output / "guards.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if row["actor_rmse"] > 0.12 or row["saturation"] > 0.65:
                    reason = "actor_drift_or_saturation"
                    l.stop_requested = True
            elapsed = time.monotonic() - start
            if (
                baseline_physics
                and elapsed - last_physics >= a.physics_guard_seconds
                and elapsed < a.seconds - 120
            ):
                guard_checkpoint = a.output / "guard_latest.pt"
                l.save(guard_checkpoint)
                evaluation_dir = a.output / f"physics_guard_{delta}"
                run_tool(
                    "evaluate_go2_omnidirectional.py",
                    [
                        "--checkpoint",
                        guard_checkpoint,
                        "--output",
                        evaluation_dir,
                        "--quick",
                        "--max-level",
                        c.cfg.terrain_max_level,
                        "--stair-step-increment",
                        c.cfg.terrain_stair_increment,
                        "--seconds",
                        20,
                        "--seeds",
                        2426,
                    ],
                    a.output / f"physics_guard_{delta}.log",
                    timeout=180,
                )
                physics = json.loads((evaluation_dir / "evaluation.json").read_text())["aggregate"]
                regressions = physical_regressions(
                    physics, {k: baseline_physics[k] for k in physics}
                )
                physics_strikes = physics_strikes + 1 if regressions else 0
                dump(
                    a.output / f"physics_guard_{delta}.json",
                    dict(regressions=regressions, strikes=physics_strikes),
                )
                if physics_strikes >= 2:
                    reason = "physical_regression"
                    l.stop_requested = True
                last_physics = time.monotonic() - start
            if (
                psutil.virtual_memory().available < 600 * 2**20
                or shutil.disk_usage(a.output).free < 15 * 2**30
            ):
                reason = "resource_guard"
                l.stop_requested = True
            if time.monotonic() - start >= a.seconds:
                reason = "wall_budget"
                l.stop_requested = True

        l.on_iteration = callback
        result = l.train()
        l.save()
        coverage.observe()
        stairs.observe()
        coverage.save(a.output)
        dump(a.output / "stairs_coverage.json", stairs.summary())
        dump(a.output / "curriculum.json", c.curriculum.summary())
        dump(
            a.output / "summary.json",
            dict(
                stop_reason=reason,
                result=result,
                wall_s=time.monotonic() - start,
                new_transitions=l.sim_steps - origin["sim_steps"],
                new_updates={
                    k: l.update_counts[k] - origin["update_counts"][k] for k in l.update_counts
                },
                wm_sampled_starts_by_terrain=l.buffer.wm_level_counts.tolist(),
                actor_starts_domain_terrain_speed=l.extra_state.get(
                    "actor_starts_domain_terrain_speed", torch.zeros(6, 8, 5)
                )
                .cpu()
                .tolist(),
                unknown_speed_index=4,
                checkpoint=str((a.output / "last.pt").resolve()),
            ),
        )
        if reason in ("actor_drift_or_saturation", "resource_guard", "physical_regression"):
            l.save(a.output / "rejected_candidate.pt")
            l.pi.load_state_dict(source_actor)
            l.opt_pi.load_state_dict(source_opt)
            l.save(a.output / "actor_rollback.pt")
    except BaseException as e:
        dump(a.output / "failure.json", dict(type=type(e).__name__, message=str(e)))
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
