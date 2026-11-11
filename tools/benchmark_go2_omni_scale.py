"""Bounded full-network collector scaling test, with real Actor optimizer steps.

Source replay/environment cannot be resumed at a different N. Explicitly transfer
weights/optimizers/counters and refill replay, retaining source robot contracts.
All updates use the shared learner.train(); benchmark policies are not deployment
candidates. The warm fill covers at least one ten-second episode per environment.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_omni_common import ROOT, configuration, environment
from validate_go2_migration import ResourceMonitor, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, required=True)
    p.add_argument(
        "--checkpoint", type=Path, default=ROOT / "results/omnidirectional_terrain/stageB/last.pt"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=128)
    a = p.parse_args()
    if a.envs < 1024 or a.envs % 1024 or a.iterations < 32:
        raise ValueError("Require 1024-multiple environments and >=32 full updates")
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = configuration()
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    original_iteration = int(source["iteration"])
    original_steps = int(source["sim_steps"])
    original_counts = source["update_counts"].copy()
    schedule = source["extra_state"]["omni_stage_schedule"]
    cfg.algo.num_envs = a.envs
    cfg.algo.iterations = original_iteration + a.iterations
    cfg.algo.wm_only_iters = schedule["wm_only_iters"]
    cfg.algo.critic_warmup_iters = schedule["critic_warmup_iters"]
    cfg.algo.warmup_steps = 512
    cfg.algo.ckpt_every = 1000000000
    cfg.algo.log_every = 16
    # N-changing transfer is deliberately distinct from replay/curriculum resume.
    transfer = {k: v for k, v in source.items() if k not in ("replay", "curriculum", "extra_state")}
    transfer["extra_state"] = {"benchmark_source": str(a.checkpoint.resolve())}
    transfer_path = a.output / "weights_optimizers_transfer.pt"
    torch.save(transfer, transfer_path)
    sampler_state = source["command_sampler"]
    source_mastery = int(source["curriculum"]["tensors"]["mastery"].min())
    del transfer, source
    OmegaConf.save(cfg, a.output / "config.yaml")
    dump(
        a.output / "provenance.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source=str(a.checkpoint.resolve()),
            source_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            envs=a.envs,
            replay_semantics="fresh replay after N-changing weights/optimizer transfer; not exact resume",
            terrain_start_mastery=source_mastery,
            full_actor_updates=True,
            warm_fill_vector_steps=512,
            measured_outer_iterations=a.iterations,
        ),
    )
    monitor = ResourceMonitor()
    monitor.thread.start()
    process = psutil.Process()
    process.cpu_percent()
    started = time.monotonic()
    env = None
    l = None
    records = []
    steady_started = None
    stop_reason = "completed"
    try:
        env = environment(cfg, a.envs)
        env.init_state()
        l = SuperMimicLearner(
            env, build_learner_config(cfg), a.output, device="cuda", backend="mujoco"
        )
        l.load(transfer_path)
        cmd = env.command_manager.get_term("velocity")
        cmd.sampler.load_state_dict(sampler_state)
        cmd.curriculum.mastery.fill_(source_mastery)
        cmd.curriculum.terrain_level.fill_(source_mastery)
        cmd.curriculum.terrain_level[cmd.command_domain == 0] = 0
        env.reset()
        l.obs_hist.zero_()
        torch.cuda.reset_peak_memory_stats()

        def callback(learner):
            nonlocal steady_started, stop_reason
            delta = learner.iteration - original_iteration
            # Discard initialization, full warm fill and first 16 optimizer steps.
            if delta == 16:
                learner.profile_ms.clear()
                steady_started = time.monotonic()
            if delta % 16 == 0:
                row = dict(
                    iteration=delta,
                    new_transitions=learner.sim_steps - original_steps,
                    updates={
                        k: learner.update_counts[k] - original_counts[k] for k in original_counts
                    },
                    wall_s=time.monotonic() - started,
                    rss_mib=process.memory_info().rss / 2**20,
                    cpu_percent=process.cpu_percent(),
                    host_available_mib=psutil.virtual_memory().available / 2**20,
                    gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    gpu_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                )
                records.append(row)
                with (a.output / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if (
                process.memory_info().rss > 6.5 * 2**30
                or psutil.virtual_memory().available < 600 * 2**20
            ):
                stop_reason = "memory_guard"
                learner.stop_requested = True

        l.on_iteration = callback
        result = l.train()
        updates = {k: l.update_counts[k] - original_counts[k] for k in original_counts}
        profile = {
            k: dict(
                count=len(v),
                mean_ms=float(torch.tensor(v).mean()),
                median_ms=float(torch.tensor(v).median()),
                p90_ms=float(torch.tensor(v).quantile(0.9)),
            )
            for k, v in l.profile_ms.items()
            if v
        }
        steady_wall = time.monotonic() - steady_started if steady_started else None
        expected = dict(
            wm=a.iterations * cfg.algo.wm_batches,
            actor=a.iterations,
            critic=a.iterations * cfg.algo.critic_epochs * cfg.algo.critic_minibatches,
        )
        completed = updates == expected and stop_reason == "completed"
        dump(
            a.output / "summary.json",
            dict(
                completed=completed,
                stop_reason=stop_reason,
                envs=a.envs,
                result=result,
                new_transitions=l.sim_steps - original_steps,
                updates=updates,
                expected_updates=expected,
                profile=profile,
                process_wall_s=time.monotonic() - started,
                steady_iterations=a.iterations - 16,
                steady_wall_including_final_saves_s=steady_wall,
                simulation_vector_steps_s=1000 / profile["simulation"]["mean_ms"],
                simulation_transitions_s=a.envs * 1000 / profile["simulation"]["mean_ms"],
                wm_updates_s=1000 / profile["wm_fit"]["mean_ms"],
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                max_logged_rss_mib=max(r["rss_mib"] for r in records),
                domain_env_counts=torch.bincount(cmd.command_domain, minlength=6).tolist(),
                networks=dict(
                    actor_input=213,
                    wm_input=142,
                    critic_input=56,
                    wm_hidden=list(cfg.algo.wm_hidden),
                    policy_hidden=list(cfg.algo.policy_hidden),
                    wm_batch=cfg.algo.wm_batch,
                    pi_batch=cfg.algo.pi_batch,
                    wm_window=8,
                    horizon=32,
                    buffer_steps=cfg.algo.buffer_steps,
                ),
            ),
        )
        if not completed:
            raise RuntimeError("Scale benchmark did not complete all requested updates")
    except BaseException as exc:
        dump(
            a.output / "failure.json",
            dict(type=type(exc).__name__, error=str(exc), wall_s=time.monotonic() - started),
        )
        raise
    finally:
        monitor.finish(a.output)
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
