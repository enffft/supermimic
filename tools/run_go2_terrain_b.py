"""Bounded experiment B orchestration, using the single shared learner loop.

1024-multiple collectors only. Warmstart does not authorize actor unfreezing.
The benchmark runs H32 backward + critic updates with actor optimizer frozen.
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_go2_migration import ResourceMonitor, dump

ROOT = Path(__file__).resolve().parents[1]


def digest(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Coverage:
    def __init__(self, learner):
        self.l = learner
        self.counts = {
            key: torch.zeros(5, 5, device=learner.device, dtype=torch.long)
            for key in (
                "transitions",
                "episode_starts",
                "falls",
                "done",
                "probes",
                "proxy_contact_switches",
                "nonflat_proxy_contacts",
            )
        }
        for key, tensor in learner.extra_state.get("coverage", {}).items():
            self.counts[key].copy_(tensor)
        self.last_contact = None
        self.last_steps = learner.sim_steps

    @torch.no_grad()
    def observe(self):
        # New ring rows since last callback, at most a few vector steps per outer.
        b = self.l.buffer
        n = (self.l.sim_steps - self.last_steps) // b.N
        rows = (torch.arange(n, device=b.device) + b.ptr - n) % b.T
        for row in rows:
            d = {k: v[row] for k, v in b.data.items()}
            idx = d["terrain_type"] * 5 + d["terrain_level"]
            s = unflatten_state(d["s1"])
            feet, _ = go2_feet_world(self.l.mdp.source_state(s))
            heights = self.l.mdp.terrain.ground_height(feet[..., :2])
            contact = feet[..., 2] - heights - 0.022 < 0.025  # geometry proxy, NOT force
            entry = self.l.env.command_manager.get_term("velocity").curriculum.origins.to(b.device)
            base_height = entry[d["terrain_level"], d["terrain_type"], 2]
            nonflat = ((heights - base_height[:, None]).abs() > 0.002) & contact
            switches = (
                torch.zeros_like(idx)
                if self.last_contact is None
                else ((contact != self.last_contact) & (d["ep_step"] > 0)[:, None]).sum(-1)
            )
            self.last_contact = contact
            values = dict(
                transitions=torch.ones_like(idx),
                episode_starts=d["ep_step"] == 0,
                falls=d["done"] & (d["ep_step"] < 499),
                done=d["done"],
                probes=d["terrain_probe"],
                proxy_contact_switches=switches,
                nonflat_proxy_contacts=nonflat.any(-1),
            )
            for key, value in values.items():
                self.counts[key].view(-1).scatter_add_(0, idx, value.long())
        self.last_steps = self.l.sim_steps
        self.l.extra_state["coverage"] = self.counts

    def summary(self):
        result = {k: v.cpu().tolist() for k, v in self.counts.items()}
        result["contact_definition"] = (
            "per-foot FK clearance<2.5cm; geometric proxy, not measured contact"
        )
        b = self.l.buffer
        for horizon in (8, 32):
            w = b.sample_windows(2048, horizon)
            result[f"valid_h{horizon}_fraction_sampled"] = float(w["mask"][:, -1].mean())
        result["replay_transitions"] = b.size * b.N
        actions = b.data["a"][: b.size]
        result["executed_action_abs_max"] = float(actions.abs().max())
        result["executed_action_saturation_fraction"] = float((actions.abs() > 0.98).float().mean())
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("benchmark", "warmstart"), required=True)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    if args.envs < 1024 or args.envs % 1024:
        raise ValueError("Formal collectors must be a multiple of 1024")
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        raise ValueError("Resume must use an existing run directory")
    out = args.output
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose(
            "config",
            overrides=[
                "task=go2_velocity/longrun_B",
                f"algo.num_envs={args.envs}",
                *args.overrides,
            ],
        )
    if args.mode == "benchmark":
        cfg.algo.iterations = 64
        cfg.algo.wm_batches = 2
        cfg.algo.buffer_steps = 128
        cfg.algo.ckpt_every = 64
    if cfg.algo.iterations > cfg.algo.wm_only_iters:
        raise ValueError(
            "This Stage I entry point requires a separate validated actor-unfreeze gate"
        )
    OmegaConf.save(cfg, out / ("config_resume.yaml" if args.resume else "config.yaml"))
    dump(
        out / "source.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            dirty=subprocess.check_output(["git", "status", "--short"], text=True),
            argv=sys.argv,
            actor_sha256=hashlib.file_digest(
                Path(cfg.algo.init_actor_from).open("rb"), "sha256"
            ).hexdigest(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            formal_networks=True,
        ),
    )
    if (
        json.loads((out / "source.json").read_text())["actor_sha256"]
        != "0b3d8e4bb26173d13a6ce0b99d5eecf43005927a8ab2e01947c4d0311009487a"
    ):
        raise ValueError("Old actor SHA differs from audited source")
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
    started = time.perf_counter()
    env, learner = None, None
    process = psutil.Process()
    process.cpu_percent()
    try:
        env = make_go2_env(env_cfg, args.envs)
        env.init_state()
        learner = SuperMimicLearner(
            env, build_learner_config(cfg), out, device="cuda", backend="mujoco"
        )
        if args.resume:
            learner.load(args.resume)
        else:
            learner.save(out / "initial.pt", include_optimizers=False)
        coverage = Coverage(learner)
        actor_digest = digest(learner.pi)
        torch.cuda.reset_peak_memory_stats()
        records = []

        def callback(l):
            coverage.observe()
            if not l.iteration % l.cfg.log_every or l.iteration == l.cfg.iterations:
                if (
                    digest(l.pi) != actor_digest
                    or l.update_counts["actor"] != 0
                    or len(l.opt_pi.state)
                ):
                    raise RuntimeError("Frozen actor contract violated")
                row = dict(
                    iteration=l.iteration,
                    real_transitions=l.sim_steps,
                    **l.update_counts,
                    wall_s=l.wall_time,
                    rss_mib=process.memory_info().rss / 2**20,
                    cpu_percent=process.cpu_percent(),
                    available_host_mib=psutil.virtual_memory().available / 2**20,
                    gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    gpu_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                    actor_digest=actor_digest,
                    resume_mode=l.resume_mode,
                )
                records.append(row)
                with (out / "budget.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                dump(out / "coverage.json", coverage.summary())
                dump(
                    out / "curriculum.json",
                    env.command_manager.get_term("velocity").curriculum.summary(),
                )
                print(json.dumps(row), flush=True)
            if (
                process.memory_info().rss > 6.5 * 2**30
                or psutil.virtual_memory().available < 600 * 2**20
            ):
                raise MemoryError("Resource guard: RSS>6.5GiB or available host RAM<600MiB")

        learner.on_iteration = callback
        result = learner.train()
        if args.mode == "benchmark":
            # Identical production H32 graph and backward, no unsafe actor step.
            for _ in range(4):
                learner.policy_update(update_actor=False)
        if digest(learner.pi) != actor_digest:
            raise RuntimeError("Frozen actor changed")
        timings = {
            k: dict(
                count=len(v),
                median_ms=float(np.median(v)),
                mean_ms=float(np.mean(v)),
                p90_ms=float(np.percentile(v, 90)),
            )
            for k, v in learner.profile_ms.items()
        }
        dump(
            out / "summary.json",
            dict(
                result=result,
                profile=timings,
                total_process_wall_s=time.perf_counter() - started,
                updates=learner.update_counts,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                simulation_vector_steps_s=1000 / timings["simulation"]["mean_ms"],
                simulation_transitions_s=args.envs * 1000 / timings["simulation"]["mean_ms"],
                wm_updates_s=1000 / timings["wm_fit"]["mean_ms"],
                frozen_actor=True,
                checkpoint=str((out / "last.pt").resolve()),
            ),
        )
    except BaseException as exc:
        dump(
            out / "failure.json",
            dict(
                type=type(exc).__name__,
                error=str(exc),
                elapsed_s=time.perf_counter() - started,
                peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
        )
        if learner is not None:
            learner.save(out / "interrupted.pt")
        raise
    finally:
        monitor.finish(out)
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
