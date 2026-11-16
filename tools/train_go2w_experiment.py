"""Bounded Go2W reward/coverage experiments using the shared training loop."""

import argparse
import json
import os
import resource
import signal
import time
from dataclasses import fields
from pathlib import Path

import psutil
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.checkpoint import initialize_experiment
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.world_model import (
    export_world_model,
    load_world_model,
    tensor_digest,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("persistent", "persistent_stop", "persistent_lateral", "joint_range", "omni_v3"), required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fresh", action="store_true", help="New Actor/Critic; reuse --wm-from unless explicitly creating --new-wm")
    parser.add_argument("--wm-from", type=Path, help="Continue this shared WM, including optimizer and normalizers")
    parser.add_argument("--actor-from", type=Path, help="Proprioceptive policy bootstrap asset for a fresh Actor")
    parser.add_argument("--new-wm", action="store_true", help="Explicit one-time creation of a new dynamics model")
    parser.add_argument("--publish-wm-to", type=Path, help="Defaults to --wm-from, updating the same asset with history")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--wm-only", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if sum((bool(args.initialize_from), bool(args.resume), args.fresh)) != 1:
        parser.error("Choose exactly one of --initialize-from, --resume and --fresh")
    if args.new_wm and args.wm_from:
        parser.error("--new-wm and --wm-from are mutually exclusive")
    if args.actor_from and not args.fresh:
        parser.error("--actor-from requires --fresh")
    if args.fresh and not args.wm_from and not args.new_wm:
        parser.error("New Actor requires --wm-from; only --new-wm permits a fresh dynamics model")
    if min(args.envs, args.batch, args.seconds, args.iterations) <= 0 or (args.wm_only is not None and args.wm_only < 0):
        parser.error("Sizes and budget must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"):
        owner = compose("config", overrides=[f"task=go2w_velocity/{args.profile}"])
    raw = OmegaConf.to_container(owner.algo, resolve=True)
    cfg = SuperMimicConfig(**{f.name: raw[f.name] for f in fields(SuperMimicConfig) if f.name in raw})
    cfg.iterations = args.iterations
    cfg.seed = args.seed
    cfg.wm_batch = cfg.pi_batch = args.batch
    cfg.wm_only_iters = args.wm_only if args.wm_only is not None else (0 if args.wm_from else 500)
    cfg.critic_warmup_iters = 50
    cfg.log_every = 20 if args.iterations > 20 else 1
    cfg.ckpt_every = 500
    cfg.numbered_ckpt_every = 0
    cfg.log_backend = "none"
    env = None
    start = time.monotonic()
    stopped = "iteration_budget"
    write_json(args.output / "launch.json", dict(vars(args), pid=os.getpid(), source_root=str(root)) | {
        "output": str(args.output), "initialize_from": str(args.initialize_from), "resume": str(args.resume),
        "wm_from": str(args.wm_from),
        "publish_wm_to": str(args.publish_wm_to),
        "actor_from": str(args.actor_from),
    })
    try:
        env = make_go2w_env(load_config(args.profile), args.envs)
        env.reset()
        learner = SuperMimicLearner(env, cfg, args.output, device="cuda:0", backend="mujoco")
        if args.initialize_from:
            write_json(args.output / "initialization_audit.json", initialize_experiment(learner, args.initialize_from))
        elif args.resume:
            learner.load(args.resume)
        if args.wm_from:
            write_json(args.output / "world_model_asset.json", load_world_model(learner, args.wm_from))
        if args.actor_from:
            actor_asset = torch.load(args.actor_from, map_location="cpu", weights_only=True)
            if actor_asset["task_contract"] != env.supermimic_contract:
                raise ValueError("Actor bootstrap task contract differs")
            learner.pi.load_state_dict(actor_asset["pi"])
            learner.extra_state["actor_bootstrap"] = actor_asset["provenance"]
        initial_wm_updates = learner.update_counts["wm"]
        initial_wm_digest = tensor_digest(learner.wm.state_dict())
        command = env.command_manager.get_term("velocity")
        write_json(args.output / "task_contract.json", env.supermimic_contract)
        write_json(args.output / "initial_curriculum.json", command.curriculum.summary())

        def request_stop(signum, frame):
            nonlocal stopped
            stopped = f"signal_{signum}"
            learner.stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        def observe(current):
            nonlocal stopped
            elapsed = time.monotonic() - start
            if elapsed >= args.seconds:
                stopped = "wall_clock_budget"
                current.stop_requested = True
            if current.iteration % cfg.log_every == 0 or current.stop_requested:
                status = {
                    "iteration": current.iteration, "elapsed_seconds": elapsed,
                    "sim_steps": current.sim_steps, "updates": dict(current.update_counts),
                    "available_RAM_MiB": psutil.virtual_memory().available / 2**20,
                    "process_RSS_MiB": psutil.Process().memory_info().rss / 2**20,
                    "GPU_allocated_MiB": torch.cuda.memory_allocated() / 2**20,
                    "GPU_peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
                    "curriculum": command.curriculum.summary(),
                }
                write_json(args.output / "progress.json", status)
                with (args.output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(status) + "\n")
                print(json.dumps({k: v for k, v in status.items() if k != "curriculum"}), flush=True)

        learner.on_iteration = observe
        summary = learner.train()
        summary["continual_world_model"] = {
            "tensor_sha256_before": initial_wm_digest,
            "tensor_sha256_after": tensor_digest(learner.wm.state_dict()),
            "source_updates": initial_wm_updates,
            "new_world_model_updates": learner.update_counts["wm"] - initial_wm_updates,
        }
        learner.save()
        asset_path = args.publish_wm_to or args.wm_from or args.output / "world_model.pt"
        expected = initial_wm_digest if args.wm_from and asset_path.resolve() == args.wm_from.resolve() else None
        write_json(args.output / "world_model_publish.json", export_world_model(args.output / "last.pt", asset_path, expected_digest=expected))
        summary.update(
            stop_reason=stopped, elapsed_seconds=time.monotonic() - start,
            updates=dict(learner.update_counts), curriculum=command.curriculum.summary(),
            peak_CPU_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            peak_GPU_MiB=torch.cuda.max_memory_allocated() / 2**20,
        )
        write_json(args.output / "summary.json", summary)
    except Exception as error:
        write_json(args.output / "failure.json", {"error": repr(error), "elapsed_seconds": time.monotonic() - start})
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
