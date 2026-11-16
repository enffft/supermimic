"""Wall-clock bounded Go2W training with the repository's upstream RSL-RL PPO."""
import argparse
import contextlib
import io
import hashlib
import importlib.metadata
import json
import signal
import time
from pathlib import Path

import psutil
import torch
from rsl_rl.runners import OnPolicyRunner
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.ppo import Go2WPpoEnv, training_config


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class BudgetReached(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-from", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--seconds", type=float, default=9000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.envs, args.rollout, args.minibatch, args.epochs, args.iterations, args.seconds) <= 0:
        parser.error("All sizes and budgets must be positive")
    cfg = training_config(args.envs, args.rollout, args.minibatch, args.epochs)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    start = time.monotonic()
    env = None
    stopped = "iteration_budget"
    stop_requested = False
    runner = None
    try:
        env = make_go2w_env(load_config("omni_v3"), args.envs)
        env.reset()
        bridge = Go2WPpoEnv(env)
        runner = OnPolicyRunner(bridge, cfg, log_dir=str(args.output / "tensorboard"), device="cuda:0")
        asset = torch.load(args.actor_from, weights_only=True, map_location="cpu")
        if asset["task_contract"] != env.supermimic_contract:
            raise ValueError("Bootstrap task contract differs from training task")
        runner.alg.actor.mlp.policy.load_state_dict(asset["pi"])
        write_json(args.output / "task_contract.json", env.supermimic_contract)
        write_json(args.output / "ppo_config.json", cfg)
        write_json(args.output / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()} | {
            "source": str(Path(__file__).resolve().parents[1]), "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
            "actor_asset_sha256": hashlib.sha256(args.actor_from.read_bytes()).hexdigest(),
            "wm": "not applicable: model-free PPO; shared WM untouched",
            "differences": ["Real on-policy rollouts replace imagined rollouts and replay-start lookahead",
                            "Same persistent coverage environments also contribute PPO gradients",
                            "Gaussian starts at previous collection std .02, then learns its std",
                            "Fresh privileged critic, native PPO normalization/GAE/clipping/KL schedule"],
        })
        command = env.command_manager.get_term("velocity")
        write_json(args.output / "initial_curriculum.json", command.curriculum.summary())
        last_save = time.monotonic()
        saved_2h = False
        completed = 0
        def save(name):
            policy = runner.alg.actor.mlp.policy
            atomic_save({"pi": {k: v.detach().cpu() for k,v in policy.state_dict().items()},
                         "args": {"policy_hidden": cfg["actor"]["hidden_dims"]},
                         "task_contract": env.supermimic_contract, "algorithm": "PPO",
                         "iteration": completed, "transitions": bridge.transitions,
                         "elapsed_seconds": time.monotonic() - start}, args.output / f"{name}.pt")
            atomic_save(runner.alg.save() | {"iter": completed,
                        "curriculum": command.curriculum.state_dict()}, args.output / "ppo_resume.pt")
        def request_stop(signum, frame):
            nonlocal stop_requested, stopped
            stop_requested, stopped = True, f"signal_{signum}"
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        native_log = runner.logger.log
        def log(**kw):
            nonlocal stopped, last_save, saved_2h, completed
            completed = kw["it"] + 1
            elapsed = time.monotonic() - start
            status = dict(iteration=completed, elapsed_seconds=elapsed, transitions=bridge.transitions,
                          collect_seconds=kw["collect_time"], learn_seconds=kw["learn_time"],
                          loss=kw["loss_dict"], learning_rate=kw["learning_rate"],
                          action_std=kw["action_std"].mean().item(),
                          mean_episode_reward=sum(runner.logger.rewbuffer)/max(1,len(runner.logger.rewbuffer)),
                          mean_episode_length=sum(runner.logger.lenbuffer)/max(1,len(runner.logger.lenbuffer)),
                          GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                          RSS_MiB=psutil.Process().memory_info().rss/2**20,
                          available_RAM_MiB=psutil.virtual_memory().available/2**20)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(status) + "\n")
            write_json(args.output / "progress.json", status | {"curriculum": command.curriculum.summary()})
            if completed % 10 == 0:
                native_log(**kw, print_minimal=True)
                print(json.dumps(status), flush=True)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    native_log(**kw, print_minimal=True)
            if hasattr(env, "go2w_metrics"):
                metrics = env.go2w_metrics
                counts = metrics["transitions"].clamp_min(1)
                data = {key: (value / counts).tolist() for key,value in metrics.items()}
                with (args.output / "go2w_diagnostics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"iteration": completed, "counts": metrics["transitions"].tolist(), **data})+"\n")
                with torch.inference_mode():
                    for value in metrics.values():
                        value.zero_()
            if elapsed >= 7200 and not saved_2h:
                save("at_2h")
                saved_2h = True
            if time.monotonic() - last_save >= 300:
                save("last")
                last_save = time.monotonic()
            if elapsed >= args.seconds or stop_requested:
                if not stop_requested:
                    stopped = "wall_clock_budget"
                raise BudgetReached
        runner.logger.log = log
        # Native runner retains all rollout, GAE and PPO optimization behavior.
        try:
            runner.learn(args.iterations)
        except BudgetReached:
            pass
        save("last")
        write_json(args.output / "summary.json", dict(stop_reason=stopped, iterations=completed,
            transitions=bridge.transitions, elapsed_seconds=time.monotonic()-start,
            curriculum=command.curriculum.summary(), peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20))
    except Exception as error:
        write_json(args.output / "failure.json", {"error": repr(error), "elapsed_seconds": time.monotonic()-start})
        raise
    finally:
        if runner is not None:
            runner.logger.stop_logging_writer()
            if runner.logger.writer is not None:
                runner.logger.writer.close()
        if env is not None:
            env.close()
            env.backend.close()

if __name__ == "__main__":
    main()
