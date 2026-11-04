#!/usr/bin/env python3
"""Fixed-seed MuJoCo residual-alpha sweep for historical M1 Action22 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_CHECKPOINTS = {
    "015m": Path("logs/m1_prior_actor/m1_action22_v1_LM50_2048_1h_20261103/ckpt_wall_015m.pt"),
    "030m": Path("logs/m1_prior_actor/m1_action22_v1_LM50_2048_1h_20261103/ckpt_wall_030m.pt"),
    "045m": Path("logs/m1_prior_actor/m1_action22_v1_LM50_2048_1h_20261103/ckpt_wall_045m.pt"),
    "060m": Path("logs/m1_prior_actor/m1_action22_v1_LM50_2048_1h_20261103/ckpt_wall_060m.pt"),
    "final": Path("logs/m1_prior_actor/m1_action22_v1_LM50_2048_1h_20261103/ckpt_latest.pt"),
}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sweep(*, output_dir: Path, num_envs: int = 32, steps: int = 300,
              seed: int = 20261104, device: str = "cuda:0",
              checkpoints: dict[str, Path] | None = None,
              alphas: tuple[float, ...] = ALPHAS, resume: bool = True) -> dict:
    checkpoint_map = checkpoints or DEFAULT_CHECKPOINTS
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = {name: str(path) for name, path in checkpoint_map.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"historical checkpoint binaries missing: {missing}")
    planned = [
        output_dir / f"{name}_{task}_actor_alpha_{alpha:.2f}.json"
        for name in checkpoint_map for task in ("L", "M") for alpha in alphas
    ]
    planned += [output_dir / f"final_{task}_prior_direct.json" for task in ("L", "M")]
    results = []

    def run_case(name: str, path: Path, task: str, alpha: float,
                 controller: str, target: Path) -> dict:
        checkpoint_hash = sha256(path)
        if target.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite existing sweep result: {target}")
            saved = json.loads(target.read_text())
            contract = saved.get("evaluation_contract", {})
            expected_controller = controller
            if (
                saved.get("checkpoint_sha256") != checkpoint_hash
                or saved.get("task_type") != task
                or saved.get("controller", "actor") != expected_controller
                or saved.get("seed") != seed
                or saved.get("num_envs") != num_envs
                or saved.get("steps") != steps
                or contract.get("seed") != seed
                or contract.get("num_envs") != num_envs
                or contract.get("steps") != steps
            ):
                raise ValueError(f"existing sweep result has a mismatched contract: {target}")
            print(f"resuming past existing checkpoint={name} task={task} controller={controller} alpha={alpha:.2f}")
            return saved

        command = [
            sys.executable, str(Path(__file__).with_name("evaluate_m1_action22_residual_alpha_case.py")),
            "--checkpoint", str(path), "--task-type", task, "--alpha", str(alpha),
            "--controller", controller, "--num-envs", str(num_envs),
            "--steps", str(steps), "--seed", str(seed), "--device", device,
            "--out", str(target),
        ]
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(),
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"evaluation child failed ({completed.returncode}) for {name}/{task}/alpha={alpha}:\n"
                f"stdout tail:\n{completed.stdout[-4000:]}\nstderr tail:\n{completed.stderr[-4000:]}"
            )
        result = json.loads(target.read_text())
        result["checkpoint_label"] = name
        result["checkpoint_sha256"] = checkpoint_hash
        result["evaluation_contract"] = {
            "seed": seed, "num_envs": num_envs, "steps": steps,
            "domain_randomization": False, "obs_noise": False,
            "action_noise": 0.0, "actor_residual_alpha": alpha,
            "controller": controller,
            "task_type": task,
            "initial_condition_sha256": result.get("initial_condition_sha256"),
            "fixed_group_names": sorted(result.get("groups", {})),
        }
        target.write_text(json.dumps(result, indent=2) + "\n")
        return result

    for name, path in checkpoint_map.items():
        for task in ("L", "M"):
            for alpha in alphas:
                target = output_dir / f"{name}_{task}_actor_alpha_{alpha:.2f}.json"
                result = run_case(name, path, task, alpha, "actor", target)
                results.append({"file": target.name, "checkpoint": name, "task": task, "alpha": alpha,
                                "groups": result["groups"]})
                print(f"completed checkpoint={name} task={task} alpha={alpha:.2f}")

    # Alpha zero is also run through the explicit Prior Direct controller. These
    # results check the evaluation path, with later states allowed to diverge.
    final_path = checkpoint_map["final"]
    for task in ("L", "M"):
        target = output_dir / f"final_{task}_prior_direct.json"
        result = run_case("final", final_path, task, 0.0, "prior_direct", target)
        results.append({"file": target.name, "checkpoint": "final", "task": task,
                        "alpha": 0.0, "controller": "prior_direct", "groups": result["groups"]})
        print(f"completed checkpoint=final task={task} controller=prior_direct")

    summary = {
        "audit": "m1_action22_actor_residual_alpha_sweep_v1",
        "checkpoints": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                        for name, path in checkpoint_map.items()},
        "alphas": list(alphas), "tasks": ["L", "M"],
        "seed": seed, "num_envs": num_envs, "steps": steps,
        "later_states_diverge_by_controller": True,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_sweep(output_dir=args.output_dir, num_envs=args.num_envs,
              steps=args.steps, seed=args.seed, device=args.device, resume=not args.no_resume)


if __name__ == "__main__":
    main()
