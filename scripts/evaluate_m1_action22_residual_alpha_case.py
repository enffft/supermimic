#!/usr/bin/env python3
"""Run one isolated fixed-seed Action22 residual-alpha evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_m1_prior_joint import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-type", choices=("L", "M"), required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--controller", choices=("actor", "prior_direct"), default="actor")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, task_type=args.task_type, controller=args.controller,
        actor_residual_alpha=args.alpha, num_envs=args.num_envs, steps=args.steps,
        seed=args.seed, device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
