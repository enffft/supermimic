"""Resume an existing UniLab flat checkpoint without changing its file."""

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def digest():
        return hashlib.sha256(args.source.read_bytes()).hexdigest()

    before = digest()
    values = torch.load(args.source, map_location="cpu", weights_only=True)
    fields_from_source = {
        f.name: values["args"][f.name] for f in fields(SuperMimicConfig) if f.name in values["args"]
    }
    cfg = SuperMimicConfig(**fields_from_source)
    env = make_go2_env(load_config(), 16)
    env.init_state()
    try:
        learner = SuperMimicLearner(env, cfg, args.output, device="cuda", backend="mujoco")
        learner.load(args.source)
        assert env.supermimic_contract == values["task_contract"]
        assert all(
            torch.equal(v.cpu(), values["pi"][k].cpu()) for k, v in learner.pi.state_dict().items()
        )
        learner.collect(8)
        loss, error = learner.wm_update()
        assert torch.isfinite(loss).all() and digest() == before
        result = {
            "source": str(args.source),
            "sha256": before,
            "device": "cuda",
            "legacy_contract_equal": True,
            "policy_tensors_identical": True,
            "resume_iteration": learner.iteration,
            "wm_update_loss": float(loss),
            "wm_update_err8": float(error),
            "finite": True,
            "source_not_modified": True,
        }
        (args.output / "resume.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result)
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
