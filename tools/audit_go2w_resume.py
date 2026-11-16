"""Load a full smoke checkpoint and verify CPU-mapped replay/optimizer recovery and BPTT."""

import argparse
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import write_json
from unisupermimic.tasks.velocity.go2w.world_model import tensor_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = SuperMimicConfig(**saved["args"])
    n = saved["replay"]["N"]
    cfg.compile = False
    env = make_go2w_env(load_config("train"), n)
    try:
        env.reset()
        learner = SuperMimicLearner(env, cfg, args.output.parent, device="cuda:0", backend="mujoco")
        learner.load(args.checkpoint)
        assert tensor_digest(learner.pi.state_dict()) == tensor_digest(
            saved.get("pi_live", saved["pi"])
        )
        assert tensor_digest(learner.wm.state_dict()) == tensor_digest(saved["wm"])
        assert learner.update_counts == saved["update_counts"]
        assert learner.buffer.size == saved["replay"]["size"]
        assert learner.buffer.history.size == saved["replay"]["history"]["size"]
        assert learner._v_mean.device == learner.device
        for value in learner.extra_state.values():
            if isinstance(value, torch.Tensor):
                assert value.device == learner.device
        assert learner.buffer.data["done"][
            learner.buffer._row(torch.tensor(learner.buffer.size - 1, device=learner.device))
        ].all()
        del saved
        before = dict(learner.update_counts)
        learner.collect(2)
        learner.env.prepare_supermimic_iteration(learner)
        wm = learner.wm_update()
        pi = learner.policy_update()
        assert all(torch.isfinite(value).all() for value in (*wm, *pi))
        report = dict(
            passed=True,
            environment_count=n,
            buffer_steps=learner.buffer.T,
            replay_size=learner.buffer.size,
            history_size=learner.buffer.history.size,
            before=before,
            after=dict(learner.update_counts),
            wm_loss=float(wm[0]),
            pi_loss=float(pi[0]),
            actor_gradnorm=float(pi[3]),
            peak_GPU_MiB=torch.cuda.max_memory_allocated() / 2**20,
            resume_semantics=learner.resume_mode,
        )
        write_json(args.output, report)
        print(report)
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
