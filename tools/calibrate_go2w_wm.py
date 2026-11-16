"""Bounded WM-only OOD diagnostic on new fresh-policy data; not a formal run."""

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.checkpoint import initialize_wm_only
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.world_model import export_world_model, tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--file-sha256", required=True)
    parser.add_argument("--tensor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()
    if args.updates <= 0 or args.output.resolve() == args.wm.resolve().parent:
        parser.error("Require positive bounded updates and a separate output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        owner = compose("config", overrides=["task=go2w_velocity/omni_v3", "algo.resume=false",
                                             "algo.checkpoint=null", "algo.load_run=null"])
    raw = OmegaConf.to_container(owner.algo, resolve=True)
    field_names = {f.name for f in fields(SuperMimicConfig)}
    launcher_fields = {"checkpoint", "algo", "algo_log_name", "resume", "num_envs", "load_run"}
    if unknown := raw.keys() - field_names - launcher_fields:
        raise ValueError(f"Unknown learner configuration fields: {sorted(unknown)}")
    if any(raw.get(key) for key in ("resume", "checkpoint", "load_run")):
        raise ValueError("Calibration requires fresh Actor/Critic; resume fields must be empty")
    cfg = SuperMimicConfig(
        **{f.name: raw[f.name] for f in fields(SuperMimicConfig) if f.name in raw}
    )
    cfg.buffer_steps = 256
    cfg.wm_batch = 128
    cfg.horizon = 16
    cfg.compile = False
    cfg.critic_terrain_conditioning = True
    cfg.pi_ema = 0
    task = load_config("omni_v3")
    task.commands["velocity"].terrain_mode_conditioning = True
    task.commands["velocity"].persistent_wm_coverage = False
    task.commands["velocity"].sampler.domain_weights = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    env = make_go2w_env(task, 32)
    try:
        c = env.command_manager.get_term("velocity")
        c.curriculum.enabled = False
        c.curriculum.phase = "fixed"
        c.curriculum.terrain_type.zero_()
        c.curriculum.terrain_level.zero_()
        env.reset()
        learner = SuperMimicLearner(env, cfg, args.output, device="cuda:0", backend="mujoco")
        initial = initialize_wm_only(
            learner,
            args.wm,
            expected_file_sha256=args.file_sha256,
            expected_tensor_sha256=args.tensor_sha256,
        )
        initial_actor = tensor_digest(learner.pi.state_dict())
        initial_critic = tensor_digest(learner.critic.state_dict())
        learner.collect(256)
        start = time.monotonic()
        losses = []
        for index in range(args.updates):
            loss, _ = learner.wm_update()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite calibration WM loss")
            losses.append(float(loss))
            if (index + 1) % 20 == 0:
                print(
                    json.dumps(
                        dict(updates=index + 1, loss=losses[-1], seconds=time.monotonic() - start)
                    ),
                    flush=True,
                )
        assert tensor_digest(learner.pi.state_dict()) == initial_actor
        assert tensor_digest(learner.critic.state_dict()) == initial_critic
        assert not learner.opt_pi.state and not learner.opt_v.state
        learner.save()
        candidate = export_world_model(
            args.output / "last.pt", args.output / "calibrated_candidate.pt"
        )
        summary = dict(
            scope="32-env WM-only diagnostic; no Actor/Critic updates; not formal training",
            source_initialization=initial,
            wm_updates=args.updates,
            seconds=time.monotonic() - start,
            transitions=learner.sim_steps,
            losses=losses,
            candidate=candidate,
            actor_unchanged=initial_actor,
            critic_unchanged=initial_critic,
            gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
