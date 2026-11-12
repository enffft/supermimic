"""Explicit bounded Go2W validation and stage migration, never a scale sweep.

Use a fresh --output directory. --stage-from loads only v2-compatible full
checkpoints into that new run; the source experiment is never written.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.terrain_plane import terrain_plane_normal


@torch.no_grad()
def evaluate_world_model(learner, batch=8):
    """Open-loop H1/H8/H32 errors on recorded actions, masked across resets."""
    output = {}
    for horizon in (1, 8, 32):
        if learner.buffer.size < horizon:
            output[f"H{horizon}"] = "SKIPPED: insufficient replay length"
            continue
        window = learner.buffer.sample_windows(batch, horizon)
        state = learner._unflatten_state(window["s"][:, 0])
        first_scan = learner._terrain_scan(state)
        for step in range(horizon):
            state = learner._wm_step(
                state, window["a"][:, step], learner.mdp.normalize_context(window["xi"][:, step])
            )
        target = learner._unflatten_state(window["s1"][:, -1])
        mask = window["mask"][:, -1]
        if not bool(mask.any()):
            output[f"H{horizon}"] = {
                "valid_sequences": 0,
                "status": "SKIPPED: all sequences cross resets",
            }
            continue
        components = {}
        for name, prediction, truth in zip(
            (
                "position",
                "rotation_matrix",
                "linear_velocity",
                "angular_velocity",
                "leg_position",
                "joint_velocity",
            ),
            state,
            target,
        ):
            error = (prediction - truth).flatten(1).square().mean(-1)
            components[name] = float(((error * mask).sum() / mask.sum().clamp_min(1)).sqrt())
        for name, indices in (("leg_velocity", slice(0, 12)), ("wheel_velocity", slice(12, 16))):
            error = (state[5][:, indices] - target[5][:, indices]).square().mean(-1)
            components[name] = float(((error * mask).sum() / mask.sum().clamp_min(1)).sqrt())
        components["valid_sequences"] = int(mask.sum())
        components["scan_change_rms"] = float(
            (learner._terrain_scan(state) - first_scan).square().mean().sqrt()
        )
        output[f"H{horizon}"] = components
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "train", "eval"), default="sim")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--wm-batch", type=int)
    parser.add_argument("--pi-batch", type=int)
    parser.add_argument("--wm-batches", type=int, default=1)
    parser.add_argument("--buffer-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--critic-minibatches", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stage", choices=("flat", "mixed", "terrain", "curriculum"), default="terrain"
    )
    parser.add_argument("--stage-from", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Exact v2 checkpoint for --mode eval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "eval" and args.checkpoint is None:
        parser.error("--mode eval requires --checkpoint")
    if args.stage_from and args.mode != "train":
        parser.error("Stage migration is a train-mode operation")
    if args.envs < 1 or args.steps < 1 or not 1 <= args.iterations <= 20:
        parser.error("Validation requires positive sizes and at most 20 iterations")
    if any(
        x is not None and x < 1 for x in (args.batch, args.wm_batch, args.pi_batch, args.wm_batches)
    ):
        parser.error("Batch sizes and WM update count must be positive")
    if args.buffer_steps < 8 or not 8 <= args.warmup_steps <= args.buffer_steps:
        parser.error("Replay and warmup must cover H8; warmup must fit in replay")
    if min(args.critic_epochs, args.critic_minibatches) < 1:
        parser.error("Critic update counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    start = time.perf_counter()
    env = make_go2w_env(load_config(args.stage), args.envs)
    try:
        env.reset()
        created = time.perf_counter() - start
        model = env.get_playback_model(0)
        result = dict(
            num_envs=args.envs,
            creation_seconds=created,
            nq=model.nq,
            nv=model.nv,
            nu=model.nu,
            simulation_dt=model.opt.timestep,
            contact_cone=int(model.opt.cone),
            impratio=float(model.opt.impratio),
        )
        if args.mode == "sim":
            started = time.perf_counter()
            action = torch.zeros(args.envs, 16, device=env.device)
            action[:, 12:] = 0.1
            for _ in range(args.steps):
                transition = env.step(action)
                if not torch.isfinite(transition.reward).all():
                    raise FloatingPointError("nonfinite real reward")
            c = env.command_manager.get_term("velocity")
            state = c.state()
            result.update(
                sim_steps_per_second=args.steps * args.envs / (time.perf_counter() - started),
                state_shape=list(flatten_state(state).shape),
                reward_mean=float(transition.reward.mean()),
                wheel_velocity_mean=state[5][:, 12:].mean(0).tolist(),
                torque_abs_max=env.velocity_measured_torque.abs().max(0).values.tolist(),
                terrain_normal=terrain_plane_normal(c.mdp.terrain.local_scan(*state[:2]), state[1])[
                    0
                ].tolist(),
            )
            env.reset()
            assert (env.current_info()["a_prev"] == 0).all()
        else:
            cfg = SuperMimicConfig(
                iterations=args.iterations,
                wm_only_iters=2,
                wm_batches=args.wm_batches,
                wm_batch=args.wm_batch or args.batch,
                pi_batch=args.pi_batch or args.batch,
                wm_window=8,
                horizon=args.horizon,
                hist=5,
                goal_mode="direct",
                goal_frames=1,
                wm_structured=True,
                terrain_training=True,
                terrain_conditioning=True,
                terrain_sensor_grad=True,
                checkpoint_replay=True,
                compile=False,
                amp="none",
                no_sync=False,
                pi_opt="adam",
                wm_opt="adam",
                warmup_steps=args.warmup_steps,
                buffer_steps=args.buffer_steps,
                profile=True,
                strict_finite=True,
                log_every=1,
                log_backend="none",
                critic_epochs=args.critic_epochs,
                critic_minibatches=args.critic_minibatches,
                seed=2026,
            )
            if args.mode == "eval":
                saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                cfg = SuperMimicConfig(**saved["args"])
                cfg.compile = False
                del saved
            learner = SuperMimicLearner(env, cfg, args.output, device=args.device, backend="mujoco")
            if args.stage_from:
                learner.load(args.stage_from, allow_command_distribution_change=True)
                # Additional bounded iterations, preserving restored optimizer age.
                learner.cfg.iterations = learner.iteration + args.iterations
            if args.mode == "eval":
                learner.load(args.checkpoint)
            else:
                result["training"] = learner.train()
            result["updates"] = learner.update_counts
            result["wm_evaluation"] = evaluate_world_model(learner)
        result.update(
            peak_CPU_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            peak_GPU_MiB=(
                torch.cuda.max_memory_allocated(args.device) / 2**20
                if args.device.startswith("cuda")
                else 0
            ),
            status="PASS",
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"status": "FAIL", "error": str(error)})
        )
        raise
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
