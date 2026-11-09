"""Bounded validation orchestration; training always calls SuperMimicLearner.train."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2.checkpoint import map_actor
from unisupermimic.tasks.velocity.go2.config import enable_randomization, load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.terrain import apply_profile, bind_perception

ROOT = Path(__file__).resolve().parents[1]


def import_actor(policy):
    manifest = json.loads(
        (ROOT / "migration/source_reference/checkpoint_manifest.json").read_text()
    )
    path = Path(manifest["path"])
    if hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != manifest["sha256"]:
        raise ValueError("reference checkpoint fingerprint changed")
    return map_actor(policy, torch.load(path, map_location="cpu", weights_only=False, mmap=True))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class ResourceMonitor:
    """Sample whole-process GPU allocations, including Warp and CUDA context."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                )
                for line in raw.splitlines():
                    pid, amount = line.split(",")
                    if int(pid.strip()) == os.getpid():
                        self.samples.append(float(amount.strip()))
            except (ValueError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(0.1)

    def finish(self, out):
        self.stop_event.set()
        self.thread.join(timeout=4)
        dump(
            out / "resource_usage.json",
            {
                "sampling_interval_s": 0.1,
                "samples": len(self.samples),
                "peak_process_gpu_mib": max(self.samples) if self.samples else None,
                "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "note": "nvidia-smi process allocation includes PyTorch/Warp/context; a sampled peak is a lower bound.",
            },
        )


def physical(env, args, out):
    mdp = env.supermimic_mdp
    pi = Policy(42, 3, hist=5, num_joints=12, goal_mode="direct").to(env.device)
    mapping = import_actor(pi) if args.reference_actor else {"loaded": "fresh"}
    results = []
    command = env.command_manager.get_term("velocity")
    commands = [
        (0.5, 0, 0),
        (-0.3, 0, 0),
        (0, 0.25, 0),
        (0, -0.25, 0),
        (0, 0, 0.5),
        (0, 0, -0.5),
        (0, 0, 0),
    ]
    for goal in commands:
        command.cfg.fixed_command = goal
        env.reset()
        history = torch.zeros(env.num_envs, 5, 42, device=env.device)
        velocities = []
        yaw = []
        heights = []
        actions = []
        resets = 0
        parity = []
        synchronize(env.device)
        started = time.perf_counter()
        with torch.no_grad():
            for step in range(args.steps):
                info = env.current_info()
                obs = mdp.obs(unflatten_state(info["s"]), info["a_prev"], info["xi"], info["f"])
                history = torch.cat((history[:, 1:], obs[:, None]), 1)
                history = torch.where(
                    (info["ep_step"] == 0)[:, None, None], obs[:, None].expand_as(history), history
                )
                action = pi(history, info["g"])
                state = env.step(action)
                tr = state.info["supermimic"]
                assert torch.equal(tr["g"], info["g"]), "command shifted before reward"
                torch.testing.assert_close(tr["s"], info["s"], atol=1e-6, rtol=1e-6)
                assert all(torch.isfinite(v).all() for v in tr.values()), "non-finite transition"
                initial, nxt = unflatten_state(tr["s"]), unflatten_state(tr["s1"])
                real_recomputed = mdp.reward_transition(
                    initial,
                    nxt,
                    tr["g"],
                    tr["a"],
                    tr["a_prev"],
                    measured_torque=env.velocity_measured_torque,
                )
                torch.testing.assert_close(real_recomputed, tr["r"], atol=1e-6, rtol=1e-6)
                proxy = mdp.reward_transition(initial, nxt, tr["g"], tr["a"], tr["a_prev"])
                parity.append(float((proxy - tr["r"]).abs().max()))
                resets += int(tr["done"].sum())
                current = state.info["supermimic_state"]
                fresh = current["ep_step"] == 0
                if fresh.any():
                    assert (current["a_prev"][fresh] == 0).all(), "reset action history is dirty"
                    assert (current["s"][fresh, 2] > 0.3).all(), (
                        "reset state still contains terminal pose"
                    )
                body_v = (nxt[1].transpose(-1, -2) @ nxt[2][..., None]).squeeze(-1)
                velocities.append(body_v[:, :2].cpu())
                yaw.append((nxt[1].transpose(-1, -2) @ nxt[3][..., None]).squeeze(-1)[:, 2].cpu())
                heights.append(nxt[0][:, 2].cpu())
                actions.append(action.cpu())
        synchronize(env.device)
        a = torch.stack(actions)
        v = torch.stack(velocities)
        w = torch.stack(yaw)
        results.append(
            {
                "command": goal,
                "steps": args.steps,
                "envs": env.num_envs,
                "wall_s": time.perf_counter() - started,
                "mean_velocity_xy": v.mean((0, 1)).tolist(),
                "mean_yaw_rate": float(w.mean()),
                "velocity_rmse": float(((v - v.new_tensor(goal[:2])) ** 2).mean().sqrt()),
                "yaw_rmse": float(((w - goal[2]) ** 2).mean().sqrt()),
                "done_count": resets,
                "minimum_root_height": float(torch.stack(heights).min()),
                "action_abs_max": float(a.abs().max()),
                "saturation_fraction": float((a.abs() > 0.98).float().mean()),
                "reward_proxy_max_abs_gap": max(parity),
            }
        )
    dump(
        out / "physical.json",
        {
            "actor": mapping,
            "backend": args.backend,
            "cases": results,
            "note": "Short physical/command protocol check; command tracking acceptance requires longer matched evaluation.",
        },
    )


def training(env, args, out):
    cfg = SuperMimicConfig(
        goal_mode="direct",
        goal_frames=1,
        hist=5,
        wm_batch=args.batch,
        pi_batch=args.batch,
        buffer_steps=128,
        compile=False,
        amp="none",
        no_sync=False,
        pi_opt="adam",
        wm_opt="adam",
        wm_window=8,
        wm_batches=1,
        warmup_steps=64,
        iterations=args.iterations,
        wm_only_iters=2,
        log_every=1,
        critic_epochs=5,
        profile=True,
        seed=2026,
    )
    learner = SuperMimicLearner(env, cfg, out, device=args.device, backend=args.backend)
    mapping = import_actor(learner.pi) if args.reference_actor else {"loaded": "fresh"}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # Full loop runs unchanged. One bounded profiler covers every actual stage.
    with torch.profiler.profile(
        activities=activities, record_shapes=False, profile_memory=True
    ) as profiler:
        summary = learner.train()
    profiler.export_chrome_trace(str(out / "profiler_trace.json"))
    (out / "profiler_operators.txt").write_text(
        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=35)
    )
    assert all(
        torch.isfinite(p).all()
        for m in (learner.pi, learner.wm, learner.critic)
        for p in m.parameters()
    )
    assert all(torch.isfinite(p.grad).all() for p in learner.pi.parameters() if p.grad is not None)
    checks = []
    for horizon in (1, 8, 32):
        window = learner.buffer.sample_windows(args.batch, horizon)
        with torch.no_grad():
            state = unflatten_state(window["s"][:, 0])
            for i in range(horizon):
                state = learner.wm.step(
                    state, window["a"][:, i], learner.mdp.normalize_context(window["xi"][:, i])
                )
            error = learner.wm.state_error(state, unflatten_state(window["s1"][:, -1]))
        mask = window["mask"][:, -1]
        assert mask.any(), f"No valid replay segment for WM horizon {horizon}"
        assert torch.isfinite(error).all()
        checks.append(
            {
                "horizon": horizon,
                "valid_samples": int(mask.sum()),
                "state_rmse": float(
                    (error.square().mean(-1) * mask).sum().div(mask.sum().clamp_min(1)).sqrt()
                ),
            }
        )
    timings = {
        key: {
            "count": len(v),
            "mean_ms": float(np.mean(v)),
            "median_ms": float(np.median(v)),
            "min_ms": min(v),
            "max_ms": max(v),
            "samples_ms": v,
        }
        for key, v in learner.profile_ms.items()
    }
    start, hist = learner.buffer.sample_starts(64, 5)
    with torch.no_grad():
        obs = learner.mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"])
        actor = learner.pi(obs, start["g"])
        perturbation = []
        for joint in range(12):
            a = torch.zeros(64, 12, device=learner.device)
            a[:, joint] = 0.1
            s = unflatten_state(start["s"])
            xi = learner.mdp.normalize_context(start["xi"])
            plus = learner.wm.step(s, a, xi)[4][:, joint]
            minus = learner.wm.step(s, -a, xi)[4][:, joint]
            diff = plus - minus
            perturbation.append(
                {
                    "joint": joint,
                    "positive_response_fraction": float((diff > 0).float().mean()),
                    "mean_q_positive_minus_negative": float(diff.mean()),
                }
            )
    report = {
        "summary": summary,
        "config": asdict(cfg),
        "actor": mapping,
        "device": args.device,
        "backend": args.backend,
        "envs": env.num_envs,
        "network_parameters": {
            k: sum(p.numel() for p in m.parameters())
            for k, m in [("wm", learner.wm), ("actor", learner.pi), ("critic", learner.critic)]
        },
        "synchronized_timings": timings,
        "wm_predictions": checks,
        "wm_action_perturbations": perturbation,
        "actor_output": {
            "abs_max": float(actor.abs().max()),
            "saturation_fraction": float((actor.abs() > 0.98).float().mean()),
            "finite_gradients": True,
        },
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else 0,
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()
        if args.device.startswith("cuda")
        else 0,
        "limitation": "Full production network widths; bounded smoke batches and iterations are explicit, no speedup claim.",
    }
    dump(out / "training.json", report)


def terrain_check(env, args, out):
    import mujoco

    # Public detached playback oracle: contact diagnostics, never backend internals.
    # Nominal terrain tests have no DR, so oracle and simulation share parameters.
    if args.domain_randomization:
        raise ValueError(
            "Detached contact oracle does not include runtime DR; test terrain without DR"
        )
    model = env.get_playback_model(0)
    data = mujoco.MjData(model)
    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "go2_terrain")
    contact_stats = {
        "terrain_contacts": 0,
        "elevated_contacts": 0,
        "maximum_force_norm": 0.0,
        "measurement": "public playback model + public physics snapshot, mj_forward/mj_contactForce; not live backend contact buffer",
    }

    def contacts():
        parts = env.get_physics_state_layout().split_state(env.get_physics_state_snapshot())
        for index in range(env.num_envs):
            data.time = parts.time[index]
            data.qpos[:] = parts.qpos[index]
            data.qvel[:] = parts.qvel[index]
            mujoco.mj_forward(model, data)
            for index in range(data.ncon):
                con = data.contact[index]
                if terrain_id >= 0 and terrain_id in con.geom:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    contact_stats["terrain_contacts"] += 1
                    contact_stats["elevated_contacts"] += int(con.pos[2] > 0.005)
                    contact_stats["maximum_force_norm"] = max(
                        contact_stats["maximum_force_norm"], float(np.linalg.norm(force[:3]))
                    )

    samples = []
    scanner = None
    if args.terrain != "flat":
        scanner = bind_perception(env, np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [1.0, 0.0]]))
    for i in range(args.steps):
        state = env.step(torch.zeros(env.num_envs, 12, device=env.device))
        tr = state.info["supermimic"]
        assert torch.isfinite(tr["s1"]).all()
        if i % 10 == 0:
            contacts()
        if i % 20 == 0:
            samples.append(
                {
                    "step": i,
                    "local_root_height": tr["s1"][:, 2].tolist(),
                    "scan_world_height": None if scanner is None else scanner.scan().tolist(),
                }
            )
        if i == args.steps // 2:
            env.reset(torch.tensor([0], device=env.device))
    if scanner is not None:
        # Separate controlled drop over the first raised section. Normal reset
        # remains at the verified approach border; this is a contact probe only.
        env.reset()
        probe = bind_perception(
            env, np.array([[1.3, -0.3], [1.3, 0.3], [1.9, -0.3], [1.9, 0.3]]), alignment="world"
        )
        height = torch.as_tensor(probe.scan(), device=env.device).max(-1).values
        c = env.command_manager.get_term("velocity")
        original_offset = c.cfg.spawn_offset
        c.cfg.spawn_offset = (
            1.6,
            0.0,
            float((height - env.scene.env_origins(env.device)[:, 2]).max()),
        )
        env.reset()
        c.cfg.spawn_offset = original_offset
        for i in range(50):
            state = env.step(torch.zeros(env.num_envs, 12, device=env.device))
            assert torch.isfinite(state.info["supermimic"]["s1"]).all()
            if i % 5 == 0:
                contacts()
        assert contact_stats["terrain_contacts"] > 0, "No generated-terrain contacts detected"
        assert contact_stats["maximum_force_norm"] > 0, "No terrain contact force detected"
    dump(
        out / "terrain.json",
        {
            "profile": args.terrain,
            "backend": args.backend,
            "samples": samples,
            "spawn_origins": env.scene.env_origins(env.device).cpu().tolist(),
            "heading": args.heading,
            "quarter_turns": args.quarter_turns,
            "training_performed": False,
            "contacts": contact_stats,
            "controlled_drop_steps": 50 if scanner is not None else 0,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["physics", "train", "terrain"])
    p.add_argument("--backend", choices=["mujoco", "mjwarp"], default="mujoco")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference-actor", action="store_true")
    p.add_argument("--domain-randomization", action="store_true")
    p.add_argument(
        "--terrain", choices=["flat", "rough", "slope", "stairs", "mixed"], default="flat"
    )
    p.add_argument("--heading", type=float, default=0.0)
    p.add_argument("--quarter-turns", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.envs < 1 or args.steps < 1 or args.iterations < 3 or args.batch < 2:
        p.error("positive envs/steps and iterations>=3, batch>=2 required")
    if args.mode == "train" and args.terrain != "flat":
        p.error("Complex terrain training is gated until flat baseline acceptance")
    if args.mode == "train" and args.iterations > 100:
        p.error("This validation entrypoint is bounded to at most 100 iterations")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite validation run: {out}")
    out.mkdir(parents=True)
    torch.manual_seed(2026)
    cfg = apply_profile(
        load_config(args.backend),
        args.terrain,
        backend=args.backend,
        heading=args.heading,
        quarter_turns=args.quarter_turns,
    )
    if args.domain_randomization:
        cfg = enable_randomization(cfg)
    dump(out / "env_config.json", asdict(cfg))
    dump(out / "invocation.json", vars(args))
    monitor = ResourceMonitor()
    monitor.thread.start()
    env = None
    try:
        env = make_go2_env(cfg, num_envs=args.envs, backend_type=args.backend)
        env.init_state()
        dump(
            out / "runtime.json",
            {
                "env_device": str(env.device),
                "learner_device": args.device,
                "backend_class": type(env.backend).__name__,
                "backend_capabilities": str(env.backend.get_tensor_capabilities()),
                "contract": env.supermimic_contract,
            },
        )
        {"physics": physical, "train": training, "terrain": terrain_check}[args.mode](
            env, args, out
        )
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        monitor.finish(out)


if __name__ == "__main__":
    main()
