"""Bounded MuJoCo terrain validation. Shared learner owns every training update.

Creates a NEW output directory; never overwrites previous evaluations/checkpoints.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_go2_migration import ResourceMonitor, dump, import_actor

from unisupermimic.rl.algos.supermimic.geometry import so3_log, unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.contract import DEFAULT_Q, JOINTS, soft_limits
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world


def geometry(env, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = env.supermimic_mdp.terrain
    cmd = env.command_manager.get_term("velocity")
    model = env.get_playback_model(0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "go2_terrain")
    if gid < 0:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    origins = cmd._origins.cpu().numpy()
    points = origins[0, :2] + np.random.default_rng(4).uniform([-0.4, -0.5], [6.5, 0.5], (256, 2))
    if field.constant is not None:
        actual = np.zeros(len(points))
    else:
        actual = np.array(
            [
                10.0
                - mujoco.mj_rayHfield(model, data, gid, np.r_[xy, 10.0], np.array([0.0, 0.0, -1.0]))
                for xy in points
            ]
        )
    sampled = field.ground_height(torch.tensor(points, dtype=torch.float32)).numpy()
    error = np.abs(sampled - actual)
    if error.max() > 1e-5:
        raise AssertionError(f"collision alignment error {error.max()}")
    p = cmd.state()[0][:1].clone()
    p[:, 0] += 1.5
    R = cmd.state()[1][:1]
    scan = field.local_scan(p, R)[0].numpy().reshape(11, 7)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    image = axs[0].imshow(scan.T, origin="lower", extent=[-0.42, 0.78, -0.36, 0.36], aspect="auto")
    fig.colorbar(image, ax=axs[0], label="base z - ground (m)")
    axs[0].set(
        xlabel="body forward (m)", ylabel="body left (m)", title="77-point scan (yaw aligned)"
    )
    axs[1].scatter(actual, sampled, s=7)
    axs[1].set(xlabel="MuJoCo ray collision height (m)", ylabel="Torch sample height (m)")
    if field.constant is None:
        xx = np.linspace(origins[0, 0] - 0.4, origins[0, 0] + 7.0, 1000)
        yy = np.full_like(xx, origins[0, 1])
        zz = field.ground_height(torch.tensor(np.stack([xx, yy], -1), dtype=torch.float32)).numpy()
        axs[2].plot(xx - origins[0, 0], zz)
        axs[2].set(xlabel="entry displacement (m)", ylabel="world ground height (m)")
    fig.suptitle("Compiled collision terrain: " + cmd.cfg.terrain_profile)
    fig.tight_layout()
    fig.savefig(out / "height_scan.png", dpi=160)
    plt.close(fig)
    result = {
        "max_collision_error_m": float(error.max()),
        "mean_collision_error_m": float(error.mean()),
        "ray_count": len(points),
        "fingerprint": field.fingerprint,
        "height_shape": list(field.heights.shape),
        "spawn_origins": origins.tolist(),
        "coordinate_system": "world",
        "sampler": "MuJoCo triangle interpolation",
        "runtime_raycast": False,
        "entry_local_height_m": env.supermimic_mdp.local_height(cmd.state()).tolist(),
    }
    if cmd.curriculum is not None:
        all_origins = cmd.curriculum.origins
        lines = []
        for col in range(all_origins.shape[1]):
            origin = all_origins[cmd.cfg.terrain_fixed_level, col]
            xy = origin[None, :2] + torch.tensor([[0.0, 0.0], [0.8, 0.0], [1.5, 0.0], [2.5, 0.0]])
            lines.append(field.ground_height(xy).tolist())
        result["entry_to_interior_world_heights_m"] = lines
    dump(out / "geometry.json", result)
    return result


def physics(env, learner, steps, out):
    cmd = env.command_manager.get_term("velocity")
    mdp = env.supermimic_mdp
    records = []
    finite = True
    for goal in ((0.5, 0, 0), (-0.3, 0, 0), (0, 0.25, 0), (0, 0, 0.6), (0, 0, 0)):
        cmd.cfg.fixed_command = goal
        env.reset()
        learner.obs_hist.zero_()
        start = cmd.state()[0].clone()
        term = 0
        reset_count = 0
        consistency = 0.0
        reward_sum = 0.0
        heights = []
        torque_gap = []
        # Collect actual executed transitions through the existing replay path.
        for i in range(steps):
            learner.collect(1)
            tr = env.state.info["supermimic"]
            finite = finite and all(torch.isfinite(t).all() for t in tr.values())
            s0, s1 = unflatten_state(tr["s"]), unflatten_state(tr["s1"])
            real = mdp.reward_transition(
                s0,
                s1,
                tr["g"],
                tr["a_clean"],
                tr["a_prev"],
                executed_action=tr["a"],
                measured_torque=env.velocity_measured_torque,
            )
            consistency = max(consistency, float((real - tr["r"]).abs().max()))
            imagined = mdp.reward_transition(
                s0, s1, tr["g"], tr["a_clean"], tr["a_prev"], executed_action=tr["a"]
            )
            torque_gap.append(float((real - imagined).abs().mean()))
            term += int(tr["term"].sum())
            reset_count += int(tr["done"].sum())
            reward_sum += float(tr["r"].mean())
            heights.append(float(mdp.local_height(s1).min()))
        records.append(
            {
                "goal": goal,
                "steps": steps,
                "terminated_transitions": term,
                "done_transitions": reset_count,
                "max_recomputed_real_reward_error": consistency,
                "mean_real_vs_pd_proxy_reward_gap": float(np.mean(torque_gap)),
                "mean_step_reward": reward_sum / steps,
                "min_root_clearance_m": min(heights),
                "net_displacement_m": (cmd.state()[0] - start).mean(0).tolist(),
            }
        )
    # Controlled local contact probes: dataset coverage, not normal curriculum births.
    normal_offset = cmd.cfg.spawn_offset
    for offset in (1.3, 2.0):
        cmd.cfg.spawn_offset = (offset, 0.0, 0.0)
        cmd.cfg.fixed_command = (0.5, 0, 0)
        env.reset()
        for _ in range(64):
            learner.collect(1)
        tr = env.state.info["supermimic"]
        assert torch.isfinite(tr["s1"]).all()
        records.append(
            {
                "contact_probe_entry_offset_m": offset,
                "steps": 64,
                "root_clearance_m": mdp.local_height(unflatten_state(tr["s1"])).tolist(),
            }
        )
    if cmd.curriculum is not None:
        c = cmd.curriculum
        dump(
            out / "curriculum.json",
            {
                "enabled": c.enabled,
                "fixed_level": cmd.cfg.terrain_fixed_level,
                **{
                    key: getattr(c, key).cpu().tolist()
                    for key in (
                        "terrain_level",
                        "terrain_type",
                        "mastery",
                        "episode_progress",
                        "commanded_progress",
                        "episode_fall",
                        "velocity_tracking_error",
                        "steps",
                        "last_decision",
                    )
                },
            },
        )
    cmd.cfg.spawn_offset = normal_offset
    assert finite
    cmd.cfg.fixed_command = (0.5, 0, 0)
    env.reset()
    dump(
        out / "physics.json",
        {
            "finite": bool(finite),
            "envs": env.num_envs,
            "cases": records,
            "acceleration": "both control-period FD; legacy instantaneous substep acceleration remains different",
            "torque": "real measured jointactuatorfrc vs imagination PD proxy",
            "note": "bounded command/protocol test; no terrain locomotion success claim",
        },
    )


def prediction_diagnostics(l, out):
    result = []
    for horizon in (1, 8, 16, 32):
        window = l.buffer.sample_windows(64, horizon)
        with torch.no_grad():
            s = unflatten_state(window["s"][:, 0])
            for i in range(horizon):
                s = l._wm_step(s, window["a"][:, i], l.mdp.normalize_context(window["xi"][:, i]))
            target = unflatten_state(window["s1"][:, -1])
            err = l.wm.state_error(s, target)
            mask = window["mask"][:, -1].bool()
            if not mask.any():
                result.append({"horizon": horizon, "valid_samples": 0})
                continue

            def rmse(t):
                return float(t[mask].square().mean().sqrt())

            posp, _ = go2_feet_world(l.mdp.source_state(s))
            post, _ = go2_feet_world(l.mdp.source_state(target))
            result.append(
                {
                    "horizon": horizon,
                    "valid_samples": int(mask.sum()),
                    "position_rmse_m": rmse(err[..., :3]),
                    "attitude_rmse_rad": rmse(err[..., 3:6]),
                    "linear_velocity_rmse_m_s": rmse(err[..., 6:9]),
                    "angular_velocity_rmse_rad_s": rmse(err[..., 9:12]),
                    "joint_position_rmse_rad": rmse(err[..., 12:24]),
                    "joint_velocity_rmse_rad_s": rmse(err[..., 24:36]),
                    "foot_position_rmse_m": rmse(posp - post),
                }
            )
    dump(out / "predictions.json", result)
    return result


def matched_prediction_diagnostics(l, out):
    """Same H32-valid starts/actions for each horizon; expose selection counts."""
    window = l.buffer.sample_windows(256, 32)
    mask = window["mask"][:, -1].bool()
    result = []
    with torch.no_grad():
        state = unflatten_state(window["s"][:, 0])
        for i in range(32):
            state = l._wm_step(
                state, window["a"][:, i], l.mdp.normalize_context(window["xi"][:, i])
            )
            if i + 1 not in (1, 8, 16, 32):
                continue
            target = unflatten_state(window["s1"][:, i])
            error = l.wm.state_error(state, target)

            def rmse(t):
                return float(t[mask].square().mean().sqrt()) if mask.any() else None

            feet, _ = go2_feet_world(l.mdp.source_state(state))
            true_feet, _ = go2_feet_world(l.mdp.source_state(target))
            result.append(
                {
                    "horizon": i + 1,
                    "valid_samples": int(mask.sum()),
                    "requested_samples": 256,
                    "position_rmse_m": rmse(error[..., :3]),
                    "attitude_rmse_rad": rmse(error[..., 3:6]),
                    "linear_velocity_rmse_m_s": rmse(error[..., 6:9]),
                    "angular_velocity_rmse_rad_s": rmse(error[..., 9:12]),
                    "joint_position_rmse_rad": rmse(error[..., 12:24]),
                    "joint_velocity_rmse_rad_s": rmse(error[..., 24:36]),
                    "foot_position_rmse_m": rmse(feet - true_feet),
                }
            )
    dump(
        out / "matched_predictions.json",
        {
            "selection": "identical starts surviving full H32; potentially easier than failing contact transitions",
            "results": result,
        },
    )
    return result


def gradient_diagnostics(env, l, out, *, offset=0.0, settle_steps=25):
    """Independent public MuJoCo snapshot oracle; never mutate live backend state."""
    cmd = env.command_manager.get_term("velocity")
    original_offset = cmd.cfg.spawn_offset
    cmd.cfg.spawn_offset = (offset, 0.0, 0.0)
    env.reset()
    l.collect(settle_steps)
    model = env.get_playback_model(0)
    snapshot = env.get_physics_state_layout().split_state(env.get_physics_state_snapshot())
    qi = np.array(env.backend.get_joint_state_qpos_indices(JOINTS))
    vi = np.array(env.backend.get_joint_state_qvel_indices(JOINTS))
    layout = env.backend.get_root_state_layout("robot/base")
    rq = np.array(layout.qpos_indices)
    rv = np.array(layout.qvel_indices)
    ctrl_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    limits = soft_limits().numpy()
    q0 = np.array(DEFAULT_Q)
    live = cmd.state()
    s = tuple(t[:1].detach().to(l.device) for t in live)
    xi = l.mdp.normalize_context(cmd.xi[:1].to(l.device))
    # Stay strictly inside actuator clipping for all three FD step sizes.
    a = l.pi(l.obs_hist[:1], cmd.g[:1].to(l.device)).detach().clamp(-0.8, 0.8)

    def sim(action):
        d = mujoco.MjData(model)
        d.qpos[:] = snapshot.qpos[0]
        d.qvel[:] = snapshot.qvel[0]
        d.time = float(snapshot.time[0])
        mujoco.mj_forward(model, d)
        d.ctrl[ctrl_ids] = np.clip(q0 + 0.25 * np.clip(action, -1, 1), limits[:, 0], limits[:, 1])
        for _ in range(4):
            mujoco.mj_step(model, d)
        return np.r_[d.qpos[rq[:3]], d.qvel[rv[:3]], d.qpos[qi], d.qvel[vi]]

    # q, qdot, p and v have different units; comparisons are per-component,
    # signs exclude numerically zero responses. Report grouped norms separately.
    def output(action):
        nxt = l._wm_step(s, action[None], xi)
        return torch.cat((nxt[0], nxt[2], nxt[4], nxt[5]), -1)[0]

    jac = torch.autograd.functional.jacobian(output, a[0]).detach().cpu().numpy()
    contact_data = mujoco.MjData(model)
    contact_data.qpos[:] = snapshot.qpos[0]
    contact_data.qvel[:] = snapshot.qvel[0]
    mujoco.mj_forward(model, contact_data)
    records = []
    for eps in (0.01, 0.03, 0.1):
        finite = np.zeros_like(jac)
        for i in range(12):
            plus = a[0].cpu().numpy().copy()
            minus = plus.copy()
            plus[i] += eps
            minus[i] -= eps
            finite[:, i] = (sim(plus) - sim(minus)) / (2 * eps)
        groups = {}
        for name, sl, unit in [
            ("position", slice(0, 3), "m/action"),
            ("linear_velocity", slice(3, 6), "m/s/action"),
            ("joint_position", slice(6, 18), "rad/action"),
            ("joint_velocity", slice(18, 30), "rad/s/action"),
        ]:
            actual, pred = finite[sl], jac[sl]
            active = (np.abs(actual) > 1e-5) & (np.abs(pred) > 1e-7)
            agreement = (
                float(np.mean(np.sign(actual[active]) == np.sign(pred[active])))
                if active.any()
                else None
            )
            groups[name] = {
                "units": unit,
                "active_components": int(active.sum()),
                "sign_agreement": agreement,
                "opposite_sign_fraction": None if agreement is None else 1 - agreement,
                "physical_gradient_norm": float(np.linalg.norm(actual)),
                "wm_gradient_norm": float(np.linalg.norm(pred)),
                "normalized_error": float(
                    np.linalg.norm(pred - actual) / (np.linalg.norm(actual) + 1e-8)
                ),
            }
        records.append({"epsilon": eps, "groups": groups})
    # Oracle sanity at the same state/action: reproduce one live physics step.
    expected = sim(a[0].cpu().numpy())
    tr = env.step(a.to(env.device).expand(env.num_envs, 12).contiguous()).info["supermimic"]
    nxt = unflatten_state(tr["s1"])
    actual = torch.cat((nxt[0], nxt[2], nxt[4], nxt[5]), -1)[0].cpu().numpy()
    # Env collection exploration is disabled for the controlled oracle comparison.
    result = {
        "samples": records,
        "state": "controlled local support/contact probe; all 12 action dimensions",
        "action_anchor_abs_max": float(a.abs().max()),
        "finite_difference_inside_action_clip": True,
        "entry_offset_m": offset,
        "settle_steps": settle_steps,
        "contacts_in_public_oracle": contact_data.ncon,
        "root_local_height_m": float(l.mdp.local_height(s)[0]),
        "position_world_m": s[0][0].detach().cpu().tolist(),
        "oracle_live_max_error": float(np.max(np.abs(expected - actual))),
        "limitations": [
            "not a contact-continuity proof",
            "bounded local probes; broad contact-state coverage required before long training",
            "finite-difference steps are in normalized action units",
            "signs alone do not measure gradient magnitude accuracy",
        ],
    }
    cmd.cfg.spawn_offset = original_offset
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        choices=("flat", "rough", "slope", "slope_down", "stairs", "stairs_down", "mixed"),
        default="rough",
    )
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--wm-only", type=int, default=2)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--conditioning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument(
        "--mode", choices=("smoke", "visualize", "evaluate", "diagnose"), default="smoke"
    )
    p.add_argument("--wm-checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.iterations > 300:
        raise ValueError("This validator does not authorize long training")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    dump(out / "invocation.json", vars(args))
    cfg = load_config(terrain=True)
    command = cfg.commands["velocity"]
    command.terrain_profile = args.profile
    command.terrain_fixed_level = args.level
    command.terrain_curriculum = args.curriculum
    env = make_go2_env(cfg, args.envs)
    env.init_state()
    monitor = ResourceMonitor()
    monitor.thread.start()
    try:
        geometry(env, out)
        if args.mode == "visualize":
            return
        manifest = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "migration/source_reference/checkpoint_manifest.json"
            ).read_text()
        )
        lc = SuperMimicConfig(
            goal_mode="direct",
            goal_frames=1,
            hist=5,
            terrain_training=True,
            terrain_conditioning=args.conditioning,
            terrain_sensor_grad=True,
            compile=False,
            amp="none",
            no_sync=False,
            pi_opt="adam",
            wm_opt="adam",
            wm_batch=args.batch,
            pi_batch=args.batch,
            wm_batches=1,
            buffer_steps=512,
            warmup_steps=64,
            iterations=args.iterations,
            wm_only_iters=args.wm_only,
            log_every=1,
            profile=True,
            log_backend="none",
            seed=2026,
            action_noise=0.0,
            init_actor_from=manifest["path"],
        )
        learner = SuperMimicLearner(env, lc, out, device=args.device, backend="mujoco")
        mapping = learner.initialization["actor"]
        torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
        if args.mode == "diagnose":
            import hashlib

            if args.wm_checkpoint is None:
                raise ValueError("diagnose requires --wm-checkpoint")
            values = torch.load(args.wm_checkpoint, map_location=learner.device, weights_only=True)
            if values.get("task_contract", {}).get(
                "terrain_fingerprint"
            ) != env.supermimic_contract.get("terrain_fingerprint"):
                raise ValueError("Diagnostic WM and physics terrain differ")
            if values.get("args", {}).get("terrain_conditioning", False) != args.conditioning:
                raise ValueError("Diagnostic WM conditioning differs")
            if values.get("args", {}).get("wm_structured", False):
                raise ValueError("Diagnostic owner currently uses unstructured WM")
            learner.wm.load_state_dict(values["wm"], strict=True)
            cases = [
                gradient_diagnostics(env, learner, out, offset=offset, settle_steps=40)
                for offset in (0.0, 1.2, 2.0)
            ]
            dump(out / "action_gradients.json", cases)
            prediction_diagnostics(learner, out)
            matched_prediction_diagnostics(learner, out)
            dump(
                out / "wm_source.json",
                {
                    "path": str(args.wm_checkpoint),
                    "sha256": hashlib.sha256(args.wm_checkpoint.read_bytes()).hexdigest(),
                    "actor": mapping,
                    "actor_updated": False,
                    "wm_updated": False,
                    "physical_probe_steps": 120,
                },
            )
            return
        physics(env, learner, args.steps, out)
        if args.mode == "evaluate":
            return
        # Actual training uses Gaussian/OU exploration. Physics/oracle use none.
        learner.cfg.action_noise = 0.1
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        started = time.perf_counter()

        # Capture one REAL call per stage; synchronized range timings cover all calls.
        # Bounded traces avoid recording hundreds of CPU physics steps into gigabytes.
        def capture_once(owner, name, label):
            original = getattr(owner, name)
            captured = False

            def wrapped(*call_args, **call_kwargs):
                nonlocal captured
                if captured:
                    return original(*call_args, **call_kwargs)
                captured = True
                with torch.profiler.profile(
                    activities=activities, record_shapes=False, profile_memory=True
                ) as prof:
                    value = original(*call_args, **call_kwargs)
                prof.export_chrome_trace(str(out / f"{label}_profiler_trace.json"))
                (out / f"{label}_profiler_operators.txt").write_text(
                    prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35)
                )
                return value

            setattr(owner, name, wrapped)

        capture_once(env, "step", "simulation")
        capture_once(learner, "wm_update", "wm_fit")
        capture_once(learner, "policy_update", "actor_critic")
        summary = learner.train()
        elapsed = time.perf_counter() - started
        prediction_diagnostics(learner, out)
        learner.cfg.action_noise = 0.0
        env.set_collection_exploration(0.0)
        gradient_cases = [
            gradient_diagnostics(env, learner, out, offset=offset) for offset in (0.0, 1.2, 2.0)
        ]
        dump(out / "action_gradients.json", gradient_cases)
        assert all(
            torch.isfinite(t).all()
            for module in (learner.pi, learner.wm, learner.critic)
            for t in module.parameters()
        )
        assert all(t.grad is None or torch.isfinite(t.grad).all() for t in learner.pi.parameters())
        starts, hist = learner.buffer.sample_starts(64, 5)
        with torch.no_grad():
            obs = learner.mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"])
            action = learner.pi(obs, starts["g"])
        dump(
            out / "training.json",
            {
                "summary": summary,
                "elapsed_s": elapsed,
                "config": asdict(lc),
                "actor": mapping,
                "network_parameters": {
                    name: sum(t.numel() for t in module.parameters())
                    for name, module in [
                        ("wm", learner.wm),
                        ("actor", learner.pi),
                        ("critic", learner.critic),
                    ]
                },
                "timings": {
                    name: {
                        "count": len(v),
                        "mean_ms": float(np.mean(v)),
                        "median_ms": float(np.median(v)),
                        "max_ms": max(v),
                    }
                    for name, v in learner.profile_ms.items()
                },
                "actor_abs_max": float(action.abs().max()),
                "actor_saturation_fraction": float((action.abs() > 0.98).float().mean()),
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device.startswith("cuda")
                else 0,
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()
                if args.device.startswith("cuda")
                else 0,
                "finite_parameters_and_actor_gradients": True,
                "horizon": 32,
                "wm_window": 8,
                "warmup_note": "two WM-only iterations for plumbing SMOKE only; insufficient for terrain policy optimization",
            },
        )
        # Explicit/manual reset windows must be closed before A/B or H32 replay.
        rows = (
            torch.arange(learner.buffer.size, device=learner.device)
            + learner.buffer.ptr
            - learner.buffer.size
        ) % learner.buffer.T
        replay = learner.buffer.data
        difference = (replay["s1"][rows[:-1]] - replay["s"][rows[1:]]).abs().amax(-1)
        continuous = ~replay["done"][rows[:-1]]
        alignment = float(difference[continuous].max()) if continuous.any() else 0.0
        assert alignment < 1e-5
        dump(
            out / "replay_alignment.json",
            {
                "nonreset_max_abs_error": alignment,
                "size": learner.buffer.size,
                "ptr": learner.buffer.ptr,
                "done_boundaries": int(replay["done"][rows].sum()),
                "coordinates": "world; immutable terrain fingerprint",
            },
        )
        # Raw transitions enable identical-data A/B WM validation, not a second trainer.
        torch.save(
            {
                "data": {k: v[: learner.buffer.size].cpu() for k, v in learner.buffer.data.items()},
                "task_contract": env.supermimic_contract,
                "ptr": learner.buffer.ptr,
                "size": learner.buffer.size,
            },
            out / "replay_snapshot.pt",
        )
    finally:
        monitor.finish(out)
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
