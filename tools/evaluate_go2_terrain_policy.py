"""Fixed-seed physical policy evaluation and independent skill-curriculum test.

Evaluation collectors are small and independent of the 2048 training collector.
No WM/actor/critic optimization is performed. Optional video uses public snapshots.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_go2_terrain_b import digest
from validate_go2_migration import dump

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fixed", "curriculum"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(terrain=True)
    c = cfg.commands["velocity"]
    c.terrain_profile = "mixed"
    c.terrain_curriculum = a.mode == "curriculum"
    c.terrain_phase = "skill" if c.terrain_curriculum else "fixed"
    c.terrain_fixed_level = 0
    c.fixed_command, c.standing_fraction = (0.25 if c.terrain_curriculum else 0.4, 0.0, 0.0), 0.0
    c.resampling_time_range = (10.0, 10.0) if c.terrain_curriculum else (20.0, 20.0)
    cfg.max_episode_seconds = 10.0 if c.terrain_curriculum else 20.0
    env = make_go2_env(cfg, 25)
    env.init_state()
    manifest = json.loads(
        (ROOT / "migration/source_reference/checkpoint_manifest.json").read_text()
    )
    lc = SuperMimicConfig(
        goal_mode="direct",
        goal_frames=1,
        hist=5,
        terrain_training=True,
        terrain_conditioning=True,
        compile=False,
        amp="none",
        no_sync=False,
        buffer_steps=64,
        init_actor_from=manifest["path"],
        action_noise=0.0,
        seed=2426,
    )
    l = SuperMimicLearner(env, lc, a.output / "collector", device="cuda", backend="mujoco")
    original_digest = digest(l.pi)
    checkpoint = torch.load(a.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    l.pi.load_state_dict(checkpoint["pi"], strict=True)
    final_digest = digest(l.pi)
    cmd = env.command_manager.get_term("velocity")
    model = env.get_playback_model(0)
    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "go2_terrain")
    foot_ids = {
        i
        for i in range(model.ngeom)
        if "foot_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
    }
    oracle = mujoco.MjData(model)
    renderer, writer = None, None
    records, history = [], []
    if a.video:
        import imageio.v2 as imageio

        renderer = mujoco.Renderer(model, height=360, width=640)
        writer = imageio.get_writer(a.output / "stairs_up_level1.mp4", fps=25)
    try:
        if a.mode == "curriculum":
            env.reset(seed=2426)
            for step in range(500 * a.episodes):
                l.collect(1)
                if (step + 1) % 500 == 0:
                    history.append(dict(vector_steps=step + 1, **cmd.curriculum.summary()))
            dump(a.output / "curriculum_history.json", history)
            dump(a.output / "curriculum.json", cmd.curriculum.summary())
        else:
            for seed in (2426, 2427, 2428):
                cmd.curriculum.terrain_level[:] = torch.arange(25) // 5
                cmd.curriculum.mastery.copy_(cmd.curriculum.terrain_level)
                env.reset(seed=seed)
                generator = torch.Generator().manual_seed(seed)
                cmd.g[:, 0] = 0.35 + 0.1 * torch.rand(25, generator=generator)
                l.obs_hist.zero_()
                start = cmd.state()[0].clone()
                active = torch.ones(25, dtype=torch.bool)
                metrics = {
                    k: torch.zeros(25)
                    for k in (
                        "steps",
                        "actual_vx",
                        "actual_vy",
                        "actual_wz",
                        "xy_sq",
                        "yaw_sq",
                        "progress",
                        "expected",
                        "height",
                        "saturation",
                        "return",
                        "proxy_slip",
                        "max_lateral",
                        "falls",
                        "passed",
                        "timeouts",
                        "oracle_samples",
                        "ground_contact_samples",
                        "body_contact_samples",
                    )
                }
                command = cmd.g.clone()
                for t in range(1000):
                    l.collect(1)
                    tr = env.state.info["supermimic"]
                    s = unflatten_state(tr["s1"])
                    p1, R, v, w, *_ = s
                    local_v = (R.transpose(-1, -2) @ v[..., None]).squeeze(-1)
                    local_w = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
                    disp = p1[:, :2] - start[:, :2]
                    feet, fv = go2_feet_world(env.supermimic_mdp.source_state(s))
                    ground = env.supermimic_mdp.terrain.ground_height(feet[..., :2])
                    proxy = (feet[..., 2] - ground - 0.022) < 0.025
                    metrics["actual_vx"] += local_v[:, 0] * active
                    metrics["actual_vy"] += local_v[:, 1] * active
                    metrics["actual_wz"] += local_w[:, 2] * active
                    metrics["steps"] += active
                    metrics["xy_sq"] += (
                        (local_v[:, :2] - tr["g"][:, :2]).square().sum(-1)
                    ) * active
                    metrics["yaw_sq"] += (local_w[:, 2] - tr["g"][:, 2]).square() * active
                    metrics["expected"] += (
                        (R[:, :2, :2] @ tr["g"][:, :2, None]).squeeze(-1)[:, 0].clamp_min(0)
                        * 0.02
                        * active
                    )
                    metrics["progress"] = torch.where(active, disp[:, 0], metrics["progress"])
                    metrics["max_lateral"] = torch.where(
                        active,
                        torch.maximum(metrics["max_lateral"], disp[:, 1].abs()),
                        metrics["max_lateral"],
                    )
                    metrics["height"] += env.supermimic_mdp.local_height(s) * active
                    metrics["return"] += tr["r"] * active
                    metrics["saturation"] += (tr["a_clean"].abs() > 0.98).float().mean(-1) * active
                    metrics["proxy_slip"] += (
                        (fv[..., :2].norm(dim=-1) * proxy).sum(-1)
                        / proxy.sum(-1).clamp_min(1)
                        * active
                    )
                    if t % 25 == 0:
                        snapshot = env.get_physics_state_layout().split_state(
                            env.get_physics_state_snapshot()
                        )
                        for i in (active & ~tr["done"]).nonzero().flatten().tolist():
                            oracle.qpos[:] = snapshot.qpos[i]
                            oracle.qvel[:] = snapshot.qvel[i]
                            mujoco.mj_forward(model, oracle)
                            contacts = [
                                ct for ct in oracle.contact if terrain_id in (ct.geom1, ct.geom2)
                            ]
                            body = any(
                                (ct.geom2 if ct.geom1 == terrain_id else ct.geom1) not in foot_ids
                                for ct in contacts
                            )
                            metrics["oracle_samples"][i] += 1
                            metrics["ground_contact_samples"][i] += bool(contacts)
                            metrics["body_contact_samples"][i] += body
                    crossed = (disp[:, 0] >= 6.8) & (metrics["max_lateral"] <= 0.6) & ~tr["term"]
                    metrics["passed"] += crossed & active
                    metrics["falls"] += tr["term"] & active
                    metrics["timeouts"] += tr["done"] & ~tr["term"] & active & ~crossed
                    if renderer is not None and seed == 2426 and t % 2 == 0 and t < 500:
                        snapshot = env.get_physics_state_layout().split_state(
                            env.get_physics_state_snapshot()
                        )
                        oracle.qpos[:] = snapshot.qpos[8]
                        oracle.qvel[:] = snapshot.qvel[8]
                        mujoco.mj_forward(model, oracle)
                        camera = mujoco.MjvCamera()
                        camera.lookat[:] = snapshot.qpos[8, :3]
                        camera.distance, camera.azimuth, camera.elevation = 2.1, 135.0, -20.0
                        renderer.update_scene(oracle, camera=camera)
                        writer.append_data(renderer.render())
                    active &= ~tr["done"] & ~crossed
                    if not active.any():
                        break
                for i in range(25):
                    count = max(float(metrics["steps"][i]), 1.0)
                    oracle_n = max(float(metrics["oracle_samples"][i]), 1.0)
                    records.append(
                        dict(
                            seed=seed,
                            type=TYPES[i % 5],
                            level=i // 5,
                            command=command[i].tolist(),
                            steps=int(count),
                            xy_rmse_m_s=float((metrics["xy_sq"][i] / count).sqrt()),
                            yaw_rmse_rad_s=float((metrics["yaw_sq"][i] / count).sqrt()),
                            actual_body_vx_m_s=float(metrics["actual_vx"][i] / count),
                            actual_body_vy_m_s=float(metrics["actual_vy"][i] / count),
                            actual_body_wz_rad_s=float(metrics["actual_wz"][i] / count),
                            max_lateral_m=float(metrics["max_lateral"][i]),
                            reached_exit_distance=bool(metrics["progress"][i] >= 6.8),
                            actual_progress_m=float(metrics["progress"][i]),
                            commanded_progress_m=float(metrics["expected"][i]),
                            progress_ratio=float(
                                metrics["progress"][i] / metrics["expected"][i].clamp_min(1e-6)
                            ),
                            mean_local_height_m=float(metrics["height"][i] / count),
                            action_saturation_fraction=float(metrics["saturation"][i] / count),
                            proxy_foot_slip_m_s=float(metrics["proxy_slip"][i] / count),
                            fall=bool(metrics["falls"][i]),
                            passed=bool(metrics["passed"][i]),
                            timeout=bool(metrics["timeouts"][i]),
                            measured_ground_contact_sample_fraction=float(
                                metrics["ground_contact_samples"][i] / oracle_n
                            ),
                            measured_nonfoot_contact_sample_fraction=float(
                                metrics["body_contact_samples"][i] / oracle_n
                            ),
                            real_return=float(metrics["return"][i]),
                        )
                    )
            dump(a.output / "fixed_evaluation.json", records)
        dump(
            a.output / "summary.json",
            dict(
                mode=a.mode,
                actor_sha_before=original_digest,
                actor_sha_after=final_digest,
                identical_to_original_actor=original_digest == final_digest,
                actor_input=213,
                actor_output=12,
                seeds=[2426, 2427, 2428] if a.mode == "fixed" else [2426],
                vector_steps=l.sim_steps // 25,
                real_transitions=l.sim_steps,
                optimizer_updates=0,
                episode_seconds=cfg.max_episode_seconds,
                full_tile_progress_target_m=6.8,
                protocol="Fixed: .35-.45m/s commands by seed, 20s budget, finish at 6.8m in .6m corridor; curriculum: .25m/s, 10s, entry target1.2m",
                contact_note="Foot slip is a geometric clearance proxy; contact/body events use public-model MuJoCo forward every25steps",
            ),
        )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
