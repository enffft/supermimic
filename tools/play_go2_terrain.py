"""Play a blind Go2 policy in MuJoCo on a selected terrain and difficulty.

Default uses MuJoCo's interactive viewer. --headless runs the same physical
rollout without a display and prints per-episode motion/termination metrics.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "results/actor_training_B/last.pt"
TERRAINS = ("flat", "rough", "slope", "slope_down", "stairs", "stairs_down")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--terrain", choices=TERRAINS, default="stairs")
    parser.add_argument("--level", type=int, choices=range(8), default=4)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--entry-offset",
        type=float,
        default=0.0,
        help="Explicit maneuver reset along terrain entry, metres (0..2)",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2426)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON metrics path; never overwrite")
    args = parser.parse_args()
    if (
        args.seconds <= 0
        or not 0 <= args.entry_offset <= 2
        or args.episodes < 1
        or not all(math.isfinite(v) for v in (args.vx, args.vy, args.wz))
    ):
        parser.error("seconds and episodes must be positive, and commands finite")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.headless:
        # MuJoCo passive viewer owns a GUI thread; Xlib must be thread-safe
        # before the first GLFW connection (Linux desktop).
        library = ctypes.util.find_library("X11")
        if library is not None:
            ctypes.CDLL(library).XInitThreads()
    values = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    contract = values.get("task_contract", {})
    cfg = load_config(terrain=True)
    command = cfg.commands["velocity"]
    command.terrain_profile = args.terrain
    command.terrain_curriculum = False
    command.terrain_fixed_level = args.level
    if args.level > 4:
        if args.terrain not in ("stairs", "stairs_down"):
            parser.error("Only stairs may exceed Level4")
        command.terrain_max_level = 7
        command.sampler.stairs_safe_spawn = True
        command.terrain_coverage_weights = (0.2, 0.4, 0.3, 0.1, 0, 0, 0, 0)
    command.fixed_command = (args.vx, args.vy, args.wz)
    command.spawn_offset = (args.entry_offset, 0.0, 0.0)
    command.resampling_time_range = (max(args.seconds + 1, 21.0),) * 2
    cfg.max_episode_seconds = args.seconds
    cfg.seed = args.seed
    env = make_go2_env(cfg, num_envs=1, backend_type="mujoco")
    viewer = None
    try:
        env.init_state()
        for key in ("actor", "joints", "context", "action", "sim_dt", "ctrl_dt"):
            if contract.get(key) != env.supermimic_contract.get(key):
                raise ValueError(f"Checkpoint does not match Go2 deployment contract: {key}")
        from unisupermimic.tasks.velocity.go2.contract import MODEL

        source_model = MODEL.with_name("terrain_robot.xml") if contract.get("terrain") else MODEL
        if contract.get("model_sha256") != hashlib.sha256(source_model.read_bytes()).hexdigest():
            raise ValueError("Checkpoint robot model does not match the audited Go2 source")
        if values.get("args", {}).get("goal_mode") != "direct":
            raise ValueError("Checkpoint must contain a Go2 direct-goal blind actor")
        actor = Policy(
            42,
            3,
            hist=5,
            num_joints=12,
            goal_mode="direct",
            hidden=tuple(values["args"]["policy_hidden"]),
        ).to(args.device)
        actor.load_state_dict(values["pi"], strict=True)
        actor.eval()
        model = env.get_playback_model(0)
        data = mujoco.MjData(model)
        if not args.headless:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(model, data)
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        history = torch.zeros(1, 5, 42, device=args.device)
        records = []
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            history.zero_()
            start = env.current_info()["s"][0, :3].clone()
            xy_error = yaw_error = reward = slip = saturation = 0.0
            fall = False
            steps = 0
            for step in range(math.ceil(args.seconds / 0.02)):
                if viewer is not None and not viewer.is_running():
                    break
                tick = time.monotonic()
                info = env.current_info()
                state = unflatten_state(info["s"])
                obs = env.supermimic_mdp.obs(state, info["a_prev"], info["xi"], info["f"]).to(
                    args.device
                )
                if int(info["ep_step"][0]) == 0:
                    history = obs[:, None].expand(-1, 5, -1).clone()
                else:
                    history = torch.cat((history[:, 1:], obs[:, None]), 1)
                with torch.no_grad():
                    action = actor(history, info["g"].to(args.device)).to(env.device)
                transition = env.step(action).info["supermimic"]
                nxt = unflatten_state(transition["s1"])
                p, R, v, w, *_ = nxt
                local_v = (R.transpose(-1, -2) @ v[..., None]).squeeze(-1)
                local_w = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
                goal = transition["g"]
                xy_error += float((local_v[:, :2] - goal[:, :2]).square().sum())
                yaw_error += float((local_w[:, 2] - goal[:, 2]).square().sum())
                reward += float(transition["r"].sum())
                feet, fv = go2_feet_world(env.supermimic_mdp.source_state(nxt))
                terrain = env.supermimic_mdp.terrain
                ground = terrain.ground_height(feet[..., :2]) if terrain is not None else 0.0
                proxy = (feet[..., 2] - ground - 0.022) < 0.025
                slip += float(
                    ((fv[..., :2].norm(dim=-1) * proxy).sum(-1) / proxy.sum(-1).clamp_min(1)).sum()
                )
                saturation += float((action.abs() > 0.98).float().mean())
                steps += 1
                fall = bool(transition["term"][0])
                if viewer is not None:
                    snap = env.get_physics_state_layout().split_state(
                        env.get_physics_state_snapshot()
                    )
                    with viewer.lock():
                        data.qpos[:] = snap.qpos[0]
                        data.qvel[:] = snap.qvel[0]
                        data.time = float(snap.time[0])
                        mujoco.mj_forward(model, data)
                        viewer.cam.lookat[:] = snap.qpos[0, :3]
                    viewer.sync()
                    if not args.no_realtime:
                        time.sleep(max(0.0, 0.02 - (time.monotonic() - tick)))
                if bool(transition["done"][0]):
                    break
            end = p[0].cpu() if steps else start
            record = dict(
                episode=episode,
                seed=args.seed + episode,
                steps=steps,
                elapsed_sim_s=steps * 0.02,
                fallen=fall,
                displacement_world_m=(end - start).tolist(),
                velocity_xy_rmse_m_s=math.sqrt(xy_error / max(steps, 1)),
                yaw_rmse_rad_s=math.sqrt(yaw_error / max(steps, 1)),
                reward_sum=reward,
                mean_geometric_foot_slip_m_s=slip / max(steps, 1),
                actor_saturation_fraction=saturation / max(steps, 1),
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if viewer is not None and not viewer.is_running():
                break
        result = dict(
            checkpoint=str(args.checkpoint.resolve()),
            terrain=args.terrain,
            level=args.level,
            command=[args.vx, args.vy, args.wz],
            episodes=records,
            note="Foot slip uses geometric clearance proxy; fall is environment hard termination.",
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as f:
                json.dump(result, f, indent=2)
    finally:
        if viewer is not None:
            viewer.close()
            # Passive viewer close requests exit asynchronously; wait for its
            # GL thread before model/data and their context are destroyed.
            deadline = time.monotonic() + 5.0
            while viewer.is_running() and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.2)
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
