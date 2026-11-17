"""Play the exact unified Go2W Actor on deterministic physical evaluation conditions.

Use --demo for signed flat commands and representative single/continuous stairs.
Levels are one-based (1..10). Default is MuJoCo desktop viewer; --headless runs
the same simulation without a window. No World Model, training or exploration.
"""

import argparse
import ctypes
import ctypes.util
import json
import math
import time
from pathlib import Path

import mujoco
import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.curriculum import DOMAINS
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import write_json


def scenarios(args):
    if not args.demo:
        return [(args.terrain, args.level, (args.vx, args.vy, args.wz))]
    return [
        ("flat", 1, goal)
        for goal in [
            (0, 0, 0),
            (0.3, 0, 0),
            (-0.2, 0, 0),
            (0, 0.2, 0),
            (0, -0.2, 0),
            (0, 0, 0.35),
            (0, 0, -0.35),
        ]
    ] + [
        ("single_step_up", 1, (0.3, 0, 0)),
        ("single_step_up", 9, (0.3, 0, 0)),
        ("single_step_down", 5, (0.3, 0, 0)),
        ("single_step_down", 9, (0.3, 0, 0)),
        ("multi_stairs_up", 6, (0.3, 0, 0)),
        ("multi_stairs_down", 6, (0.3, 0, 0)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--terrain", choices=DOMAINS, default="flat")
    parser.add_argument("--level", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0)
    parser.add_argument("--wz", type=float, default=0)
    parser.add_argument(
        "--yaw",
        type=float,
        default=0,
        help="Initial yaw in radians; evaluator center condition is zero",
    )
    parser.add_argument(
        "--seconds", type=float, default=10, help="Maximum first-episode duration (0..10s)"
    )
    parser.add_argument("--episodes", type=int, default=1, help="Repeat the scenario/demo sequence")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        not 0 < args.seconds <= 10
        or args.episodes < 1
        or not all(math.isfinite(v) for v in (args.vx, args.vy, args.wz, args.yaw))
    ):
        parser.error("Require finite commands/yaw, 0 < seconds <= 10 and positive episodes")
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite existing output")
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = saved["args"]
    env = make_go2w_env(load_config("train"), 1)
    viewer = None
    records = []
    try:
        if saved.get("task_contract") != env.supermimic_contract:
            raise ValueError("Checkpoint differs from unified Go2W task contract")
        command = env.command_manager.get_term("velocity")
        command.curriculum.enabled = False
        command.curriculum.coverage_level.fill_(-1)
        actor = Policy(
            50,
            7,
            hist=cfg["hist"],
            num_joints=16,
            hidden=cfg["policy_hidden"],
            goal_mode="direct",
            direct_spec=command.mdp.policy_spec,
        )
        actor.load_state_dict(saved["pi"], strict=True)
        actor.eval()
        iteration = saved["iteration"]
        del saved
        if not args.headless:
            library = ctypes.util.find_library("X11")
            if library:
                ctypes.CDLL(library).XInitThreads()
            from mujoco import viewer as mujoco_viewer

            model = env.get_playback_model(0)
            data = mujoco.MjData(model)
            viewer = mujoco_viewer.launch_passive(model, data)
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.opt.geomgroup[:] = 1
        with torch.no_grad():
            for episode in range(args.episodes):
                for terrain, level, goal in scenarios(args):
                    if viewer is not None and not viewer.is_running():
                        break
                    domain = DOMAINS.index(terrain)
                    command.command_domain.fill_(domain)
                    command.curriculum.terrain_type.fill_(max(0, domain - 1))
                    command.curriculum.terrain_level.fill_(level - 1)
                    env.reset()
                    env.set_episode_length_buf(torch.zeros(1, dtype=torch.int64, device=env.device))
                    command.g[:, :3] = torch.tensor(goal)
                    views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
                    qpos = views["qpos"].clone()
                    root = torch.zeros(1, 7)
                    root[:, :3] = command._origins
                    root[:, 2] = (
                        command.mdp.terrain.ground_height(root[:, :2]) + command.cfg.spawn_height
                    )
                    root[:, 3] = math.cos(args.yaw / 2)
                    root[:, 6] = math.sin(args.yaw / 2)
                    qpos[:, command.root_q] = root
                    env.backend.set_state_tensor(
                        torch.arange(1), qpos.contiguous(), views["qvel"].clone().contiguous()
                    )
                    history = torch.zeros(1, cfg["hist"], 50)
                    actual_sum = torch.zeros(3)
                    error_sum = torch.zeros(3)
                    reward = 0.0
                    reason = "evaluation_horizon"
                    fallen = False
                    traversed = False
                    print(
                        f"iteration={iteration} {terrain} level={level} goal={goal} yaw={args.yaw}",
                        flush=True,
                    )
                    steps = 0
                    for step in range(math.ceil(args.seconds / 0.02)):
                        if viewer is not None and not viewer.is_running():
                            break
                        tick = time.monotonic()
                        obs = command.mdp.obs(
                            command.state(), command.a_prev, command.xi, command.f
                        )
                        history = (
                            obs[:, None].expand(-1, cfg["hist"], -1).clone()
                            if step == 0
                            else torch.cat((history[:, 1:], obs[:, None]), 1)
                        )
                        action = actor(history, command.g).clamp(-1, 1)
                        result = env.step(action)
                        state = env.velocity_next_state
                        actual = torch.cat(
                            (
                                matvec(state[1].transpose(-1, -2), state[2])[:, :2],
                                matvec(state[1].transpose(-1, -2), state[3])[:, 2:3],
                            ),
                            -1,
                        )[0]
                        actual_sum += actual
                        error_sum += (actual - torch.tensor(goal)).square()
                        reward += float(result.reward[0])
                        fallen = bool(command.mdp.hard_termination(state)[0])
                        traversed = (
                            bool(env.termination_manager.get_term("traversal_finished")[0])
                            and not fallen
                        )
                        steps += 1
                        ended = bool(result.terminated[0] or result.truncated[0])
                        if ended:
                            reason = (
                                "+".join(
                                    name
                                    for name in env.termination_manager.active_terms
                                    if bool(env.termination_manager.get_term(name)[0])
                                )
                                or "unknown_termination"
                            )
                        if viewer is not None:
                            if not ended:
                                views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
                                with viewer.lock():
                                    data.qpos[:] = views["qpos"][0].numpy()
                                    data.qvel[:] = views["qvel"][0].numpy()
                                    data.time = steps * 0.02
                                    mujoco.mj_forward(model, data)
                                    viewer.cam.lookat[:] = state[0][0].numpy()
                            viewer.set_texts(
                                (
                                    None,
                                    None,
                                    f"it {iteration} | {terrain} L{level}",
                                    f"cmd {goal} | t={steps * 0.02:.2f}s | {reason}",
                                )
                            )
                            viewer.sync()
                            if not args.no_realtime:
                                time.sleep(max(0, 0.02 - (time.monotonic() - tick)))
                        if ended:
                            break
                    row = dict(
                        episode=episode,
                        terrain=terrain,
                        level=level,
                        goal=goal,
                        initial_yaw_rad=args.yaw,
                        steps=steps,
                        fall=fallen,
                        traversal_success=traversed,
                        terminal_reason=reason,
                        duration_seconds=steps * 0.02,
                        mean_actual_vx_vy_yaw=(actual_sum / max(steps, 1)).tolist(),
                        rmse_vx_vy_yaw=(error_sum / max(steps, 1)).sqrt().tolist(),
                        return_value=reward,
                    )
                    records.append(row)
                    print(json.dumps(row), flush=True)
        if args.output:
            write_json(
                args.output,
                dict(
                    checkpoint=str(args.checkpoint.resolve()),
                    iteration=iteration,
                    scope="frozen Actor; actual MuJoCo; one-based levels; no exploration; first episode only",
                    rows=records,
                ),
            )
    finally:
        if viewer is not None:
            viewer.close()
            time.sleep(0.3)
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
