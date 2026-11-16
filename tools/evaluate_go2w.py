"""Fixed real-physics matrix for one 7D-conditioned Go2W Actor (no WM rollout)."""

import argparse
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.curriculum import DOMAINS, SINGLE_HEIGHTS, STAIR_HEIGHTS
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import write_json


def matrix(seeds):
    cases = []
    for seed in seeds:
        for axis, goal in [
            ("stop", (0, 0, 0)),
            ("forward", (0.3, 0, 0)),
            ("backward", (-0.2, 0, 0)),
            ("left", (0, 0.2, 0)),
            ("right", (0, -0.2, 0)),
            ("yaw_left", (0, 0, 0.35)),
            ("yaw_right", (0, 0, -0.35)),
        ]:
            cases.append(dict(name="flat_" + axis, domain=0, level=0, goal=goal, seed=seed))
        for domain in range(1, 8):
            for level in range(10):
                cases.append(
                    dict(
                        name=f"{DOMAINS[domain]}_level{level + 1}",
                        domain=domain,
                        level=level,
                        goal=(0.3, 0, 0),
                        seed=seed,
                    )
                )
    return cases


def evaluate(checkpoint, output, seeds=(2026, 2027, 2028)):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = saved["args"]
    trained_iteration = saved["iteration"]
    cases = matrix(seeds)
    task = load_config("train")
    env = make_go2w_env(task, len(cases))
    try:
        c = env.command_manager.get_term("velocity")
        cur = c.curriculum
        cur.enabled = False
        cur.coverage_level.fill_(-1)
        c.command_domain[:] = torch.tensor([case["domain"] for case in cases])
        cur.terrain_type[:] = (c.command_domain - 1).clamp_min(0)
        cur.terrain_level[:] = torch.tensor([case["level"] for case in cases])
        env.reset()
        if saved.get("task_contract") != env.supermimic_contract:
            raise ValueError("Evaluation task contract differs")
        goals = torch.tensor([case["goal"] for case in cases], dtype=torch.float32)
        c.g[:, :3] = goals
        actor = Policy(
            50,
            7,
            hist=cfg["hist"],
            num_joints=16,
            hidden=cfg["policy_hidden"],
            goal_mode="direct",
            direct_spec=c.mdp.policy_spec,
        )
        actor.load_state_dict(saved["pi"], strict=True)
        actor.eval()
        del saved
        n = len(cases)
        # Fixed, recorded seed-dependent initial yaw perturbations. No action noise.
        root = torch.zeros(n, 13)
        root[:, :3] = c._origins
        root[:, 2] = c.mdp.terrain.ground_height(root[:, :2]) + c.cfg.spawn_height
        yaw = torch.tensor([0.025 * ((case["seed"] - 2026) % 3 - 1) for case in cases])
        root[:, 3] = torch.cos(yaw / 2)
        root[:, 6] = torch.sin(yaw / 2)
        views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        qpos = views["qpos"].clone()
        qpos[:, c.root_q] = root[:, :7]
        env.backend.set_state_tensor(
            torch.arange(n), qpos.contiguous(), views["qvel"].clone().contiguous()
        )

        history = torch.zeros(n, cfg["hist"], 50)
        active = torch.ones(n, dtype=torch.bool)
        count = torch.zeros(n)
        err = torch.zeros(n, 3)
        fall = torch.zeros(n, dtype=torch.bool)
        success = torch.zeros_like(fall)
        distance = torch.zeros(n)
        yaw_distance = torch.zeros(n)
        reward = torch.zeros(n)
        start = c.state()[0].clone()
        last = start.clone()
        with torch.no_grad():
            for step in range(500):
                obs = c.mdp.obs(c.state(), c.a_prev, c.xi, c.f)
                history = torch.cat((history[:, 1:], obs[:, None]), 1)
                if step == 0:
                    history = obs[:, None].expand(-1, cfg["hist"], -1).clone()
                action = actor(history, c.g).clamp(-1, 1)
                result = env.step(action)
                s = env.velocity_next_state
                actual = torch.cat(
                    (
                        matvec(s[1].transpose(-1, -2), s[2])[:, :2],
                        matvec(s[1].transpose(-1, -2), s[3])[:, 2:3],
                    ),
                    -1,
                )
                err += (actual - goals).square() * active[:, None]
                count += active
                distance += torch.where(active, (s[0][:, :2] - last[:, :2]).norm(dim=-1), 0)
                yaw_distance += torch.where(active, actual[:, 2] * goals[:, 2].sign() * 0.02, 0)
                reward += result.reward * active
                fall |= active & c.mdp.hard_termination(s)
                # Auto-reset clears the event; termination tags expose the pre-reset success.
                traversed = (
                    (~c.mdp.hard_termination(s)) & result.terminated & (c.command_domain >= 4)
                )
                success |= active & traversed
                last = s[0].clone()
                active &= ~(result.terminated | result.truncated)
                if not active.any():
                    break
        rows = []
        for i, case in enumerate(cases):
            rmse = (err[i] / count[i].clamp_min(1)).sqrt()
            moving = (
                distance[i] > 0.3
                if abs(case["goal"][0]) + abs(case["goal"][1]) > 0.05
                else yaw_distance[i] > 0.3
                if abs(case["goal"][2]) > 0.05
                else True
            )
            passed = (
                bool(success[i])
                if case["domain"] >= 4
                else bool(not fall[i] and count[i] >= 499 and rmse.max() < 0.15 and moving)
            )
            rows.append(
                case
                | dict(
                    initial_yaw_rad=float(yaw[i]),
                    steps=int(count[i]),
                    rmse_vx_vy_yaw=rmse.tolist(),
                    fall=bool(fall[i]),
                    traversal_success=bool(success[i]),
                    passed=passed,
                    path_m=float(distance[i]),
                    return_value=float(reward[i]),
                )
            )
        grouped = {}
        for domain in range(8):
            selected = [r for r in rows if r["domain"] == domain]
            grouped[DOMAINS[domain]] = dict(
                episodes=len(selected),
                passed=sum(r["passed"] for r in selected),
                fall_rate=sum(r["fall"] for r in selected) / len(selected),
                success_rate=sum(r["passed"] for r in selected) / len(selected),
                mean_rmse_vx_vy_yaw=torch.tensor([r["rmse_vx_vy_yaw"] for r in selected])
                .mean(0)
                .tolist(),
            )
        result = dict(
            checkpoint=str(checkpoint),
            scope="real MuJoCo; fixed commands; initial yaw seeds; policy frozen",
            rows=rows,
            by_domain=grouped,
            trained_iterations=trained_iteration,
            single_heights_m=SINGLE_HEIGHTS,
            multi_riser_heights_m=STAIR_HEIGHTS,
        )
        write_json(output, result)
        return result
    finally:
        env.close()
        env.backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.output, args.seeds)
    print(json.dumps(result["by_domain"], indent=2))


if __name__ == "__main__":
    main()
