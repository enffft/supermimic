"""Fixed real-physics matrix for one 7D-conditioned Go2W Actor (no WM rollout)."""

import argparse
import json
from dataclasses import asdict
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


def evaluate(
    checkpoint, output, seeds=(2026, 2027, 2028), trace_output=None, collect_rewards=False
):
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
        # Training staggers initial episode ages. Every evaluation starts at t=0.
        env.set_episode_length_buf(torch.zeros(env.num_envs, dtype=torch.int64, device=env.device))
        assert not env.episode_length_buf.any()
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
        unique_seeds = sorted(set(seeds))
        yaw_map = {
            seed: (-0.025 + 0.05 * i / (len(unique_seeds) - 1) if len(unique_seeds) > 1 else 0.0)
            for i, seed in enumerate(unique_seeds)
        }
        yaw = torch.tensor([yaw_map[case["seed"]] for case in cases])
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
        actual_sum = torch.zeros(n, 3)
        steady_error = torch.zeros(n, 3)
        steady_count = torch.zeros(n)
        terminal_reasons = ["running"] * n
        signed_distance = torch.zeros(n)
        terminal_position = torch.zeros(n, 3)
        trace_ids = [
            i
            for i, case in enumerate(cases)
            if case["seed"] == unique_seeds[len(unique_seeds) // 2]
            and (
                case["domain"] == 0
                or (case["domain"] in (4, 5) and case["level"] in (0, 5, 8, 9))
                or (case["domain"] in (6, 7) and case["level"] in (0, 5, 9))
            )
        ]
        traces = []
        fall = torch.zeros(n, dtype=torch.bool)
        success = torch.zeros_like(fall)
        distance = torch.zeros(n)
        yaw_distance = torch.zeros(n)
        reward = torch.zeros(n)
        reward_sums = {}
        reward_parity_error = 0.0
        start = c.state()[0].clone()
        last = start.clone()
        with torch.no_grad():
            for step in range(500):
                obs = c.mdp.obs(c.state(), c.a_prev, c.xi, c.f)
                history = torch.cat((history[:, 1:], obs[:, None]), 1)
                if step == 0:
                    history = obs[:, None].expand(-1, cfg["hist"], -1).clone()
                if trace_output is not None:
                    native = env.backend.get_state_views(("qpos",), device="cpu")["qpos"]
                    traces.append(
                        dict(qpos=native[trace_ids].clone(), active=active[trace_ids].clone())
                    )
                action = actor(history, c.g).clamp(-1, 1)
                result = env.step(action)
                s = env.velocity_next_state
                if collect_rewards:
                    pre = env.velocity_pre_action
                    tags = env.velocity_terrain_tags
                    terms = c.mdp.terms(
                        pre["state"],
                        s,
                        pre["goal"],
                        env.velocity_executed_action,
                        pre["previous_action"],
                        command_domain=tags["command_domain"],
                        traversal=tags["traversal"],
                        success_peak=tags["success_peak"],
                    )
                    reconstructed = torch.stack(list(terms.values())).sum(0) * c.mdp.control.dt
                    reward_parity_error = max(
                        reward_parity_error,
                        float((reconstructed - result.reward)[active].abs().max()),
                    )
                    for name, value in terms.items():
                        reward_sums.setdefault(name, torch.zeros(n)).add_(value * active)
                actual = torch.cat(
                    (
                        matvec(s[1].transpose(-1, -2), s[2])[:, :2],
                        matvec(s[1].transpose(-1, -2), s[3])[:, 2:3],
                    ),
                    -1,
                )
                err += (actual - goals).square() * active[:, None]
                actual_sum += actual * active[:, None]
                if step >= 100:
                    steady_error += (actual - goals).square() * active[:, None]
                    steady_count += active
                signed_distance += torch.where(
                    active,
                    (actual[:, :2] * goals[:, :2]).sum(-1)
                    / goals[:, :2].norm(dim=-1).clamp_min(0.01)
                    * 0.02,
                    0.0,
                )
                terminal_position[active] = s[0][active]
                count += active
                distance += torch.where(active, (s[0][:, :2] - last[:, :2]).norm(dim=-1), 0)
                yaw_distance += torch.where(active, actual[:, 2] * goals[:, 2].sign() * 0.02, 0)
                reward += result.reward * active
                fall |= active & c.mdp.hard_termination(s)
                # Auto-reset clears the event; termination tags expose the pre-reset success.
                traversed = env.termination_manager.get_term(
                    "traversal_finished"
                ).clone() & ~c.mdp.hard_termination(s)
                success |= active & traversed
                last = s[0].clone()
                ended = active & (result.terminated | result.truncated)
                term_values = {
                    name: env.termination_manager.get_term(name).clone()
                    for name in env.termination_manager.active_terms
                }
                for i in ended.nonzero().flatten().tolist():
                    reasons = [name for name, value in term_values.items() if bool(value[i])]
                    terminal_reasons[i] = (
                        "+".join(reasons)
                        if reasons
                        else ("time_limit" if result.truncated[i] else "unknown_termination")
                    )
                active &= ~(result.terminated | result.truncated)
                if not active.any():
                    break
        rows = []
        for i, case in enumerate(cases):
            rmse = (err[i] / count[i].clamp_min(1)).sqrt()
            moving = (
                signed_distance[i] > 0.3
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
                    duration_seconds=float(count[i] * 0.02),
                    terminal_reason=terminal_reasons[i]
                    if terminal_reasons[i] != "running"
                    else "evaluation_horizon",
                    full_duration=bool(count[i] >= 499),
                    mean_actual_vx_vy_yaw=(actual_sum[i] / count[i].clamp_min(1)).tolist(),
                    steady_rmse_vx_vy_yaw=(steady_error[i] / steady_count[i].clamp_min(1))
                    .sqrt()
                    .tolist()
                    if steady_count[i] > 0
                    else None,
                    signed_progress_m=float(signed_distance[i]),
                    signed_yaw_rad=float(yaw_distance[i]),
                    final_position_world=terminal_position[i].tolist(),
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
                mean_duration_seconds=sum(r["duration_seconds"] for r in selected) / len(selected),
                full_duration_rate=sum(r["full_duration"] for r in selected) / len(selected),
                success_rate=sum(r["passed"] for r in selected) / len(selected),
                mean_rmse_vx_vy_yaw=torch.tensor([r["rmse_vx_vy_yaw"] for r in selected])
                .mean(0)
                .tolist(),
            )
        result = dict(
            checkpoint=str(checkpoint),
            evaluation_protocol="t0 counters; frozen policy; first episode; explicit termination reasons; signed motion; full 500-step non-stair admission",
            initial_episode_counter=0,
            time_horizon_seconds=10.0,
            yaw_map=yaw_map,
            pass_criteria=dict(
                tracking_max_rmse=0.15,
                min_translation_m=0.3,
                min_yaw_rad=0.3,
                minimum_nonstair_steps=499,
                stairs="physical traversal_finished event; no fall",
            ),
            scope="real MuJoCo; fixed commands; initial yaw seeds; policy frozen",
            rows=rows,
            by_domain=grouped,
            trained_iterations=trained_iteration,
            single_heights_m=SINGLE_HEIGHTS,
            multi_riser_heights_m=STAIR_HEIGHTS,
        )
        if collect_rewards:
            if reward_parity_error > 1e-6:
                raise RuntimeError(
                    f"Recomputed reward differs from physical manager: {reward_parity_error}"
                )
            contributions = []
            for i, case in enumerate(cases):
                contributions.append(
                    case
                    | dict(
                        steps=int(count[i]),
                        rate_terms={
                            k: float(v[i] / count[i].clamp_min(1)) for k, v in reward_sums.items()
                        },
                        return_terms={
                            k: float(v[i] * c.mdp.control.dt) for k, v in reward_sums.items()
                        },
                    )
                )
            by_domain = {}
            for domain, name in enumerate(DOMAINS):
                selected = torch.tensor([case["domain"] == domain for case in cases])
                by_domain[name] = dict(
                    physical_steps=int(count[selected].sum()),
                    pooled_rate_terms={
                        k: float(v[selected].sum() / count[selected].sum())
                        for k, v in reward_sums.items()
                    },
                    mean_episode_return_terms={
                        k: float(v[selected].mean() * c.mdp.control.dt)
                        for k, v in reward_sums.items()
                    },
                )
            result["reward_analysis"] = dict(
                configuration=asdict(c.mdp.reward_cfg),
                dt=c.mdp.control.dt,
                rate_semantics="weighted rates before dt; only first-episode live steps; endpoint saturated PD torque",
                recomputed_reward_max_abs_error=reward_parity_error,
                max_step_reward_normalizer=c.mdp.max_step_reward,
                rows=contributions,
                by_domain=by_domain,
            )
        if trace_output is not None:
            trace_output = Path(trace_output)
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    cases=[cases[i] for i in trace_ids],
                    qpos=torch.stack([t["qpos"] for t in traces]),
                    active=torch.stack([t["active"] for t in traces]),
                    dt=0.02,
                    rows=[rows[i] for i in trace_ids],
                ),
                trace_output,
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
    parser.add_argument("--traces", type=Path)
    parser.add_argument(
        "--rewards",
        action="store_true",
        help="Collect weighted per-term physical reward contributions",
    )
    parser.add_argument("--protocol", choices=("unified", "motion_prior"), default="unified")
    parser.add_argument("--switches", action="store_true")
    parser.add_argument("--terrain-review", action="store_true")
    parser.add_argument("--holdout", type=Path)
    args = parser.parse_args()
    if args.protocol == "motion_prior":
        from tools.evaluate_go2w_motion_prior import evaluate as evaluate_motion_prior

        report = evaluate_motion_prior(
            args.checkpoint,
            args.output,
            args.seeds,
            args.traces,
            args.holdout,
            args.switches,
            args.terrain_review,
        )
        print(json.dumps(report["pass_rates"], indent=2))
        return
    result = evaluate(args.checkpoint, args.output, args.seeds, args.traces, args.rewards)
    print(json.dumps(result["by_domain"], indent=2))


if __name__ == "__main__":
    main()
