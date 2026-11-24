"""Flat and transition evaluations on independent, ordinary-reset MuJoCo episodes."""

import math

import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, matvec
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import write_json


def fixed_cases(seeds, switches=False, terrain=False):
    families = {
        "standing": [(0.0, 0.0, 0.0)],
        "forward": [(v, 0, 0) for v in (0.10, 0.30, 0.50, 0.70, 0.375)],
        "backward": [(v, 0, 0) for v in (-0.10, -0.20, -0.30, -0.175)],
        "left": [(0, v, 0) for v in (0.10, 0.20, 0.30, 0.175)],
        "right": [(0, v, 0) for v in (-0.10, -0.20, -0.30, -0.175)],
        "yaw_left": [(0, 0, v) for v in (0.20, 0.35, 0.40, 0.60, 0.475)],
        "yaw_right": [(0, 0, v) for v in (-0.20, -0.35, -0.40, -0.60, -0.475)],
        "forward_yaw": [(0.3, 0, 0.3), (0.3, 0, -0.3), (0.45, 0, 0.35), (0.45, 0, -0.35)],
        "forward_lateral": [(0.3, 0.1, 0), (0.3, -0.1, 0), (0.25, 0.175, 0), (0.25, -0.175, 0)],
        "lateral_yaw": [(0, 0.15, 0.4), (0, -0.15, -0.4), (0, 0.18, -0.35), (0, -0.18, 0.35)],
    }
    cases = []
    for seed in seeds:
        for group, goals in families.items():
            for i, goal in enumerate(goals):
                cases.append(
                    dict(name=f"{group}_{i}", family=group, goal=goal, seed=seed, domain=0)
                )
        if switches:
            for name, goals in {
                "start_stop": [(0, 0, 0), (0.30, 0, 0), (0, 0, 0)],
                "forward_reverse": [(0.30, 0, 0), (-0.20, 0, 0)],
                "lateral_reverse": [(0, 0.20, 0), (0, -0.20, 0)],
                "yaw_reverse": [(0, 0, 0.40), (0, 0, -0.40)],
                "forward_to_lateral": [(0.30, 0, 0), (0, 0.20, 0)],
                "forward_to_turn": [(0.30, 0, 0), (0.30, 0, 0.40)],
            }.items():
                cases.append(
                    dict(
                        name=name,
                        family="switch",
                        goal=goals[0],
                        sequence=goals,
                        seed=seed,
                        domain=0,
                    )
                )
        if terrain:
            for d in range(1, 8):
                for level in (0, 2):
                    cases.append(
                        dict(
                            name=f"terrain_{d}_{level}",
                            family="terrain",
                            goal=(0.3, 0, 0),
                            seed=seed,
                            domain=d,
                            level=level,
                        )
                    )
    return cases


def evaluate(
    checkpoint, output, seeds, trace_output=None, holdout=None, switches=False, terrain=False
):
    torch.set_num_threads(1)
    cases = fixed_cases(seeds, switches, terrain)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    profile = "motion_prior" if "motion_prior" in saved else "train"
    cfg, contract = saved["args"], saved["task_contract"]
    env = make_go2w_env(load_config(profile), len(cases))
    try:
        if env.supermimic_contract != contract:
            raise ValueError("Evaluation task contract differs")
        term = env.command_manager.get_term("velocity")
        cur = term.curriculum
        cur.enabled = False
        cur.coverage_level.fill_(-1)
        domain = torch.tensor([row["domain"] for row in cases])
        term.command_domain.copy_(domain)
        cur.terrain_type.copy_((domain - 1).clamp_min(0))
        cur.terrain_level.copy_(torch.tensor([row.get("level", 0) for row in cases]))
        env.reset()
        env.set_episode_length_buf(torch.zeros(env.num_envs, dtype=torch.long))
        n = len(cases)
        yaw_levels = torch.linspace(-0.025, 0.025, len(seeds)) if len(seeds) > 1 else torch.zeros(1)
        yaw = torch.tensor([yaw_levels[list(seeds).index(row["seed"])] for row in cases])
        views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        qpos = views["qpos"].clone()
        root = qpos[:, term.root_q].clone()
        root[:, 3], root[:, 6] = (yaw / 2).cos(), (yaw / 2).sin()
        qpos[:, term.root_q] = root
        env.backend.set_state_tensor(
            torch.arange(n), qpos.contiguous(), views["qvel"].clone().contiguous()
        )
        actor = Policy(
            50,
            7,
            hist=cfg["hist"],
            num_joints=16,
            hidden=cfg["policy_hidden"],
            goal_mode="direct",
            direct_spec=term.mdp.policy_spec,
        )
        actor.load_state_dict(saved["pi"], strict=True)
        actor.eval()
        goals = torch.tensor([r["goal"] for r in cases], dtype=torch.float32)
        segments = torch.tensor(
            [
                [
                    r.get("sequence", [r["goal"]])[min(i, len(r.get("sequence", [0])) - 1)]
                    for r in cases
                ]
                for i in range(3)
            ]
        )
        # A 4.2 s .70 m/s run fits inside the existing ±3.2 m flat tile boundary.
        durations = torch.tensor(
            [
                900
                if r["domain"]
                else 200 * len(r["sequence"])
                if "sequence" in r
                else 500
                if max(abs(r["goal"][0]), abs(r["goal"][1])) <= 0.30
                else 210
                for r in cases
            ]
        )
        reference_indices = [i for i, r in enumerate(cases) if r["seed"] == seeds[len(seeds) // 2]]
        trace_rows = []
        history = torch.zeros(n, cfg["hist"], 50)
        active = torch.ones(n, dtype=torch.bool)
        elapsed = torch.zeros(n)
        fall, strict_success = torch.zeros_like(active), torch.zeros_like(active)
        errors = torch.zeros(n, 3)
        sums = torch.zeros(n, 3)
        segment_errors = torch.zeros(3, n, 3)
        segment_seen = torch.zeros(3, n)
        latest_reward = torch.zeros(n)
        holdout_rows = {k: [] for k in ("before", "actions", "after", "active")}
        save_ids = torch.arange(n)
        max_steps = int(durations.max())
        with torch.inference_mode():
            for step in range(max_steps):
                segment = min(step // 200, 2)
                goal = segments[segment].clone()
                term.g[:, :3] = goal
                state = term.state()
                obs = term.mdp.obs(state, term.a_prev, term.xi, term.f)
                if step == 0:
                    history[:] = obs[:, None]
                else:
                    history = torch.cat((history[:, 1:], obs[:, None]), 1)
                if trace_output and step % 2 == 0:
                    native = env.backend.get_state_views(("qpos",), device="cpu")["qpos"]
                    trace_rows.append(
                        (native[reference_indices].clone(), active[reference_indices].clone())
                    )
                action = actor(history, term.g).clamp(-1, 1)
                if holdout:
                    holdout_rows["before"].append(flatten_state(state)[save_ids].clone())
                    holdout_rows["actions"].append(action[save_ids].clone())
                result = env.step(action)
                after = env.velocity_next_state
                actual = torch.cat(
                    (
                        matvec(after[1].transpose(-1, -2), after[2])[:, :2],
                        matvec(after[1].transpose(-1, -2), after[3])[:, 2:3],
                    ),
                    -1,
                )
                delta = (actual - goal).square()
                errors += delta * active[:, None]
                sums += actual * active[:, None]
                elapsed += active
                latest_reward += result.reward * active
                eligible = (step % 200 >= 50) & active
                segment_errors[segment] += delta * eligible[:, None]
                segment_seen[segment] += eligible
                fall |= active & term.mdp.hard_termination(after)
                strict_success |= active & env.termination_manager.get_term("traversal_finished")
                if holdout:
                    holdout_rows["after"].append(flatten_state(after)[save_ids].clone())
                    holdout_rows["active"].append(
                        (active & ~result.terminated & ~result.truncated)[save_ids].clone()
                    )
                active &= ~(result.terminated | result.truncated) & (step + 1 < durations)
                if not active.any():
                    break
        rmse = (errors / elapsed[:, None].clamp_min(1)).sqrt()
        mean = sums / elapsed[:, None].clamp_min(1)
        rows = []
        for i, case in enumerate(cases):
            g = goals[i]
            moving = bool(((g.abs() < 0.05) | (g * mean[i] > 0.25 * g.square())).all())
            passed = bool(
                not fall[i] and elapsed[i] >= durations[i] - 1 and rmse[i].max() < 0.15 and moving
            )
            if "sequence" in case:
                k = len(case["sequence"])
                shot = (segment_errors[:k, i] / segment_seen[:k, i, None].clamp_min(1)).sqrt()
                passed = bool(not fall[i] and elapsed[i] >= durations[i] - 1 and shot.max() < 0.18)
            if case["domain"] >= 4:
                passed = bool(strict_success[i] and not fall[i])
            duration = float(elapsed[i]) * 0.02
            rows.append(
                {
                    **case,
                    "passed": passed,
                    "fall": bool(fall[i]),
                    "rmse": rmse[i].tolist(),
                    "mean_actual": mean[i].tolist(),
                    "duration_seconds": duration,
                    "terminal_reason": "fall"
                    if fall[i]
                    else "completed"
                    if elapsed[i] >= durations[i] - 1
                    else "boundary",
                    "initial_yaw_rad": float(yaw[i]),
                    "reward": float(latest_reward[i]),
                }
            )
        ordinary = [r for r in rows if r["family"] not in ("switch", "terrain")]

        def fraction(rows, key):
            return sum(r[key] for r in rows) / max(1, len(rows))

        pass_rates = {
            g: fraction([r for r in ordinary if r["family"] == g], "passed")
            for g in dict.fromkeys(r["family"] for r in ordinary)
        }
        pass_rates["coupled"] = fraction(
            [r for r in ordinary if "_" in r["family"] and not r["family"].startswith("yaw")],
            "passed",
        )
        standard = {
            r["family"]: r
            for r in ordinary
            if r["goal"]
            in (
                (0.0, 0.0, 0.0),
                (0.3, 0, 0),
                (-0.2, 0, 0),
                (0, 0.2, 0),
                (0, -0.2, 0),
                (0, 0, 0.35),
                (0, 0, -0.35),
            )
        }
        report = dict(
            rows=rows,
            pass_rates=pass_rates,
            flat_pass_rate=fraction(ordinary, "passed"),
            fall_rate=fraction(ordinary, "fall"),
            rmse=[
                math.sqrt(sum(r["rmse"][a] ** 2 for r in ordinary) / len(ordinary))
                for a in range(3)
            ],
            switch_pass_rate=fraction([r for r in rows if r["family"] == "switch"], "passed"),
            terrain_pass_rate=fraction([r for r in rows if r["family"] == "terrain"], "passed"),
            standard_flat_rows=[
                r
                for r in ordinary
                if r["family"] in standard and r["goal"] == standard[r["family"]]["goal"]
            ],
        )
        if trace_output:
            torch.save(
                dict(
                    cases=[cases[i] for i in reference_indices],
                    qpos=torch.stack([r[0] for r in trace_rows]),
                    active=torch.stack([r[1] for r in trace_rows]),
                    dt=0.04,
                    rows=[rows[i] for i in reference_indices],
                    profile="motion_prior",
                ),
                trace_output,
            )
        if holdout:
            batches = {k: torch.stack(v) for k, v in holdout_rows.items()}
            sample_start = torch.arange(0, max(0, len(batches["active"]) - 7), 25)
            windows = sample_start[:, None] + torch.arange(8)
            ids = batches["active"][windows].all(1) & (
                sample_start[:, None] // 200 == (sample_start[:, None] + 7) // 200
            )
            start, index = ids.nonzero(as_tuple=True)
            initial = batches["before"][sample_start[start], index]
            window_rows = windows[start].T
            holdout_cases = [cases[i] for i in index.tolist()]
            torch.save(
                dict(
                    cases=holdout_cases,
                    initial=initial,
                    actions=batches["actions"][window_rows, index[None]],
                    truth=batches["after"][window_rows, index[None]],
                    valid=batches["active"][window_rows, index[None]],
                    terrain_fingerprint=term.mdp.terrain.fingerprint,
                    contract=contract,
                    profile="motion_prior",
                    scope="independent ordinary-reset Actor-only MuJoCo windows",
                ),
                holdout,
            )
        write_json(output, report)
        return report
    finally:
        env.close()
        env.backend.close()
