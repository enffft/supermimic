"""Flat and transition evaluations on independent, ordinary-reset MuJoCo episodes."""

import math

import torch

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, matvec
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import write_json


def fixed_cases(seeds, switches=False, terrain=False, *, basic=False):
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
    if basic:
        families = {
            "standing": [(0, 0, 0)],
            "forward": [(0.10, 0, 0), (0.20, 0, 0)],
            "backward": [(-0.10, 0, 0), (-0.20, 0, 0)],
            "left": [(0, 0.10, 0), (0, 0.15, 0)],
            "right": [(0, -0.10, 0), (0, -0.15, 0)],
            "yaw_left": [(0, 0, 0.20), (0, 0, 0.40)],
            "yaw_right": [(0, 0, -0.20), (0, 0, -0.40)],
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
    checkpoint,
    output,
    seeds,
    trace_output=None,
    holdout=None,
    switches=False,
    terrain=False,
    *,
    protocol="motion_prior",
):
    torch.set_num_threads(1)
    legacy = protocol == "motion_prior"
    cases = fixed_cases(seeds, switches, terrain, basic=protocol == "motion_prior_basic")
    segment_steps = 200 if legacy else 100
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
                else segment_steps * len(r["sequence"])
                if "sequence" in r
                else 200
                if not legacy
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
        boundary = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)
        errors = torch.zeros(n, 3)
        sums = torch.zeros(n, 3)
        segment_errors = torch.zeros(3, n, 3)
        segment_seen = torch.zeros(3, n)
        segment_sums = torch.zeros(3, n, 3)
        latest_reward = torch.zeros(n)
        holdout_rows = {k: [] for k in ("before", "actions", "after", "active")}
        save_ids = torch.arange(n)
        max_steps = int(durations.max())
        with torch.inference_mode():
            for step in range(max_steps):
                segment = min(step // segment_steps, 2)
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
                eligible = (step % segment_steps >= (50 if legacy else 25)) & active
                segment_errors[segment] += delta * eligible[:, None]
                segment_sums[segment] += actual * eligible[:, None]
                segment_seen[segment] += eligible
                fall |= active & term.mdp.hard_termination(after)
                boundary |= active & env.termination_manager.get_term("flat_domain_boundary")
                timed_out |= active & env.termination_manager.get_term("time_out")
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
            complete = elapsed[i] >= durations[i] - int(legacy)
            invalid = bool(not legacy and boundary[i] and not fall[i] and not complete)
            if not legacy:
                limit = torch.maximum(g.new_tensor([0.05, 0.05, 0.10]), 0.35 * g.abs())
                moving = ((g == 0) | (g * mean[i] >= 0.4 * g.square())).all()
                passed = bool(not fall[i] and complete and (rmse[i] <= limit).all() and moving)
                if case["domain"]:
                    passed = bool(not fall[i] and complete and rmse[i].max() < 0.15)
            if "sequence" in case:
                k = len(case["sequence"])
                shot = (segment_errors[:k, i] / segment_seen[:k, i, None].clamp_min(1)).sqrt()
                if legacy:
                    passed = bool(not fall[i] and complete and shot.max() < 0.18)
                else:
                    seq = segments[:k, i]
                    actual = segment_sums[:k, i] / segment_seen[:k, i, None].clamp_min(1)
                    moving = ((seq == 0) | (seq * actual >= 0.4 * seq.square())).all()
                    limit = torch.maximum(seq.new_tensor([0.06, 0.06, 0.12]), 0.40 * seq.abs())
                    passed = bool(not fall[i] and complete and (shot <= limit).all() and moving)
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
                    "valid": not invalid,
                    "terminal_reason": ("fall" if legacy else "physical_fall")
                    if fall[i]
                    else "completed"
                    if complete or (strict_success[i] and case["domain"] >= 4)
                    else ("boundary" if legacy else "flat_boundary")
                    if boundary[i]
                    else "time_limit"
                    if timed_out[i]
                    else "incomplete",
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
            protocol=protocol,
            flat_window_seconds=4.0 if not legacy else None,
            switch_segment_seconds=segment_steps * 0.02,
            invalid_count=sum(not r["valid"] for r in rows),
            pass_counts={
                family: {
                    "passed": sum(r["passed"] for r in rows if r["family"] == family),
                    "attempts": sum(r["family"] == family for r in rows),
                }
                for family in dict.fromkeys(r["family"] for r in rows)
            },
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
                sample_start[:, None] // segment_steps
                == (sample_start[:, None] + 7) // segment_steps
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


def extend_physical_holdout(path, checkpoint):
    """Add eight-step held-out trials around reference targets and velocity impulses."""
    from omegaconf import OmegaConf

    from tools.train_go2w_motion_prior import owner_config
    from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat
    from unisupermimic.tasks.velocity.go2w.contract import JOINTS
    from unisupermimic.tasks.velocity.go2w.experiment import ROOT
    from unisupermimic.tasks.velocity.go2w.motion_prior import MotionBank, reference_action

    torch.set_num_threads(1)
    holdout = torch.load(path, weights_only=True, map_location="cpu")
    cfg = OmegaConf.to_container(owner_config().motion_prior, resolve=True)
    bank = MotionBank(ROOT / cfg["bank"], "cpu", cfg)
    kept = torch.arange(len(bank.rows))[bank.nominal.abs().amax(-1) <= 0.45][::3]
    indices = kept.repeat(2)
    reference = torch.arange(len(indices)) < len(kept)
    frames = bank.reset_starts[indices, 12]
    cases = [
        dict(
            domain=0,
            family=bank.rows[i]["motion_category"],
            source="reference_pd" if guided else "perturbed_actor",
            goal=bank.nominal[i].tolist(),
        )
        for i, guided in zip(indices.tolist(), reference.tolist())
    ]
    env = make_go2w_env(load_config("motion_prior"), len(cases))
    try:
        term = env.command_manager.get_term("velocity")
        term.curriculum.enabled = False
        env.reset()
        env.set_episode_length_buf(torch.zeros(len(cases), dtype=torch.long))
        term.g[:, :3] = bank.nominal[indices]
        ground = term._origins.clone()
        ground[:, 2] = term.mdp.terrain.ground_height(ground[:, :2])
        root, angle, rate = bank.reset_state(indices, frames, ground, torch.zeros(len(cases)))
        views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        qpos, qvel = views["qpos"].clone(), views["qvel"].clone()
        leg_columns = env.backend.get_joint_state_qpos_indices(JOINTS)
        qpos[:, term.root_q] = torch.where(reference[:, None], root[:, :7], qpos[:, term.root_q])
        qpos[:, leg_columns] = torch.where(reference[:, None], angle, qpos[:, leg_columns])
        body_rate = matvec(quat_to_mat(root[:, 3:7]).transpose(-1, -2), root[:, 10:13])
        root_velocity = torch.cat((root[:, 7:10], body_rate), -1)
        rng = torch.Generator().manual_seed(76631)
        impulse = torch.zeros_like(root_velocity)
        impulse[:, :2] = 0.25 * torch.randn(len(cases), 2, generator=rng)
        impulse[:, 5] = 0.20 * torch.randn(len(cases), generator=rng)
        qvel[:, term.root_v] = torch.where(reference[:, None], root_velocity, impulse)
        qvel[:, term.v_indices] = torch.where(reference[:, None], rate, qvel[:, term.v_indices])
        env.backend.set_state_tensor(torch.arange(len(cases)), qpos.contiguous(), qvel.contiguous())
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
        policy = Policy(
            50,
            7,
            hist=saved["args"]["hist"],
            num_joints=16,
            hidden=saved["args"]["policy_hidden"],
            goal_mode="direct",
            direct_spec=term.mdp.policy_spec,
        )
        policy.load_state_dict(saved["pi"], strict=True)
        policy.eval()
        initial = flatten_state(term.state()).clone()
        mixed = env.action_manager.get_term("mixed")
        actions, states, masks = [], [], []
        alive = torch.ones(len(cases), dtype=torch.bool)
        with torch.inference_mode():
            for j in range(8):
                obs = term.mdp.obs(term.state(), term.a_prev, term.xi, term.f)
                hist = (
                    obs[:, None].expand(-1, saved["args"]["hist"], -1)
                    if j == 0
                    else torch.cat((hist[:, 1:], obs[:, None]), 1)
                )
                policy_action = policy(hist, term.g).clamp(-1, 1)
                guide, _ = reference_action(
                    bank.positions[indices, frames + j + 1],
                    bank.data["wheel_qd"][indices, frames + j + 1],
                    mixed.cfg.control,
                    mixed.default,
                    mixed.limits,
                )
                result = env.step(torch.where(reference[:, None], guide, policy_action))
                actions.append(env.velocity_executed_action.clone())
                states.append(flatten_state(env.velocity_next_state).clone())
                alive &= ~(result.terminated | result.truncated)
                masks.append(alive.clone())
        holdout.update(
            cases=holdout["cases"] + cases,
            initial=torch.cat((holdout["initial"], initial)),
            actions=torch.cat((holdout["actions"], torch.stack(actions)), 1),
            truth=torch.cat((holdout["truth"], torch.stack(states)), 1),
            valid=torch.cat((holdout["valid"], torch.stack(masks)), 1),
            scope="Independent physical windows from ordinary Actor, reference PD and perturbed Actor.",
        )
        temp = path.with_suffix(".tmp")
        torch.save(holdout, temp)
        temp.replace(path)
    finally:
        env.close()
        env.backend.close()
