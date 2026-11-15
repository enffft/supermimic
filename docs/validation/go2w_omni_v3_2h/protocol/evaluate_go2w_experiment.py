"""First-episode physics and fresh-trajectory WM assessment, without training."""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.velocity.go2.command_sampler import DOMAINS, command_modes
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT, POLICY_SPEC
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env


def scenarios():
    flat = [
        ("stop", (0., 0., 0.)), ("forward_030", (.3, 0., 0.)),
        ("forward_060", (.6, 0., 0.)), ("backward_015", (-.15, 0., 0.)),
        ("left_006", (0., .06, 0.)), ("right_006", (0., -.06, 0.)),
        ("left_015", (0., .15, 0.)), ("right_015", (0., -.15, 0.)),
        ("yaw_left_020", (0., 0., .2)), ("yaw_right_020", (0., 0., -.2)),
        ("yaw_left_035", (0., 0., .35)), ("yaw_right_035", (0., 0., -.35)),
        ("mixed", (.25, .05, .1)), ("lateral_sine", (0., 0., 0.)),
        ("yaw_sine", (0., 0., 0.)),
    ]
    out = [dict(name=name, domain=0, level=0, goal=goal) for name, goal in flat]
    out += [dict(name=f"{DOMAINS[d]}_L{level}", domain=d, level=level, goal=(.3, 0., 0.))
            for d in range(1, 6) for level in range(8)]
    return out


@torch.no_grad()
def assess(args):
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    started = time.monotonic()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    actor = Policy(50, 3, hist=5, num_joints=16, hidden=tuple(saved["args"]["policy_hidden"]),
                   goal_mode="direct", direct_spec=POLICY_SPEC).cuda().eval()
    actor.load_state_dict(saved["pi"])
    wm = WorldModel(hidden=saved["args"]["wm_hidden"], num_joints=16, context_dim=0,
                    structured=True, terrain_dim=77, state_layout=LAYOUT).cuda().eval()
    wm.load_state_dict(saved["wm"])
    contract, iteration = saved["task_contract"], saved["iteration"]
    del saved
    cases = scenarios()
    replicas = len(args.seeds) * args.replicas
    n = len(cases) * replicas
    env = make_go2w_env(load_config(args.profile), n)
    try:
        if env.supermimic_contract != contract:
            raise ValueError("Evaluation profile does not exactly match checkpoint task contract")
        env.reset()
        c = env.command_manager.get_term("velocity")
        terrain_gpu = c.mdp.terrain.to("cuda:0")
        domain = torch.tensor([case["domain"] for case in cases]).repeat_interleave(replicas)
        level = torch.tensor([case["level"] for case in cases]).repeat_interleave(replicas)
        fixed_goal = torch.tensor([case["goal"] for case in cases]).repeat_interleave(replicas, dim=0)
        c.command_domain.copy_(domain)
        cur = c.curriculum
        cur.enabled, cur.phase, cur.fixed_domain = False, "fixed", c.command_domain
        cur.terrain_type.copy_(torch.where(domain > 0, domain - 1, torch.arange(n) % 5))
        cur.terrain_level.copy_(level)
        cur.mastery.copy_(level)
        cur.steps.zero_()
        cur.episode_fall.zero_()
        env.reset()
        env.set_episode_length_buf(torch.zeros(n, dtype=torch.long, device=env.device))
        env.set_collection_exploration(0.)
        # Three reset seeds sample small legal perturbations on the SAME atlas.
        offsets = torch.cat([torch.rand(args.replicas, 3, generator=torch.Generator().manual_seed(seed))
                             for seed in args.seeds])
        offsets = ((offsets - .5) * torch.tensor([.04, .04, .05])).repeat(len(cases), 1)
        root = torch.zeros(n, 13)
        root[:, :3] = c.state()[0]
        root[:, :2] += offsets[:, :2]
        root[:, 3] = torch.cos(offsets[:, 2] / 2)
        root[:, 6] = torch.sin(offsets[:, 2] / 2)
        q = torch.cat((c.mdp.default_q.expand(n, 12), torch.zeros(n, 4)), -1)
        reset_mask = torch.ones(n, dtype=torch.bool, device=env.device)
        with env._reset_state.scoped_device_event_tensor(reset_mask):
            c.robot.write_motion_state_tensor_to_sim(root_state=root, position=q,
                                                     velocity=torch.zeros(n, 16), reset_mask=reset_mask)
        initial = c.state()
        start = initial[0][:, :2].clone()
        ground0 = c.mdp.terrain.ground_height(start)
        live, falls = torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        history = torch.zeros(n, 5, 50)
        sq, actual = torch.zeros(n, 3), torch.zeros(n, 3)
        all_sq, command_sum = torch.zeros(n, 3), torch.zeros(n, 3)
        frames, count = torch.zeros(n), torch.zeros(n)
        progress, lateral, ground_gain = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        action_abs, action_saturation = torch.zeros(n, 12), torch.zeros(n, 12)
        wm_errors = {h: dict(count=torch.zeros(n), position=torch.zeros(n), linear_velocity=torch.zeros(n),
                              angular_velocity=torch.zeros(n), leg_position=torch.zeros(n),
                              leg_velocity=torch.zeros(n), wheel_velocity=torch.zeros(n)) for h in (1, 8, 16)}
        window = []
        heldout = []
        for step in range(500):
            goal = fixed_goal.clone()
            wave = math.sin(2 * math.pi * step * .02 / 6.)
            goal[13 * replicas:14 * replicas, 1] = .15 * wave
            goal[14 * replicas:15 * replicas, 2] = .3 * wave
            c.g.copy_(goal)
            c.command_mode.copy_(command_modes(goal))
            c.crossing_episode.copy_(domain != 0)
            info = env.current_info()
            state = unflatten_state(info["s"], LAYOUT)
            obs = c.mdp.obs(state, info["a_prev"], info["xi"], info["f"])
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            history = torch.where((info["ep_step"] == 0)[:, None, None], obs[:, None].expand_as(history), history)
            action = actor(history.cuda(), goal.cuda()).cpu()
            if not torch.isfinite(action).all():
                raise FloatingPointError("Nonfinite evaluation action")
            before = live.clone()
            row = env.step(action).info[SUPERMIMIC_TRANSITION_KEY]
            if not (torch.equal(row["g"], goal) and torch.equal(row["command_domain"], domain)
                    and torch.equal(row["terrain_level"], level)):
                raise ValueError("Evaluation command or terrain metadata drift")
            state = unflatten_state(row["s1"], LAYOUT)
            measured = torch.cat((matvec(state[1].transpose(-1, -2), state[2])[:, :2],
                                  matvec(state[1].transpose(-1, -2), state[3])[:, 2:3]), -1)
            if not torch.isfinite(measured).all():
                raise FloatingPointError("Nonfinite physical velocity")
            eligible = live & (step >= 50)
            sq += (measured - goal).square() * eligible[:, None]
            actual += measured * eligible[:, None]
            command_sum += goal * eligible[:, None]
            all_sq += (measured - goal).square() * live[:, None]
            count += eligible
            frames += live
            delta = state[0][:, :2] - start
            progress = torch.where(live, delta[:, 0], progress)
            lateral = torch.maximum(lateral, delta[:, 1].abs() * live)
            ground_gain = torch.maximum(ground_gain, (c.mdp.terrain.ground_height(state[0][:, :2]) - ground0) * live)
            action_abs += row["a"][:, :12].abs() * live[:, None]
            action_saturation += (row["a"][:, :12].abs() >= .98) * live[:, None]
            falls |= live & row["term"]
            live &= ~row["done"]
            if step % 160 >= 144:
                window.append({key: row[key].clone() for key in ("s", "s1", "a", "done")} | {"live": before})
            if step % 160 == 159:
                if args.save_windows:
                    heldout.append({
                        key: torch.stack([item[key] for item in window], 1)
                        for key in ("s", "s1", "a", "done", "live")
                    })
                prediction = unflatten_state(window[0]["s"].cuda(), LAYOUT)
                valid = window[0]["live"].clone()
                for k in range(16):
                    scan = terrain_gpu.local_scan(prediction[0], prediction[1], sensor_grad=True)
                    prediction = wm.step(prediction, window[k]["a"].cuda(), torch.empty(n, 0, device="cuda"), scan)
                    h = k + 1
                    if h in wm_errors:
                        target = unflatten_state(window[k]["s1"].cuda(), LAYOUT)
                        errors = {
                            "position": (prediction[0] - target[0]).square().mean(-1),
                            "linear_velocity": (prediction[2] - target[2]).square().mean(-1),
                            "angular_velocity": (prediction[3] - target[3]).square().mean(-1),
                            "leg_position": (prediction[4] - target[4]).square().mean(-1),
                            "leg_velocity": (prediction[5][:, :12] - target[5][:, :12]).square().mean(-1),
                            "wheel_velocity": (prediction[5][:, 12:] - target[5][:, 12:]).square().mean(-1),
                        }
                        wm_errors[h]["count"] += valid
                        for name, error in errors.items():
                            if not torch.isfinite(error).all():
                                raise FloatingPointError("Nonfinite WM prediction")
                            wm_errors[h][name] += error.cpu() * valid
                    valid &= ~window[k]["done"]
                window.clear()
        rmse = (all_sq / frames.clamp_min(1)[:, None]).sqrt()
        survived = (frames == 500) & ~falls
        entry = survived & (progress >= 2.1) & (lateral <= .6) & (rmse[:, :2].square().sum(-1).sqrt() < .35) & (rmse[:, 2] < .4)
        rows = []
        for i, case in enumerate(cases):
            part = slice(i * replicas, (i + 1) * replicas)
            samples = count[part].sum()
            wm_result = {}
            for h, values in wm_errors.items():
                num = float(values["count"][part].sum())
                wm_result[str(h)] = {"valid_windows": num, "RMSE": {
                    key: float((value[part].sum() / max(num, 1)).sqrt())
                    for key, value in values.items() if key != "count"
                } if num else None}
            rows.append(case | {
                "replicas": replicas, "falls": int(falls[part].sum()), "survived_10s": int(survived[part].sum()),
                "eligible_samples": int(samples),
                "RMSE_vx_vy_yaw": (sq[part].sum(0) / samples).sqrt().tolist() if samples else None,
                "mean_actual_vx_vy_yaw": (actual[part].sum(0) / samples).tolist() if samples else None,
                "mean_command_vx_vy_yaw": (command_sum[part].sum(0) / samples).tolist() if samples else None,
                "progress_m": progress[part].tolist(), "max_ground_gain_m": ground_gain[part].tolist(),
                "entry_successes": int(entry[part].sum()) if case["domain"] else None,
                "climb_successes": int((entry[part] & (ground_gain[part] >= .03 * case["level"] - .005)).sum())
                if case["domain"] == 4 and case["level"] > 0 else None,
                "leg_action_abs_mean": (action_abs[part].sum(0) / frames[part].sum().clamp_min(1)).tolist(),
                "leg_action_saturation_fraction": (action_saturation[part].sum(0) / frames[part].sum().clamp_min(1)).tolist(),
                "fresh_trajectory_WM": wm_result,
            })
        result = dict(checkpoint=str(args.checkpoint), profile=args.profile, iteration=iteration,
                      elapsed_seconds=time.monotonic() - started, seeds=args.seeds, replicas_per_seed=args.replicas,
                      seed_scope="legal spawn xy/yaw perturbations; same deterministic terrain atlas; one training seed",
                      entry_success="10s survival, progress >=2.1m (70% of .3m/s x10s), lateral<=.6m, xyRMSE<.35,yawRMSE<.4",
                      climb_success="entry_success plus actual center ground gain >= configured riser -5mm",
                      WM_protocol="3 fresh on-policy windows per trial; masks exclude reset crossing; never replay samples; dynamic scan each prediction",
                      rows=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        if args.save_windows:
            args.save_windows.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "windows": {key: torch.cat([item[key] for item in heldout]) for key in heldout[0]},
                "domain": domain.repeat(len(heldout)), "level": level.repeat(len(heldout)),
                "task_contract": contract, "checkpoint": str(args.checkpoint),
                "note": "Fresh physical evaluation trajectories, never inserted into training replay",
            }, args.save_windows)
        print(f"Saved {args.output}", flush=True)
    finally:
        env.close()
        env.backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", default="terrain", choices=("terrain", "persistent", "persistent_stop", "persistent_lateral", "joint_range", "omni_v3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--save-windows", type=Path)
    assess(parser.parse_args())


if __name__ == "__main__":
    main()
