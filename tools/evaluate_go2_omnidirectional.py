"""Independent real-physics directional grid and terrain perturbation evaluation.

Seeds explicitly perturb initial XY/yaw/joints, so repeats are not identical
nominal resets. No exploration, WM, optimizer or training curriculum is used.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world

FLAT = [
    ("stand", (0, 0, 0)),
    ("forward_low", (0.2, 0, 0)),
    ("forward_medium", (0.4, 0, 0)),
    ("forward_high", (0.8, 0, 0)),
    ("backward_low", (-0.15, 0, 0)),
    ("backward_medium", (-0.3, 0, 0)),
    ("lateral_left", (0, 0.2, 0)),
    ("lateral_right", (0, -0.2, 0)),
    ("turn_left", (0, 0, 0.5)),
    ("turn_right", (0, 0, -0.5)),
    ("forward_turn", (0.4, 0, 0.3)),
    ("forward_lateral", (0.3, 0.2, 0)),
    ("mixed", (0.3, 0.15, 0.3)),
    ("reverse_turn", (-0.2, 0, -0.3)),
    ("lateral_small", (0, 0.05, 0)),
    ("turn_small", (0, 0, -0.12)),
    ("backward_boundary", (-0.48, 0, 0)),
    ("lateral_boundary", (0, -0.38, 0)),
]
PERTURBATIONS = [(0.4, 0, 0), (0.4, 0.05, 0), (0.4, -0.05, 0), (0.4, 0, 0.1), (0.4, 0, -0.1)]
TYPES = ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down")


STAIRS_OMNI = [
    (0.4, 0.15, 0),
    (0.4, -0.15, 0),
    (0.4, 0, 0.3),
    (0.4, 0, -0.3),
    (0.4, 0.15, 0.3),
    (0.4, -0.15, -0.3),
    (0.3, 0.2, 0.4),
    (0.3, -0.2, -0.4),
    (0.2, 0.3, 0),
    (0.2, 0, 0.6),
]


def stairs_speed_commands(level):
    vy, wz = ((0.08, 0.15), (0.15, 0.30), (0.22, 0.45), (0.30, 0.60))[level]
    return [
        (0, vy, 0),
        (0, -vy, 0),
        (0, 0, wz),
        (0, 0, -wz),
        (0.4, vy, wz),
        (0.4, -vy, -wz),
        (-0.15, 0, 0),
    ]


def evaluate(
    checkpoint,
    output,
    seeds=(2426, 2427, 2428, 2429, 2430),
    seconds=20,
    quick=False,
    stairs_omni=False,
    terrain_max_level=4,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    values = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    actor = Policy(
        42,
        3,
        hist=5,
        num_joints=12,
        goal_mode="direct",
        hidden=tuple(values["args"]["policy_hidden"]),
    )
    actor.load_state_dict(values["pi"])
    actor_device = "cuda" if stairs_omni and torch.cuda.is_available() else "cpu"
    actor.to(actor_device).eval()
    records = []
    for domain in ("flat", "terrain", "stairs_maneuver") if stairs_omni else ("flat", "terrain"):
        scenarios = (
            [(name, 0, 0, cmd) for name, cmd in FLAT]
            if domain == "flat"
            else [
                (f"{TYPES[t]}_L{level}_cmd{i}", t, level, cmd)
                for level in range(1, terrain_max_level + 1)
                for t in range(5)
                if t >= 3 or level <= 4
                for i, cmd in enumerate(PERTURBATIONS)
            ]
        )
        if stairs_omni and domain == "flat":
            scenarios += [
                (f"flat_expanded_{i}", 0, 0, g)
                for i, g in enumerate(((0, 0.3, 0), (0, -0.3, 0), (0, 0, 0.65), (0, 0, -0.65)))
            ]
        if domain == "stairs_maneuver":
            scenarios = [
                (f"{TYPES[t]}_L{level}_omni{i}", t, level, g)
                for t in (3, 4)
                for level in range(1, terrain_max_level + 1)
                for i, g in enumerate(STAIRS_OMNI)
            ]
            scenarios += [
                (f"{TYPES[t]}_L{level}_S{speed}_{i}", t, level, g)
                for t in (3, 4)
                for level in range(1, terrain_max_level + 1)
                for speed in range(4)
                for i, g in enumerate(stairs_speed_commands(speed))
            ]
        if quick:
            scenarios = (
                [
                    s
                    for s in scenarios
                    if s[0]
                    in (
                        "stand",
                        "forward_medium",
                        "backward_medium",
                        "lateral_left",
                        "lateral_right",
                        "turn_left",
                        "turn_right",
                    )
                ]
                if domain == "flat"
                else [
                    s
                    for s in scenarios
                    if s[1] in (3, 4) and s[2] in (1, 4) and s[3] == (0.4, 0, 0)
                ]
            )
        episode_seconds = min(seconds, 4.0) if domain == "stairs_maneuver" else seconds
        cfg = load_config(terrain=True)
        c = cfg.commands["velocity"]
        c.terrain_max_level = terrain_max_level
        if terrain_max_level == 7:
            c.terrain_coverage_weights = (0.2, 0.4, 0.3, 0.1, 0, 0, 0, 0)
            c.sampler.stairs_safe_spawn = True
        c.terrain_profile = "flat" if domain == "flat" else "mixed"
        c.terrain_curriculum = False
        c.terrain_phase = "fixed"
        c.fixed_command = (0, 0, 0)
        c.resampling_time_range = (seconds + 1, seconds + 1)
        if domain == "stairs_maneuver":
            c.spawn_offset = (1.6, 0, 0)
        cfg.max_episode_seconds = episode_seconds
        c.reset_jitter = (0.02, 0.03, 0.01)
        env = make_go2_env(cfg, len(scenarios))
        env.init_state()
        cmd = env.command_manager.get_term("velocity")
        for key in ("actor", "joints", "context", "action", "sim_dt", "ctrl_dt"):
            if values["task_contract"][key] != env.supermimic_contract[key]:
                raise ValueError(f"Actor deployment protocol mismatch: {key}")
        try:
            for seed in seeds:
                n = len(scenarios)
                if cmd.curriculum is not None:
                    cmd.curriculum.terrain_type[:] = torch.tensor([s[1] for s in scenarios])
                    cmd.curriculum.terrain_level[:] = torch.tensor([s[2] for s in scenarios])
                env.reset(seed=seed)
                cmd.g[:] = torch.tensor([s[3] for s in scenarios])
                start = cmd.state()[0].clone()
                active = torch.ones(n, dtype=torch.bool)
                hist = torch.zeros(n, 5, 42)
                keys = (
                    "steps",
                    "vx",
                    "vy",
                    "wz",
                    "xy_sq",
                    "yaw_sq",
                    "height",
                    "sat",
                    "reward",
                    "slip",
                    "yaw_displacement",
                    "expected",
                    "max_lateral",
                    "outside_tile",
                )
                stats = {k: torch.zeros(n) for k in keys}
                falls = torch.zeros(n, dtype=torch.bool)
                end = start.clone()
                platform_reached = torch.zeros(n, dtype=torch.bool)
                max_terrain_change = torch.zeros(n)
                tile_entry = (
                    cmd.curriculum.origins[
                        cmd.curriculum.terrain_level, cmd.curriculum.terrain_type, :2
                    ]
                    if domain != "flat"
                    else start[:, :2]
                )
                entry_ground = cmd.mdp.terrain.ground_height(tile_entry)
                target_ground = cmd.mdp.terrain.ground_height(
                    tile_entry + start.new_tensor([3.4, 0])
                )
                terrain_sign = torch.where(target_ground >= entry_ground, 1.0, -1.0)
                for _ in range(math.ceil(episode_seconds / 0.02)):
                    info = env.current_info()
                    s = unflatten_state(info["s"])
                    obs = cmd.mdp.obs(s, info["a_prev"], info["xi"], info["f"])
                    hist = torch.where(
                        (info["ep_step"] == 0)[:, None, None],
                        obs[:, None].expand(-1, 5, -1),
                        torch.cat((hist[:, 1:], obs[:, None]), 1),
                    )
                    with torch.no_grad():
                        a = actor(hist.to(actor_device), info["g"].to(actor_device)).cpu()
                    tr = env.step(a).info["supermimic"]
                    nxt = unflatten_state(tr["s1"])
                    p, R, v, w, *_ = nxt
                    lv = (R.transpose(-1, -2) @ v[..., None]).squeeze(-1)
                    lw = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
                    terrain = cmd.mdp.terrain
                    ground = terrain.ground_height(p[:, :2])
                    local = p[:, :2] - tile_entry
                    terrain_change = (ground - entry_ground) * terrain_sign
                    max_terrain_change = torch.maximum(max_terrain_change, terrain_change * active)
                    platform_reached |= (
                        active
                        & ~tr["term"]
                        & (local[:, 0] >= 3.0)
                        & (local[:, 0] <= 3.8)
                        & (local[:, 1].abs() <= 0.6)
                        & ((ground - target_ground).abs() < 0.025)
                    )
                    feet, fv = go2_feet_world(cmd.mdp.source_state(nxt))
                    proxy = feet[..., 2] - terrain.ground_height(feet[..., :2]) - 0.022 < 0.025
                    vals = dict(
                        steps=torch.ones(n),
                        vx=lv[:, 0],
                        vy=lv[:, 1],
                        wz=lw[:, 2],
                        xy_sq=(lv[:, :2] - tr["g"][:, :2]).square().sum(-1),
                        yaw_sq=(lw[:, 2] - tr["g"][:, 2]).square(),
                        height=p[:, 2] - ground,
                        sat=(a.abs() > 0.98).float().mean(-1),
                        reward=tr["r"],
                        outside_tile=(
                            (
                                p[:, :2]
                                - (
                                    start[:, :2]
                                    + p.new_tensor((1.8 if domain == "stairs_maneuver" else 3.4, 0))
                                )
                            )
                            .abs()
                            .amax(-1)
                            > 3.5
                        ).float()
                        if domain != "flat"
                        else torch.zeros(n),
                        slip=(fv[..., :2].norm(dim=-1) * proxy).sum(-1)
                        / proxy.sum(-1).clamp_min(1),
                        yaw_displacement=lw[:, 2] * 0.02,
                        expected=(R[:, :2, :2] @ tr["g"][:, :2, None])
                        .squeeze(-1)[:, 0]
                        .clamp_min(0)
                        * 0.02,
                    )
                    for k, val in vals.items():
                        stats[k] += val * active
                    stats["max_lateral"] = torch.maximum(
                        stats["max_lateral"], (p[:, 1] - start[:, 1]).abs() * active
                    )
                    end = torch.where(active[:, None], p, end)
                    falls |= active & tr["term"]
                    active &= ~tr["done"]
                    if not active.any():
                        break
                for i, (name, t, level, g) in enumerate(scenarios):
                    st = max(float(stats["steps"][i]), 1)
                    disp = (end[i] - start[i]).tolist()
                    rec = dict(
                        domain=domain,
                        scenario=name,
                        terrain="flat" if domain == "flat" else TYPES[t],
                        level=level,
                        seed=seed,
                        command=list(g),
                        actual_vx_m_s=float(stats["vx"][i] / st),
                        actual_vy_m_s=float(stats["vy"][i] / st),
                        actual_wz_rad_s=float(stats["wz"][i] / st),
                        xy_rmse_m_s=float((stats["xy_sq"][i] / st).sqrt()),
                        yaw_rmse_rad_s=float((stats["yaw_sq"][i] / st).sqrt()),
                        fall=bool(falls[i]),
                        survival_s=st * 0.02,
                        displacement_world_m=disp,
                        angular_displacement_rad=float(stats["yaw_displacement"][i]),
                        relative_height_m=float(stats["height"][i] / st),
                        saturation=float(stats["sat"][i] / st),
                        reward_sum=float(stats["reward"][i]),
                        geometric_slip_m_s=float(stats["slip"][i] / st),
                        expected_forward_progress_m=float(stats["expected"][i]),
                        max_lateral_m=float(stats["max_lateral"][i]),
                        outside_tile_fraction=float(stats["outside_tile"][i] / st),
                        central_platform_reached=bool(platform_reached[i])
                        if domain == "terrain"
                        else False,
                        max_terrain_change_m=float(max_terrain_change[i]),
                        central_platform_height_change_m=float(
                            (target_ground[i] - entry_ground[i]).abs()
                        ),
                    )
                    rec["crossing_eligible"] = domain == "terrain" and g == (0.4, 0, 0)
                    rec["crossing_passed"] = (
                        rec["crossing_eligible"]
                        and not rec["fall"]
                        and disp[0] >= 6.8
                        and rec["max_lateral_m"] <= 0.6
                    )
                    records.append(rec)
            print(json.dumps(dict(domain=domain, episodes=len(scenarios) * len(seeds))), flush=True)
        finally:
            env.close()
            env.backend.close()
    groups = {}
    for r in records:
        groups.setdefault(r["scenario"], []).append(r)
    aggregate = {
        name: {
            key: sum(r[key] for r in rs) / len(rs)
            for key in (
                "actual_vx_m_s",
                "actual_vy_m_s",
                "actual_wz_rad_s",
                "xy_rmse_m_s",
                "yaw_rmse_rad_s",
                "fall",
                "survival_s",
                "saturation",
                "crossing_passed",
                "geometric_slip_m_s",
                "outside_tile_fraction",
                "central_platform_reached",
                "max_terrain_change_m",
            )
        }
        for name, rs in groups.items()
    }
    result = dict(
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.file_digest(open(checkpoint, "rb"), "sha256").hexdigest(),
        seeds=list(seeds),
        seconds=seconds,
        maneuver_seconds=min(seconds, 4.0) if stairs_omni else None,
        actor_device=actor_device,
        safe_spawn=terrain_max_level == 7,
        platform_metric="center x within [3.0,3.8]m of entry, lateral <=.6m, collision height within 2.5cm of center, alive; distinct from full crossing",
        initial_perturbation="XY +/-2cm, yaw +/-0.03rad, joints +/-0.01rad",
        records=records,
        aggregate=aggregate,
    )
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--stairs-omni", action="store_true")
    p.add_argument("--max-level", type=int, choices=(4, 7), default=4)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[2426, 2427, 2428, 2429, 2430])
    a = p.parse_args()
    evaluate(a.checkpoint, a.output, a.seeds, a.seconds, a.quick, a.stairs_omni, a.max_level)


if __name__ == "__main__":
    main()
