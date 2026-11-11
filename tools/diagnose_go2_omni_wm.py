"""Independent multi-command heldout trajectories and calibrated action FD probes."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.contract import JOINTS
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_go2_omnidirectional import FLAT, TYPES
from evaluate_go2_terrain_b import physical_fd


def collect(checkpoint, out, stairs_speed=None, terrain_max_level=4):
    out.mkdir(parents=True, exist_ok=False)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    pi = Policy(
        42,
        3,
        hist=5,
        num_joints=12,
        goal_mode="direct",
        hidden=tuple(source["args"]["policy_hidden"]),
    )
    pi.load_state_dict(source["pi"])
    pi.eval()
    cfg = load_config(terrain=True)
    cmdcfg = cfg.commands["velocity"]
    cmdcfg.terrain_max_level = terrain_max_level
    if terrain_max_level == 7:
        cmdcfg.terrain_coverage_weights = (0.2, 0.4, 0.3, 0.1, 0, 0, 0, 0)
        cmdcfg.sampler.stairs_safe_spawn = True
    cmdcfg.terrain_profile = "mixed"
    cmdcfg.terrain_phase = "fixed"
    cmdcfg.fixed_command = (0.4, 0, 0)
    cmdcfg.resampling_time_range = (11, 11)
    cmdcfg.reset_jitter = (0.02, 0.03, 0.01)
    cfg.seed = 3526
    scenarios = [(name, 0, 0, g) for name, g in FLAT] + [
        (f"{TYPES[t]}_L{level}", t, level, (0.4, 0, 0)) for level in (1, 2, 4) for t in range(5)
    ]
    if stairs_speed is not None:
        from evaluate_go2_omnidirectional import stairs_speed_commands

        scenarios = [
            (f"{TYPES[t]}_L{level}_S{stairs_speed}_{i}", t, level, g)
            for t in (3, 4)
            for level in range(1, terrain_max_level + 1)
            for i, g in enumerate(stairs_speed_commands(stairs_speed))
        ]
        cmdcfg.spawn_offset = (1.6, 0, 0)
    env = make_go2_env(cfg, len(scenarios))
    env.init_state()
    c = env.command_manager.get_term("velocity")
    model = env.get_playback_model(0)
    mujoco.mj_saveModel(model, str(out / "terrain.mjb"))
    layout = env.backend.get_root_state_layout("robot/base")
    layout = dict(
        qi=list(map(int, env.backend.get_joint_state_qpos_indices(JOINTS))),
        vi=list(map(int, env.backend.get_joint_state_qvel_indices(JOINTS))),
        rq=list(map(int, layout.qpos_indices)),
        rv=list(map(int, layout.qvel_indices)),
    )
    rows = []
    probes = []
    history = torch.zeros(len(scenarios), 5, 42)
    try:
        c.curriculum.terrain_type[:] = torch.tensor([s[1] for s in scenarios])
        c.curriculum.terrain_level[:] = torch.tensor([s[2] for s in scenarios])
        env.reset(seed=3526)
        c.g[:] = torch.tensor([s[3] for s in scenarios])
        for t in range(300):
            if stairs_speed is not None:
                c.g[:] = torch.tensor([s[3] for s in scenarios])
            info = env.current_info()
            state = unflatten_state(info["s"])
            obs = c.mdp.obs(state, info["a_prev"], info["xi"], info["f"])
            history = torch.where(
                (info["ep_step"] == 0)[:, None, None],
                obs[:, None].expand(-1, 5, -1),
                torch.cat((history[:, 1:], obs[:, None]), 1),
            )
            with torch.no_grad():
                a = pi(history, info["g"])
            if t == 80:
                snap = env.get_physics_state_layout().split_state(env.get_physics_state_snapshot())
                for i in (
                    list(range(len(scenarios)))
                    if stairs_speed is not None
                    else [2, 5, 6, 7, 8, 9, *range(len(FLAT), len(scenarios))]
                ):
                    probes.append(
                        dict(
                            s=info["s"][i].clone(),
                            xi=info["xi"][i].clone(),
                            a=a[i].clamp(-0.8, 0.8).clone(),
                            qpos=torch.tensor(snap.qpos[i].copy()),
                            qvel=torch.tensor(snap.qvel[i].copy()),
                            time=float(snap.time[i]),
                            index=i,
                            scenario=scenarios[i][0],
                        )
                    )
            tr = env.step(a).info["supermimic"]
            rows.append({k: tr[k].cpu().clone() for k in ("s", "s1", "a", "xi", "done", "g")})
        data = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        torch.save(
            dict(
                data=data,
                probes=probes,
                field=asdict(c.mdp.terrain),
                layout=layout,
                scenarios=scenarios,
                contract=env.supermimic_contract,
            ),
            out / "heldout.pt",
        )
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    source=str(checkpoint),
                    seed=3526,
                    transitions=300 * len(scenarios),
                    commands=[s[3] for s in scenarios],
                    training_replay=False,
                    exploration=0,
                ),
                indent=2,
            )
        )
    finally:
        env.close()
        env.backend.close()


def evaluate(checkpoint, dataset, out):
    out.mkdir(parents=True, exist_ok=False)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    held = torch.load(dataset / "heldout.pt", weights_only=True, map_location="cpu")
    device = "cuda"
    for key in (
        "actor",
        "joints",
        "context",
        "action",
        "sim_dt",
        "ctrl_dt",
        "state",
        "model_sha256",
        "terrain_fingerprint",
    ):
        if source["task_contract"].get(key) != held["contract"].get(key):
            raise ValueError(f"Holdout mismatch {key}")
    wm = WorldModel(
        hidden=tuple(source["args"]["wm_hidden"]), num_joints=12, context_dim=16, terrain_dim=77
    ).to(device)
    wm.load_state_dict(source["wm"])
    field = TerrainHeightField(**held["field"]).to(device)
    mdp = Go2VelocityMdp(device, field)

    def step(s, a, xi):
        return wm.step(s, a, mdp.normalize_context(xi), field.local_scan(s[0], s[1]))

    d = held["data"]
    indices = [
        (e, t)
        for e in range(len(held["scenarios"]))
        for t in (30, 80, 140, 210)
        if not d["done"][t : t + 32, e].any()
    ]
    batch = {
        k: torch.stack([d[k][t : t + 32, e] for e, t in indices]).to(device)
        for k in ("s", "s1", "a", "xi")
    }
    state = unflatten_state(batch["s"][:, 0])
    records = []
    with torch.no_grad():
        for t in range(32):
            state = step(state, batch["a"][:, t], batch["xi"][:, t])
            if t + 1 not in (1, 8, 16, 32):
                continue
            target = unflatten_state(batch["s1"][:, t])
            error = wm.state_error(state, target)
            feet, _ = go2_feet_world(mdp.source_state(state))
            truefeet, _ = go2_feet_world(mdp.source_state(target))
            for e, (name, *_) in enumerate(held["scenarios"]):
                mask = torch.tensor([x[0] == e for x in indices], device=device)
                if not mask.any():
                    continue
                row = dict(scenario=name, horizon=t + 1, samples=int(mask.sum()))
                for key, value in dict(
                    position_m=error[:, :3],
                    attitude_rad=error[:, 3:6],
                    velocity_m_s=error[:, 6:9],
                    joint_position_rad=error[:, 12:24],
                    joint_velocity_rad_s=error[:, 24:36],
                    foot_position_m=feet - truefeet,
                ).items():
                    row[key] = float(value[mask].square().mean().sqrt())
                records.append(row)
    model = mujoco.MjModel.from_binary_path(str(dataset / "terrain.mjb"))
    gradients = []
    calibration = []
    for probe in held["probes"]:
        i = probe["index"]
        s = unflatten_state(probe["s"][None].to(device))
        xi = probe["xi"][None].to(device)
        anchor = probe["a"].to(device)
        # Verify the public snapshot oracle against the actual recorded action.
        real, _ = physical_fd(model, held["layout"], probe, d["a"][80, i].numpy())
        truth = unflatten_state(d["s1"][80, i][None])
        truth = torch.cat((truth[0], truth[2], truth[4], truth[5]), -1)[0].numpy()
        calibration.append(float(np.abs(real - truth).max()))

        def output(a):
            nxt = step(s, a[None], xi)
            return torch.cat((nxt[0], nxt[2], nxt[4], nxt[5]), -1)[0]

        jac = torch.autograd.functional.jacobian(output, anchor).detach().cpu().numpy()
        for eps in (0.01, 0.03, 0.07):
            fd = []
            contacts = 0
            for axis in range(12):
                delta = np.zeros(12)
                delta[axis] = eps
                plus, nc = physical_fd(model, held["layout"], probe, probe["a"].numpy() + delta)
                minus, _ = physical_fd(model, held["layout"], probe, probe["a"].numpy() - delta)
                fd.append((plus - minus) / (2 * eps))
                contacts = nc
            fd = np.stack(fd, -1)
            for name, sl in (("base_velocity", slice(3, 6)), ("joint_velocity", slice(18, 30))):
                a, b = jac[sl], fd[sl]
                valid = np.abs(b) > 1e-3
                gradients.append(
                    dict(
                        scenario=probe["scenario"],
                        epsilon=eps,
                        quantity=name,
                        contact_count=contacts,
                        nonzero_components=int(valid.sum()),
                        sign_agreement=float(np.mean((a[valid] * b[valid]) > 0))
                        if valid.any()
                        else None,
                        wm_norm=float(np.linalg.norm(a)),
                        physics_norm=float(np.linalg.norm(b)),
                        normalized_error=float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8)),
                    )
                )
    if max(calibration) > 1e-3:
        raise ValueError(f"Uncalibrated physical oracle: {max(calibration)}")
    (out / "diagnostics.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(checkpoint),
                prediction=records,
                gradients=gradients,
                oracle_max_error=max(calibration),
                same_h32_valid_windows_for_all_horizons=True,
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["collect", "evaluate"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--stairs-speed", type=int, choices=range(4))
    p.add_argument("--max-level", type=int, choices=(4, 7), default=4)
    a = p.parse_args()
    if a.mode == "collect":
        collect(a.checkpoint, a.output, a.stairs_speed, a.max_level)
    else:
        evaluate(a.checkpoint, a.dataset, a.output)


if __name__ == "__main__":
    main()
