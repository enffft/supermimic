"""Real MuJoCo solver contact/foot trajectories from the public UniLab compiled model.

Detached native MjData own the complete integration/contact state. No private
backend arrays, contact proxies presented as forces, or WM is used for physics.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    flatten_state,
    matvec,
    quat_to_mat,
    unflatten_state,
)
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.contract import DEFAULT_Q, JOINTS, soft_limits
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp

LEGS = ("FL", "FR", "RL", "RR")


def state_tuple(source):
    R = quat_to_mat(source[..., 3:7])
    return (
        source[..., :3],
        R,
        source[..., 19:22],
        matvec(R, source[..., 22:25]),
        source[..., 7:19],
        source[..., 25:37],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[2426, 2427, 2428, 2429, 2430])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    pi = (
        Policy(
            42,
            3,
            hist=5,
            num_joints=12,
            goal_mode="direct",
            hidden=tuple(saved["args"]["policy_hidden"]),
        )
        .cuda()
        .eval()
    )
    pi.load_state_dict(saved["pi"])
    cfg = load_config(terrain=True)
    c = cfg.commands["velocity"]
    c.terrain_max_level = 7
    c.terrain_coverage_weights = (0.2, 0.4, 0.3, 0.1, 0, 0, 0, 0)
    c.sampler.stairs_safe_spawn = True
    c.terrain_profile = "mixed"
    c.terrain_curriculum = False
    c.terrain_phase = "fixed"
    c.fixed_command = (0.4, 0, 0)
    c.reset_jitter = (0.02, 0.03, 0.01)
    c.resampling_time_range = (21, 21)
    cfg.max_episode_seconds = a.seconds
    cases = [("flat", 0, 0)] + [
        (f"stairs_{d}_L{l}", t, l) for d, t in [("up", 3), ("down", 4)] for l in (4, 5, 6, 7)
    ]
    env = make_go2_env(cfg, len(cases))
    env.init_state()
    cmd = env.command_manager.get_term("velocity")
    field = cmd.mdp.terrain
    mdp = cmd.mdp
    for key in ("actor", "joints", "context", "action", "sim_dt", "ctrl_dt", "terrain_fingerprint"):
        assert saved["task_contract"][key] == env.supermimic_contract[key], key
    mujoco.mj_saveModel(env.get_playback_model(0), str(a.output / "terrain.mjb"))
    model = mujoco.MjModel.from_binary_path(str(a.output / "terrain.mjb"))
    qi = np.asarray(env.backend.get_joint_state_qpos_indices(JOINTS))
    vi = np.asarray(env.backend.get_joint_state_qvel_indices(JOINTS))
    layout = env.backend.get_root_state_layout("robot/base")
    rq = np.asarray(layout.qpos_indices)
    rv = np.asarray(layout.qvel_indices)
    act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS])
    footids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{l}_foot_collision")
            for l in LEGS
        ]
    )
    terrainid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "go2_terrain")
    limits = soft_limits().numpy()
    assert np.all(model.geom_size[footids, 0] == 0.022)
    alltraces = []
    summaries = []
    peaks = []
    crossings = []
    calibration = []
    fk_errors = []
    velocity_errors = []
    contact_events = 0
    origins = cmd.curriculum.origins
    with gzip.open(a.output / "contacts.jsonl.gz", "wt") as log:
        for seed in a.seeds:
            cmd.curriculum.terrain_type[:] = torch.tensor([x[1] for x in cases])
            cmd.curriculum.terrain_level[:] = torch.tensor([x[2] for x in cases])
            env.reset(seed=seed)
            snap = env.get_physics_state_layout().split_state(env.get_physics_state_snapshot())
            n = len(cases)
            ds = [mujoco.MjData(model) for _ in cases]
            scratch = mujoco.MjData(model)
            for i, d in enumerate(ds):
                d.qpos[:] = snap.qpos[i]
                d.qvel[:] = snap.qvel[i]
                d.time = float(snap.time[i])
                mujoco.mj_forward(model, d)

            def sources():
                return torch.tensor(
                    np.stack(
                        [
                            np.r_[
                                d.qpos[rq[:7]],
                                d.qpos[qi],
                                d.qvel[rv[:3]],
                                d.qvel[rv[3:6]],
                                d.qvel[vi],
                            ]
                            for d in ds
                        ]
                    ),
                    dtype=torch.float32,
                )

            s = sources()
            start = s[:, :3].clone()
            prev_a = torch.zeros(n, 12)
            goal = torch.tensor([0.4, 0, 0]).expand(n, -1)
            xi = cmd.xi.clone()
            history = None
            alive = torch.ones(n, dtype=torch.bool)
            rows = []
            swing_peak = np.zeros((n, 4))
            air_start = np.zeros((n, 4))
            was_air = np.zeros((n, 4), bool)
            entries = torch.stack([origins[l, t, :2] for _, t, l in cases])
            axis = torch.linspace(-0.4, 7.3, 1541)
            sections = field.ground_height(
                entries[:, None, :] + torch.stack((axis, torch.zeros_like(axis)), -1)
            )
            edge_lists = []
            for i in range(n):
                slopes = np.gradient(sections[i].numpy(), axis.numpy())
                idx = np.flatnonzero(np.abs(slopes) > 1.0)
                groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
                edge_lists.append(
                    [
                        (
                            float(axis[g[len(g) // 2]]),
                            float(sections[i, g[0]]),
                            float(sections[i, min(g[-1] + 1, len(axis) - 1)]),
                        )
                        for g in groups
                        if len(g)
                    ]
                )
            for step in range(round(a.seconds / 0.02)):
                obs = mdp.obs(state_tuple(s), prev_a, xi, torch.zeros(n, dtype=torch.long))
                history = (
                    obs[:, None].expand(-1, 5, -1).clone()
                    if history is None
                    else torch.cat((history[:, 1:], obs[:, None]), 1)
                )
                with torch.no_grad():
                    action = pi(history.cuda(), goal.cuda()).cpu()
                true_contact = np.zeros((n, 4), bool)
                riser = np.zeros((n, 4), bool)
                tread = np.zeros((n, 4), bool)
                forces = np.zeros((n, 4, 3))
                body = np.zeros(n, bool)
                torque = np.zeros((n, 12))
                for i, d in enumerate(ds):
                    if not alive[i]:
                        continue
                    d.ctrl[act] = np.clip(
                        np.array(DEFAULT_Q) + 0.25 * action[i].numpy(), limits[:, 0], limits[:, 1]
                    )
                    for sub in range(4):
                        mujoco.mj_step(model, d)
                        for k, ct in enumerate(d.contact):
                            if terrainid not in (ct.geom1, ct.geom2):
                                continue
                            other = int(ct.geom2 if ct.geom1 == terrainid else ct.geom1)
                            leg = np.flatnonzero(footids == other)
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, d, k, force)
                            world = ct.frame.reshape(3, 3).T @ force[:3]
                            j = int(leg[0]) if len(leg) else -1
                            if np.linalg.norm(force[:3]) > 2:
                                if j >= 0:
                                    true_contact[i, j] = True
                                    near_edge = any(
                                        abs(float(ct.pos[0] - entries[i, 0]) - e[0]) < 0.06
                                        and abs(e[2] - e[1]) > 0.03
                                        for e in edge_lists[i]
                                    )
                                    riser[i, j] |= abs(ct.frame[2]) < 0.7 and near_edge
                                    tread[i, j] |= abs(ct.frame[2]) >= 0.7
                                    forces[i, j] += world / 4
                                else:
                                    name = (
                                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                                        or ""
                                    )
                                    body[i] |= any(x in name for x in ("base", "head"))
                            log.write(
                                json.dumps(
                                    dict(
                                        seed=seed,
                                        scenario=cases[i][0],
                                        step=step,
                                        substep=sub,
                                        geom1=int(ct.geom1),
                                        geom2=int(ct.geom2),
                                        foot=j,
                                        position=ct.pos.tolist(),
                                        normal=ct.frame[:3].tolist(),
                                        force_contact=force.tolist(),
                                        force_world=world.tolist(),
                                        distance=float(ct.dist),
                                    )
                                )
                                + "\n"
                            )
                            contact_events += 1
                    torque[i] = d.actuator_force[act]
                nxt = sources()
                if step == 0:
                    tr = env.step(action).info["supermimic"]
                    expected = mdp.source_state(unflatten_state(tr["s1"]))
                    calibration.append(float((expected - nxt).abs().max()))
                feet, vel = go2_feet_world(nxt)
                ground = field.ground_height(feet[..., :2])
                clearance = feet[..., 2] - ground - 0.022
                local = feet.clone()
                local[..., :2] -= entries[:, None, :]
                # Kinematics on a separate MjData must not disturb the real solver warmstart.
                for i, d in enumerate(ds):
                    if not alive[i]:
                        continue
                    scratch.qpos[:] = d.qpos
                    scratch.qvel[:] = d.qvel
                    mujoco.mj_forward(model, scratch)
                    fk_errors.append(
                        float(np.abs(scratch.geom_xpos[footids] - feet[i].numpy()).max())
                    )
                    for j, gid in enumerate(footids):
                        v = np.zeros(6)
                        mujoco.mj_objectVelocity(
                            model, scratch, mujoco.mjtObj.mjOBJ_GEOM, int(gid), v, 0
                        )
                        velocity_errors.append(float(np.abs(v[3:] - vel[i, j].numpy()).max()))
                lv = matvec(state_tuple(nxt)[1].transpose(-1, -2), nxt[:, 19:22])
                pitch = torch.asin((-state_tuple(nxt)[1][:, 2, 0]).clamp(-1, 1))
                term = mdp.hard_termination(state_tuple(nxt))
                next_distance = np.full((n, 4), np.nan)
                for i in range(n):
                    if not alive[i]:
                        continue
                    for j in range(4):
                        x = float(local[i, j, 0])
                        oldx = float((go2_feet_world(s)[0][i, j, 0] - entries[i, 0]))
                        edges = edge_lists[i]
                        ahead = [e for e in edges if e[0] >= x]
                        next_distance[i, j] = ahead[0][0] - x if ahead else np.nan
                        for ex, z0, z1 in edges:
                            if oldx < ex <= x and step > 25:
                                crossings.append(
                                    dict(
                                        scenario=cases[i][0],
                                        seed=seed,
                                        foot=j,
                                        time=step * 0.02,
                                        edge_x=ex,
                                        bottom_z=float(feet[i, j, 2] - 0.022),
                                        tread_z=max(z0, z1),
                                        clear=bool(feet[i, j, 2] - 0.022 >= max(z0, z1) + 0.005),
                                        riser_contact=bool(riser[i, j]),
                                    )
                                )
                        air = not true_contact[i, j] and step > 25
                        if air and not was_air[i, j]:
                            air_start[i, j] = float(ground[i, j])
                            swing_peak[i, j] = 0
                        if air:
                            swing_peak[i, j] = max(
                                swing_peak[i, j], float(feet[i, j, 2] - 0.022 - air_start[i, j])
                            )
                        if was_air[i, j] and true_contact[i, j]:
                            peaks.append(
                                dict(
                                    scenario=cases[i][0],
                                    seed=seed,
                                    foot=j,
                                    peak_m=swing_peak[i, j],
                                    landing_vz_m_s=float(vel[i, j, 2]),
                                    landing_force_N=float(np.linalg.norm(forces[i, j])),
                                )
                            )
                        was_air[i, j] = air
                rows.append(
                    dict(
                        s=flatten_state(state_tuple(s)),
                        s1=flatten_state(state_tuple(nxt)),
                        a=action.clone(),
                        a_prev=prev_a.clone(),
                        xi=xi.clone(),
                        g=goal.clone(),
                        done=term | ~alive,
                        active=alive.clone(),
                        feet=feet,
                        foot_velocity=vel,
                        clearance=clearance,
                        contact=torch.tensor(true_contact),
                        force=torch.tensor(forces),
                        riser=torch.tensor(riser),
                        tread=torch.tensor(tread),
                        body_contact=torch.tensor(body),
                        torque=torch.tensor(torque),
                        next_edge_distance=torch.tensor(next_distance),
                        base_pitch=pitch,
                        body_velocity=lv,
                        joint_acceleration=(nxt[:, 25:] - s[:, 25:]) / 0.02,
                    )
                )
                prev_a = action
                s = nxt
                alive &= ~term
            data = {k: torch.stack([x[k] for x in rows]) for k in rows[0]}
            alltraces.append(
                dict(
                    seed=seed,
                    data=data,
                    entries=entries,
                    section_x=axis,
                    section_z=sections,
                    edges=edge_lists,
                )
            )
            for i, (name, t, l) in enumerate(cases):
                mask = data["active"][:, i]
                st = int(mask.sum())
                pos = unflatten_state(data["s1"][mask, i])[0]
                disp = pos[-1] - start[i]
                pk = [x["peak_m"] for x in peaks if x["scenario"] == name and x["seed"] == seed]
                cross = [x for x in crossings if x["scenario"] == name and x["seed"] == seed]
                fallen = bool(data["done"][mask, i].any())
                maxlat = float((pos[:, 1] - start[i, 1]).abs().max())
                summaries.append(
                    dict(
                        scenario=name,
                        seed=seed,
                        level=l,
                        fall=fallen,
                        survival_s=st * 0.02,
                        progress_m=float(disp[0]),
                        max_lateral_m=maxlat,
                        crossing=bool(not fallen and disp[0] >= 6.8 and maxlat <= 0.6),
                        xy_rmse_m_s=float(
                            (data["body_velocity"][mask, i, :2] - goal[i, :2])
                            .square()
                            .sum(-1)
                            .mean()
                            .sqrt()
                        ),
                        stalled_s=float(
                            ((data["body_velocity"][mask, i, 0] < 0.05).float().sum() * 0.02)
                        ),
                        peak_mean_m=float(np.mean(pk)) if pk else None,
                        peak_p50_m=float(np.median(pk)) if pk else None,
                        peak_p90_m=float(np.quantile(pk, 0.9)) if pk else None,
                        foot_riser_contact_steps=int(data["riser"][mask, i].sum()),
                        body_terrain_contact_steps=int(data["body_contact"][mask, i].sum()),
                        edge_crossings=len(cross),
                        clear_edge_crossings=sum(x["clear"] for x in cross),
                        saturation=float((data["a"][mask, i].abs() > 0.98).float().mean()),
                        support_fraction=float(data["contact"][mask, i].float().mean()),
                        all_four_air_fraction=float(
                            (~data["contact"][mask, i].any(-1)).float().mean()
                        ),
                        joint_acceleration_rms=float(
                            data["joint_acceleration"][mask, i].square().mean().sqrt()
                        ),
                        base_vz_rms=float(data["s1"][mask, i, 9].square().mean().sqrt()),
                    )
                )
            print(json.dumps(dict(seed=seed, completed_cases=len(cases))), flush=True)
    env.close()
    env.backend.close()
    torch.save(
        dict(
            traces=alltraces,
            cases=cases,
            field=asdict(field),
            contract=saved["task_contract"],
            layout=dict(qi=qi.tolist(), vi=vi.tolist(), rq=rq.tolist(), rv=rv.tolist()),
        ),
        a.output / "trajectories.pt",
    )
    result = dict(
        checkpoint=str(a.checkpoint),
        sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
        seeds=a.seeds,
        command=[0.4, 0, 0],
        dt=0.02,
        force_dt=0.005,
        physics="actual native mj_step on detached public compiled model; complete integration state owned by this diagnostic",
        unilab_one_step_max_error=max(calibration),
        foot_fk_position_max_error_m=max(fk_errors),
        foot_fk_velocity_max_error_m_s=max(velocity_errors),
        foot_radius_m=0.022,
        contact_events=contact_events,
        summaries=summaries,
        swing_peaks=peaks,
        edge_crossings=crossings,
        geom_names=[
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
        ],
    )
    (a.output / "diagnostics.json").write_text(json.dumps(result, indent=2))
    assert max(calibration) < 0.001, calibration
    assert max(fk_errors) < 1e-4, max(fk_errors)
    assert max(velocity_errors) < 0.001, max(velocity_errors)


if __name__ == "__main__":
    main()
