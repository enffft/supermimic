"""Independent fixed trajectories and MuJoCo action-Jacobian oracle for B.

Never trains a policy or world model. Save held-out physics once and reuse the
exact same H32-valid starting states for every checkpoint and every horizon.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2.config import load_config
from unisupermimic.tasks.velocity.go2.contract import DEFAULT_Q, JOINTS, soft_limits
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.feet import go2_feet_world
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_go2_migration import dump

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down")


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(terrain=True)
    c = cfg.commands["velocity"]
    c.terrain_profile, c.terrain_curriculum, c.terrain_phase = "mixed", False, "fixed"
    c.fixed_command, c.standing_fraction = (0.4, 0.0, 0.0), 0.0
    c.resampling_time_range = (10.0, 10.0)
    cfg.seed = 2326  # held-out collection RNG; geometry seed remains audited 2026
    env = make_go2_env(cfg, 25)
    env.init_state()
    manifest = json.loads(
        (ROOT / "migration/source_reference/checkpoint_manifest.json").read_text()
    )
    lc = SuperMimicConfig(
        goal_mode="direct",
        goal_frames=1,
        hist=5,
        terrain_training=True,
        terrain_conditioning=True,
        compile=False,
        amp="none",
        no_sync=False,
        pi_opt="adam",
        wm_opt="adam",
        buffer_steps=512,
        init_actor_from=manifest["path"],
        action_noise=0.05,
        seed=2326,
    )
    l = SuperMimicLearner(env, lc, out / "collector", device="cuda", backend="mujoco")
    cmd = env.command_manager.get_term("velocity")
    cmd.curriculum.terrain_level[:] = torch.arange(25) // 5
    cmd.curriculum.mastery.copy_(cmd.curriculum.terrain_level)
    model = env.get_playback_model(0)
    mujoco.mj_saveModel(model, str(out / "terrain.mjb"))
    layout = env.backend.get_root_state_layout("robot/base")
    metadata = dict(
        qi=[int(x) for x in env.backend.get_joint_state_qpos_indices(JOINTS)],
        vi=[int(x) for x in env.backend.get_joint_state_qvel_indices(JOINTS)],
        rq=[int(x) for x in layout.qpos_indices],
        rv=[int(x) for x in layout.qvel_indices],
    )
    chunks, probes = [], []
    try:
        for offset, steps in ((0.0, 500), (0.8, 160), (1.3, 160), (2.0, 160)):
            cmd.cfg.spawn_offset = (offset, 0.0, 0.0)
            env.reset(seed=2326 + int(offset * 100))
            l.buffer.ptr = l.buffer.size = 0
            l.obs_hist.zero_()
            for t in range(steps):
                l.collect(1)
                if t == 39:
                    snapshot = env.get_physics_state_layout().split_state(
                        env.get_physics_state_snapshot()
                    )
                    info = l._state_info()
                    s = unflatten_state(info["s"])
                    obs = l.mdp.obs(s, info["a_prev"], info["xi"], info["f"])
                    hist = torch.cat((l.obs_hist[:, 1:], obs[:, None]), 1)
                    with torch.no_grad():
                        anchor = l.pi(hist, info["g"]).clamp(-0.8, 0.8)
                    for i in [0, *range(5, 15)]:  # flat + every type level 1 and 2
                        probes.append(
                            dict(
                                s=info["s"][i].cpu(),
                                xi=info["xi"][i].cpu(),
                                a=anchor[i].cpu(),
                                qpos=torch.tensor(snapshot.qpos[i].copy()),
                                qvel=torch.tensor(snapshot.qvel[i].copy()),
                                time=float(snapshot.time[i]),
                                terrain_type=i % 5,
                                level=i // 5,
                                offset=offset,
                            )
                        )
            chunk = {k: v[:steps].detach().cpu().clone() for k, v in l.buffer.data.items()}
            chunk["terrain_type"] = (torch.arange(25) % 5).expand(steps, 25).clone()
            chunk["terrain_level"] = (torch.arange(25) // 5).expand(steps, 25).clone()
            chunk["entry_offset_m"] = offset
            chunks.append(chunk)
        torch.save(
            dict(
                chunks=chunks,
                probes=probes,
                layout=metadata,
                field=asdict(env.supermimic_mdp.terrain),
                seed=2326,
                goal=(0.4, 0.0, 0.0),
                exploration=0.05,
                contract=env.supermimic_contract,
            ),
            out / "heldout.pt",
        )
        dump(
            out / "manifest.json",
            dict(
                real_transitions=sum(len(x["s"]) * 25 for x in chunks),
                probe_states=len(probes),
                seed=2326,
                geometry_seed=2026,
                command=[0.4, 0, 0],
                exploration=0.05,
                starts="entry and controlled drop/contact probes; probe traversals are NOT successes",
                data_not_added_to_training_replay=True,
                types=TYPES,
            ),
        )
    finally:
        env.close()
        env.backend.close()


def matched_windows(data, device):
    windows = []
    for chunk in data["chunks"]:
        # Deliberately fixed starts, no checkpoint-dependent sample selection.
        starts = [20, 50, 85, 120] if len(chunk["s"]) < 200 else [30, 140, 260, 400]
        for e in range(25):
            for t in starts:
                windows.append(
                    {k: chunk[k][t : t + 32, e] for k in ("s", "s1", "a", "xi", "done")}
                    | dict(kind=e % 5, level=e // 5)
                )
    batch = {
        k: torch.stack([w[k] for w in windows]).to(device) for k in ("s", "s1", "a", "xi", "done")
    }
    batch["kind"] = torch.tensor([w["kind"] for w in windows], device=device)
    batch["level"] = torch.tensor([w["level"] for w in windows], device=device)
    batch["valid"] = ~batch["done"][:, :31].any(-1)
    return batch


def physical_fd(model, layout, probe, action):
    d = mujoco.MjData(model)
    d.qpos[:] = probe["qpos"].numpy()
    d.qvel[:] = probe["qvel"].numpy()
    d.time = probe["time"]
    mujoco.mj_forward(model, d)
    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "go2_terrain")
    contacts = sum(terrain_id in (c.geom1, c.geom2) for c in d.contact)
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS]
    limits = soft_limits().numpy()
    d.ctrl[act_ids] = np.clip(
        np.array(DEFAULT_Q) + 0.25 * np.clip(action, -1, 1), limits[:, 0], limits[:, 1]
    )
    for _ in range(4):
        mujoco.mj_step(model, d)
    return np.r_[
        d.qpos[np.array(layout["rq"][:3])],
        d.qvel[np.array(layout["rv"][:3])],
        d.qpos[np.array(layout["qi"])],
        d.qvel[np.array(layout["vi"])],
    ], contacts


def evaluate(dataset, checkpoint, out):
    out.mkdir(parents=True, exist_ok=False)
    data = torch.load(dataset / "heldout.pt", weights_only=True, map_location="cpu")
    values = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not values["args"].get("terrain_conditioning"):
        raise ValueError("This is experiment B only")
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
        if values["task_contract"].get(key) != data["contract"].get(key):
            raise ValueError(f"Holdout protocol mismatch: {key}")
    device = "cuda"
    wm = WorldModel(
        hidden=tuple(values["args"]["wm_hidden"]), num_joints=12, context_dim=16, terrain_dim=77
    ).to(device)
    wm.load_state_dict(values["wm"])
    field = TerrainHeightField(**data["field"]).to(device)
    mdp = Go2VelocityMdp(device, field)

    def step(state, action, xi):
        return wm.step(
            state, action, mdp.normalize_context(xi), field.local_scan(state[0], state[1])
        )

    batch = matched_windows(data, device)
    state = unflatten_state(batch["s"][:, 0])
    records = []
    with torch.no_grad():
        for t in range(32):
            state = step(state, batch["a"][:, t], batch["xi"][:, t])
            if t + 1 not in (1, 8, 16, 32):
                continue
            target = unflatten_state(batch["s1"][:, t])
            err = wm.state_error(state, target)
            feet, _ = go2_feet_world(mdp.source_state(state))
            true_feet, _ = go2_feet_world(mdp.source_state(target))
            for kind in range(5):
                for level in range(5):
                    mask = batch["valid"] & (batch["kind"] == kind) & (batch["level"] == level)

                    def rmse(x):
                        return float(x[mask].square().mean().sqrt()) if mask.any() else None

                    records.append(
                        dict(
                            type=TYPES[kind],
                            level=level,
                            horizon=t + 1,
                            samples=int(mask.sum()),
                            position_m=rmse(err[:, :3]),
                            attitude_rad=rmse(err[:, 3:6]),
                            velocity_m_s=rmse(err[:, 6:9]),
                            angular_velocity_rad_s=rmse(err[:, 9:12]),
                            joint_position_rad=rmse(err[:, 12:24]),
                            joint_velocity_rad_s=rmse(err[:, 24:36]),
                            foot_position_m=rmse(feet - true_feet),
                        )
                    )
    dump(out / "prediction.json", records)
    model = mujoco.MjModel.from_binary_path(str(dataset / "terrain.mjb"))
    calibration = []
    for probe in data["probes"]:
        chunk = next(c for c in data["chunks"] if c["entry_offset_m"] == probe["offset"])
        e = probe["level"] * 5 + probe["terrain_type"]
        actual_state = unflatten_state(chunk["s1"][40, e][None])
        actual = torch.cat(
            (actual_state[0], actual_state[2], actual_state[4], actual_state[5]), -1
        )[0].numpy()
        predicted, _ = physical_fd(model, data["layout"], probe, chunk["a"][40, e].numpy())
        error = np.abs(predicted - actual)
        calibration.append(
            dict(
                type=TYPES[probe["terrain_type"]],
                level=probe["level"],
                offset=probe["offset"],
                position_max_m=float(error[:3].max()),
                velocity_max_m_s=float(error[3:6].max()),
                q_max_rad=float(error[6:18].max()),
                qd_max_rad_s=float(error[18:].max()),
            )
        )
    dump(out / "oracle_calibration.json", calibration)
    if max(r["qd_max_rad_s"] for r in calibration) > 1e-3:
        raise ValueError(
            "Independent oracle does not reproduce recorded real physics within 1e-3 rad/s"
        )
    gradients = []
    for probe in data["probes"]:
        s = unflatten_state(probe["s"][None].to(device))
        xi = probe["xi"][None].to(device)
        anchor = probe["a"].to(device)

        def output(action):
            nxt = step(s, action[None], xi)
            return torch.cat((nxt[0], nxt[2], nxt[4], nxt[5]), -1)[0]

        jac = (
            torch.autograd.functional.jacobian(output, anchor, vectorize=True)
            .detach()
            .cpu()
            .numpy()
        )
        action = anchor.cpu().numpy()
        for eps in (0.01, 0.03, 0.1):
            fd = np.zeros_like(jac)
            contacts = 0
            for j in range(12):
                perturb = np.eye(12)[j] * eps
                plus, contacts = physical_fd(model, data["layout"], probe, action + perturb)
                minus, _ = physical_fd(model, data["layout"], probe, action - perturb)
                fd[:, j] = (plus - minus) / (2 * eps)
            groups = {}
            for name, sl in (
                ("position", slice(0, 3)),
                ("velocity", slice(3, 6)),
                ("joint_position", slice(6, 18)),
                ("joint_velocity", slice(18, 30)),
            ):
                actual, pred = fd[sl], jac[sl]
                active = (np.abs(actual) > 1e-5) & (np.abs(pred) > 1e-7)
                groups[name] = dict(
                    active_components=int(active.sum()),
                    sign_agreement=float((np.sign(actual[active]) == np.sign(pred[active])).mean())
                    if active.any()
                    else None,
                    normalized_error=float(
                        np.linalg.norm(pred - actual) / (np.linalg.norm(actual) + 1e-8)
                    ),
                    physical_norm=float(np.linalg.norm(actual)),
                    wm_norm=float(np.linalg.norm(pred)),
                )
            gradients.append(
                dict(
                    type=TYPES[probe["terrain_type"]],
                    level=probe["level"],
                    offset_m=probe["offset"],
                    epsilon=eps,
                    actual_ground_contacts=contacts,
                    groups=groups,
                )
            )
    dump(out / "gradients.json", gradients)
    # Gate defined in physical units; each type must pass, not just aggregate loss.
    selected = [r for r in records if r["level"] in (1, 2) and r["horizon"] == 32]
    gs = [
        r["groups"]["joint_velocity"]
        for r in gradients
        if r["level"] in (1, 2) and r["epsilon"] == 0.03
    ]
    prediction_ok = all(
        r["samples"] >= 8
        and r["joint_position_rad"] < 0.12
        and r["velocity_m_s"] < 0.25
        and r["position_m"] < 0.15
        and r["attitude_rad"] < 0.15
        for r in selected
    )
    gradient_ok = all(
        g["sign_agreement"] is not None
        and g["sign_agreement"] >= 0.8
        and g["normalized_error"] < 0.6
        for g in gs
    )
    dump(
        out / "acceptance.json",
        dict(
            iteration=values["iteration"],
            update_counts=values.get("update_counts"),
            real_transitions=values["sim_steps"],
            prediction_pass=prediction_ok,
            physical_gradient_pass=gradient_ok,
            actor_unfreeze_allowed=prediction_ok and gradient_ok,
            minimum_probe_sign_agreement=min(g["sign_agreement"] or 0 for g in gs),
            mean_probe_sign_agreement=float(np.mean([g["sign_agreement"] or 0 for g in gs])),
            mean_probe_normalized_error=float(np.mean([g["normalized_error"] for g in gs])),
            thresholds="level1/2 H32 n>=8,q<.12rad,v<.25m/s,p<.15m,R<.15rad; each contact probe qd Jacobian agreement>=.8,error<.6",
            limitations="Contact discontinuities and FD epsilon dependence reported separately; passing signs alone is insufficient",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("collect", "evaluate"), required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "collect":
        build(a.output)
    else:
        evaluate(a.dataset, a.checkpoint, a.output)


if __name__ == "__main__":
    main()
