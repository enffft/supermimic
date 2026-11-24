"""Measure reference control on real batched MuJoCo steps."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from tools.train_go2w_motion_prior import owner_config
from unisupermimic.rl.algos.supermimic.geometry import matvec, quat_to_mat
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import JOINTS
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import ROOT, write_json
from unisupermimic.tasks.velocity.go2w.motion_prior import MotionBank, reference_action

COMMANDS = (
    [(0, 0, 0)]
    + [(v, 0, 0) for v in (0.10, 0.20, -0.10, -0.20)]
    + [(0, v, 0) for v in (0.05, -0.05, 0.10, -0.10, 0.15, -0.15)]
    + [(0, 0, v) for v in (0.20, -0.20, 0.40, -0.40)]
    + [(0.10, 0.05, 0), (0.10, -0.05, 0), (0.10, 0, 0.20), (0.10, 0, -0.20)]
)
MODES = ("standing_rollout", "rsi_rollout", "local_8", "local_16", "local_32")


@torch.inference_mode()
def audit(output):
    if output.exists() or output.with_suffix(".pt").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = OmegaConf.to_container(owner_config().motion_prior, resolve=True)
    bank = MotionBank(ROOT / cfg["bank"], "cpu", cfg)
    index = []
    for command in COMMANDS:
        found = (bank.nominal == torch.tensor(command, dtype=torch.float32)).all(-1).nonzero()
        if len(found) != 1:
            raise ValueError(f"Command does not identify a unique clip: {command}")
        index.append(int(found[0]))
    ids = torch.tensor(index).repeat(len(MODES))
    mode = torch.arange(len(MODES)).repeat_interleave(len(index))
    start = torch.where(mode == 0, 0, bank.reset_starts[ids, 12])
    end = torch.minimum(torch.tensor([300, 300, 8, 16, 32])[mode], 300 - start)
    task = load_config("motion_prior")
    env = make_go2w_env(task, len(ids))
    try:
        term = env.command_manager.get_term("velocity")
        term.curriculum.enabled = False
        env.reset()
        env.set_episode_length_buf(torch.zeros_like(ids))
        mixed = env.action_manager.get_term("mixed")
        mixed.collection_noise_std = 0.0
        origin = term._origins.clone()
        origin[:, 2] = term.mdp.terrain.ground_height(origin[:, :2])
        root, q, qd = bank.reset_state(ids, start, origin, torch.zeros_like(start).float())
        views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        qp, qv = views["qpos"].clone(), views["qvel"].clone()
        position_columns = env.backend.get_joint_state_qpos_indices(JOINTS)
        rsi = (mode != 0)[:, None]
        qp[:, term.root_q] = torch.where(rsi, root[:, :7], qp[:, term.root_q])
        qp[:, position_columns] = torch.where(rsi, q, qp[:, position_columns])
        qv[:, term.root_v] = torch.where(
            rsi,
            torch.cat(
                (
                    root[:, 7:10],
                    matvec(quat_to_mat(root[:, 3:7]).transpose(-1, -2), root[:, 10:13]),
                ),
                -1,
            ),
            qv[:, term.root_v],
        )
        qv[:, term.v_indices] = torch.where(rsi, qd, qv[:, term.v_indices])
        env.backend.set_state_tensor(torch.arange(len(ids)), qp.contiguous(), qv.contiguous())

        n = len(ids)
        live, fallen = torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        elapsed, plateau_steps, clips, saturation = [torch.zeros(n) for _ in range(4)]
        actual_sum, plateau_sum, tracking = [torch.zeros(n, 3) for _ in range(3)]
        leg_error, wheel_error, perturbation = [torch.zeros(n) for _ in range(3)]
        noise = torch.zeros(n, 16)
        rng = torch.Generator().manual_seed(8821)
        scale = noise.new_tensor(
            [cfg.get("local_leg_std", 0.01)] * 12 + [cfg.get("local_wheel_std", 0.03)] * 4
        )
        model = env.get_playback_model(0)
        data = mujoco.MjData(model)
        wheels = {
            model.geom(f"robot/{leg}_wheel_collision").id: i
            for i, leg in enumerate(("FL", "FR", "RL", "RR"))
        }
        previous = np.zeros((n, 4), dtype=bool)
        contact_changes = np.zeros(n, dtype=int)
        torque_limit = mixed._target.new_tensor(
            mixed.cfg.control.leg_effort_limits + mixed.cfg.control.wheel_effort_limits
        )
        shown = torch.tensor([0, 1, 7, 8, 11, 12, len(index) + 7, len(index) + 8])
        traces, active_trace = [], []
        for step in range(int(end.max())):
            frame = (start + step).clamp_max(299)
            goal = bank.data["command"][ids, frame]
            term.g[:, :3] = goal
            action, clipped = reference_action(
                bank.positions[ids, frame + 1],
                bank.velocity[ids, frame + 1, 12:],
                mixed.cfg.control,
                mixed.default,
                mixed.limits,
            )
            noise.mul_(cfg.get("local_correlation", 0.8)).add_(
                scale * torch.randn(n, 16, generator=rng)
            )
            proposed = action + noise * (mode >= 2)[:, None]
            executed = proposed.clamp(-1, 1)
            if step % 2 == 0:
                traces.append(
                    env.backend.get_state_views(("qpos",), device="cpu")["qpos"][shown].clone()
                )
                active_trace.append(live[shown].clone())
            result = env.step(executed)
            state = env.velocity_next_state
            rotation = state[1].transpose(-1, -2)
            velocity = torch.cat(
                (matvec(rotation, state[2])[:, :2], matvec(rotation, state[3])[:, 2:3]), -1
            )
            steady = live & ((goal - bank.nominal[ids]).abs().amax(-1) < 1e-5)
            steps_target = bank.positions[ids, frame + 1]
            leg_error += (state[4] - steps_target).square().mean(-1) * live
            wheel_error += (state[5][:, 12:] - bank.velocity[ids, frame + 1, 12:]).square().mean(
                -1
            ) * live
            perturbation += (executed[:, :12] - action[:, :12]).square().mean(-1) * live
            actual_sum += velocity * live[:, None]
            plateau_sum += velocity * steady[:, None]
            tracking += (velocity - goal).square() * live[:, None]
            elapsed += live
            plateau_steps += steady
            clips += (clipped | (proposed != executed)).float().mean(-1) * live
            saturation += (env.velocity_measured_torque.abs() >= torque_limit * 0.999).float().mean(
                -1
            ) * live
            native = env.backend.get_state_views(("qpos",), device="cpu")["qpos"]
            for lane in live.nonzero().flatten().tolist():
                data.qpos[:] = native[lane].numpy()
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                mujoco.mj_collision(model, data)
                contacts = np.zeros(4, dtype=bool)
                for pair in data.contact:
                    a, b = int(pair.geom1), int(pair.geom2)
                    if a in wheels and model.geom_bodyid[b] == 0:
                        contacts[wheels[a]] = True
                    if b in wheels and model.geom_bodyid[a] == 0:
                        contacts[wheels[b]] = True
                if step:
                    contact_changes[lane] += np.count_nonzero(previous[lane] != contacts)
                previous[lane] = contacts
            fallen |= live & term.mdp.hard_termination(state)
            live &= ~result.terminated & ~result.truncated & (step + 1 < end)
        rows = []
        for lane, (clip, source) in enumerate(zip(ids.tolist(), mode.tolist())):
            count = max(1, float(elapsed[lane]))
            rows.append(
                dict(
                    name=f"{MODES[source]}_{clip}",
                    domain=0,
                    category=bank.rows[clip]["motion_category"],
                    command=bank.nominal[clip].tolist(),
                    goal=bank.nominal[clip].tolist(),
                    collection_type=MODES[source],
                    start_frame=int(start[lane]),
                    requested_steps=int(end[lane]),
                    mean_actual=(actual_sum[lane] / count).tolist(),
                    plateau_mean_actual=(
                        plateau_sum[lane] / plateau_steps[lane].clamp_min(1)
                    ).tolist(),
                    plateau_samples=int(plateau_steps[lane]),
                    tracking_rmse=(tracking[lane] / count).sqrt().tolist(),
                    joint_position_rmse=float((leg_error[lane] / count).sqrt()),
                    wheel_velocity_rmse=float((wheel_error[lane] / count).sqrt()),
                    leg_action_perturbation_rms=float((perturbation[lane] / count).sqrt()),
                    survival_time=count * 0.02,
                    duration_seconds=count * 0.02,
                    fall=bool(fallen[lane]),
                    passed=not bool(fallen[lane]) and count == int(end[lane]),
                    terminal_reason="fall" if fallen[lane] else "completed",
                    contact_transition_count=int(contact_changes[lane]),
                    action_clipping_fraction=float(clips[lane] / count),
                    torque_saturation_fraction=float(saturation[lane] / count),
                )
            )
        write_json(
            output,
            {
                "bank_identity": bank.identity,
                "dt": 0.02,
                "rows": rows,
                "contact_measurement": "Native MuJoCo collision query at each controlled post-step pose",
            },
        )
        torch.save(
            dict(
                qpos=torch.stack(traces),
                active=torch.stack(active_trace),
                dt=0.04,
                cases=[rows[i] for i in shown],
                rows=[rows[i] for i in shown],
                profile="reference_audit",
            ),
            output.with_suffix(".pt"),
        )
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
