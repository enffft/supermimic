#!/usr/bin/env python3
"""Short CUDA MuJoCo validation of the V3.3 L reset and reference action map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.reference import decode_future31
from supermimic.robots.spec import M1_JOINTS
from supermimic.tasks.action_contract import inverse_m1_setpoint_action


PRIOR = Path(
    "/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/"
    "cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt"
)
PRIOR_SHA256 = "05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c"
COMMANDS = ((0.0, "stop"), (0.12, "forward_id"), (-0.08, "backward_id"))
WHEEL_NAMES = ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")
LEG_NAMES = (
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT",
)
ARM_NAMES = tuple(f"piper_x_joint{i}" for i in range(1, 7))


def _array(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(getattr(value, "_tensor", None), torch.Tensor):
        return value._tensor.detach().cpu()
    if hasattr(value, "numpy"):
        result = value.numpy()
        if isinstance(result, torch.Tensor):
            return result.detach().cpu()
        return torch.as_tensor(result)
    return torch.as_tensor(value)


def _euler_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    sin_pitch = (2 * (w * y - z * x)).clamp(-1, 1)
    pitch = torch.asin(sin_pitch)
    return roll, pitch


def run(*, num_envs: int, steps: int, device: str, out: Path | None) -> dict:
    if num_envs < len(COMMANDS):
        raise ValueError("num_envs must be at least 3, one for each fixed command")
    if steps < 10:
        raise ValueError("use at least 10 control steps to cover one prior horizon")

    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3",
        num_envs=num_envs, history_length=5, episode_length_s=10.0,
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False, replay_capacity=32,
        seed=20261103, m1_joint_task_type="L",
        m1_joint_vx_range=(-0.09, 0.145), m1_joint_vy_range=(0.0, 0.0),
        m1_joint_wz_range=(0.0, 0.0),
    ), device=device)
    try:
        prior = load_m1_motion_prior(
            PRIOR, device=device, expected_sha256=PRIOR_SHA256,
            input_compatibility="legacy_v1",
        )
        prior.eval()
        obs, _ = env.reset(seed=20261103)
        state_io = StateIO(env)
        state = state_io.read()
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("m1_prior_joint")
        task_code = torch.zeros(num_envs, dtype=torch.long, device=device)
        command_id = torch.arange(num_envs, device=device) % len(COMMANDS)
        velocity = torch.zeros(num_envs, 3, dtype=state.dtype, device=device)
        for idx, (vx, _) in enumerate(COMMANDS):
            velocity[command_id == idx, 0] = vx
        command_term.task_type[:] = task_code
        command_term.velocity_command_body[:] = velocity
        command_term.target_position_world[:] = 0.0

        initial_state = state.clone()
        history = state[:, None].expand(-1, 4, -1).clone()
        phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        reference = torch.zeros(num_envs, 10, 31, dtype=state.dtype, device=device)
        reference_state = torch.zeros(num_envs, 10, 57, dtype=state.dtype, device=device)
        spec = env.scene["robot"].cfg.robot_spec if hasattr(env.scene["robot"].cfg, "robot_spec") else None
        if spec is None:
            from supermimic.robots.spec import get_robot_spec
            spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3")
        wheel_ids = torch.tensor([M1_JOINTS.index(name) for name in WHEEL_NAMES], device=device)
        leg_ids = torch.tensor([M1_JOINTS.index(name) for name in LEG_NAMES], device=device)
        arm_ids = torch.tensor([M1_JOINTS.index(name) for name in ARM_NAMES], device=device)
        action_term = env.action_manager.get_term("mixed")

        rows = {name: {
            "command_vx_m_s": vx,
            "envs": int((command_id == i).sum()),
            "signed_forward_displacement_m": [], "body_vx_m_s": [], "body_vy_m_s": [],
            "body_yaw_rate_rad_s": [], "root_height_m": [], "root_roll_rad": [], "root_pitch_rad": [],
            "reference_wheel_rates_rad_s": [], "actual_wheel_rates_rad_s": [],
            "wheel_rate_rmse_rad_s": [], "leg_reference_rmse_rad": [], "piper_reference_rmse_rad": [],
            "position_target_rmse_rad": [], "action_saturation_fraction": [], "target_saturation_fraction": [],
            "actuator_force_abs_mean": [], "contact_count": [], "wheel_ground_contact_count": [],
        } for i, (vx, name) in enumerate(COMMANDS)}

        model = env.sim.mj_model
        wheel_geom_ids = set()
        for gid in range(model.ngeom):
            geom_name = __import__("mujoco").mj_id2name(model, __import__("mujoco").mjtObj.mjOBJ_GEOM, gid) or ""
            if any(token in geom_name.lower() for token in ("wheel", "roller", "foot")):
                wheel_geom_ids.add(gid)
        terrain_geom_ids = set()
        for gid in range(model.ngeom):
            geom_name = __import__("mujoco").mj_id2name(model, __import__("mujoco").mjtObj.mjOBJ_GEOM, gid) or ""
            if "terrain" in geom_name.lower() or "ground" in geom_name.lower() or "floor" in geom_name.lower():
                terrain_geom_ids.add(gid)

        initial_root_q = initial_state[:, 3:7]
        initial_root_pos = initial_state[:, :3]
        with torch.inference_mode():
            for step in range(steps):
                if step == 0 or bool((phase == 0).any()):
                    goal = task_goal8(state, task_code, velocity, torch.zeros_like(velocity))
                    next_reference = prior(history, goal, mode="prior_mean")
                    reference = torch.where((phase == 0)[:, None, None], next_reference, reference)
                    decoded = decode_future31(state, next_reference)
                    reference_state = torch.where((phase == 0)[:, None, None], decoded, reference_state)
                current_ref = reference_state[torch.arange(num_envs, device=device), phase]
                q_target = current_ref[:, 7:29]
                wheel_target = current_ref[:, 35:57].index_select(-1, wheel_ids)
                action = inverse_m1_setpoint_action(q_target, wheel_target, robot_spec=spec)
                _, _, terminated, truncated, _ = env.step(action)
                state = state_io.read()
                ref_q = q_target[:, :]
                q_actual = robot.data.joint_pos[:, :]
                setpoint = action_term.physical_setpoint
                target_error = (setpoint[:, :].index_select(-1, torch.tensor([M1_JOINTS.index(n) for n in LEG_NAMES + ARM_NAMES], device=device)) -
                                torch.cat((ref_q.index_select(-1, leg_ids), ref_q.index_select(-1, arm_ids)), dim=-1))
                body_v = quat_apply_inverse(state[:, 3:7], state[:, 29:32])
                roll, pitch = _euler_roll_pitch(state[:, 3:7])
                forward = quat_apply_inverse(initial_root_q, state[:, :3] - initial_root_pos)[:, 0]
                wheel_actual = state[:, 35:57].index_select(-1, wheel_ids)
                wheel_err = wheel_actual - wheel_target
                forces = getattr(robot.data, "qfrc_actuator", None)
                force_mean = forces.abs().mean(-1) if forces is not None else torch.full((num_envs,), float("nan"), device=device)
                actual_q = state[:, 7:29]
                # Position saturation means the policy envelope is at its edge, not the physical joint limit.
                action_sat = (action.abs() >= 0.98).float().mean(-1)
                pos_ids = torch.tensor([i for i, name in enumerate(M1_JOINTS) if name not in WHEEL_NAMES], device=device)
                position_bounds = torch.tensor(spec.position_target_bounds, device=device, dtype=state.dtype)
                pos_target = setpoint.index_select(-1, pos_ids)
                pos_bounds = position_bounds.index_select(0, pos_ids)
                target_sat = ((pos_target - pos_bounds[:, 0] < 1e-4) | (pos_bounds[:, 1] - pos_target < 1e-4)).float().mean(-1)

                contact_count = torch.zeros(num_envs)
                wheel_ground = torch.zeros(num_envs)
                try:
                    data = env.sim.data
                    nacon = _array(data.nacon).reshape(-1)
                    ncontact = int(nacon[0])
                    geom = _array(data.contact.geom).reshape(-1, 2)[:ncontact]
                    worldid = _array(data.contact.worldid).reshape(-1)[:ncontact]
                    for env_i in range(num_envs):
                        contact_count[env_i] = (worldid == env_i).sum()
                    if wheel_geom_ids and terrain_geom_ids:
                        g1, g2 = geom[:, 0], geom[:, 1]
                        is_wheel_ground = (
                            (torch.isin(g1, torch.tensor(list(wheel_geom_ids))) & torch.isin(g2, torch.tensor(list(terrain_geom_ids)))) |
                            (torch.isin(g2, torch.tensor(list(wheel_geom_ids))) & torch.isin(g1, torch.tensor(list(terrain_geom_ids))))
                        )
                        for env_i in range(num_envs):
                            wheel_ground[env_i] = ((worldid == env_i) & is_wheel_ground).sum()
                except Exception:
                    nacon = None

                valid = ~(terminated | truncated)
                for idx, (_, name) in enumerate(COMMANDS):
                    mask = (command_id == idx) & valid
                    row = rows[name]
                    for key, values in (
                        ("signed_forward_displacement_m", forward), ("body_vx_m_s", body_v[:, 0]),
                        ("body_vy_m_s", body_v[:, 1]), ("body_yaw_rate_rad_s", state[:, 34]),
                        ("root_height_m", state[:, 2]), ("root_roll_rad", roll), ("root_pitch_rad", pitch),
                        ("reference_wheel_rates_rad_s", wheel_target.mean(-1)),
                        ("actual_wheel_rates_rad_s", wheel_actual.mean(-1)),
                        ("wheel_rate_rmse_rad_s", wheel_err.square().mean(-1).sqrt()),
                        ("leg_reference_rmse_rad", (actual_q.index_select(-1, leg_ids) - q_target.index_select(-1, leg_ids)).square().mean(-1).sqrt()),
                        ("piper_reference_rmse_rad", (actual_q.index_select(-1, arm_ids) - q_target.index_select(-1, arm_ids)).square().mean(-1).sqrt()),
                        ("position_target_rmse_rad", target_error.square().mean(-1).sqrt()),
                        ("action_saturation_fraction", action_sat), ("target_saturation_fraction", target_sat),
                        ("actuator_force_abs_mean", force_mean), ("contact_count", contact_count.to(device)),
                        ("wheel_ground_contact_count", wheel_ground.to(device)),
                    ):
                        row[key].append(values[mask].detach().cpu())
                history = torch.cat((history[:, 1:], state[:, None]), dim=1)
                phase = (phase + 1) % 10
                phase[terminated | truncated] = 0

        def summarize(values):
            if not values:
                return None
            vals = torch.cat(values).float()
            return float(vals.mean()) if vals.numel() else None

        initial_foot_contact = _array(env.sim.data.nacon).reshape(-1).tolist() if hasattr(env.sim.data, "nacon") else None
        result = {
            "prior_path": str(PRIOR), "prior_sha256": PRIOR_SHA256,
            "input_compatibility": "legacy_v1", "pose_profile": "dataset_v3_3_l_high_carry",
            "action_contract": "v3", "device": device, "num_envs": num_envs, "steps": steps,
            "seed": 20261103, "sim_step_dt_s": float(env.step_dt),
            "initial_root_height_mean_m": float(initial_state[:, 2].mean()),
            "initial_joint_positions_rad": {name: float(initial_state[:, 7 + i].mean()) for i, name in enumerate(M1_JOINTS)},
            "initial_contact_world_contact_total": initial_foot_contact,
            "contact_geom_detection": {"wheel_geom_ids": sorted(wheel_geom_ids), "terrain_geom_ids": sorted(terrain_geom_ids)},
            "groups": {},
        }
        for _, name in COMMANDS:
            result["groups"][name] = {key: summarize(values) for key, values in rows[name].items() if isinstance(values, list)}
            result["groups"][name]["envs"] = rows[name]["envs"]
            result["groups"][name]["command_vx_m_s"] = rows[name]["command_vx_m_s"]
        encoded = json.dumps(result, indent=2)
        print(encoded)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(encoded + "\n")
        return result
    finally:
        env.close()


def main(
    num_envs: int = 12,
    steps: int = 50,
    device: str = "cuda:0",
    out: Path | None = Path("/tmp/m1_v33_physics_validation.json"),
) -> None:
    run(num_envs=num_envs, steps=steps, device=device, out=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("/tmp/m1_v33_physics_validation.json"))
    args = parser.parse_args()
    main(**vars(args))
