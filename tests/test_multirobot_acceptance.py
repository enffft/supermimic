"""Small acceptance checks for the Go2 and M1 adapters.

GPU checks intentionally use 8–16 environments and short runs. They are not training
benchmarks; they pin down the simulator/model interfaces and the model-based semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
M1_ACCEPTED = (
    "a_wheel_000000",
    "a_wheel_000002",
    "b_arm_000006",
    "c_assisted_000029",
    "c_assisted_000039",
    "c_assisted_000149",
)
M1_BANK = Path("/home/yifeiy/AMP-LOCO-M1/dataset/m1_motion_bank_v1")
M1_OUTPUT = ROOT / "data/npz/m1_piper"
M1_MANIFEST = ROOT / "configs/m1_piper_first6.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_compiled_robot_joint_order_matches_robot_spec():
    """MuJoCo qpos/qvel addresses and the public actor order agree by name."""
    import mujoco

    from supermimic.robots.mjlab_assets import load_robot_spec_model
    from supermimic.robots.spec import get_robot_spec

    for name in ("g1", "go2", "m1_piper"):
        spec = get_robot_spec(name)
        model = load_robot_spec_model(name)
        scalar = sorted(
            (
                int(model.jnt_qposadr[i]),
                int(model.jnt_dofadr[i]),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
            )
            for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        )
        assert tuple(x[2] for x in scalar) == spec.joint_names
        assert [x[0] for x in scalar] == list(range(7, 7 + spec.num_joints))
        assert [x[1] for x in scalar] == list(range(6, 6 + spec.num_joints))
    m1 = get_robot_spec("m1_piper")
    assert tuple(m1.joint_names[i] for i in (3, 7, 11, 15)) == m1.wheel_joint_names


def test_m1_first_phase_manifest_and_conversion_integrity():
    """The six approved NPZs still match their 79-D source and wheel sidecars."""
    from scipy.spatial.transform import Rotation
    from scipy.spatial.transform import Slerp

    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, get_robot_spec

    if not M1_BANK.exists() or not M1_OUTPUT.exists() or not M1_MANIFEST.exists():
        pytest.skip("local M1 Motion Bank and converted NPZs are not available")
    manifest = list(csv.DictReader(M1_MANIFEST.open()))
    selected = tuple(row["name"] for row in manifest if row["keep"] == "1")
    assert selected == M1_ACCEPTED
    files = discover_npz(M1_OUTPUT, manifest=M1_MANIFEST)
    assert tuple(path.stem for path in files) == M1_ACCEPTED

    bank_manifest = json.loads((M1_BANK / "manifest.json").read_text())
    assert tuple(bank_manifest["amp_joint_order"]) == M1_AMP_JOINTS
    assert bank_manifest["amp_frame_dim"] == 79
    spec = get_robot_spec("m1_piper")
    joint_index = {name: i for i, name in enumerate(M1_JOINTS)}
    amp_index = {name: i for i, name in enumerate(M1_AMP_JOINTS)}
    wheel_index = [joint_index[name] for name in spec.wheel_joint_names]

    for name in M1_ACCEPTED:
        source = M1_BANK / "motions" / f"{name}.txt"
        sidecar_path = source.with_suffix(".npz")
        with source.open() as f:
            clip = json.load(f)
        frames = np.asarray(clip["Frames"], dtype=np.float32)
        assert frames.ndim == 2 and frames.shape[1] == 79 and np.isfinite(frames).all()
        dt_src = float(clip["FrameDuration"])
        t_src = np.arange(len(frames), dtype=np.float64) * dt_src
        t_dst = np.arange(int(np.floor(t_src[-1] / 0.02 + 1e-8)) + 1, dtype=np.float64) * 0.02
        with np.load(M1_OUTPUT / f"{name}.npz", allow_pickle=False) as out:
            assert float(out["fps"]) == pytest.approx(50.0)
            assert float(out["frame_dt"]) == pytest.approx(0.02)
            assert float(out["source_frame_dt"]) == pytest.approx(dt_src)
            assert str(out["source_hash"]) == _sha256(source)
            assert str(out["sidecar_hash"]) == _sha256(sidecar_path)
            assert tuple(out["joint_names"].tolist()) == M1_JOINTS
            assert tuple(out["amp_joint_names"].tolist()) == M1_AMP_JOINTS
            assert tuple(out["wheel_joint_names"].tolist()) == spec.wheel_joint_names
            assert tuple(out["body_names"].tolist()) == spec.tracked_body_names
            assert out["joint_pos"].shape == (len(t_dst), 22)
            assert out["joint_vel"].shape == (len(t_dst), 22)
            assert out["root_pos_w"].shape == (len(t_dst), 3)

            interp = lambda x: np.stack([np.interp(t_dst, t_src, x[:, j]) for j in range(x.shape[1])], -1)
            np.testing.assert_allclose(out["root_pos_w"], interp(frames[:, :3]), atol=2e-6, rtol=0)
            rotations = Rotation.from_quat(frames[:, 3:7])
            q_xyzw = Slerp(t_src, rotations)(t_dst).as_quat().astype(np.float32)
            np.testing.assert_allclose(out["root_quat_w"], q_xyzw[:, [3, 0, 1, 2]], atol=3e-6, rtol=0)
            for name_i in M1_AMP_JOINTS:
                j = joint_index[name_i]
                a = amp_index[name_i]
                np.testing.assert_allclose(out["joint_pos"][:, j], np.interp(t_dst, t_src, frames[:, 7 + a]), atol=2e-6, rtol=0)
                np.testing.assert_allclose(out["joint_vel"][:, j], np.interp(t_dst, t_src, frames[:, 46 + a]), atol=2e-6, rtol=0)

            with np.load(sidecar_path, allow_pickle=False) as side:
                assert tuple(side["wheel_joint_names"].tolist()) == spec.wheel_joint_names
                rates_valid = np.asarray(side["wheel_rate_valid"], bool) & np.isfinite(side["wheel_angular_velocity_rad_s"])
                angle_valid = np.isfinite(side["wheel_angle_rad"])
                expected_wheel_valid = np.zeros_like(out["wheel_reference_valid"])
                expected_joint_valid = np.ones_like(out["joint_valid"])
                for wi in range(4):
                    if rates_valid[:, wi].all() and angle_valid[:, wi].all():
                        expected_wheel_valid[:, wi] = True
                        expected_joint_valid[:, wheel_index[wi]] = True
                    else:
                        expected_joint_valid[:, wheel_index[wi]] = False
                np.testing.assert_array_equal(out["wheel_reference_valid"], expected_wheel_valid)
                np.testing.assert_array_equal(out["joint_valid"], expected_joint_valid)
                body_lin = interp(frames[:, 40:43])
                fallback_lin_w = Rotation.from_quat(q_xyzw).apply(body_lin)
                if "base_linear_velocity_world_m_s" in side and np.isfinite(side["base_linear_velocity_world_m_s"]).all():
                    expected_lin_w = interp(side["base_linear_velocity_world_m_s"])
                else:
                    expected_lin_w = fallback_lin_w
                np.testing.assert_allclose(out["root_lin_vel_w"], expected_lin_w, atol=3e-5, rtol=0)
                for wi, j in enumerate(wheel_index):
                    if expected_wheel_valid[:, wi].all():
                        wheel_angle = np.interp(t_dst, t_src, np.unwrap(side["wheel_angle_rad"][:, wi]))
                        wheel_rate = np.interp(t_dst, t_src, side["wheel_angular_velocity_rad_s"][:, wi])
                        np.testing.assert_allclose(out["joint_pos"][:, j], wheel_angle, atol=3e-5, rtol=0)
                        np.testing.assert_allclose(out["joint_vel"][:, j], wheel_rate, atol=3e-5, rtol=0)


@pytest.fixture(scope="module")
def m1_env():
    if not torch.cuda.is_available():
        pytest.skip("M1 MuJoCo-Warp acceptance requires CUDA")
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(M1_OUTPUT, names=list(M1_ACCEPTED), manifest=M1_MANIFEST)
    if len(files) != len(M1_ACCEPTED):
        pytest.skip("the six accepted M1 motion NPZs are not available")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(path) for path in files], robot_name="m1_piper", task_name="motion_tracking",
            num_envs=16, replay_capacity=512, episode_length_s=20.0, action_noise_std=0.0,
            obs_noise=False, reset_noise=False, domain_randomization=True, sampling_mode="start",
            record_transitions=True, anchor_obs=True, seed=29,
        ),
        device="cuda:0",
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    yield env
    env.close()


def _m1_heads(env):
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.mjlab_assets import load_robot_spec_model
    from supermimic.robots.spec import get_robot_spec
    from supermimic.world_model.heads import AnalyticHeads

    spec = get_robot_spec("m1_piper")
    robot = env.scene["robot"]
    return AnalyticHeads(
        G1Kinematics(load_robot_spec_model("m1_piper"), device=env.device),
        spec.tracked_body_names, spec.anchor_body_name, spec.root_body_name,
        robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
        default_base_height=spec.default_base_height,
        reward_anchor_at_start=True,
    )


def _ref_from_m1_command(cmd, frame: torch.Tensor | None = None):
    from supermimic.world_model.heads import ReferenceFrame

    library, frame = cmd.library, cmd.frame if frame is None else frame
    bodies, ref = library.bodies(frame), library.ref(frame)
    return ReferenceFrame(
        bodies.pos_w, bodies.quat_w, bodies.lin_vel_w, bodies.ang_vel_w,
        ref.root_pos_w, ref.root_quat_w,
    )


def _newest_m1_frame(policy_obs: torch.Tensor) -> torch.Tensor:
    n = policy_obs.shape[0]
    chunks = []
    start = 0
    for width in (22, 22, 3, 3, 22):
        term = policy_obs[:, start * 10 : (start + width) * 10].reshape(n, 10, width)
        chunks.append(term[:, -1])
        start += width
    return torch.cat(chunks, dim=-1)


def test_m1_real_and_analytic_observation_parity_and_action_order(m1_env):
    from supermimic.envs.state_io import StateIO
    from supermimic.robots.spec import get_robot_spec
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    env = m1_env
    spec = get_robot_spec("m1_piper")
    obs, _ = env.reset()
    robot = env.scene["robot"]
    recorder = env.recorder_manager.get_term("replay")
    io = StateIO(env)
    action_term = env.action_manager.get_term("mixed")
    assert tuple(robot.joint_names) == spec.joint_names
    assert action_term._joint_ids.tolist() == list(range(spec.num_joints))
    assert tuple(spec.joint_names[i] for i in action_term._wheel_action_idx.tolist()) == spec.wheel_joint_names
    assert tuple(spec.joint_names[i] for i in (3, 7, 11, 15)) == spec.wheel_joint_names

    # P0-B: two named action coordinates update only their matching position/velocity targets.
    action = torch.zeros(env.num_envs, spec.num_joints, device=env.device)
    wheel_i, arm_i = 3, 16
    action[:, wheel_i] = 0.4
    action[:, arm_i] = 0.2
    action_term.process_actions(action)
    action_term.apply_actions()
    after_pos = robot.data.joint_pos_target.clone()
    after_vel = robot.data.joint_vel_target.clone()
    expected_pos = robot.data.default_joint_pos.clone()
    expected_pos[:, arm_i] += 0.2 * spec.action_scales[arm_i]
    expected_vel = torch.zeros_like(after_vel)
    expected_vel[:, wheel_i] = 0.4 * spec.action_scales[wheel_i]
    torch.testing.assert_close(after_pos, expected_pos, atol=2e-5, rtol=0.0)
    torch.testing.assert_close(after_vel, expected_vel, atol=2e-5, rtol=0.0)

    # Reset so both sides start with the same simulator history. Domain-randomised encoder bias
    # remains active, while observation corruption and action noise are disabled.
    obs, _ = env.reset()
    action_term = env.action_manager.get_term("mixed")
    q0 = robot.data.default_joint_pos[0]
    wheel_idx = tuple(spec.joint_names.index(name) for name in spec.wheel_joint_names)
    expected_history: list[torch.Tensor] = []
    observation_error_max = 0.0
    cases = ("default", "wheel_nonzero", "wheel_over_2pi", "wheel_velocity", "piper_position") * 2
    for case in cases:
        s = io.read()
        if case == "wheel_nonzero":
            s[:, 7 + wheel_idx[0]] = 0.73
        elif case == "wheel_over_2pi":
            s[:, 7 + wheel_idx[0]] = 2.0 * math.pi + 0.41
        elif case == "wheel_velocity":
            s[:, io.nq + 6 + wheel_idx[1]] = 1.6
        elif case == "piper_position":
            s[:, 7 + 16] = q0[16] + 0.27
        io.write(torch.arange(env.num_envs, device=env.device), s)
        action = torch.zeros(env.num_envs, spec.num_joints, device=env.device)
        action[:, 3] = 0.2
        action[:, 16] = -0.15
        obs, _, terminated, timeout, _ = env.step(action)
        assert not (terminated | timeout).any()
        analytic = analytic_observation(io.read(), env.action_manager.action, recorder.ctx.read(), q0, wheel_idx)
        newest = _newest_m1_frame(obs["policy"])
        observation_error_max = max(observation_error_max, float((analytic - newest).abs().max()))
        torch.testing.assert_close(analytic, newest, atol=3e-5, rtol=0.0)
        expected_history.append(analytic)

    flattened = flatten_history_term_major(torch.stack(expected_history[-10:], dim=0), num_joints=22)
    assert obs["policy"].shape == (16, 720)
    history_error_max = float((flattened - obs["policy"]).abs().max())
    torch.testing.assert_close(flattened, obs["policy"], atol=3e-5, rtol=0.0)
    print("M1 real/analytic observation parity:", {
        "single_frame_max_abs_error": observation_error_max,
        "flattened_720d_max_abs_error": history_error_max,
        "wheel_indices": wheel_idx,
        "tested_cases": sorted(set(cases)),
    })


def test_m1_real_and_analytic_reward_terms_match(m1_env):
    from supermimic.envs.state_io import StateIO

    env = m1_env
    heads = _m1_heads(env)
    io = StateIO(env)
    env.reset()
    cmd = env.command_manager.get_term("motion")
    samples = {}
    for step in range(12):
        state_before = io.read()
        ref_before = _ref_from_m1_command(cmd, cmd.frame.clone())
        action = torch.zeros(env.num_envs, 22, device=env.device)
        action[:, 16 + step % 6] = 0.08 * math.sin(step)
        action[:, 3] = 0.04
        _, _, terminated, timeout, _ = env.step(action)
        alive = ~(terminated | timeout)
        assert alive.any()
        ours = heads.reward(
            io.read(), env.action_manager.action, env.action_manager.prev_action,
            ref_before, anchor_state=state_before,
        )
        # RewardManager.get_term is its already weighted, dt-scaled per-step value.
        for name, value in ours.items():
            if name == "total":
                continue
            theirs = env.reward_manager.get_term(name)
            row = samples.setdefault(name, {"real": [], "analytic": [], "abs": []})
            row["real"].append(theirs[alive].detach().cpu())
            row["analytic"].append(value[alive].detach().cpu())
            row["abs"].append((value[alive] - theirs[alive]).abs().detach().cpu())
    assert set(samples) == {
        "motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori",
        "motion_body_lin_vel", "motion_body_ang_vel", "action_rate_l2", "joint_limit",
    }
    import json

    report = {}
    for name, row in samples.items():
        real = torch.cat(row["real"])
        analytic = torch.cat(row["analytic"])
        absolute = torch.cat(row["abs"])
        report[name] = {
            "real_mean": float(real.mean()),
            "analytic_mean": float(analytic.mean()),
            "absolute_error_mean": float(absolute.mean()),
            "absolute_error_max": float(absolute.max()),
            "relative_mean_error": float(absolute.mean() / real.abs().mean().clamp_min(1e-8)),
        }
    print("M1 real/analytic reward parity:", json.dumps(report, sort_keys=True))
    assert max(x["absolute_error_max"] for x in report.values()) < 3e-4, report


def test_go2_zero_action_stand_and_termination_features():
    if not torch.cuda.is_available():
        pytest.skip("Go2 MuJoCo-Warp acceptance requires CUDA")
    import mujoco
    import warp as wp
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.terminations import free_root_tilted, free_root_too_low
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.robots.spec import get_robot_spec
    from supermimic.world_model.heads import AnalyticHeads

    spec = get_robot_spec("go2")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[], robot_name="go2", task_name="velocity_command", num_envs=16,
            episode_length_s=20.0, action_noise_std=0.0, obs_noise=False, reset_noise=False,
            domain_randomization=False, record_transitions=False, seed=3,
            velocity_gaussian_noise_std=0.0, velocity_ou_env_fraction=0.0, velocity_ou_noise_std=0.0,
            velocity_curriculum=False, velocity_curriculum_fixed_stage=0,
        ),
        device="cuda:0",
    )
    try:
        env.reset()
        robot = env.scene["robot"]
        assert tuple(robot.joint_names) == spec.joint_names
        root_z_hist = []
        tilt_hist = []
        force_tensor_hist = []
        ang_vel_hist = []
        for _ in range(400):  # 8 s at the 20 ms control period.
            env.step(torch.zeros(16, 12, device=env.device))
            root_z_hist.append(robot.data.root_link_pos_w[:, 2].detach().clone())
            g_local = quat_apply_inverse(robot.data.root_link_quat_w, robot.data.gravity_vec_w)
            tilt_hist.append((1.0 + g_local[:, 2]).abs().detach().clone())
            force_tensor_hist.append(robot.data.actuator_force.abs().detach().clone())
            ang_vel_hist.append(robot.data.root_link_ang_vel_b.abs().max().detach().clone())

        root_z_hist = torch.stack(root_z_hist)
        tilt_hist = torch.stack(tilt_hist)
        force_tensor_hist = torch.stack(force_tensor_hist)
        ang_vel_hist = torch.stack(ang_vel_hist)
        final_z = robot.data.root_link_pos_w[:, 2]
        final_tilt = tilt_hist[-1]
        assert float(root_z_hist[-100:].mean()) == pytest.approx(0.282, abs=0.015)
        assert float(root_z_hist[-100:].std()) < 0.01
        assert float(final_tilt.max()) < 0.05
        assert float(ang_vel_hist[-100:].max()) < 0.5
        assert float(force_tensor_hist.max()) < 23.5
        assert env.reset_buf.sum().item() == 0
        assert env.cfg.terminations["root_height"].params["threshold"] == pytest.approx(0.17)

        # Confirm all four named foot collision geoms contact the terrain in every env.
        ground_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        foot_ids = [mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{n}_collision") for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
        contact = env.sim.data.struct.contact
        world = wp.to_torch(contact.worldid)
        geom = wp.to_torch(contact.geom)
        dim = wp.to_torch(contact.dim)
        active = (dim > 0) & (world >= 0)
        for env_i in range(16):
            local = active & (world == env_i)
            pairs = geom[local]
            for foot_id in foot_ids:
                touches = ((pairs[:, 0] == ground_id) & (pairs[:, 1] == foot_id)) | ((pairs[:, 1] == ground_id) & (pairs[:, 0] == foot_id))
                assert bool(touches.any()), (env_i, foot_id)
        force_limit = torch.as_tensor(env.sim.mj_model.actuator_forcerange[:, 1], device=env.device).abs()
        saturation_fraction = float((force_tensor_hist >= force_limit[None, None, :] * 0.999).float().mean())
        ground_contact_geoms = set()
        ground_contacts = geom[active & (world >= 0)]
        for pair in ground_contacts.detach().cpu().tolist():
            if ground_id in pair:
                ground_contact_geoms.add(pair[1] if pair[0] == ground_id else pair[0])
        ground_contact_names = sorted(env.sim.mj_model.geom(i).name for i in ground_contact_geoms)
        bad_leg_contacts = [name for name in ground_contact_names if any(x in name.lower() for x in ("hip", "thigh", "calf"))]
        assert not bad_leg_contacts, bad_leg_contacts
        joint_margin = torch.minimum(
            robot.data.joint_pos - robot.data.soft_joint_pos_limits[..., 0],
            robot.data.soft_joint_pos_limits[..., 1] - robot.data.joint_pos,
        )
        foot_body_ids = [robot.body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
        final_foot_z = robot.data.body_link_pos_w[:, foot_body_ids, 2]

        # The free-task feature thresholds must produce exactly the environment's labels.
        thresh = float(env.cfg.terminations["root_height"].params["threshold"])
        real_low = free_root_too_low(env, SceneEntityCfg("robot"), thresh)
        tilt_threshold = 1.0 - math.cos(math.radians(70.0))
        real_tilt = free_root_tilted(env, SceneEntityCfg("robot"), tilt_threshold)
        # Use the actual root positions for the differentiable World Model feature path.
        feat = AnalyticHeads.free_termination_features(robot.data.root_link_pos_w, robot.data.root_link_quat_w, spec.default_base_height)
        torch.testing.assert_close(real_low, robot.data.root_link_pos_w[:, 2] < thresh)
        torch.testing.assert_close(real_tilt, feat[:, 1].abs() > tilt_threshold)
        assert not real_low.any() and not real_tilt.any()

        # Exercise both failure sides of the shared labels on deliberately perturbed states.
        from supermimic.envs.state_io import StateIO
        io = StateIO(env)
        state = io.read()
        state[0, 2] = thresh - 0.01
        state[1, 3:7] = torch.tensor([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], device=env.device)
        io.write(torch.tensor([0, 1], device=env.device), state[:2])
        feat = AnalyticHeads.free_termination_features(robot.data.root_link_pos_w, robot.data.root_link_quat_w, spec.default_base_height)
        real_low = free_root_too_low(env, SceneEntityCfg("robot"), thresh)
        real_tilt = free_root_tilted(env, SceneEntityCfg("robot"), tilt_threshold)
        torch.testing.assert_close(real_low, robot.data.root_link_pos_w[:, 2] < thresh)
        torch.testing.assert_close(real_tilt, feat[:, 1].abs() > tilt_threshold)
        assert real_low[0] and real_tilt[1]
        print("Go2 zero-action stand:", {
            "root_z_last_100_mean": float(root_z_hist[-100:].mean()),
            "root_z_last_100_std": float(root_z_hist[-100:].std()),
            "root_z_final_min": float(final_z.min()), "root_z_final_max": float(final_z.max()),
            "final_tilt_feature_max": float(final_tilt.max()),
            "root_ang_vel_last_100_max": float(ang_vel_hist[-100:].max()),
            "actuator_force_abs_max": float(force_tensor_hist.max()),
            "torque_saturation_fraction": saturation_fraction,
            "foot_body_z_mean": float(final_foot_z.mean()),
            "min_soft_joint_limit_margin_rad": float(joint_margin.min()),
            "ground_contact_geoms": ground_contact_names,
            "reset_count": int(env.reset_buf.sum()), "foot_contact_envs": 16,
            "root_height_threshold_m": thresh,
        })
    finally:
        env.close()


def test_go2_reward_parity_ou_execution_and_replay_action():
    if not torch.cuda.is_available():
        pytest.skip("Go2 MuJoCo-Warp acceptance requires CUDA")
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.robots.spec import get_robot_spec
    from supermimic.tasks.velocity_reward import VelocityRobotParams, velocity_reward_terms

    spec = get_robot_spec("go2")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[], robot_name="go2", task_name="velocity_command", num_envs=16,
            episode_length_s=20.0, action_noise_std=0.0, obs_noise=False, reset_noise=False,
            domain_randomization=False, record_transitions=True, seed=19,
            velocity_gaussian_noise_std=0.1, velocity_ou_env_fraction=0.25,
            velocity_ou_noise_std=0.3, velocity_ou_theta=2.0,
            velocity_curriculum=False, velocity_curriculum_fixed_stage=0,
        ),
        device="cuda:0",
    )
    try:
        env.reset()
        action = torch.full((16, spec.num_joints), 0.05, device=env.device)
        _, _, terminated, truncated, _ = env.step(action)
        assert not (terminated | truncated).any()

        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        command = env.command_manager.get_term("velocity")
        q0 = robot.data.default_joint_pos[0].detach()
        params = VelocityRobotParams(
            default_joint_pos=q0,
            soft_joint_limits=robot.data.soft_joint_pos_limits[0],
            action_scales=q0.new_tensor(spec.action_scales),
            kp=q0.new_tensor(spec.kp),
            kd=q0.new_tensor(spec.kd),
            effort_limits=torch.full_like(q0, 23.5),
            dt=env.step_dt,
        )
        state = StateIO(env).read()
        # RewardManager evaluates before the final MuJoCo forward pass; use the exact
        # real simulator tensors cached at that reward timestamp for a like-for-like call.
        reward_inputs = env._supermimic_go2_reward_cache["inputs"]
        direct = velocity_reward_terms(
            reward_inputs["state_prev"], reward_inputs["state"], reward_inputs["goal"],
            reward_inputs["action"], reward_inputs["previous_action"], reward_inputs["params"],
            measured_torque=reward_inputs["torque"],
            measured_joint_acceleration=reward_inputs["joint_acceleration"],
        )
        imagined_proxy = velocity_reward_terms(
            reward_inputs["state_prev"], reward_inputs["state"], reward_inputs["goal"],
            reward_inputs["action"], reward_inputs["previous_action"], reward_inputs["params"],
        )
        parity = {}
        proxy_error = {}
        for name, expected in direct.items():
            actual = env.reward_manager.get_term(name)
            torch.testing.assert_close(actual, expected * env.step_dt, atol=2.0e-6, rtol=1.0e-5, msg=lambda msg: f"{name}: {msg}")
            parity[name] = float((actual - expected * env.step_dt).abs().max())
            proxy_error[name] = float((imagined_proxy[name] - expected).abs().mean())

        # The policy action stays clean; the actuator and replay use the noisy executed action.
        torch.testing.assert_close(term.raw_action, action)
        assert bool((term.executed_raw - action).abs().max() > 0)
        assert float(term.executed_raw.abs().max()) <= 1.0
        expected_target = robot.data.default_joint_pos + term.executed_raw * q0.new_tensor(spec.action_scales)
        torch.testing.assert_close(robot.data.joint_pos_target, expected_target, atol=1.0e-6, rtol=0.0)
        replay = env.recorder_manager.get_term("replay").buffer
        torch.testing.assert_close(replay.a[:, 0], action)
        torch.testing.assert_close(replay.a_exec[:, 0], term.executed_raw)
        ou_fraction = float(term._ou_mask.float().mean())
        assert 0.0 < ou_fraction < 1.0
        print("Go2 reward/action parity:", json.dumps({
            "per_term_max_abs_error": parity,
            "proxy_vs_real_per_term_mean_abs_error_rate": proxy_error,
            "ou_env_fraction": ou_fraction,
            "clean_action_abs_mean": float(term.raw_action.abs().mean()),
            "executed_action_abs_mean": float(term.executed_raw.abs().mean()),
            "replay_matches_executed_action": True,
        }, sort_keys=True))
    finally:
        env.close()


def test_m1_eight_step_wm_backward_and_32_step_actor_gradient(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("World Model / Actor BPTT acceptance requires CUDA")
    from argparse import Namespace

    from supermimic.scripts.diagnose_multirobot import run

    report_path = tmp_path / "m1_short_diagnostic.json"
    result = run(
        Namespace(
            robot="m1_piper", device="cuda:0", num_envs=8, steps=24,
            wm_updates=1, window_batch=4, diagnostic_windows=4, imag_batch=4,
            action_noise=0.02, domain_randomization=False, seed=71,
            output=report_path,
        )
    )
    assert report_path.exists()
    assert result["policy_optimizer_steps"] == 0
    assert len(result["world_model_updates"]) == 1
    assert math.isfinite(result["world_model_updates"][0]["loss"])
    horizons = {item["horizon"]: item for item in result["imagination"]}
    assert set(horizons) == {8, 32}
    assert math.isfinite(horizons[8]["actor_grad_norm"])
    assert math.isfinite(horizons[32]["actor_grad_norm"])
