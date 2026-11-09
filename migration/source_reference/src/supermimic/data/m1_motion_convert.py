"""Convert the local AMP-LOCO-M1 Motion Bank into SuperMimic's NPZ schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, get_robot_spec


DEFAULT_BANK = Path("/home/yifeiy/AMP-LOCO-M1/dataset/m1_motion_bank_v1")
DEFAULT_OUTPUT = Path("data/npz/m1_piper")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _interp(values: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.stack([np.interp(t_dst, t_src, values[:, j]) for j in range(values.shape[1])], axis=-1)


def _angular_velocity_body(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rotations = Rotation.from_quat(quat_xyzw)
    result = np.zeros((len(quat_xyzw), 3), dtype=np.float32)
    if len(quat_xyzw) > 1:
        result[:-1] = np.stack([(rotations[i].inv() * rotations[i + 1]).as_rotvec() / dt for i in range(len(quat_xyzw) - 1)])
        result[-1] = result[-2]
    return result


def convert_m1_motion_bank(
    bank_dir: str | Path = DEFAULT_BANK,
    output_dir: str | Path = DEFAULT_OUTPUT,
    names: list[str] | None = None,
    limit: int | None = None,
    target_dt: float = 0.02,
) -> list[Path]:
    """Convert 79-D JSON frames plus valid sidecar wheel data; source assets stay read-only."""
    bank_dir, output_dir = Path(bank_dir), Path(output_dir)
    motions_dir = bank_dir / "motions"
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    source_amp_names = tuple(manifest["amp_joint_order"])
    if source_amp_names != M1_AMP_JOINTS:
        raise ValueError(f"AMP joint order mismatch: {source_amp_names}")
    if manifest.get("amp_frame_dim") != 79 or manifest.get("wheel_dofs_in_amp_frames") is not False:
        raise ValueError("unexpected M1 Motion Bank schema")
    wanted = set(names) if names else None
    files = sorted(motions_dir.glob("*.txt"))
    if wanted is not None:
        files = [p for p in files if p.stem in wanted]
    if limit is not None:
        files = files[:limit]
    if target_dt <= 0:
        raise ValueError("target_dt must be positive")
    if not files:
        raise FileNotFoundError(f"no selected M1 motion files in {motions_dir}")

    robot = get_robot_spec("m1_piper")
    model = load_robot_spec_model("m1_piper")
    body_names = robot.tracked_body_names
    # G1Kinematics stores bodies with MuJoCo's world body (id 0) omitted.
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) - 1 for name in body_names]
    if any(i < 0 for i in body_ids):
        raise ValueError(f"M1 model missing a tracked body: {body_names}")
    joint_qpos = {}
    joint_dof = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name != "root":
            joint_qpos[name] = int(model.jnt_qposadr[j])
            joint_dof[name] = int(model.jnt_dofadr[j])

    output_dir.mkdir(parents=True, exist_ok=True)
    from supermimic.kinematics.g1_fk import G1Kinematics

    fk = G1Kinematics(model, device="cpu")
    outputs = []
    for txt in files:
        data = json.loads(txt.read_text())
        frames = np.asarray(data["Frames"], dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != 79 or not np.isfinite(frames).all():
            raise ValueError(f"{txt}: expected finite (T,79), got {frames.shape}")
        dt = float(data["FrameDuration"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"{txt}: invalid FrameDuration={dt}")
        T_source = frames.shape[0]
        dt_source = float(data["FrameDuration"])
        t_source = np.arange(T_source, dtype=np.float64) * dt_source
        T = int(np.floor(t_source[-1] / target_dt + 1e-8)) + 1
        t_target = np.arange(T, dtype=np.float64) * target_dt
        root_pos = _interp(frames[:, :3], t_source, t_target).astype(np.float32)
        source_rot = Rotation.from_quat(frames[:, 3:7])
        from scipy.spatial.transform import Slerp

        q_xyzw = Slerp(t_source, source_rot)(t_target).as_quat().astype(np.float32)
        q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
        joint_pos = np.tile(np.asarray(robot.default_joint_pos, dtype=np.float32), (T, 1))
        joint_vel = np.zeros_like(joint_pos)
        amp_source_index = {name: i for i, name in enumerate(source_amp_names)}
        robot_index = {name: i for i, name in enumerate(M1_JOINTS)}
        for name in M1_AMP_JOINTS:
            j = robot_index[name]
            joint_pos[:, j] = np.interp(t_target, t_source, frames[:, 7 + amp_source_index[name]])
            joint_vel[:, j] = np.interp(t_target, t_source, frames[:, 46 + amp_source_index[name]])

        sidecar_path = txt.with_suffix(".npz")
        wheel_rate_valid = np.zeros((T, len(robot.wheel_joint_names)), dtype=bool)
        wheel_ref_valid = np.zeros_like(wheel_rate_valid)
        wheel_angles = np.full((T, len(robot.wheel_joint_names)), np.nan, dtype=np.float32)
        wheel_rates = np.full_like(wheel_angles, np.nan)
        category = str(data.get("MotionMode", "unknown"))
        root_lin_w_sidecar_valid = False
        if sidecar_path.exists():
            with np.load(sidecar_path, allow_pickle=False) as side:
                side_names = tuple(str(n) for n in side["wheel_joint_names"])
                if side_names != robot.wheel_joint_names:
                    raise ValueError(f"{sidecar_path}: wheel order mismatch: {side_names}")
                source_angles = np.asarray(side["wheel_angle_rad"], dtype=np.float32)
                source_rates = np.asarray(side["wheel_angular_velocity_rad_s"], dtype=np.float32)
                source_rate_valid = np.asarray(side["wheel_rate_valid"], dtype=bool) & np.isfinite(source_rates)
                for wi in range(len(robot.wheel_joint_names)):
                    if source_rate_valid[:, wi].all() and np.isfinite(source_angles[:, wi]).all():
                        wheel_angles[:, wi] = np.interp(t_target, t_source, np.unwrap(source_angles[:, wi]))
                        wheel_rates[:, wi] = np.interp(t_target, t_source, source_rates[:, wi])
                        wheel_rate_valid[:, wi] = True
                wheel_ref_valid = wheel_rate_valid & np.isfinite(wheel_angles)
                if "motion_category" in side:
                    category = str(side["motion_category"][0])
                if "base_linear_velocity_world_m_s" in side:
                    candidate = np.asarray(side["base_linear_velocity_world_m_s"], dtype=np.float32)
                    # The sidecar velocity is world-frame and may be absent in legacy data.
                    side_lin_w = _interp(candidate, t_source, t_target)
                    valid_src = np.isfinite(candidate).all(axis=-1)
                    if valid_src.all():
                        root_lin_w_sidecar_valid = True
                    else:
                        root_lin_w_sidecar_valid = False
        for j, name in enumerate(robot.wheel_joint_names):
            idx = robot_index[name]
            valid_angle = np.isfinite(wheel_angles[:, j])
            valid_rate = wheel_rate_valid[:, j]
            joint_pos[valid_angle, idx] = wheel_angles[valid_angle, j]
            joint_vel[valid_rate, idx] = wheel_rates[valid_rate, j]
            wheel_ref_valid[:, j] = valid_angle & valid_rate

        qpos = np.zeros((T, model.nq), dtype=np.float32)
        qvel = np.zeros((T, model.nv), dtype=np.float32)
        qpos[:, :3], qpos[:, 3:7] = root_pos, q_wxyz
        # The AMP frame stores base linear velocity in the body frame (40:43);
        # rotate it with the resampled quaternion to match StateIO/MuJoCo world qvel.
        root_lin_body = _interp(frames[:, 40:43], t_source, t_target).astype(np.float32)
        root_lin_w = Rotation.from_quat(q_xyzw).apply(root_lin_body).astype(np.float32)
        if root_lin_w_sidecar_valid:
            root_lin_w = side_lin_w.astype(np.float32)
        root_ang_body = _interp(frames[:, 43:46], t_source, t_target).astype(np.float32)
        root_ang_w = Rotation.from_quat(q_xyzw).apply(root_ang_body).astype(np.float32)
        qvel[:, :3] = root_lin_w
        # M1 AMP stores body angular velocity. MuJoCo's free-joint qvel uses body coordinates.
        qvel[:, 3:6] = root_ang_body
        for name, idx in robot_index.items():
            qpos[:, joint_qpos[name]] = joint_pos[:, idx]
            qvel[:, joint_dof[name]] = joint_vel[:, idx]
        with torch.no_grad():
            out = fk(torch.from_numpy(qpos), torch.from_numpy(qvel))
            bodies = {key: val.numpy()[:, body_ids] for key, val in out.items()}
        joint_valid = np.ones((T, len(M1_JOINTS)), dtype=bool)
        for j, name in enumerate(robot.wheel_joint_names):
            joint_valid[:, robot_index[name]] = wheel_ref_valid[:, j]

        output = output_dir / f"{txt.stem}.npz"
        np.savez_compressed(
            output,
            fps=np.asarray(1.0 / target_dt, dtype=np.float32),
            frame_dt=np.asarray(target_dt, dtype=np.float32),
            source_frame_dt=np.asarray(dt_source, dtype=np.float32),
            joint_pos=joint_pos, joint_vel=joint_vel,
            root_pos_w=root_pos, root_quat_w=q_wxyz,
            root_lin_vel_w=root_lin_w, root_ang_vel_w=root_ang_w,
            joint_names=np.asarray(M1_JOINTS), amp_joint_names=np.asarray(M1_AMP_JOINTS),
            joint_valid=joint_valid, wheel_rate_valid=wheel_rate_valid,
            wheel_reference_valid=wheel_ref_valid, wheel_joint_names=np.asarray(robot.wheel_joint_names),
            body_pos_w=bodies["pos"], body_quat_w=bodies["quat"],
            body_lin_vel_w=bodies["lin_vel"], body_ang_vel_w=bodies["ang_vel"],
            body_names=np.asarray(body_names), motion_category=np.asarray(category),
            source=np.asarray(str(txt.resolve())), source_hash=np.asarray(_sha256(txt)),
            sidecar_hash=np.asarray(_sha256(sidecar_path) if sidecar_path.exists() else ""),
            dt=np.asarray(target_dt, dtype=np.float32),
        )
        outputs.append(output)
        valid_fraction = float(wheel_ref_valid.mean()) if wheel_ref_valid.size else 0.0
        print(f"{txt.stem}: {T_source}->{T} frames, {1.0/dt_source:.1f}->{1.0/target_dt:.1f} Hz, category={category}, wheel-reference-valid={valid_fraction:.3f} -> {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", action="append", dest="names")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    convert_m1_motion_bank(args.bank_dir, args.output_dir, args.names, args.limit)


if __name__ == "__main__":
    main()
