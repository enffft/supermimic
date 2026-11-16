"""Native collision rays, wheel FK/radius/direction and physical reward parity."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import write_json
from unisupermimic.tasks.velocity.go2w.traversal import wheel_centers


def audit(output):
    torch.set_num_threads(1)
    env = make_go2w_env(load_config("train"), 32)
    try:
        env.reset()
        c = env.command_manager.get_term("velocity")
        field = c.mdp.terrain
        model = env.get_playback_model(0)
        data = mujoco.MjData(model)
        views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        data.qpos[:] = views["qpos"][0].numpy()
        data.qvel[:] = views["qvel"][0].numpy()
        mujoco.mj_forward(model, data)
        group = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        rays = []
        for level in range(10):
            for domain in range(1, 8):
                origin = c.curriculum.origins[level, domain - 1]
                for offset in (0.0, 0.65, 0.79, 0.81, 1.0, 1.19, 1.21, 1.59, 1.61, 1.99, 2.01, 2.8):
                    xy = origin[:2] + origin.new_tensor([offset, 0.0])
                    geom = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(
                        model,
                        data,
                        np.array([*xy.tolist(), 5.0]),
                        np.array([0.0, 0.0, -1.0]),
                        group,
                        1,
                        -1,
                        geom,
                    )
                    if distance < 0:
                        raise AssertionError("Terrain ray missed")
                    actual = 5 - distance
                    predicted = float(field.ground_height(xy))
                    rays.append(
                        dict(
                            level=level + 1,
                            domain=domain,
                            offset=offset,
                            height=actual,
                            scan_height=predicted,
                            error=abs(actual - predicted),
                        )
                    )
        max_error = max(r["error"] for r in rays)
        faces = []
        for level in range(10):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"traversal_{level}_3_0")
            left = model.geom_pos[gid, 0] - model.geom_size[gid, 0]
            point = model.geom_pos[gid].copy()
            point[0] = left - 0.2
            geom = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(model, data, point, np.array([1.0, 0.0, 0.0]), group, 1, -1, geom)
            faces.append(
                dict(
                    level=level + 1,
                    height=float(2 * model.geom_size[gid, 2]),
                    horizontal_ray_distance=float(dist),
                    geom=int(geom[0]),
                    expected_geom=gid,
                )
            )
        native = []
        wheel_rows = []
        for leg in ("FL", "FR", "RL", "RR"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{leg}_wheel_link")
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{leg}_wheel_collision")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_wheel_joint")
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            n = model.mesh_vertnum[mid]
            vertices = (
                model.mesh_vert[start : start + n] @ data.geom_xmat[gid].reshape(3, 3).T
                + data.geom_xpos[gid]
            )
            center = data.xpos[bid]
            axis = data.xaxis[jid]
            relative = vertices - center
            radial = relative - (relative @ axis)[:, None] * axis
            radius = float(np.linalg.norm(radial, axis=-1).max())
            forward = np.cross(axis, [0, 0, 1])
            native.append(center.copy())
            wheel_rows.append(
                dict(
                    leg=leg,
                    radius_m=radius,
                    axis_world=axis.tolist(),
                    positive_rolling_direction=forward.tolist(),
                    forward_sign=1 if forward[0] > 0 else -1,
                )
            )
        predicted = wheel_centers(c.state())[0].numpy()
        fk_error = float(np.max(np.abs(predicted - np.array(native))))
        parity = []
        targets = c.traversal.clone()
        ids = c.target_id.clone()
        for step in range(5):
            action = torch.rand(32, 16) * 0.01
            result = env.step(action)
            tags = env.velocity_terrain_tags
            pre = env.velocity_pre_action
            nxt = env.velocity_next_state
            imagined = c.mdp.reward_transition(
                pre["state"],
                nxt,
                pre["goal"],
                env.velocity_executed_action,
                pre["previous_action"],
                **c.mdp.imagination_reward_kwargs(
                    dict(
                        command_domain=tags["command_domain"],
                        traversal=tags["traversal"],
                        success_peak=tags["success_peak"],
                    ),
                    1,
                ),
            )
            parity.append(float((result.reward - imagined).abs().max()))
            same = ids == c.target_id
            torch.testing.assert_close(targets[same], c.traversal[same], atol=0, rtol=0)
            targets = c.traversal.clone()
            ids = c.target_id.clone()
        passed = (
            max_error < 0.008
            and fk_error < 1e-5
            and max(parity) < 1e-6
            and all(
                abs(r["horizontal_ray_distance"] - 0.2) < 1e-6 and r["geom"] == r["expected_geom"]
                for r in faces
            )
        )
        passed &= all(
            abs(r["radius_m"] - 0.086) < 1e-5 and r["forward_sign"] == 1 for r in wheel_rows
        )
        report = dict(
            passed=bool(passed),
            collision_scan_max_abs_m=max_error,
            wheel_fk_max_abs_m=fk_error,
            reward_parity_max_abs=max(parity),
            vertical_faces=faces,
            wheels=wheel_rows,
            terrain_rays=rays,
        )
        write_json(output, report)
        print(
            {
                k: v
                for k, v in report.items()
                if k not in ["terrain_rays", "vertical_faces", "wheels"]
            }
        )
        return 0 if passed else 2
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(audit(parser.parse_args().output))
