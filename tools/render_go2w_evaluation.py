"""Render saved first-episode physical evaluation traces; never steps a policy or WM."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env


def render(traces, output):
    torch.set_num_threads(1)
    saved = torch.load(traces, map_location="cpu", weights_only=True)
    profile = saved.get("profile", "train")
    env = make_go2w_env(load_config("motion_prior" if profile == "reference_audit" else profile), 1)
    try:
        model = env.get_playback_model(0)
        data = mujoco.MjData(model)
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base")
        renderer = mujoco.Renderer(model, height=240, width=400)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 1
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 1.6
        camera.azimuth = 120
        camera.elevation = -20
        groups = {
            "flat": [
                (0, "flat_stop"),
                (0, "flat_forward"),
                (0, "flat_left"),
                (0, "flat_right"),
                (0, "flat_yaw_left"),
                (0, "flat_yaw_right"),
            ],
            "stairs": [
                (4, "single_step_up_level1"),
                (4, "single_step_up_level9"),
                (5, "single_step_down_level9"),
                (6, "multi_stairs_up_level6"),
                (7, "multi_stairs_down_level6"),
                (6, "multi_stairs_up_level10"),
            ],
        }
        if saved.get("profile") == "motion_prior":
            groups = {
                "flat": [
                    (0, name)
                    for name in (
                        "standing_0",
                        "forward_1",
                        "backward_1",
                        "left_1",
                        "right_1",
                        "yaw_left_1",
                    )
                ],
                "coupled": [
                    (0, name)
                    for name in (
                        "yaw_right_1",
                        "forward_yaw_0",
                        "forward_lateral_0",
                        "lateral_yaw_0",
                        "lateral_yaw_1",
                        "forward_yaw_1",
                    )
                ],
            }
        if profile == "reference_audit":
            groups = {
                "reference_pd": [(case["domain"], case["name"]) for case in saved["cases"][:6]]
            }
        output.mkdir(parents=True, exist_ok=True)
        for label, group in groups.items():
            indices = [
                next(
                    i
                    for i, c in enumerate(saved["cases"])
                    if c["domain"] == d and c["name"] == name
                )
                for d, name in group
            ]
            with imageio.get_writer(
                output / f"{label}.mp4", fps=25, codec="libx264", quality=7
            ) as writer:
                last_valid = {i: 0 for i in indices}
                dt = saved["dt"]
                for step in range(0, len(saved["qpos"]), round(0.04 / dt)):
                    tiles = []
                    for i in indices:
                        if saved["active"][step, i]:
                            last_valid[i] = step
                        native = saved["qpos"][last_valid[i], i].numpy()
                        data.qpos[:] = native
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = data.xpos[base]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        frame = Image.fromarray(renderer.render().copy())
                        draw = ImageDraw.Draw(frame)
                        row = saved["rows"][i]
                        draw.rectangle((0, 0, 400, 46), fill=(20, 20, 20))
                        draw.text((7, 5), f"{row['name']} | cmd {row['goal']}", fill="white")
                        text = f"t={min(step * dt, row['duration_seconds']):.2f}s | "
                        text += "active" if saved["active"][step, i] else row["terminal_reason"]
                        text += f" | pass={row['passed']}"
                        draw.text((7, 25), text, fill="white")
                        tiles.append(np.array(frame))
                    writer.append_data(
                        np.concatenate(
                            (np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)),
                            axis=0,
                        )
                    )
            print(output / f"{label}.mp4", flush=True)
        renderer.close()
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.traces, args.output)
