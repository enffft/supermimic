"""Replay kinematic Go2W clips in MuJoCo. Press H to show the controls."""

import argparse
from collections.abc import Iterable
import csv
import json
from pathlib import Path
from queue import SimpleQueue
from time import perf_counter, sleep

import mujoco
import numpy as np

from . import LEGS, ROLLING, STANCE, SWING
from .kinematics import Robot, hull
from .quality_check import load_clip
from .trajectories import heading_velocity, rotations

OVERLAYS = ("base", "com", "target", "fk", "support", "contacts")
QUALITIES = ("qualified", "candidate", "rejected")
MODES = {STANCE: "STANCE", ROLLING: "ROLLING", SWING: "SWING"}
COLORS = ((1., .38, .27, 1.), (.31, .74, 1., 1.), (.48, .87, .35, 1.), (.99, .74, .26, 1.))
BASE_COLOR = (.31, .81, 1., .85)
COM_COLOR = (1., .79, .27, .95)
SUPPORT_COLOR = (.29, .96, .61, .85)
FLAT = np.eye(3).ravel()


class Playback:
    def __init__(self, dataset, category=None, quality=None, motion=None):
        self.dataset = Path(dataset).resolve()
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        self.dt = manifest["dt"]
        self.rows = manifest["motions"]
        robot = self.robot = Robot()
        if (manifest["quaternion_order"] != "wxyz"
                or manifest["leg_joint_names"] + manifest["wheel_joint_names"] != list(robot.joint_names)
                or manifest["wheel_order"] != list(LEGS)):
            raise ValueError("Viewer requires the named Go2W joint order and wxyz quaternions")
        audit = self.dataset / "reports/quality_report.csv"
        if audit.exists():
            with audit.open() as stream:
                status = {row["motion_id"]: row["quality_status"] for row in csv.DictReader(stream)}
            self.rows = [{**row, "quality_status": status.get(row["motion_id"], row["quality_status"])}
                         for row in self.rows]
        self.categories = tuple(dict.fromkeys(row["motion_category"] for row in self.rows))
        # Collision primitives are the official flat model's body/leg geometry.
        # Use the existing wheel visual meshes to avoid rendering overlapping copies.
        robot.model.geom_rgba[robot.wheel_geoms] = 0
        for geom in robot.ground_geoms:
            name = robot.model.geom(int(geom)).name
            rgb = (.20, .25, .31) if "base_" in name else (.68, .74, .80)
            robot.model.geom_rgba[geom] = (*rgb, 1.)
        robot.model.geom_rgba[robot.floor] = (.18, .21, .23, 1.)
        self.overlays = set(OVERLAYS)
        self.category, self.quality = category, quality
        self.paused, self.loop, self.show_help = False, True, False
        self.speed = 1.
        self.filter(motion)

    def filter(self, motion=None):
        self.matches = [
            row for row in self.rows if
            (self.category is None or row["motion_category"] == self.category)
            and (self.quality is None or row["quality_status"] == self.quality)
            and (self.dataset / row["file_path"]).is_file()
        ]
        if not self.matches:
            raise ValueError("No playable clips match those filters")
        index = next((i for i, row in enumerate(self.matches) if row["motion_id"] == motion), None)
        if motion is not None and index is None:
            raise ValueError(f"Motion {motion} is not in this selection")
        self.select(0 if index is None else index)

    def cycle_filter(self, field, values):
        options = (None, *values)
        start = options.index(getattr(self, field))
        keys = {"category": "motion_category", "quality": "quality_status"}
        other = "quality" if field == "category" else "category"
        for offset in range(1, len(options) + 1):
            value = options[(start + offset) % len(options)]
            other_value = getattr(self, other)
            if any((value is None or row[keys[field]] == value)
                   and (other_value is None or row[keys[other]] == other_value)
                   and (self.dataset / row["file_path"]).is_file() for row in self.rows):
                setattr(self, field, value)
                self.filter()
                return

    def select(self, index):
        self.index = index % len(self.matches)
        self.row = self.matches[self.index]
        path = (self.dataset / self.row["file_path"]).resolve()
        if not path.is_relative_to(self.dataset):
            raise ValueError("Motion path escaped the dataset")
        self.clip = load_clip(path)
        quat = self.clip["base_quat"]
        self.yaw = np.unwrap(2*np.arctan2(quat[:, 3], quat[:, 0]))
        self.reference = heading_velocity(
            self.clip["base_lin_vel"], self.clip["base_ang_vel"], rotations(self.yaw))
        self.reset()

    def reset(self):
        self.cursor = 0.
        self.frame = 0

    def step(self, frames):
        self.paused = True
        self.frame = int(np.clip(self.frame + frames, 0, len(self.yaw)-1))
        self.cursor = self.frame * self.dt

    def advance(self, seconds):
        if self.paused:
            return
        self.cursor += seconds * self.speed
        if self.cursor >= len(self.yaw) * self.dt:
            if self.loop:
                self.reset()
            else:
                self.step(len(self.yaw))
        self.frame = min(int(self.cursor / self.dt + 1e-8), len(self.yaw)-1)

    def show_frame(self):
        clip, i = self.clip, self.frame
        self.robot.set_pose(clip["base_pos"][i], clip["base_quat"][i], clip["leg_q"][i], clip["wheel_q"][i])

    def status(self):
        c, i = self.clip, self.frame
        def number(values):
            return " ".join(f"{v:+.3f}" for v in values)
        return (
            ("Motion", self.row["motion_id"]),
            ("Category / quality", f"{self.row['motion_category']} / {self.row['quality_status']}"),
            ("Selection", f"{self.index+1}/{len(self.matches)} | {self.category or 'all'} | {self.quality or 'all'}"),
            ("Frame / time", f"{i}/{len(self.yaw)-1}  {c['time'][i]:.2f} s  "
                             f"{self.speed:g}x  {'paused' if self.paused else 'playing'}"),
            ("Nominal vx vy wz", number(c["command_nominal"])),
            ("Reference vx vy wz", number(self.reference[i])),
            ("Base xyz / yaw", f"{number(c['base_pos'][i])} m  {self.yaw[i]:+.3f} rad"),
            ("CoM xyz", number(c["com_pos"][i]) + " m"),
            ("CoM support margin", f"{c['support_margin'][i]*1000:+.1f} mm"),
            ("FL FR RL RR", " ".join(MODES.get(mode, "?") for mode in c["motion_phase"][i])),
            ("IK error (mm)", number(1000*c["ik_error"][i])),
            ("Rolling / lateral (m/s)", number(c["rolling_residual"][i]) + " / " + number(c["lateral_residual"][i])),
        )


def add_geom(scene, kind, position, color, size):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo overlay scene capacity exceeded")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(position, dtype=float),
                        FLAT, np.asarray(color, dtype=np.float32))
    scene.ngeom += 1
    return geom


def line(scene, start, end, color, width=2.):
    if np.linalg.norm(end-start) < 1e-6:
        return
    geom = add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, start, color, [0, 0, 0])
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)


def sphere(scene, point, color, radius):
    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, point, color, [radius]*3)


def path_trace(scene, points, color, width=2., dashed=False):
    if len(points) < 2:
        return
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        if not dashed or i % 2 == 0:
            line(scene, a, b, color, width)


def draw(scene, playback, offset=0):
    scene.ngeom = offset
    c, i = playback.clip, playback.frame
    options = playback.overlays
    # Show history and a short preview in the moving camera.
    lo, hi = max(0, i-85), min(len(playback.yaw), i+45)
    segment = slice(lo, hi, 3)
    if "base" in options:
        path_trace(scene, c["base_pos"][segment], BASE_COLOR, 2.5)
        location = c["base_pos"][i] + [0, 0, .07]
        end = location + .17*np.array([np.cos(playback.yaw[i]), np.sin(playback.yaw[i]), 0])
        line(scene, location, end, BASE_COLOR, 4)
    if "com" in options:
        path_trace(scene, c["com_pos"][segment], COM_COLOR, 2)
        sphere(scene, c["com_pos"][i], COM_COLOR, .018)
    if "support" in options:
        support = hull(c["contact_point"][i, c["contact_mask"][i], :2])
        if len(support) >= 3:
            vertices = np.column_stack((support, np.full(len(support), .005)))
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                line(scene, a, b, SUPPORT_COLOR, 3)
            center = vertices.mean(axis=0)
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                for u in np.linspace(0, 1, 16):
                    line(scene, center, (1-u)*a + u*b, (*SUPPORT_COLOR[:3], .22), 2)
        sphere(scene, np.r_[c["com_pos"][i, :2], .013], COM_COLOR, .014)

    d = playback.robot.data
    for wheel, color in enumerate(COLORS):
        if "target" in options:
            path_trace(scene, c["wheel_center_target"][segment, wheel], (*color[:3], .9), 2.8, dashed=True)
        if "fk" in options:
            path_trace(scene, c["wheel_center_fk"][segment, wheel], (*color[:3], .55), 1.5)
        if "contacts" in options:
            grounded = c["contact_mask"][i, wheel]
            sphere(scene, c["contact_point"][i, wheel], color, .016 if grounded else .023)
        # A rim stripe attached to the measured wheel-body frame exposes spin
        # even when the tire mesh is viewed in silhouette.
        body = playback.robot.wheel_bodies[wheel]
        side = 1 if LEGS[wheel].endswith("L") else -1
        wheel_frame = d.xmat[body].reshape(3, 3)
        axis = wheel_frame[:, 1]
        radial = wheel_frame[:, 2]
        hub = d.xpos[body] + side*.078*axis
        line(scene, hub + .022*radial, hub + .076*radial, color, 3)


def camera_setup(camera, playback):
    camera.distance = 1.55
    camera.azimuth = 128.
    camera.elevation = -26.
    camera.lookat[:] = playback.clip["base_pos"][playback.frame] - [0, 0, .16]


def live(playback):
    import glfw
    import mujoco.viewer

    pending = SimpleQueue()
    playback.show_frame()
    help_text = ("Space pause   Left/Right frame   N/P clips   C category   Q quality\n"
                 "+/- speed   L loop   Home reset   1-6 overlays   H help   Esc quit")
    with mujoco.viewer.launch_passive(playback.robot.model, playback.robot.data,
                                      key_callback=pending.put, show_left_ui=False, show_right_ui=False) as handle:
        camera_setup(handle.cam, playback)
        handle.opt.geomgroup[3] = 1
        last = perf_counter()
        while handle.is_running():
            while not pending.empty():
                key = pending.get()
                if key == glfw.KEY_ESCAPE:
                    return
                if key == glfw.KEY_SPACE:
                    playback.paused = not playback.paused
                elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                    playback.step(1 if key == glfw.KEY_RIGHT else -1)
                elif key in (glfw.KEY_N, glfw.KEY_P):
                    playback.select(playback.index + (1 if key == glfw.KEY_N else -1))
                elif key == glfw.KEY_C:
                    playback.cycle_filter("category", playback.categories)
                elif key == glfw.KEY_Q:
                    playback.cycle_filter("quality", QUALITIES)
                elif key == glfw.KEY_HOME:
                    playback.reset()
                elif key == glfw.KEY_L:
                    playback.loop = not playback.loop
                elif key in (glfw.KEY_EQUAL, glfw.KEY_MINUS):
                    playback.speed = float(np.clip(playback.speed * (2 if key == glfw.KEY_EQUAL else .5), .125, 4))
                elif key == glfw.KEY_H:
                    playback.show_help = not playback.show_help
                elif glfw.KEY_1 <= key <= glfw.KEY_6:
                    playback.overlays.symmetric_difference_update({OVERLAYS[key - glfw.KEY_1]})
            now = perf_counter()
            with handle.lock():
                playback.advance(now-last)
                playback.show_frame()
                handle.cam.lookat[:] = playback.clip["base_pos"][playback.frame] - [0, 0, .16]
                draw(handle.user_scn, playback)
            labels, values = map("\n".join, zip(*playback.status()))
            handle.set_texts([
                (100, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values),
                (100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 help_text if playback.show_help else "H for controls", "Loop on" if playback.loop else "Loop off"),
            ])
            handle.sync(state_only=True)
            last = now
            sleep(.01)


def render_frames(playback, frames: Iterable[int], output):
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera_setup(camera, playback)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    model = playback.robot.model
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 600
    with mujoco.Renderer(model, width=960, height=600, max_geom=1500) as renderer:
        for frame in frames:
            if not 0 <= frame < len(playback.yaw):
                raise ValueError(f"Frame {frame} outside the clip")
            playback.frame = frame
            playback.show_frame()
            camera.lookat[:] = playback.clip["base_pos"][frame] - [0, 0, .16]
            renderer.update_scene(playback.robot.data, camera, scene_option=option)
            draw(renderer.scene, playback, renderer.scene.ngeom)
            image = Image.fromarray(renderer.render())
            labels = [f"{label}   {value}" for label, value in playback.status()]
            ImageDraw.Draw(image).multiline_text((13, 13), "\n".join(labels), fill="white",
                                                  stroke_width=1, stroke_fill="#182028", spacing=3)
            image.save(output / f"{playback.row['motion_id']}_{frame:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--category")
    parser.add_argument("--quality", choices=QUALITIES)
    parser.add_argument("--motion", help="Start with this motion ID")
    parser.add_argument("--render-dir", type=Path, help="Save PNG frames using offscreen MuJoCo")
    parser.add_argument("--frames", type=int, nargs="+", default=[100])
    args = parser.parse_args()
    try:
        playback = Playback(args.dataset, args.category, args.quality, args.motion)
        if args.render_dir:
            render_frames(playback, args.frames, args.render_dir)
        else:
            live(playback)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
