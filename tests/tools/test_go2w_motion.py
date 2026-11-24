from collections import Counter

import mujoco
import numpy as np
import pytest
import yaml
from tools.go2w_motion import CONFIG, DIAGONALS, LEGS, MODEL, ROLLING, SWING
from tools.go2w_motion.generate import commands, generate_dataset, synthesize
from tools.go2w_motion.kinematics import IDENTITY, Robot, support_geometry
from tools.go2w_motion.quality_check import audit_dataset, check_clip, load_clip
from tools.go2w_motion.trajectories import (
    integrate,
    smootherstep,
    swing,
    swing_bump,
    trapezoid_integral,
)
from tools.go2w_motion.viewer import Playback


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def robot():
    return Robot()


@pytest.mark.parametrize("vx,vy,wz", [(0.4, 0, 0), (-0.3, 0, 0), (0, 0, -0.7), (0.4, -0.2, 0.7)])
def test_integrate_heading_twist(vx, vy, wz):
    dt, duration = .02, 2.
    trace = np.tile([vx, vy, wz], (101, 1))
    xyz, yaw = integrate(trace, dt, .4)
    theta = wz*duration
    expected = np.array([vx, vy])*duration
    if wz:
        expected = np.array([vx*np.sin(theta) + vy*(np.cos(theta)-1),
                             vx*(1-np.cos(theta)) + vy*np.sin(theta)])/wz
    np.testing.assert_allclose(xyz[-1], [*expected, .4], atol=1e-12)
    assert yaw[-1] == pytest.approx(theta)


def test_exact_unique_balanced_command_grid(config):
    jobs = list(commands(config))
    assert len({job[0] for job in jobs}) == len({job[2] for job in jobs}) == len(jobs) == 100
    assert Counter(group for _, group, _ in jobs) == {
        "forward_backward": 22, "yaw": 16, "lateral": 14,
        "forward_yaw": 16, "forward_lateral": 16, "lateral_yaw": 16,
    }
    by_category = {(g, c) for _, g, c in jobs}
    for _, group, (vx, vy, wz) in jobs:
        assert (group, (vx, -vy, -wz)) in by_category
    assert {(np.sign(vy), np.sign(wz)) for _, g, (_, vy, wz) in jobs if g == "lateral_yaw"} == {
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    }


def test_c2_endpoints_and_nonzero_rolling_derivatives():
    t = .24
    positions = (np.array([0.1, 0.2]), np.array([0.2, 0.4]))
    velocities = (np.array([.21, .02]), np.array([.35, -.04]))
    accelerations = (np.array([.03, -.01]), np.array([-.08, .05]))

    def evaluate(u):
        return swing(*positions, *velocities, *accelerations, t, u)

    h = 1e-4
    for i, sign in ((0, 1), (1, -1)):
        a, b, c, d = (evaluate(i + sign*k*h) for k in range(4))
        np.testing.assert_allclose(a, positions[i], atol=1e-12)
        np.testing.assert_allclose(sign*(-3*a+4*b-c)/(2*h*t), velocities[i], atol=1e-5)
        np.testing.assert_allclose((2*a-5*b+4*c-d)/(h*t)**2, accelerations[i], atol=5e-5)
        for func, value in ((smootherstep, i), (swing_bump, 0)):
            a, b, c, d = (func(i + sign*k*h) for k in range(4))
            assert a == pytest.approx(value)
            assert sign*(-3*a+4*b-c)/(2*h) == pytest.approx(0, abs=2e-6)
            assert (2*a-5*b+4*c-d)/h**2 == pytest.approx(0, abs=2e-4)
    assert swing_bump(.5) == 1.


def test_named_mapping_matches_terrain_robot(robot):
    m = robot.model
    expected = tuple(f"robot/{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
    expected += tuple(f"robot/{leg}_wheel_joint" for leg in LEGS)
    assert robot.joint_names == expected
    np.testing.assert_array_equal(m.actuator_trnid[robot.actuators, 0], robot.joints)
    assert list(robot.actuators) != sorted(robot.actuators)
    assert robot.qpos[3] != robot.qpos[0] + 3
    terrain = mujoco.MjModel.from_xml_path(str(MODEL.with_name("terrain_robot.xml")))
    for name in expected:
        np.testing.assert_allclose(m.joint(name).axis, terrain.joint(name).axis)
        np.testing.assert_allclose(m.joint(name).range, terrain.joint(name).range)
    np.testing.assert_allclose(robot.default_q, [.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5])
    np.testing.assert_allclose(robot.action_limits, robot.soft)
    assert robot.wheel_rate_limit == 15.


def test_ik_roundtrip_via_mujoco_and_unreachable(robot):
    rng = np.random.default_rng(42)
    for _ in range(8):
        q = robot.stand_q + rng.uniform(-.12, .12, 12)
        robot.set_pose(np.zeros(3), IDENTITY, q)
        centers = robot.data.xpos[robot.wheel_bodies].copy()
        solved, valid = robot.inverse(centers)
        assert valid.all()
        np.testing.assert_allclose(solved.ravel(), q, atol=1e-12)
    impossible = centers.copy()
    impossible[0] += [0, 0, -1]
    _, valid = robot.inverse(impossible)
    assert valid.tolist() == [False, True, True, True]


def test_support_geometry_and_wheel_stand(robot):
    ground = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    assert support_geometry(ground, [.3, 0.])[1] == pytest.approx(.7)
    assert support_geometry(ground[:3], [0., 1.])[1] == pytest.approx(-1/np.sqrt(2))
    assert support_geometry(ground[[0, 2]], [0., 1.])[1] == pytest.approx(-1/np.sqrt(2))
    robot.set_pose(np.array([0., 0., robot.stand_height]), IDENTITY, robot.stand_q)
    assert np.max(np.abs(robot.contact_points()[:, 2])) < 1e-4
    assert robot.spawn_clearance.min() > .025
    assert not robot.collisions(.003)


@pytest.mark.parametrize("speed", [.30, -.20])
def test_wheel_sign_integration_and_four_wheel_rolling(robot, config, speed):
    nominal = (speed, 0., 0.)
    clip = synthesize(robot, nominal, config, config["planner"]["swing_candidates"][0])
    assert np.all(clip["motion_phase"] == ROLLING)
    assert clip["wheel_rate_valid"].all()
    np.testing.assert_allclose(clip["wheel_qd"][100]*robot.wheel_radius, speed, atol=.005)
    np.testing.assert_allclose(clip["wheel_q"], trapezoid_integral(clip["wheel_qd"], config["dt"]))
    assert check_clip(clip, nominal, robot, config)["quality_status"] == "qualified"


@pytest.mark.parametrize("nominal", [(0., .10, 0.), (0., 0., .30), (0., 0., -.30)])
def test_diagonal_swings_alternate_and_stay_candidates(robot, config, nominal):
    clip = synthesize(robot, nominal, config, config["planner"]["diagonal_swing_candidates"][0], gait="diagonal")
    swing_mask = clip["motion_phase"] == SWING
    start = np.flatnonzero(np.diff(np.r_[False, swing_mask.any(axis=1)].astype(int)) == 1)
    pairs = [tuple(np.flatnonzero(swing_mask[i])) for i in start]
    assert len(pairs) >= 2 and all(pair in DIAGONALS for pair in pairs)
    assert all(a != b for a, b in zip(pairs, pairs[1:]))
    assert not clip["wheel_rate_valid"][swing_mask].any()
    q = config["quality"]
    assert np.max(np.abs(clip["contact_point"][~swing_mask][:, 2])) < q["contact_height"]
    assert clip["contact_point"][swing_mask][:, 2].max() >= q["swing_min_peak_height"]
    good = check_clip(clip, nominal, robot, config, gait="diagonal")
    assert good["quality_status"] == "candidate"
    assert good["failure_reasons"] == ["dynamic_diagonal_support"]
    assert check_clip(clip, nominal, robot, config)["quality_status"] == "rejected"
    spoofed = {**clip, "command": clip["command"]*.8}
    assert check_clip(spoofed, nominal, robot, config, gait="diagonal")["quality_status"] == "rejected"


def test_disk_audit_playback_and_refusal_to_overwrite(tmp_path):
    path = tmp_path / "bank"
    summary = generate_dataset(path, category="standing")
    assert summary["categories"]["standing"]["qualified"] == 1
    clip_path = path / "motions" / "standing" / "standing.npz"
    persisted = load_clip(clip_path)
    playback = Playback(path, quality="qualified")
    assert persisted.keys() == playback.clip.keys()
    for name in persisted:
        np.testing.assert_array_equal(persisted[name], playback.clip[name])
    playback.step(1)
    assert playback.frame == 1 and playback.paused
    playback.paused = False
    playback.advance(10.)
    playback.show_frame()
    assert playback.frame == 0
    np.testing.assert_allclose(playback.robot.data.qpos[playback.robot.qpos[12:]], 0)
    with pytest.raises(FileExistsError):
        generate_dataset(path, category="standing")
    persisted["base_quat"][10] = np.nan
    np.savez_compressed(clip_path, **persisted)
    corrupted = audit_dataset(path)
    assert corrupted["categories"]["standing"]["rejected"] == 1
    assert corrupted["manifest_disagreements"] == ["standing"]
