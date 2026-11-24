"""Offline Go2W kinematic references on a flat MuJoCo plane."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src/unisupermimic/assets/robots/go2w/flat.xml"
TRAIN = ROOT / "src/unisupermimic/conf/supermimic/task/go2w_velocity/train.yaml"
CONFIG = Path(__file__).with_name("config.yaml")
LEGS = ("FL", "FR", "RL", "RR")
STANCE, ROLLING, SWING = range(3)

# Shape (T is frame count), units, coordinate system and provenance.
FIELDS = {
    "time": (("T",), "s", "clip clock"),
    "command_nominal": ((3,), "m/s, m/s, rad/s", "requested heading vx, vy, wz"),
    "command": (("T", 3), "m/s, m/s, rad/s", "planned heading vx, vy, wz"),
    "base_pos_nominal": (("T", 3), "m", "world, integrated command"),
    "base_pos": (("T", 3), "m", "world, corrected reference"),
    "base_quat": (("T", 4), "1", "world-from-base quaternion wxyz"),
    "base_lin_vel": (("T", 3), "m/s", "world, derivative of reference position"),
    "base_ang_vel": (("T", 3), "rad/s", "world, derivative of reference orientation"),
    "com_pos": (("T", 3), "m", "world, FK subtree CoM"),
    "leg_q": (("T", 12), "rad", "logical joint order, IK state"),
    "leg_qd": (("T", 12), "rad/s", "logical joint order, derivative of IK state"),
    "wheel_q": (("T", 4), "rad", "logical wheel order, unwrapped kinematic integral"),
    "wheel_qd": (("T", 4), "rad/s", "logical wheel order, kinematic rolling estimate"),
    "wheel_center_target": (("T", 4, 3), "m", "world, planned"),
    "wheel_center_fk": (("T", 4, 3), "m", "world, rebuilt from reference pose"),
    "contact_point": (("T", 4, 3), "m", "world, lowest wheel mesh point, including swing"),
    "contact_mask": (("T", 4), "bool", "planned support"),
    "wheel_rate_valid": (("T", 4), "bool", "support, within ground tolerance and valid IK"),
    "motion_phase": (("T", 4), "enum", "planned STANCE=0, ROLLING=1, SWING=2"),
    "ik_failed": (("T", 4), "bool", "analytic workspace/limit solver failure"),
    "ik_error": (("T", 4), "m", "target to FK distance"),
    "rolling_residual": (("T", 4), "m/s", "FK contact velocity along rolling direction"),
    "lateral_residual": (("T", 4), "m/s", "FK contact velocity along horizontal wheel axis"),
    "support_margin": (("T",), "m", "signed CoM projection to support polygon edges"),
}
