"""Small, explicit robot contracts shared by the simulator and learning stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_modes: tuple[str, ...]  # "position" or "velocity", in policy order.
    action_scales: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    default_base_height: float
    control_dt: float
    tracked_body_names: tuple[str, ...]
    root_body_name: str
    anchor_body_name: str
    model_path: Path | None = None
    wheel_joint_names: tuple[str, ...] = ()
    amp_joint_names: tuple[str, ...] = ()
    foot_geom_regex: str = ".*"

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        for label, values in (
            ("default_joint_pos", self.default_joint_pos),
            ("action_modes", self.action_modes),
            ("action_scales", self.action_scales),
            ("kp", self.kp),
            ("kd", self.kd),
        ):
            if len(values) != n:
                raise ValueError(f"{self.name}: {label} has {len(values)} entries; expected {n}")
        if len(set(self.joint_names)) != n:
            raise ValueError(f"{self.name}: joint_names contains duplicates")
        if any(m not in ("position", "velocity") for m in self.action_modes):
            raise ValueError(f"{self.name}: action_modes entries must be position or velocity")
        if not set(self.wheel_joint_names).issubset(self.joint_names):
            raise ValueError(f"{self.name}: wheel joints are missing from action joint list")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def delta_dim(self) -> int:
        return 12 + 2 * self.num_joints

    @property
    def obs_dim(self) -> int:
        return 3 * self.num_joints + 6

    @property
    def amp_dim(self) -> int:
        return len(self.amp_joint_names)


GO2_JOINTS = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
GO2_DEFAULT = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)

M1_LEG_JOINTS = (
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT", "FAR_FOOT_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT", "FBL_FOOT_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT", "RAR_FOOT_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT", "RBL_FOOT_JOINT",
)
M1_ARM_JOINTS = tuple(f"piper_x_joint{i}" for i in range(1, 7))
M1_JOINTS = M1_LEG_JOINTS + M1_ARM_JOINTS
M1_AMP_JOINTS = tuple(
    n for n in M1_LEG_JOINTS if not n.endswith("FOOT_JOINT")
) + M1_ARM_JOINTS
M1_DEFAULT = (
    0.0, 1.1, -2.0, 0.0,
    0.0, 1.1, -2.0, 0.0,
    0.0, -1.1, 2.0, 0.0,
    0.0, -1.1, 2.0, 0.0,
    0.0, 2.55, -1.50, -0.80, 0.0, 0.0,
)


def get_robot_spec(name: str) -> RobotSpec:
    key = name.lower().replace("-", "_")
    if key == "g1":
        # Keep this definition aligned with mjlab's G1 asset-zoo actuator order.
        from supermimic.data.motion_convert import G1_JOINT_NAMES
        g1_default = tuple(
            -0.312 if "hip_pitch" in n else
            0.669 if "knee" in n else
            -0.363 if "ankle_pitch" in n else
            0.6 if "elbow" in n else
            0.2 if n == "left_shoulder_roll_joint" else
            -0.2 if n == "right_shoulder_roll_joint" else
            0.2 if "shoulder_pitch" in n else 0.0
            for n in G1_JOINT_NAMES
        )
        tracked = (
            "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
            "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
        )

        return RobotSpec(
            name="g1", joint_names=G1_JOINT_NAMES,
            default_joint_pos=g1_default,
            action_modes=("position",) * 29, action_scales=(1.0,) * 29,
            kp=(1.0,) * 29, kd=(1.0,) * 29,
            default_base_height=0.76, control_dt=0.02,
            tracked_body_names=tracked, root_body_name="pelvis",
            anchor_body_name="torso_link", foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
            amp_joint_names=G1_JOINT_NAMES,
        )
    if key == "go2":
        return RobotSpec(
            name="go2", joint_names=GO2_JOINTS, default_joint_pos=GO2_DEFAULT,
            action_modes=("position",) * 12, action_scales=(0.25,) * 12,
            kp=(40.0,) * 12, kd=(1.0,) * 12,
            default_base_height=0.42, control_dt=0.02,
            tracked_body_names=("base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            root_body_name="base", anchor_body_name="base",
            model_path=Path("/home/yifeiy/ws_parkour/lidar_locomotion/legged_gym/resources/robots/go2/urdf/go2.urdf"),
            foot_geom_regex=r"^(FL|FR|RL|RR)_foot_collision$", amp_joint_names=(),
        )
    if key in ("m1", "m1_piper", "m1piper"):
        return RobotSpec(
            name="m1_piper", joint_names=M1_JOINTS, default_joint_pos=M1_DEFAULT,
            action_modes=tuple("velocity" if n in M1_LEG_JOINTS and n.endswith("FOOT_JOINT") else "position" for n in M1_JOINTS),
            action_scales=tuple(5.0 if n.endswith("FOOT_JOINT") else 0.25 for n in M1_JOINTS),
            kp=tuple(0.0 if n.endswith("FOOT_JOINT") else (32.0 if n.startswith("piper_") else 250.0) for n in M1_JOINTS),
            kd=tuple(0.5 if n.endswith("FOOT_JOINT") else (1.0 if n.startswith("piper_") else 9.0) for n in M1_JOINTS),
            default_base_height=0.3892858, control_dt=0.02,
            tracked_body_names=("BASE_LINK", "FAR_ABAD_LINK", "FAR_HIP_LINK", "FAR_KNEE_LINK", "FBL_ABAD_LINK", "FBL_HIP_LINK", "FBL_KNEE_LINK", "RAR_ABAD_LINK", "RAR_HIP_LINK", "RAR_KNEE_LINK", "RBL_ABAD_LINK", "RBL_HIP_LINK", "RBL_KNEE_LINK", "piper_x_link7"),
            root_body_name="BASE_LINK", anchor_body_name="BASE_LINK",
            model_path=Path("/home/yifeiy/AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf"),
            wheel_joint_names=tuple(n for n in M1_LEG_JOINTS if n.endswith("FOOT_JOINT")),
            amp_joint_names=M1_AMP_JOINTS, foot_geom_regex=r".*FOOT.*",
        )
    raise ValueError(f"unknown robot {name!r}; choose g1, go2, or m1_piper")
