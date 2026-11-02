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
    effort_limits: tuple[float, ...] = ()
    actuator_model_version: str = "mjlab_builtin_position_pd_v1"
    default_pose_profile: str = "default_v1"
    action_contract_version: str = "normalized_position_v1"
    position_target_bounds: tuple[tuple[float, float], ...] = ()

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
        if self.effort_limits and len(self.effort_limits) != n:
            raise ValueError(f"{self.name}: effort_limits has {len(self.effort_limits)} entries; expected {n}")
        if self.position_target_bounds and len(self.position_target_bounds) != n:
            raise ValueError(f"{self.name}: position_target_bounds must have {n} entries")
        if self.position_target_bounds:
            for i, (mode, bounds, default) in enumerate(zip(
                    self.action_modes, self.position_target_bounds, self.default_joint_pos)):
                lo, hi = bounds
                if lo > hi:
                    raise ValueError(f"{self.name}: target bounds are inverted for {self.joint_names[i]}")
                if mode == "position" and not lo <= default <= hi:
                    raise ValueError(f"{self.name}: default pose is outside target bounds for {self.joint_names[i]}")

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
M1_CARRY_DEFAULT = (
    0.0, 1.1, -2.0, 0.0,
    0.0, 1.1, -2.0, 0.0,
    0.0, -1.1, 2.0, 0.0,
    0.0, -1.1, 2.0, 0.0,
    0.0, 0.50, -0.40, 0.30, 0.0, 0.0,
)
# Absolute radians from AMP Bank v2.5 reference p1/p99 plus a 0.05 rad
# margin, clipped to the URDF 90% soft limits and Piper SDK limits. Wheel
# slots use explicit unused [0,0] sentinels.
M1_POSITION_TARGET_BOUNDS_V3_BY_NAME = {
    "FAR_ABAD_JOINT": (-0.31298, 0.36099),
    "FAR_HIP_JOINT": (0.53814, 1.56619),
    "FAR_KNEE_JOINT": (-2.52090, -1.66946),
    "FBL_ABAD_JOINT": (-0.35488, 0.31071),
    "FBL_HIP_JOINT": (0.56885, 1.58435),
    "FBL_KNEE_JOINT": (-2.52090, -1.68615),
    "RAR_ABAD_JOINT": (-0.30770, 0.34571),
    "RAR_HIP_JOINT": (-1.57743, -0.61997),
    "RAR_KNEE_JOINT": (1.57674, 2.52090),
    "RBL_ABAD_JOINT": (-0.37172, 0.30733),
    "RBL_HIP_JOINT": (-1.59972, -0.60404),
    "RBL_KNEE_JOINT": (1.59689, 2.52090),
    "piper_x_joint1": (-0.53457, 0.61986),
    "piper_x_joint2": (0.15700, 2.84280),
    "piper_x_joint3": (-1.88899, -0.30207),
    "piper_x_joint4": (-0.41415, 0.60037),
    "piper_x_joint5": (-0.36002, 0.29504),
    # q6 has no motion support in this bank; this is a conservative bounded
    # control envelope, not a claim that the prior learned q6 motion.
    "piper_x_joint6": (-0.05, 0.05),
}
M1_ACTION_SCALES = tuple(
    5.0 if n.endswith("FOOT_JOINT") else
    0.35 if n in ("piper_x_joint2", "piper_x_joint4") else
    0.25
    for n in M1_JOINTS
)


def get_robot_spec(name: str, *, m1_pose_profile: str | None = None,
                   m1_action_contract: str = "v2") -> RobotSpec:
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
            effort_limits=(23.7, 23.7, 35.55) * 4,
            actuator_model_version="go2_urdf_builtin_pd_kp40_kd1_effort_23p7_35p55_v2",
        )
    if key in ("m1", "m1_piper", "m1piper"):
        pose_profile = m1_pose_profile or "legacy_manipulation"
        if pose_profile not in ("legacy_manipulation", "dataset_carry"):
            raise ValueError("M1 pose profile must be legacy_manipulation or dataset_carry")
        if m1_action_contract not in ("v2", "v3"):
            raise ValueError("M1 action contract must be v2 or v3")
        defaults = M1_DEFAULT if pose_profile == "legacy_manipulation" else M1_CARRY_DEFAULT
        action_modes = tuple(
            "velocity" if n in M1_LEG_JOINTS and n.endswith("FOOT_JOINT") else "position"
            for n in M1_JOINTS
        )
        scales = M1_ACTION_SCALES
        target_bounds = ()
        actuator_version = "m1_urdf_builtin_pd_v2"
        contract_version = "m1_mixed_action_v2"
        if m1_action_contract == "v3":
            # Position scales are unused by v3; retaining zeros avoids implying
            # the legacy symmetric mapping still controls physical targets.
            scales = tuple(5.0 if mode == "velocity" else 0.0 for mode in action_modes)
            target_bounds = tuple(M1_POSITION_TARGET_BOUNDS_V3_BY_NAME.get(n, (0.0, 0.0)) for n in M1_JOINTS)
            for joint_name, q0, bounds in zip(M1_JOINTS, defaults, target_bounds):
                if joint_name in M1_POSITION_TARGET_BOUNDS_V3_BY_NAME and not bounds[0] <= q0 <= bounds[1]:
                    raise ValueError(f"M1 pose {pose_profile} default for {joint_name} is outside v3 target bounds")
            actuator_version = f"m1_urdf_builtin_pd_action_contract_v3_{pose_profile}_v1"
            contract_version = "m1_mixed_action_v3"
        return RobotSpec(
            name="m1_piper", joint_names=M1_JOINTS, default_joint_pos=defaults,
            action_modes=action_modes,
            action_scales=scales,
            kp=tuple(0.0 if n.endswith("FOOT_JOINT") else (32.0 if n.startswith("piper_") else 250.0) for n in M1_JOINTS),
            kd=tuple(0.5 if n.endswith("FOOT_JOINT") else (1.0 if n.startswith("piper_") else 9.0) for n in M1_JOINTS),
            default_base_height=0.3892858, control_dt=0.02,
            tracked_body_names=("BASE_LINK", "FAR_ABAD_LINK", "FAR_HIP_LINK", "FAR_KNEE_LINK", "FBL_ABAD_LINK", "FBL_HIP_LINK", "FBL_KNEE_LINK", "RAR_ABAD_LINK", "RAR_HIP_LINK", "RAR_KNEE_LINK", "RBL_ABAD_LINK", "RBL_HIP_LINK", "RBL_KNEE_LINK", "piper_x_link7"),
            root_body_name="BASE_LINK", anchor_body_name="BASE_LINK",
            model_path=Path("/home/yifeiy/AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf"),
            wheel_joint_names=tuple(n for n in M1_LEG_JOINTS if n.endswith("FOOT_JOINT")),
            amp_joint_names=M1_AMP_JOINTS, foot_geom_regex=r".*FOOT.*",
            actuator_model_version=actuator_version,
            default_pose_profile=("m1_pose_dataset_carry_v1" if pose_profile == "dataset_carry"
                                  else "m1_pose_legacy_manipulation_v1"),
            action_contract_version=contract_version,
            position_target_bounds=target_bounds,
        )
    raise ValueError(f"unknown robot {name!r}; choose g1, go2, or m1_piper")
