"""MuJoCo/mjlab model adapters for the two non-zoo robots."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from supermimic.robots.spec import M1_ARM_JOINTS, M1_LEG_JOINTS, M1_JOINTS, get_robot_spec


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> None:
    """The source Go2 URDF leaves collision geom names blank; name by owning link."""
    for body in spec.bodies:
        for i, geom in enumerate(body.geoms):
            if not geom.name:
                suffix = "_collision" if len(body.geoms) == 1 else f"_collision_{i}"
                geom.name = f"{body.name}{suffix}"


def _m1_urdf_spec(path: Path) -> mujoco.MjSpec:
    """Read the source M1 URDF without editing it; MuJoCo has no DAE decoder.

    The vendor package also ships STL equivalents for the six moving Piper links.
    The fixed gripper meshes have no STL counterpart, so their collision meshes are
    omitted; their URDF inertial elements and the articulated arm remain intact.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for collision in list(link.findall("collision")):
            mesh = collision.find("./geometry/mesh")
            if mesh is not None and Path(mesh.attrib["filename"]).suffix.lower() == ".dae":
                if Path(mesh.attrib["filename"]).stem in {"J7", "J8", "gripper_base_100_v2"}:
                    link.remove(collision)
    for mesh in root.iter("mesh"):
        relative = Path(mesh.attrib["filename"])
        if relative.suffix.lower() == ".dae":
            stem = relative.stem
            if stem.startswith("J") and stem[1:].isdigit():
                stem = f"Link{stem[1:]}"
            relative = relative.parent.parent / f"{stem}.STL"
        mesh.attrib["filename"] = str((path.parent / relative).resolve())
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="wb") as tmp:
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        tmp.flush()
        # MjSpec loads mesh assets at parse time, so it remains compilable after the
        # temporary descriptor is removed.
        spec = mujoco.MjSpec.from_file(tmp.name)
    root_body = spec.body("BASE_LINK")
    if root_body is None:
        raise ValueError(f"{path}: expected BASE_LINK root body")
    _name_unnamed_geoms(spec)
    # Keep fixed gripper links (notably link7/TCP) addressable for FK and validation.
    spec.compiler.fusestatic = False
    root_body.add_freejoint(name="root")
    return spec


def load_robot_spec_model(name: str) -> mujoco.MjModel:
    """Compile an inspectable MuJoCo model and verify its floating-base joint map."""
    robot = get_robot_spec(name)
    if robot.name == "g1":
        from supermimic.data.motion_convert import load_g1_model

        model = load_g1_model()
    elif robot.name == "go2":
        spec = mujoco.MjSpec.from_file(str(robot.model_path))
        _name_unnamed_geoms(spec)
        # Preserve the fixed foot links so reward/FK code can address the actual feet.
        spec.compiler.fusestatic = False
        root = spec.body(robot.root_body_name)
        if root is None:
            raise ValueError(f"{robot.name}: no root body {robot.root_body_name!r}")
        root.add_freejoint(name="root")
        model = spec.compile()
    else:
        model = _m1_urdf_spec(robot.model_path).compile()

    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    )
    if model.nq != 7 + robot.num_joints or model.nv != 6 + robot.num_joints:
        raise ValueError(f"{robot.name}: compiled nq/nv={model.nq}/{model.nv}, expected {7 + robot.num_joints}/{6 + robot.num_joints}")
    if set(names) != set(robot.joint_names):
        raise ValueError(f"{robot.name}: URDF joints differ; missing={set(robot.joint_names)-set(names)}, extra={set(names)-set(robot.joint_names)}")
    return model


def get_robot_entity_cfg(name: str):
    """Build a mjlab EntityCfg with name-matched initial state and native PD drives."""
    from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    robot = get_robot_spec(name)
    if robot.name == "g1":
        from mjlab.asset_zoo.robots import get_g1_robot_cfg

        return get_g1_robot_cfg()

    default_pos = dict(zip(robot.joint_names, robot.default_joint_pos))
    if robot.name == "go2":
        def spec_fn():
            spec = mujoco.MjSpec.from_file(str(robot.model_path))
            _name_unnamed_geoms(spec)
            spec.compiler.fusestatic = False
            spec.body(robot.root_body_name).add_freejoint(name="root")
            return spec

        actuators = (
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                stiffness=40.0, damping=1.0, effort_limit=23.5,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_calf_joint",),
                stiffness=40.0, damping=1.0, effort_limit=23.5,
            ),
        )
        collision_cfg = CollisionCfg(
            geom_names_expr=(r".*_collision.*",), contype=1, conaffinity=1,
            condim={r".*_foot_collision": 6, r".*_collision.*": 1},
            priority={r".*_foot_collision": 1, ".*": 0},
            friction={r".*_foot_collision": (1.0, 5e-3, 5e-4)},
        )
    else:
        def spec_fn():
            return _m1_urdf_spec(robot.model_path)

        pos_leg = tuple(n for n in M1_LEG_JOINTS if n not in robot.wheel_joint_names)
        actuators = (
            BuiltinPositionActuatorCfg(
                target_names_expr=pos_leg,
                stiffness=250.0, damping=9.0, effort_limit=180.0,
            ),
            BuiltinVelocityActuatorCfg(
                target_names_expr=robot.wheel_joint_names,
                damping=0.5, effort_limit=28.68,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=M1_ARM_JOINTS,
                stiffness=32.0, damping=1.0, effort_limit=100.0,
            ),
        )
        collision_cfg = CollisionCfg(geom_names_expr=(".*",), contype=1, conaffinity=1, condim=3, priority=0)

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot.default_base_height),
            joint_pos=default_pos,
            joint_vel={".*": 0.0},
        ),
        collisions=(collision_cfg,),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
    )
