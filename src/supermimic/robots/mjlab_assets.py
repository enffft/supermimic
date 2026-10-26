"""MuJoCo/mjlab model adapters for the two non-zoo robots."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
import hashlib
import math
import shutil
import subprocess
from pathlib import Path

import mujoco

from supermimic.robots.spec import M1_ARM_JOINTS, M1_LEG_JOINTS, M1_JOINTS, get_robot_spec


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> None:
    """The source Go2 URDF leaves collision geom names blank; name by owning link."""
    for body in spec.bodies:
        for i, geom in enumerate(body.geoms):
            # Keep collision proxies in a viewer-hidden group. They remain active
            # for physics, while the separately added visual mesh is what users see.
            geom.group = 3
            if not geom.name:
                suffix = "_collision" if len(body.geoms) == 1 else f"_collision_{i}"
                geom.name = f"{body.name}{suffix}"


def _stl_face_count(path: Path) -> int:
    """Count triangles in either binary or ASCII STL without decoding the mesh."""
    with path.open("rb") as stream:
        header = stream.read(84)
    if len(header) == 84:
        count = int.from_bytes(header[80:84], "little")
        if path.stat().st_size == 84 + count * 50:
            return count
    return path.read_bytes().lower().count(b"facet normal")


def _dae_export_axes(path: Path) -> tuple[str, str]:
    """Choose Blender export axes that preserve a Z-up URDF mesh frame."""
    try:
        root = ET.parse(path).getroot()
        up_axis = root.find("./{*}asset/{*}up_axis")
        source_up = (
            up_axis.text.strip().upper()
            if up_axis is not None and up_axis.text
            else ""
        )
    except ET.ParseError:
        source_up = ""
    # Blender imports Collada node transforms in the declared up-axis frame. The
    # Go2 and Piper assets declare Z_UP and URDF expects link-local Z_UP; exporting
    # them through Blender's default Y_UP mapping rotates their vertices again.
    # Keep the established mapping for other/unspecified Collada conventions.
    return ("Y", "Z") if source_up == "Z_UP" else ("-Z", "Y")


def _convert_visual_mesh(path: Path, *, max_faces: int = 160_000) -> Path:
    """Convert DAE or an oversized STL visual to a MuJoCo-friendly OBJ."""
    path = path.resolve()
    suffix = path.suffix.lower()
    # URDF mesh vertices are expressed in the mesh's local frame. Blender's STL
    # importer preserves it. For Collada, respect its declared up axis: Z_UP is
    # already the URDF link convention and must not be rotated into Y_UP.
    axis_forward, axis_up = _dae_export_axes(path) if suffix == ".dae" else ("Y", "Z")
    # Invalidate earlier Collada exports, which used one axis conversion for
    # every asset regardless of the source <up_axis> declaration. This keeps
    # oversized STL conversions on their existing cache key.
    fingerprint = (
        f"v4:{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}:"
        f"{max_faces}:{axis_forward}:{axis_up}"
    )
    key = hashlib.sha1(fingerprint.encode()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "supermimic" / "dae_visuals"
    output = cache_dir / f"{path.stem}_{key}.obj"
    material = output.with_suffix(".mtl")
    if output.is_file() and material.is_file():
        return output

    blender = shutil.which("blender")
    if blender is None:
        raise RuntimeError(
            f"Cannot render URDF visual {path}: MuJoCo does not load DAE meshes and Blender was not found."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    converter = r'''import bpy, sys
from pathlib import Path
args = sys.argv[sys.argv.index("--") + 1:]
source, output, max_faces = Path(args[0]).resolve(), Path(args[1]).resolve(), int(args[2])
axis_forward, axis_up = args[3], args[4]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
if source.suffix.lower() == ".dae":
    bpy.ops.wm.collada_import(filepath=str(source))
else:
    bpy.ops.import_mesh.stl(filepath=str(source))
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("Visual mesh contains no mesh objects: " + str(source))
for obj in meshes:
    if len(obj.data.polygons) > max_faces:
        modifier = obj.modifiers.new(name="mujoco_visual_decimation", type="DECIMATE")
        modifier.ratio = max_faces / len(obj.data.polygons)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
bpy.ops.object.select_all(action="DESELECT")
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.export_scene.obj(filepath=str(output), use_selection=True,
                         use_materials=True, use_triangles=True,
                         # Export in the URDF link frame selected from the
                         # source format/up-axis convention.
                         axis_forward=axis_forward, axis_up=axis_up)
'''
    with tempfile.TemporaryDirectory(prefix="sm_dae_", dir=cache_dir) as tmp:
        tmp_obj = Path(tmp) / output.name
        script = Path(tmp) / "convert.py"
        script.write_text(converter)
        result = subprocess.run(
            [
                blender, "-b", "--python", str(script), "--", str(path), str(tmp_obj),
                str(max_faces), axis_forward, axis_up,
            ],
            capture_output=True,
            text=True,
        )
        tmp_mtl = tmp_obj.with_suffix(".mtl")
        if result.returncode != 0 or not tmp_obj.is_file() or not tmp_mtl.is_file():
            raise RuntimeError(
                f"Failed converting URDF visual mesh {path} with Blender:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        # Blender's OBJ refers to the matching MTL basename. Rename both only
        # after a complete conversion so interrupted imports cannot leave a bad cache.
        tmp_obj.replace(output)
        tmp_mtl.replace(material)
    return output


def _renderable_visual_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".dae":
        return _convert_visual_mesh(path)
    if suffix == ".stl" and _stl_face_count(path) > 180_000:
        return _convert_visual_mesh(path)
    return path


def _obj_diffuse_color(path: Path) -> tuple[float, float, float] | None:
    """Read the primary diffuse color Blender exported from a Collada material."""
    material_path = path.with_suffix(".mtl")
    if not material_path.is_file():
        return None
    for line in material_path.read_text(errors="ignore").splitlines():
        if line.startswith("Kd "):
            values = tuple(float(value) for value in line.split()[1:4])
            if len(values) == 3:
                return values
    return None


def _urdf_xyz(text: str | None, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not text:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"Expected three URDF coordinates, got {text!r}")
    return values


def _quat_from_urdf_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _m1_stl_counterpart(uri: Path, urdf_dir: Path) -> Path | None:
    """Return the source Piper-X STL matching a DAE visual, when one exists."""
    if uri.suffix.lower() != ".dae":
        return None
    stem = uri.stem
    if stem.startswith("J") and stem[1:].isdigit():
        stem = f"Link{stem[1:]}"
    candidate = (urdf_dir / uri.parent.parent / f"{stem}.STL").resolve()
    return candidate if candidate.is_file() else None


def _m1_visual_rpy(uri: Path, rpy: tuple[float, float, float]) -> tuple[float, float, float]:
    """Correct the redundant quarter-turn on the M1 z-up quadruped STLs."""
    if (
        uri.suffix.lower() == ".stl"
        and "zg_m1" in uri.parts
        and math.isclose(rpy[0], -math.pi / 2, abs_tol=1e-5)
        and math.isclose(rpy[1], 0.0, abs_tol=1e-5)
        and math.isclose(rpy[2], 0.0, abs_tol=1e-5)
    ):
        return (0.0, 0.0, 0.0)
    return rpy


def _add_urdf_visuals(
    spec: mujoco.MjSpec,
    root: ET.Element,
    urdf_dir: Path,
    *,
    prefer_m1_stl_visuals: bool = False,
) -> None:
    """Add URDF visuals explicitly; MuJoCo's URDF reader imports collisions only."""
    mesh_names: dict[tuple[Path, tuple[float, float, float]], str] = {}
    for link in root.findall("link"):
        body = spec.body(link.attrib["name"])
        if body is None:
            continue
        for visual_index, visual in enumerate(link.findall("visual")):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            origin = visual.find("origin")
            pos = _urdf_xyz(origin.attrib.get("xyz") if origin is not None else None)
            rpy = _urdf_xyz(origin.attrib.get("rpy") if origin is not None else None)
            color_node = visual.find("./material/color")
            rgba = tuple(float(x) for x in color_node.attrib["rgba"].split()) if color_node is not None else (0.7, 0.7, 0.7, 1.0)
            if len(rgba) != 4:
                raise ValueError(f"{link.attrib['name']}: visual rgba must contain four values")

            mesh_node = geometry.find("mesh")
            if mesh_node is not None:
                uri = Path(mesh_node.attrib["filename"])
                # In the M1 quadruped portion, the STL vertices already use the
                # z-up link frame: their long hip/knee axes follow the URDF's
                # negative-z joint offsets, and collision proxies are defined in
                # that same frame. The source URDF repeats a -90 degree visual
                # roll, which turns those meshes sideways relative to the joints.
                if prefer_m1_stl_visuals:
                    rpy = _m1_visual_rpy(uri, rpy)
                quat = _quat_from_urdf_rpy(rpy)
                mesh_path = uri if uri.is_absolute() else urdf_dir / uri
                if prefer_m1_stl_visuals:
                    # The M1 package ships corresponding Piper-X STL assets next
                    # to the DAE visuals. They use the URDF link frame directly;
                    # load them as renderable geoms instead of hiding them as
                    # collision proxies while trying to render the DAE meshes.
                    mesh_path = _m1_stl_counterpart(uri, urdf_dir) or mesh_path
                dae_visual = mesh_path.suffix.lower() == ".dae"
                mesh_path = _renderable_visual_path(mesh_path.resolve())
                # The Go2 URDF paints every Collada mesh white, while the DAE
                # contains its actual part colors. Use the primary DAE material
                # when there is no explicit URDF color or it is plain white.
                if dae_visual and (
                    color_node is None or all(abs(value - 1.0) < 1e-6 for value in rgba)
                ):
                    diffuse = _obj_diffuse_color(mesh_path)
                    if diffuse is not None:
                        rgba = (*diffuse, rgba[3])
                scale = _urdf_xyz(mesh_node.attrib.get("scale"), (1.0, 1.0, 1.0))
                mesh_key = (mesh_path, scale)
                mesh_name = mesh_names.get(mesh_key)
                if mesh_name is None:
                    key = hashlib.sha1(f"{mesh_path}:{scale}".encode()).hexdigest()[:12]
                    mesh_name = f"visual_mesh_{key}"
                    spec.add_mesh(name=mesh_name, file=str(mesh_path), scale=scale, smoothnormal=1)
                    mesh_names[mesh_key] = mesh_name
                body.add_geom(
                    name=f"visual_{link.attrib['name']}_{visual_index}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                    pos=pos,
                    quat=quat,
                    contype=0,
                    conaffinity=0,
                    mass=0.0,
                    rgba=rgba,
                    group=1,
                )
                continue

            quat = _quat_from_urdf_rpy(rpy)

            box = geometry.find("box")
            sphere = geometry.find("sphere")
            cylinder = geometry.find("cylinder")
            if box is not None:
                size = tuple(value / 2 for value in _urdf_xyz(box.attrib["size"]))
                geom_type = mujoco.mjtGeom.mjGEOM_BOX
            elif sphere is not None:
                size = (float(sphere.attrib["radius"]),)
                geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
            elif cylinder is not None:
                size = (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]) / 2)
                geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            else:
                continue
            body.add_geom(
                name=f"visual_{link.attrib['name']}_{visual_index}",
                type=geom_type,
                size=size,
                pos=pos,
                quat=quat,
                contype=0,
                conaffinity=0,
                mass=0.0,
                rgba=rgba,
                group=1,
            )


def _go2_urdf_spec(path: Path) -> mujoco.MjSpec:
    tree = ET.parse(path)
    spec = mujoco.MjSpec.from_file(str(path))
    _name_unnamed_geoms(spec)
    # URDF loading sets this flag to drop visual-only geoms during compilation.
    # We add supported visual meshes below, so retain them in the compiled model.
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    root_body = spec.body("base")
    if root_body is None:
        raise ValueError(f"{path}: expected base root body")
    root_body.add_freejoint(name="root")
    _add_urdf_visuals(spec, tree.getroot(), path.parent)
    return spec


def _m1_urdf_spec(path: Path) -> mujoco.MjSpec:
    """Load M1 collisions and add its visual STL/Collada meshes separately."""
    visual_root = ET.parse(path).getroot()
    tree = ET.parse(path)
    root = tree.getroot()
    for link in root.findall("link"):
        # MuJoCo's URDF reader does not consume <visual>; add the original visual
        # elements below after reading the collision model.
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for collision in list(link.findall("collision")):
            for mesh in collision.iter("mesh"):
                relative = Path(mesh.attrib["filename"])
                if relative.suffix.lower() == ".dae":
                    stem = relative.stem
                    if stem.startswith("J") and stem[1:].isdigit():
                        stem = f"Link{stem[1:]}"
                    stl = relative.parent.parent / f"{stem}.STL"
                    candidate = (path.parent / stl).resolve()
                    if candidate.is_file():
                        mesh.attrib["filename"] = str(candidate)
                    else:
                        # Keep the prior M1 collision policy for the fixed gripper:
                        # these fine DAE meshes are visual-only in this model.
                        link.remove(collision)
                        break
                else:
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
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    root_body.add_freejoint(name="root")
    _add_urdf_visuals(spec, visual_root, path.parent, prefer_m1_stl_visuals=True)
    return spec


def load_robot_spec_model(name: str) -> mujoco.MjModel:
    """Compile an inspectable MuJoCo model and verify its floating-base joint map."""
    robot = get_robot_spec(name)
    if robot.name == "g1":
        from supermimic.data.motion_convert import load_g1_model

        model = load_g1_model()
    elif robot.name == "go2":
        model = _go2_urdf_spec(robot.model_path).compile()
    else:
        model = _m1_urdf_spec(robot.model_path).compile()

    scalar_joints = sorted(
        (
            int(model.jnt_qposadr[i]),
            int(model.jnt_dofadr[i]),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
        )
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    )
    names = tuple(name for _, _, name in scalar_joints)
    if model.nq != 7 + robot.num_joints or model.nv != 6 + robot.num_joints:
        raise ValueError(f"{robot.name}: compiled nq/nv={model.nq}/{model.nv}, expected {7 + robot.num_joints}/{6 + robot.num_joints}")
    if names != robot.joint_names:
        if set(names) == set(robot.joint_names):
            raise ValueError(
                f"{robot.name}: compiled joint address order differs from RobotSpec; "
                f"compiled={names}, spec={robot.joint_names}"
            )
        raise ValueError(f"{robot.name}: URDF joints differ; missing={set(robot.joint_names)-set(names)}, extra={set(names)-set(robot.joint_names)}")
    for i, (qadr, dadr, name) in enumerate(scalar_joints):
        if qadr != 7 + i or dadr != 6 + i:
            raise ValueError(
                f"{robot.name}: non-contiguous scalar joint address for {name}: "
                f"qpos={qadr}, qvel={dadr}, expected={7+i}/{6+i}"
            )
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
            return _go2_urdf_spec(robot.model_path)

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
        collision_cfg = CollisionCfg(geom_names_expr=(r".*_collision.*",), contype=1, conaffinity=1, condim=3, priority=0)

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
