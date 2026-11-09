"""Read-only source/checkpoint audit and CPU export of the original mjlab scene.

Run with the original working conda Python; writes only into this experiment.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/yifeiy/ws_supermimic")
OUT = ROOT / "migration/source_reference"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = list((SOURCE / "src").rglob("*.py")) + [SOURCE / "pyproject.toml", SOURCE / "LICENSE"]
    hashes = {}
    for path in paths:
        rel = path.relative_to(SOURCE)
        target = OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for args, name in [
        (["status", "--porcelain=v1"], "git_status.txt"),
        (["diff", "--binary"], "source_changes.patch"),
    ]:
        (OUT / name).write_bytes(subprocess.check_output(["git", "-C", str(SOURCE), *args]))
    hashes["git_head"] = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    (OUT / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    import mjlab

    math_source = Path(mjlab.__file__).parent / "utils/lab_api/math.py"
    shutil.copy2(math_source, OUT / "mjlab_math.py")
    sys.path.insert(0, str(OUT / "src"))
    import mujoco
    from mjlab.scene import Scene
    from supermimic.envs.multirobot_env_cfg import make_multirobot_env_cfg
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs

    cfg = make_multirobot_env_cfg(
        SuperMimicEnvCfgArgs(
            npz_files=[], robot_name="go2", task_name="velocity_command", num_envs=1
        )
    )
    scene = Scene(cfg.scene, device="cpu")
    scene.spec.option.timestep = 0.005
    scene.spec.option.iterations = 10
    scene.spec.option.ls_iterations = 20
    model = scene.spec.compile()
    assets = ROOT / "src/unisupermimic/assets/robots/go2_velocity"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "flat.xml").write_text(scene.spec.to_xml())
    # Exported visual meshes use local experiment copies; physical geoms are unchanged.
    import xml.etree.ElementTree as ET

    tree = ET.parse(assets / "flat.xml")
    xml = tree.getroot()
    # MuJoCo 3.11 rejects an empty nested default emitted by 3.16.
    for parent in xml.iter():
        for child in list(parent):
            if child.tag == "default" and not child.attrib and not list(child):
                parent.remove(child)
    for mesh in xml.findall("./asset/mesh"):
        origin = Path(mesh.attrib["file"])
        if not origin.is_absolute():
            origin = Path(scene.spec.compiler.meshdir) / origin
        dst = assets / "meshes" / origin.name
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(origin, dst)
        mesh.attrib["file"] = str(dst.resolve())
    sensor = ET.SubElement(xml, "sensor")
    for leg in ("FL", "FR", "RL", "RR"):
        for joint in ("hip", "thigh", "calf"):
            name = f"robot/{leg}_{joint}_joint"
            ET.SubElement(sensor, "jointactuatorfrc", {"name": name + "_torque", "joint": name})
    tree.write(assets / "flat.xml", encoding="unicode")
    # The generated terrain owns the ground. A robot-only attachment prevents
    # duplicate plane/heightfield contacts, while flat.xml stays unchanged.
    import copy

    terrain_xml = copy.deepcopy(xml)
    for parent in terrain_xml.iter():
        for child in list(parent):
            if child.tag == "body" and child.attrib.get("name") == "terrain":
                parent.remove(child)
    ET.ElementTree(terrain_xml).write(assets / "terrain_robot.xml", encoding="unicode")
    import numpy as np

    physical_fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_condim",
        "geom_contype",
        "geom_conaffinity",
        "geom_priority",
        "geom_solref",
        "geom_solimp",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "qpos0",
        "key_qpos",
        "key_ctrl",
    )
    np.savez(
        OUT / "compiled_physics.npz", **{k: np.asarray(getattr(model, k)) for k in physical_fields}
    )
    ckpt = (
        SOURCE
        / "logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt"
    )
    import torch

    saved = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    (OUT / "checkpoint_config.json").write_text(
        json.dumps(saved["config"], indent=2, default=str) + "\n"
    )
    (OUT / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "path": str(ckpt),
                "sha256": hashlib.file_digest(ckpt.open("rb"), "sha256").hexdigest(),
                "keys": list(saved),
                "model_fingerprint": saved.get("robot_model_sha256"),
                "tensor_shapes": {
                    k: {n: list(t.shape) for n, t in v.items() if isinstance(t, torch.Tensor)}
                    for k, v in saved.items()
                    if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values())
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print("Exported original scene and source/checkpoint manifests:", OUT)


if __name__ == "__main__":
    main()
