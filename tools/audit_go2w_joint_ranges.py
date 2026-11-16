"""Compare the confirmed deployment sources with training target envelopes."""

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import torch
import yaml

from unisupermimic.tasks.velocity.go2w.contract import DEFAULT_Q, LEG_JOINTS, MODEL, soft_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, default=Path("/home/yifeiy/rl_sar"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.deployment_root / "policy/go2w/base.yaml"
    config = args.deployment_root / "policy/go2w/robot_lab/config.yaml"
    urdf = args.deployment_root / "src/rl_sar_zoo/go2w_description/urdf/go2w_description.urdf"
    sdk = args.deployment_root / "src/rl_sar/library/core/rl_sdk/rl_sdk.cpp"
    fsm = args.deployment_root / "src/rl_sar/fsm_robot/fsm_go2w.hpp"
    base_cfg = yaml.safe_load(base.read_text())["go2w"]
    policy_cfg = yaml.safe_load(config.read_text())["go2w/robot_lab"]
    limits = {joint.get("name"): joint.find("limit").attrib
              for joint in ET.parse(urdf).getroot().findall("joint") if joint.find("limit") is not None}
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    narrow_limits, wide_limits = soft_limits(), soft_limits(fraction=.95)
    default = torch.tensor(DEFAULT_Q)
    narrow = torch.stack((torch.maximum(default - .25, narrow_limits[:, 0]),
                          torch.minimum(default + .25, narrow_limits[:, 1])), -1)
    rows = []
    for i, name in enumerate(LEG_JOINTS):
        deploy_name = name.removeprefix("robot/")
        index = base_cfg["joint_names"].index(deploy_name)
        mechanical = [float(limits[deploy_name][key]) for key in ("lower", "upper")]
        official = model.joint(name).range.copy().tolist()
        if max(abs(a - b) for a, b in zip(mechanical, official)) > 1e-5:
            raise ValueError(f"Deployment/model position limits disagree: {name}")
        raw = [policy_cfg["clip_actions_lower"][index], policy_cfg["clip_actions_upper"][index]]
        scaled = [policy_cfg["default_dof_pos"][index] + policy_cfg["action_scale"][index] * a for a in raw]
        rows.append(dict(joint=name, deployment_index=index, mechanical_rad=mechanical,
                         old_training_target_rad=narrow[i].tolist(), full_training_target_rad=wide_limits[i].tolist(),
                         deployment_default_rad=policy_cfg["default_dof_pos"][index],
                         deployment_raw_action_clip=raw, deployment_unconstrained_target_rad=scaled,
                         training_effort_Nm=float(model.actuator(name).ctrlrange[1]),
                         deployment_torque_parameter_Nm=policy_cfg["torque_limits"][index],
                         training_kp_kd=[40., 1.], deployment_kp_kd=[policy_cfg["rl_kp"][index], policy_cfg["rl_kd"][index]]))
    files = [base, config, urdf, sdk, fsm, MODEL]
    report = dict(
        source_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, rows=rows,
        confirmed_deployment_directory=str(base.parent),
        mapping="Full action -1/0/+1 -> 95% mechanical lower/default/upper; 2.5% interval margin at each end",
        limitation="Code/model audit; no hardware motion or encoder log measured. Raw +/-100 clip is NOT a physical joint range.",
        deployment_control="Forward clips network actions; ComputeOutput scales/adds default without a joint-position clamp; torque vector is clipped separately. Firmware limits are not established by this audit.",
        deployment_fold_posture=[0., 1.36, -2.65],
        control_alignment="Position ranges match, but gains, defaults, ordering, wheel scaling and effort differ; new student is not a drop-in deployment policy.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Verified {len(rows)} leg position limits; saved {args.output}")


if __name__ == "__main__":
    main()
