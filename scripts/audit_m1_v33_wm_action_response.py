"""CPU diagnostic: test whether the checkpoint World Model predicts the correct L action direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.kinematics.rotations import heading_quat, quat_apply_inverse
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import state_history_to_cvae53
from supermimic.motion_prior.reference import decode_future31
from supermimic.robots.spec import M1_JOINTS
from supermimic.tasks.action_contract import inverse_m1_setpoint_action, m1_robot_spec_from_checkpoint
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.state_spec import StateSpecCfg


def audit(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "supermimic_m1_prior_actor_joint_v3":
        raise ValueError("expected a v3 M1 joint-training Checkpoint")
    robot_spec = m1_robot_spec_from_checkpoint(checkpoint)

    saved_wm_cfg = dict(checkpoint["config"]["world_model"])
    state_spec_cfg = dict(saved_wm_cfg.pop("state_spec"))
    state_spec_cfg["wheel_indices"] = tuple(state_spec_cfg["wheel_indices"])
    wm_cfg = WorldModelCfg(
        **saved_wm_cfg, state_spec=StateSpecCfg(**state_spec_cfg),
    )
    wm = WorldModel(wm_cfg, heads=object(), device="cpu")
    wm.load_state_dict(checkpoint["wm"], strict=True)
    wm.eval()

    source = checkpoint["prior_source"]
    prior = load_m1_motion_prior(
        source["path"], device="cpu", expected_sha256=source["sha256"],
        input_compatibility=checkpoint["prior_input_compatibility"],
    )
    initial_prior = load_m1_motion_prior(
        source["path"], device="cpu", expected_sha256=source["sha256"],
        input_compatibility=checkpoint["prior_input_compatibility"],
    )
    prior.load_state_dict(checkpoint["motion_prior"], strict=True)
    initial_prior.load_state_dict(checkpoint["prior_initial"], strict=True)
    prior.eval()
    initial_prior.eval()

    state = torch.zeros(1, 57)
    state[:, 2] = 0.47
    state[:, 3] = 1.0
    state[:, 7:29] = torch.tensor(robot_spec.default_joint_pos)
    history = state[:, None].expand(-1, 4, -1).clone()
    features = state_history_to_cvae53(history)
    xi = torch.zeros(1, 26)
    xi[:, 0] = 0.75  # nominal friction, no encoder bias or COM randomization
    wheel_ids = torch.tensor([
        M1_JOINTS.index(name) for name in (
            "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
        )
    ])

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "iteration": int(checkpoint["iteration"]),
        "wm_optimizer_steps": int(checkpoint["wm_optimizer_steps"]),
        "prior_optimizer_steps": int(checkpoint["prior_optimizer_steps"]),
        "prior_sha256": source["sha256"],
        "input_compatibility": checkpoint["prior_input_compatibility"],
        "state_source": "static dataset_v3_3_l_high_carry; State57; nominal friction; no DR",
        "action_source": "V3.3 Prior Future31 inverted through the saved asymmetric v3 contract",
        "world_model_prediction_is_not_mujoco": True,
        "commands": [],
    }
    for vx in (-0.08, 0.0, 0.12):
        goal = torch.tensor([[vx, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        row = {"command_vx_m_s": vx, "priors": {}}
        for label, model in (("prior_initial", initial_prior), ("prior_fine_tuned", prior)):
            with torch.inference_mode():
                future = model.forward_features(features, goal)
                reference = decode_future31(state, future)
                position_target = reference[..., 7:29]
                wheel_rates = reference[..., 35:57].index_select(-1, wheel_ids)
                actions = inverse_m1_setpoint_action(
                    position_target, wheel_rates, robot_spec=robot_spec,
                )
                prediction = state.clone()
                for step in range(reference.shape[1]):
                    prediction = wm.step(prediction, actions[:, step], xi)
                final_heading = heading_quat(prediction[:, 3:7])
                predicted_body_velocity = quat_apply_inverse(final_heading, prediction[:, 29:32])
            row["priors"][label] = {
                "reference_root_dx_0p2s_m": float(reference[0, -1, 0] - state[0, 0]),
                "reference_wheel_mean_rad_s_by_joint": wheel_rates.mean(1)[0].tolist(),
                "wm_predicted_root_dx_0p2s_m": float(prediction[0, 0] - state[0, 0]),
                "wm_predicted_final_body_velocity_m_s": predicted_body_velocity[0].tolist(),
                "wm_predicted_final_wheel_rate_rad_s_by_joint": prediction[0, 35:57].index_select(0, wheel_ids).tolist(),
            }
        result["commands"].append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    encoded = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
