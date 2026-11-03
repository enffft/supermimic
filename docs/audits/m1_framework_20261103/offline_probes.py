"""Read-only CPU probes for the 2026-11-03 M1 framework audit.

Run from repository root with the supermimic Conda Python. No simulator steps,
training, optimizer updates, or checkpoint writes. Synthetic loss probes are
counterexamples to objective properties, not predictions of physical behavior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from supermimic.motion_prior.losses import (
    M1JointLossCfg, _task_planner_terms, _reference_smoothness_terms, actor_losses,
)
from supermimic.robots.spec import M1_AMP_JOINTS, get_robot_spec


class RootAsTcp:
    """Synthetic differentiable FK used only to isolate loss arithmetic."""

    def __call__(self, qpos, qvel):
        return {"pos": qpos[:, None, :3]}


def main():
    torch.set_num_threads(2)
    cfg = M1JointLossCfg()
    h = 10
    state = torch.zeros(h, 1, 57)
    state[..., 2] = 0.47
    state[..., 3] = 1.0
    actions = torch.zeros(h, 1, 22)
    common = dict(
        states=state, reference_states=state.clone(), actions=actions,
        previous_action=actions[0], soft_limits=torch.tensor([[-3.0, 3.0]] * 22),
        cfg=cfg, task_type=torch.tensor([1]), velocity_command=torch.zeros(1, 3),
        fixed_world_target=torch.tensor([[0.30, 0.0, 0.47]]), fk=RootAsTcp(),
        reference_actions=actions, initial_state=state[0],
    )
    alive = torch.ones(h, 1)
    dead = torch.zeros(h, 1)
    first = dead.clone()
    first[0] = 1
    safe = actor_losses(**common, valid_mask=alive, safety_mask=alive, survival_probability=alive)
    fail = actor_losses(**common, valid_mask=dead, safety_mask=first, survival_probability=dead)
    low_states = state.clone()
    low_states[..., 2] = 0.21  # below the configured 0.22 m safety height
    low_fail = actor_losses(**{**common, "states": low_states}, valid_mask=dead,
                            safety_mask=first, survival_probability=dead)
    scalars = lambda x: {k: float(v.detach()) for k, v in x.items() if v.numel() == 1}

    states = torch.cat((state[:1], state), dim=0).requires_grad_(True)
    m = _task_planner_terms(states, torch.tensor([1]), torch.zeros(1, 3),
                            common["fixed_world_target"], RootAsTcp(), 0, cfg)
    grad, = torch.autograd.grad(m["task_m"], states)
    l = _task_planner_terms(states, torch.tensor([0]), torch.tensor([[0.12, 0., 0.]]),
                            common["fixed_world_target"], RootAsTcp(), 0, cfg)

    spec = get_robot_spec("m1_piper", m1_pose_profile="dataset_v3_3_l_high_carry", m1_action_contract="v3")
    initial = torch.zeros(1, 57)
    initial[:, 2] = spec.default_base_height
    initial[:, 3] = 1
    initial[:, 7:29] = torch.tensor(spec.default_joint_pos)
    future = torch.zeros(1, h, 31)
    future[..., 3] = 1
    future[..., 7] = 1
    for i, name in enumerate(M1_AMP_JOINTS):
        future[..., 9+i] = spec.default_joint_pos[spec.joint_names.index(name)]
    # A perfectly straight, constant-speed reference can jump from initial
    # actual v=0 to reference v=1 m/s without a boundary acceleration penalty.
    future[0, :, 0] = torch.arange(1, h+1) * cfg.control_dt
    smooth = _reference_smoothness_terms(initial, future, cfg)

    paths = [
        "src/supermimic/motion_prior/losses.py",
        "src/supermimic/motion_prior/trainer.py",
        "src/supermimic/motion_prior/features.py",
        "src/supermimic/motion_prior/rollout.py",
        "src/supermimic/policy/nets.py",
        "scripts/audit_m1_action22_real_response.py",
        "/home/yifeiy/下载/SuperMimic_M1_最终系统框架_V1_2026-11-03.md",
        "/home/yifeiy/AMP-LOCO-M1/motion_prior/cvae_features.py",
        "/home/yifeiy/AMP-LOCO-M1/motion_prior/cvae_data.py",
    ]
    report = {
        "python": sys.executable,
        "scope": "CPU loss arithmetic and source identity; no physical rollout or training",
        "supermimic_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "amp_head": subprocess.check_output(["git", "-C", "/home/yifeiy/AMP-LOCO-M1", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        "termination_counterexample": {
            "assumption": "synthetic TCP, safe geometry, WM predicts immediate termination; tests loss ordering only",
            "safe_stationary_30cm": scalars(safe), "predicted_immediate_failure": scalars(fail),
            "predicted_failure_with_low_base_021m": scalars(low_fail),
            "failure_has_lower_total": bool(fail["total"] < safe["total"]),
        },
        "task_scales": {
            "stationary_l_vx_goal_012": float(l["task_l"].detach()),
            "stationary_m_distance_030": float(m["task_m"].detach()),
            "loss_ratio_not_parameter_gradient_ratio": float((m["task_m"] / l["task_l"]).detach()),
            "m_tcp_dx_gradient_by_future_step": grad[1:, 0, 0].tolist(),
            "last_vs_first_step_gradient_ratio": float(grad[-1, 0, 0] / grad[1, 0, 0]),
        },
        "initial_velocity_boundary": {
            "actual_vx": 0.0, "reference_segment_vx": 1.0,
            "implied_first_acceleration_m_s2": 1.0/cfg.control_dt,
            "reported_smoothness_terms": scalars(smooth),
        },
    }
    # Actual checkpoint inference at nominal kinematic reset poses. The FK
    # model is built on CPU; no physics step/settling is performed.
    from supermimic.motion_prior import load_m1_motion_prior
    from supermimic.motion_prior.features import task_goal8, state_history_to_cvae53
    from supermimic.motion_prior.reference import decode_future31
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.mjlab_assets import load_robot_spec_model

    ckpt = Path("/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt")
    prior = load_m1_motion_prior(ckpt, device="cpu", input_compatibility="legacy_v1")
    fk = G1Kinematics(load_robot_spec_model("m1_piper"), device="cpu")
    tcp_index = fk.body_index("piper_x_link7")
    report["prior_sha256"] = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    report["nominal_reset_inference_not_physical_execution"] = []
    for profile, task, vx in (
        ("dataset_v3_3_l_high_carry", 0, 0.12),
        ("dataset_v3_3_l_high_carry", 0, -0.06),
        ("dataset_v3_3_l_high_carry", 1, 0.0),
        ("dataset_carry", 1, 0.0),
    ):
        rs = get_robot_spec("m1_piper", m1_pose_profile=profile, m1_action_contract="v3")
        s = initial.clone()
        s[:, 2] = rs.default_base_height
        s[:, 7:29] = torch.tensor(rs.default_joint_pos)
        hist = s[:, None].expand(-1, 4, -1).clone()
        tcp0 = fk(s[:, :29], s[:, 29:])["pos"][:, tcp_index]
        target = tcp0 + torch.tensor([[0.30, 0.0, 0.0]])
        goal = task_goal8(s, torch.tensor([task]), torch.tensor([[vx, 0., 0.]]), target)
        with torch.inference_mode():
            pred = prior(hist, goal, mode="prior_mean")
            decoded = decode_future31(s, pred)
            tcp_end = fk(decoded[:, -1, :29], decoded[:, -1, 29:])["pos"][:, tcp_index]
            cond = torch.cat((state_history_to_cvae53(hist).flatten(1), goal), dim=-1)
            z = ((cond-prior.x_mean)/prior.x_std).abs()
        report["nominal_reset_inference_not_physical_execution"].append({
            "profile": profile, "task": "L" if task == 0 else "M", "vx_goal": vx,
            "goal8": goal[0].tolist(), "condition_max_abs_zscore": float(z.max()),
            "condition_fraction_abs_z_above5": float((z > 5).float().mean()),
            "predicted_root_delta_at_02s": pred[0, -1, :3].tolist(),
            "predicted_mean_wheel_rates": pred[0, :, 27:31].mean(0).tolist(),
            "kinematic_predicted_tcp_progress_m": float((target-tcp0).norm()-(target-tcp_end).norm()),
        })
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
