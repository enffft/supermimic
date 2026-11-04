#!/usr/bin/env python3
"""Paired real-MuJoCo versus World-Model Action22 finite-difference audit.

The tool perturbs the same normalized Action22 sequence by +/- epsilon from
matched simulator states. It records the physical response and task-loss
direction at 1/2/4/8/10 steps. An audit over fewer than 128 states is explicitly
diagnostic and cannot pass the Actor gate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.features import task_goal8
from supermimic.motion_prior.losses import M1JointLossCfg, _task_planner_terms
from supermimic.motion_prior.metrics import root_tilt_and_stability
from supermimic.motion_prior.reference import future31_to_m1_action22
from supermimic.motion_prior.trainer import module_sha256
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import M1_JOINTS, get_robot_spec
from supermimic.tasks.action_contract import m1_robot_spec_from_checkpoint
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


HORIZONS = (1, 2, 4, 8, 10)


def _load_wm(checkpoint: dict, device: str):
    saved = dict(checkpoint["config"]["world_model"])
    state_spec = dict(saved.pop("state_spec"))
    if "wheel_indices" in state_spec:
        state_spec["wheel_indices"] = tuple(state_spec["wheel_indices"])
    wm = WorldModel(WorldModelCfg(**saved, state_spec=StateSpecCfg(**state_spec)), heads=object(), device=device)
    wm.load_state_dict(checkpoint["wm"], strict=True)
    wm.eval().requires_grad_(False)
    if module_sha256(wm) != checkpoint.get("world_model_sha256"):
        raise ValueError("loaded World Model does not match checkpoint world_model_sha256")
    return wm


def _scenarios():
    return (
        {"name": "L_forward", "task": 0, "cmd": (0.12, 0.0, 0.0), "offset": None},
        {"name": "L_backward", "task": 0, "cmd": (-0.06, 0.0, 0.0), "offset": None},
        {"name": "L_stop", "task": 0, "cmd": (0.0, 0.0, 0.0), "offset": None},
        {"name": "L_deceleration", "task": 0, "cmd": (0.0, 0.0, 0.0), "prefix_cmd": (0.12, 0.0, 0.0), "offset": None},
        {"name": "M_forward", "task": 1, "cmd": (0.0, 0.0, 0.0), "offset": (0.24, 0.0, 0.0)},
        {"name": "M_side", "task": 1, "cmd": (0.0, 0.0, 0.0), "offset": (0.12, 0.16, 0.0)},
        {"name": "M_up", "task": 1, "cmd": (0.0, 0.0, 0.0), "offset": (0.12, 0.0, 0.10)},
    )


def _action_directions(spec, task: int, state: torch.Tensor, target: torch.Tensor, fk, tcp_index: int):
    directions: dict[str, torch.Tensor] = {}
    wheel_ids = [spec.joint_names.index(name) for name in spec.wheel_joint_names]
    wheel = torch.zeros(22, device=state.device)
    wheel[wheel_ids] = 1.0 / len(wheel_ids) ** 0.5
    directions["wheel_common"] = wheel
    for name in spec.wheel_joint_names:
        value = torch.zeros(22, device=state.device)
        value[spec.joint_names.index(name)] = 1.0
        directions[f"wheel_{name}"] = value
    hip_id = spec.joint_names.index("FAR_HIP_JOINT")
    hip = torch.zeros(22, device=state.device)
    hip[hip_id] = 1.0
    directions["front_hip"] = hip

    # Piper direction is the local normalized Action22 direction that most
    # reduces TCP distance under the differentiable kinematic Jacobian.
    arm_ids = [spec.joint_names.index(f"piper_x_joint{i}") for i in range(1, 7)]
    qpos = state[:, :29].detach().clone().requires_grad_(True)
    tcp = fk(qpos, state[:, 29:])["pos"][:, tcp_index]
    distance = torch.linalg.vector_norm(tcp - target, dim=-1).sum()
    q_grad, = torch.autograd.grad(distance, qpos)
    scales = state.new_tensor(spec.action_scales)
    arm_direction = torch.zeros(22, device=state.device)
    arm_direction[arm_ids] = -q_grad[0, [7 + i for i in arm_ids]] * scales[arm_ids]
    if float(arm_direction.norm()) <= 1e-8:
        arm_direction[arm_ids[0]] = 1.0
    arm_direction = arm_direction / arm_direction.norm().clamp_min(1e-8)
    directions["arm_toward_tcp"] = arm_direction
    directions["arm_away_from_tcp"] = -arm_direction
    return directions


def _task_losses(initial: torch.Tensor, states: torch.Tensor, task: int,
                 command: torch.Tensor, target: torch.Tensor, fk, tcp_index: int):
    """Return the training objective per paired branch, without batch averaging.

    The FD batch contains the ``+epsilon`` and ``-epsilon`` branches. Reducing
    both rows to one task mean before differencing would erase the very signal
    this audit measures.
    """
    cfg = M1JointLossCfg()
    values = []
    for index in range(states.shape[0]):
        labels = torch.tensor([task], dtype=torch.long, device=states.device)
        terms = _task_planner_terms(
            torch.stack((initial[index:index + 1], states[index:index + 1]), dim=0),
            labels, command[index:index + 1], target[index:index + 1],
            fk, tcp_index, cfg,
        )
        if task == 0:
            values.append(terms["task_l"])
        else:
            values.append(terms["task_m"] + terms["m_base_drift"])
    return torch.stack(values)


def _paired_response(real_pair: torch.Tensor, wm_pair: torch.Tensor, epsilon: float,
                     low_signal: float) -> dict:
    """Compare central finite-difference responses for [plus, minus] branches."""
    real_delta = real_pair[0].float() - real_pair[1].float()
    wm_delta = wm_pair[0].float() - wm_pair[1].float()
    real_norm = float(real_delta.norm())
    wm_norm = float(wm_delta.norm())
    if real_norm < low_signal:
        return {
            "low_signal": True, "real_delta_norm": real_norm,
            "wm_delta_norm": wm_norm, "real_slope_norm": real_norm / (2 * epsilon),
            "wm_slope_norm": wm_norm / (2 * epsilon),
            "direction_cosine": None, "relative_response_error": None,
        }
    cosine = float(torch.dot(real_delta.flatten(), wm_delta.flatten()) /
                   (real_delta.norm() * wm_delta.norm()).clamp_min(1.0e-12))
    return {
        "low_signal": False, "real_delta_norm": real_norm,
        "wm_delta_norm": wm_norm, "real_slope_norm": real_norm / (2 * epsilon),
        "wm_slope_norm": wm_norm / (2 * epsilon), "direction_cosine": cosine,
        "relative_response_error": float((wm_delta - real_delta).norm() / real_delta.norm().clamp_min(1.0e-12)),
    }


def _task_response(real_pair: list[float], wm_pair: list[float], epsilon: float,
                   low_signal: float = 1.0e-5) -> dict:
    real_delta = float(real_pair[0] - real_pair[1])
    wm_delta = float(wm_pair[0] - wm_pair[1])
    real_slope, wm_slope = real_delta / (2 * epsilon), wm_delta / (2 * epsilon)
    low = abs(real_delta) < low_signal
    return {
        "real_plus_minus_delta": real_delta, "wm_plus_minus_delta": wm_delta,
        "real_directional_slope": real_slope, "wm_directional_slope": wm_slope,
        "low_signal": low,
        "direction_agreement": None if low else bool(real_delta * wm_delta > 0 or (real_delta == 0 and wm_delta == 0)),
    }


_OUTPUT_LOW_SIGNAL = {
    "body_velocity": 5.0e-3, "root_displacement": 2.0e-4,
    "wheel_rate": 1.0e-2, "leg_q": 1.0e-3, "arm_q": 1.0e-3,
    "tcp_position": 2.0e-4, "tilt_height": 1.0e-3,
}


def audit(
    checkpoint_path: Path, *, state_count: int = 128, seed: int = 20261104,
    device: str = "cuda:0", epsilon: float = 0.02,
    min_direction_agreement: float = 0.80,
    max_relative_response_error: float = 0.50,
    out: Path | None = None,
) -> dict:
    if state_count < 1 or epsilon <= 0:
        raise ValueError("state_count and epsilon must be positive")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    supported = {
        "supermimic_m1_action22_reference_v1", "supermimic_m1_action22_reference_v2",
    }
    if ckpt.get("format") not in supported:
        raise ValueError("finite-difference audit requires an Action22-reference V4/V5 checkpoint")
    spec = m1_robot_spec_from_checkpoint(ckpt)
    wm = _load_wm(ckpt, device)
    source = ckpt["prior_source"]
    prior = load_m1_motion_prior(
        source["path"], device=device, expected_sha256=source["sha256"],
        input_compatibility=ckpt["prior_input_compatibility"],
    )
    prior.load_state_dict(ckpt["motion_prior"], strict=True)
    prior.eval().requires_grad_(False)
    fk = G1Kinematics(load_robot_spec_model("m1_piper"), device=device)
    tcp_index = fk.body_index("piper_x_link7")
    profile = {
        "m1_pose_legacy_manipulation_v1": "legacy_manipulation",
        "m1_pose_dataset_carry_v1": "dataset_carry",
        "m1_pose_dataset_v3_3_l_high_carry_v1": "dataset_v3_3_l_high_carry",
    }[spec.default_pose_profile]
    sampler = ckpt.get("task_sampler_state", {})
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="m1_piper", task_name="m1_prior_joint",
        m1_pose_profile=profile, m1_action_contract="v3", num_envs=2,
        history_length=5, episode_length_s=10.0, action_noise_std=0.0,
        obs_noise=False, reset_noise=True, domain_randomization=False,
        record_transitions=False, replay_capacity=16, seed=seed,
        m1_joint_task_type="L", m1_joint_vx_range=tuple(sampler.get("vx_range", (-0.09, 0.145))),
        m1_joint_vy_range=tuple(sampler.get("vy_range", (-0.05, 0.055))),
        m1_joint_wz_range=tuple(sampler.get("wz_range", (-0.17, 0.18))),
    ), device=device)
    state_io = StateIO(env)
    context = SimContext(
        env, spec.foot_geom_regex, com_body_name=spec.anchor_body_name, entity_name="robot",
    )
    command_term = env.command_manager.get_term("m1_prior_joint")
    robot = env.scene["robot"]
    wheel_ids = [M1_JOINTS.index(n) for n in spec.wheel_joint_names]
    leg_ids = [M1_JOINTS.index(n) for n in spec.amp_joint_names[:12]]
    arm_ids = [M1_JOINTS.index(n) for n in spec.amp_joint_names[12:]]
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures: dict[str, int] = defaultdict(int)
    paired_state_max_error = 0.0
    paired_xi_max_error = 0.0
    task_direction: dict[str, list[bool]] = defaultdict(list)
    low_signal: dict[str, int] = defaultdict(int)
    sample_rows: list[dict] = []
    valid_state_counts: dict[str, set[int]] = defaultdict(set)
    valid_horizon_counts: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    core_outputs = ("body_velocity", "root_displacement", "wheel_rate", "leg_q", "arm_q", "tcp_position", "tilt_height")
    try:
        for scenario in _scenarios():
            for state_index in range(state_count):
                env_seed = seed + state_index
                env.reset(seed=env_seed)
                state = state_io.read()
                # Use randomized reset row 0 as the source, then copy it to the
                # paired row so the positive/negative finite difference shares
                # an exact qpos/qvel/contact setup.
                state_io.write(torch.tensor([0, 1], device=state.device), state[0:1].expand(2, -1), forward=True)
                state = state_io.read()
                xi = context.read()
                xi[:] = xi[0:1]
                task = int(scenario["task"])
                command = state.new_tensor([scenario.get("prefix_cmd", scenario["cmd"])])
                task_ids = torch.full((2,), task, dtype=torch.long, device=state.device)
                command_term.task_type[:] = task_ids
                command_term.velocity_command_body[:] = command.expand(2, -1)
                command_term.target_position_world[:] = 0.0
                tcp_initial = robot.data.body_link_pos_w[:, command_term._tcp_body_ids].mean(dim=1) - env.scene.env_origins
                if task == 1:
                    target0 = tcp_initial[0] + state.new_tensor(scenario["offset"])
                    target = target0[None].expand(2, -1).clone()
                else:
                    target = state.new_zeros(2, 3)
                command_term.target_position_world[:] = target

                # Vary the held-out state by replaying 0..4 safe Prior steps;
                # the deceleration case changes the Goal after this prefix.
                prefix_len = state_index % 5
                history_rows = [state[0].clone()]
                prefix_goal = task_goal8(state, task_ids, command.expand(2, -1), target)
                prefix_future = prior(state[:, None].expand(-1, 4, -1), prefix_goal, mode="prior_mean")
                prefix_actions = future31_to_m1_action22(prefix_future, robot_spec=spec)
                for step in range(prefix_len):
                    action = prefix_actions[:, step]
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool((terminated | truncated).any()):
                        break
                    state = state_io.read()
                    history_rows.append(state[0].clone())
                if len(history_rows) <= prefix_len:
                    failures[scenario["name"]] += 1
                    continue
                state = state_io.read()
                state_io.write(torch.tensor([0, 1], device=state.device), state[0:1].expand(2, -1), forward=True)
                state = state_io.read()
                xi = context.read(); xi[:] = xi[0:1]
                if scenario.get("prefix_cmd") is not None:
                    command = state.new_tensor([scenario["cmd"]])
                    command_term.velocity_command_body[:] = command.expand(2, -1)
                command_term.task_type[:] = task_ids
                command_term.target_position_world[:] = target
                history = torch.stack(history_rows[-4:], dim=0)
                if history.shape[0] < 4:
                    history = torch.cat((history[:1].expand(4-history.shape[0], -1), history), dim=0)
                history = history.unsqueeze(0).contiguous()
                goal = task_goal8(state[:1], task_ids[:1], command.expand(2, -1)[:1], target[:1])
                with torch.inference_mode():
                    future = prior(history, goal, mode="prior_mean")
                    base_actions = future31_to_m1_action22(future, robot_spec=spec)
                # Goal-directed Piper finite difference uses a kinematic target
                # gradient. Other direction probes are canonical Action22 axes.
                directions = _action_directions(spec, task, state[:1], target[:1], fk, tcp_index)
                directions = {
                    name: value for name, value in directions.items()
                    if name == "wheel_common" or name.startswith("wheel_")
                    or name in ("front_hip", "arm_toward_tcp", "arm_away_from_tcp")
                }
                # Store exact start state/context identity for each paired branch.
                paired_state_max_error = max(paired_state_max_error, float((state[0]-state[1]).abs().amax()))
                paired_xi_max_error = max(paired_xi_max_error, float((xi[0]-xi[1]).abs().amax()))
                for direction_name, direction in directions.items():
                    state_io.write(torch.tensor([0, 1], device=state.device), state[0:1].expand(2, -1), forward=True)
                    state_pair = state_io.read()
                    xi_pair = context.read(); xi_pair[:] = xi_pair[0:1]
                    action_sign = state.new_tensor((1.0, -1.0))[:, None]
                    unbounded_pair_actions = (
                        base_actions[0:1] + action_sign[:, None, :] * float(epsilon) * direction[None, None, :]
                    )
                    pair_actions = unbounded_pair_actions.clamp(-1.0, 1.0)
                    action_span = pair_actions[0] - pair_actions[1]
                    epsilon_effective = float((action_span * direction[None]).sum(-1).mean() * 0.5)
                    action_saturation = float((unbounded_pair_actions.abs() > 1.0).float().mean())
                    wm_state = state_pair.clone()
                    initial_pos = state_pair[:, :3].clone()
                    initial_quat = state_pair[:, 3:7].clone()
                    target_pair = target.clone()
                    outputs_real: dict[int, dict[str, list]] = {}
                    outputs_wm: dict[int, dict[str, list]] = {}
                    task_real: dict[int, list[float]] = {}
                    task_wm: dict[int, list[float]] = {}
                    done_pair = torch.zeros(2, dtype=torch.bool, device=state.device)
                    for step in range(10):
                        action = pair_actions[:, step]
                        with torch.inference_mode():
                            wm_state = wm.step(wm_state, action, xi_pair)
                        _, _, terminated, truncated, _ = env.step(action)
                        actual_state = state_io.read()
                        done_pair |= terminated | truncated
                        # mjlab auto-resets terminated rows. Discard this step
                        # and every longer horizon rather than comparing a
                        # reset state with the still-live branch.
                        if bool(done_pair.any()):
                            break
                        if step + 1 in HORIZONS:
                            actual_body_v = quat_apply_inverse(actual_state[:, 3:7], actual_state[:, 29:32])
                            model_body_v = quat_apply_inverse(wm_state[:, 3:7], wm_state[:, 29:32])
                            actual_tcp = fk(actual_state[:, :29], actual_state[:, 29:])["pos"][:, tcp_index]
                            model_tcp = fk(wm_state[:, :29], wm_state[:, 29:])["pos"][:, tcp_index]
                            actual_tilt, _ = root_tilt_and_stability(
                                actual_state[:, 3:7], actual_state[:, 2], minimum_height_m=0.22,
                                maximum_tilt_rad=0.90,
                            )
                            model_tilt, _ = root_tilt_and_stability(
                                wm_state[:, 3:7], wm_state[:, 2], minimum_height_m=0.22,
                                maximum_tilt_rad=0.90,
                            )
                            actual_outputs = {
                                "body_velocity": actual_body_v,
                                "root_displacement": quat_apply_inverse(initial_quat, actual_state[:, :3] - initial_pos),
                                "wheel_rate": actual_state[:, 35:57][:, wheel_ids],
                                "leg_q": actual_state[:, 7:29][:, leg_ids],
                                "arm_q": actual_state[:, 7:29][:, arm_ids],
                                "tcp_position": actual_tcp,
                                "tilt_height": torch.stack((actual_tilt, actual_state[:, 2]), -1),
                            }
                            model_outputs = {
                                "body_velocity": model_body_v,
                                "root_displacement": quat_apply_inverse(initial_quat, wm_state[:, :3] - initial_pos),
                                "wheel_rate": wm_state[:, 35:57][:, wheel_ids],
                                "leg_q": wm_state[:, 7:29][:, leg_ids],
                                "arm_q": wm_state[:, 7:29][:, arm_ids],
                                "tcp_position": model_tcp,
                                "tilt_height": torch.stack((model_tilt, wm_state[:, 2]), -1),
                            }
                            outputs_real[step + 1] = {
                                name: value.detach().cpu().float().tolist()
                                for name, value in actual_outputs.items()
                            }
                            outputs_wm[step + 1] = {
                                name: value.detach().cpu().float().tolist()
                                for name, value in model_outputs.items()
                            }
                            losses_real = _task_losses(state_pair, actual_state, task, command.expand(2, -1), target_pair, fk, tcp_index)
                            losses_wm = _task_losses(state_pair, wm_state, task, command.expand(2, -1), target_pair, fk, tcp_index)
                            task_real[step + 1] = [float(v) for v in losses_real.detach().cpu()]
                            task_wm[step + 1] = [float(v) for v in losses_wm.detach().cpu()]
                    for horizon in HORIZONS:
                        if horizon not in outputs_real:
                            failures[f"{scenario['name']}/{direction_name}/h{horizon}"] += 1
                            continue
                        bucket = f"{scenario['name']}/{direction_name}/h{horizon}"
                        response_row = {}
                        for output_name in core_outputs:
                            real_pair = torch.tensor(outputs_real[horizon][output_name])
                            wm_pair = torch.tensor(outputs_wm[horizon][output_name])
                            response = _paired_response(
                                real_pair, wm_pair, max(epsilon_effective, 1.0e-8), _OUTPUT_LOW_SIGNAL[output_name],
                            )
                            response["nominal_epsilon"] = float(epsilon)
                            response["effective_epsilon"] = epsilon_effective
                            response["action_saturation_fraction"] = action_saturation
                            response_row[output_name] = response
                            buckets[bucket][f"{output_name}/count"].append(1.0)
                            if response["low_signal"]:
                                low_signal[f"{bucket}/{output_name}"] += 1
                            else:
                                buckets[bucket][f"{output_name}/direction_cosine"].append(response["direction_cosine"])
                                buckets[bucket][f"{output_name}/relative_error"].append(response["relative_response_error"])
                        task_response = _task_response(
                            task_real[horizon], task_wm[horizon], max(epsilon_effective, 1.0e-8),
                        )
                        task_response["nominal_epsilon"] = float(epsilon)
                        task_response["effective_epsilon"] = epsilon_effective
                        task_response["action_saturation_fraction"] = action_saturation
                        response_row["task_loss"] = task_response
                        buckets[bucket]["task/count"].append(1.0)
                        if task_response["low_signal"]:
                            low_signal[f"{bucket}/task"] += 1
                        else:
                            task_direction[bucket].append(task_response["direction_agreement"])
                            buckets[bucket]["task/real_slope"].append(task_response["real_directional_slope"])
                            buckets[bucket]["task/wm_slope"].append(task_response["wm_directional_slope"])
                        # Retain compact paired evidence for per-state review.
                        sample_rows.append({
                            "scenario": scenario["name"], "task": "L" if task == 0 else "M",
                            "direction": direction_name, "state_index": state_index,
                            "prefix_len": prefix_len, "horizon_steps": horizon,
                            "paired_start_state_max_abs_error": float((state[0]-state[1]).abs().amax()),
                            "paired_xi_max_abs_error": float((xi_pair[0]-xi_pair[1]).abs().amax()),
                            "task_loss_real_plus_minus": task_real[horizon],
                            "task_loss_wm_plus_minus": task_wm[horizon],
                            "responses": response_row,
                        })
                        valid_state_counts[scenario["name"]].add(state_index)
                        valid_horizon_counts[scenario["name"]][horizon].add(state_index)
                        is_disagreement = (
                            task_response["direction_agreement"] is False
                            or any(
                                not response_row[name]["low_signal"]
                                and (response_row[name]["direction_cosine"] < 0.0
                                     or response_row[name]["relative_response_error"] > 2.0)
                                for name in core_outputs
                            )
                        )
                        if is_disagreement and len(sample_rows) < 500:
                            sample_rows.append({
                                "scenario": scenario["name"], "task": "L" if task == 0 else "M",
                                "direction": direction_name, "state_index": state_index,
                                "prefix_len": prefix_len, "horizon_steps": horizon,
                                "paired_start_state_max_abs_error": float((state[0]-state[1]).abs().amax()),
                                "paired_xi_max_abs_error": float((xi_pair[0]-xi_pair[1]).abs().amax()),
                                "task_loss_real_plus_minus": task_real[horizon],
                                "task_loss_wm_plus_minus": task_wm[horizon],
                                "responses": response_row,
                            })
    finally:
        env.close()

    gate_contract = ckpt.get("actor_gate_contract", {})
    bucket_summary = {}
    for key, values in buckets.items():
        bucket_summary[key] = {
            metric: {
                "count": len(samples),
                "mean": float(torch.tensor(samples, dtype=torch.float64).mean()),
                "median": float(torch.tensor(samples, dtype=torch.float64).median()),
                "p90": float(torch.quantile(torch.tensor(samples, dtype=torch.float64), 0.90)),
            }
            for metric, samples in values.items() if samples
        }
    minimum_bucket_samples = max(16, state_count // 4)
    action_output_requirements = {
        "wheel_common": ("body_velocity", "root_displacement", "wheel_rate"),
        "front_hip": ("leg_q",),
        "arm_toward_tcp": ("arm_q", "tcp_position"),
        "arm_away_from_tcp": ("arm_q", "tcp_position"),
    }
    for spec_wheel in spec.wheel_joint_names:
        action_output_requirements[f"wheel_{spec_wheel}"] = ("body_velocity", "root_displacement", "wheel_rate")
    eligible_action_buckets = []
    insufficient_action_buckets = []
    for scenario in _scenarios():
        for direction_name, required_outputs in action_output_requirements.items():
            for horizon in HORIZONS:
                key = f"{scenario['name']}/{direction_name}/h{horizon}"
                values = buckets.get(key, {})
                for output_name in required_outputs:
                    cos = values.get(f"{output_name}/direction_cosine", [])
                    rel = values.get(f"{output_name}/relative_error", [])
                    if len(cos) < minimum_bucket_samples or len(rel) != len(cos):
                        insufficient_action_buckets.append(f"{key}/{output_name}")
                    else:
                        eligible_action_buckets.append(
                            sum(v >= min_direction_agreement for v in cos) / len(cos) >= min_direction_agreement
                            and float(torch.tensor(rel).median()) <= max_relative_response_error
                        )
    eligible_task_buckets = []
    insufficient_task_buckets = []
    for scenario in _scenarios():
        required_directions = (
            ("wheel_common",) + tuple(f"wheel_{name}" for name in spec.wheel_joint_names)
            if scenario["task"] == 0 else ("arm_toward_tcp", "arm_away_from_tcp")
        )
        for direction_name in required_directions:
            for horizon in HORIZONS:
                key = f"{scenario['name']}/{direction_name}/h{horizon}"
                agreement = task_direction.get(key, [])
                if len(agreement) < minimum_bucket_samples:
                    insufficient_task_buckets.append(key)
                else:
                    eligible_task_buckets.append(
                        sum(agreement) / len(agreement) >= min_direction_agreement
                    )
    action_response_pass = bool(eligible_action_buckets) and not insufficient_action_buckets and all(eligible_action_buckets)
    task_gradient_direction_pass = bool(eligible_task_buckets) and not insufficient_task_buckets and all(eligible_task_buckets)
    all_scenarios_covered = all(len(valid_state_counts[s["name"]]) >= min(state_count, 128) for s in _scenarios())
    gate_status = "blocked"
    if ckpt.get("format") != "supermimic_m1_action22_reference_v2":
        gate_status = "blocked_legacy_v4_checkpoint_requires_new_v5_audited_snapshot"
    elif not all_scenarios_covered:
        gate_status = "blocked_insufficient_valid_heldout_coverage"
    elif not action_response_pass or not task_gradient_direction_pass:
        gate_status = "blocked_real_action_response_or_task_gradient_disagreement"
    else:
        gate_status = "pass_candidate_contract_and_trainer_gate_still_required"
    result = {
        "audit_type": "m1_wm_mujoco_finite_difference_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": ckpt.get("format"),
        "iteration": int(ckpt.get("iteration", -1)),
        "world_model_sha256": ckpt.get("world_model_sha256"),
        "robot_model_sha256": ckpt.get("robot_model_sha256"),
        "action_mapping": ckpt.get("action_mapping"),
        "physics_contract_sha256": gate_contract.get("physics_contract_sha256"),
        "task_distribution": gate_contract.get("task_distribution"),
        "task_distribution_sha256": gate_contract.get("task_distribution_sha256"),
        "reset_contract": gate_contract.get("reset_contract"),
        "heldout_state_count": min((len(v) for v in valid_state_counts.values()), default=0),
        "valid_state_count_by_scenario": {key: len(value) for key, value in valid_state_counts.items()},
        "valid_state_count_by_scenario_horizon": {
            key: {str(horizon): len(indices) for horizon, indices in sorted(value.items())}
            for key, value in valid_horizon_counts.items()
        },
        "paired_state_max_abs_error": paired_state_max_error,
        "paired_xi_max_abs_error": paired_xi_max_error,
        "device": device, "seed": seed, "epsilon_normalized": epsilon,
        "horizons_steps": list(HORIZONS),
        "thresholds": {
            "minimum_direction_agreement": min_direction_agreement,
            "maximum_relative_response_error": max_relative_response_error,
            "note": "These thresholds are explicit initial review thresholds; low-signal samples are reported separately.",
        },
        "action_response_pass": action_response_pass,
        "task_gradient_direction_pass": task_gradient_direction_pass,
        "eligible_action_bucket_count": len(eligible_action_buckets),
        "eligible_task_bucket_count": len(eligible_task_buckets),
        "insufficient_action_buckets": insufficient_action_buckets,
        "insufficient_task_buckets": insufficient_task_buckets,
        "gate_status": gate_status,
        "failures": dict(failures),
        "low_signal_sample_counts": dict(low_signal),
        "buckets": bucket_summary,
        "samples": sample_rows,
    }
    if out is not None:
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--min-direction-agreement", type=float, default=0.80)
    parser.add_argument("--max-relative-response-error", type=float, default=0.50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.checkpoint, state_count=args.state_count, seed=args.seed, device=args.device,
        epsilon=args.epsilon, min_direction_agreement=args.min_direction_agreement,
        max_relative_response_error=args.max_relative_response_error, out=args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
