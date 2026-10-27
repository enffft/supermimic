"""Evaluate a Go2 Chase Ball checkpoint against real MuJoCo rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.commands import ChaseBallCommand
from supermimic.envs.state_io import StateIO
from supermimic.imagination.compiled import VelocityImagStep
from supermimic.imagination.rollout import imagine
from supermimic.imagination.tasks import ChaseBallTask
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import CriticCfg, Policy, PolicyCfg
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.chase_reward import ChaseBallRewardCfg, chase_front_position
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _tuplify(values: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}


@torch.no_grad()
def evaluate(
    checkpoint: Path,
    device: str = "cuda:0",
    num_envs: int = 128,
    episodes: int = 128,
    target_stage: int | None = None,
    debug_vis: bool = False,
    out: Path | None = None,
) -> dict:
    if num_envs < 1 or episodes < 1:
        raise ValueError("num_envs and episodes must be positive")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck.get("config", {})
    env_cfg = config.get("env", {})
    if env_cfg.get("robot_name") != "go2" or env_cfg.get("task_name") != "chase_ball":
        raise ValueError("checkpoint must be trained with go2/chase_ball")
    p_cfg = PolicyCfg(**_tuplify(config["policy"]))
    if p_cfg.goal_encoding != "direct" or p_cfg.goal_dim != 6 or p_cfg.action_output != "tanh":
        raise ValueError("Chase evaluation requires the 6-D direct Actor with tanh action mapping")
    policy = Policy(p_cfg).to(device)
    policy.load_state_dict(ck["policy"], strict=True)
    policy.eval()

    chosen_stage = int(
        target_stage if target_stage is not None
        else ck.get("command_state", {}).get("stage", env_cfg.get("chase_initial_stage", 0))
    )
    if chosen_stage not in (0, 1, 2):
        raise ValueError("target_stage must be 0 (near), 1 (mid), or 2 (far)")
    robot_spec = get_robot_spec("go2")
    args = SuperMimicEnvCfgArgs(
        npz_files=[], robot_name="go2", task_name="chase_ball",
        num_envs=num_envs, history_length=int(env_cfg.get("history_length", p_cfg.history)),
        episode_length_s=float(env_cfg.get("episode_length_s", 10.0)),
        action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=True,
        replay_capacity=max(256, int(env_cfg.get("replay_capacity", 1000))),
        chase_auto_curriculum=False, chase_fixed_stage=chosen_stage,
        chase_tag_radius=float(env_cfg.get("chase_tag_radius", 0.20)),
        chase_approach_weight=float(env_cfg.get("chase_approach_weight", 1.5)),
        chase_near_weight=float(env_cfg.get("chase_near_weight", 3.0)),
        chase_tag_weight=float(env_cfg.get("chase_tag_weight", 15.0)),
        chase_tag_sigma=float(env_cfg.get("chase_tag_sigma", 0.30)),
        chase_facing_weight=float(env_cfg.get("chase_facing_weight", 0.5)),
        velocity_min_safe_height=float(env_cfg.get("velocity_min_safe_height", 0.22)),
        velocity_fall_height=float(env_cfg.get("velocity_fall_height", 0.17)),
        velocity_fall_tilt_deg=float(env_cfg.get("velocity_fall_tilt_deg", 70.0)),
        debug_vis=debug_vis, seed=123,
    )
    env = make_env(args, device=device)
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("chase_ball")
        if not isinstance(command, ChaseBallCommand):
            raise TypeError("environment did not construct ChaseBallCommand")
        state_io = StateIO(env)
        recorder = env.recorder_manager.get_term("replay")
        dt = float(env.step_dt)
        nj = robot_spec.num_joints
        root_lin_start = 7 + nj
        initial_state = state_io.read().clone()
        init_front = chase_front_position(initial_state)
        initial_distance = torch.linalg.vector_norm(command.goal[:, :2] - init_front[:, :2], dim=-1)
        distance_sum = torch.zeros(num_envs, device=device)
        approach_sum = torch.zeros(num_envs, device=device)
        sample_count = torch.zeros(num_envs, device=device)
        ep_steps = torch.zeros(num_envs, device=device)
        rows: list[dict[str, float]] = []
        action_saturated = 0.0
        action_count = 0.0

        while len(rows) < episodes:
            state = state_io.read()
            goal = command.goal
            front = chase_front_position(state)
            delta = goal[:, :2] - front[:, :2]
            dist = torch.linalg.vector_norm(delta, dim=-1)
            direction = delta / dist[:, None].clamp_min(1.0e-6)
            velocity = state[:, root_lin_start:root_lin_start + 2]
            distance_sum += dist
            approach_sum += (velocity * direction).sum(-1)
            sample_count += 1
            ep_steps += 1

            with torch.no_grad():
                action = policy(obs["policy"], obs["goal"])
            ptr = recorder.buffer.ptr.clone()
            obs, _, terminated, truncated, _ = env.step(action)
            # The first row written after ptr contains the just-executed action,
            # including terminal steps (the extra terminal state row follows it).
            env_ids = torch.arange(num_envs, device=device)
            executed = recorder.buffer.a_exec[env_ids, ptr]
            action_saturated += float((executed.abs() >= 0.999).sum())
            action_count += float(executed.numel())
            done = terminated | truncated
            ids = done.nonzero(as_tuple=False).flatten()
            if not ids.numel():
                continue

            for env_id in ids.tolist():
                terminal_idx = int((recorder.buffer.ptr[env_id] - 1) % recorder.buffer.T)
                final_state = recorder.buffer.s[env_id, terminal_idx]
                ball_state = recorder.buffer.goal[env_id, terminal_idx]
                final_front = chase_front_position(final_state[None])[0]
                final_distance = float(torch.linalg.vector_norm(final_front[:2] - ball_state[:2]))
                fell = bool(terminated[env_id])
                tagged = final_distance <= command.cfg.tag_radius and not fell
                timed_out = bool(truncated[env_id]) and not tagged
                displacement = float(torch.linalg.vector_norm(final_state[:2] - initial_state[env_id, :2]))
                steps = max(float(ep_steps[env_id]), 1.0)
                rows.append({
                    "initial_ball_distance": float(initial_distance[env_id]),
                    "mean_ball_distance": float(distance_sum[env_id] / sample_count[env_id].clamp_min(1.0)),
                    "final_ball_distance": final_distance,
                    "tag": float(tagged),
                    "time_to_tag": float(ep_steps[env_id] * dt) if tagged else float("nan"),
                    "robot_displacement": displacement,
                    "mean_approach_velocity": float(approach_sum[env_id] / steps),
                    "fall": float(fell),
                    "timeout": float(timed_out),
                })
                if len(rows) >= episodes:
                    break

            reset_state = state_io.read()
            for env_id in ids.tolist():
                initial_state[env_id] = reset_state[env_id]
                init_front[env_id] = chase_front_position(reset_state[env_id:env_id + 1])[0]
                initial_distance[env_id] = torch.linalg.vector_norm(command.goal[env_id, :2] - init_front[env_id, :2])
                distance_sum[env_id] = 0.0
                approach_sum[env_id] = 0.0
                sample_count[env_id] = 0.0
                ep_steps[env_id] = 0.0

        def mean(key: str) -> float:
            values = [r[key] for r in rows if torch.isfinite(torch.tensor(r[key]))]
            return sum(values) / max(len(values), 1)

        # Use the same checkpoint WM to score one- and eight-step real replay
        # windows, then its frozen dynamics to report 32-step task survival.
        wm_cfg_dict = dict(config["world_model"])
        wm_cfg_dict["hidden"] = tuple(wm_cfg_dict["hidden"])
        wm_cfg_dict["term_hidden"] = tuple(wm_cfg_dict["term_hidden"])
        wm_cfg_dict["state_spec"] = StateSpecCfg(**_tuplify(wm_cfg_dict["state_spec"]))
        wm_cfg = WorldModelCfg(**wm_cfg_dict)
        fk = G1Kinematics(load_robot_spec_model("go2"), device=device)
        heads = AnalyticHeads(
            fk, robot_spec.tracked_body_names, robot_spec.anchor_body_name, robot_spec.root_body_name,
            env.scene["robot"].data.soft_joint_pos_limits[0].clone(),
            env.scene["robot"].data.default_joint_pos[0].clone(),
            default_base_height=robot_spec.default_base_height,
        )
        wm = WorldModel(wm_cfg, heads, device)
        wm.load_state_dict(ck["wm"], strict=True)
        wm.eval()
        buffer = recorder.buffer
        mask8 = buffer.window_mask(8)
        if bool(mask8.any()):
            nse1_starts = buffer.sample_windows(min(256, int(mask8.sum())), 1)
            nse8_starts = buffer.sample_windows(min(256, int(mask8.sum())), 8)
            nse1 = float(wm.rollout_error(nse1_starts.states, nse1_starts.a_exec, nse1_starts.xi)[0])
            nse8 = float(wm.rollout_error(nse8_starts.states, nse8_starts.a_exec, nse8_starts.xi)[-1])
            start = buffer.sample_starts(min(128, int(mask8.sum())), int(env_cfg.get("history_length", 10)))
            task = ChaseBallTask(
                default_joint_pos=env.scene["robot"].data.default_joint_pos[0].clone(),
                dt=robot_spec.control_dt,
                reward_cfg=ChaseBallRewardCfg(
                    approach_weight=float(env_cfg.get("chase_approach_weight", 1.5)),
                    near_weight=float(env_cfg.get("chase_near_weight", 3.0)),
                    tag_weight=float(env_cfg.get("chase_tag_weight", 15.0)),
                    tag_radius=float(env_cfg.get("chase_tag_radius", 0.20)),
                    tag_sigma=float(env_cfg.get("chase_tag_sigma", 0.30)),
                    facing_weight=float(env_cfg.get("chase_facing_weight", 0.5)),
                    minimum_height=float(env_cfg.get("velocity_min_safe_height", 0.22)),
                ),
                fall_height=float(env_cfg.get("velocity_fall_height", 0.17)),
                fall_tilt_deg=float(env_cfg.get("velocity_fall_tilt_deg", 70.0)),
            )
            step_fn = VelocityImagStep(wm, task, task.default_joint_pos, compile_mode=None)
            roll = imagine(wm, policy, task, start, 32, obs_noise=False, step_fn=step_fn)
            survival32 = float(roll.survival[-1].mean())
        else:
            nse1 = nse8 = survival32 = float("nan")

        result = {
            "checkpoint": str(checkpoint),
            "iteration": int(ck.get("iteration", 0)),
            "world_model_updates": int(ck.get("num_updates", 0)),
            "actor_updates": int(ck.get("actor_updates", 0)),
            "target_stage": chosen_stage,
            "episodes": len(rows),
            "mean_initial_ball_distance_m": mean("initial_ball_distance"),
            "mean_ball_distance_m": mean("mean_ball_distance"),
            "final_ball_distance_m": mean("final_ball_distance"),
            "tag_rate": mean("tag"),
            "mean_time_to_tag_s": mean("time_to_tag"),
            "mean_robot_displacement_m": mean("robot_displacement"),
            "mean_approach_velocity_mps": mean("mean_approach_velocity"),
            "fall_rate": mean("fall"),
            "timeout_rate": mean("timeout"),
            "action_saturation_fraction": action_saturated / max(action_count, 1.0),
            "world_model_nse_1step": nse1,
            "world_model_nse_8step": nse8,
            "imagination_survival_32": survival32,
        }
        encoded = json.dumps(result, indent=2)
        print(encoded)
        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(encoded + "\n")
        return result
    finally:
        env.close()


def main(
    checkpoint: Path,
    device: str = "cuda:0",
    num_envs: int = 128,
    episodes: int = 128,
    target_stage: int | None = None,
    debug_vis: bool = False,
    out: Path | None = None,
) -> None:
    evaluate(checkpoint, device, num_envs, episodes, target_stage, debug_vis, out)


if __name__ == "__main__":
    tyro.cli(main)
