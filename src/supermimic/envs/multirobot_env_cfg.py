"""Minimal Go2 and M1 environment assembly; the verified G1 config stays in tracking_env_cfg."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import MixedM1ActionCfg, NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg, VelocityCommandCfg
from supermimic.envs.tracking_env_cfg import QuietSimulationCfg, SuperMimicEnvCfgArgs
from supermimic.robots.mjlab_assets import get_robot_entity_cfg
from supermimic.robots.spec import get_robot_spec


def zero_reward(env) -> object:
    """Placeholder physical reward for collection only; no Go2 training reward is implied."""
    import torch

    return torch.zeros(env.num_envs, device=env.device)


def make_multirobot_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    robot = get_robot_spec(a.robot_name)
    motion_task = a.task_name == "motion_tracking"
    velocity_task = a.task_name == "velocity_command"
    if not (motion_task or velocity_task):
        raise ValueError(f"unknown task {a.task_name!r}; use motion_tracking or velocity_command")
    if velocity_task and robot.name != "go2":
        raise ValueError("velocity_command is currently defined only for Go2")
    if motion_task and robot.name == "go2":
        raise ValueError("Go2 has no motion bank; select velocity_command")
    if motion_task and not a.npz_files:
        raise ValueError(f"{robot.name} motion_tracking requires converted NPZ motion files")

    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    joint_cfg = SceneEntityCfg("robot", joint_names=robot.joint_names)
    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=sm_obs.joint_pos_rel_masked,
            params={"biased": True, "wheel_joint_names": robot.wheel_joint_names, "asset_cfg": joint_cfg},
            noise=noise(-0.01, 0.01),
        ),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, params={"asset_cfg": joint_cfg}, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    if motion_task:
        goal_terms = {"future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"})}
        if a.anchor_obs:
            goal_terms.update({
                "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
            })
    else:
        goal_terms = {"velocity_command": ObservationTermCfg(func=sm_obs.velocity_goal, params={"command_name": "velocity"})}
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms, concatenate_terms=True, enable_corruption=a.obs_noise,
            history_length=a.history_length, flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(terms=goal_terms, concatenate_terms=True, enable_corruption=a.obs_noise),
    }

    if robot.name == "m1_piper":
        actions = {
            "mixed": MixedM1ActionCfg(
                entity_name="robot", joint_names=robot.joint_names,
                wheel_joint_names=robot.wheel_joint_names, action_scales=robot.action_scales,
                noise_std=a.action_noise_std,
            )
        }
        action_name = "mixed"
    else:
        scale = dict(zip(robot.joint_names, robot.action_scales))
        actions = {
            "joint_pos": NoisyJointPositionActionCfg(
                entity_name="robot", actuator_names=robot.joint_names, scale=scale,
                preserve_order=True, use_default_offset=True, noise_std=a.action_noise_std,
            )
        }
        action_name = "joint_pos"

    if motion_task:
        commands = {
            "motion": MultiClipMotionCommandCfg(
                npz_files=a.npz_files, anchor_body_name=robot.anchor_body_name,
                root_body_name=robot.root_body_name, body_names=robot.tracked_body_names,
                goal_joint_names=robot.amp_joint_names or None,
                entity_name="robot", future_frames=a.future_frames, bin_frames=a.bin_frames,
                pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)} if a.reset_noise else {},
                velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)} if a.reset_noise else {},
                joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
                sampling_mode=a.sampling_mode, compact_library=False,
            )
        }
        rewards = {
            "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
            "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
            "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
            "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
            "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
            "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
            "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
            "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": joint_cfg}),
        }
        terminations = {
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
            "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
            "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
        }
        # Motion Goals are rebuilt from (clip, frame) during imagination; only
        # reference-free command tasks need a Goal snapshot in each replay row.
        goal_dim = 0
        command_name = "motion"
    else:
        commands = {
            "velocity": VelocityCommandCfg(
                resampling_time_range=(10.0, 10.0),
            )
        }
        # Reward is intentionally zero in this interface bring-up. A task reward must be
        # specified before policy optimization is enabled.
        rewards = {"collection_only": RewardTermCfg(func=zero_reward, weight=0.0)}
        terminations = {
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "root_height": TerminationTermCfg(
                func=sm_term.free_root_too_low,
                params={"asset_cfg": SceneEntityCfg("robot"), "threshold": max(0.05, robot.default_base_height - a.root_height_threshold)},
            ),
            "root_tilt": TerminationTermCfg(func=sm_term.free_root_tilted, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": a.root_tilt_threshold}),
        }
        goal_dim = 3
        command_name = "velocity"

    events: dict[str, EventTermCfg] = {
        "reset_scene_to_default": EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset"),
        "foot_friction": EventTermCfg(
            mode="reset" if a.domain_randomization else "startup", func=dr.geom_friction,
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=robot.foot_geom_regex), "operation": "abs",
                    "ranges": (0.3, 1.2) if a.domain_randomization else (1.0, 1.0), "shared_random": True},
        )
    }
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup", func=dr.body_com_offset,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.root_body_name,)), "operation": "add",
                    "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.025, 0.025)}},
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset", func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={
            "capacity": a.replay_capacity, "foot_geom_regex": robot.foot_geom_regex,
            "com_body_name": robot.root_body_name, "command_name": command_name,
            "action_name": action_name, "goal_dim": goal_dim,
        })

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": get_robot_entity_cfg(robot.name)}),
        observations=observations, actions=actions, commands=commands, events=events,
        rewards=rewards, terminations=terminations, recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=robot.root_body_name, distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        # Warp's initial contact capacity is for the whole vectorized model, not one env.
        sim=QuietSimulationCfg(nconmax=max(256, a.num_envs * 16), njmax=max(512, a.num_envs * 128), mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4, episode_length_s=a.episode_length_s, seed=a.seed,
    )


def make_multirobot_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_multirobot_env_cfg(a), device=device)
