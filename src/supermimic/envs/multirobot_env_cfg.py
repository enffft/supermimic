"""Minimal Go2 and M1 environment assembly; the verified G1 config stays in tracking_env_cfg."""

from __future__ import annotations

import math

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
from supermimic.envs.mdp.commands import ChaseBallCommandCfg, MultiClipMotionCommandCfg, VelocityCommandCfg, WorldPointCommandCfg
from supermimic.envs.tracking_env_cfg import QuietSimulationCfg, SuperMimicEnvCfgArgs
from supermimic.robots.mjlab_assets import get_robot_entity_cfg
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.velocity_reward import VELOCITY_REWARD_NAMES, go2_velocity_reward_term
from supermimic.tasks.chase_reward import CHASE_REWARD_NAMES, go2_chase_reward_term
from supermimic.tasks.world_point_reward import WORLD_POINT_REWARD_NAMES, m1_world_point_reward_term


def make_multirobot_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    robot = get_robot_spec(a.robot_name)
    motion_task = a.task_name == "motion_tracking"
    velocity_task = a.task_name == "velocity_command"
    chase_task = a.task_name == "chase_ball"
    world_point_task = a.task_name == "world_point_reach"
    if not (motion_task or velocity_task or chase_task or world_point_task):
        raise ValueError(f"unknown task {a.task_name!r}; use motion_tracking, velocity_command, chase_ball, or world_point_reach")
    if (velocity_task or chase_task) and robot.name != "go2":
        raise ValueError(f"{a.task_name} is currently defined only for Go2")
    if world_point_task and robot.name != "m1_piper":
        raise ValueError("world_point_reach is currently defined only for M1-Piper")
    if motion_task and robot.name == "go2":
        raise ValueError("Go2 has no motion bank; select velocity_command or chase_ball")
    if a.motion_goal_schema == "full_body" and not (motion_task and robot.name == "g1"):
        raise ValueError("full_body Motion Goal schema is supported only for G1 motion_tracking")
    if motion_task and a.anchor_obs and a.anchor_orientation_obs:
        raise ValueError("choose position+orientation Anchor or orientation-only Anchor")
    if motion_task and not a.npz_files:
        raise ValueError(f"{robot.name} motion_tracking requires converted NPZ motion files")
    if a.anchor_obs and a.anchor_orientation_obs:
        raise ValueError("choose position+orientation Anchor or orientation-only Anchor")

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
        elif a.anchor_orientation_obs:
            goal_terms["anchor_ori"] = ObservationTermCfg(
                func=track_mdp.motion_anchor_ori_b,
                params={"command_name": "motion"},
                noise=noise(-0.05, 0.05),
            )
    elif velocity_task:
        goal_terms = {"velocity_command": ObservationTermCfg(func=sm_obs.velocity_goal, params={"command_name": "velocity"})}
    elif chase_task:
        goal_terms = {"chase_ball": ObservationTermCfg(func=sm_obs.chase_ball_goal, params={"command_name": "chase_ball"})}
    else:
        goal_terms = {"world_point_body": ObservationTermCfg(func=sm_obs.world_point_goal_body, params={"command_name": "world_point"})}
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
                preserve_order=True, use_default_offset=True,
                noise_std=a.velocity_gaussian_noise_std if robot.name == "go2" else a.action_noise_std,
                ou_env_fraction=a.velocity_ou_env_fraction if robot.name == "go2" else 0.0,
                ou_noise_std=a.velocity_ou_noise_std if robot.name == "go2" else 0.0,
                ou_theta=a.velocity_ou_theta,
                action_clip=1.0 if robot.name == "go2" else float("inf"),
            )
        }
        action_name = "joint_pos"

    if motion_task:
        commands = {
            "motion": MultiClipMotionCommandCfg(
                npz_files=a.npz_files, anchor_body_name=robot.anchor_body_name,
                root_body_name=robot.root_body_name, body_names=robot.tracked_body_names,
                goal_joint_names=robot.amp_joint_names or None,
                goal_schema=a.motion_goal_schema,
                entity_name="robot", future_frames=a.future_frames, bin_frames=a.bin_frames,
                pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)} if a.reset_noise else {},
                velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)} if a.reset_noise else {},
                joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
                sampling_mode=a.sampling_mode, compact_library=False,
                debug_vis=a.debug_vis,
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
    elif velocity_task:
        commands = {
            "velocity": VelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(a.velocity_resample_seconds, a.velocity_resample_seconds),
                auto_curriculum=a.velocity_curriculum,
                debug_vis=a.velocity_debug_vis,
                initial_stage=a.velocity_curriculum_initial_stage,
                fixed_stage=a.velocity_curriculum_fixed_stage,
                stage_a_vx_range=a.velocity_stage_a_vx_range,
                stage_b_vx_range=a.velocity_stage_b_vx_range,
                stage_b_vy_range=a.velocity_stage_b_vy_range,
                stage_b_yaw_range=a.velocity_stage_b_yaw_range,
                stage_c_vx_range=a.velocity_stage_c_vx_range,
                stage_c_vy_range=a.velocity_stage_c_vy_range,
                stage_c_yaw_range=a.velocity_stage_c_yaw_range,
                stage_a_static_fraction=a.velocity_stage_a_static_fraction,
                stage_b_keep_a_fraction=a.velocity_stage_b_keep_a_fraction,
                stage_c_keep_a_fraction=a.velocity_stage_c_keep_a_fraction,
                stage_c_keep_b_fraction=a.velocity_stage_c_keep_b_fraction,
                promotion_window_episodes=a.velocity_promotion_window_episodes,
                consecutive_passing_windows=a.velocity_consecutive_passing_windows,
                min_moving_episodes=a.velocity_min_moving_episodes,
                min_yaw_episodes=a.velocity_min_yaw_episodes,
                min_command_bin_episodes=a.velocity_min_command_bin_episodes,
                required_magnitude_coverage=a.velocity_required_magnitude_coverage,
                max_fall_rate=a.velocity_max_fall_rate,
                stage_a_max_vel_rmse=a.velocity_stage_a_max_vel_rmse,
                stage_a_max_static_speed=a.velocity_stage_a_max_static_speed,
                stage_b_max_vel_rmse=a.velocity_stage_b_max_vel_rmse,
                stage_b_max_yaw_rmse=a.velocity_stage_b_max_yaw_rmse,
                fixed_command=a.velocity_fixed_command,
                show_actual_velocity=a.velocity_show_actual_velocity,
            )
        }
        # Every adapter calls the same pure reward core; RewardManager applies dt once.
        rewards = {
            name: RewardTermCfg(
                func=go2_velocity_reward_term, weight=1.0,
                params={
                    "term_name": name,
                    "linear_tracking": a.velocity_reward_linear_tracking,
                    "yaw_tracking": a.velocity_reward_yaw_tracking,
                    "command_direction": a.velocity_reward_direction,
                    "min_safe_height": a.velocity_min_safe_height,
                },
            )
            for name in VELOCITY_REWARD_NAMES
        }
        tilt_feature_threshold = 1.0 - math.cos(math.radians(a.velocity_fall_tilt_deg))
        terminations = {
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "root_height": TerminationTermCfg(
                func=sm_term.free_root_too_low,
                params={"asset_cfg": SceneEntityCfg("robot"), "threshold": a.velocity_fall_height},
            ),
            "root_tilt": TerminationTermCfg(func=sm_term.free_root_tilted, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": tilt_feature_threshold}),
        }
        goal_dim = 3
        command_name = "velocity"
    elif chase_task:
        commands = {
            "chase_ball": ChaseBallCommandCfg(
                entity_name="robot",
                ball_center_height=0.09,
                tag_radius=a.chase_tag_radius,
                auto_curriculum=a.chase_auto_curriculum,
                initial_stage=a.chase_initial_stage,
                fixed_stage=a.chase_fixed_stage,
                promotion_window_episodes=a.chase_promotion_window_episodes,
                consecutive_passing_windows=a.chase_consecutive_passing_windows,
                promotion_success_rate=a.chase_promotion_success_rate,
                debug_vis=a.debug_vis,
            )
        }
        rewards = {
            name: RewardTermCfg(
                func=go2_chase_reward_term, weight=1.0,
                params={
                    "term_name": name,
                    "approach_weight": a.chase_approach_weight,
                    "near_weight": a.chase_near_weight,
                    "tag_weight": a.chase_tag_weight,
                    "tag_radius": a.chase_tag_radius,
                    "tag_sigma": a.chase_tag_sigma,
                    "facing_weight": a.chase_facing_weight,
                    "minimum_height": a.velocity_min_safe_height,
                },
            )
            for name in CHASE_REWARD_NAMES
        }
        tilt_feature_threshold = 1.0 - math.cos(math.radians(a.velocity_fall_tilt_deg))
        terminations = {
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "chase_tag": TerminationTermCfg(
                func=sm_term.chase_tag,
                params={"command_name": "chase_ball"},
                time_out=True,
            ),
            "root_height": TerminationTermCfg(
                func=sm_term.free_root_too_low,
                params={"asset_cfg": SceneEntityCfg("robot"), "threshold": a.velocity_fall_height},
            ),
            "root_tilt": TerminationTermCfg(
                func=sm_term.free_root_tilted,
                params={"asset_cfg": SceneEntityCfg("robot"), "threshold": tilt_feature_threshold},
            ),
        }
        goal_dim = 6
        command_name = "chase_ball"
    else:
        commands = {
            "world_point": WorldPointCommandCfg(
                entity_name="robot",
                near_target_fraction=a.m1_near_target_fraction,
                near_x=a.m1_near_x, near_y=a.m1_near_y, near_z=a.m1_near_z,
                far_x=a.m1_far_x, far_y=a.m1_far_y, far_z=a.m1_far_z,
                success_distance=a.m1_success_distance,
                auto_curriculum=a.m1_target_curriculum,
                initial_stage=a.m1_initial_target_stage,
                fixed_stage=a.m1_fixed_target_stage,
                promotion_window_episodes=a.m1_target_promotion_window_episodes,
                promotion_success_rate=a.m1_target_promotion_success_rate,
                consecutive_passing_windows=a.m1_target_consecutive_passing_windows,
                debug_vis=a.debug_vis,
            )
        }
        rewards = {
            name: RewardTermCfg(func=m1_world_point_reward_term, weight=1.0, params={"term_name": name})
            for name in WORLD_POINT_REWARD_NAMES
        }
        tilt_feature_threshold = 1.0 - math.cos(math.radians(a.velocity_fall_tilt_deg))
        terminations = {
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "root_height": TerminationTermCfg(
                func=sm_term.free_root_too_low,
                params={"asset_cfg": SceneEntityCfg("robot"), "threshold": a.m1_fall_height},
            ),
            "root_tilt": TerminationTermCfg(
                func=sm_term.free_root_tilted,
                params={"asset_cfg": SceneEntityCfg("robot"), "threshold": tilt_feature_threshold},
            ),
        }
        goal_dim = 3
        command_name = "world_point"

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

    # mjWarp's nconmax/njmax are PER WORLD, not totals across vectorized envs.
    # Scaling these by num_envs multiplies the arrays by num_envs twice and can
    # exhaust host RAM before a large GPU rollout starts.
    # Initial compiled contacts are about 84 for Go2 and 216 for M1. Keep
    # comfortable per-world headroom without multiplying these by num_envs.
    nconmax = 128 if robot.name == "go2" else 256
    njmax = 512 if robot.name == "go2" else 1024
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": get_robot_entity_cfg(robot.name)}),
        observations=observations, actions=actions, commands=commands, events=events,
        rewards=rewards, terminations=terminations, recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=robot.root_body_name, distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4, episode_length_s=a.episode_length_s, seed=a.seed,
    )


def make_multirobot_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_multirobot_env_cfg(a), device=device)
