"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig, validate_g1_actor_contract
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.tasks import ChaseBallTask, MotionTrackingTask, VelocityCommandTask, WorldPointReachTask, M1RewardTask
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy, M1ActionReferenceActorCfg, M1ActionReferenceActor
from supermimic.motion_prior import load_m1_motion_prior
from supermimic.motion_prior.exploration import M1BoundedGaussianExplorer
from supermimic.motion_prior.trainer import M1AlternatingTrainer
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed
from supermimic.training.schedule import actor_ramp_scale, scheduled_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel
from supermimic.tasks.action_contract import add_bounded_action_exploration
from supermimic.tasks.chase_reward import ChaseBallRewardCfg
from supermimic.tasks.velocity_reward import VelocityRewardCfg, VelocityRobotParams
from supermimic.tasks.world_point_reward import WorldPointRobotParams, m1_tcp_proxy_indices


M1_FIRST_PHASE_CLIPS = (
    "a_wheel_000000",
    "a_wheel_000002",
    "b_arm_000006",
    "c_assisted_000029",
    "c_assisted_000039",
    "c_assisted_000149",
)
M1_MOTION_SETS = {
    "A": M1_FIRST_PHASE_CLIPS[:2],
    "B": M1_FIRST_PHASE_CLIPS[:3],
    "C": M1_FIRST_PHASE_CLIPS,
}
M1_FIRST_PHASE_MANIFEST = Path("configs/m1_piper_first6.csv")


def _termination_mode(task_name: str) -> str:
    if task_name == "motion_tracking":
        return "learned_reference"
    if task_name == "chase_ball":
        return "analytic_fall_plus_tag_success"
    return "analytic_height_tilt"


def _format_m1_metric(value: float | None, precision: int = 4) -> str:
    """Format missing or nonfinite diagnostics as unavailable, not a NaN result."""
    if value is None:
        return "n/a"
    numeric = float(value)
    return f"{numeric:.{precision}f}" if math.isfinite(numeric) else "n/a"


def _format_m1_aggregate(aggregate: dict[str, float], count: int, key: str,
                         precision: int = 4) -> str:
    value = aggregate.get(key)
    if value is None:
        return "n/a"
    return _format_m1_metric(float(value) / max(count, 1), precision)


class Trainer:
    @staticmethod
    def _resolved_num_envs(robot_name: str, requested: int | None = None, task_name: str | None = None) -> int:
        if requested is not None:
            return requested
        name = get_robot_spec(robot_name).name
        if name == "m1_piper":
            if task_name == "m1_prior_joint":
                return 16
            return 256 if task_name == "world_point_reach" else 2048
        return 3072 if name == "go2" else 4096

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.resume is not None and cfg.init_wm_from is not None:
            raise ValueError("--resume and --init-wm-from are mutually exclusive")
        if cfg.m1_joint_training.enabled and cfg.resume is not None:
            raise ValueError("M1 Action22 runs use --m1-joint-training.resume, not legacy --resume")
        if cfg.m1_joint_training.resume is not None and cfg.init_wm_from is not None:
            raise ValueError("M1 Action22 resume includes the WM; do not also pass --init-wm-from")
        is_m1_reward = cfg.env.task_name == "m1_reward"
        is_m1_prior_joint = cfg.env.task_name == "m1_prior_joint"
        if is_m1_reward:
            if cfg.env.m1_pose_profile is None:
                cfg.env.m1_pose_profile = "dataset_v3_3_l_high_carry"
            if cfg.env.m1_action_contract not in (None, "v3"):
                raise ValueError("m1_reward requires Action Contract v3")
            cfg.env.m1_action_contract = "v3"
            if cfg.env.m1_joint_task_type == "balanced":
                cfg.env.m1_joint_balanced_fixed_quota = True
        if is_m1_prior_joint or is_m1_reward:
            if cfg.env.m1_pose_profile is None:
                cfg.env.m1_pose_profile = (
                    "dataset_v3_3_l_high_carry" if (
                        cfg.m1_joint_training.task_type == "L"
                        or cfg.env.m1_joint_balanced_fixed_quota
                    )
                    else "dataset_carry"
                )
            if cfg.env.m1_action_contract not in (None, "v3"):
                raise ValueError("M1 Action22 training requires --env.m1-action-contract v3")
            cfg.env.m1_action_contract = "v3"
            robot_spec = get_robot_spec(
                cfg.env.robot_name,
                m1_pose_profile=cfg.env.m1_pose_profile,
                m1_action_contract=cfg.env.m1_action_contract,
            )
        else:
            robot_spec = get_robot_spec(cfg.env.robot_name)
        self.robot_spec = robot_spec
        self.robot_joint_names = tuple(robot_spec.joint_names)
        self.actuator_model_version = robot_spec.actuator_model_version
        self.robot_model_sha256 = (
            self._checkpoint_sha256(robot_spec.model_path)
            if robot_spec.model_path is not None and robot_spec.model_path.is_file()
            else None
        )
        cfg.env.num_envs = self._resolved_num_envs(robot_spec.name, cfg.env.num_envs, cfg.env.task_name)
        is_motion = cfg.env.task_name == "motion_tracking"
        is_velocity = cfg.env.task_name == "velocity_command"
        is_chase = cfg.env.task_name == "chase_ball"
        is_world_point = cfg.env.task_name == "world_point_reach"
        is_m1_prior_joint = cfg.env.task_name == "m1_prior_joint"
        if cfg.m1_joint_training.enabled != is_m1_prior_joint:
            raise ValueError("m1_joint_training.enabled and env.task_name='m1_prior_joint' must be enabled together")
        if cfg.env.task_name not in ("motion_tracking", "velocity_command", "chase_ball", "world_point_reach", "m1_prior_joint", "m1_reward"):
            raise ValueError(f"unknown task {cfg.env.task_name!r}")
        if is_m1_prior_joint and cfg.env.obs_noise:
            raise ValueError(
                "M1 Action22 v1 reconstructs noise-free proprioception for BPTT; "
                "run with --env.no-obs-noise until observation noise is replayed explicitly"
            )
        if is_m1_prior_joint and cfg.env.action_noise_std != 0.0:
            raise ValueError(
                "M1 Action22 v1 uses explicit collection-time exploration; keep "
                "--env.action-noise-std 0 so Replay records the exact executed action"
            )
        if (is_velocity or is_chase) and robot_spec.name != "go2":
            raise ValueError(f"{cfg.env.task_name} is currently supported only for Go2")
        if (is_world_point or is_m1_prior_joint or is_m1_reward) and robot_spec.name != "m1_piper":
            raise ValueError(f"{cfg.env.task_name} is currently supported only for M1-Piper")
        if robot_spec.name == "go2" and not (is_velocity or is_chase):
            raise ValueError("Go2 has no Motion Library; select velocity_command or chase_ball")
        if robot_spec.name == "m1_piper" and not (is_motion or is_world_point or is_m1_prior_joint or is_m1_reward):
            raise ValueError("M1-Piper supports motion_tracking, world_point_reach, and opt-in m1_prior_joint")
        if cfg.env.anchor_obs is None:
            cfg.env.anchor_obs = robot_spec.name == "m1_piper" and is_motion
        validate_g1_actor_contract(cfg.env, robot_spec.name)
        if cfg.env.anchor_obs and cfg.env.anchor_orientation_obs:
            raise ValueError("choose position+orientation Anchor or orientation-only Anchor")
        if (cfg.env.anchor_obs or cfg.env.anchor_orientation_obs) and not is_motion:
            raise ValueError("Anchor observations are defined only for motion_tracking")
        if robot_spec.name == "m1_piper" and is_motion:
            if cfg.data.library is not None:
                raise ValueError("M1 packed libraries bypass the first-phase validated clip manifest")
            motion_set = cfg.data.m1_motion_set.upper()
            if motion_set not in M1_MOTION_SETS:
                raise ValueError(f"M1 motion set must be one of {tuple(M1_MOTION_SETS)}, got {cfg.data.m1_motion_set!r}")
            selected_set = set(M1_MOTION_SETS[motion_set])
            if cfg.data.clips is None:
                cfg.data.clips = list(M1_MOTION_SETS[motion_set])
            unapproved = set(cfg.data.clips) - selected_set
            if unapproved:
                raise ValueError(f"M1 motion set {motion_set} rejects clips outside its selection: {sorted(unapproved)}")
            if cfg.data.manifest is None:
                cfg.data.manifest = M1_FIRST_PHASE_MANIFEST

        if is_velocity or is_chase:
            cfg.wm_trainer.batch_size = cfg.env.velocity_wm_batch_size
            cfg.policy_trainer.batch_size = cfg.env.velocity_actor_batch_size
        elif is_world_point:
            cfg.wm_trainer.batch_size = cfg.env.m1_world_point_wm_batch_size
            cfg.policy_trainer.batch_size = cfg.env.m1_world_point_actor_batch_size
        elif is_m1_prior_joint:
            cfg.env.m1_joint_task_type = cfg.m1_joint_training.task_type
            cfg.env.m1_joint_l_task_fraction = cfg.m1_joint_training.l_task_fraction
            cfg.wm_trainer.batch_size = cfg.m1_joint_training.batch_size
            cfg.wm_trainer.micro_batch_size = min(
                cfg.wm_trainer.micro_batch_size or cfg.m1_joint_training.micro_batch_size,
                cfg.m1_joint_training.micro_batch_size,
            )
            cfg.wm_trainer.batches_per_update = max(cfg.wm_trainer.batches_per_update, 1)

        # Build every network dimension from the selected robot/task while leaving the
        # default G1 values and initialization path unchanged.
        n_joints = robot_spec.num_joints
        full_body_goal = is_motion and robot_spec.name == "g1" and cfg.env.motion_goal_schema == "full_body"
        goal_term_sizes = (268,) if full_body_goal else (robot_spec.amp_dim, robot_spec.amp_dim, 3, 3, 3, 1) if is_motion else (3, 3) if is_chase else (8,) if (is_m1_prior_joint or is_m1_reward) else (3,)
        goal_term_scale = (1.0,) if full_body_goal or is_m1_prior_joint or is_m1_reward else (1.0, 5.0, 1.0, 2.0, 1.0, 1.0) if is_motion else (1.0, 1.0) if is_chase else (1.0,)
        anchor_dim = 9 if cfg.env.anchor_obs else 6 if cfg.env.anchor_orientation_obs else 0
        per_frame_goal_dim = 268 if full_body_goal else (2 * robot_spec.amp_dim + 10)
        goal_dim = cfg.env.future_frames * per_frame_goal_dim + anchor_dim if is_motion else 6 if is_chase else 8 if (is_m1_prior_joint or is_m1_reward) else 3
        cfg.policy.num_joints = n_joints
        cfg.policy.history = cfg.env.history_length
        # The imagination rollout must use the same flattened observation
        # history as the environment/Actor.  Keep this synchronized when a
        # task overrides the default (Go2-SuperMimic uses five frames).
        cfg.policy_trainer.history = cfg.env.history_length
        cfg.policy.goal_dim = goal_dim
        cfg.policy.goal_interface = "motion" if is_motion else "velocity" if is_velocity else "chase_ball" if is_chase else "m1_reward" if is_m1_reward else "m1_prior_joint" if is_m1_prior_joint else "world_point_reach"
        # Keep G1/M1 on the original encoded goal path; only Go2 gets the compact
        # proprioception + velocity-command actor.
        cfg.policy.goal_encoding = cfg.env.velocity_policy_encoding if is_velocity else "direct" if (is_world_point or is_chase or is_m1_prior_joint or is_m1_reward) else "encoded"
        # Direct task actors must stay inside the action interval during BPTT.
        # A smooth tanh has useful gradients near the boundary, unlike relying on
        # a hard execution clamp after an unbounded decoder. Old checkpoints omit
        # this setting and retain their historical identity output in Play.
        cfg.policy.action_output = cfg.env.direct_action_output if cfg.policy.goal_encoding == "direct" else "identity"
        cfg.policy.goal_term_sizes = goal_term_sizes
        cfg.policy.goal_term_scale = goal_term_scale
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.anchor_orientation_obs = cfg.env.anchor_orientation_obs
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        cfg.world_model.num_joints = n_joints
        cfg.world_model.action_dim = n_joints
        cfg.world_model.xi_dim = n_joints + 4
        cfg.world_model.termination_base_height = robot_spec.default_base_height
        if is_m1_reward:
            cfg.world_model.height_threshold = robot_spec.default_base_height - cfg.env.m1_fall_height
            cfg.world_model.tilt_threshold = 1.0 - math.cos(math.radians(cfg.env.velocity_fall_tilt_deg))
        if robot_spec.name == "go2":
            if cfg.env.velocity_fall_height >= robot_spec.default_base_height:
                raise ValueError("Go2 fall height must be below the 0.42 m default base height")
            cfg.world_model.height_threshold = robot_spec.default_base_height - cfg.env.velocity_fall_height
            cfg.world_model.tilt_threshold = 1.0 - math.cos(math.radians(cfg.env.velocity_fall_tilt_deg))
        cfg.world_model.state_spec.num_joints = n_joints
        cfg.world_model.state_spec.dt = robot_spec.control_dt
        cfg.world_model.state_spec.wheel_indices = tuple(robot_spec.joint_names.index(name) for name in robot_spec.wheel_joint_names)
        cfg.world_model.state_spec.xi_offset = (0.75,) + (0.0,) * n_joints + (0.0,) * 3
        cfg.world_model.state_spec.xi_scale = (0.45,) + (0.01,) * n_joints + (0.05,) * 3

        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if is_m1_prior_joint and self.world > 1:
            raise NotImplementedError("M1 Action22-reference training is currently single-GPU only")
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if not is_motion:
            files = []
            if cfg.data.library is not None:
                raise ValueError(f"{cfg.env.task_name} does not consume a motion library")
        elif cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            npz_dir = cfg.data.npz_dir
            if robot_spec.name == "m1_piper" and npz_dir == Path("data/npz/lafan1"):
                from supermimic.data.m1_motion_convert import DEFAULT_BANK, DEFAULT_OUTPUT, convert_m1_motion_bank

                npz_dir = DEFAULT_OUTPUT
                source_names = sorted(p.stem for p in (DEFAULT_BANK / "motions").glob("*.txt"))
                selected = [name for name in source_names if cfg.data.clips is None or name in set(cfg.data.clips)]
                if cfg.data.limit is not None:
                    selected = selected[: cfg.data.limit]
                if any(not (npz_dir / f"{name}.npz").exists() for name in selected):
                    convert_m1_motion_bank(DEFAULT_BANK, npz_dir, names=selected)
                cfg.data.npz_dir = npz_dir
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if is_motion and not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot_name=robot_spec.name,
                task_name=cfg.env.task_name,
                m1_pose_profile=cfg.env.m1_pose_profile,
                m1_action_contract=cfg.env.m1_action_contract,
                num_envs=cfg.env.num_envs,
                history_length=cfg.env.history_length,
                future_frames=cfg.env.future_frames,
                motion_goal_schema=cfg.env.motion_goal_schema,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                anchor_orientation_obs=cfg.env.anchor_orientation_obs,
                velocity_gaussian_noise_std=cfg.env.velocity_gaussian_noise_std,
                velocity_ou_env_fraction=cfg.env.velocity_ou_env_fraction,
                velocity_ou_noise_std=cfg.env.velocity_ou_noise_std,
                velocity_ou_theta=cfg.env.velocity_ou_theta,
                velocity_reward_linear_tracking=cfg.env.velocity_reward_linear_tracking,
                velocity_reward_yaw_tracking=cfg.env.velocity_reward_yaw_tracking,
                velocity_reward_direction=cfg.env.velocity_reward_direction,
                velocity_reward_tracking_sigma_sq=cfg.env.velocity_reward_tracking_sigma_sq,
                velocity_reward_zero_speed_baseline=cfg.env.velocity_reward_zero_speed_baseline,
                go2_gait=cfg.env.go2_gait,
                velocity_curriculum=cfg.env.velocity_curriculum,
                velocity_curriculum_initial_stage=cfg.env.velocity_curriculum_initial_stage,
                velocity_curriculum_fixed_stage=cfg.env.velocity_curriculum_fixed_stage,
                velocity_stage_a_vx_range=cfg.env.velocity_stage_a_vx_range,
                velocity_stage_b_vx_range=cfg.env.velocity_stage_b_vx_range,
                velocity_stage_b_vy_range=cfg.env.velocity_stage_b_vy_range,
                velocity_stage_b_yaw_range=cfg.env.velocity_stage_b_yaw_range,
                velocity_stage_c_vx_range=cfg.env.velocity_stage_c_vx_range,
                velocity_stage_c_vy_range=cfg.env.velocity_stage_c_vy_range,
                velocity_stage_c_yaw_range=cfg.env.velocity_stage_c_yaw_range,
                velocity_stage_a_static_fraction=cfg.env.velocity_stage_a_static_fraction,
                velocity_stage_b_keep_a_fraction=cfg.env.velocity_stage_b_keep_a_fraction,
                velocity_stage_c_keep_a_fraction=cfg.env.velocity_stage_c_keep_a_fraction,
                velocity_stage_c_keep_b_fraction=cfg.env.velocity_stage_c_keep_b_fraction,
                velocity_promotion_window_episodes=cfg.env.velocity_promotion_window_episodes,
                velocity_consecutive_passing_windows=cfg.env.velocity_consecutive_passing_windows,
                velocity_min_moving_episodes=cfg.env.velocity_min_moving_episodes,
                velocity_min_yaw_episodes=cfg.env.velocity_min_yaw_episodes,
                velocity_min_command_bin_episodes=cfg.env.velocity_min_command_bin_episodes,
                velocity_required_magnitude_coverage=cfg.env.velocity_required_magnitude_coverage,
                velocity_max_fall_rate=cfg.env.velocity_max_fall_rate,
                velocity_stage_a_max_vel_rmse=cfg.env.velocity_stage_a_max_vel_rmse,
                velocity_stage_a_max_static_speed=cfg.env.velocity_stage_a_max_static_speed,
                velocity_stage_b_max_vel_rmse=cfg.env.velocity_stage_b_max_vel_rmse,
                velocity_stage_b_max_yaw_rmse=cfg.env.velocity_stage_b_max_yaw_rmse,
                velocity_resample_seconds=cfg.env.velocity_resample_seconds,
                velocity_min_safe_height=cfg.env.velocity_min_safe_height,
                velocity_fall_height=cfg.env.velocity_fall_height,
                velocity_fall_tilt_deg=cfg.env.velocity_fall_tilt_deg,
                chase_auto_curriculum=cfg.env.chase_auto_curriculum,
                chase_initial_stage=cfg.env.chase_initial_stage,
                chase_fixed_stage=cfg.env.chase_fixed_stage,
                chase_promotion_window_episodes=cfg.env.chase_promotion_window_episodes,
                chase_consecutive_passing_windows=cfg.env.chase_consecutive_passing_windows,
                chase_promotion_success_rate=cfg.env.chase_promotion_success_rate,
                chase_tag_radius=cfg.env.chase_tag_radius,
                chase_approach_weight=cfg.env.chase_approach_weight,
                chase_near_weight=cfg.env.chase_near_weight,
                chase_tag_weight=cfg.env.chase_tag_weight,
                chase_tag_sigma=cfg.env.chase_tag_sigma,
                chase_facing_weight=cfg.env.chase_facing_weight,
                chase_joint_torque_weight=cfg.env.chase_joint_torque_weight,
                chase_joint_acceleration_weight=cfg.env.chase_joint_acceleration_weight,
                m1_reward=cfg.env.m1_reward,
                m1_target=cfg.env.m1_target,
                m1_fall_height=cfg.env.m1_fall_height,
                m1_success_distance=cfg.env.m1_success_distance,
                m1_target_curriculum=cfg.env.m1_target_curriculum,
                m1_initial_target_stage=cfg.env.m1_initial_target_stage,
                m1_fixed_target_stage=cfg.env.m1_fixed_target_stage,
                m1_target_promotion_window_episodes=cfg.env.m1_target_promotion_window_episodes,
                m1_target_promotion_success_rate=cfg.env.m1_target_promotion_success_rate,
                m1_target_consecutive_passing_windows=cfg.env.m1_target_consecutive_passing_windows,
                m1_near_target_fraction=cfg.env.m1_near_target_fraction,
                m1_near_x=cfg.env.m1_near_x,
                m1_near_y=cfg.env.m1_near_y,
                m1_near_z=cfg.env.m1_near_z,
                m1_far_x=cfg.env.m1_far_x,
                m1_far_y=cfg.env.m1_far_y,
                m1_far_z=cfg.env.m1_far_z,
                m1_joint_velocity_weight=cfg.env.m1_joint_velocity_weight,
                m1_joint_acceleration_weight=cfg.env.m1_joint_acceleration_weight,
                m1_leg_action_magnitude_weight=cfg.env.m1_leg_action_magnitude_weight,
                m1_joint_task_type=cfg.env.m1_joint_task_type,
                m1_joint_l_task_fraction=cfg.env.m1_joint_l_task_fraction,
                m1_joint_balanced_fixed_quota=cfg.env.m1_joint_balanced_fixed_quota,
                m1_joint_vx_range=cfg.env.m1_joint_vx_range,
                m1_joint_vx_stop_fraction=cfg.env.m1_joint_vx_stop_fraction,
                m1_joint_vx_reverse_fraction=cfg.env.m1_joint_vx_reverse_fraction,
                m1_joint_vy_range=cfg.env.m1_joint_vy_range,
                m1_joint_wz_range=cfg.env.m1_joint_wz_range,
                m1_joint_target_x_range=cfg.env.m1_joint_target_x_range,
                m1_joint_target_y_range=cfg.env.m1_joint_target_y_range,
                m1_joint_target_z_range=cfg.env.m1_joint_target_z_range,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        command_name = "motion" if is_motion else "velocity" if is_velocity else "chase_ball" if is_chase else "m1_prior_joint" if (is_m1_prior_joint or is_m1_reward) else "world_point"
        self.command = env.command_manager.get_term(command_name)
        self.lib = getattr(self.command, "library", None)
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]
        if tuple(robot.joint_names) != tuple(robot_spec.joint_names):
            raise ValueError(
                f"{robot_spec.name}: mjlab entity order {tuple(robot.joint_names)} "
                f"does not match Actor/RobotSpec order {tuple(robot_spec.joint_names)}"
            )

        fk = G1Kinematics(load_robot_spec_model(robot_spec.name), device=self.device)
        heads = AnalyticHeads(
            fk, robot_spec.tracked_body_names, robot_spec.anchor_body_name, robot_spec.root_body_name,
            robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
            default_base_height=robot_spec.default_base_height,
            reward_anchor_at_start=robot_spec.name == "m1_piper",
        )
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.recorder.real_transition_callback = self.wm.update_real_transition
        if is_m1_prior_joint:
            self.policy = M1ActionReferenceActor(
                M1ActionReferenceActorCfg(
                    history=cfg.env.history_length,
                    leg_delta_max_normalized=cfg.m1_joint_training.actor_leg_delta_max_normalized,
                    arm_delta_max_normalized=cfg.m1_joint_training.actor_arm_delta_max_normalized,
                    wheel_delta_max_normalized=cfg.m1_joint_training.actor_wheel_delta_max_normalized,
                ),
            ).to(self.device)
            self.critic = None  # M1 v1 uses explicit critic-free differentiable task losses
        else:
            self.policy = Policy(cfg.policy).to(self.device)
            self.critic = Critic(cfg.critic, feat_dim=self.wm.spec.feat_dim, goal_dim=goal_dim, xi_dim=n_joints + 4, pcfg=cfg.policy).to(self.device)
        self.pretrained_wm = False
        self.init_wm_from_path: str | None = None
        self.init_wm_sha256: str | None = None
        self.init_wm_source_iteration: int | None = None
        if cfg.init_wm_from is not None:
            self._initialize_world_model(cfg.init_wm_from, robot_spec.joint_names)
        if is_motion:
            self.task = MotionTrackingTask(
                self.lib, cfg.env.future_frames, cfg.env.anchor_obs,
                cfg.env.anchor_orientation_obs,
                goal_schema=cfg.env.motion_goal_schema,
            )
        elif is_velocity:
            reward_cfg = VelocityRewardCfg(
                linear_tracking=cfg.env.velocity_reward_linear_tracking,
                yaw_tracking=cfg.env.velocity_reward_yaw_tracking,
                command_direction=cfg.env.velocity_reward_direction,
                tracking_sigma_sq=cfg.env.velocity_reward_tracking_sigma_sq,
                zero_speed_baseline=cfg.env.velocity_reward_zero_speed_baseline,
                gait=cfg.env.go2_gait,
                min_safe_height=cfg.env.velocity_min_safe_height,
            )
            default_joint_pos = robot.data.default_joint_pos[0].detach().clone()
            reward_params = VelocityRobotParams(
                default_joint_pos=default_joint_pos,
                soft_joint_limits=robot.data.soft_joint_pos_limits[0].detach().clone(),
                action_scales=default_joint_pos.new_tensor(robot_spec.action_scales),
                kp=default_joint_pos.new_tensor(robot_spec.kp),
                kd=default_joint_pos.new_tensor(robot_spec.kd),
                effort_limits=default_joint_pos.new_tensor(robot_spec.effort_limits),
                dt=robot_spec.control_dt,
                cfg=reward_cfg,
            )
            self.task = VelocityCommandTask(
                params=reward_params,
                dt=robot_spec.control_dt,
                reward_cfg=reward_cfg,
                fall_height=cfg.env.velocity_fall_height,
                tilt_feature_threshold=1.0 - math.cos(math.radians(cfg.env.velocity_fall_tilt_deg)),
            )
        elif is_chase:
            chase_default = robot.data.default_joint_pos[0].detach().clone()
            chase_reward_params = VelocityRobotParams(
                default_joint_pos=chase_default,
                soft_joint_limits=robot.data.soft_joint_pos_limits[0].detach().clone(),
                action_scales=chase_default.new_tensor(robot_spec.action_scales),
                kp=chase_default.new_tensor(robot_spec.kp), kd=chase_default.new_tensor(robot_spec.kd),
                effort_limits=chase_default.new_tensor(robot_spec.effort_limits), dt=robot_spec.control_dt,
            )
            self.task = ChaseBallTask(
                default_joint_pos=chase_default, params=chase_reward_params,
                dt=robot_spec.control_dt,
                reward_cfg=ChaseBallRewardCfg(
                    approach_weight=cfg.env.chase_approach_weight,
                    near_weight=cfg.env.chase_near_weight,
                    tag_weight=cfg.env.chase_tag_weight,
                    tag_radius=cfg.env.chase_tag_radius,
                    tag_sigma=cfg.env.chase_tag_sigma,
                    facing_weight=cfg.env.chase_facing_weight,
                    minimum_height=cfg.env.velocity_min_safe_height,
                    joint_torque_weight=cfg.env.chase_joint_torque_weight,
                    joint_acceleration_weight=cfg.env.chase_joint_acceleration_weight,
                ),
                fall_height=cfg.env.velocity_fall_height,
                fall_tilt_deg=cfg.env.velocity_fall_tilt_deg,
            )
        elif is_world_point or is_m1_reward:
            tcp_indices = m1_tcp_proxy_indices(heads.fk)
            world_point_params = WorldPointRobotParams(
                default_joint_pos=robot.data.default_joint_pos[0].detach().clone(),
                soft_joint_limits=robot.data.soft_joint_pos_limits[0].detach().clone(),
                dt=robot_spec.control_dt,
                fall_height=cfg.env.m1_fall_height,
                success_distance=cfg.env.m1_success_distance,
                tilt_feature_threshold=1.0 - math.cos(math.radians(cfg.env.velocity_fall_tilt_deg if is_m1_reward else 70.0)),
                joint_velocity_weight=cfg.env.m1_joint_velocity_weight,
                joint_acceleration_weight=cfg.env.m1_joint_acceleration_weight,
                leg_action_magnitude_weight=cfg.env.m1_leg_action_magnitude_weight,
            )
            task_cls = M1RewardTask if is_m1_reward else WorldPointReachTask
            reward_kwargs = {"reward_cfg": cfg.env.m1_reward, "base_height": robot_spec.default_base_height,
                             "terminal_distance_weight": cfg.policy_trainer.m1_terminal_distance_weight,
                             "resample_on_success": cfg.env.m1_target.resample_on_success,
                             "task_balanced": cfg.env.m1_joint_task_type == "balanced"} if is_m1_reward else {}
            self.task = task_cls(
                **reward_kwargs,
                params=world_point_params,
                fk=heads.fk,
                tcp_body_indices=tcp_indices,
                dt=robot_spec.control_dt,
            )
        else:
            self.task = None
        for m in (self.wm, self.policy, self.critic):
            if m is None:
                continue
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = None
        self.m1_joint_trainer = None
        if is_m1_prior_joint:
            prior = load_m1_motion_prior(
                cfg.m1_joint_training.prior_checkpoint,
                device=self.device,
                expected_sha256=cfg.m1_joint_training.expected_prior_sha256,
                input_compatibility=cfg.m1_joint_training.prior_input_compatibility,
            )
            tcp_index = heads.fk.body_index("piper_x_link7")
            self.m1_joint_trainer = M1AlternatingTrainer(
                wm=self.wm,
                wm_trainer=self.wm_trainer,
                prior=prior,
                actor=self.policy,
                buffer=self.buffer,
                default_joint_pos=robot.data.default_joint_pos[0].detach().clone(),
                soft_limits=robot.data.soft_joint_pos_limits[0].detach().clone(),
                fk=heads.fk,
                tcp_index=tcp_index,
                cfg=cfg.m1_joint_training,
                robot_joint_names=self.robot_joint_names,
                robot_model_sha256=self.robot_model_sha256,
                command=self.command,
                robot_spec=robot_spec,
            )
            if cfg.m1_joint_training.resume is not None:
                self._load_m1_joint_checkpoint(cfg.m1_joint_training.resume)
            if self.is_main:
                print(
                    f"[INFO] M1 CVAE strict load: {prior.source['path']} "
                    f"sha256={prior.source['sha256']} step={prior.source['step']}; "
                    f"prior_state_source={cfg.m1_joint_training.state_source}; "
                    f"input_compatibility={prior.input_compatibility}", flush=True,
                )
        else:
            self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.task, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            critic_count = n_params(self.critic) / 1e6 if self.critic is not None else 0.0
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {critic_count:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward if is_motion else 1.0
        self.has_tracking_return = is_motion
        self.is_go2_velocity = robot_spec.name == "go2" and is_velocity
        self.is_go2_chase = robot_spec.name == "go2" and is_chase
        self.is_go2_task = self.is_go2_velocity or self.is_go2_chase
        self.is_m1_motion = robot_spec.name == "m1_piper" and is_motion
        self.is_m1_reward = is_m1_reward
        if is_m1_reward:
            self._m1_reward_hold = torch.zeros(env.num_envs, device=self.device)
            self._m1_reward_success = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.is_m1_world_point = robot_spec.name == "m1_piper" and is_world_point
        self.is_m1_prior_joint = robot_spec.name == "m1_piper" and is_m1_prior_joint
        self.go2_action_stats = self._empty_go2_action_stats() if self.is_go2_task else None
        self.task_diag_sum: dict[str, torch.Tensor] = {}
        self.motion_names = [Path(f).stem for f in files]
        self.motion_return_sum = torch.zeros(len(self.motion_names), device=self.device)
        self.motion_return_count = torch.zeros(len(self.motion_names), device=self.device)
        self.motion_success_sum = torch.zeros(len(self.motion_names), device=self.device)
        self.motion_success_count = torch.zeros(len(self.motion_names), device=self.device)
        if self.is_m1_motion:
            self.m1_arm_ids = torch.tensor(
                [robot.joint_names.index(f"piper_x_joint{i}") for i in range(1, 7)],
                dtype=torch.long, device=self.device,
            )
            self.m1_tcp_body_idx = self.lib.body_names.index("piper_x_link7")
        else:
            self.m1_arm_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self.m1_tcp_body_idx = -1
        self.m1_ep_clip_return = torch.zeros(env.num_envs, device=self.device)
        self.m1_metric_count = torch.zeros(env.num_envs, device=self.device)
        self.m1_ep_metrics = {
            key: torch.zeros(env.num_envs, device=self.device)
            for key in ("root_pos_error", "root_ori_error", "arm_joint_error", "tcp_pos_error")
        }
        self.m1_step_metric_sum: dict[str, torch.Tensor] = {}
        self.m1_step_metric_count = torch.zeros((), device=self.device)
        self.real_reward_term_sum: dict[str, torch.Tensor] = {}
        self.iteration = 0
        self.num_updates = 0
        self.wm_only_updates = 0
        self.actor_updates = 0
        self.actor_start_iteration: int | None = None
        self._utd_acc = 0.0
        if cfg.m1_joint_training.resume is not None:
            self._load_m1_joint_checkpoint(cfg.m1_joint_training.resume)
        elif cfg.resume is not None:
            self.load(cfg.resume)
        if self.is_main:
            (self.log_dir / "config.json").write_text(
                json.dumps(self._run_config_snapshot(), indent=2, default=str)
            )

    # ------------------------------------------------------------------ checkpoints

    @staticmethod
    def _as_tuple(value):
        return tuple(value) if isinstance(value, (list, tuple)) else value

    @staticmethod
    def _checkpoint_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _initialize_world_model(self, path: Path, joint_names: tuple[str, ...]) -> None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"pretrained World Model checkpoint does not exist: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # Init-WM intentionally consumes only the model and its provenance.  Actor,
        # critic, optimizers and replay belong to the new experiment.
        required = {"wm", "config"}
        missing = required - set(ck)
        if missing:
            raise ValueError(f"{path}: incomplete training checkpoint, missing {sorted(missing)}")
        saved = ck["config"]
        saved_env = saved.get("env", {})
        saved_wm = saved.get("world_model", {})
        saved_spec = saved_wm.get("state_spec", {})
        saved_robot = ck.get("robot_name", saved.get("robot_name", saved_env.get("robot_name")))
        saved_task = ck.get("task_name", saved.get("task_name", saved_env.get("task_name")))
        current_robot = self.cfg.env.robot_name
        current_task = self.cfg.env.task_name
        compatible_tasks = {
            "go2": {"velocity_command", "chase_ball"},
            "m1_piper": {"world_point_reach", "m1_prior_joint"},
        }
        if saved_robot != current_robot:
            raise ValueError(f"--init-wm-from robot mismatch: saved={saved_robot!r}, current={current_robot!r}")
        if saved_task not in compatible_tasks.get(current_robot, set()) or current_task not in compatible_tasks.get(current_robot, set()):
            raise ValueError(f"--init-wm-from task is not compatible with {current_robot}: saved={saved_task!r}, current={current_task!r}")
        saved_joints = tuple(ck.get("robot_joint_names", saved.get("robot_joint_names", ())))
        if saved_joints != tuple(joint_names):
            raise ValueError(f"pretrained World Model joint order mismatch: saved={saved_joints}, current={tuple(joint_names)}")
        saved_actuator = ck.get("actuator_model_version", saved.get("actuator_model_version"))
        if saved_actuator != self.actuator_model_version:
            raise ValueError(
                "pretrained World Model actuator model is incompatible: "
                f"saved={saved_actuator!r}, current={self.actuator_model_version!r}"
            )
        saved_model_hash = ck.get("robot_model_sha256", saved.get("robot_model_sha256"))
        if saved_model_hash != self.robot_model_sha256:
            raise ValueError(
                "pretrained World Model robot URDF fingerprint is incompatible: "
                f"saved={saved_model_hash!r}, current={self.robot_model_sha256!r}"
            )
        current_robot_spec = self.robot_spec
        physical_fields = {
            "control_dt": float(current_robot_spec.control_dt),
            "action_modes": tuple(current_robot_spec.action_modes),
            "action_scales": tuple(current_robot_spec.action_scales),
            "kp": tuple(current_robot_spec.kp),
            "kd": tuple(current_robot_spec.kd),
            "effort_limits": tuple(current_robot_spec.effort_limits),
        }
        for key, current_value in physical_fields.items():
            saved_value = saved.get(key, saved_env.get(key))
            if saved_value is not None:
                old = self._as_tuple(saved_value)
                if key == "control_dt":
                    if not math.isclose(float(old), float(current_value), rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError(f"pretrained World Model {key} mismatch: saved={old}, current={current_value}")
                elif old != current_value:
                    raise ValueError(f"pretrained World Model {key} mismatch: saved={old}, current={current_value}")
        for key in ("running_normalization", "running_norm_min_std"):
            old = saved_wm.get(key, False if key == "running_normalization" else 1e-2)
            new = getattr(self.cfg.world_model, key)
            if old != new:
                raise ValueError(f"pretrained World Model {key} mismatch: saved={old}, current={new}")
        current_spec = self.robot_spec
        expected_dims = {"num_joints": current_spec.num_joints, "action_dim": current_spec.num_joints, "xi_dim": current_spec.num_joints + 4}
        for key, expected in expected_dims.items():
            actual = saved_wm.get(key)
            if actual != expected or getattr(self.cfg.world_model, key) != expected:
                raise ValueError(f"pretrained World Model {key} mismatch: saved={actual}, expected={expected}")
        expected_state_dim = 13 + 2 * expected_dims["num_joints"]
        saved_state_dim = ck.get("state_dim", saved.get("state_dim"))
        if saved_state_dim is not None and int(saved_state_dim) != expected_state_dim:
            raise ValueError(f"pretrained World Model state dimension mismatch: saved={saved_state_dim}, current={expected_state_dim}")
        if int(self.wm.spec.state_dim) != expected_state_dim:
            raise AssertionError("World Model state dimension invariant failed")

        # Old checkpoints did not persist a named joint-order table. The robot identity
        # plus the 12-D weight layout is checked here; new checkpoints save and compare
        # the explicit RobotSpec order below.
        saved_joint_names = ck.get("robot_joint_names", saved.get("robot_joint_names"))
        if saved_joint_names is not None and tuple(saved_joint_names) != tuple(joint_names):
            raise ValueError(
                "pretrained World Model joint order mismatch: "
                f"saved={tuple(saved_joint_names)}, current={tuple(joint_names)}"
            )
        saved_dt = float(saved_spec.get("dt", float("nan")))
        if not math.isclose(saved_dt, float(self.cfg.world_model.state_spec.dt), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"pretrained World Model control dt mismatch: saved={saved_dt}")
        state_fields = (
            "dt", "num_joints", "wheel_indices", "err_scale", "out_scale", "in_scale",
            "explicit_integration", "xi_offset", "xi_scale",
        )
        current_spec = asdict(self.cfg.world_model.state_spec)
        for key in state_fields:
            if key not in saved_spec:
                raise ValueError(f"pretrained checkpoint has no StateSpec field {key!r}")
            old, new = self._as_tuple(saved_spec[key]), self._as_tuple(current_spec[key])
            if old != new:
                raise ValueError(f"pretrained World Model StateSpec.{key} mismatch: saved={old}, current={new}")
        if tuple(saved_wm.get("hidden", ())) != tuple(self.cfg.world_model.hidden):
            raise ValueError("pretrained World Model Dynamics MLP hidden widths do not match")
        if tuple(saved_wm.get("term_hidden", ())) != tuple(self.cfg.world_model.term_hidden):
            raise ValueError("pretrained World Model Termination Head widths do not match")
        for key in ("height_threshold", "tilt_threshold", "termination_base_height"):
            old, new = float(saved_wm.get(key, float("nan"))), float(getattr(self.cfg.world_model, key))
            if not math.isclose(old, new, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"pretrained World Model {key} termination semantics mismatch: saved={old}, current={new}")
        if not self._tree_is_finite(ck["wm"]):
            raise ValueError("pretrained World Model contains NaN or Inf parameters")

        self.wm.load_state_dict(ck["wm"], strict=True)
        self.pretrained_wm = True
        self.init_wm_from_path = str(path)
        self.init_wm_sha256 = self._checkpoint_sha256(path)
        self.init_wm_source_iteration = int(ck.get("iteration", -1))
        print(
            f"[INFO] initialized {current_robot} World Model from {path} "
            f"(sha256={self.init_wm_sha256}; strict state-dict load)",
            flush=True,
        )

    def _effective_wm_only_target(self) -> int:
        if getattr(self, "is_go2_task", False):
            return 0 if self.pretrained_wm else int(self.cfg.schedule.wm_only_iterations)
        if getattr(self, "is_m1_prior_joint", False):
            # The new Planner is sensitive to WM extrapolation error even when
            # weights came from native M1 Reach. Refit/evaluate on this run's
            # collected distribution before allowing a Planner update.
            return int(self.cfg.schedule.m1_wm_only_iterations)
        if self.is_m1_motion or getattr(self, "is_m1_world_point", False) or getattr(self, "is_m1_reward", False) or getattr(self, "is_m1_prior_joint", False):
            return 0 if self.pretrained_wm else int(self.cfg.schedule.m1_wm_only_iterations)
        if self.cfg.env.robot_name == "g1" and self.cfg.env.task_name == "motion_tracking":
            return int(self.cfg.schedule.g1_motion_wm_only_updates)
        return 0

    def _run_config_snapshot(self) -> dict:
        config = asdict(self.cfg)
        config.update({
            "robot_joint_names": self.robot_joint_names,
            "actuator_model_version": self.actuator_model_version,
            "robot_model_sha256": self.robot_model_sha256,
            "robot_name": self.cfg.env.robot_name,
            "task_name": self.cfg.env.task_name,
            "goal_dim": self.cfg.policy.goal_dim,
            "action_dim": self.cfg.policy.num_joints,
            "state_dim": self.wm.spec.state_dim,
            "policy_mode": self.cfg.policy.goal_encoding,
            "termination_mode": _termination_mode(self.cfg.env.task_name),
            "pretrained_wm": self.pretrained_wm,
            "init_wm_from": self.init_wm_from_path,
            "init_wm_sha256": self.init_wm_sha256,
            "init_wm_source_iteration": self.init_wm_source_iteration,
            "effective_wm_only_updates": self._effective_wm_only_target(),
            "control_dt": self.cfg.world_model.state_spec.dt,
            "action_modes": tuple(self.robot_spec.action_modes),
            "action_scales": tuple(self.robot_spec.action_scales),
            "position_target_bounds": tuple(self.robot_spec.position_target_bounds),
            "default_pose_profile": self.robot_spec.default_pose_profile,
            "action_contract_version": self.robot_spec.action_contract_version,
            "kp": tuple(self.robot_spec.kp),
            "kd": tuple(self.robot_spec.kd),
            "effort_limits": tuple(self.robot_spec.effort_limits),
        })
        if self.is_m1_reward:
            from supermimic.tasks.action_contract import serialize_m1_action_contract
            config["m1_reward_action_mapping"] = serialize_m1_action_contract(self.robot_spec)
        return config

    @staticmethod
    def _tree_is_finite(value) -> bool:
        if isinstance(value, torch.Tensor):
            return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(Trainer._tree_is_finite(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(Trainer._tree_is_finite(v) for v in value)
        return True

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        if self.is_m1_prior_joint:
            state = self.m1_joint_trainer.checkpoint(
                iteration=self.iteration,
                config_snapshot=self._run_config_snapshot(),
                command_state=self.command.state_dict(),
                wm_only_updates=self.wm_only_updates,
            )
            if not self._tree_is_finite(state["wm"]) or not self._tree_is_finite(state["motion_prior"]) \
                    or not self._tree_is_finite(state["actor"]) or not self._tree_is_finite(state["wm_optimizer"]) \
                    or not self._tree_is_finite(state["prior_optimizer"]) or not self._tree_is_finite(state["actor_optimizer"]):
                print(f"[WARN] skipped joint checkpoint {name}: model or optimizer has NaN/Inf", flush=True)
                return
            path = self.log_dir / name
            tmp_path = path.with_name(path.name + ".tmp")
            try:
                torch.save(state, tmp_path)
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            if (name == "ckpt_latest.pt" and self.m1_joint_trainer.actor_optimizer_steps == 0):
                initial_path = self.log_dir / "ckpt_initial_actor.pt"
                if not initial_path.exists():
                    initial_tmp = initial_path.with_name(initial_path.name + ".tmp")
                    try:
                        torch.save(state, initial_tmp)
                        os.replace(initial_tmp, initial_path)
                    finally:
                        if initial_tmp.exists():
                            initial_tmp.unlink()
            return
        state = {
            "training_wall_seconds": getattr(self, "_training_elapsed_before", 0.) + (time.time() - self._training_start_time if hasattr(self, "_training_start_time") else 0.),
            "iteration": self.iteration,
            "num_updates": self.num_updates,
            "wm": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "config": self._run_config_snapshot(),
            "wm_only_updates": self.wm_only_updates,
            "actor_updates": self.actor_updates,
            "actor_start_iteration": self.actor_start_iteration,
            "utd_acc": self._utd_acc,
            "pretrained_wm": self.pretrained_wm,
            "init_wm_from": self.init_wm_from_path,
            "init_wm_sha256": self.init_wm_sha256,
            "init_wm_source_iteration": self.init_wm_source_iteration,
            "effective_wm_only_updates": self._effective_wm_only_target(),
            "policy_mode": self.cfg.policy.goal_encoding,
            "robot_joint_names": self.robot_joint_names,
            "actuator_model_version": self.actuator_model_version,
            "robot_model_sha256": self.robot_model_sha256,
            "robot_name": self.cfg.env.robot_name,
            "task_name": self.cfg.env.task_name,
            "policy_encoding_mode": self.cfg.policy.goal_encoding,
            "action_output_mode": self.cfg.policy.action_output,
            "goal_dim": self.cfg.policy.goal_dim,
            "action_dim": self.cfg.policy.num_joints,
            "state_dim": self.wm.spec.state_dim,
            "termination_mode": _termination_mode(self.cfg.env.task_name),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        if not all(self._tree_is_finite(state[k]) for k in ("wm", "policy", "critic", "wm_opt", "policy_opt", "critic_opt")):
            print(f"[WARN] skipped {name}: model or optimizer state contains NaN/Inf; keeping the previous valid checkpoint", flush=True)
            return
        if hasattr(self.command, "bin_return"):
            state["bin_return"] = self.command.bin_return
            state["bin_memory"] = self.command.bin_memory
        if hasattr(self.command, "state_dict"):
            state["command_state"] = self.command.state_dict()
        action_term = self.recorder.action_term
        if hasattr(action_term, "state_dict"):
            state["action_state"] = action_term.state_dict()
        path = self.log_dir / name
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def load(self, path: Path) -> None:
        if self.is_m1_prior_joint:
            raise ValueError("M1 Action22 runs require --m1-joint-training.resume v4 checkpoint")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        saved_config = ck.get("config", {})
        saved_env = saved_config.get("env", {})
        saved_robot = ck.get("robot_name", saved_config.get("robot_name", saved_env.get("robot_name")))
        saved_task = ck.get("task_name", saved_config.get("task_name", saved_env.get("task_name")))
        if saved_robot is not None and saved_robot != self.cfg.env.robot_name:
            raise ValueError(f"checkpoint robot {saved_robot!r} cannot resume as {self.cfg.env.robot_name!r}")
        if saved_task is not None and saved_task != self.cfg.env.task_name:
            raise ValueError(f"checkpoint task {saved_task!r} cannot resume as {self.cfg.env.task_name!r}")
        saved_actuator = ck.get("actuator_model_version", saved_config.get("actuator_model_version"))
        if self.cfg.env.robot_name == "go2" and saved_actuator != self.actuator_model_version:
            raise ValueError(
                "Go2 checkpoint actuator model is incompatible with this run; "
                "the old knee effort limit changed, so its World Model cannot be resumed"
            )
        saved_model_hash = ck.get("robot_model_sha256", saved_config.get("robot_model_sha256"))
        if saved_model_hash is not None and saved_model_hash != self.robot_model_sha256:
            raise ValueError("checkpoint robot URDF fingerprint differs from the current model")
        saved_norm = saved_config.get("world_model", {}).get("running_normalization", False)
        if saved_norm != self.cfg.world_model.running_normalization:
            raise ValueError(f"checkpoint World Model running_normalization={saved_norm} differs from requested config")
        saved_term_mode = ck.get("termination_mode")
        current_term_mode = _termination_mode(self.cfg.env.task_name)
        if saved_term_mode is not None and saved_term_mode != current_term_mode:
            raise ValueError(f"checkpoint termination mode {saved_term_mode!r} differs from {current_term_mode!r}")
        for key, actual in (("goal_dim", self.cfg.policy.goal_dim), ("action_dim", self.cfg.policy.num_joints), ("state_dim", self.wm.spec.state_dim)):
            expected = ck.get(key)
            if expected is not None and expected != actual:
                raise ValueError(f"checkpoint {key}={expected} but this run expects {actual}")
        saved_policy_mode = ck.get(
            "policy_mode", ck.get("config", {}).get("policy", {}).get("goal_encoding", "encoded")
        )
        if saved_policy_mode != self.cfg.policy.goal_encoding:
            raise ValueError(
                f"checkpoint Actor mode {saved_policy_mode!r} cannot resume into "
                f"{self.cfg.policy.goal_encoding!r}; use it for evaluation or initialize only the World Model"
            )
        saved_action_output = ck.get(
            "action_output_mode", ck.get("config", {}).get("policy", {}).get("action_output", "identity")
        )
        if saved_action_output != self.cfg.policy.action_output:
            raise ValueError(
                f"checkpoint action output mapping {saved_action_output!r} cannot resume into "
                f"{self.cfg.policy.action_output!r}; start a fresh learner or use the saved mapping for evaluation"
            )
        if self.is_m1_reward:
            from supermimic.tasks.action_contract import serialize_m1_action_contract
            if saved_config.get("m1_reward_action_mapping") != serialize_m1_action_contract(self.robot_spec):
                raise ValueError("m1_reward resume requires the identical saved pose/action contract")
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if hasattr(self.command, "bin_return") and "bin_return" in ck:
            self.command.bin_return.copy_(ck["bin_return"])
            self.command.bin_memory.copy_(ck["bin_memory"])
        if "command_state" in ck and hasattr(self.command, "load_state_dict"):
            self.command.load_state_dict(ck["command_state"])
        action_term = self.recorder.action_term
        if "action_state" in ck and hasattr(action_term, "load_state_dict"):
            action_term.load_state_dict(ck["action_state"])
        self._training_elapsed_before = float(ck.get("training_wall_seconds", 0.))
        self.iteration = ck["iteration"]
        self.num_updates = ck.get("num_updates", 0)
        self.wm_only_updates = ck.get("wm_only_updates", 0)
        self.actor_updates = ck.get("actor_updates", 0)
        self.actor_start_iteration = ck.get("actor_start_iteration", ck.get("actor_start_update"))
        self._utd_acc = ck.get("utd_acc", 0.0)
        self.pretrained_wm = bool(ck.get("pretrained_wm", False))
        self.init_wm_from_path = ck.get("init_wm_from")
        self.init_wm_sha256 = ck.get("init_wm_sha256")
        self.init_wm_source_iteration = ck.get("init_wm_source_iteration")
        if "torch_rng_state" in ck:
            torch.set_rng_state(ck["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in ck:
            torch.cuda.set_rng_state(ck["cuda_rng_state"].cpu(), self.device)
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    def _load_m1_joint_checkpoint(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"M1 joint checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.iteration = self.m1_joint_trainer.load_checkpoint(
            checkpoint, expected_source=self.m1_joint_trainer.prior.source,
        )
        self.num_updates = self.m1_joint_trainer.wm_update_events
        self.actor_updates = self.m1_joint_trainer.actor_optimizer_steps
        self.wm_only_updates = int(checkpoint.get("wm_only_updates", 0))
        saved_config = checkpoint.get("config", {})
        self.pretrained_wm = bool(saved_config.get("pretrained_wm", False))
        self.init_wm_from_path = saved_config.get("init_wm_from")
        self.init_wm_sha256 = saved_config.get("init_wm_sha256")
        self.init_wm_source_iteration = saved_config.get("init_wm_source_iteration")
        print(f"[INFO] resumed versioned M1 joint checkpoint from {path} at iteration {self.iteration}", flush=True)

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs, state_history: torch.Tensor | None = None,
            reference_phase: torch.Tensor | None = None) -> torch.Tensor:
        if self.is_m1_prior_joint:
            if state_history is None or reference_phase is None:
                raise ValueError("M1 Action22 acting needs State57 history and current plan phase")
            action, _ = self.m1_joint_trainer.act(obs["policy"], obs["goal"], state_history, reference_phase)
            return action
        return self.policy(obs["policy"], obs["goal"])

    @torch.no_grad()
    def _measure_m1_joint_task_step(
        self, state: torch.Tensor, task_type: torch.Tensor,
        velocity_command: torch.Tensor, fixed_world_target: torch.Tensor,
        initial_base_position: torch.Tensor, previous_ee_distance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Measure the just-finished real state against its pre-reset task context."""
        from supermimic.motion_prior.metrics import root_tilt_and_stability
        from supermimic.motion_prior.velocity import velocity_command_errors

        task_type = task_type.to(device=state.device, dtype=torch.long)
        is_m = task_type == 1
        velocity_xy_error, velocity_xy_squared_error, yaw_rate_error, body_vertical_velocity = (
            velocity_command_errors(state, velocity_command)
        )
        from supermimic.robots.spec import M1_JOINTS

        wheel_ids = [M1_JOINTS.index(n) for n in ("FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT")]
        fk = self.m1_joint_trainer.fk
        bodies = fk(state[:, :29], state[:, 29:])
        tcp_position = bodies["pos"][:, self.m1_joint_trainer.tcp_index]
        distance = torch.linalg.vector_norm(tcp_position - fixed_world_target, dim=-1)
        distance = torch.where(is_m, distance, torch.zeros_like(distance))
        progress = torch.where(is_m, previous_ee_distance - distance, torch.zeros_like(distance))
        displacement = torch.linalg.vector_norm(state[:, :2] - initial_base_position[:, :2], dim=-1)
        base_tilt, base_stable = root_tilt_and_stability(
            state[:, 3:7], state[:, 2] + self.env.scene.env_origins[:, 2],
            minimum_height_m=self.command.cfg.success_minimum_base_height_m,
            maximum_tilt_rad=self.command.cfg.success_maximum_tilt_rad,
        )
        return {
            "base_velocity_xy_error": torch.linalg.vector_norm(velocity_xy_error, dim=-1),
            "base_velocity_xy_squared_error": velocity_xy_squared_error,
            "base_vx_squared_error": velocity_xy_error[:, 0].square(),
            "base_vy_squared_error": velocity_xy_error[:, 1].square(),
            "command_vx": velocity_command[:, 0],
            "command_vy": velocity_command[:, 1],
            "command_wz": velocity_command[:, 2],
            "base_vx_body": velocity_command[:, 0] + velocity_xy_error[:, 0],
            "base_vy_body": velocity_command[:, 1] + velocity_xy_error[:, 1],
            "positive_command_negative_vx": (
                (velocity_command[:, 0] > 0.05)
                & ((velocity_command[:, 0] + velocity_xy_error[:, 0]) < 0.0)
            ).to(state.dtype),
            "base_yaw_rate": state[:, 34],
            "wheel_rate_abs_mean": state[:, 35:57][:, wheel_ids].abs().mean(-1),
            "base_vertical_velocity": body_vertical_velocity,
            "yaw_rate_error": yaw_rate_error.abs(),
            "ee_distance": distance,
            "ee_progress": progress,
            "base_displacement": displacement,
            "world_displacement": displacement,
            "ee_success": (
                (distance < self.command.cfg.success_distance) & is_m & base_stable
            ).float(),
            "joint_limit_violation": (
                (state[:, 7:29] < self.env.scene["robot"].data.soft_joint_pos_limits[0, :, 0])
                | (state[:, 7:29] > self.env.scene["robot"].data.soft_joint_pos_limits[0, :, 1])
            ).float().mean(-1),
            "base_height": state[:, 2],
            "base_tilt_rad": base_tilt,
            "base_stable": base_stable.to(state.dtype),
        }

    @torch.no_grad()
    def _update_m1_metrics(self, alive: torch.Tensor) -> None:
        if not self.is_m1_motion:
            return
        from mjlab.utils.lab_api.math import quat_error_magnitude

        cmd = self.command
        robot = self.env.scene["robot"]
        values = {
            "root_pos_error": torch.linalg.vector_norm(cmd.root_pos_w - cmd.robot_root_pos_w, dim=-1),
            "root_ori_error": quat_error_magnitude(cmd.root_quat_w, cmd.robot_root_quat_w),
            "arm_joint_error": torch.linalg.vector_norm(
                cmd.joint_pos[:, self.m1_arm_ids] - robot.data.joint_pos[:, self.m1_arm_ids], dim=-1
            ) / self.m1_arm_ids.numel() ** 0.5,
            "tcp_pos_error": torch.linalg.vector_norm(
                cmd.body_pos_w[:, self.m1_tcp_body_idx] - cmd.robot_body_pos_w[:, self.m1_tcp_body_idx], dim=-1
            ),
        }
        self.m1_metric_count += alive
        self.m1_step_metric_count += alive.sum()
        for key, value in values.items():
            self.m1_ep_metrics[key] += value * alive
            self.m1_step_metric_sum[key] = self.m1_step_metric_sum.get(key, value.new_zeros(())) + value[alive].sum()

    @torch.no_grad()
    def _finish_m1_episodes(self, done: torch.Tensor, terminated: torch.Tensor, clip_ids: torch.Tensor) -> None:
        if not self.is_m1_motion:
            return
        ids = done.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        selected_clip = clip_ids[ids]
        returns = self.m1_ep_clip_return[ids]
        self.motion_return_sum.index_add_(0, selected_clip, returns)
        self.motion_return_count.index_add_(0, selected_clip, torch.ones_like(returns))

        counts = self.m1_metric_count[ids].clamp_min(1.0)
        means = {key: value[ids] / counts for key, value in self.m1_ep_metrics.items()}
        # Explicit engineering evaluation thresholds; they do not affect the reward.
        success = (
            (means["root_pos_error"] < 0.10)
            & (means["root_ori_error"] < 0.25)
            & (means["arm_joint_error"] < 0.20)
            & (means["tcp_pos_error"] < 0.08)
            & ~terminated[ids]
            & (self.m1_metric_count[ids] > 0)
        ).float()
        self.motion_success_sum.index_add_(0, selected_clip, success)
        self.motion_success_count.index_add_(0, selected_clip, torch.ones_like(success))
        self.m1_ep_clip_return[ids] = 0.0
        self.m1_metric_count[ids] = 0.0
        for value in self.m1_ep_metrics.values():
            value[ids] = 0.0

    def _record_real_reward_terms(self) -> None:
        names = self.env.reward_manager.active_terms
        for name in names:
            value = self.env.reward_manager.get_term(name).mean().detach()
            self.real_reward_term_sum[name] = self.real_reward_term_sum.get(name, value.new_zeros(())) + value
        if self.is_m1_reward:
            from supermimic.motion_prior.velocity import velocity_command_errors
            from supermimic.tasks.world_point_reward import tcp_position_world

            cache = self.env._supermimic_m1_reward_cache
            state, context = cache["state_next"], cache["context"]
            is_l, is_m = context[:, 0] < 0.5, context[:, 0] >= 0.5
            xy_error, xy_sq, yaw_error, _ = velocity_command_errors(state, context[:, 1:4])
            distance = (tcp_position_world(state, self.task.fk, self.task.tcp_body_indices) - context[:, 4:7]).norm(dim=-1)
            stable = (self.task.termination_features(state) < 0).all(-1)
            within = is_m & stable & (distance < self.cfg.env.m1_success_distance)
            if self.cfg.env.m1_target.resample_on_success:
                reached = (cache["terms"]["m_success"] > 0) & stable
                self._m1_reward_success |= reached
                m_metrics = {"ee_distance": distance, "within_8cm": within.float(),
                             "goals_per_env_minute": reached.float() * (60. / self.env.step_dt),
                             "episode_any_goal_success_so_far": self._m1_reward_success.float()}
            else:
                self._m1_reward_hold = torch.where(within, self._m1_reward_hold + self.env.step_dt, 0.0)
                self._m1_reward_success |= self._m1_reward_hold >= 0.5
                m_metrics = {"ee_distance": distance, "within_8cm": within.float(),
                             "stable_hold_success_so_far": self._m1_reward_success.float()}
            metrics = {}
            for mask, prefix, values in (
                (is_l, "real/L", {"velocity_xy_mse": xy_sq, "yaw_rate_mse": yaw_error.square(),
                                 "command_vx": context[:, 1], "base_vx": xy_error[:, 0] + context[:, 1],
                                 "wrong_direction": ((context[:, 1] > 0.05) & (xy_error[:, 0] + context[:, 1] < 0)).float()}),
                (is_m, "real/M", m_metrics),
            ):
                if bool(mask.any()):
                    metrics.update({f"{prefix}/{key}": value[mask].mean() for key, value in values.items()})
            done = self.env.reset_terminated | self.env.reset_time_outs
            self._m1_reward_hold[done] = 0.0
            self._m1_reward_success[done] = False
            for key, value in metrics.items():
                self.task_diag_sum[key] = self.task_diag_sum.get(key, value.new_zeros(())) + value.detach()
        if self.cfg.env.robot_name.lower().replace("-", "_") == "g1" and self.cfg.env.task_name == "motion_tracking":
            # Diagnostics only: expose drift that a pure-proprio Actor cannot
            # observe, without changing the reward or policy inputs.
            for metric_name in (
                "error_anchor_pos", "error_anchor_yaw", "error_root_pos", "error_root_yaw",
                "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel",
            ):
                value = self.command.metrics.get(metric_name)
                if value is not None:
                    key = f"drift/{metric_name}"
                    self.task_diag_sum[key] = self.task_diag_sum.get(key, value.new_zeros(())) + value.mean().detach()
        if self.is_go2_task:
            cache = getattr(self.env, "_supermimic_go2_reward_cache", None) or getattr(self.env, "_supermimic_chase_reward_cache", None)
            if cache is not None:
                inputs = cache.get("inputs", {})
                for key, value in (("joint_torque_rms", inputs.get("torque")), ("joint_acceleration_rms", inputs.get("joint_acceleration"))):
                    if value is not None:
                        metric = value.square().mean().sqrt().detach()
                        self.task_diag_sum[key] = self.task_diag_sum.get(key, metric.new_zeros(())) + metric
                if inputs.get("torque_proxy") is not None:
                    value = (inputs["torque"] - inputs["torque_proxy"]).square().mean().sqrt().detach()
                    self.task_diag_sum["torque_proxy_rms_error"] = self.task_diag_sum.get("torque_proxy_rms_error", value.new_zeros(())) + value
                if inputs.get("acceleration_fd") is not None:
                    value = (inputs["joint_acceleration"] - inputs["acceleration_fd"]).square().mean().sqrt().detach()
                    self.task_diag_sum["acceleration_fd_rms_error"] = self.task_diag_sum.get("acceleration_fd_rms_error", value.new_zeros(())) + value
        elif self.is_m1_world_point:
            cache = getattr(self.env, "_supermimic_world_point_reward_cache", None)
            if cache is not None:
                action = cache["action"]
                leg = action[..., (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)]
                wheels = action[..., (3, 7, 11, 15)]
                arm = action[..., 16:22]
                for key, value in (("action/leg_abs_mean", leg.abs().mean()), ("action/wheel_abs_mean", wheels.abs().mean()), ("action/arm_abs_mean", arm.abs().mean()),
                                    ("action/leg_saturation_fraction", (leg.abs() >= 1.0 - 1e-7).float().mean()),
                                    ("action/wheel_saturation_fraction", (wheels.abs() >= 1.0 - 1e-7).float().mean()),
                                    ("action/arm_saturation_fraction", (arm.abs() >= 1.0 - 1e-7).float().mean())):
                    self.task_diag_sum[key] = self.task_diag_sum.get(key, value.new_zeros(())) + value.detach()
                torque = cache.get("actual_torque")
                if torque is not None and torque.shape[-1] >= 22:
                    leg_torque = torque[..., (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)]
                    limits = 180.0
                    for key, value in (("actuator/leg_torque_rms", leg_torque.square().mean().sqrt()),
                                        ("actuator/leg_torque_limit_fraction", (leg_torque.abs() >= limits * (1 - 1e-6)).float().mean())):
                        self.task_diag_sum[key] = self.task_diag_sum.get(key, value.new_zeros(())) + value.detach()

    def _empty_go2_action_stats(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros((), device=self.device)
            for name in (
                "clean_sum", "clean_sq_sum", "clean_count",
                "clean_abs_sum",
                "exec_sum", "exec_sq_sum", "exec_count",
                "exec_abs_sum",
                "gauss_sum", "gauss_sq_sum", "gauss_count",
                "ou_sum", "ou_sq_sum", "ou_count",
                "clipped", "action_count", "terminated", "env_steps", "ou_envs", "env_count",
            )
        }

    def _accumulate_go2_action_stats(
        self,
        clean: torch.Tensor,
        executed: torch.Tensor,
        applied_noise: torch.Tensor,
        ou_mask: torch.Tensor,
        terminated: torch.Tensor,
    ) -> None:
        stats = self.go2_action_stats
        if stats is None:
            return
        def add_moments(prefix: str, value: torch.Tensor) -> None:
            if value.numel() == 0:
                return
            stats[f"{prefix}_sum"] += value.sum()
            stats[f"{prefix}_sq_sum"] += value.square().sum()
            stats[f"{prefix}_count"] += value.numel()

        add_moments("clean", clean)
        add_moments("exec", executed)
        stats["clean_abs_sum"] += clean.abs().sum()
        stats["exec_abs_sum"] += executed.abs().sum()
        add_moments("gauss", applied_noise[~ou_mask])
        add_moments("ou", applied_noise[ou_mask])
        stats["clipped"] += (executed.abs() >= 1.0 - 1.0e-7).sum()
        stats["action_count"] += executed.numel()
        stats["terminated"] += terminated.sum()
        stats["env_steps"] += terminated.numel()
        stats["ou_envs"] += ou_mask.sum()
        stats["env_count"] += ou_mask.numel()

    def _write_go2_action_stats(self, iteration: int) -> None:
        if self.writer is None or self.go2_action_stats is None:
            return
        stats = self.go2_action_stats

        def std(prefix: str) -> float:
            count = stats[f"{prefix}_count"].clamp_min(1.0)
            mean = stats[f"{prefix}_sum"] / count
            variance = stats[f"{prefix}_sq_sum"] / count - mean.square()
            return float(variance.clamp_min(0.0).sqrt())

        self.writer.add_scalar("go2/clean_action_std", std("clean"), iteration)
        self.writer.add_scalar("go2/executed_action_std", std("exec"), iteration)
        self.writer.add_scalar("go2/action_abs_mean", float(stats["clean_abs_sum"] / stats["clean_count"].clamp_min(1.0)), iteration)
        self.writer.add_scalar("go2/gaussian_noise_std_actual", std("gauss"), iteration)
        self.writer.add_scalar("go2/ou_noise_std_actual", std("ou"), iteration)
        self.writer.add_scalar(
            "go2/action_clip_fraction",
            float(stats["clipped"] / stats["action_count"].clamp_min(1.0)), iteration,
        )
        self.writer.add_scalar(
            "go2/episode_termination_fraction",
            float(stats["terminated"] / stats["env_steps"].clamp_min(1.0)), iteration,
        )
        self.writer.add_scalar(
            "go2/ou_environment_fraction",
            float(stats["ou_envs"] / stats["env_count"].clamp_min(1.0)), iteration,
        )
        self.go2_action_stats = self._empty_go2_action_stats()

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        wm_only_target = self._effective_wm_only_target()
        obs, _ = env.reset()
        if self.is_m1_prior_joint:
            initial_state = self.recorder.state_io.read()
            m1_state_history = initial_state[:, None, :].expand(-1, 4, -1).clone()
            m1_reference_phase = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
            self._m1_ee_hold_seconds = torch.zeros(env.num_envs, device=self.device)
            self._m1_episode_reach_success = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
            self._m1_finished_reach_successes = 0.0
            self._m1_finished_reach_episodes = 0.0
            self._m1_joint_real_step_metrics: dict[str, float] = {}
            self._m1_wm_replay_diagnostics: dict[str, float] = {}
            self._m1_metric_sum: dict[str, torch.Tensor] = {}
            self._m1_metric_count: dict[str, int] = {}
            self._m1_episode_counts = {"L": 0, "M": 0}
            self._m1_episode_falls = {"L": 0, "M": 0}
            exploration_cfg = self.cfg.m1_joint_training
            self._m1_explorer = M1BoundedGaussianExplorer(
                leg_std=exploration_cfg.exploration_leg_std,
                arm_std=exploration_cfg.exploration_arm_std,
                wheel_std=exploration_cfg.exploration_wheel_std,
                env_fraction=exploration_cfg.exploration_env_fraction,
                decay_end_fraction=exploration_cfg.exploration_decay_end_fraction,
            )
            initial_actor_path = self.log_dir / "ckpt_initial_actor.pt"
            if (
                self.iteration == 0 and self.m1_joint_trainer.actor_optimizer_steps == 0
                and not initial_actor_path.exists()
            ):
                self.save("ckpt_initial_actor.pt")
        t_start = time.time()
        self._training_start_time = t_start
        checkpoint_interval_seconds = max(0.0, float(sch.checkpoint_interval_minutes) * 60.0)
        next_wall_checkpoint = checkpoint_interval_seconds
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_counts: dict[str, int] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        finished_falls: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            m1_diagnostics_due = self.is_m1_prior_joint and (self.iteration + 1) % max(1, sch.log_interval // 5) == 0
            pre_frame = self.command.frame.clone() if self.is_m1_motion else None
            action = self.act(obs, m1_state_history, m1_reference_phase) if self.is_m1_prior_joint else self.act(obs)
            if self.is_m1_prior_joint:
                exploration_cfg = cfg.m1_joint_training
                phase_used = m1_reference_phase.clone()
                if exploration_cfg.exploration_enabled:
                    noise, explore_mask, noise_std = self._m1_explorer.sample(
                        action, iteration=self.iteration,
                        total_iterations=max(1, sch.total_iterations),
                    )
                else:
                    noise = torch.zeros_like(action)
                    explore_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
                    noise_std = torch.zeros(22, dtype=action.dtype, device=action.device)
                # Noise is installed explicitly on the M1 action term. The
                # ActionManager still observes/stores the raw Actor action;
                # ReplayRecorder writes the exact final bounded execution.
                action_term = self.recorder.action_term
                action_term.set_exploration(noise, explore_mask)
                executed_candidate = add_bounded_action_exploration(action, noise, explore_mask)
                self._m1_metric_sum["real/exploration_selected_envs"] = self._m1_metric_sum.get(
                    "real/exploration_selected_envs", action.new_zeros(()),
                ) + explore_mask.float().sum()
                self._m1_metric_count["real/exploration_selected_envs"] = (
                    self._m1_metric_count.get("real/exploration_selected_envs", 0) + env.num_envs
                )
                self._m1_metric_sum["real/exploration_noise_abs_sum"] = self._m1_metric_sum.get(
                    "real/exploration_noise_abs_sum", action.new_zeros(()),
                ) + (noise * explore_mask[:, None]).abs().sum()
                self._m1_metric_count["real/exploration_noise_abs_sum"] = (
                    self._m1_metric_count.get("real/exploration_noise_abs_sum", 0) + action.numel()
                )
                self._m1_metric_sum["real/exploration_action_clip_fraction"] = self._m1_metric_sum.get(
                    "real/exploration_action_clip_fraction", action.new_zeros(()),
                ) + (executed_candidate.abs() >= 1.0 - 1e-7).float().sum()
                self._m1_metric_count["real/exploration_action_clip_fraction"] = (
                    self._m1_metric_count.get("real/exploration_action_clip_fraction", 0) + action.numel()
                )
                for group, ids in (
                    ("leg", [i for i, n in enumerate(self.robot_joint_names) if ("FOOT" not in n and not n.startswith("piper_"))]),
                    ("wheel", [i for i, n in enumerate(self.robot_joint_names) if n.endswith("FOOT_JOINT")]),
                    ("arm", [i for i, n in enumerate(self.robot_joint_names) if n.startswith("piper_")]),
                ):
                    self._m1_metric_sum[f"real/exploration/{group}_std"] = self._m1_metric_sum.get(
                        f"real/exploration/{group}_std", action.new_zeros(()),
                    ) + noise_std[ids].mean()
                    self._m1_metric_count[f"real/exploration/{group}_std"] = (
                        self._m1_metric_count.get(f"real/exploration/{group}_std", 0) + 1
                    )
                wm_prediction = None
                if m1_diagnostics_due:
                    with torch.no_grad():
                        wm_prediction = self.wm.step(
                            m1_state_history[:, -1], executed_candidate, self.recorder.prev_xi,
                        )
                task_type_before = self.command.task_type.clone()
                velocity_command_before = self.command.velocity_command_body.clone()
                fixed_target_before = self.command.target_position_world.clone()
                initial_base_before = self.command.initial_base_position_world.clone()
                previous_ee_distance = self.command._last_ee_distance.clone()
            action_buffer_ptr = (
                self.buffer.ptr.clone()
                if self.is_go2_task or self.is_m1_prior_joint else None
            )
            action_ou_mask = (
                self.recorder.action_term._ou_mask.clone() if self.is_go2_task else None
            )
            obs, rew, term, tout, extras = env.step(action)
            if self.is_m1_prior_joint:
                next_state = self.recorder.state_io.read()
                # ManagerBasedRlEnv resets terminated worlds before returning. Recover
                # their just-finished terminal state from Replay's explicit terminal row.
                ended = (term | tout).nonzero(as_tuple=False).flatten()
                metric_next_state = next_state.clone()
                if ended.numel():
                    terminal_row = (self.buffer.ptr[ended] - 1) % self.buffer.T
                    metric_next_state[ended] = self.buffer.s[ended, terminal_row]
                if m1_diagnostics_due:
                    # env.step() auto-resets terminal environments before it
                    # returns, which clears ActionTerm.executed_raw and its
                    # physical setpoint. ReplayRecorder wrote this transition
                    # before reset, so read the exact per-step executed action
                    # from the captured ring-buffer cursor instead.
                    m1_env_ids = torch.arange(env.num_envs, device=self.device)
                    executed_action = self.buffer.a_exec[m1_env_ids, action_buffer_ptr]
                    real_metrics = self.m1_joint_trainer.record_real_step(
                        metric_next_state, action, phase_used, term,
                        truncated=tout,
                        executed_action=executed_action,
                    )
                    self._m1_joint_real_step_metrics = dict(real_metrics)
                task_metrics = self._measure_m1_joint_task_step(
                    metric_next_state, task_type_before, velocity_command_before,
                    fixed_target_before, initial_base_before, previous_ee_distance,
                )
                is_l = task_type_before == 0
                is_m = task_type_before == 1
                if m1_diagnostics_due:
                    self._m1_joint_real_step_metrics["real/L_sample_count"] = float(is_l.sum())
                    self._m1_joint_real_step_metrics["real/M_sample_count"] = float(is_m.sum())
                l_metrics = {
                    "base_velocity_xy_error": task_metrics["base_velocity_xy_error"],
                    "base_velocity_xy_squared_error": task_metrics["base_velocity_xy_squared_error"],
                    "yaw_rate_error": task_metrics["yaw_rate_error"],
                    "base_vertical_velocity": task_metrics["base_vertical_velocity"],
                    "world_displacement": task_metrics["world_displacement"],
                    "base_height": task_metrics["base_height"],
                }
                m_metrics = {
                    "ee_distance": task_metrics["ee_distance"],
                    "ee_progress": task_metrics["ee_progress"],
                    "ee_success": task_metrics["ee_success"],
                    "base_displacement": task_metrics["base_displacement"],
                    "base_height": task_metrics["base_height"],
                    "base_tilt_rad": task_metrics["base_tilt_rad"],
                    "base_stable": task_metrics["base_stable"],
                    "joint_limit_violation": task_metrics["joint_limit_violation"],
                }
                if m1_diagnostics_due:
                    from supermimic.motion_prior.velocity import task_masked_mean

                    l_rmse_sq = task_masked_mean(task_metrics["base_velocity_xy_squared_error"], is_l)
                    if l_rmse_sq is not None:
                        vector_rmse = l_rmse_sq.clamp_min(0).sqrt()
                        vx_rmse_sq = task_masked_mean(task_metrics["base_vx_squared_error"], is_l)
                        vy_rmse_sq = task_masked_mean(task_metrics["base_vy_squared_error"], is_l)
                        self._m1_joint_real_step_metrics["real/L/base_velocity_xy_vector_rmse"] = float(vector_rmse)
                        self._m1_joint_real_step_metrics["real/L/base_velocity_xy_rmse"] = float(vector_rmse)
                        self._m1_joint_real_step_metrics["real/L/base_velocity_xy_per_axis_rmse"] = float(
                            ((vx_rmse_sq + vy_rmse_sq) * 0.5).clamp_min(0).sqrt()
                        )
                        self._m1_joint_real_step_metrics["real/L/vx_rmse"] = float(vx_rmse_sq.clamp_min(0).sqrt())
                        self._m1_joint_real_step_metrics["real/L/vy_rmse"] = float(vy_rmse_sq.clamp_min(0).sqrt())
                        for name, value in l_metrics.items():
                            if name == "base_velocity_xy_squared_error":
                                continue
                            metric_mask = is_l
                            if name == "positive_command_negative_vx_fraction":
                                metric_mask = is_l & (task_metrics["command_vx"] > 0.05)
                            mean = task_masked_mean(value, metric_mask)
                            if mean is not None:
                                self._m1_joint_real_step_metrics[f"real/L/{name}"] = float(mean)
                    for name, value in m_metrics.items():
                        mean = task_masked_mean(value, is_m)
                        if mean is not None:
                            self._m1_joint_real_step_metrics[f"real/M/{name}"] = float(mean)
                for name, values, mask in (
                    ("real/L/base_velocity_xy_squared_error", task_metrics["base_velocity_xy_squared_error"], is_l),
                    ("real/L/base_vx_squared_error", task_metrics["base_vx_squared_error"], is_l),
                    ("real/L/base_vy_squared_error", task_metrics["base_vy_squared_error"], is_l),
                    (
                        "real/L/positive_command_negative_vx_fraction",
                        task_metrics["positive_command_negative_vx"],
                        is_l & (task_metrics["command_vx"] > 0.05),
                    ),
                    ("real/L/yaw_rate_error", task_metrics["yaw_rate_error"], is_l),
                    ("real/L/base_vertical_velocity", task_metrics["base_vertical_velocity"], is_l),
                    ("real/L/command_vx", task_metrics["command_vx"], is_l),
                    ("real/L/command_vy", task_metrics["command_vy"], is_l),
                    ("real/L/command_wz", task_metrics["command_wz"], is_l),
                    ("real/L/base_vx_body", task_metrics["base_vx_body"], is_l),
                    ("real/L/base_vy_body", task_metrics["base_vy_body"], is_l),
                    ("real/L/base_yaw_rate", task_metrics["base_yaw_rate"], is_l),
                    ("real/L/wheel_rate_abs_mean", task_metrics["wheel_rate_abs_mean"], is_l),
                    ("real/L/world_displacement", task_metrics["world_displacement"], is_l),
                    ("real/L/base_height", task_metrics["base_height"], is_l),
                    ("real/M/ee_distance", task_metrics["ee_distance"], is_m),
                    ("real/M/ee_progress", task_metrics["ee_progress"], is_m),
                    ("real/M/ee_success", task_metrics["ee_success"], is_m),
                    ("real/M/base_displacement", task_metrics["base_displacement"], is_m),
                    ("real/M/base_height", task_metrics["base_height"], is_m),
                    ("real/M/joint_limit_violation", task_metrics["joint_limit_violation"], is_m),
                ):
                    selected = values[mask]
                    if selected.numel():
                        total = selected.detach().sum()
                        self._m1_metric_sum[name] = self._m1_metric_sum.get(name, torch.zeros_like(total)) + total
                        self._m1_metric_count[name] = self._m1_metric_count.get(name, 0) + int(selected.numel())
                saturation_sum = (action.abs() >= 0.98).float().sum().detach()
                self._m1_metric_sum["real/action_saturation"] = self._m1_metric_sum.get(
                    "real/action_saturation", torch.zeros_like(saturation_sum),
                ) + saturation_sum
                self._m1_metric_count["real/action_saturation"] = self._m1_metric_count.get("real/action_saturation", 0) + action.numel()
                if m1_diagnostics_due and self.command.cfg.task_type == "L":
                    self._m1_joint_real_step_metrics["real/base_velocity_xy_rmse"] = self._m1_joint_real_step_metrics["real/L/base_velocity_xy_rmse"]
                    self._m1_joint_real_step_metrics["real/yaw_rate_error"] = self._m1_joint_real_step_metrics["real/L/yaw_rate_error"]
                    self._m1_joint_real_step_metrics["real/base_vertical_velocity"] = self._m1_joint_real_step_metrics["real/L/base_vertical_velocity"]
                    self._m1_joint_real_step_metrics["real/world_displacement"] = self._m1_joint_real_step_metrics["real/L/world_displacement"]
                elif m1_diagnostics_due and self.command.cfg.task_type == "M":
                    self._m1_joint_real_step_metrics["real/ee_distance"] = self._m1_joint_real_step_metrics["real/M/ee_distance"]
                    self._m1_joint_real_step_metrics["real/ee_progress"] = self._m1_joint_real_step_metrics["real/M/ee_progress"]
                    self._m1_joint_real_step_metrics["real/base_displacement"] = self._m1_joint_real_step_metrics["real/M/base_displacement"]
                if m1_diagnostics_due:
                    self._m1_joint_real_step_metrics["real/action_saturation"] = float((action.abs() >= 0.98).float().mean())
                    self._m1_joint_real_step_metrics["imag/real_gap"] = float(
                        self.wm.spec.normalized_sq_error(wm_prediction, metric_next_state).mean()
                    )
                    one_step_error = self.wm.spec.error_vector(wm_prediction, metric_next_state) * self.wm.spec.err_scale
                    self._m1_joint_real_step_metrics["wm/real_one_step/root_position_rmse_m"] = float(
                        one_step_error[..., 0:3].square().sum(-1).mean().sqrt()
                    )
                    self._m1_joint_real_step_metrics["wm/real_one_step/root_orientation_rmse_rad"] = float(
                        one_step_error[..., 3:6].square().sum(-1).mean().sqrt()
                    )
                    self._m1_joint_real_step_metrics["wm/real_one_step/root_velocity_rmse_m_s"] = float(
                        one_step_error[..., 6:9].square().sum(-1).mean().sqrt()
                    )
                    self._m1_joint_real_step_metrics["wm/real_one_step/root_angular_velocity_rmse_rad_s"] = float(
                        one_step_error[..., 9:12].square().sum(-1).mean().sqrt()
                    )
                    q_error = one_step_error[..., 12:34]
                    qd_error = one_step_error[..., 34:56]
                    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS

                    leg_error_ids = torch.tensor(
                        [M1_JOINTS.index(name) for name in M1_AMP_JOINTS[:12]], device=self.device,
                    )
                    arm_error_ids = torch.tensor(
                        [M1_JOINTS.index(name) for name in M1_AMP_JOINTS[12:]], device=self.device,
                    )
                    wheel_error_ids = torch.tensor(self.wm.spec.wheel_indices, device=self.device)
                    for prefix, source, indices, unit in (
                        ("leg_position", q_error, leg_error_ids, "rad"),
                        ("arm_position", q_error, arm_error_ids, "rad"),
                        ("wheel_position", q_error, wheel_error_ids, "rad"),
                        ("leg_velocity", qd_error, leg_error_ids, "rad_s"),
                        ("arm_velocity", qd_error, arm_error_ids, "rad_s"),
                        ("wheel_velocity", qd_error, wheel_error_ids, "rad_s"),
                    ):
                        self._m1_joint_real_step_metrics[f"wm/real_one_step/{prefix}_rmse_{unit}"] = float(
                            source.index_select(-1, indices).square().mean().sqrt()
                        )
                    predicted_tcp = self.m1_joint_trainer.fk(wm_prediction[:, :29], wm_prediction[:, 29:])["pos"][:, self.m1_joint_trainer.tcp_index]
                    actual_tcp = self.m1_joint_trainer.fk(metric_next_state[:, :29], metric_next_state[:, 29:])["pos"][:, self.m1_joint_trainer.tcp_index]
                    self._m1_joint_real_step_metrics["wm/real_one_step/ee_fk_error_m"] = float(
                        torch.linalg.vector_norm(predicted_tcp - actual_tcp, dim=-1).mean()
                    )
                reach = task_type_before == 1
                within = (
                    reach & task_metrics["base_stable"].bool()
                    & (task_metrics["ee_distance"] < self.command.cfg.success_distance)
                )
                hold_next = torch.where(
                    within, self._m1_ee_hold_seconds + float(env.step_dt),
                    torch.zeros_like(self._m1_ee_hold_seconds),
                )
                episode_success_next = self._m1_episode_reach_success | (hold_next >= 0.5)
                done_ids = (term | tout).nonzero(as_tuple=False).flatten()
                finished_m = done_ids[reach[done_ids]]
                if finished_m.numel():
                    survived_to_end = ~term[finished_m]
                    self._m1_finished_reach_successes += float(
                        (episode_success_next[finished_m] & survived_to_end).sum()
                    )
                    self._m1_finished_reach_episodes += float(finished_m.numel())
                self._m1_ee_hold_seconds = hold_next
                self._m1_episode_reach_success = episode_success_next
                if done_ids.numel():
                    for task_code, task_name in ((0, "L"), (1, "M")):
                        task_done = done_ids[task_type_before[done_ids] == task_code]
                        self._m1_episode_counts[task_name] += int(task_done.numel())
                        self._m1_episode_falls[task_name] += int(term[task_done].sum())
                    self._m1_ee_hold_seconds[done_ids] = 0.0
                    self._m1_episode_reach_success[done_ids] = False
                if m1_diagnostics_due and self._m1_finished_reach_episodes:
                    self._m1_joint_real_step_metrics["real/ee_success_hold_rate"] = (
                        self._m1_finished_reach_successes / self._m1_finished_reach_episodes
                    )
                state_history_next = torch.cat((m1_state_history[:, 1:], next_state[:, None]), dim=1)
                if done_ids.numel():
                    state_history_next[done_ids] = next_state[done_ids, None, :].expand(-1, 4, -1)
                m1_state_history = state_history_next
                m1_reference_phase = (m1_reference_phase + 1) % 10
                m1_reference_phase[term | tout] = 0
                # Real MuJoCo mechanical power is reported as simulator actuator
                # power, not battery/electrical energy and not the planner's proxy.
                qfrc = getattr(env.sim.data, "qfrc_actuator", None)
                if qfrc is not None and m1_diagnostics_due:
                    force = qfrc[:, self.recorder.state_io.joint_dof_addr]
                    qd = metric_next_state[:, 35:57]
                    positive_power = torch.relu(force * qd)
                    self._m1_joint_real_step_metrics["real/actuator_force_abs_mean"] = float(force.abs().mean())
                    self._m1_joint_real_step_metrics["real/joint_velocity_abs_mean"] = float(qd.abs().mean())
                    self._m1_joint_real_step_metrics["real/actuator_mechanical_power_positive_w"] = float(positive_power.sum(-1).mean())
                    self._m1_joint_real_step_metrics["real/actuator_mechanical_work_step_j"] = float(positive_power.sum(-1).mean() * self.cfg.world_model.state_spec.dt)
            if self.is_go2_task:
                env_ids = torch.arange(env.num_envs, device=self.device)
                executed = self.buffer.a_exec[env_ids, action_buffer_ptr]
                clean = self.buffer.a[env_ids, action_buffer_ptr]
                applied_noise = self.recorder.action_term.applied_noise.clone()
                self._accumulate_go2_action_stats(clean, executed, applied_noise, action_ou_mask, term)
            self._record_real_reward_terms()
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if self.is_m1_motion:
                clip_ids = self.lib.frame_clip[pre_frame]
                self.m1_ep_clip_return += rew
                self._update_m1_metrics(~done)
                self._finish_m1_episodes(done, term, clip_ids)
            ids = done.nonzero().squeeze(-1)
            if ids.numel() > 0:
                lens = self.ep_len[ids]
                if self.has_tracking_return:
                    norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                    finished_returns.extend(norm_ret.tolist())
                elif self.is_go2_task or self.is_m1_world_point or self.is_m1_prior_joint or self.is_m1_reward:
                    finished_returns.extend(self.ep_reward_sum[ids].tolist())
                    finished_falls.extend(term[ids].float().tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            if not self.is_m1_prior_joint:
                torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    # A configured UTD ratio must never force a sample from an
                    # empty or shorter-than-window Replay. Keep the accumulator
                    # pending; the next real steps will make a valid window.
                    if self.is_m1_prior_joint:
                        # total_rows is a cheap host counter; a full GPU replay
                        # mask scan every simulation step was otherwise
                        # forcing synchronization over E×T entries. WM sampling
                        # still validates the actual same-episode windows.
                        replay_warm = self.buffer.total_rows >= env.num_envs * (int(cfg.wm_trainer.window) + 1)
                    else:
                        replay_warm = (
                            self.buffer.size >= int(cfg.wm_trainer.window) + 1
                            and bool(self.buffer.window_mask(int(cfg.wm_trainer.window)).any())
                        )
                    if not replay_warm:
                        break
                    self._utd_acc -= 1.0
                    wm_lr = scheduled_lr(sch.lr_schedule, self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    policy_enabled = wm_only_target <= 0 or self.wm_only_updates >= wm_only_target
                    if self.is_m1_prior_joint:
                        if wm_only_target > 0 and not policy_enabled:
                            self.wm_only_updates += 1
                        prior_lr = scheduled_lr(
                            sch.lr_schedule, self.iteration, sch.total_iterations,
                            cfg.m1_joint_training.prior_lr, sch.lr_final_ratio, sch.lr_decay_fraction,
                        )
                        actor_scheduled_lr = scheduled_lr(
                            sch.lr_schedule, self.iteration, sch.total_iterations,
                            cfg.m1_joint_training.actor_lr, sch.lr_final_ratio, sch.lr_decay_fraction,
                        )
                        self.wm_trainer.set_lr(wm_lr)
                        self.m1_joint_trainer.set_lr(prior_lr=prior_lr, actor_lr=actor_scheduled_lr)
                        t0 = time.perf_counter()
                        joint_metrics = self.m1_joint_trainer.update_event(
                            policy_allowed=policy_enabled,
                            config_snapshot=self._run_config_snapshot(),
                        )
                        update_time = time.perf_counter() - t0
                        timers["wm"] += update_time
                        timers["policy"] += update_time if policy_enabled else 0.0
                        self.actor_updates = self.m1_joint_trainer.actor_optimizer_steps
                        self.num_updates += 1
                        for k, v in joint_metrics.items():
                            agg[k] = agg.get(k, 0.0) + v
                            agg_counts[k] = agg_counts.get(k, 0) + 1
                        agg["wm/lr"] = wm_lr
                        agg["prior/lr"] = prior_lr
                        agg["actor/lr"] = actor_scheduled_lr
                        agg["schedule/wm_only_updates"] = float(self.wm_only_updates)
                        agg["schedule/actor_updates"] = float(self.actor_updates)
                        agg_n += 1
                        continue
                    actor_scheduled_lr = scheduled_lr(sch.lr_schedule, self.iteration, sch.total_iterations, cfg.policy_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    critic_lr = scheduled_lr(sch.lr_schedule, self.iteration, sch.total_iterations, cfg.policy_trainer.critic_lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(wm_lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer, task_balanced=self.is_m1_reward and self.task.task_balanced)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    m_pi = {}
                    ramp_scale = 0.0
                    actor_lr = 0.0
                    if wm_only_target > 0 and not policy_enabled:
                        self.wm_only_updates += 1
                    else:
                        if self.actor_start_iteration is None:
                            self.actor_start_iteration = self.iteration
                            if self.is_go2_task and self.is_main:
                                with torch.no_grad():
                                    start_err = self.wm_trainer.online_error(self.buffer, n=256, k=8)
                                    self.go2_actor_start_nse8 = float(start_err[-1])
                                if self.go2_actor_start_nse8 > 0.5:
                                    print(
                                        f"[WARN] Go2 actor starts with 8-step WM NSE={self.go2_actor_start_nse8:.4f} "
                                        "(configured review threshold 0.5); see wm/actor_start_nse8",
                                        flush=True,
                                    )
                        t0 = time.perf_counter()
                        for _ in range(sch.policy_updates_per_wm_update):
                            ramp_updates = max(int(sch.actor_ramp_iterations), 0)
                            if (self.is_go2_task or self.is_m1_reward) and ramp_updates > 0:
                                ramp_scale = actor_ramp_scale(self.actor_updates + 1, ramp_updates)
                            else:
                                ramp_scale = 1.0
                            actor_lr = actor_scheduled_lr * ramp_scale
                            self.policy_trainer.set_lr(actor_lr, critic_lr)
                            m_pi = self.policy_trainer.update(self.buffer)
                            self.actor_updates += 1
                        if sch.policy_updates_per_wm_update:
                            torch.cuda.synchronize()
                            timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        key = f"wm/{k}"
                        agg[key] = agg.get(key, 0.0) + v
                        agg_counts[key] = agg_counts.get(key, 0) + 1
                    for k, v in m_pi.items():
                        key = f"policy/{k}"
                        agg[key] = agg.get(key, 0.0) + v
                        agg_counts[key] = agg_counts.get(key, 0) + 1
                    agg["wm/lr"] = wm_lr
                    agg["policy/actor_lr"] = actor_lr
                    agg["policy/critic_lr"] = critic_lr
                    agg["policy/actor_ramp_scale"] = ramp_scale
                    agg["schedule/wm_only_updates"] = float(self.wm_only_updates)
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                if self.is_m1_prior_joint:
                    torch.cuda.synchronize()
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                    len_sum = allreduce_mean(float(sum(finished_lens))) * self.world
                    fall_sum = allreduce_mean(float(sum(finished_falls))) * self.world
                    finished_lens = [len_sum / cnt] if cnt > 0 else []
                    finished_falls = [fall_sum / cnt] if cnt > 0 and self.is_go2_task else []
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_counts, agg_n = {}, {}, 0
                    finished_returns, finished_lens = [], []
                    finished_falls = []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    last_value_keys = (
                        "policy/actor_lr", "policy/critic_lr", "policy/actor_ramp_scale",
                        "wm/lr", "prior/lr", "actor/lr", "schedule/wm_only_updates",
                    )
                    for k, v in agg.items():
                        metric_count = agg_counts.get(k, agg_n) if self.is_m1_prior_joint or self.is_m1_reward else agg_n
                        denominator = 1 if k in last_value_keys else metric_count
                        self.writer.add_scalar(k, v / max(denominator, 1), it)
                if self.is_main and self.policy_trainer is not None and self.policy_trainer.last_imag_diagnostics:
                    self.writer.add_text(
                        "imag/step_diagnostics",
                        json.dumps(self.policy_trainer.last_imag_diagnostics, sort_keys=True),
                        it,
                    )
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    if self.cfg.env.task_name == "motion_tracking":
                        self.writer.add_scalar("train/episode_norm_return", tr, it)
                    elif self.is_go2_velocity:
                        self.writer.add_scalar("go2/real_episode_return", tr, it)
                    elif self.is_go2_chase:
                        self.writer.add_scalar("go2/chase_episode_return", tr, it)
                        # Timeout success is the command's hard tag event, never
                        # the complement of the fall flag (ordinary timeouts are
                        # not successes).
                        tag_rate = float(self.command.last_success_rate)
                        if not math.isfinite(tag_rate):
                            tag_rate = float(self.command.metrics["tagged"].float().mean())
                        self.writer.add_scalar("go2/chase_timeout_success_fraction", tag_rate, it)
                    elif self.is_m1_world_point:
                        self.writer.add_scalar("m1/world_point_episode_return", tr, it)
                    elif self.is_m1_reward:
                        self.writer.add_scalar("m1/reward_only_episode_return", tr, it)
                    elif self.is_m1_prior_joint:
                        self.writer.add_scalar("m1/prior_joint_episode_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                    if self.is_m1_prior_joint:
                        self.writer.add_scalar("real/episode_length", sum(finished_lens) / len(finished_lens), it)
                else:
                    tr = float("nan")
                if self.is_go2_task and self.is_main and finished_falls:
                    self.writer.add_scalar("go2/fall_rate", sum(finished_falls) / len(finished_falls), it)
                if self.is_m1_world_point and self.is_main and finished_falls:
                    fall_rate = sum(finished_falls) / len(finished_falls)
                    self.writer.add_scalar("m1/world_point_fall_rate", fall_rate, it)
                    self.writer.add_scalar("m1/fall_rate", fall_rate, it)
                if self.is_m1_reward and self.is_main and finished_falls:
                    self.writer.add_scalar("m1/reward_only_fall_rate", sum(finished_falls) / len(finished_falls), it)
                if self.is_m1_prior_joint and self.is_main and finished_falls:
                    self.writer.add_scalar("m1/fall_rate", sum(finished_falls) / len(finished_falls), it)
                if self.is_go2_velocity and self.is_main:
                    self.writer.add_scalar("go2/curriculum_stage", float(self.command.stage), it)
                    self.writer.add_scalar("go2/curriculum_eval_count", float(self.command.eval_count), it)
                    # Always expose the deployment-facing command/response
                    # error, even before a full curriculum evaluation window
                    # is complete.
                    self.writer.add_scalar("go2/command_speed", float(self.command.metrics["command_speed"].mean()), it)
                    self.writer.add_scalar("go2/real_vel_error_xy", float(self.command.metrics["real_vel_error_xy"].mean()), it)
                    self.writer.add_scalar("go2/real_yaw_error", float(self.command.metrics["real_yaw_error"].mean()), it)
                    for key, value in self.command.last_eval_metrics.items():
                        if isinstance(value, (float, int)) and math.isfinite(float(value)):
                            self.writer.add_scalar(f"go2/{key}", float(value), it)
                    action_term = self.recorder.action_term
                    if hasattr(action_term, "_ou_mask"):
                        self.writer.add_scalar("go2/ou_env_fraction", float(action_term._ou_mask.float().mean()), it)
                        self.writer.add_scalar("go2/action_noise_abs_mean", float((action_term.executed_raw - action_term.raw_action).abs().mean()), it)
                    self._write_go2_action_stats(it)
                if self.is_go2_chase and self.is_main:
                    self.writer.add_scalar("go2/chase_curriculum_stage", float(self.command.stage), it)
                    if math.isfinite(self.command.last_success_rate):
                        self.writer.add_scalar("go2/chase_success_rate_window", self.command.last_success_rate, it)
                    self.writer.add_scalar("go2/chase_tag_fraction_step", float(self.command.metrics["tagged"].mean()), it)
                    self.writer.add_scalar("go2/chase_ball_distance", float(self.command.metrics["ball_distance"].mean()), it)
                    self.writer.add_scalar("go2/chase_approach_speed", float(self.command.metrics["approach_speed"].mean()), it)
                    self.writer.add_scalar("go2/chase/tag_rate", float(self.command.metrics["tagged"].mean()), it)
                    self.writer.add_scalar("go2/chase/mean_ball_distance", float(self.command.metrics["ball_distance"].mean()), it)
                    self.writer.add_scalar("go2/chase/approach_velocity", float(self.command.metrics["approach_speed"].mean()), it)
                    self.writer.add_scalar("go2/chase/robot_displacement", float(self.command.metrics["robot_displacement"].mean()), it)
                    self._write_go2_action_stats(it)
                if self.is_m1_motion and self.is_main:
                    metric_count = float(self.m1_step_metric_count)
                    if metric_count > 0:
                        for key, value in self.m1_step_metric_sum.items():
                            self.writer.add_scalar(f"m1/{key}", float(value / metric_count), it)
                    for i, name in enumerate(self.motion_names):
                        count = float(self.motion_return_count[i])
                        if count > 0:
                            self.writer.add_scalar(f"m1/per_motion_return/{name}", float(self.motion_return_sum[i] / count), it)
                        success_count = float(self.motion_success_count[i])
                        if success_count > 0:
                            self.writer.add_scalar(f"m1/per_motion_success_rate/{name}", float(self.motion_success_sum[i] / success_count), it)
                    total_success_count = float(self.motion_success_count.sum())
                    if total_success_count > 0:
                        self.writer.add_scalar("m1/motion_success_rate", float(self.motion_success_sum.sum() / total_success_count), it)
                if self.is_m1_world_point and self.is_main:
                    tcp_distance = float(self.command.metrics["target_distance"].mean())
                    self.writer.add_scalar(
                        "m1/world_point_target_distance",
                        tcp_distance, it,
                    )
                    self.writer.add_scalar(
                        "m1/world_point_instant_success_fraction",
                        float(self.command.metrics["success"].mean()), it,
                    )
                    self.writer.add_scalar("m1/world_point_target_stage", float(self.command.stage), it)
                    if math.isfinite(self.command.last_success_rate):
                        self.writer.add_scalar("m1/world_point_success_rate_window", self.command.last_success_rate, it)
                        self.writer.add_scalar("m1/success_rate", self.command.last_success_rate, it)
                    self.writer.add_scalar("m1/tcp_distance_mean", tcp_distance, it)
                    self.writer.add_scalar("m1/tcp_distance_final", tcp_distance, it)
                    cache = getattr(self.env, "_supermimic_world_point_reward_cache", None)
                    if cache is not None:
                        self.writer.add_scalar("m1/base_height_mean", float(cache["state_next"][..., 2].mean()), it)
                if self.is_m1_prior_joint and self.is_main:
                    real_window: dict[str, float] = {}
                    for key, total in self._m1_metric_sum.items():
                        count = self._m1_metric_count.get(key, 0)
                        if count <= 0:
                            continue
                        value = float(total / count)
                        if key in (
                            "real/L/base_velocity_xy_squared_error",
                            "real/L/base_vx_squared_error", "real/L/base_vy_squared_error",
                        ):
                            value = max(value, 0.0) ** 0.5
                            key = {
                                "real/L/base_velocity_xy_squared_error": "real/L/base_velocity_xy_vector_rmse",
                                "real/L/base_vx_squared_error": "real/L/vx_rmse",
                                "real/L/base_vy_squared_error": "real/L/vy_rmse",
                            }[key]
                        real_window[key] = value
                    if "real/L/vx_rmse" in real_window and "real/L/vy_rmse" in real_window:
                        real_window["real/L/base_velocity_xy_per_axis_rmse"] = (
                            (real_window["real/L/vx_rmse"] ** 2 + real_window["real/L/vy_rmse"] ** 2) / 2.0
                        ) ** 0.5
                        real_window["real/L/base_velocity_xy_rmse"] = real_window["real/L/base_velocity_xy_vector_rmse"]
                    real_window["real/L_sample_count"] = float(
                        self._m1_metric_count.get("real/L/yaw_rate_error", 0)
                    )
                    real_window["real/M_sample_count"] = float(
                        self._m1_metric_count.get("real/M/ee_distance", 0)
                    )
                    if self._m1_episode_counts["L"]:
                        real_window["real/L_fall_rate"] = (
                            self._m1_episode_falls["L"] / self._m1_episode_counts["L"]
                        )
                    if self._m1_episode_counts["M"]:
                        real_window["real/M_fall_rate"] = (
                            self._m1_episode_falls["M"] / self._m1_episode_counts["M"]
                        )
                    if self.command.cfg.task_type == "L" and "real/L/base_velocity_xy_rmse" in real_window:
                        real_window["real/base_velocity_xy_rmse"] = real_window["real/L/base_velocity_xy_rmse"]
                        real_window["real/yaw_rate_error"] = real_window["real/L/yaw_rate_error"]
                        real_window["real/base_vertical_velocity"] = real_window["real/L/base_vertical_velocity"]
                        real_window["real/world_displacement"] = real_window["real/L/world_displacement"]
                    if self.command.cfg.task_type == "M" and "real/M/ee_distance" in real_window:
                        real_window["real/ee_distance"] = real_window["real/M/ee_distance"]
                        real_window["real/ee_progress"] = real_window["real/M/ee_progress"]
                        real_window["real/base_displacement"] = real_window["real/M/base_displacement"]
                    for key, value in real_window.items():
                        self.writer.add_scalar(key, value, it)
                    # These are latest-step diagnostics; task scores above are
                    # masked means over the complete logging interval.
                    for key, value in self._m1_joint_real_step_metrics.items():
                        if key.startswith("wm/") or key == "real/ee_success_hold_rate":
                            self.writer.add_scalar(key, value, it)
                    self._m1_joint_log_metrics = {**self._m1_joint_real_step_metrics, **real_window}
                    self.writer.add_scalar("prior/actor_ready", float(self.m1_joint_trainer.actor_ready), it)
                    self.writer.add_scalar("prior/state_source_privileged", 1.0, it)
                    self.writer.add_scalar("prior/l_task_fraction_real", float((self.command.task_type == 0).float().mean()), it)
                    self.writer.add_scalar("env/task_l_count", float((self.command.task_type == 0).sum()), it)
                    self.writer.add_scalar("env/task_m_count", float((self.command.task_type == 1).sum()), it)
                    self.writer.add_scalar("real/episodes_l_window", float(self._m1_episode_counts["L"]), it)
                    self.writer.add_scalar("real/episodes_m_window", float(self._m1_episode_counts["M"]), it)
                    self.writer.add_scalar("prior/optimizer_steps", float(self.m1_joint_trainer.prior_optimizer_steps), it)
                    self.writer.add_scalar(
                        "prior/calibration_rolled_back_optimizer_steps",
                        float(self.m1_joint_trainer._prior_calibration_rolled_back_updates), it,
                    )
                    self.writer.add_scalar(
                        "prior/calibration_status_code",
                        float({"not_started": 0, "snapshot_saved": 1, "in_progress": 2,
                               "accepted": 3, "rejected_rollback": 4,
                               "no_calibration_to_validate": 5}.get(
                                   self.m1_joint_trainer._prior_calibration_status, -1)), it,
                    )
                    self.writer.add_scalar("actor/optimizer_steps", float(self.m1_joint_trainer.actor_optimizer_steps), it)
                    self.writer.add_scalar("wm/update_events", float(self.m1_joint_trainer.wm_update_events), it)
                    self.writer.add_scalar("wm/optimizer_steps", float(self.m1_joint_trainer.wm_optimizer_steps), it)
                    stage_code = {"wm_warmup": 0, "actor_prior_frozen": 1,
                                  "prior_calibration": 2, "actor_adaptation": 3,
                                  "validation": 4}.get(self.m1_joint_trainer.stage, -1)
                    self.writer.add_scalar("schedule/m1_joint_phase", stage_code, it)
                if self.is_main:
                    task_prefix = "go2" if self.is_go2_task else ("m1" if self.is_m1_motion or self.is_m1_world_point or self.is_m1_prior_joint or self.is_m1_reward else "g1")
                    for key, value in self.real_reward_term_sum.items():
                        self.writer.add_scalar(f"{task_prefix}/reward_real/{key}", float(value / sch.log_interval), it)
                    for key, value in self.task_diag_sum.items():
                        self.writer.add_scalar(f"{task_prefix}/{key}", float(value / sch.log_interval), it)
                    for key, value in agg.items():
                        if key.startswith("policy/imag_r/"):
                            term = key.removeprefix("policy/imag_r/")
                            term_count = agg_counts.get(key, agg_n) if self.is_m1_reward else agg_n
                            self.writer.add_scalar(f"{task_prefix}/reward_imag/{term}", value / max(term_count, 1), it)
                with torch.no_grad():
                    if self.is_main and self.is_m1_prior_joint and bool(self.buffer.window_mask(8).any()):
                        wm_components = self.wm_trainer.online_error_components(
                            self.buffer, n=min(256, max(self.env.num_envs, 8)), k=8,
                        )
                        for key, value in wm_components.items():
                            self.writer.add_scalar(f"wm/replay_open_loop/{key}", value, it)
                        self._m1_wm_replay_diagnostics = wm_components
                        err = None
                    elif self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=256, k=8)
                        self.writer.add_scalar("wm/nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main and self.is_go2_task and hasattr(self, "go2_actor_start_nse8"):
                    self.writer.add_scalar("wm/actor_start_nse8", self.go2_actor_start_nse8, it)
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("wm/pretrained_initialized", float(self.pretrained_wm), it)
                    self.writer.add_scalar("wm/pretrained_source_iteration", float(self.init_wm_source_iteration or 0), it)
                    self.writer.add_scalar("wm/optimizer_steps", float(self.num_updates), it)
                    self.writer.add_scalar("wm/updates_per_env_step_actual", float(self.num_updates / max(it, 1)), it)
                    self.writer.add_scalar("schedule/wm_only_updates", self.wm_only_updates, it)
                    self.writer.add_scalar("schedule/actor_updates", self.actor_updates, it)
                    if "sampling_entropy" in self.command.metrics:
                        self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    if hasattr(self.command, "bin_return"):
                        self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    pi_count = agg_counts.get("policy/imag_return_undisc", agg_n) if self.is_m1_reward else agg_n
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(pi_count,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(pi_count,1):.3f} | "
                        f"WM-only {self.wm_only_updates}/{wm_only_target} actor-ramp {agg.get('policy/actor_ramp_scale', 0.0):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    if self.is_m1_prior_joint:
                        real = getattr(self, "_m1_joint_log_metrics", self._m1_joint_real_step_metrics)
                        phase = self.m1_joint_trainer.stage
                        wm_nse1 = self._m1_wm_replay_diagnostics.get("nse_1step", float("nan"))
                        wm_nse8 = self._m1_wm_replay_diagnostics.get("nse_8step", float("nan"))
                        wm_ready = self.m1_joint_trainer.last_wm_readiness.get("schedule/wm_ready", 0.0)
                        readiness = self.m1_joint_trainer.actor_readiness_diagnostics
                        actor_gate_failures = [
                            key.removeprefix("schedule/actor_readiness_").removesuffix("_pass")
                            for key, value in readiness.items()
                            if key.endswith("_pass") and not bool(value)
                        ]
                        wm_gate = self.m1_joint_trainer.last_wm_readiness
                        wm_gate_failures = [
                            key.removeprefix("schedule/wm_readiness_").removesuffix("_pass")
                            for key, value in wm_gate.items()
                            if key.endswith("_pass") and not bool(value)
                        ]
                        gpu_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                        task_summary = []
                        for key in (
                            "real/base_velocity_xy_rmse", "real/yaw_rate_error",
                            "real/M/ee_distance", "real/M/ee_progress",
                        ):
                            if key in real:
                                task_summary.append(f"{key.removeprefix('real/') } {real[key]:.4f}")
                        task_text = " | ".join(task_summary) if task_summary else "task metrics pending"
                        prior_summary = "/".join((
                            _format_m1_aggregate(agg, agg_n, "prior/task_loss"),
                            _format_m1_aggregate(agg, agg_n, "prior/smooth_loss"),
                            _format_m1_aggregate(agg, agg_n, "prior/grad_norm", 3),
                        ))
                        actor_summary = "/".join((
                            _format_m1_aggregate(agg, agg_n, "actor/tracking_error"),
                            _format_m1_aggregate(agg, agg_n, "actor/grad_norm", 3),
                        ))
                        wm_summary = "/".join((
                            _format_m1_metric(wm_nse1), _format_m1_metric(wm_nse8),
                        ))
                        msg = (
                            f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | "
                            f"M1 WM NSE1/8 {wm_summary} | "
                            f"prior task/smooth/grad {prior_summary} | "
                            f"actor track/grad {actor_summary} | "
                            f"{task_text} | phase {phase} | wm_ready {int(bool(wm_ready))} | "
                            f"actor_ready {int(bool(readiness['schedule/actor_ready']))}"
                            f"[{','.join(actor_gate_failures) or 'pass'}] "
                            f"wm_gate [{','.join(wm_gate_failures) or 'pass'}] | "
                            f"updates wm/prior/actor {self.m1_joint_trainer.wm_optimizer_steps}/"
                            f"{self.m1_joint_trainer.prior_optimizer_steps}/{self.m1_joint_trainer.actor_optimizer_steps} | "
                            f"GPU peak {gpu_peak:.2f} GiB | t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                        )
                    print(msg, flush=True)
                agg, agg_counts, agg_n = {}, {}, 0
                finished_returns, finished_lens = [], []
                finished_falls = []
                self.real_reward_term_sum = {}
                self.task_diag_sum = {}
                self.m1_step_metric_sum = {}
                self.m1_step_metric_count.zero_()
                if self.is_m1_prior_joint:
                    self._m1_metric_sum = {}
                    self._m1_metric_count = {}
                    self._m1_episode_counts = {"L": 0, "M": 0}
                    self._m1_episode_falls = {"L": 0, "M": 0}
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if checkpoint_interval_seconds and time.time() - t_start >= next_wall_checkpoint:
                while time.time() - t_start >= next_wall_checkpoint:
                    wall_minutes = int(round(next_wall_checkpoint / 60.0))
                    self.save(f"ckpt_wall_{wall_minutes:03d}m.pt")
                    self.save("ckpt_latest.pt")
                    next_wall_checkpoint += checkpoint_interval_seconds
            if sch.stop_at_iteration is not None and self.iteration >= sch.stop_at_iteration:
                if self.is_main:
                    print(f"[INFO] stop_at_iteration reached ({self.iteration}), stopping", flush=True)
                break
            if sch.max_hours is not None:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
