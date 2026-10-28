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

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.tasks import ChaseBallTask, MotionTrackingTask, VelocityCommandTask, WorldPointReachTask
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import actor_ramp_scale, scheduled_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel
from supermimic.tasks.velocity_reward import VelocityRewardCfg, VelocityRobotParams
from supermimic.tasks.world_point_reward import WorldPointRobotParams, m1_tcp_proxy_indices
from supermimic.tasks.chase_reward import ChaseBallRewardCfg


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


class Trainer:
    @staticmethod
    def _resolved_num_envs(robot_name: str, requested: int | None = None, task_name: str | None = None) -> int:
        if requested is not None:
            return requested
        name = get_robot_spec(robot_name).name
        if name == "m1_piper":
            return 256 if task_name == "world_point_reach" else 2048
        return 3072 if name == "go2" else 4096

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.resume is not None and cfg.init_wm_from is not None:
            raise ValueError("--resume and --init-wm-from are mutually exclusive")
        robot_spec = get_robot_spec(cfg.env.robot_name)
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
        if cfg.env.task_name not in ("motion_tracking", "velocity_command", "chase_ball", "world_point_reach"):
            raise ValueError(f"unknown task {cfg.env.task_name!r}")
        if (is_velocity or is_chase) and robot_spec.name != "go2":
            raise ValueError(f"{cfg.env.task_name} is currently supported only for Go2")
        if is_world_point and robot_spec.name != "m1_piper":
            raise ValueError("world_point_reach is currently supported only for M1-Piper")
        if robot_spec.name == "go2" and not (is_velocity or is_chase):
            raise ValueError("Go2 has no Motion Library; select velocity_command or chase_ball")
        if robot_spec.name == "m1_piper" and not (is_motion or is_world_point):
            raise ValueError("M1-Piper supports motion_tracking and world_point_reach")
        if cfg.env.anchor_obs is None:
            cfg.env.anchor_obs = robot_spec.name == "m1_piper" and is_motion
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

        # Build every network dimension from the selected robot/task while leaving the
        # default G1 values and initialization path unchanged.
        n_joints = robot_spec.num_joints
        full_body_goal = is_motion and robot_spec.name == "g1" and cfg.env.motion_goal_schema == "full_body"
        goal_term_sizes = (268,) if full_body_goal else (robot_spec.amp_dim, robot_spec.amp_dim, 3, 3, 3, 1) if is_motion else (3, 3) if is_chase else (3,)
        goal_term_scale = (1.0,) if full_body_goal else (1.0, 5.0, 1.0, 2.0, 1.0, 1.0) if is_motion else (1.0, 1.0) if is_chase else (1.0,)
        anchor_dim = 9 if cfg.env.anchor_obs else 6 if cfg.env.anchor_orientation_obs else 0
        per_frame_goal_dim = 268 if full_body_goal else (2 * robot_spec.amp_dim + 10)
        goal_dim = cfg.env.future_frames * per_frame_goal_dim + anchor_dim if is_motion else 6 if is_chase else 3
        cfg.policy.num_joints = n_joints
        cfg.policy.history = cfg.env.history_length
        cfg.policy.goal_dim = goal_dim
        cfg.policy.goal_interface = "motion" if is_motion else "velocity" if is_velocity else "chase_ball" if is_chase else "world_point_reach"
        # Keep G1/M1 on the original encoded goal path; only Go2 gets the compact
        # proprioception + velocity-command actor.
        cfg.policy.goal_encoding = cfg.env.velocity_policy_encoding if is_velocity else "direct" if (is_world_point or is_chase) else "encoded"
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
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        command_name = "motion" if is_motion else "velocity" if is_velocity else "chase_ball" if is_chase else "world_point"
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
        else:
            tcp_indices = m1_tcp_proxy_indices(heads.fk)
            world_point_params = WorldPointRobotParams(
                default_joint_pos=robot.data.default_joint_pos[0].detach().clone(),
                soft_joint_limits=robot.data.soft_joint_pos_limits[0].detach().clone(),
                dt=robot_spec.control_dt,
                fall_height=cfg.env.m1_fall_height,
                success_distance=cfg.env.m1_success_distance,
                joint_velocity_weight=cfg.env.m1_joint_velocity_weight,
                joint_acceleration_weight=cfg.env.m1_joint_acceleration_weight,
                leg_action_magnitude_weight=cfg.env.m1_leg_action_magnitude_weight,
            )
            self.task = WorldPointReachTask(
                params=world_point_params,
                fk=heads.fk,
                tcp_body_indices=tcp_indices,
                dt=robot_spec.control_dt,
            )
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.task, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward if is_motion else 1.0
        self.has_tracking_return = is_motion
        self.is_go2_velocity = robot_spec.name == "go2" and is_velocity
        self.is_go2_chase = robot_spec.name == "go2" and is_chase
        self.is_go2_task = self.is_go2_velocity or self.is_go2_chase
        self.is_m1_motion = robot_spec.name == "m1_piper" and is_motion
        self.is_m1_world_point = robot_spec.name == "m1_piper" and is_world_point
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
        if cfg.resume is not None:
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
            "m1_piper": {"world_point_reach"},
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
        current_robot_spec = get_robot_spec(current_robot)
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
        current_spec = get_robot_spec(current_robot)
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
        if self.is_m1_motion or getattr(self, "is_m1_world_point", False):
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
            "action_modes": tuple(get_robot_spec(self.cfg.env.robot_name).action_modes),
            "action_scales": tuple(get_robot_spec(self.cfg.env.robot_name).action_scales),
            "kp": tuple(get_robot_spec(self.cfg.env.robot_name).kp),
            "kd": tuple(get_robot_spec(self.cfg.env.robot_name).kd),
            "effort_limits": tuple(get_robot_spec(self.cfg.env.robot_name).effort_limits),
        })
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
        state = {
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

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

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
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        finished_falls: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            pre_frame = self.command.frame.clone() if self.is_m1_motion else None
            action = self.act(obs)
            action_buffer_ptr = self.buffer.ptr.clone() if self.is_go2_task else None
            action_ou_mask = (
                self.recorder.action_term._ou_mask.clone() if self.is_go2_task else None
            )
            obs, rew, term, tout, extras = env.step(action)
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
                elif self.is_go2_task or self.is_m1_world_point:
                    finished_returns.extend(self.ep_reward_sum[ids].tolist())
                    finished_falls.extend(term[ids].float().tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
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
                    if self.buffer.size < int(cfg.wm_trainer.window) + 1 or not bool(self.buffer.window_mask(int(cfg.wm_trainer.window)).any()):
                        break
                    self._utd_acc -= 1.0
                    wm_lr = scheduled_lr(sch.lr_schedule, self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    actor_scheduled_lr = scheduled_lr(sch.lr_schedule, self.iteration, sch.total_iterations, cfg.policy_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    critic_lr = scheduled_lr(sch.lr_schedule, self.iteration, sch.total_iterations, cfg.policy_trainer.critic_lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(wm_lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    m_pi = {}
                    ramp_scale = 0.0
                    actor_lr = 0.0
                    policy_enabled = wm_only_target <= 0 or self.wm_only_updates >= wm_only_target
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
                            if self.is_go2_task and ramp_updates > 0:
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
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["wm/lr"] = wm_lr
                    agg["policy/actor_lr"] = actor_lr
                    agg["policy/critic_lr"] = critic_lr
                    agg["policy/actor_ramp_scale"] = ramp_scale
                    agg["schedule/wm_only_updates"] = float(self.wm_only_updates)
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
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
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    finished_falls = []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    last_value_keys = (
                        "policy/actor_lr", "policy/critic_lr", "policy/actor_ramp_scale",
                        "wm/lr", "schedule/wm_only_updates",
                    )
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (1 if k in last_value_keys else agg_n), it)
                if self.is_main and self.policy_trainer.last_imag_diagnostics:
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
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                if self.is_go2_task and self.is_main and finished_falls:
                    self.writer.add_scalar("go2/fall_rate", sum(finished_falls) / len(finished_falls), it)
                if self.is_m1_world_point and self.is_main and finished_falls:
                    fall_rate = sum(finished_falls) / len(finished_falls)
                    self.writer.add_scalar("m1/world_point_fall_rate", fall_rate, it)
                    self.writer.add_scalar("m1/fall_rate", fall_rate, it)
                if self.is_go2_velocity and self.is_main:
                    self.writer.add_scalar("go2/curriculum_stage", float(self.command.stage), it)
                    self.writer.add_scalar("go2/curriculum_eval_count", float(self.command.eval_count), it)
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
                if self.is_main:
                    task_prefix = "go2" if self.is_go2_task else ("m1" if self.is_m1_motion or self.is_m1_world_point else "g1")
                    for key, value in self.real_reward_term_sum.items():
                        self.writer.add_scalar(f"{task_prefix}/reward_real/{key}", float(value / sch.log_interval), it)
                    for key, value in self.task_diag_sum.items():
                        self.writer.add_scalar(f"{task_prefix}/{key}", float(value / sch.log_interval), it)
                    for key, value in agg.items():
                        if key.startswith("policy/imag_r/"):
                            term = key.removeprefix("policy/imag_r/")
                            self.writer.add_scalar(f"{task_prefix}/reward_imag/{term}", value / max(agg_n, 1), it)
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
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
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"WM-only {self.wm_only_updates}/{wm_only_target} actor-ramp {agg.get('policy/actor_ramp_scale', 0.0):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                finished_falls = []
                self.real_reward_term_sum = {}
                self.task_diag_sum = {}
                self.m1_step_metric_sum = {}
                self.m1_step_metric_count.zero_()
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.stop_at_iteration is not None and self.iteration >= sch.stop_at_iteration:
                if self.is_main:
                    print(f"[INFO] stop_at_iteration reached ({self.iteration}), stopping", flush=True)
                break
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
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
