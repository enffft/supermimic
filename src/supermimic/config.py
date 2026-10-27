"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir
    m1_motion_set: str = "C"  # independent runs: A=2 wheels, B=A+arm, C=all six validated clips


@dataclass
class EnvCfg:
    robot_name: str = "g1"  # g1 | go2 | m1_piper
    task_name: str = "motion_tracking"  # motion_tracking | velocity_command | chase_ball | world_point_reach
    velocity_policy_encoding: Literal["direct", "encoded"] = "direct"  # encoded is for legacy Go2 checkpoints
    direct_action_output: Literal["identity", "tanh"] = "tanh"  # set identity only to resume/evaluate legacy direct checkpoints
    velocity_wm_batch_size: int = 1024
    velocity_actor_batch_size: int = 1024
    num_envs: int | None = None  # Defaults: Go2 3072, M1-Piper 2048, G1 4096; CLI overrides remain explicit.
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    history_length: int = 10
    future_frames: int = 10
    motion_goal_schema: Literal["root_joint", "full_body"] = "root_joint"
    compact_library: bool = False
    anchor_obs: bool | None = None  # default: enabled for M1, disabled for G1/Go2
    anchor_orientation_obs: bool = False  # G1 orientation-only (6D) alternative; mutually exclusive with anchor_obs
    velocity_debug_objective: bool = False  # smoke-test only; not a selected Go2 reward
    velocity_gaussian_noise_std: float = 0.18
    velocity_ou_env_fraction: float = 0.30
    velocity_ou_noise_std: float = 0.60
    velocity_ou_theta: float = 2.0
    velocity_reward_linear_tracking: float = 3.0
    velocity_reward_yaw_tracking: float = 1.2
    velocity_reward_direction: float = 0.5
    velocity_curriculum: bool = True
    velocity_curriculum_initial_stage: int = 0
    velocity_curriculum_fixed_stage: int | None = None  # A=0, B=1, C=2; None means automatic
    velocity_stage_a_vx_range: tuple[float, float] = (0.15, 0.55)
    velocity_stage_b_vx_range: tuple[float, float] = (-0.2, 0.7)
    velocity_stage_b_vy_range: tuple[float, float] = (-0.15, 0.15)
    velocity_stage_b_yaw_range: tuple[float, float] = (-0.4, 0.4)
    velocity_stage_c_vx_range: tuple[float, float] = (-0.4, 1.0)
    velocity_stage_c_vy_range: tuple[float, float] = (-0.3, 0.3)
    velocity_stage_c_yaw_range: tuple[float, float] = (-0.8, 0.8)
    velocity_stage_a_static_fraction: float = 0.15
    velocity_stage_b_keep_a_fraction: float = 0.35
    velocity_stage_c_keep_a_fraction: float = 0.25
    velocity_stage_c_keep_b_fraction: float = 0.25
    velocity_promotion_window_episodes: int = 256
    velocity_consecutive_passing_windows: int = 2
    velocity_min_moving_episodes: int = 128
    velocity_min_yaw_episodes: int = 64
    velocity_min_command_bin_episodes: int = 2
    velocity_required_magnitude_coverage: float = 2.0 / 3.0
    velocity_max_fall_rate: float = 0.10
    velocity_stage_a_max_vel_rmse: float = 0.18
    velocity_stage_a_max_static_speed: float = 0.12
    velocity_stage_b_max_vel_rmse: float = 0.20
    velocity_stage_b_max_yaw_rmse: float = 0.20
    velocity_resample_seconds: float = 10.0
    velocity_min_safe_height: float = 0.22  # zero-action stable Go2 root height was 0.2827 m
    velocity_fall_height: float = 0.17
    velocity_fall_tilt_deg: float = 70.0
    chase_auto_curriculum: bool = True
    chase_initial_stage: int = 0
    chase_fixed_stage: int | None = None
    chase_promotion_window_episodes: int = 64
    chase_consecutive_passing_windows: int = 2
    chase_promotion_success_rate: float = 0.35
    chase_tag_radius: float = 0.20
    chase_approach_weight: float = 1.5
    chase_near_weight: float = 3.0
    chase_tag_weight: float = 15.0
    chase_tag_sigma: float = 0.30
    chase_facing_weight: float = 0.5
    chase_joint_torque_weight: float = -2.0e-5
    chase_joint_acceleration_weight: float = -2.5e-7
    m1_fall_height: float = 0.20
    m1_success_distance: float = 0.08
    m1_target_curriculum: bool = True
    m1_initial_target_stage: int = 0
    m1_fixed_target_stage: int | None = None
    m1_target_promotion_window_episodes: int = 128
    m1_target_promotion_success_rate: float = 0.35
    m1_target_consecutive_passing_windows: int = 2
    m1_near_target_fraction: float = 0.95
    m1_near_x: tuple[float, float] = (0.12, 0.36)
    m1_near_y: tuple[float, float] = (-0.20, 0.20)
    m1_near_z: tuple[float, float] = (-0.15, 0.15)
    m1_far_x: tuple[float, float] = (1.00, 1.40)
    m1_far_y: tuple[float, float] = (-0.40, 0.40)
    m1_far_z: tuple[float, float] = (-0.10, 0.25)
    m1_world_point_wm_batch_size: int = 256
    m1_world_point_actor_batch_size: int = 256
    m1_joint_velocity_weight: float = -1.0e-6
    m1_joint_acceleration_weight: float = -1.0e-7
    m1_leg_action_magnitude_weight: float = -0.002


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    wm_only_iterations: int = 200  # WM update events before Go2 actor/critic updates
    m1_wm_only_iterations: int = 200  # let the M1 WM fit the selected motion bank before BPTT
    g1_motion_wm_only_updates: int = 0  # opt-in WM prefit for a new G1 motion architecture
    actor_ramp_iterations: int = 1000  # actual Actor optimizer updates after WM-only warmup
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
    init_wm_from: Path | None = None
