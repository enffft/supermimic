"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.motion_prior.losses import M1JointLossCfg
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
    m1_pose_profile: Literal[
        "legacy_manipulation", "dataset_carry", "dataset_v3_3_l_high_carry",
    ] | None = None
    m1_action_contract: Literal["v2", "v3"] | None = None
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
    # G1's deployable Actor contract is proprioception history + the external
    # reference Goal. Anchor-relative pose is a simulator-only privileged signal
    # and must be opted out of explicitly for historical experiments.
    g1_pure_proprio: bool = True
    velocity_debug_objective: bool = False  # smoke-test only; not a selected Go2 reward
    # Match Go2-SuperMimic's action exploration: 0.1 iid noise on 75% of
    # environments and temporally correlated OU noise on the remaining 25%.
    velocity_gaussian_noise_std: float = 0.10
    velocity_ou_env_fraction: float = 0.25
    velocity_ou_noise_std: float = 0.60
    velocity_ou_theta: float = 2.0
    velocity_reward_linear_tracking: float = 3.0
    velocity_reward_yaw_tracking: float = 1.2
    velocity_reward_direction: float = 0.5
    # Narrower tracking kernels and a zero-speed baseline prevent an idle
    # policy from receiving most of the reward for a nonzero command.
    velocity_reward_tracking_sigma_sq: float = 0.09
    velocity_reward_zero_speed_baseline: bool = True
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
    # The reference chase task uses the Gaussian tag reward directly; retain
    # the legacy term for checkpoint compatibility but disable it by default.
    chase_near_weight: float = 0.0
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
    m1_joint_task_type: str = "balanced"  # balanced | L | M; LM is research-only in v1
    m1_joint_l_task_fraction: float = 0.5
    m1_joint_balanced_fixed_quota: bool = False
    # V3.3 L frame-wise effective Goal P1/P99, rounded outward slightly.
    m1_joint_vx_range: tuple[float, float] = (-0.09, 0.145)
    m1_joint_vx_stop_fraction: float = 0.15
    m1_joint_vx_reverse_fraction: float = 0.35
    m1_joint_vy_range: tuple[float, float] = (-0.05, 0.055)
    m1_joint_wz_range: tuple[float, float] = (-0.17, 0.18)
    m1_joint_target_x_range: tuple[float, float] = (0.12, 0.36)
    m1_joint_target_y_range: tuple[float, float] = (-0.20, 0.20)
    m1_joint_target_z_range: tuple[float, float] = (-0.15, 0.15)


def validate_g1_actor_contract(env: EnvCfg, robot_name: str | None = None) -> None:
    """Reject non-deployable G1 motion inputs before constructing an experiment."""
    name = (robot_name or env.robot_name).lower().replace("-", "_")
    if name != "g1" or env.task_name != "motion_tracking" or not env.g1_pure_proprio:
        return
    if env.anchor_obs or env.anchor_orientation_obs:
        raise ValueError(
            "G1 pure-proprio Actor cannot use anchor-relative pose; set "
            "env.anchor_obs=False and env.anchor_orientation_obs=False (or "
            "explicitly set env.g1_pure_proprio=False for a legacy privileged run)"
        )
    if env.motion_goal_schema != "root_joint":
        raise ValueError(
            "G1 pure-proprio training requires the deployable 68D/frame root_joint "
            "Goal; full_body Goal is a separate research experiment"
        )


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
    checkpoint_interval_minutes: float = 0.0  # optional wall-clock snapshots; 0 disables
    max_hours: float | None = None
    # Optional absolute global-iteration stop; leaves total_iterations/LR schedule unchanged.
    stop_at_iteration: int | None = None


@dataclass
class M1JointTrainingCfg:
    """Staged M1 Action22-reference training and checkpoint contract v4."""

    enabled: bool = False
    prior_checkpoint: Path = Path("/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt")
    expected_prior_sha256: str | None = "05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c"
    resume: Path | None = None  # separate format from old M1/G1/Go2 checkpoints
    state_source: Literal["privileged", "estimated"] = "privileged"
    prior_mode: Literal["prior_mean", "prior_sample"] = "prior_mean"
    prior_input_compatibility: Literal["legacy_v1", "train_constant_compat_v1"] = "legacy_v1"
    prior_horizon: int = 10
    actor_horizon: int = 10
    replanning_cadence: int = 10
    wm_updates_per_event: int = 1
    # Prior stays frozen until matching held-out MuJoCo evidence is supplied.
    prior_updates_per_event: int = 0
    actor_updates_per_event: int = 1
    training_mode: Literal["staged_action22_v1"] = "staged_action22_v1"
    actor_update_mode: Literal["reference_only", "tracking_diagnostic", "wm_task_gated"] = "wm_task_gated"
    actor_wm_gate_evidence_path: Path | None = None
    actor_leg_delta_max_normalized: float = 0.05
    actor_arm_delta_max_normalized: float = 0.05
    actor_wheel_delta_max_normalized: float = 0.10
    actor_gradient_diagnostic_interval: int = 50
    prior_gate_evidence_path: Path | None = None
    exploration_enabled: bool = True
    exploration_env_fraction: float = 0.25
    exploration_leg_std: float = 0.02
    exploration_arm_std: float = 0.015
    exploration_wheel_std: float = 0.05
    exploration_decay_end_fraction: float = 0.75
    prior_lr: float = 1.0e-5
    actor_lr: float = 2.0e-4
    batch_size: int = 32
    micro_batch_size: int = 8
    actor_pretrain_updates: int = 384
    freeze_prior: bool = True
    wm_readiness_min_events: int = 200
    # Optional limits are deliberately unset until task-specific held-out WM
    # calibration exists. Finite replay losses alone are not a quality claim.
    wm_readiness_max_nse_1: float | None = None
    wm_readiness_max_nse_horizon: float | None = None
    grad_clip_norm: float = 1.0
    readiness_window_steps: int = 500
    readiness_min_samples: int = 100
    readiness_min_survival: float = 0.90
    readiness_max_root_tracking_error: float = 0.20  # root xyz Euclidean error, metres
    readiness_max_arm_tracking_error: float = 0.25  # Piper joint position RMSE, radians
    readiness_max_wheel_tracking_error: float = 0.75  # wheel angular-rate RMSE, rad/s
    readiness_max_action_saturation: float = 0.10
    task_type: str = "balanced"  # balanced | L | M; LM is not enabled in v1
    l_task_fraction: float = 0.5
    balanced_replay_sampling: bool = True
    loss: M1JointLossCfg = field(default_factory=M1JointLossCfg)


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
    m1_joint_training: M1JointTrainingCfg = field(default_factory=M1JointTrainingCfg)
    resume: Path | None = None
    init_wm_from: Path | None = None
