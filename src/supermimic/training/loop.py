"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.tasks import MotionTrackingTask, VelocityCommandTask
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import actor_ramp_scale, cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel
from supermimic.tasks.velocity_reward import VelocityRewardCfg, VelocityRobotParams


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


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        robot_spec = get_robot_spec(cfg.env.robot_name)
        if cfg.env.num_envs is None:
            cfg.env.num_envs = 2048 if robot_spec.name == "m1_piper" else 4096
        is_motion = cfg.env.task_name == "motion_tracking"
        if cfg.env.task_name not in ("motion_tracking", "velocity_command"):
            raise ValueError(f"unknown task {cfg.env.task_name!r}")
        if cfg.env.task_name == "velocity_command" and robot_spec.name != "go2":
            raise ValueError("velocity_command is currently supported only for Go2")
        if robot_spec.name == "go2" and is_motion:
            raise ValueError("Go2 has no Motion Library; select env.task_name=velocity_command")
        if robot_spec.name != "go2" and not is_motion:
            raise ValueError("velocity_command is currently supported only for Go2")
        if cfg.env.anchor_obs is None:
            cfg.env.anchor_obs = robot_spec.name == "m1_piper"
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

        # Build every network dimension from the selected robot/task while leaving the
        # default G1 values and initialization path unchanged.
        n_joints = robot_spec.num_joints
        goal_term_sizes = (robot_spec.amp_dim, robot_spec.amp_dim, 3, 3, 3, 1) if is_motion else (3,)
        goal_term_scale = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0) if is_motion else (1.0,)
        goal_dim = cfg.env.future_frames * (2 * robot_spec.amp_dim + 10) + (9 if cfg.env.anchor_obs else 0) if is_motion else 3
        cfg.policy.num_joints = n_joints
        cfg.policy.history = cfg.env.history_length
        cfg.policy.goal_dim = goal_dim
        cfg.policy.goal_interface = "motion" if is_motion else "velocity"
        cfg.policy.goal_term_sizes = goal_term_sizes
        cfg.policy.goal_term_scale = goal_term_scale
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
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
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if not is_motion:
            files = []
            if cfg.data.library is not None:
                raise ValueError("Go2 velocity-command task does not consume a motion library")
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
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                velocity_gaussian_noise_std=cfg.env.velocity_gaussian_noise_std,
                velocity_ou_env_fraction=cfg.env.velocity_ou_env_fraction,
                velocity_ou_noise_std=cfg.env.velocity_ou_noise_std,
                velocity_ou_theta=cfg.env.velocity_ou_theta,
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
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        command_name = "motion" if is_motion else "velocity"
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
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, feat_dim=self.wm.spec.feat_dim, goal_dim=goal_dim, xi_dim=n_joints + 4, pcfg=cfg.policy).to(self.device)
        if is_motion:
            self.task = MotionTrackingTask(self.lib, cfg.env.future_frames, cfg.env.anchor_obs)
        else:
            reward_cfg = VelocityRewardCfg(min_safe_height=cfg.env.velocity_min_safe_height)
            default_joint_pos = robot.data.default_joint_pos[0].detach().clone()
            reward_params = VelocityRobotParams(
                default_joint_pos=default_joint_pos,
                soft_joint_limits=robot.data.soft_joint_pos_limits[0].detach().clone(),
                action_scales=default_joint_pos.new_tensor(robot_spec.action_scales),
                kp=default_joint_pos.new_tensor(robot_spec.kp),
                kd=default_joint_pos.new_tensor(robot_spec.kd),
                effort_limits=default_joint_pos.new_full((n_joints,), 23.5),
                dt=robot_spec.control_dt,
                cfg=reward_cfg,
            )
            self.task = VelocityCommandTask(
                params=reward_params,
                dt=robot_spec.control_dt,
                reward_cfg=reward_cfg,
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
        self.is_go2_velocity = robot_spec.name == "go2" and not is_motion
        self.is_m1_motion = robot_spec.name == "m1_piper" and is_motion
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

    # ------------------------------------------------------------------ checkpoints

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
            "config": asdict(self.cfg),
            "wm_only_updates": self.wm_only_updates,
            "actor_updates": self.actor_updates,
            "actor_start_iteration": self.actor_start_iteration,
            "utd_acc": self._utd_acc,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        if hasattr(self.command, "bin_return"):
            state["bin_return"] = self.command.bin_return
            state["bin_memory"] = self.command.bin_memory
        if hasattr(self.command, "state_dict"):
            state["command_state"] = self.command.state_dict()
        action_term = self.recorder.action_term
        if hasattr(action_term, "state_dict"):
            state["action_state"] = action_term.state_dict()
        torch.save(state, self.log_dir / name)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
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

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        wm_only_target = (
            sch.wm_only_iterations if self.is_go2_velocity
            else sch.m1_wm_only_iterations if self.is_m1_motion
            else 0
        )
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
            obs, rew, term, tout, extras = env.step(action)
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
                elif self.is_go2_velocity:
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
                    self._utd_acc -= 1.0
                    wm_lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    actor_scheduled_lr = cosine_lr(self.iteration, sch.total_iterations, cfg.policy_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    critic_lr = cosine_lr(self.iteration, sch.total_iterations, cfg.policy_trainer.critic_lr, sch.lr_final_ratio, sch.lr_decay_fraction)
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
                            if self.is_go2_velocity and self.is_main:
                                with torch.no_grad():
                                    start_err = self.wm_trainer.online_error(self.buffer, n=256, k=8)
                                    self.go2_actor_start_nse8 = float(start_err[-1])
                                if self.go2_actor_start_nse8 > 0.5:
                                    print(
                                        f"[WARN] Go2 actor starts with 8-step WM NSE={self.go2_actor_start_nse8:.4f} "
                                        "(configured review threshold 0.5); see wm/actor_start_nse8",
                                        flush=True,
                                    )
                        ramp_updates = max(int(sch.actor_ramp_iterations), 0)
                        if self.is_go2_velocity and ramp_updates > 0:
                            ramp_scale = actor_ramp_scale(self.iteration, self.actor_start_iteration, ramp_updates)
                        else:
                            ramp_scale = 1.0
                        actor_lr = actor_scheduled_lr * ramp_scale
                        self.policy_trainer.set_lr(actor_lr, critic_lr)
                        t0 = time.perf_counter()
                        for _ in range(sch.policy_updates_per_wm_update):
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
                    finished_falls = [fall_sum / cnt] if cnt > 0 and self.is_go2_velocity else []
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    finished_falls = []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k not in ("policy/actor_lr", "policy/critic_lr", "wm/lr", "schedule/wm_only_updates") else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    if self.cfg.env.task_name == "motion_tracking":
                        self.writer.add_scalar("train/episode_norm_return", tr, it)
                    elif self.is_go2_velocity:
                        self.writer.add_scalar("go2/real_episode_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                if self.is_go2_velocity and self.is_main and finished_falls:
                    self.writer.add_scalar("go2/fall_rate", sum(finished_falls) / len(finished_falls), it)
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
                if self.is_main:
                    task_prefix = "go2" if self.is_go2_velocity else ("m1" if self.is_m1_motion else "g1")
                    for key, value in self.real_reward_term_sum.items():
                        self.writer.add_scalar(f"{task_prefix}/reward_real/{key}", float(value / sch.log_interval), it)
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=256, k=8)
                        self.writer.add_scalar("wm/nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main and self.is_go2_velocity and hasattr(self, "go2_actor_start_nse8"):
                    self.writer.add_scalar("wm/actor_start_nse8", self.go2_actor_start_nse8, it)
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
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
                        f"WM-only {self.wm_only_updates}/{wm_only_target} actor-ramp {agg.get('policy/actor_ramp_scale', 0.0)/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                finished_falls = []
                self.real_reward_term_sum = {}
                self.m1_step_metric_sum = {}
                self.m1_step_metric_count.zero_()
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
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
