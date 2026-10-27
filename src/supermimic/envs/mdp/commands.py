"""Multi-clip motion command with future-frame goals and return-based reset sampling.

Exposes the same attribute API as mjlab's ``MotionCommand`` (``anchor_pos_w``,
``robot_body_pos_w``, ``body_pos_relative_w`` ...) so mjlab's tracking reward
functions work unchanged, plus:

* ``goal``        (N, K*68) future-frame goal features (plan D5)
* ``root_*``      reference pelvis quantities for the paper's root terminations
* ``clip_end``    (N,) True once the clip's last frame has been reached (time-out)
* return-based reset sampling over 1 s bins of the whole library (plan D14)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from supermimic.envs.mdp.motion_library import MotionLibrary

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class MultiClipMotionCommand(CommandTerm):
    cfg: MultiClipMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_root_body_index = self.robot.body_names.index(cfg.root_body_name)

        default_joint_pos = self.robot.data.default_joint_pos[0].detach()
        self.library = MotionLibrary(
            cfg.npz_files, cfg.body_names, default_joint_pos, device=self.device, bin_frames=cfg.bin_frames,
            compact=cfg.compact_library, store_dtype=torch.float16 if cfg.library_fp16 else torch.float32,
            joint_names=tuple(self.robot.joint_names), goal_joint_names=cfg.goal_joint_names,
            root_body_name=cfg.root_body_name,
        )
        print(f"[INFO] {self.library.summary()}")
        lib = self.library

        n = self.num_envs
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._refresh_ref()
        self.body_pos_relative_w = torch.zeros(n, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(n, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Return-based reset sampling state.
        self.bin_return = torch.full((lib.num_bins,), cfg.g_max, device=self.device)
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=self.device)
        self.max_ep_steps = min(int(math.ceil(env.cfg.episode_length_s / env.step_dt)) + 2, cfg.max_logged_steps)
        # Evaluation hook: when >= 0, the next reset of that env starts exactly at this global frame.
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.ep_rewards = torch.zeros(n, self.max_ep_steps, device=self.device)
        self.ep_bins = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._gamma_pow = cfg.gamma ** torch.arange(self.max_ep_steps, device=self.device, dtype=torch.float64)
        self.bootstrap_fn = None  # optional: env_ids -> value estimate for truncated episodes

        for k in ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
                  "error_joint_pos", "error_joint_vel", "sampling_entropy", "episode_return_to_go"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ reference (mjlab MotionCommand API)

    def _refresh_ref(self) -> None:
        """Cache the reference state / tracked-body states of the current target frames."""
        self._ref = self.library.ref(self.frame)
        self._bodies = self.library.bodies(self.frame)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def goal(self) -> torch.Tensor:
        return self.library.goal_window(self.frame, self.cfg.future_frames)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref.joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._bodies.lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._bodies.ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w[:, self.motion_anchor_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref.root_pos_w + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref.root_quat_w

    # ------------------------------------------------------------------ robot

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_root_body_index]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_root_body_index]

    # ------------------------------------------------------------------ metrics

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    # ------------------------------------------------------------------ reward bookkeeping for the sampler

    def _log_step_rewards(self, env_mask: torch.Tensor | None = None) -> None:
        """Append this step's simulator reward (and the bin it was earned in) to each env's episode log."""
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_mask is None else env_mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        self.ep_rewards[ids, pos] = reward[ids]
        self.ep_bins[ids, pos] = self.library.bin_of_frame(self.frame[ids])
        self.ep_len[ids] = torch.minimum(self.ep_len[ids] + 1, torch.full_like(pos, self.max_ep_steps))

    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        """Fold the finished episodes' discounted returns-to-go into the per-bin estimates."""
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return
        T = int(lens.max().item())
        r = self.ep_rewards[env_ids, :T].double()
        bins = self.ep_bins[env_ids, :T]
        tmask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        r = r * tmask
        gp = self._gamma_pow[:T]
        # Return-to-go R_t = sum_{s>=t} gamma^{s-t} r_s via a reverse cumulative sum.
        disc = r * gp[None, :]
        rtg = torch.flip(torch.cumsum(torch.flip(disc, [1]), 1), [1]) / gp[None, :]
        # Truncated (timed-out) episodes: bootstrap with the critic if available, otherwise
        # drop the steps too close to the truncation point (plan D14).
        terminated = self._env.termination_manager.terminated[env_ids] if hasattr(self._env, "termination_manager") else torch.ones_like(lens, dtype=torch.bool)
        truncated = ~terminated & ~self.clip_end[env_ids]
        if truncated.any():
            if self.bootstrap_fn is not None:
                v = self.bootstrap_fn(env_ids).double()  # (n,)
                remaining = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(0).double()
                rtg = rtg + truncated[:, None] * (cfg.gamma ** remaining) * v[:, None]
            else:
                remaining = lens[:, None] - torch.arange(T, device=self.device)[None, :]
                tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        valid = tmask.reshape(-1)
        b = bins.reshape(-1)[valid]
        g = rtg.reshape(-1)[valid].float()
        if b.numel() == 0:
            return
        sums = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, g)
        cnts = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, torch.ones_like(g))
        touched = cnts > 0
        mean = sums[touched] / cnts[touched]
        self.bin_return[touched] = (1 - cfg.ema_alpha) * self.bin_return[touched] + cfg.ema_alpha * mean
        self.metrics["episode_return_to_go"][env_ids] = rtg[:, 0].float()
        self.ep_len[env_ids] = 0

    # ------------------------------------------------------------------ sampling

    def _sample_frames(self, n: int) -> torch.Tensor:
        lib = self.library
        cfg = self.cfg
        if cfg.sampling_mode == "start":
            clips = torch.randint(0, lib.num_clips, (n,), device=self.device)
            return lib.clip_start[clips]
        if cfg.sampling_mode == "uniform":
            # Uniform over frames (long clips get proportionally more starts), leaving >= 2 steps.
            f = torch.randint(0, lib.num_frames, (n,), device=self.device)
            return torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(0)
        assert cfg.sampling_mode == "return"
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lib.bin_start[b] + (u * (lib.bin_end[b] - lib.bin_start[b]).float()).long()
        f = torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(lib.clip_start[lib.frame_clip[f]])
        ent = -(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        self.metrics["sampling_entropy"][:] = ent
        return f

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        # Episode bookkeeping for the envs being reset (terminal reward is already in reward_buf).
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        if self.ep_len.sum() > 0 or hasattr(self._env, "reward_buf"):
            self._log_step_rewards(mask)
            self._finalize_episodes(env_ids)
        self.ep_len[env_ids] = 0

        sampled = self._sample_frames(len(env_ids))
        forced = self.forced_frame[env_ids]
        self.frame[env_ids] = torch.where(forced >= 0, forced, sampled)
        self.clip_end[env_ids] = False

        r = self.library.ref(self.frame[env_ids])
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids]
        root_ori = r.root_quat_w.clone()
        root_lin_vel = r.root_lin_vel_w.clone()
        root_ang_vel = r.root_ang_vel_w.clone()
        joint_pos = r.joint_pos.clone()
        joint_vel = r.joint_vel

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = root_pos + rs[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), root_ori)
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = root_lin_vel + rs[:, :3]
        root_ang_vel = root_ang_vel + rs[:, 3:]
        lo, hi = self.cfg.joint_position_range
        if hi > lo:
            joint_pos = joint_pos + sample_uniform(lo, hi, joint_pos.shape, device=self.device)

        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)

    def update_relative_body_poses(self) -> None:
        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
        delta_pos_w = robot_anchor_pos
        delta_pos_w[..., 2] = anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos)

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # Clip end must be judged against the clip of the *current* frame (frame+1 may already be
        # the first frame of the next clip in the concatenated library).
        last = self.library.last_frame_of(self.frame)
        if env_ids is None:
            # Per-step path: log the reward earned against the current target frame for the envs
            # that were NOT just reset (their terminal reward was logged in _resample_command).
            reset_buf = getattr(self._env, "reset_buf", None)
            self._log_step_rewards(None if reset_buf is None else ~reset_buf)
            self.frame += 1
        else:
            self.frame[env_ids] += 1
        over = self.frame > last
        if over.any():
            self.clip_end |= over
            self.frame = torch.where(over, last, self.frame)
        self._refresh_ref()
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Deterministic reset to exact reference frames (no noise); used by evaluation."""
        self.ep_len[env_ids] = 0
        self.frame[env_ids] = frame
        self.clip_end[env_ids] = False
        r = self.library.ref(frame)
        self._write_reference_state_to_sim(
            env_ids, r.root_pos_w + self._env.scene.env_origins[env_ids], r.root_quat_w, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_pos, r.joint_vel
        )

    # ------------------------------------------------------------------ debug visualization (viewer only)

    _ghost_model = None

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw the reference pose as a translucent ghost robot (same approach as mjlab's MotionCommand)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            import copy

            m = copy.deepcopy(self._env.sim.mj_model)
            has_visual_geoms = bool(((m.geom_contype == 0) & (m.geom_conaffinity == 0)).any())
            for gi in range(m.ngeom):
                # Some imported URDFs (M1-Piper) provide collision geoms but no
                # separate visual meshes. In that case the collision shapes are
                # the only usable reference model and must remain visible.
                if has_visual_geoms and (m.geom_contype[gi] != 0 or m.geom_conaffinity[gi] != 0):
                    m.geom_rgba[gi, 3] = 0  # hide collision geoms
                else:
                    m.geom_rgba[gi] = (0.2, 0.6, 1.0, 0.45)
            self._ghost_model = m
        indexing = self.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy().reshape(-1)
        joint_q = indexing.joint_q_adr.cpu().numpy().reshape(-1)
        if free_q.size != 7 or joint_q.size != self.library.num_joints:
            raise RuntimeError(
                f"ghost qpos indexing mismatch: free={free_q.size}, joints={joint_q.size}, "
                f"expected 7 and {self.library.num_joints}"
            )
        root_pos = self.root_pos_w.cpu().numpy()
        root_quat = self.root_quat_w.cpu().numpy()
        joint_pos_t = self.joint_pos
        valid = self.library.joint_valid
        if valid is not None:
            # Invalid M1 wheel reference samples are visualized at the robot's
            # documented default wheel angle; they are not added to Goal targets.
            defaults = self.robot.data.default_joint_pos
            joint_pos_t = torch.where(valid[self.frame], joint_pos_t, defaults)
        joint_pos = joint_pos_t.cpu().numpy()
        import numpy as np

        for e in env_indices:
            qpos = self._ghost_model.qpos0.copy()
            qpos[free_q[0:3]] = root_pos[e]
            qpos[free_q[3:7]] = root_quat[e]
            qpos[joint_q] = joint_pos[e]
            visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{e}")


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(CommandTermCfg):
    npz_files: list[str]
    anchor_body_name: str
    root_body_name: str = "pelvis"
    body_names: tuple[str, ...]
    goal_joint_names: tuple[str, ...] | None = None
    entity_name: str = "robot"
    future_frames: int = 10
    bin_frames: int = 50
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    # Return-based sampling (plan D14).
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.05
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    max_logged_steps: int = 1024
    compact_library: bool = False
    library_fp16: bool = True
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> MultiClipMotionCommand:
        return MultiClipMotionCommand(self, env)


class VelocityCommand(CommandTerm):
    """Body-frame velocity commands with real-simulation gated A/B/C progression."""

    cfg: VelocityCommandCfg

    def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.stage = int(cfg.fixed_stage if cfg.fixed_stage is not None else cfg.initial_stage)
        self.stage = max(0, min(self.stage, 2))
        self.eval_count = 0
        self.promotion_streak = 0
        self.promotion_count = 0
        self._episode_stage = torch.full((self.num_envs,), self.stage, dtype=torch.long, device=self.device)
        self._episode_command = torch.zeros_like(self._command)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_move_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_move_error_sq = torch.zeros(self.num_envs, device=self.device)
        self._episode_yaw_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_yaw_error_sq = torch.zeros(self.num_envs, device=self.device)
        self._episode_static_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_static_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_height_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._eval_window: list[dict[str, float | int]] = []
        self.last_eval_metrics: dict[str, float] = {}
        self.metrics["command_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["real_vel_error_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["real_yaw_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def goal(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["command_speed"] = torch.linalg.vector_norm(self._command[:, :2], dim=-1)
        lin = self.robot.data.root_link_lin_vel_b[:, :2]
        yaw = self.robot.data.root_link_ang_vel_b[:, 2]
        self.metrics["real_vel_error_xy"] = torch.linalg.vector_norm(self._command[:, :2] - lin, dim=-1)
        self.metrics["real_yaw_error"] = torch.abs(self._command[:, 2] - yaw)

    def record_real_step(
        self,
        command: torch.Tensor,
        lin_vel_xy: torch.Tensor,
        yaw_rate: torch.Tensor,
        base_height: torch.Tensor,
        terminated: torch.Tensor,
    ) -> None:
        """Accumulate metrics from the pre-reset MuJoCo reward evaluation."""
        move = torch.linalg.vector_norm(command[:, :2], dim=-1) > self.cfg.moving_command_threshold
        yaw_active = command[:, 2].abs() > self.cfg.yaw_command_threshold
        err_xy_sq = (lin_vel_xy - command[:, :2]).square().sum(-1)
        err_yaw_sq = (yaw_rate - command[:, 2]).square()
        static = ~move & ~yaw_active
        self._episode_steps += 1
        self._episode_move_count += move
        self._episode_move_error_sq += err_xy_sq * move
        self._episode_yaw_count += yaw_active
        self._episode_yaw_error_sq += err_yaw_sq * yaw_active
        self._episode_static_count += static
        self._episode_static_speed_sum += torch.linalg.vector_norm(lin_vel_xy, dim=-1) * static
        self._episode_height_sum += base_height
        self._episode_fall |= terminated.bool()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._finish_episodes(env_ids)
        ids = env_ids
        n = len(ids)
        self._command[ids] = self._commands_for_reset(n)
        self._episode_stage[ids] = self.stage
        self._episode_command[ids] = self._command[ids]
        self._episode_steps[ids] = 0
        self._episode_move_count[ids] = 0
        self._episode_move_error_sq[ids] = 0
        self._episode_yaw_count[ids] = 0
        self._episode_yaw_error_sq[ids] = 0
        self._episode_static_count[ids] = 0
        self._episode_static_speed_sum[ids] = 0
        self._episode_height_sum[ids] = 0
        self._episode_fall[ids] = False

    def _sample_a(self, n: int) -> torch.Tensor:
        out = torch.zeros(n, 3, device=self.device)
        moving = torch.rand(n, device=self.device) >= self.cfg.stage_a_static_fraction
        lo, hi = self.cfg.stage_a_vx_range
        out[:, 0] = lo + torch.rand(n, device=self.device) * (hi - lo)
        out[~moving] = 0.0
        return out

    def _sample_b(self, n: int) -> torch.Tensor:
        out = torch.empty(n, 3, device=self.device)
        for i, (lo, hi) in enumerate((self.cfg.stage_b_vx_range, self.cfg.stage_b_vy_range, self.cfg.stage_b_yaw_range)):
            out[:, i] = lo + torch.rand(n, device=self.device) * (hi - lo)
        return out

    def _sample_c(self, n: int) -> torch.Tensor:
        out = torch.empty(n, 3, device=self.device)
        for i, (lo, hi) in enumerate((self.cfg.stage_c_vx_range, self.cfg.stage_c_vy_range, self.cfg.stage_c_yaw_range)):
            out[:, i] = lo + torch.rand(n, device=self.device) * (hi - lo)
        return out

    def _sample_commands(self, n: int, stage: int) -> torch.Tensor:
        if self.cfg.fixed_stage is not None or not self.cfg.auto_curriculum:
            stage = int(self.cfg.fixed_stage if self.cfg.fixed_stage is not None else self.cfg.initial_stage)
        if stage == 0:
            return self._sample_a(n)
        u = torch.rand(n, device=self.device)
        if stage == 1:
            keep_a = u < self.cfg.stage_b_keep_a_fraction
            out = self._sample_b(n)
            if keep_a.any():
                out[keep_a] = self._sample_a(int(keep_a.sum()))
            return out
        keep_a = u < self.cfg.stage_c_keep_a_fraction
        keep_b = (u >= self.cfg.stage_c_keep_a_fraction) & (
            u < self.cfg.stage_c_keep_a_fraction + self.cfg.stage_c_keep_b_fraction
        )
        out = self._sample_c(n)
        if keep_a.any():
            out[keep_a] = self._sample_a(int(keep_a.sum()))
        if keep_b.any():
            out[keep_b] = self._sample_b(int(keep_b.sum()))
        return out

    def _commands_for_reset(self, n: int) -> torch.Tensor:
        """Return either the persistent play command or a newly sampled train goal."""
        if self.cfg.fixed_command is None:
            return self._sample_commands(n, self.stage)
        fixed = torch.as_tensor(self.cfg.fixed_command, dtype=self._command.dtype, device=self.device)
        return fixed.expand(n, 3).clone()

    @staticmethod
    def _command_bins(command: torch.Tensor) -> tuple[int, int]:
        vx, vy = float(command[0]), float(command[1])
        speed = math.hypot(vx, vy)
        if speed < 0.05:
            return -1, -1
        if abs(vx) >= abs(vy):
            direction = 0 if vx >= 0 else 1  # forward / backward
        else:
            direction = 2 if vy >= 0 else 3  # left / right
        magnitude = 0 if speed < 0.25 else (1 if speed < 0.55 else 2)
        return direction, magnitude

    def _finish_episodes(self, env_ids: torch.Tensor) -> None:
        valid = self._episode_steps[env_ids] > 0
        ids = env_ids[valid]
        if ids.numel() == 0:
            return
        device_rows = torch.stack(
            [
                self._episode_stage[ids].float(),
                torch.sqrt(self._episode_move_error_sq[ids] / self._episode_move_count[ids].clamp_min(1)),
                torch.sqrt(self._episode_yaw_error_sq[ids] / self._episode_yaw_count[ids].clamp_min(1)),
                self._episode_static_speed_sum[ids] / self._episode_static_count[ids].clamp_min(1),
                self._episode_move_count[ids],
                self._episode_yaw_count[ids],
                self._episode_static_count[ids],
                self._episode_fall[ids].float(),
                self._episode_steps[ids],
                self._episode_height_sum[ids] / self._episode_steps[ids].clamp_min(1),
            ],
            dim=-1,
        )
        rows = device_rows.detach().cpu().tolist()
        commands = self._episode_command[ids].detach().cpu().tolist()
        for row, command in zip(rows, commands, strict=True):
            ep_stage = int(row[0])
            if ep_stage != self.stage:
                continue
            direction, magnitude = self._command_bins(command)
            self._eval_window.append(
                {
                    "vel_rmse": row[1],
                    "yaw_rmse": row[2],
                    "static_speed": row[3],
                    "moving_steps": row[4],
                    "yaw_steps": row[5],
                    "static_steps": row[6],
                    "fall": row[7],
                    "length": row[8],
                    "height": row[9],
                    "direction": direction,
                    "magnitude": magnitude,
                }
            )
            if len(self._eval_window) >= self.cfg.promotion_window_episodes:
                self._evaluate_window()

    def _evaluate_window(self) -> None:
        rows = self._eval_window[: self.cfg.promotion_window_episodes]
        del self._eval_window[: self.cfg.promotion_window_episodes]
        if not rows:
            return
        moving = [r["vel_rmse"] for r in rows if r["moving_steps"] > 0]
        yaw = [r["yaw_rmse"] for r in rows if r["yaw_steps"] > 0]
        static = [r["static_speed"] for r in rows if r["static_steps"] > 0]
        fall_rate = sum(float(r["fall"]) for r in rows) / len(rows)
        directions = [0, 0, 0, 0]
        magnitudes = [0, 0, 0]
        for r in rows:
            if r["direction"] >= 0:
                directions[int(r["direction"])] += 1
                magnitudes[int(r["magnitude"])] += 1
        vel_rmse = math.sqrt(sum(float(x) ** 2 for x in moving) / max(len(moving), 1)) if moving else float("nan")
        yaw_rmse = math.sqrt(sum(float(x) ** 2 for x in yaw) / max(len(yaw), 1)) if yaw else float("nan")
        static_speed = sum(float(x) for x in static) / len(static) if static else float("nan")
        direction_coverage = sum(x >= self.cfg.min_command_bin_episodes for x in directions) / 4.0
        magnitude_coverage = sum(x >= self.cfg.min_command_bin_episodes for x in magnitudes) / 3.0
        self.eval_count += 1
        self.last_eval_metrics = {
            "real_vel_rmse_xy": vel_rmse,
            "real_yaw_rmse": yaw_rmse,
            "static_speed": static_speed,
            "fall_rate": fall_rate,
            "episode_length": sum(float(r["length"]) for r in rows) / len(rows),
            "mean_base_height": sum(float(r["height"]) for r in rows) / len(rows),
            "command_bin_coverage": direction_coverage,
            "magnitude_bin_coverage": magnitude_coverage,
            "eval_episodes": float(len(rows)),
            "eval_count": float(self.eval_count),
        }
        passed = False
        if self.stage == 0:
            passed = (
                len(moving) >= self.cfg.min_moving_episodes
                and bool(static)
                and vel_rmse < self.cfg.stage_a_max_vel_rmse
                and static_speed < self.cfg.stage_a_max_static_speed
                and fall_rate < self.cfg.max_fall_rate
            )
        elif self.stage == 1:
            passed = (
                len(moving) >= self.cfg.min_moving_episodes
                and len(yaw) >= self.cfg.min_yaw_episodes
                and vel_rmse < self.cfg.stage_b_max_vel_rmse
                and yaw_rmse < self.cfg.stage_b_max_yaw_rmse
                and fall_rate < self.cfg.max_fall_rate
                and direction_coverage >= 1.0
                and magnitude_coverage >= self.cfg.required_magnitude_coverage
            )
        if self.cfg.auto_curriculum and self.cfg.fixed_stage is None and self.stage < 2:
            self.promotion_streak = self.promotion_streak + 1 if passed else 0
            self.last_eval_metrics["promotion_pass"] = float(passed)
            self.last_eval_metrics["promotion_streak"] = float(self.promotion_streak)
            if self.promotion_streak >= self.cfg.consecutive_passing_windows:
                self.stage += 1
                self.promotion_count += 1
                self.promotion_streak = 0
                self._eval_window.clear()
        else:
            self.last_eval_metrics["promotion_pass"] = float(passed)
            self.last_eval_metrics["promotion_streak"] = float(self.promotion_streak)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        # A sampled command is held exactly constant until the next resampling event.
        return

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw body-frame commanded/actual velocity in world coordinates.

        The yaw cue is a short curved arrow around the base; its radius is fixed so
        yaw rate is never mistaken for translational velocity.
        """
        import numpy as np

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w
        base = root_pos + root_pos.new_tensor([0.0, 0.0, 0.32])
        cmd_world = quat_apply(
            root_quat,
            torch.cat([self._command[:, :2], torch.zeros_like(self._command[:, :1])], dim=-1),
        )
        actual_body = self.robot.data.root_link_lin_vel_b
        actual_world = quat_apply(root_quat, actual_body)
        for e in env_indices:
            origin = base[e].detach().cpu().numpy()
            speed = float(torch.linalg.vector_norm(self._command[e, :2]))
            if speed > 1e-4:
                direction = cmd_world[e].detach().cpu().numpy()
                length = min(1.2, max(0.12, speed * 0.8))
                end = origin + direction / max(float(np.linalg.norm(direction)), 1e-8) * length
                visualizer.add_arrow(origin, end, (0.1, 0.9, 0.2, 1.0), width=0.025, label=f"command_{e}")
            if self.cfg.show_actual_velocity:
                actual_speed = float(torch.linalg.vector_norm(actual_world[e, :2]))
                if actual_speed > 1e-4:
                    direction = actual_world[e].detach().cpu().numpy()
                    length = min(1.2, max(0.12, actual_speed * 0.8))
                    end = origin + direction / max(float(np.linalg.norm(direction)), 1e-8) * length
                    visualizer.add_arrow(origin, end, (0.1, 0.55, 1.0, 1.0), width=0.018, label=f"actual_{e}")

            yaw_rate = float(self._command[e, 2])
            if abs(yaw_rate) > 1e-4:
                # Three tangent segments make a compact curved indicator. Positive
                # yaw follows +z; negative yaw follows -z. Radius does not depend
                # on linear command magnitude.
                radius = 0.18
                extent = min(1.35, max(0.25, abs(yaw_rate) * 0.9))
                sign = 1.0 if yaw_rate > 0 else -1.0
                angle0 = -0.5 * extent
                points = []
                for j in range(4):
                    angle = angle0 + sign * extent * j / 3.0
                    local = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.08])
                    points.append(origin + local)
                for j in range(3):
                    visualizer.add_arrow(points[j], points[j + 1], (1.0, 0.65, 0.05, 1.0), width=0.018, label=f"yaw_{e}")

    def state_dict(self) -> dict:
        return {
            "stage": self.stage,
            "eval_count": self.eval_count,
            "promotion_streak": self.promotion_streak,
            "promotion_count": self.promotion_count,
            "eval_window": list(self._eval_window),
            "last_eval_metrics": dict(self.last_eval_metrics),
        }

    def load_state_dict(self, state: dict) -> None:
        self.stage = int(state.get("stage", self.stage))
        self.eval_count = int(state.get("eval_count", 0))
        self.promotion_streak = int(state.get("promotion_streak", 0))
        self.promotion_count = int(state.get("promotion_count", 0))
        self._eval_window = list(state.get("eval_window", []))
        self.last_eval_metrics = dict(state.get("last_eval_metrics", {}))


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    auto_curriculum: bool = True
    initial_stage: int = 0
    fixed_stage: int | None = None
    stage_a_static_fraction: float = 0.15
    stage_a_vx_range: tuple[float, float] = (0.15, 0.55)
    stage_b_vx_range: tuple[float, float] = (-0.2, 0.7)
    stage_b_vy_range: tuple[float, float] = (-0.15, 0.15)
    stage_b_yaw_range: tuple[float, float] = (-0.4, 0.4)
    stage_c_vx_range: tuple[float, float] = (-0.4, 1.0)
    stage_c_vy_range: tuple[float, float] = (-0.3, 0.3)
    stage_c_yaw_range: tuple[float, float] = (-0.8, 0.8)
    stage_b_keep_a_fraction: float = 0.35
    stage_c_keep_a_fraction: float = 0.25
    stage_c_keep_b_fraction: float = 0.25
    moving_command_threshold: float = 0.05
    yaw_command_threshold: float = 0.05
    promotion_window_episodes: int = 256
    consecutive_passing_windows: int = 2
    min_moving_episodes: int = 128
    min_yaw_episodes: int = 64
    min_command_bin_episodes: int = 2
    required_magnitude_coverage: float = 2.0 / 3.0
    max_fall_rate: float = 0.10
    stage_a_max_vel_rmse: float = 0.18
    stage_a_max_static_speed: float = 0.12
    stage_b_max_vel_rmse: float = 0.20
    stage_b_max_yaw_rmse: float = 0.20
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    fixed_command: tuple[float, float, float] | None = None
    show_actual_velocity: bool = False

    def __post_init__(self):
        fractions = (
            self.stage_a_static_fraction,
            self.stage_b_keep_a_fraction,
            self.stage_c_keep_a_fraction,
            self.stage_c_keep_b_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("curriculum sampling fractions must be within [0, 1]")
        if self.stage_c_keep_a_fraction + self.stage_c_keep_b_fraction > 1.0:
            raise ValueError("Stage C retained Stage A/B fractions cannot sum above 1")
        ranges = (
            self.stage_a_vx_range,
            self.stage_b_vx_range,
            self.stage_b_vy_range,
            self.stage_b_yaw_range,
            self.stage_c_vx_range,
            self.stage_c_vy_range,
            self.stage_c_yaw_range,
        )
        if any(len(bounds) != 2 or bounds[0] > bounds[1] for bounds in ranges):
            raise ValueError("each curriculum command range must be an ordered (min, max) pair")
        if self.promotion_window_episodes < 1 or self.consecutive_passing_windows < 1:
            raise ValueError("curriculum evaluation windows must be positive")
        if self.fixed_stage is not None and self.fixed_stage not in (0, 1, 2):
            raise ValueError("fixed_stage must be A=0, B=1, or C=2")
        if self.fixed_command is not None and (len(self.fixed_command) != 3 or not all(math.isfinite(float(x)) for x in self.fixed_command)):
            raise ValueError("fixed_command must contain finite (vx, vy, yaw_rate) values")

    def build(self, env: ManagerBasedRlEnv) -> VelocityCommand:
        return VelocityCommand(self, env)


class WorldPointCommand(CommandTerm):
    """Episode-fixed env-local world target for M1-Piper TCP reaching."""

    cfg: WorldPointCommandCfg

    def __init__(self, cfg: WorldPointCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.target_position_world = torch.zeros(self.num_envs, 3, device=self.device)
        self._tcp_body_ids = torch.tensor(
            [self.robot.body_names.index("piper_x_link7"), self.robot.body_names.index("piper_x_link8")],
            dtype=torch.long, device=self.device,
        )
        self.metrics["target_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self._base_trace: dict[int, list] = {}

    @property
    def goal(self) -> torch.Tensor:
        # Recorder stores this invariant target; Actor observations are computed
        # separately from the current robot state.
        return self.target_position_world

    @property
    def command(self) -> torch.Tensor:
        return self.target_position_world

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        root_pos = self.robot.data.root_link_pos_w[env_ids]
        root_quat = self.robot.data.root_link_quat_w[env_ids]
        tcp_origins = self.robot.data.body_link_pos_w[env_ids][:, self._tcp_body_ids].mean(dim=1)
        env_origins = self._env.scene.env_origins[env_ids]
        count = int(env_ids.numel())
        near = torch.rand(count, device=self.device) < self.cfg.near_target_fraction
        offsets = torch.zeros(count, 3, device=self.device)
        for mask, bounds in (
            (near, (self.cfg.near_x, self.cfg.near_y, self.cfg.near_z)),
            (~near, (self.cfg.far_x, self.cfg.far_y, self.cfg.far_z)),
        ):
            num = int(mask.sum().item())
            if not num:
                continue
            for axis, (lo, hi) in enumerate(bounds):
                offsets[mask, axis] = lo + torch.rand(num, device=self.device) * (hi - lo)
        target_global = tcp_origins + quat_apply(root_quat, offsets)
        target_local = target_global - env_origins
        target_local[:, 2].clamp_(self.cfg.min_target_height, self.cfg.max_target_height)
        self.target_position_world[env_ids] = target_local
        for env_id in env_ids.tolist():
            self._base_trace[int(env_id)] = []

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        return

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Resample after reset poses have been written and forwarded to MuJoCo.

        mjlab resets command terms before ``scene.write_data_to_sim`` and
        ``sim.forward``. Sampling a TCP-relative point in ``reset`` therefore
        sees stale body positions, especially for nonzero environment origins.
        The zero-dt command update runs after forward; refresh only reset envs here
        so the stored target is derived from their actual post-reset TCP.
        """
        super().compute(dt, env_ids)
        if env_ids is not None:
            if float(dt) != 0.0 or env_ids.numel() == 0:
                return
            reset_ids = env_ids
        elif isinstance(dt, torch.Tensor):
            reset_ids = (dt.reshape(-1) <= 0.0).nonzero(as_tuple=False).flatten()
        else:
            return
        if reset_ids.numel() == 0:
            return
        self._resample_command(reset_ids)
        self._update_metrics()

    def _update_metrics(self) -> None:
        tcp = self.robot.data.body_link_pos_w[:, self._tcp_body_ids].mean(dim=1)
        target_global = self.target_position_world + self._env.scene.env_origins
        self.metrics["target_distance"] = torch.linalg.vector_norm(tcp - target_global, dim=-1)
        self.metrics["success"] = (self.metrics["target_distance"] < self.cfg.success_distance).float()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        import numpy as np

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        tcp = self.robot.data.body_link_pos_w[:, self._tcp_body_ids].mean(dim=1)
        target = self.target_position_world + self._env.scene.env_origins
        base = self.robot.data.root_link_pos_w
        for env_id in env_indices:
            e = int(env_id)
            p_tcp = tcp[e].detach().cpu().numpy()
            p_target = target[e].detach().cpu().numpy()
            p_base = base[e].detach().cpu().numpy()
            visualizer.add_sphere(p_target, radius=0.045, color=(1.0, 0.2, 0.15, 1.0), label=f"target_{e}")
            visualizer.add_sphere(p_tcp, radius=0.025, color=(0.1, 0.8, 1.0, 1.0), label=f"tcp_{e}")
            visualizer.add_arrow(p_tcp, p_target, (1.0, 0.8, 0.1, 1.0), width=0.012, label=f"tcp_error_{e}")
            path = self._base_trace.setdefault(e, [])
            if not path or np.linalg.norm(path[-1] - p_base) > 0.05:
                path.append(p_base.copy())
                del path[:-80]
            for j in range(1, len(path)):
                visualizer.add_arrow(path[j - 1], path[j], (0.25, 0.75, 0.35, 0.65), width=0.008, label=f"base_path_{e}_{j}")


@dataclass(kw_only=True)
class WorldPointCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    near_target_fraction: float = 0.6
    near_x: tuple[float, float] = (0.05, 0.30)
    near_y: tuple[float, float] = (-0.20, 0.20)
    near_z: tuple[float, float] = (-0.15, 0.15)
    far_x: tuple[float, float] = (1.00, 1.40)
    far_y: tuple[float, float] = (-0.40, 0.40)
    far_z: tuple[float, float] = (-0.10, 0.25)
    min_target_height: float = 0.10
    max_target_height: float = 1.20
    success_distance: float = 0.08
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def __post_init__(self):
        if not 0.0 <= self.near_target_fraction <= 1.0:
            raise ValueError("near_target_fraction must be within [0, 1]")
        for name in ("near_x", "near_y", "near_z", "far_x", "far_y", "far_z"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be an ordered (min, max) pair")
        if not 0 < self.success_distance or not 0 < self.min_target_height < self.max_target_height:
            raise ValueError("world-point target heights and success distance are invalid")

    def build(self, env: ManagerBasedRlEnv) -> WorldPointCommand:
        return WorldPointCommand(self, env)
