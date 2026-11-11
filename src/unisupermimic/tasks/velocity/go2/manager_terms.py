"""Go2 Manager terms. Commands resample after reward and are snapshotted in replay."""

import math
from dataclasses import dataclass, field

import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, ManagerTermBase

from unisupermimic.rl.algos.supermimic.geometry import matvec, quat_to_mat

from .command_sampler import TerrainAwareVelocitySampler, VelocitySamplerCfg, fixed_domains
from .contract import DEFAULT_Q, JOINTS, soft_limits
from .mdp import Go2VelocityMdp


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    reset_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)  # eval XY/yaw/joints
    sampling_mode: str = "legacy_stage"
    sampler: VelocitySamplerCfg = field(default_factory=VelocitySamplerCfg)
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    # Fixed stage is explicit; automatic promotion is gated pending matched evaluation.
    stage: int = 0
    standing_fraction: float = 0.05
    fixed_command: tuple[float, float, float] | None = None
    spawn_heading: float = 0.0
    spawn_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_randomization: bool = False
    terrain_profile: str | None = None  # None preserves legacy flat path
    terrain_aware: bool = False
    terrain_seed: int = 2026
    terrain_quarter_turns: int = 0
    terrain_curriculum: bool = False
    terrain_fixed_level: int = 0
    terrain_max_level: int = 4
    terrain_recycle_probability: float = 0.15
    terrain_phase: str = "skill"
    terrain_coverage_weights: tuple[float, ...] = (0.2, 0.4, 0.3, 0.1, 0.0)
    terrain_min_episodes: int = 3
    terrain_target_progress: float = 1.2
    terrain_flat_probability: float = 0.15
    terrain_probe_probability: float = 0.0

    def build(self, env):
        return VelocityCommand(self, env)


class VelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.stage not in (0, 1, 2):
            raise ValueError("Go2 command stage must be 0, 1, or 2")
        self.robot = env.scene[cfg.entity_name]
        if tuple(self.robot.joint_names) != JOINTS:
            raise ValueError("Go2 requires exact named FL/FR/RL/RR joint order")
        self.mdp = Go2VelocityMdp(env.device)
        self.g = torch.zeros(self.num_envs, 3, device=env.device)
        self.xi = torch.zeros(self.num_envs, 16, device=env.device)
        self.xi[:, 0] = 1.0
        self.a_prev = torch.zeros(self.num_envs, 12, device=env.device)
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.int32, device=env.device)
        self.f = torch.zeros(self.num_envs, dtype=torch.long, device=env.device)
        self._origins = env.scene.env_origins(env.device).clone()
        self.curriculum = None
        if cfg.sampling_mode not in ("legacy_stage", "terrain_adaptive"):
            raise ValueError("Unknown velocity command sampling mode")
        self.sampler = (
            TerrainAwareVelocitySampler(cfg.sampler, env.device)
            if cfg.sampling_mode == "terrain_adaptive"
            else None
        )
        self.command_domain = (
            fixed_domains(self.num_envs, cfg.sampler.domain_weights, env.device)
            if self.sampler is not None
            else None
        )
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=env.device)
        self.crossing_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)
        # Resolve names once; public qpos/qvel are authoritative after the last
        # physics substep (body sensors can otherwise lag by one substep).
        self.q_indices = torch.as_tensor(
            env.backend.get_joint_state_qpos_indices(JOINTS), device=env.device
        )
        self.v_indices = torch.as_tensor(
            env.backend.get_joint_state_qvel_indices(JOINTS), device=env.device
        )
        layout = env.backend.get_root_state_layout("robot/base")
        self.root_q = torch.as_tensor(layout.qpos_indices, device=env.device)
        self.root_v = torch.as_tensor(layout.qvel_indices, device=env.device)

    @property
    def tensor_sensor_names(self):
        return tuple(name + "_torque" for name in JOINTS)

    @property
    def command(self):
        return self.g

    def bind_read_phase(self):
        self._env.scene.read_plan.refresh()

    def state(self):
        plan = self._env.scene.read_plan
        if plan is None or not plan.ready:
            n, d = self.num_envs, self._env.device
            p = torch.zeros(n, 3, device=d)
            p[:, 2] = 0.42
            return (
                p,
                torch.eye(3, device=d).expand(n, 3, 3),
                torch.zeros(n, 3, device=d),
                torch.zeros(n, 3, device=d),
                self.mdp.default_q.expand(n, 12),
                torch.zeros(n, 12, device=d),
            )
        views = self._env.backend.get_state_views(("qpos", "qvel"), device=self._device)
        root_q = views["qpos"][:, self.root_q]
        root_v = views["qvel"][:, self.root_v]
        R = quat_to_mat(root_q[:, 3:7])
        return (
            root_q[:, :3] if self.cfg.terrain_aware else root_q[:, :3] - self._origins,
            R,
            root_v[:, :3],
            matvec(R, root_v[:, 3:6]),
            views["qpos"][:, self.q_indices],
            views["qvel"][:, self.v_indices],
        )

    def _resample_command(self, mask):
        if self.cfg.fixed_command is not None:
            sampled = self.g.new_tensor(self.cfg.fixed_command).expand_as(self.g)
        elif self.sampler is not None:
            level = (
                self.curriculum.terrain_level
                if self.curriculum is not None
                else torch.zeros_like(self.command_domain)
            )
            sampled, mode, crossing, hold = self.sampler.sample(self.command_domain, level)
            if self.sampler.cfg.stairs_omni:
                # Episode intent is latched at reset. A maneuver resample can
                # never turn its accumulated history into traversal evidence.
                ongoing_stairs = (self.command_domain >= 4) & (self.ep_step > 0)
                crossing = crossing & ~ongoing_stairs
                hold = torch.where(
                    ongoing_stairs, hold.clamp(*self.sampler.cfg.stairs_maneuver_hold_range), hold
                )
            self.command_mode.copy_(torch.where(mask, mode, self.command_mode))
            self.crossing_episode.copy_(torch.where(mask, crossing, self.crossing_episode))
            self.time_left.copy_(torch.where(mask, hold, self.time_left))
            if self.curriculum is not None:
                self.curriculum.crossing_episode.copy_(self.crossing_episode)
        else:
            lo, hi = (
                ((0.25, 0.0, 0.0), (0.6, 0.0, 0.0)),
                ((-0.2, -0.15, -0.4), (0.7, 0.15, 0.4)),
                ((-0.4, -0.3, -0.8), (1.0, 0.3, 0.8)),
            )[self.cfg.stage]
            sampled = self._rng.uniform(
                self.g.new_tensor(lo), self.g.new_tensor(hi), tuple(self.g.shape)
            )
            sampled = torch.where(
                (self._rng.random((self.num_envs,)) < self.cfg.standing_fraction)[:, None],
                0.0,
                sampled,
            )
        self.g.copy_(torch.where(mask[:, None], sampled, self.g))

    def reset(self, env_ids=None, *, publish_metrics=True):
        ids = resolve_reset_env_ids(
            env_ids, num_envs=self.num_envs, device=self._device, owner="Go2Velocity.reset"
        )
        mask = self._reset_mask_buffer.resolve(ids, owner="Go2Velocity.reset")
        if ids.numel() == 0:
            return {}
        if self.curriculum is not None:
            self._origins[ids] = self.curriculum.reset(ids, self._rng)
        early_command = self.sampler is not None and self.sampler.cfg.stairs_omni
        if early_command:
            self.ep_step[ids] = 0
            self._resample_command(mask)
        root = torch.zeros(self.num_envs, 13, device=self._device)
        root[:, :3] = self._origins
        root[:, :3] += root.new_tensor(self.cfg.spawn_offset)
        if early_command:
            # Pure lateral/turning at the flat approach never reaches stairs.
            # Maneuvers start on an existing tread; traversals retain entry reset.
            maneuver = (self.command_domain >= 4) & ~self.crossing_episode
            direction = root.new_tensor(
                (math.cos(self.cfg.spawn_heading), math.sin(self.cfg.spawn_heading))
            )
            root[:, :2] += (
                maneuver[:, None] * direction * self.sampler.cfg.stairs_maneuver_entry_offset
            )
        if self.curriculum is not None:
            direction = root.new_tensor(
                (math.cos(self.cfg.spawn_heading), math.sin(self.cfg.spawn_heading))
            )
            root[:, :2] += self.curriculum.probe_offset[:, None] * direction
        if self.cfg.terrain_aware and self.mdp.terrain is not None:
            root[:, 2] = self.mdp.terrain.ground_height(root[:, :2]) + self.cfg.spawn_offset[2]
        root[:, 2] += 0.42
        if self.curriculum is not None:
            self.curriculum.start_xy[ids] = root[ids, :2]
        root[:, 3] = math.cos(self.cfg.spawn_heading / 2)
        root[:, 6] = math.sin(self.cfg.spawn_heading / 2)
        position = self.mdp.default_q.expand(self.num_envs, 12).contiguous()
        if any(self.cfg.reset_jitter):
            xy, yaw, joints = self.cfg.reset_jitter
            root[:, :2] += self._rng.uniform(-xy, xy, (self.num_envs, 2))
            angle = self.cfg.spawn_heading + self._rng.uniform(-yaw, yaw, (self.num_envs,))
            root[:, 3], root[:, 6] = torch.cos(angle / 2), torch.sin(angle / 2)
            position = position + self._rng.uniform(-joints, joints, (self.num_envs, 12))
        if self.cfg.sampler.stairs_safe_spawn and self.mdp.terrain is not None:
            from .reset_geometry import safe_spawn_lift

            # Explicit new-atlas reset protocol, opt-in and checkpointed in sampler cfg.
            root[:, 2] += safe_spawn_lift(root, position, self.mdp.terrain)
        if self.curriculum is not None:
            self.curriculum.start_xy[ids] = root[ids, :2]
        self.robot.write_motion_state_tensor_to_sim(
            root_state=root,
            position=position,
            velocity=torch.zeros_like(self.a_prev),
            reset_mask=mask,
        )
        self.a_prev[ids] = 0
        if self.cfg.domain_randomization:
            self.xi[ids, 1:13] = self._rng.uniform(-0.01, 0.01, (ids.numel(), 12))
        self.ep_step[ids] = 0
        self.f[ids] = 0
        if not early_command:
            self._resample_command(mask)
        if self.sampler is None:
            self.time_left[ids] = self.cfg.resampling_time_range[0]
        self.command_counter[ids] += 1
        return {}

    def _update_metrics(self):
        pass

    def refresh_reset(self, reset_mask):
        pass

    def post_compute_reset(self):
        pass

    def _update_command(self):
        self.a_prev.copy_(self._env.action_manager.action)
        self.ep_step.add_(1)
        self.f.add_(1)


@dataclass(kw_only=True)
class VelocityActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    ou_env_fraction: float = 0.25
    ou_noise_std: float = 0.3
    ou_theta: float = 2.0

    def build(self, env):
        return VelocityAction(self, env)


class VelocityAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 12, device=env.device)
        self._target = torch.zeros_like(self._raw)
        self.executed_raw = torch.zeros_like(self._raw)
        self.ou_state = torch.zeros_like(self._raw)
        self.ou_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)
        self.collection_noise_std = 0.0
        self.default = self._raw.new_tensor(DEFAULT_Q)
        self.limits = soft_limits(env.device)

    @property
    def action_dim(self):
        return 12

    @property
    def raw_action(self):
        return self._raw

    @property
    def tensor_target(self):
        return self._target

    def process_actions(self, actions):
        self._raw.copy_(actions.clamp(-1, 1))
        noise = torch.zeros_like(actions)
        if self.collection_noise_std > 0:
            noise = self.collection_noise_std * torch.randn_like(actions)
            if self.cfg.ou_env_fraction > 0 and self.cfg.ou_noise_std > 0:
                dt = self._env.cfg.ctrl_dt
                self.ou_state.add_(
                    -self.cfg.ou_theta * self.ou_state * dt
                    + self.cfg.ou_noise_std * math.sqrt(dt) * torch.randn_like(actions)
                )
                noise = torch.where(self.ou_mask[:, None], self.ou_state, noise)
        self.executed_raw.copy_((self._raw + noise).clamp(-1, 1))
        # Snapshot before automatic reset clears ActionTerm buffers.
        self._env.velocity_executed_action = self.executed_raw.detach().clone()

    def apply_actions(self):
        self._target.copy_(
            (self.default + 0.25 * self.executed_raw).clamp(self.limits[:, 0], self.limits[:, 1])
        )
        # Public Entity API materializes the joint->actuator mapping by name.
        self._entity.set_full_width_joint_position_target(self._target)

    def reset(self, env_ids=None):
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._env.device,
            owner="Go2VelocityAction.reset",
        )
        self._raw[ids] = 0.0
        self.executed_raw[ids] = 0.0
        self.ou_state[ids] = 0.0
        self.ou_mask[ids] = (
            torch.rand(ids.numel(), device=self._env.device) < self.cfg.ou_env_fraction
        )


def observation(env):
    c = env.command_manager.get_term("velocity")
    return c.mdp.obs(c.state(), c.a_prev, c.xi, c.f)


def reward(env):
    c = env.command_manager.get_term("velocity")
    pre = env.velocity_pre_action
    # Snapshot post-physics state before any manager reset or command resampling.
    env.velocity_next_state = tuple(t.detach().clone() for t in c.state())
    if c.curriculum is not None:
        c.curriculum.observe(
            env.velocity_next_state,
            pre["goal"],
            c.mdp.hard_termination(env.velocity_next_state),
            c.cfg.spawn_heading,
        )
    views = env.scene.read_plan.sensor_tensor_views(c.robot, c.tensor_sensor_names)
    torque = torch.cat([views.values[name] for name in c.tensor_sensor_names], -1)
    env.velocity_measured_torque = torque.detach().clone()
    return c.mdp.reward_transition(
        pre["state"],
        env.velocity_next_state,
        pre["goal"],
        pre["action"],
        pre["previous_action"],
        executed_action=env.velocity_executed_action,
        measured_torque=torque,
    )


def fallen(env):
    c = env.command_manager.get_term("velocity")
    return c.mdp.hard_termination(c.state())


class ContextEvent(ManagerTermBase):
    """Optional source DR: startup COM xyz; shared foot friction at each reset.

    Encoder bias lives only in proprioception/context, never in position targets.
    Backend capability binding raises if the requested physical write is unsupported.
    """

    uses_startup_rows = True

    def __init__(self, cfg, env):
        super().__init__(env)
        self.startup = cfg.mode == "startup"
        robot = env.scene["robot"]
        self.robot = robot
        body_ids, _ = robot.find_bodies("robot/base", preserve_order=True)
        geom_ids, _ = robot.find_geoms(r"robot/(FL|FR|RL|RR)_foot_collision", preserve_order=True)
        self.body_ids, self.ipos = robot.bind_body_ipos_write(body_ids, term_name="go2_context")
        self.geom_ids, self.friction = robot.bind_geom_friction_write(
            geom_ids, term_name="go2_context"
        )

    @staticmethod
    def selected(defaults, ids):
        values = defaults.to(device=ids.device, dtype=torch.float32)
        return (
            values.expand((ids.numel(), *values.shape)).clone()
            if values.ndim == 2
            else values.index_select(0, ids).clone()
        )

    def __call__(self, env, env_ids):
        ids = resolve_reset_env_ids(
            env_ids, num_envs=env.num_envs, device=env.device, owner="Go2ContextEvent"
        )
        if ids.numel() == 0:
            return
        c = env.command_manager.get_term("velocity")
        if self.startup:
            c.xi[ids, 13:16] = env.rng.uniform(-0.025, 0.025, (ids.numel(), 3))
        else:
            c.xi[ids, 0] = env.rng.uniform(0.3, 1.2, (ids.numel(),))
        ipos = self.selected(self.ipos, ids) + c.xi[ids, 13:16].unsqueeze(1)
        if self.startup:
            self.robot.record_startup_body_ipos_to_sim(
                ipos, body_ids=self.body_ids, env_ids=ids, term_name="go2_context"
            )
        else:
            self.robot.write_body_ipos_to_sim(
                ipos, body_ids=self.body_ids, env_ids=ids, term_name="go2_context"
            )
            fr = self.selected(self.friction, ids)
            fr[..., 0] = c.xi[ids, :1]
            self.robot.write_geom_friction_to_sim(
                fr, geom_ids=self.geom_ids, env_ids=ids, term_name="go2_context"
            )


def flat_domain_boundary(env):
    """Episode truncation before leaving the known flat atlas tile, never a fall.

    General flat commands need room behind and beside the robot; spawn at tile
    center and keep the whole footprint inside its 8m flat collision geometry.
    """
    c = env.command_manager.get_term("velocity")
    if c.sampler is None or c.curriculum is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (c.command_domain == 0) & (
        (c.state()[0][:, :2] - c._origins[:, :2]).abs().amax(-1) > 3.2
    )
