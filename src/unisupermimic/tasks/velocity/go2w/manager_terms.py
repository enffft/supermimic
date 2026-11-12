"""Independent mixed action and command terms; reuse Uni terrain sampling/math."""

from dataclasses import dataclass, field

import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg

from unisupermimic.rl.algos.supermimic.geometry import matvec, quat_to_mat
from unisupermimic.tasks.velocity.go2.command_sampler import (
    TerrainAwareVelocitySampler, VelocitySamplerCfg, fixed_domains,
)

from .contract import JOINTS, LEG_JOINTS, MixedControlCfg, mixed_torque
from .mdp import Go2WVelocityMdp
from .reward_core import VelocityRewardCfg


@dataclass(kw_only=True)
class Go2WMixedActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    control: MixedControlCfg = field(default_factory=MixedControlCfg)

    def build(self, env):
        return Go2WMixedAction(self, env)


class Go2WMixedAction(ActionTerm):
    requires_substep_state_feedback = True
    uses_tensor_state_packet = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        cfg.control.validate()
        if tuple(self._entity.joint_names) != JOINTS:
            raise ValueError("Go2W Entity must declare logical leg12 + wheel4 joint order")
        self.q_indices = torch.as_tensor(env.backend.get_joint_state_qpos_indices(LEG_JOINTS), device=env.device)
        self.v_indices = torch.as_tensor(env.backend.get_joint_state_qvel_indices(JOINTS), device=env.device)
        self._raw = torch.zeros(self.num_envs, 16, device=env.device)
        self.executed_raw = torch.zeros_like(self._raw)
        self._target = torch.zeros_like(self._raw)
        self.collection_noise_std = 0.0
        from .contract import DEFAULT_Q, soft_limits
        self.default = self._raw.new_tensor(DEFAULT_Q)
        self.limits = soft_limits(env.device)

    @property
    def action_dim(self):
        return 16

    @property
    def raw_action(self):
        return self._raw

    @property
    def tensor_target(self):
        return self._target

    def process_actions(self, actions):
        self._raw.copy_(actions.clamp(-1, 1))
        noise = torch.randn_like(actions) * self.collection_noise_std if self.collection_noise_std > 0 else 0.0
        self.executed_raw.copy_((self._raw + noise).clamp(-1, 1))
        self._env.velocity_executed_action = self.executed_raw.detach().clone()

    def apply_actions(self):
        views = self._env.backend.get_state_views(("qpos", "qvel"), device=self._device)
        torque = mixed_torque(views["qpos"][:, self.q_indices], views["qvel"][:, self.v_indices],
                              self.executed_raw, self.cfg.control, self.default, self.limits)
        self._target.copy_(torque)
        # The public API maps named joints to MJCF gear=1 torque motors.
        # set_joint_velocity_target merely writes ctrl; it cannot implement speed PD.
        self._entity.set_joint_effort_target(self._target)

    def reset(self, env_ids=None):
        ids = resolve_reset_env_ids(env_ids, num_envs=self.num_envs, device=self._device, owner="Go2WMixedAction")
        self._raw[ids] = 0
        self.executed_raw[ids] = 0
        self._target[ids] = 0


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    sampling_mode: str = "terrain_adaptive"
    sampler: VelocitySamplerCfg = field(default_factory=VelocitySamplerCfg)
    reward: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)
    fixed_command: tuple[float, float, float] | None = None
    terrain_profile: str = "mixed"
    terrain_curriculum: bool = True
    terrain_fixed_level: int = 0
    terrain_max_level: int = 4
    terrain_phase: str = "skill"
    terrain_seed: int = 2026
    terrain_coverage_weights: tuple[float, ...] = (0.2, 0.4, 0.3, 0.1, 0.0)
    terrain_min_episodes: int = 3
    terrain_recycle_probability: float = 0.15
    terrain_target_progress: float = 1.2
    spawn_height: float = 0.42
    spawn_heading: float = 0.0

    def build(self, env):
        return VelocityCommand(self, env)


class VelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.sampling_mode != "terrain_adaptive" or cfg.sampler.stairs_omni or cfg.spawn_heading != 0:
            raise ValueError("Go2W v1 uses the conservative six-domain sampler and heading zero")
        self.robot = env.scene[cfg.entity_name]
        if tuple(self.robot.joint_names) != JOINTS:
            raise ValueError("Go2W named joint order mismatch")
        self.mdp = Go2WVelocityMdp(env.device, control=env.cfg.actions["mixed"].control, reward=cfg.reward)
        self.g = torch.zeros(self.num_envs, 3, device=env.device)
        self.xi = torch.zeros(self.num_envs, self.mdp.context_dim, device=env.device)
        self.a_prev = torch.zeros(self.num_envs, 16, device=env.device)
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.int32, device=env.device)
        self.f = torch.zeros(self.num_envs, dtype=torch.long, device=env.device)
        self._origins = env.scene.env_origins(env.device).clone()
        self.curriculum = None
        self.sampler = TerrainAwareVelocitySampler(cfg.sampler, env.device)
        self.command_domain = fixed_domains(self.num_envs, cfg.sampler.domain_weights, env.device)
        self.command_mode = torch.zeros_like(self.f)
        self.crossing_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)
        self.q_indices = torch.as_tensor(env.backend.get_joint_state_qpos_indices(LEG_JOINTS), device=env.device)
        self.v_indices = torch.as_tensor(env.backend.get_joint_state_qvel_indices(JOINTS), device=env.device)
        layout = env.backend.get_root_state_layout("robot/base")
        self.root_q = torch.as_tensor(layout.qpos_indices, device=env.device)
        self.root_v = torch.as_tensor(layout.qvel_indices, device=env.device)

    @property
    def tensor_sensor_names(self):
        return tuple(joint + "_torque" for joint in JOINTS)

    @property
    def command(self):
        return self.g

    def bind_read_phase(self):
        self._env.scene.read_plan.refresh()

    def state(self):
        plan = self._env.scene.read_plan
        if plan is None or not plan.ready:
            p = self.g.new_zeros(self.num_envs, 3)
            p[:, 2] = self.cfg.spawn_height
            return (p, torch.eye(3, device=self._device).expand(self.num_envs, 3, 3),
                    torch.zeros_like(p), torch.zeros_like(p),
                    self.mdp.default_q.expand(self.num_envs, 12), torch.zeros_like(self.a_prev))
        views = self._env.backend.get_state_views(("qpos", "qvel"), device=self._device)
        root, velocity = views["qpos"][:, self.root_q], views["qvel"][:, self.root_v]
        rotation = quat_to_mat(root[:, 3:7])
        return (root[:, :3], rotation, velocity[:, :3], matvec(rotation, velocity[:, 3:6]),
                views["qpos"][:, self.q_indices], views["qvel"][:, self.v_indices])

    def _resample_command(self, mask):
        level = self.curriculum.terrain_level if self.curriculum is not None else torch.zeros_like(self.f)
        goal, mode, crossing, hold = self.sampler.sample(self.command_domain, level)
        if self.cfg.fixed_command is not None:
            from unisupermimic.tasks.velocity.go2.command_sampler import command_modes
            goal = goal.new_tensor(self.cfg.fixed_command).expand_as(goal)
            mode = command_modes(goal)
            crossing = (self.command_domain != 0) & (mode == 1)
        self.g.copy_(torch.where(mask[:, None], goal, self.g))
        self.command_mode.copy_(torch.where(mask, mode, self.command_mode))
        self.crossing_episode.copy_(torch.where(mask, crossing, self.crossing_episode))
        self.time_left.copy_(torch.where(mask, hold, self.time_left))
        if self.curriculum is not None:
            self.curriculum.crossing_episode.copy_(self.crossing_episode)

    def reset(self, env_ids=None, *, publish_metrics=True):
        ids = resolve_reset_env_ids(env_ids, num_envs=self.num_envs, device=self._device, owner="Go2WVelocity")
        mask = self._reset_mask_buffer.resolve(ids, owner="Go2WVelocity")
        if ids.numel() == 0:
            return {}
        if self.curriculum is not None:
            self._origins[ids] = self.curriculum.reset(ids, self._rng)
        root = self.g.new_zeros(self.num_envs, 13)
        root[:, :3] = self._origins
        if self.mdp.terrain is not None:
            root[:, 2] = self.mdp.terrain.ground_height(root[:, :2])
        root[:, 2] += self.cfg.spawn_height
        root[:, 3] = 1.0
        # Full simulator state retains wheel qpos. Only the MDP projection omits it.
        position = torch.cat((self.mdp.default_q.expand(self.num_envs, 12),
                              self.g.new_zeros(self.num_envs, 4)), -1)
        self.robot.write_motion_state_tensor_to_sim(root_state=root, position=position,
                                                    velocity=torch.zeros_like(self.a_prev), reset_mask=mask)
        self.a_prev[ids] = 0
        self.ep_step[ids] = 0
        self.f[ids] = 0
        self._resample_command(mask)
        self.command_counter[ids] += 1
        return {}

    def _update_metrics(self):
        pass

    def refresh_reset(self, reset_mask):
        pass

    def post_compute_reset(self):
        pass

    def _update_command(self):
        # Executed, clipped/noisy action, not the ActionManager clean proposal.
        live = ~self._env.reset_buf
        self.a_prev.copy_(torch.where(live[:, None], self._env.velocity_executed_action, 0.0))
        self.ep_step.add_(live.to(self.ep_step.dtype))
        self.f.add_(live.to(self.f.dtype))


def observation(env):
    c = env.command_manager.get_term("velocity")
    return c.mdp.obs(c.state(), c.a_prev, c.xi, c.f)


def reward(env):
    c = env.command_manager.get_term("velocity")
    pre = env.velocity_pre_action
    env.velocity_next_state = tuple(t.detach().clone() for t in c.state())
    if c.curriculum is not None:
        c.curriculum.observe(env.velocity_next_state, pre["goal"], c.mdp.hard_termination(env.velocity_next_state), 0.0)
    views = env.scene.read_plan.sensor_tensor_views(c.robot, c.tensor_sensor_names)
    torque = torch.cat([views.values[name] for name in c.tensor_sensor_names], -1)
    env.velocity_measured_torque = torque.detach().clone()
    return c.mdp.reward_transition(pre["state"], env.velocity_next_state, pre["goal"],
                                  env.velocity_executed_action, pre["previous_action"], measured_torque=torque)


def fallen(env):
    c = env.command_manager.get_term("velocity")
    return c.mdp.hard_termination(c.state())
