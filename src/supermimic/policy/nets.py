"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass(frozen=True)
class ActorInputContract:
    """Deployment-facing inputs accepted by the Actor.

    ``obs_hist`` is the five-term proprioception history (joint position,
    joint velocity, base angular velocity, projected gravity, previous clean
    action). ``goal`` is an external task reference. World-model state, ``xi``,
    root error, contacts, torque and critic features are absent by design.
    """

    history: int
    num_joints: int
    goal_dim: int

    @property
    def proprioception_dim_per_frame(self) -> int:
        return 3 * self.num_joints + 6

    @property
    def obs_dim(self) -> int:
        return self.history * self.proprioception_dim_per_frame

    def validate(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> None:
        if obs_hist.ndim != 2 or obs_hist.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Actor expects obs_hist shape (N, {self.obs_dim}), got {tuple(obs_hist.shape)}"
            )
        if goal.ndim != 2 or goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"Actor expects goal shape (N, {self.goal_dim}), got {tuple(goal.shape)}"
            )
        if obs_hist.shape[0] != goal.shape[0]:
            raise ValueError("Actor observation and Goal batch sizes must match")


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    num_joints: int = 29
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos 29, joint_vel 29, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    goal_term_sizes: tuple[int, ...] = (29, 29, 3, 3, 3, 1)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # 680 (+9 with anchor_obs)
    goal_interface: str = "motion"
    # Old checkpoints omit this field and therefore keep the encoded architecture.
    goal_encoding: Literal["encoded", "direct"] = "encoded"
    # ``identity`` is the historical motion/legacy actor contract. New direct
    # command/task actors use tanh so imagined gradients stay inside the same
    # bounded action support used by physical execution.
    action_output: Literal["identity", "tanh"] = "identity"


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    sizes = (cfg.num_joints, cfg.num_joints, 3, 3, cfg.num_joints)
    for d, s in zip(sizes, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = cfg.goal_term_sizes
    if len(sizes) != len(cfg.goal_term_scale) or sum(sizes) == 0:
        raise ValueError("goal_term_sizes and goal_term_scale must describe a non-empty goal layout")
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)])
    k, extra = divmod(goal_dim, sum(sizes))
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int | None = None, act_dim: int | None = None):
        super().__init__()
        obs_dim = obs_dim or cfg.history * (3 * cfg.num_joints + 6)
        act_dim = act_dim or cfg.num_joints
        self.goal_encoding = cfg.goal_encoding
        self.action_output = cfg.action_output
        if self.goal_encoding == "direct":
            input_dim = obs_dim + int(cfg.goal_dim)
            self.register_buffer("goal_scale", _goal_scale_vector(cfg, int(cfg.goal_dim)))
        elif self.goal_encoding == "encoded":
            input_dim = cfg.latent_dim + obs_dim
        else:
            raise ValueError(f"unknown goal encoding {self.goal_encoding!r}")
        self.mlp = MLP(input_dim, cfg.decoder_hidden, act_dim, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z_or_goal: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        obs_scaled = obs_hist * self.scale
        if self.goal_encoding == "direct":
            # Direct velocity mode follows [scaled 420-D proprioception, scaled 3-D goal].
            action = self.mlp(torch.cat([obs_scaled, z_or_goal * self.goal_scale], dim=-1))
            return torch.tanh(action) if self.action_output == "tanh" else action
        return self.mlp(torch.cat([z_or_goal, obs_scaled], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.input_contract = ActorInputContract(cfg.history, cfg.num_joints, cfg.goal_dim)
        # Preserve the encoded checkpoint module names exactly. Direct mode creates no
        # unused encoder parameters or buffers.
        self.encoders = nn.ModuleDict(
            {cfg.goal_interface: MotionEncoder(cfg)} if cfg.goal_encoding == "encoded" else {}
        )
        self.decoder = Decoder(cfg)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str | None = None) -> torch.Tensor:
        self.input_contract.validate(obs_hist, goal)
        if self.cfg.goal_encoding == "direct":
            return self.decoder(goal, obs_hist)
        interface = interface or self.cfg.goal_interface
        return self.decoder(self.encoders[interface](goal), obs_hist)


@dataclass(frozen=True)
class M1PriorActorCfg:
    """Versioned Actor contract for the M1 CVAE joint-training experiment."""

    history: int = 10
    num_joints: int = 22
    reference_horizon: int = 10
    reference_embedding: int = 64
    hidden: tuple[int, ...] = (512, 256)
    version: str = "m1_prior_conditioned_actor_v1"


class M1PriorConditionedActor(nn.Module):
    """M1 mixed-action Actor conditioned on the complete future reference.

    This is a separate schema from ``Policy``; legacy policy state dictionaries
    and their forward signature remain unchanged. The reference path is
    intentionally differentiable with respect to Future31 during Planner BPTT.
    """

    def __init__(self, cfg: M1PriorActorCfg = M1PriorActorCfg()):
        super().__init__()
        if cfg.num_joints != 22 or cfg.reference_horizon != 10:
            raise ValueError("M1PriorConditionedActor v1 requires 22 actions and a 10-frame reference")
        self.cfg = cfg
        obs_dim = cfg.history * (3 * cfg.num_joints + 6)
        self.reference_encoder = nn.Sequential(
            nn.Linear(cfg.reference_horizon * 31, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, cfg.reference_embedding), nn.SiLU(),
        )
        widths = (obs_dim + 8 + cfg.reference_embedding + 2, *cfg.hidden)
        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(widths[:-1], widths[1:])):
            layers.extend((nn.Linear(din, dout), nn.LayerNorm(dout), nn.SiLU()))
        final = nn.Linear(widths[-1], cfg.num_joints)
        nn.init.uniform_(final.weight, -0.003, 0.003)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.decoder = nn.Sequential(*layers)
        # SuperMimic flattens temporal observations term-major:
        # [all joint-pos frames, all joint-vel frames, all angular-velocity
        # frames, all gravity frames, all previous-action frames].
        # M1 proprio contains no base linear velocity term.
        self.register_buffer("obs_scale", torch.cat((
            torch.ones(cfg.history * 22),
            torch.full((cfg.history * 22,), 1.0 / 5.0),
            torch.full((cfg.history * 3,), 1.0 / 2.0),
            torch.ones(cfg.history * 3),
            torch.ones(cfg.history * 22),
        )))
        self.register_buffer("goal_scale", torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]))
        self.register_buffer("reference_scale", torch.tensor(
            [2.0] * 3 + [1.0] * 6 + [0.5] * 12 + [0.5] * 6 + [0.1] * 4
        ).repeat(cfg.reference_horizon))

    def forward(
        self,
        proprio_history: torch.Tensor,
        task_goal: torch.Tensor,
        future_reference: torch.Tensor,
        reference_phase: torch.Tensor,
    ) -> torch.Tensor:
        batch = proprio_history.shape[0]
        expected_obs = self.cfg.history * (3 * self.cfg.num_joints + 6)
        if proprio_history.shape != (batch, expected_obs):
            raise ValueError(f"proprio_history must be [B,{expected_obs}]")
        if task_goal.shape != (batch, 8):
            raise ValueError("task_goal must be [B,8]")
        if future_reference.shape != (batch, 10, 31):
            raise ValueError("future_reference must be [B,10,31]")
        phase = reference_phase.reshape(batch).to(dtype=proprio_history.dtype)
        if torch.any((phase < 0) | (phase >= self.cfg.reference_horizon)):
            raise ValueError("reference_phase must lie in [0,9]")
        phase_angle = phase * (2.0 * torch.pi / self.cfg.reference_horizon)
        phase_code = torch.stack((phase_angle.sin(), phase_angle.cos()), dim=-1)
        reference_embedding = self.reference_encoder(
            (future_reference.reshape(batch, -1) * self.reference_scale).to(proprio_history.dtype)
        )
        inputs = torch.cat((
            proprio_history * self.obs_scale,
            task_goal * self.goal_scale,
            reference_embedding,
            phase_code,
        ), dim=-1)
        return torch.tanh(self.decoder(inputs))


class Critic(nn.Module):
    """V(canonical state 71, goal 680, xi 33, root error 4) -> scalar (one per interface)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)
