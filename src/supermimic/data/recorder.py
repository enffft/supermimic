"""mjlab RecorderTerm that streams simulator transitions into a ReplayBuffer.

Hook usage (see ``mjlab/managers/recorder_manager.py``):

* ``record_post_reset(ids)``  fresh episode start: remember s_0, xi, target frame.
* ``record_post_step()``      for envs that did not reset: write row (s_t, a_t, a_exec, xi, r, ...),
                              then remember s_{t+1} as the next row's state.
* ``record_pre_reset(ids)``   terminal step: write the last transition row plus a *terminal* row
                              holding s_{T+1} with its termination label; qpos/qvel are final here
                              even though derived kinematics are one substep stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ReplayRecorder(RecorderTerm):
    def __init__(self, cfg: ReplayRecorderCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.state_io = StateIO(env)
        self.ctx = SimContext(env, p["foot_geom_regex"], p.get("com_body_name", "torso_link"))
        self.command_name = p.get("command_name", "motion")
        self.action_name = p.get("action_name", "joint_pos")
        self.goal_dim = int(p.get("goal_dim", 0))
        n = env.num_envs
        self.state_dim = self.state_io.state_dim
        self.act_dim = int(env.action_manager.total_action_dim)
        self.xi_dim = self.ctx.xi_dim
        self.buffer = ReplayBuffer(n, int(p["capacity"]), self.state_dim, self.act_dim, self.xi_dim, env.device, self.goal_dim)
        self.prev_s = torch.zeros(n, self.state_dim, device=env.device)
        self.prev_xi = torch.zeros(n, self.xi_dim, device=env.device)
        self.prev_goal = torch.zeros(n, self.goal_dim, device=env.device) if self.goal_dim else None
        self.prev_frame = torch.zeros(n, dtype=torch.long, device=env.device)
        self.prev_clip = torch.zeros(n, dtype=torch.long, device=env.device)
        self.ep_start = torch.ones(n, dtype=torch.bool, device=env.device)
        self.enabled = True
        # Set by Trainer after constructing the World Model. This callback only
        # sees new real transitions; sampled replay windows never update moments.
        self.real_transition_callback = None

    # lazy handles (managers are built after the recorder in load_managers order? no: recorder is last)
    @property
    def command(self):
        return self._env.command_manager.get_term(self.command_name)

    @property
    def action_term(self):
        return self._env.action_manager.get_term(self.action_name)

    def _remember(self, ids: torch.Tensor) -> None:
        cmd = self.command
        self.prev_s[ids] = self.state_io.read()[ids]
        self.prev_xi[ids] = self.ctx.read()[ids]
        frame = getattr(cmd, "frame", None)
        self.prev_frame[ids] = frame[ids] if frame is not None else 0
        lib = getattr(cmd, "library", None)
        self.prev_clip[ids] = lib.frame_clip[frame[ids]] if lib is not None and frame is not None else 0
        if self.prev_goal is not None:
            goal = getattr(cmd, "goal", None)
            if goal is None or goal.shape[-1] != self.goal_dim:
                raise ValueError(f"command {self.command_name!r} must expose a {self.goal_dim}-D goal")
            self.prev_goal[ids] = goal[ids]

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled:
            return
        self._remember(env_ids)
        self.ep_start[env_ids] = True

    def _write_rows(self, ids: torch.Tensor, done_next: torch.Tensor) -> None:
        env = self._env
        s_next = self.state_io.read()[ids]
        if self.real_transition_callback is not None:
            self.real_transition_callback(self.prev_s[ids], s_next)
        self.buffer.add(
            ids,
            s=self.prev_s[ids],
            a=env.action_manager.action[ids],
            a_exec=self.action_term.executed_raw[ids],
            xi=self.prev_xi[ids],
            r=env.reward_buf[ids],
            done=done_next,
            frame=self.prev_frame[ids],
            clip=self.prev_clip[ids],
            terminal=torch.zeros_like(done_next, dtype=torch.bool),
            ep_start=self.ep_start[ids],
            goal=self.prev_goal[ids] if self.prev_goal is not None else None,
        )
        self.ep_start[ids] = False

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled or env_ids.numel() == 0:
            return
        env = self._env
        terminated = env.reset_terminated[env_ids].float()
        self._write_rows(env_ids, terminated)
        # Terminal row: the state reached by the last action (no transition out of it).
        cmd = self.command
        s_next = self.state_io.read()[env_ids]
        zeros_a = torch.zeros(env_ids.numel(), self.act_dim, device=env.device)
        frame = getattr(cmd, "frame", None)
        frame = frame[env_ids] if frame is not None else torch.zeros(env_ids.numel(), dtype=torch.long, device=env.device)
        lib = getattr(cmd, "library", None)
        clip = lib.frame_clip[frame] if lib is not None else torch.zeros_like(frame)
        self.buffer.add(
            env_ids,
            s=s_next,
            a=zeros_a,
            a_exec=zeros_a,
            xi=self.prev_xi[env_ids],
            r=torch.zeros(env_ids.numel(), device=env.device),
            done=terminated,
            frame=frame,
            clip=clip,
            terminal=torch.ones(env_ids.numel(), dtype=torch.bool, device=env.device),
            ep_start=torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device),
            goal=self.prev_goal[env_ids] if self.prev_goal is not None else None,
        )
        self.buffer.end_episode(env_ids)

    def record_post_step(self) -> None:
        if not self.enabled:
            return
        env = self._env
        mask = ~env.reset_buf
        ids = mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        self._write_rows(ids, torch.zeros(ids.numel(), device=env.device))
        self._remember(ids)


@dataclass
class ReplayRecorderCfg(RecorderTermCfg):
    func: type = ReplayRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000, "foot_geom_regex": r"^(left|right)_foot[1-7]_collision$"})
