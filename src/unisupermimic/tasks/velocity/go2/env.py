"""Independent Go2 Manager environment, using the shared SuperMimic learner."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv

from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY


class Go2VelocityEnv(ManagerBasedRlEnv):
    @property
    def supermimic_contract(self):
        contract = {
            "schema": "go2_velocity_direct_v2",
            "state": "p,quat_wxyz,v_world,w_world,q,qd",
            "actor": "term_major_5x42_fixed_scale_silu_tanh_goal3",
            "joints": list(self.scene["robot"].joint_names),
            "context": "friction,bias12,com3",
            "model_sha256": hashlib.sha256(
                Path(self.cfg.scene.model_file).read_bytes()
            ).hexdigest(),
            "action": "clip[-1,1]; q_default+0.25*a; clamp_soft_limits",
            "reward": asdict(self.supermimic_mdp.params.cfg),
            "reward_manager": {name: asdict(term) for name, term in self.cfg.rewards.items()},
            "terminations": {name: asdict(term) for name, term in self.cfg.terminations.items()},
            "acceleration": "control_period_finite_difference",
            "real_torque": "jointactuatorfrc_sensor",
            "imagined_torque": "pd_proxy",
            "terrain": None if self.cfg.scene.terrain is None else asdict(self.cfg.scene.terrain),
            "commands": asdict(self.cfg.commands["velocity"]),
            "actions": {name: asdict(term) for name, term in self.cfg.actions.items()},
            "max_episode_seconds": self.cfg.max_episode_seconds,
            "auto_reset": self.cfg.auto_reset,
            "events": {name: asdict(term) for name, term in self.cfg.events.items()},
            "sim_dt": self.cfg.sim_dt,
            "ctrl_dt": self.cfg.ctrl_dt,
        }
        # Checkpoints use weights_only loading: persist callable names, never classes.
        return json.loads(
            json.dumps(
                contract,
                default=lambda value: (
                    f"{value.__module__}.{value.__qualname__}" if callable(value) else str(value)
                ),
            )
        )

    def set_collection_exploration(self, standard_deviation):
        self.action_manager.get_term("joint_pos").collection_noise_std = float(standard_deviation)

    @property
    def supermimic_mdp(self):
        return self.command_manager.get_term("velocity").mdp

    def current_info(self):
        c = self.command_manager.get_term("velocity")
        return {
            k: v.detach().clone()
            for k, v in {
                "s": flatten_state(c.state()),
                "f": c.f,
                "xi": c.xi,
                "a_prev": c.a_prev,
                "ep_step": c.ep_step,
                "g": c.g,
            }.items()
        }

    def apply_action(self, actions, state):
        c = self.command_manager.get_term("velocity")
        self.velocity_pre_action = {
            "state": tuple(t.detach().clone() for t in c.state()),
            "goal": c.g.detach().clone(),
            "context": c.xi.detach().clone(),
            "frames": c.f.detach().clone(),
            "ep_step": c.ep_step.detach().clone(),
            "previous_action": c.a_prev.detach().clone(),
            "action": actions.clamp(-1, 1).detach().clone(),
        }
        return super().apply_action(actions.clamp(-1, 1), state)

    def _update_state_in_read_phase(self, state):
        updated = super()._update_state_in_read_phase(state)
        pre = self.velocity_pre_action
        updated.info[SUPERMIMIC_TRANSITION_KEY] = {
            k: v.detach().clone()
            for k, v in {
                "s": flatten_state(pre["state"]),
                "s1": flatten_state(self.velocity_next_state),
                "a": self.velocity_executed_action,
                "a_clean": pre["action"],
                "a_prev": pre["previous_action"],
                "g": pre["goal"],
                "xi": pre["context"],
                "f": pre["frames"],
                "ep_step": pre["ep_step"],
                "r": updated.reward,
                "term": updated.terminated,
                "done": updated.terminated | updated.truncated,
            }.items()
        }
        updated.info[SUPERMIMIC_STATE_KEY] = self.current_info()
        return updated

    def reset(self, env_ids=None, *, seed=None, options=None):
        obs, info = super().reset(env_ids, seed=seed)
        current = self.current_info()
        self._state.info[SUPERMIMIC_STATE_KEY] = current
        info[SUPERMIMIC_STATE_KEY] = current
        return obs, info


def make_go2_env(cfg, num_envs=1, backend_type="mujoco"):
    if (
        set(cfg.actions) != {"joint_pos"}
        or set(cfg.rewards) != {"velocity"}
        or cfg.rewards["velocity"].weight != 1.0
    ):
        raise ValueError(
            "Go2 baseline requires one joint_pos action and one unscaled velocity reward"
        )
    if backend_type not in ("mujoco", "mjwarp"):
        raise ValueError("Go2VelocitySuperMimic supports only audited mujoco and mjwarp backends")
    if cfg.scale_rewards_by_dt or cfg.ctrl_dt != 0.02 or cfg.sim_dt != 0.005:
        raise ValueError(
            "Go2 baseline requires reward already scaled by dt, ctrl_dt=.02, sim_dt=.005"
        )
    if cfg.scene.terrain is not None and backend_type == "mjwarp":
        raise NotImplementedError(
            "UniSim 2.7.1 mjwarp does not support generated terrain or height scanners"
        )
    cfg.validate()
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        base_name="robot/base",
        **env_backend_kwargs(cfg),
    )
    try:
        return Go2VelocityEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise
