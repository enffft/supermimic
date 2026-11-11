"""Independent Go2 Manager environment, using the shared SuperMimic learner."""

import hashlib
import json
import math
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
        if not any(self.cfg.commands["velocity"].reset_jitter):
            contract["commands"].pop("reset_jitter", None)
        if self.cfg.commands["velocity"].sampling_mode == "legacy_stage":
            for key in ("sampling_mode", "sampler"):
                contract["commands"].pop(key, None)
        if "sampler" in contract["commands"]:
            from .command_sampler import VelocitySamplerCfg

            defaults = asdict(VelocitySamplerCfg())
            # Adding explicit default range fields preserves early-adaptation
            # checkpoint contracts; nondefault ranges remain strictly recorded.
            for key in (
                "flat_velocity_levels",
                "terrain_velocity_ranges",
                "stairs_omni",
                "stairs_mode_weights",
                "stairs_speed_limits",
                "stairs_vx_range",
                "stairs_maneuver_hold_range",
                "stairs_maneuver_entry_offset",
            ):
                if json.dumps(contract["commands"]["sampler"][key]) == json.dumps(defaults[key]):
                    contract["commands"]["sampler"].pop(key)
        new_fields = (
            "terrain_profile",
            "terrain_aware",
            "terrain_seed",
            "terrain_quarter_turns",
            "terrain_curriculum",
            "terrain_fixed_level",
            "terrain_max_level",
            "terrain_recycle_probability",
            "terrain_phase",
            "terrain_coverage_weights",
            "terrain_min_episodes",
            "terrain_target_progress",
            "terrain_flat_probability",
            "terrain_probe_probability",
        )
        if not self.cfg.commands["velocity"].terrain_aware:
            # Preserve v2 flat checkpoint contract byte-for-byte.
            for key in new_fields:
                contract["commands"].pop(key, None)
        else:
            contract["schema"] = "go2_velocity_blind_terrain_v3"
            contract["state_coordinates"] = "world"
            contract["terrain_fingerprint"] = self.supermimic_mdp.terrain.fingerprint
            contract["terrain_height"] = "compiled_collision_triangle; root_xy; per_foot_xy"
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
        self.velocity_terrain_tags = (
            {}
            if c.curriculum is None
            else {
                "terrain_type": c.curriculum.terrain_type.clone(),
                "terrain_level": c.curriculum.terrain_level.clone(),
                "terrain_probe": c.curriculum.probe.clone(),
            }
        )
        if c.sampler is not None:
            self.velocity_terrain_tags.update(
                command_domain=c.command_domain.clone(),
                command_mode=c.command_mode.clone(),
                crossing_episode=c.crossing_episode.clone(),
            )
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
        updated.info[SUPERMIMIC_TRANSITION_KEY].update(self.velocity_terrain_tags)
        return updated

    def reset(self, env_ids=None, *, seed=None, options=None):
        obs, info = super().reset(env_ids, seed=seed)
        current = self.current_info()
        self._state.info[SUPERMIMIC_STATE_KEY] = current
        info[SUPERMIMIC_STATE_KEY] = current
        return obs, info


def make_go2_env(cfg, num_envs=1, backend_type="mujoco"):
    command_cfg = cfg.commands["velocity"]
    if command_cfg.terrain_profile is not None:
        from .terrain import apply_profile, terrain_profile

        apply_profile(
            cfg,
            command_cfg.terrain_profile,
            backend=backend_type,
            seed=command_cfg.terrain_seed,
            quarter_turns=command_cfg.terrain_quarter_turns,
            heading=command_cfg.spawn_heading,
        )
        if command_cfg.terrain_profile != "flat":
            entry_turns = round(command_cfg.spawn_heading / (math.pi / 2))
            if not math.isclose(command_cfg.spawn_heading, entry_turns * math.pi / 2, abs_tol=1e-6):
                raise NotImplementedError(
                    "Terrain entry headings currently require multiples of pi/2"
                )
            cfg.scene.terrain = terrain_profile(
                command_cfg.terrain_profile,
                seed=command_cfg.terrain_seed,
                quarter_turns=command_cfg.terrain_quarter_turns,
                levels=True,
                entry_quarter_turns=entry_turns,
            )
        command_cfg.terrain_aware = True
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
    if command_cfg.sampling_mode == "terrain_adaptive":
        if command_cfg.terrain_profile != "mixed" or command_cfg.spawn_heading != 0:
            raise ValueError("Adaptive domains require the audited mixed atlas, heading zero")
        if command_cfg.sampler.terrain_hold_seconds < cfg.max_episode_seconds:
            raise ValueError("Terrain command must cover the full episode")
        from unilab.managers import TerminationTermCfg

        from .manager_terms import flat_domain_boundary

        cfg.terminations["flat_domain_boundary"] = TerminationTermCfg(
            func=flat_domain_boundary, time_out=True
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
        env = Go2VelocityEnv(cfg, backend, num_envs)
        if command_cfg.terrain_aware:
            from .curriculum import TerrainCurriculum
            from .heightfield import TerrainHeightField

            command = env.command_manager.get_term("velocity")
            if cfg.scene.terrain is None:
                command.mdp.terrain = TerrainHeightField.flat(env.device)
            else:
                spawn = backend.get_terrain_spawn_data()
                origins = spawn.terrain_origins
                command.mdp.terrain = TerrainHeightField.from_mujoco(
                    env.get_playback_model(0),
                    device=env.device,
                    metadata={
                        "seed": command_cfg.terrain_seed,
                        "origins": origins.tolist(),
                        "generator": asdict(cfg.scene.terrain),
                    },
                )
                command.curriculum = TerrainCurriculum(
                    num_envs,
                    origins,
                    enabled=command_cfg.terrain_curriculum,
                    fixed_level=command_cfg.terrain_fixed_level,
                    max_level=command_cfg.terrain_max_level,
                    recycle_probability=command_cfg.terrain_recycle_probability,
                    phase=command_cfg.terrain_phase,
                    coverage_weights=command_cfg.terrain_coverage_weights,
                    min_episodes=command_cfg.terrain_min_episodes,
                    target_progress=command_cfg.terrain_target_progress,
                    flat_probability=command_cfg.terrain_flat_probability,
                    probe_probability=command_cfg.terrain_probe_probability,
                    device=env.device,
                )
                if command.sampler is not None:
                    command.curriculum.fixed_domain = command.command_domain
                    command.curriculum.flat_probability = 0.0
                # All compiled atlas entry footprints must be finite and planar.
                field = command.mdp.terrain
                points = torch.tensor(origins.copy(), dtype=torch.float32, device=env.device)
                offsets = points.new_tensor([[-0.3, -0.2], [-0.3, 0.2], [0.3, -0.2], [0.3, 0.2]])
                samples = field.ground_height(points[..., None, :2] + offsets)
                if (
                    not torch.isfinite(samples).all()
                    or (samples.max(-1).values - samples.min(-1).values).max() > 0.008
                ):
                    raise ValueError("Compiled terrain entry footprint is not safely planar")
                command.curriculum.origins[..., 2] = field.ground_height(points[..., :2])
        return env
    except Exception:
        backend.cleanup_scene_assets()
        raise
