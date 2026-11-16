"""Go2W environment, reusing the established direct-task transition lifecycle."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.managers import TerminationTermCfg

from unisupermimic.tasks.velocity.go2.env import Go2VelocityEnv
from unisupermimic.tasks.velocity.go2.heightfield import TerrainHeightField
from unisupermimic.tasks.velocity.go2.manager_terms import flat_domain_boundary

from .contract import EFFORT, JOINTS, LAYOUT, LEG_JOINTS, MODEL, POLICY_SPEC, WHEEL_JOINTS
from .terrain import Go2WTerrainCurriculum, mixed_terrain, terrain_contract


def validate_model(path):
    """Tiny XML-only audit, before backend construction; no simulator imports."""
    root = ET.parse(path).getroot()
    names = [j.get("name") for j in root.findall(".//worldbody//joint")]
    if set(names) != set(JOINTS) or len(names) != 16:
        raise ValueError("Go2W physical joint names differ from compact state/action protocol")
    motors = list(root.find("actuator"))
    if len(motors) != 16 or {m.get("joint") for m in motors} != set(JOINTS):
        raise ValueError("Expected exactly one torque motor per Go2W joint")
    for motor in motors:
        if (
            motor.tag != "motor"
            or motor.get("gear") != "1"
            or motor.get("name") != motor.get("joint")
        ):
            raise ValueError("Expected named gear=1 torque motors")
        effort = EFFORT[JOINTS.index(motor.get("joint"))]
        inherited = "-45.43 45.43" if motor.get("class") == "knee" else "-23.7 23.7"
        if tuple(map(float, motor.get("ctrlrange", inherited).split())) != (-effort, effort):
            raise ValueError("Go2W model motor effort limits changed")
    for joint in root.findall(".//worldbody//joint"):
        if joint.get("name") in WHEEL_JOINTS and (
            joint.get("axis") != "0 1 0" or joint.get("limited") != "false"
        ):
            raise ValueError("Go2W wheels must be continuous Y-axis hinges")
    mesh = root.find("asset/mesh")
    mesh_path = Path(path).parent / root.find("compiler").get("meshdir", "") / mesh.get("file")
    provenance = json.loads((MODEL.parent / "provenance.json").read_text())
    if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != provenance["wheel_sha256"]:
        raise ValueError("Go2W wheel collision asset fingerprint changed")


class Go2WVelocityEnv(Go2VelocityEnv):
    """Parent supplies pre/post/reset snapshots, replay commands and executed actions.

    These methods have no joint-width assumptions. Model/command/action/reward
    owners are all Go2W-specific; no Go2 environment factory is invoked.
    """

    def apply_action(self, actions, state):
        result = super().apply_action(actions, state)
        c = self.command_manager.get_term("velocity")
        if c.cfg.unified_curriculum:
            self.velocity_terrain_tags.update(
                traversal=c.traversal.clone(),
                target_id=c.target_id.clone(),
                success_peak=c.success_peak.clone(),
            )
        return result

    @staticmethod
    def validate_distribution_migration(old, new):
        from .checkpoint import validate_stage_migration

        validate_stage_migration(old, new)

    def prepare_supermimic_iteration(self, learner):
        from .diagnostics import begin_iteration

        begin_iteration(self, learner)

    def finish_supermimic_iteration(self, learner):
        from .diagnostics import finish_iteration

        finish_iteration(self, learner)

    def set_collection_exploration(self, standard_deviation):
        self.action_manager.get_term("mixed").collection_noise_std = float(standard_deviation)

    @property
    def supermimic_contract(self):
        cfg = self.cfg
        paths = (Path(cfg.scene.model_file), MODEL.parent / "assets/wheel.stl")
        contract = {
            "schema": "go2w_velocity_supermimic_v2",
            "task": "Go2WVelocitySuperMimic",
            "state_layout": asdict(LAYOUT),
            "state": "p_world,quat_wxyz,v_world,w_world,q_leg12,qd_leg12,qd_wheel4",
            "state_coordinates": "world",
            "actor": asdict(self.supermimic_mdp.policy_spec),
            "actor_observations": "q_leg-default,qd_leg,qd_wheel,imu_omega,projected_gravity,executed_action",
            "joints": list(JOINTS),
            "position_joints": list(LEG_JOINTS),
            "wheel_joints": list(WHEEL_JOINTS),
            "physical_qpos_joints": [
                j.get("name") for j in ET.parse(paths[0]).getroot().findall(".//worldbody//joint")
            ],
            "physical_actuators": [
                m.get("name") for m in ET.parse(paths[0]).getroot().find("actuator")
            ],
            "control_protocol": "substep_effort;leg_position_PD;wheel_velocity_PD;gear1;saturated_v1",
            "actions": {name: asdict(term) for name, term in cfg.actions.items()},
            "commands": asdict(cfg.commands["velocity"]),
            "reward": asdict(self.supermimic_mdp.reward_cfg),
            "reward_protocol": "terrain_plane_alignment_domain_penalties_v2_no_target_detach",
            "curriculum_protocol": "wm_coverage_then_skill_v2_episode_fixed_domain",
            "reward_manager": {name: asdict(term) for name, term in cfg.rewards.items()},
            "terminations": {name: asdict(term) for name, term in cfg.terminations.items()},
            "context": "nominal_physics_empty_context",
            "model_assets_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "terrain": terrain_contract(cfg.scene.terrain),
            "terrain_fingerprint": self.supermimic_mdp.terrain.fingerprint,
            "terrain_sensor": "world_collision_triangles;predicted_pose_each_step;11x7_yaw_scan",
            "real_torque": "last_substep_jointactuatorfrc_noiseless",
            "imagined_torque": "same_saturated_mixed_PD_at_predicted_endpoint",
            "acceleration": "control_period_finite_difference",
            "sim_dt": cfg.sim_dt,
            "ctrl_dt": cfg.ctrl_dt,
            "max_episode_seconds": cfg.max_episode_seconds,
            "auto_reset": cfg.auto_reset,
            "events": {name: asdict(term) for name, term in cfg.events.items()},
        }
        # Learner additionally strictly checks structured integration,
        # terrain conditioning/gradients, history, and network sizes on resume.
        # Preserve the exact default v2 contract for existing checkpoints.
        if not cfg.commands["velocity"].persistent_wm_coverage:
            contract["commands"].pop("persistent_wm_coverage")
        else:
            contract["curriculum_protocol"] = "persistent_wm_eighth_domain_level_quotas"
        for reward in (contract["reward"], contract["commands"]["reward"]):
            if not cfg.commands["velocity"].unified_curriculum:
                for key in (
                    "wheel_radius",
                    "wheel_consistency",
                    "side_drift",
                    "stair_progress",
                    "stair_success",
                    "progress_gamma",
                ):
                    reward.pop(key)
            if reward["stop_reward_command_sigma"] == reward["stop_command_sigma"]:
                reward.pop("stop_reward_command_sigma")
            if not any(reward["tracking_error_cost"]):
                reward.pop("tracking_error_cost")
            if not reward["physical_leg_action_rate"]:
                reward.pop("physical_leg_action_rate")
        if cfg.commands["velocity"].actor_level_lookahead == 0:
            contract["commands"].pop("actor_level_lookahead")
        if not cfg.commands["velocity"].terrain_mode_conditioning:
            contract["commands"].pop("terrain_mode_conditioning")
        else:
            contract["terrain_mode"] = {
                "names": ["flat", "rough", "slope", "stair"],
                "domain_to_mode": [0, 1, 2, 2, 3, 3],
                "source": "environment_task_manager; deployment_requires_explicit_mode",
                "goal": "velocity3_then_mode_onehot4",
            }
        if not cfg.commands["velocity"].unified_curriculum:
            contract["commands"].pop("unified_curriculum")
            contract["commands"].pop("curriculum")
        else:
            contract["curriculum_protocol"] = "ten_level_per_skill_hysteresis_review_and_coverage"
            contract["reward_protocol"] = "shared_endpoint_PD_traversal_potential_once_budget"
            contract["traversal_protocol"] = (
                "target10;immutable_episode;success_hold10;no_actor_target"
            )
        control = contract["actions"]["mixed"]["control"]
        if not control["full_leg_target_range"] and control["leg_limit_fraction"] == 0.90:
            control.pop("full_leg_target_range")
            control.pop("leg_limit_fraction")
        if cfg.actions["mixed"].control.full_leg_target_range:
            contract["control_protocol"] = (
                "substep_effort;leg_default_centered_joint_range_PD;wheel_velocity_PD;gear1;saturated"
            )
        return json.loads(
            json.dumps(
                contract,
                default=lambda x: f"{x.__module__}.{x.__qualname__}" if callable(x) else str(x),
            )
        )


def make_go2w_env(cfg, num_envs=1, backend_type="mujoco"):
    c = cfg.commands["velocity"]
    if c.actor_level_lookahead not in (0, 1):
        raise ValueError("Go2W actor starts admit at most one level beyond measured mastery")
    if backend_type != "mujoco":
        raise NotImplementedError("Go2W v2 supports only the MuJoCo torque-motor backend")
    if (
        set(cfg.actions) != {"mixed"}
        or set(cfg.rewards) != {"velocity"}
        or cfg.rewards["velocity"].weight != 1
        or cfg.scale_rewards_by_dt
    ):
        raise ValueError("Go2W requires one mixed action and dt-scaled shared velocity reward")
    if cfg.sim_dt != 0.005 or cfg.ctrl_dt != 0.02 or cfg.actions["mixed"].control.dt != cfg.ctrl_dt:
        raise ValueError("Go2W v2 requires sim_dt=.005 and control/reward dt=.02")
    unified = c.unified_curriculum
    if c.terrain_profile != "mixed" or c.terrain_max_level != (9 if unified else 7):
        raise ValueError("Go2W curriculum uses the same eight-level level0..7 mixed atlas")
    if c.sampler.terrain_level_count != (10 if unified else 8):
        raise ValueError("Go2W command sampler requires eight terrain levels")
    if c.sampler.terrain_hold_seconds < cfg.max_episode_seconds:
        raise ValueError("Terrain commands must span the whole episode")
    cfg.scene.model_file = str(MODEL.with_name("terrain_robot.xml"))
    if unified:
        from .manager_terms import traversal_finished
        from .terrain import unified_terrain

        cfg.scene.terrain = unified_terrain(c.terrain_seed)
        cfg.scene.fragment_files.append(str(MODEL.parents[2] / "terrains/go2w_traversal.xml"))
        cfg.terminations["traversal_finished"] = TerminationTermCfg(func=traversal_finished)
    else:
        cfg.scene.terrain = mixed_terrain(c.terrain_seed)
    cfg.terminations["flat_domain_boundary"] = TerminationTermCfg(
        func=flat_domain_boundary, time_out=True
    )
    validate_model(cfg.scene.model_file)
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
        env = Go2WVelocityEnv(cfg, backend, num_envs)
        command = env.command_manager.get_term("velocity")
        origins = backend.get_terrain_spawn_data().terrain_origins
        command.mdp.terrain = TerrainHeightField.from_mujoco(
            env.get_playback_model(0),
            device=env.device,
            metadata={
                "seed": c.terrain_seed,
                "origins": origins.tolist(),
                "generator": terrain_contract(cfg.scene.terrain),
            },
        )
        if unified:
            from .heightfield import TraversalHeightField

            command.mdp.terrain = TraversalHeightField.from_field(
                command.mdp.terrain, env.get_playback_model(0)
            )
            from .curriculum import UnifiedCurriculum

            command.curriculum = UnifiedCurriculum(
                num_envs, origins, c.curriculum, command.command_domain, env.device
            )
        else:
            command.curriculum = Go2WTerrainCurriculum(
                num_envs,
                origins,
                enabled=c.terrain_curriculum,
                fixed_level=c.terrain_fixed_level,
                max_level=c.terrain_max_level,
                recycle_probability=c.terrain_recycle_probability,
                phase=c.terrain_phase,
                coverage_weights=c.terrain_coverage_weights,
                min_episodes=c.terrain_min_episodes,
                target_progress=c.terrain_target_progress,
                flat_probability=0.0,
                device=env.device,
            )
        command.curriculum.fixed_domain = command.command_domain
        if c.persistent_wm_coverage and not unified:
            command.curriculum.configure_persistent_coverage()
        points = torch.as_tensor(origins.copy(), device=env.device, dtype=torch.float32)
        offsets = points.new_tensor([[-0.4, -0.25], [-0.4, 0.25], [0.4, -0.25], [0.4, 0.25]])
        ground = command.mdp.terrain.ground_height(points[..., None, :2] + offsets)
        if not torch.isfinite(ground).all() or (ground.amax(-1) - ground.amin(-1)).max() > 0.008:
            raise ValueError("Go2W terrain entry footprint must be finite and planar")
        command.curriculum.origins[..., 2] = command.mdp.terrain.ground_height(points[..., :2])
        return env
    except Exception:
        backend.cleanup_scene_assets()
        raise
