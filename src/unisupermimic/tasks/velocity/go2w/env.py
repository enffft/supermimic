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
from .terrain import Go2WTerrainCurriculum, mixed_terrain


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
        if motor.tag != "motor" or motor.get("gear") != "1" or motor.get("name") != motor.get("joint"):
            raise ValueError("Expected named gear=1 torque motors")
        effort = EFFORT[JOINTS.index(motor.get("joint"))]
        inherited = "-45.43 45.43" if motor.get("class") == "knee" else "-23.7 23.7"
        if tuple(map(float, motor.get("ctrlrange", inherited).split())) != (-effort, effort):
            raise ValueError("Go2W model motor effort limits changed")
    for joint in root.findall(".//worldbody//joint"):
        if joint.get("name") in WHEEL_JOINTS and (joint.get("axis") != "0 1 0" or joint.get("limited") != "false"):
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

    def set_collection_exploration(self, standard_deviation):
        self.action_manager.get_term("mixed").collection_noise_std = float(standard_deviation)

    @property
    def supermimic_contract(self):
        cfg = self.cfg
        paths = (Path(cfg.scene.model_file), MODEL.parent / "assets/wheel.stl")
        contract = {
            "schema": "go2w_velocity_supermimic_v1",
            "task": "Go2WVelocitySuperMimic",
            "state_layout": asdict(LAYOUT),
            "state": "p_world,quat_wxyz,v_world,w_world,q_leg12,qd_leg12,qd_wheel4",
            "state_coordinates": "world",
            "actor": asdict(POLICY_SPEC),
            "actor_observations": "q_leg-default,qd_leg,qd_wheel,imu_omega,projected_gravity,executed_action",
            "joints": list(JOINTS),
            "position_joints": list(LEG_JOINTS),
            "wheel_joints": list(WHEEL_JOINTS),
            "physical_qpos_joints": [j.get("name") for j in ET.parse(paths[0]).getroot().findall(".//worldbody//joint")],
            "physical_actuators": [m.get("name") for m in ET.parse(paths[0]).getroot().find("actuator")],
            "control_protocol": "substep_effort;leg_position_PD;wheel_velocity_PD;gear1;saturated_v1",
            "actions": {name: asdict(term) for name, term in cfg.actions.items()},
            "commands": asdict(cfg.commands["velocity"]),
            "reward": asdict(self.supermimic_mdp.reward_cfg),
            "reward_manager": {name: asdict(term) for name, term in cfg.rewards.items()},
            "terminations": {name: asdict(term) for name, term in cfg.terminations.items()},
            "context": "nominal_physics_empty_context",
            "model_assets_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "terrain": asdict(cfg.scene.terrain),
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
        return json.loads(json.dumps(contract, default=lambda x: (
            f"{x.__module__}.{x.__qualname__}" if callable(x) else str(x))))


def make_go2w_env(cfg, num_envs=1, backend_type="mujoco"):
    c = cfg.commands["velocity"]
    if backend_type != "mujoco":
        raise NotImplementedError("Go2W v1 supports only the MuJoCo torque-motor backend")
    if (set(cfg.actions) != {"mixed"} or set(cfg.rewards) != {"velocity"}
            or cfg.rewards["velocity"].weight != 1 or cfg.scale_rewards_by_dt):
        raise ValueError("Go2W requires one mixed action and dt-scaled shared velocity reward")
    if cfg.sim_dt != 0.005 or cfg.ctrl_dt != 0.02 or cfg.actions["mixed"].control.dt != cfg.ctrl_dt:
        raise ValueError("Go2W v1 requires sim_dt=.005 and control/reward dt=.02")
    if c.terrain_profile != "mixed" or c.terrain_max_level != 4:
        raise ValueError("Go2W curriculum uses the same conservative level0..4 mixed atlas")
    if c.sampler.terrain_hold_seconds < cfg.max_episode_seconds:
        raise ValueError("Terrain commands must span the whole episode")
    cfg.scene.model_file = str(MODEL.with_name("terrain_robot.xml"))
    cfg.scene.terrain = mixed_terrain(c.terrain_seed)
    cfg.terminations["flat_domain_boundary"] = TerminationTermCfg(func=flat_domain_boundary, time_out=True)
    validate_model(cfg.scene.model_file)
    cfg.validate()
    backend = create_backend(backend_type, cfg.scene, num_envs, cfg.sim_dt,
                             body_state_required=True, base_name="robot/base", **env_backend_kwargs(cfg))
    try:
        env = Go2WVelocityEnv(cfg, backend, num_envs)
        command = env.command_manager.get_term("velocity")
        origins = backend.get_terrain_spawn_data().terrain_origins
        command.mdp.terrain = TerrainHeightField.from_mujoco(env.get_playback_model(0), device=env.device,
            metadata={"seed": c.terrain_seed, "origins": origins.tolist(), "generator": asdict(cfg.scene.terrain)})
        command.curriculum = Go2WTerrainCurriculum(num_envs, origins, enabled=c.terrain_curriculum,
            fixed_level=c.terrain_fixed_level, max_level=c.terrain_max_level,
            recycle_probability=c.terrain_recycle_probability, phase=c.terrain_phase,
            coverage_weights=c.terrain_coverage_weights, min_episodes=c.terrain_min_episodes,
            target_progress=c.terrain_target_progress, flat_probability=0.0, device=env.device)
        command.curriculum.fixed_domain = command.command_domain
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
