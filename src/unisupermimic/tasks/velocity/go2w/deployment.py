"""Read-only adapter for the user's rl_sar Go2W policy in physics diagnostics.

It translates observation/joint/action conventions explicitly. It does not
load this teacher into the student, modify deployment files, or control a robot.
"""

import hashlib
from pathlib import Path

import torch
import yaml

from unisupermimic.rl.algos.supermimic.geometry import matvec

from .contract import JOINTS


class DeploymentPolicy:
    def __init__(self, directory, num_envs, control, default, limits, device="cpu"):
        directory = Path(directory)
        base_path, cfg_path = directory / "base.yaml", directory / "robot_lab/config.yaml"
        base = yaml.safe_load(base_path.read_text())["go2w"]
        cfg = yaml.safe_load(cfg_path.read_text())["go2w/robot_lab"]
        if (cfg["observations"] != ["ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]
                or cfg["observations_history"] or cfg["num_observations"] != 57
                or cfg["wheel_indices"] != [12, 13, 14, 15]):
            raise ValueError("Unsupported deployment observation convention")
        names = ["robot/" + name.replace("_foot_joint", "_wheel_joint") for name in base["joint_names"]]
        if set(names) != set(JOINTS) or len(names) != 16:
            raise ValueError("Deployment joint list differs from Go2W")
        self.to_deploy = torch.tensor([JOINTS.index(name) for name in names], device=device)
        self.to_logical = torch.argsort(self.to_deploy)
        self.cfg, self.control = cfg, control
        self.default, self.limits = default.to(device), limits.to(device)
        self.deploy_default = torch.tensor(cfg["default_dof_pos"], device=device)
        self.scale = torch.tensor(cfg["action_scale"], device=device)
        self.lower = torch.tensor(cfg["clip_actions_lower"], device=device)
        self.upper = torch.tensor(cfg["clip_actions_upper"], device=device)
        self.previous = torch.zeros(num_envs, 16, device=device)
        policy = directory / "robot_lab" / cfg["model_name"]
        self.policy = torch.jit.load(str(policy), map_location=device).eval()
        self.provenance = {
            "files_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (base_path, cfg_path, policy)},
            "joint_mapping_deploy_from_logical": self.to_deploy.cpu().tolist(),
            "deployment_kp": cfg["rl_kp"], "deployment_kd": cfg["rl_kd"],
            "note": "Inference adapter only; physics uses the training controller, so gains/effort differ from deployment",
        }
        self.last_target = torch.zeros(num_envs, 12, device=device)
        self.last_clipped = torch.zeros(num_envs, 12, dtype=torch.bool, device=device)

    @torch.no_grad()
    def action(self, state, command, episode_start):
        _, rotation, _, angular, q, qd = state
        self.previous[episode_start] = 0
        q16 = torch.cat((q, torch.zeros_like(qd[..., 12:])), -1)[:, self.to_deploy]
        relative = q16 - self.deploy_default
        relative[:, 12:] = 0
        obs = torch.cat((
            matvec(rotation.transpose(-1, -2), angular) * self.cfg["ang_vel_scale"],
            -rotation[..., 2, :], command * command.new_tensor(self.cfg["commands_scale"]),
            relative * self.cfg["dof_pos_scale"], qd[:, self.to_deploy] * self.cfg["dof_vel_scale"],
            self.previous,
        ), -1).clamp(-self.cfg["clip_obs"], self.cfg["clip_obs"])
        raw = self.policy(obs)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Nonfinite deployment policy output")
        raw = raw.clamp(self.lower, self.upper)
        self.previous.copy_(raw)
        scaled = raw * self.scale
        target = (scaled + self.deploy_default)[:, self.to_logical][:, :12]
        self.last_clipped = (target < self.limits[:, 0]) | (target > self.limits[:, 1])
        target = target.clamp(self.limits[:, 0], self.limits[:, 1])
        self.last_target.copy_(target)
        if self.control.full_leg_target_range:
            spans = torch.where(target >= self.default, self.limits[:, 1] - self.default,
                                self.default - self.limits[:, 0])
            legs = (target - self.default) / spans
        else:
            legs = (target - self.default) / self.control.leg_position_scale
        wheels = scaled[:, self.to_logical][:, 12:] / self.control.wheel_velocity_scale
        return torch.cat((legs, wheels), -1).clamp(-1, 1)
