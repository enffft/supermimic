"""Export both Go2 actors through the existing rl_sar TorchScript interface."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import yaml

from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots.spec import get_robot_spec


class DirectMean(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.register_buffer("scale", torch.cat((policy.decoder.scale, policy.decoder.goal_scale)))
        self.net = copy.deepcopy(policy.decoder.mlp.net).cpu().eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x * self.scale))


class RlSarHistoryActor(nn.Module):
    """rl_sar time-major raw 5x45 -> SM term-major 210 + current command 3."""
    def __init__(self, mean):
        super().__init__()
        self.mean = mean

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        frames = history.reshape(-1, 5, 45)
        obs = torch.cat((frames[:, :, :12].flatten(1), frames[:, :, 12:24].flatten(1),
                         frames[:, :, 24:27].flatten(1), frames[:, :, 27:30].flatten(1),
                         frames[:, :, 30:42].flatten(1), frames[:, -1, 42:45]), dim=-1)
        return self.mean(obs)


def load_policy(checkpoint: Path):
    """PPO is converted to the existing inference-only SM schema for play/eval."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = Path(json.loads((checkpoint / "progress.json").read_text())["checkpoint"])
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "actor_state_dict" in ckpt:
        meta = json.loads((checkpoint.parent / "comparison_config.json").read_text())
        config = copy.deepcopy(meta["reference_config"])
        config["env"] = {**config["env"], **meta["env_args"]}
        cfg = PolicyCfg(**config["policy"])
        policy = Policy(cfg)
        source = ckpt["actor_state_dict"]
        state = {"decoder.scale": source["mlp.0.scale"][:210],
                 "decoder.goal_scale": source["mlp.0.scale"][210:]}
        state.update({"decoder.mlp.net." + k.removeprefix("mlp.1."): v
                      for k, v in source.items() if k.startswith("mlp.1.")})
        policy.load_state_dict(state, strict=True)
        algorithm = "PPO"
    else:
        config = ckpt["config"]
        policy = Policy(PolicyCfg(**config["policy"]))
        policy.load_state_dict(ckpt["policy"], strict=True)
        algorithm = ckpt.get("source_algorithm", "SuperMimic")
    cfg = policy.cfg
    if (cfg.history, cfg.num_joints, cfg.goal_dim, cfg.goal_encoding, cfg.action_output) != (5, 12, 3, "direct", "tanh"):
        raise ValueError("export requires the Go2 direct/tanh 5-frame velocity actor")
    if config["env"]["robot_name"] != "go2" or config["env"]["task_name"] != "velocity_command":
        raise ValueError("not a Go2 velocity checkpoint")
    saved_actuator = ckpt.get("actuator_model_version", config.get("actuator_model_version"))
    if saved_actuator != get_robot_spec("go2").actuator_model_version:
        raise ValueError(f"checkpoint actuator contract {saved_actuator!r} differs from current Go2")
    return policy.cpu().eval(), config, algorithm


def export(checkpoint: Path, output: Path, policy_name: str):
    torch.set_num_threads(1)
    if checkpoint.is_dir():
        checkpoint = Path(json.loads((checkpoint / "progress.json").read_text())["checkpoint"])
    policy, config, algorithm = load_policy(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    wrapped = RlSarHistoryActor(DirectMean(policy)).eval()
    scripted = torch.jit.script(wrapped)
    scripted.save(str(output / "policy.pt"))
    generator = torch.Generator().manual_seed(2026)
    frames = torch.randn(16, 5, 45, generator=generator)
    # Distinct temporal values and commands expose ordering/stale-command bugs.
    obs = torch.cat([frames[:, :, a:b].flatten(1) for a, b in ((0, 12), (12, 24), (24, 27), (27, 30), (30, 42))], -1)
    with torch.no_grad():
        expected = policy(obs, frames[:, -1, 42:])
        actual = torch.jit.load(str(output / "policy.pt"))(frames.flatten(1))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    spec = get_robot_spec("go2")
    settings = {
        "model_name": "policy.pt", "dt": 0.005, "decimation": 4,
        "num_observations": 45, "num_of_dofs": 12,
        "observations": ["dof_pos", "dof_vel", "ang_vel", "gravity_vec", "actions", "commands"],
        "observations_history": [4, 3, 2, 1, 0], "observations_history_priority": "time",
        "history_warmstart": True, "clip_obs": 1e9,
        "clip_actions_lower": [-1.] * 12, "clip_actions_upper": [1.] * 12,
        "rl_kp": list(spec.kp), "rl_kd": list(spec.kd),
        "fixed_kp": [60.] * 12, "fixed_kd": [5.] * 12,
        "action_scale": list(spec.action_scales), "wheel_indices": [],
        "lin_vel_scale": 1., "ang_vel_scale": 1., "dof_pos_scale": 1., "dof_vel_scale": 1.,
        "commands_scale": [1., 1., 1.], "torque_limits": list(spec.effort_limits),
        "default_dof_pos": list(spec.default_joint_pos),
        "joint_mapping": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
    }
    (output / "config.yaml").write_text(yaml.safe_dump({"go2/" + policy_name: settings}, sort_keys=False))
    torch.save({"policy": policy.state_dict(), "config": config, "source_algorithm": algorithm,
                "actuator_model_version": spec.actuator_model_version,
                "inference_only": True, "source_checkpoint": str(checkpoint.resolve())}, output / "policy_inference.pt")
    manifest = {
        "algorithm": algorithm, "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / "policy.pt").read_bytes()).hexdigest(),
        "policy_name": policy_name, "policy_joint_order": list(spec.joint_names),
        "actuator_model_version": spec.actuator_model_version,
        "input": "225 raw time-major floats, oldest first; per-frame q_rel12,qd12,omega_body3,gravity_body3,previous_clean_action12,command3",
        "training_input": "210 term-major history + latest 3-D body-frame command; fixed scaling inside model",
        "history_reset": "repeat first measured frame five times",
        "python_max_action_error": (actual - expected).abs().max().item(),
        "physical_robot_tested": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rl-sar", type=Path, default=Path.home() / "rl_sar")
    p.add_argument("--policy-name", choices=("supermimic_velocity", "ppo_velocity"), required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    export(a.checkpoint, a.output or a.rl_sar / "policy/go2" / a.policy_name, a.policy_name)


if __name__ == "__main__":
    main()
