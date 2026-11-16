"""Read-only WM warm-start and physical action-gradient audit; never trains."""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, matvec, so3_log
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2w.checkpoint import initialize_wm_only
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.world_model import tensor_digest


def axes(state):
    rotation = state[1].transpose(-1, -2)
    return torch.cat(
        (matvec(rotation, state[2])[..., :2], matvec(rotation, state[3])[..., 2:3]), -1
    )


def prediction_rmse(predicted, actual):
    errors = {
        "position_m": predicted[0] - actual[0],
        "orientation_rad": so3_log(actual[1].transpose(-1, -2) @ predicted[1]),
        "linear_velocity_m_s": predicted[2] - actual[2],
        "angular_velocity_rad_s": predicted[3] - actual[3],
        "leg_position_rad": predicted[4] - actual[4],
        "leg_velocity_rad_s": predicted[5][:, :12] - actual[5][:, :12],
        "wheel_velocity_rad_s": predicted[5][:, 12:] - actual[5][:, 12:],
    }
    return {key: float(value.square().mean().sqrt()) for key, value in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--file-sha256", required=True)
    parser.add_argument("--tensor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode-conditioning",
        action="store_true",
        help="Audit the new 7D Actor goal and privileged terrain Critic",
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--settle-steps", type=int, nargs="+", default=[0, 5])
    args = parser.parse_args()
    if any(not math.isfinite(e) or e <= 0 for e in args.epsilons) or min(args.settle_steps) < 0:
        parser.error("Require positive finite perturbations and nonnegative settling steps")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        owner = compose("config", overrides=["task=go2w_velocity/omni_v3", "algo.resume=false",
                                             "algo.checkpoint=null", "algo.load_run=null"])
    raw = OmegaConf.to_container(owner.algo, resolve=True)
    field_names = {f.name for f in fields(SuperMimicConfig)}
    launcher_fields = {"checkpoint", "algo", "algo_log_name", "resume", "num_envs", "load_run"}
    if unknown := raw.keys() - field_names - launcher_fields:
        raise ValueError(f"Unknown learner configuration fields: {sorted(unknown)}")
    if any(raw.get(key) for key in ("resume", "checkpoint", "load_run")):
        raise ValueError("Audit requires new models; launcher resume fields must be empty")
    cfg = SuperMimicConfig(
        **{f.name: raw[f.name] for f in fields(SuperMimicConfig) if f.name in raw}
    )
    cfg.buffer_steps = 32
    cfg.horizon = 16
    cfg.compile = False
    cfg.pi_ema = 0
    cfg.critic_terrain_conditioning = args.mode_conditioning
    task = load_config("omni_v3")
    task.commands["velocity"].persistent_wm_coverage = False
    task.commands["velocity"].sampler.domain_weights = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    task.commands["velocity"].fixed_command = (0.0, 0.0, 0.0)
    task.auto_reset = False
    task.commands["velocity"].terrain_mode_conditioning = args.mode_conditioning
    env = make_go2w_env(task, 32)
    try:
        command = env.command_manager.get_term("velocity")
        command.curriculum.enabled = False
        command.curriculum.phase = "fixed"
        command.curriculum.terrain_type.zero_()
        command.curriculum.terrain_level.zero_()
        env.reset()
        learner = SuperMimicLearner(env, cfg, args.output, device="cpu", backend="mujoco")
        initialization = initialize_wm_only(
            learner,
            args.wm,
            expected_file_sha256=args.file_sha256,
            expected_tensor_sha256=args.tensor_sha256,
        )
        asset = torch.load(args.wm, weights_only=True, map_location="cpu", mmap=True)
        reference = WorldModel(
            hidden=cfg.wm_hidden,
            num_joints=16,
            context_dim=0,
            structured=True,
            terrain_dim=77,
            state_layout=LAYOUT,
        )
        reference.load_state_dict(asset["wm"], strict=True)
        zero = torch.zeros(32, 16)
        # Equal inputs must produce identical H1/H8 predictions before any update.
        initial = tuple(item.detach().clone() for item in command.state())
        a, b = initial, initial
        parity = {}
        with torch.no_grad():
            for step in range(8):
                a = learner._wm_step(a, zero, torch.empty(32, 0))
                b = reference.step(
                    b,
                    zero,
                    torch.empty(32, 0),
                    command.mdp.terrain.local_scan(b[0], b[1], sensor_grad=True),
                )
                if step in (0, 7):
                    torch.testing.assert_close(flatten_state(a), flatten_state(b), atol=0, rtol=0)
                    parity[str(step + 1)] = 0.0
        rows = []
        boundaries = []
        for settle in args.settle_steps:
            for epsilon in args.epsilons:
                env.reset()
                for _ in range(settle):
                    result = env.step(zero)
                    if (result.terminated | result.truncated).any():
                        (args.output / "settle_failure.json").write_text(
                            json.dumps(
                                dict(
                                    step=_ + 1,
                                    state=flatten_state(command.state()).tolist(),
                                    terminated=result.terminated.tolist(),
                                    truncated=result.truncated.tolist(),
                                    local_height=command.mdp.local_height(command.state()).tolist(),
                                ),
                                indent=2,
                            )
                        )
                        raise RuntimeError(
                            "Neutral full-range controller terminated during audit settling"
                        )
                start = tuple(item.detach().clone() for item in command.state())
                # All replicas must have the same contact state, not just poses.
                for tensor in start:
                    torch.testing.assert_close(
                        tensor, tensor[:1].expand_as(tensor), atol=1e-6, rtol=1e-5
                    )
                if command.mdp.hard_termination(start).any():
                    raise RuntimeError("Neutral full-range controller fell during audit settling")
                actions = torch.cat((torch.eye(16), -torch.eye(16))) * epsilon
                predicted = start
                for step in range(1, 17):
                    result = env.step(actions)
                    if (result.terminated | result.truncated).any():
                        boundaries.append(
                            dict(
                                settle_steps=settle,
                                epsilon=epsilon,
                                horizon=step,
                                terminated=result.terminated.tolist(),
                                truncated=result.truncated.tolist(),
                            )
                        )
                        break
                    with torch.no_grad():
                        predicted = learner._wm_step(predicted, actions, torch.empty(32, 0))
                    if step not in (1, 8, 16):
                        continue
                    actual_state = command.state()
                    measured = axes(actual_state).detach()
                    physical = ((measured[:16] - measured[16:]) / (2 * epsilon)).T
                    wm_fd_values = axes(predicted).detach()
                    wm_fd = ((wm_fd_values[:16] - wm_fd_values[16:]) / (2 * epsilon)).T
                    single = tuple(item[:1] for item in start)

                    def predict(action):
                        state = single
                        for _ in range(step):
                            state = learner._wm_step(state, action[None], torch.empty(1, 0))
                        return axes(state)[0]

                    jacobian = torch.autograd.functional.jacobian(predict, torch.zeros(16))
                    cosine = torch.nn.functional.cosine_similarity(physical, jacobian, dim=-1)
                    row = dict(
                        settle_steps=settle,
                        epsilon=epsilon,
                        horizon=step,
                        axes=["vx", "vy", "yaw"],
                        physical_jacobian=physical.tolist(),
                        wm_finite_difference=wm_fd.tolist(),
                        wm_autograd=jacobian.tolist(),
                        cosine=cosine.tolist(),
                        physical_norm=physical.norm(dim=-1).tolist(),
                        predicted_norm=jacobian.norm(dim=-1).tolist(),
                        physical_falls=int(command.mdp.hard_termination(actual_state).sum()),
                        prediction_rmse=prediction_rmse(predicted, actual_state),
                        initial_state=flatten_state(tuple(x[:1] for x in start)).tolist(),
                        all_finite=bool(
                            torch.isfinite(jacobian).all() and torch.isfinite(physical).all()
                        ),
                    )
                    rows.append(row)
                    (args.output / "probe_rows.json").write_text(json.dumps(rows, indent=2))
                    print(
                        json.dumps(
                            {k: v for k, v in row.items() if "jacobian" not in k and "wm_" not in k}
                        ),
                        flush=True,
                    )
        # A nonpositive inner product means ascending the WM's velocity gradient
        # locally decreases the corresponding physical velocity in this probe.
        failures = [
            dict(
                settle_steps=r["settle_steps"],
                epsilon=r["epsilon"],
                horizon=r["horizon"],
                axis=axis,
                cosine=r["cosine"][i],
            )
            for r in rows
            for i, axis in enumerate(r["axes"])
            if r["physical_norm"][i] > 1e-3 and r["cosine"][i] <= 0
        ]
        result = dict(
            scope="pre-training WM gate; existing eight-level atlas; flat neutral contact probes only",
            mode_conditioning=args.mode_conditioning,
            actor_input_dim=learner.pi.dec[0].in_features,
            critic_input_dim=learner.critic.norm.mean.numel(),
            initialization=initialization,
            prediction_parity_max_abs=parity,
            source_tensor_sha256=tensor_digest(learner.wm.state_dict()),
            probes=rows,
            episode_boundaries=boundaries,
            gradient_direction_gate_passed=not failures,
            direction_failures=failures,
            limitation="Failure is local evidence; does not establish every policy trajectory has wrong gradients.",
        )
        (args.output / "audit.json").write_text(json.dumps(result, indent=2))
        return 2 if failures else 0
    except BaseException as exc:
        (args.output / "crash.json").write_text(
            json.dumps(dict(error=type(exc).__name__, message=str(exc)), indent=2)
        )
        raise
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
