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
from unisupermimic.tasks.velocity.go2w.contract import JOINTS, LAYOUT
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


def contact_signature(env):
    """Read actual compiled contacts at the current physics state, never Actor inputs."""
    import mujoco

    model = env.get_playback_model(0)
    data = mujoco.MjData(model)
    views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
    data.qpos[:] = views["qpos"][0].numpy()
    data.qvel[:] = views["qvel"][0].numpy()
    mujoco.mj_forward(model, data)
    wheels = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{leg}_wheel_collision")
        for leg in ("FL", "FR", "RL", "RR")
    ]
    counts = [sum(gid in contact.geom for contact in data.contact) for gid in wheels]
    return dict(
        wheel_contacts=counts,
        total_contacts=data.ncon,
        category="wheel_contact" if any(counts) else "no_wheel_contact",
    )


def evaluate_direction_gate(rows, boundaries, horizon, expected_cases):
    """Unused longer horizons are warnings. Missing required probes fail closed."""
    required = sorted({1, horizon})
    failures, warnings = [], []
    for row in rows:
        for i, axis in enumerate(row["axes"]):
            if not row["all_finite"] or (row["physical_norm"][i] > 1e-3 and row["cosine"][i] <= 0):
                item = dict(
                    case=row["case"],
                    epsilon=row["epsilon"],
                    horizon=row["horizon"],
                    axis=axis,
                    cosine=row["cosine"][i],
                )
                (failures if row["horizon"] in required else warnings).append(item)
    observed = {(r["case"], r["epsilon"], r["horizon"]) for r in rows}
    missing = [
        dict(case=c, epsilon=e, horizon=h)
        for c, e in expected_cases
        for h in required
        if (c, e, h) not in observed
    ]
    contacts = sorted({r["contact"]["category"] for r in rows if r["horizon"] in required})
    sampled = any(r["action_source"] == "actor_plus_collection_noise" for r in rows)
    coverage = sampled and len(contacts) >= 2
    return dict(
        actor_horizon=horizon,
        required_horizons=required,
        direction_failures=failures,
        diagnostic_warnings=warnings,
        missing_required_probes=missing,
        contact_categories=contacts,
        sampled_action_coverage=sampled,
        coverage_passed=coverage,
        episode_boundaries=boundaries,
        gradient_direction_gate_passed=not failures and not missing and coverage,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--file-sha256", required=True)
    parser.add_argument("--tensor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="train")
    parser.add_argument("--mode-conditioning", action="store_true")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002, 0.005])
    parser.add_argument("--settle-steps", type=int, nargs="+", default=[0, 5, 10])
    args = parser.parse_args()
    if any(not math.isfinite(e) or e <= 0 for e in args.epsilons) or min(args.settle_steps) < 0:
        parser.error("Require finite positive perturbations and nonnegative settling steps")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        owner = compose(
            "config",
            overrides=[
                f"task=go2w_velocity/{args.profile}",
                "algo.resume=false",
                "algo.checkpoint=null",
                "algo.load_run=null",
            ],
        )
    raw = OmegaConf.to_container(owner.algo, resolve=True)
    names = {f.name for f in fields(SuperMimicConfig)}
    allowed = {"checkpoint", "algo", "algo_log_name", "resume", "num_envs", "load_run"}
    if unknown := raw.keys() - names - allowed:
        raise ValueError(f"Unknown learner fields: {sorted(unknown)}")
    cfg = SuperMimicConfig(**{k: v for k, v in raw.items() if k in names})
    configured_horizon = cfg.horizon
    cfg.buffer_steps = 32
    cfg.compile = False
    cfg.pi_ema = 0
    task = load_config(args.profile)
    if args.mode_conditioning:
        task.commands["velocity"].terrain_mode_conditioning = True
        cfg.critic_terrain_conditioning = True
    task.commands["velocity"].persistent_wm_coverage = False
    task.commands["velocity"].sampler.domain_weights = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    task.commands["velocity"].fixed_command = (0.3, 0.0, 0.0)
    task.auto_reset = False
    env = make_go2w_env(task, 32)
    try:
        command = env.command_manager.get_term("velocity")
        command.curriculum.enabled = False
        command.curriculum.phase = "fixed"
        command.curriculum.terrain_type.zero_()
        command.curriculum.terrain_level.zero_()
        command.command_domain.zero_()
        if hasattr(command.curriculum, "coverage_level"):
            command.curriculum.coverage_level.fill_(-1)
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
        initial = tuple(t.detach().clone() for t in command.state())
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
        native_pre = {}
        original_apply_action = env.apply_action

        def capture_native_pre(actions, state):
            views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
            native_pre.update({k: v[:1].clone() for k, v in views.items()})
            return original_apply_action(actions, state)

        env.apply_action = capture_native_pre
        import copy

        policy_norms = (
            copy.deepcopy(learner.pi.obs_norm.state_dict()),
            copy.deepcopy(learner.pi.goal_norm.state_dict()),
        )
        rows, boundaries, expected = [], [], []
        horizons = sorted({1, configured_horizon, 16})
        for settle in args.settle_steps:
            for sample_seed in [None, 2026, 2027]:
                case = f"initial_{settle}_action_{sample_seed}"
                for epsilon in args.epsilons:
                    expected.append((case, epsilon))
                    env.reset()
                    env.set_collection_exploration(0.0)
                    if sample_seed is None:
                        terminated = False
                        for _ in range(settle):
                            result = env.step(zero)
                            if (result.terminated | result.truncated).any():
                                terminated = True
                                break
                        if terminated:
                            boundaries.append(dict(case=case, epsilon=epsilon, horizon=0))
                            continue
                        base = torch.zeros(16)
                        action_source = "neutral_control"
                    else:
                        # Use the actual collector, including Actor normalizers, five-frame
                        # histories, executed exploration and policy-generated initial states.
                        torch.manual_seed(sample_seed)
                        learner.pi.obs_norm.load_state_dict(policy_norms[0])
                        learner.pi.goal_norm.load_state_dict(policy_norms[1])
                        learner.obs_hist.zero_()
                        env.set_autoreset(True)
                        learner.collect(settle + 1)
                        env.set_autoreset(False)
                        base = env.velocity_executed_action[0].clone()
                        # Restore the PRE-action state of the actual sampled transition.
                        # The 32 finite-difference replicas must start from identical physics.
                        env.backend.set_state_tensor(
                            torch.arange(32),
                            native_pre["qpos"].expand(32, -1).contiguous(),
                            native_pre["qvel"].expand(32, -1).contiguous(),
                        )
                        command.a_prev[:] = env.velocity_pre_action["previous_action"][:1]
                        command.ep_step[:] = command.ep_step[:1].clone() - 1
                        command.f[:] = command.f[:1].clone() - 1
                        # collect() updates WM normalizers: undo them for an original-asset audit.
                        learner.wm.load_state_dict(asset["wm"], strict=True)
                        env.set_collection_exploration(0.0)
                        action_source = "actor_plus_collection_noise"
                    start = tuple(t.detach().clone() for t in command.state())
                    for t in start:
                        torch.testing.assert_close(t, t[:1].expand_as(t), atol=1e-6, rtol=1e-5)
                    contact = contact_signature(env)
                    actions = base[None] + torch.cat((torch.eye(16), -torch.eye(16))) * epsilon
                    predicted = start
                    contact_trace = []
                    for step in range(1, max(horizons) + 1):
                        result = env.step(actions)
                        contact_trace.append(contact_signature(env))
                        if (result.terminated | result.truncated).any():
                            boundaries.append(
                                dict(
                                    case=case,
                                    epsilon=epsilon,
                                    horizon=step,
                                    terminated=result.terminated.tolist(),
                                    truncated=result.truncated.tolist(),
                                )
                            )
                            break
                        with torch.no_grad():
                            predicted = learner._wm_step(predicted, actions, torch.empty(32, 0))
                        if step not in horizons:
                            continue
                        truth = command.state()
                        actual = axes(truth).detach()
                        physical = ((actual[:16] - actual[16:]) / (2 * epsilon)).T
                        fd = axes(predicted).detach()
                        wm_fd = ((fd[:16] - fd[16:]) / (2 * epsilon)).T
                        single = tuple(t[:1] for t in start)

                        def predict(action):
                            state = single
                            for _ in range(step):
                                state = learner._wm_step(state, action[None], torch.empty(1, 0))
                            return axes(state)[0]

                        jac = torch.autograd.functional.jacobian(predict, base)
                        cosine = torch.nn.functional.cosine_similarity(physical, jac, dim=-1)
                        row = dict(
                            case=case,
                            settle_steps=settle,
                            epsilon=epsilon,
                            horizon=step,
                            action_source=action_source,
                            sample_seed=sample_seed,
                            base_action=base.tolist(),
                            action_min=float(actions.min()),
                            action_max=float(actions.max()),
                            contact=contact,
                            contact_trace=list(contact_trace),
                            axes=["vx", "vy", "yaw"],
                            physical_jacobian=physical.tolist(),
                            wm_autograd=jac.tolist(),
                            wm_finite_difference=wm_fd.tolist(),
                            cosine=cosine.tolist(),
                            physical_norm=physical.norm(dim=-1).tolist(),
                            predicted_norm=jac.norm(dim=-1).tolist(),
                            prediction_rmse=prediction_rmse(predicted, truth),
                            initial_state=flatten_state(single).tolist(),
                            all_finite=bool(
                                torch.isfinite(jac).all() and torch.isfinite(physical).all()
                            ),
                        )
                        rows.append(row)
                        (args.output / "probe_rows.json").write_text(json.dumps(rows, indent=2))
                        print(
                            json.dumps(
                                dict(
                                    case=case,
                                    epsilon=epsilon,
                                    horizon=step,
                                    cosine=row["cosine"],
                                    contact=contact,
                                )
                            ),
                            flush=True,
                        )
        result = evaluate_direction_gate(rows, boundaries, configured_horizon, expected)
        result.update(
            scope="multi-state multi-scale real-physics direction gate; actual collector actions and pre-action states",
            sampling_protocol="collect() normalizers/history/exploration; restore full actual pre-action state including wheel angles",
            initialization=initialization,
            actor_input_dim=learner.pi.dec[0].in_features,
            critic_input_dim=learner.critic.norm.mean.numel(),
            probes=rows,
            prediction_parity_max_abs=parity,
            source_tensor_sha256=tensor_digest(learner.wm.state_dict()),
            resolved_algo=raw,
        )
        (args.output / "audit.json").write_text(json.dumps(result, indent=2))
        return 0 if result["gradient_direction_gate_passed"] else 2
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
