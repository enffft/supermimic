"""Brief physical-target distillation from an existing Go2W policy, then stop.

This only initializes the proprioceptive Actor. World Model weights, statistics
and optimizer remain in the continually trained asset and are not fitted here.
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT, POLICY_SPEC, leg_position_target
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env


def remap_offset_action(action, default, limits):
    result = action.clone()
    span = torch.where(action[..., :12] >= 0, limits[:, 1] - default, default - limits[:, 0])
    result[..., :12] = .25 * action[..., :12] / span
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=800)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    started = time.monotonic()
    source = torch.load(args.teacher, map_location="cpu", weights_only=True, mmap=True)
    control = source["task_contract"]["actions"]["mixed"]["control"]
    if control.get("full_leg_target_range", False) or control["leg_position_scale"] != .25:
        raise ValueError("Bootstrap teacher must use the audited .25-rad offset mapping")
    teacher = Policy(50, 3, hist=5, num_joints=16, hidden=source["args"]["policy_hidden"],
                     goal_mode="direct", direct_spec=POLICY_SPEC).cuda().eval()
    teacher.load_state_dict(source["pi"])
    student = deepcopy(teacher).train()
    cfg = load_config("terrain")
    target_cfg = load_config("omni_v3")
    cfg.commands["velocity"].sampler = deepcopy(target_cfg.commands["velocity"].sampler)
    cfg.commands["velocity"].terrain_phase = "skill"
    env = make_go2w_env(cfg, args.envs)
    target_env = None
    try:
        # Only build the target to capture its exact task identity and control.
        target_env = make_go2w_env(target_cfg, 1)
        target_env.reset()
        target_contract = target_env.supermimic_contract
        mdp = target_env.supermimic_mdp.to("cuda:0")
        default, limits = mdp.default_q, mdp.limits
        env.reset()
        history = torch.zeros(args.envs, 5, 50, device="cuda")
        inputs, goals, labels = [], [], []
        falls = 0
        with torch.no_grad():
            for _ in range(args.steps):
                info = {key: value.cuda() for key, value in env.current_info().items()}
                state = unflatten_state(info["s"], LAYOUT)
                obs = mdp.obs(state, info["a_prev"], info["xi"], info["f"])
                history = torch.cat((history[:, 1:], obs[:, None]), 1)
                history = torch.where((info["ep_step"] == 0)[:, None, None], obs[:, None].expand_as(history), history)
                action = teacher(history, info["g"])
                mapped = history.clone()
                mapped[..., -16:] = remap_offset_action(history[..., -16:], default, limits)
                inputs.append(mapped.cpu())
                goals.append(info["g"].cpu())
                labels.append(remap_offset_action(action, default, limits).cpu())
                transition = env.step(action.cpu()).info[SUPERMIMIC_TRANSITION_KEY]
                falls += int(transition["term"].sum())
        x, g, y = (torch.cat(values).cuda() for values in (inputs, goals, labels))
        # Start near the smaller output coordinates before the short regression.
        with torch.no_grad():
            representative = y[:, :12].mean(0)
            span = torch.where(representative >= 0, limits[:, 1] - default, default - limits[:, 0])
            scale = .25 / span
            student.dec[-1].weight[:12].mul_(scale[:, None])
            student.dec[-1].bias[:12].mul_(scale)
        order = torch.randperm(len(x), device="cuda")
        split = int(.9 * len(x))
        train_ids, valid_ids = order[:split], order[split:]
        optimizer = torch.optim.Adam(student.parameters(), lr=2e-4)
        target = leg_position_target(y, mdp.control, default, limits)
        for step in range(args.updates):
            ids = train_ids[torch.randint(len(train_ids), (1024,), device="cuda")]
            predicted = student(x[ids], g[ids])
            joint_error = (leg_position_target(predicted, mdp.control, default, limits) - target[ids]) / .25
            loss = joint_error.square().mean() + (predicted[:, 12:] - y[ids, 12:]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            if (step + 1) % 200 == 0:
                print(json.dumps(dict(update=step + 1, loss=float(loss))), flush=True)
        with torch.no_grad():
            pred = student(x[valid_ids], g[valid_ids])
            error = leg_position_target(pred, mdp.control, default, limits) - target[valid_ids]
            report = dict(teacher=str(args.teacher.resolve()), samples=len(x), updates=args.updates,
                          validation_scope="random held-out transitions from the same short collection",
                          leg_target_RMSE_rad=float(error.square().mean().sqrt()),
                          leg_target_max_error_rad=float(error.abs().max()),
                          wheel_action_RMSE=float((pred[:, 12:] - y[valid_ids, 12:]).square().mean().sqrt()),
                          collection_falls=falls, elapsed_seconds=time.monotonic() - started,
                          use="Actor initialization only; subsequent training is standard SuperMimic")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pi = {key: value.cpu() for key, value in student.state_dict().items()}
        torch.save(dict(pi=pi, task_contract=target_contract, provenance=report), args.output)
        # Evaluation-only checkpoint; does not replace the WM or its optimizer.
        asset = torch.load(args.wm, map_location="cpu", weights_only=True, mmap=True)
        torch.save(dict(pi=pi, wm=asset["wm"], args=source["args"], iteration=0,
                        task_contract=target_contract), args.output.with_name("bootstrap_evaluation.pt"))
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()
        env.backend.close()
        if target_env is not None:
            target_env.close()
            target_env.backend.close()


if __name__ == "__main__":
    main()
