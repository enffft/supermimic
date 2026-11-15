"""Compare WMs on the same held-out physical transitions, including action remapping."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import (
    LAYOUT,
    MixedControlCfg,
    leg_position_target,
    soft_limits,
)
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    cfg = load_config(args.profile)
    env = make_go2w_env(cfg, 1)
    try:
        if env.supermimic_contract != saved["task_contract"]:
            raise ValueError("WM evaluation profile differs from checkpoint")
        source_contract, target_contract = dataset["task_contract"], saved["task_contract"]
        for key in ("model_assets_sha256", "terrain_fingerprint", "state_layout", "joints", "sim_dt", "ctrl_dt"):
            if source_contract[key] != target_contract[key]:
                raise ValueError(f"Holdout physical protocol differs: {key}")
        source_control = MixedControlCfg(**source_contract["actions"]["mixed"]["control"])
        target_control = cfg.actions["mixed"].control
        a, b = asdict(source_control), asdict(target_control)
        for item in (a, b):
            for key in ("full_leg_target_range", "leg_limit_fraction", "leg_position_scale"):
                item.pop(key)
        if a != b:
            raise ValueError("Holdout remapping requires unchanged controller gains, effort and wheel velocity scale")
        command = env.command_manager.get_term("velocity")
        default = command.mdp.default_q
        source_limits = soft_limits(fraction=source_control.leg_limit_fraction)
        target_limits = command.mdp.limits
        windows = dataset["windows"]
        target = leg_position_target(windows["a"], source_control, default, source_limits)
        if ((target < target_limits[:, 0] - 1e-6) | (target > target_limits[:, 1] + 1e-6)).any():
            raise ValueError("Holdout contains targets outside evaluated controller range")
        action = windows["a"].clone()
        if target_control.full_leg_target_range:
            spans = torch.where(target >= default, target_limits[:, 1] - default, default - target_limits[:, 0])
            action[..., :12] = (target - default) / spans
        else:
            action[..., :12] = (target - default) / target_control.leg_position_scale
        reconstructed = leg_position_target(action, target_control, default, target_limits)
        torch.testing.assert_close(target, reconstructed, atol=1e-6, rtol=1e-5)
        wm = WorldModel(hidden=saved["args"]["wm_hidden"], num_joints=16, context_dim=0,
                        structured=True, terrain_dim=77, state_layout=LAYOUT).cuda().eval()
        wm.load_state_dict(saved["wm"])
        field = command.mdp.terrain.to("cuda")
        state = unflatten_state(windows["s"][:, 0].cuda(), LAYOUT)
        n = len(windows["s"])
        valid = windows["live"][:, 0].clone()
        scopes = dataset["domain"] * 8 + dataset["level"]
        out = {}
        for step in range(16):
            state = wm.step(state, action[:, step].cuda(), torch.empty(n, 0, device="cuda"),
                            field.local_scan(state[0], state[1], sensor_grad=True))
            horizon = step + 1
            if horizon in (1, 8, 16):
                truth = unflatten_state(windows["s1"][:, step].cuda(), LAYOUT)
                errors = {"position": (state[0] - truth[0]).square().mean(-1).cpu(),
                          "linear_velocity": (state[2] - truth[2]).square().mean(-1).cpu(),
                          "angular_velocity": (state[3] - truth[3]).square().mean(-1).cpu(),
                          "leg_position": (state[4] - truth[4]).square().mean(-1).cpu(),
                          "leg_velocity": (state[5][:, :12] - truth[5][:, :12]).square().mean(-1).cpu(),
                          "wheel_velocity": (state[5][:, 12:] - truth[5][:, 12:]).square().mean(-1).cpu()}
                rows = []
                for domain in range(6):
                    for level in range(8):
                        selected = valid & (scopes == domain * 8 + level)
                        num = int(selected.sum())
                        rows.append(dict(domain=domain, level=level, valid_windows=num,
                                         RMSE={key: float(value[selected].mean().sqrt()) for key, value in errors.items()} if num else None))
                out[str(horizon)] = rows
            valid &= ~windows["done"][:, step].bool()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), dataset=str(args.dataset),
                                              action_remapping="Identical physical PD targets verified; wheel actions unchanged",
                                              horizons=out), indent=2))
        print(f"Saved {args.output}")
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
