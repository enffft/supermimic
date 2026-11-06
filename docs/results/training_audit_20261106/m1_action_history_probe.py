"""CPU frozen-physics counterfactual: feed back ONLY previous clean actions.

Not a physical rollout or a historical Replay reconstruction. No optimization.
"""
import json
from pathlib import Path

import torch
from supermimic.policy.nets import Policy, PolicyCfg

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "logs/m1_reward/v2_lm50_2048_3p5h_seed2026"


def main():
    torch.set_num_threads(1)
    results = []
    for name in ("ckpt_wall_015m.pt", "ckpt_wall_060m.pt", "ckpt_wall_090m.pt",
                 "ckpt_wall_100m.pt", "ckpt_wall_105m.pt", "ckpt_wall_120m.pt", "ckpt_latest.pt"):
        saved = torch.load(RUN / name, map_location="cpu", weights_only=False, mmap=True)
        policy = Policy(PolicyCfg(**saved["config"]["policy"])).eval()
        policy.load_state_dict(saved["policy"])
        h, nj = policy.cfg.history, policy.cfg.num_joints
        fixed = torch.cat([torch.zeros(4, h * nj), torch.zeros(4, h * nj), torch.zeros(4, h * 3),
                           torch.tensor([[0., 0., -1.]]).repeat(4, h)], -1)
        goal = torch.tensor([[.12, 0, 0, 0, 0, 0, 1, 0], [-.12, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1.5, 0, .3, 0, 1], [0, 0, 0, -1.5, 0, .3, 0, 1]])
        a_hist = torch.zeros(4, h, nj)
        rows = []
        with torch.no_grad():
            for step in range(64):
                action = policy(torch.cat([fixed, a_hist.flatten(1)], -1), goal)
                if step in (0, 1, 4, 8, 16, 32, 63):
                    rows.append({"step": step, "abs_action_mean": float(action.abs().mean()),
                                 "saturation_over_0p95": float((action.abs() > .95).float().mean()),
                                 "tanh_slope_mean": float((1 - action.square()).mean()),
                                 "L_command_difference": float((action[0] - action[1]).abs().mean()),
                                 "M_goal_difference": float((action[2] - action[3]).abs().mean())})
                a_hist = torch.cat([a_hist[:, 1:], action[:, None]], 1)
        results.append({"checkpoint": name, "rows": rows})
        print(name, rows[-1], flush=True)
    Path(__file__).with_name("m1_action_history_feedback.json").write_text(json.dumps({
        "method": "identical nominal synthetic physical observations held fixed, previous-action history alone evolves for 64 calls; four L/M goals; frozen Actor; CPU; no optimization",
        "limitations": "Artificial controlled input test, not physical simulation or original training causality",
        "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
