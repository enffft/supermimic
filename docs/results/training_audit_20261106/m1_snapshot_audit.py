"""CPU-only frozen Actor checkpoint audit on identical synthetic inputs."""
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from supermimic.policy.nets import Policy, PolicyCfg

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "logs/m1_reward/v2_lm50_2048_3p5h_seed2026"
OUT = Path(__file__).parent


def main():
    torch.set_num_threads(1)
    rows = []
    for path in sorted(RUN.glob("ckpt_wall_*.pt")) + [RUN / "ckpt_latest.pt"]:
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        policy = Policy(PolicyCfg(**saved["config"]["policy"])).eval()
        policy.load_state_dict(saved["policy"])
        h, nj = policy.cfg.history, policy.cfg.num_joints
        # q_rel=0, qd=0, omega=0, gravity=(0,0,-1), previous action=0.
        obs = torch.cat([torch.zeros(4, h * nj), torch.zeros(4, h * nj),
                         torch.zeros(4, h * 3), torch.tensor([[0., 0., -1.]]).repeat(4, h),
                         torch.zeros(4, h * nj)], -1)
        goal = torch.tensor([[.12, 0, 0, 0, 0, 0, 1, 0], [-.12, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1.5, 0, .3, 0, 1], [0, 0, 0, -1.5, 0, .3, 0, 1]])
        x = torch.cat([obs * policy.decoder.scale, goal * policy.decoder.goal_scale], -1)
        with torch.no_grad():
            hidden = policy.decoder.mlp.net[:-1](x)
            logits = policy.decoder.mlp.net[-1](hidden)
            action = logits.tanh()
            row = {"checkpoint": path.name, "iteration": saved["iteration"],
                   "hours": saved["training_wall_seconds"] / 3600,
                   "actor_updates": saved["actor_updates"],
                   "fixed_input_mean_abs_action": float(action.abs().mean()),
                   "fixed_input_saturated_fraction_0p95": float((action.abs() > .95).float().mean()),
                   "fixed_input_mean_abs_logit": float(logits.abs().mean()),
                   "fixed_input_mean_tanh_slope": float((1 - action.square()).mean()),
                   "fixed_input_L_command_action_difference": float((action[0] - action[1]).abs().mean()),
                   "fixed_input_M_target_action_difference": float((action[2] - action[3]).abs().mean()),
                   "last_hidden_abs_mean": float(hidden.abs().mean()),
                   "last_weight_frobenius": float(policy.decoder.mlp.net[-1].weight.norm()),
                   "last_bias_abs_mean": float(policy.decoder.mlp.net[-1].bias.abs().mean()),
                   "policy_weights_finite": all(bool(p.isfinite().all()) for p in policy.parameters())}
        rows.append(row)
        print(path.name, json.dumps(row), flush=True)
        del policy, saved
    (OUT / "m1_fixed_input_snapshots.json").write_text(json.dumps({
        "method": "42 five-minute snapshots plus final; identical synthetic nominal proprioception and four L/M goals; CPU only; no updates",
        "limitations": "Synthetic frozen inputs isolate network drift; not historical Replay, not a physical rollout or task success evaluation",
        "snapshots": rows}, indent=2) + "\n")
    reader = EventAccumulator(str(RUN / "tb"), size_guidance={"scalars": 0}).Reload()
    tags = [t for t in reader.Tags()["scalars"] if t.startswith(("policy/", "wm/nse", "m1/real/", "m1/reward_real/", "time/hours", "schedule/actor_updates"))]
    curves = {tag: [{"step": r.step, "wall_time": r.wall_time, "value": r.value}
                    for r in reader.Scalars(tag)] for tag in tags}
    (OUT / "m1_selected_training_scalars.json").write_text(json.dumps(curves, indent=2) + "\n")


if __name__ == "__main__":
    main()
