"""Collect a short replay and diagnose WM termination/survival/BPTT for Go2 or M1.

This is a small acceptance diagnostic, not the normal training entry point: it performs a
few configured World Model updates, computes Actor gradients without stepping the Actor
optimizer, and writes a JSON report. Example:

  PYTHONPATH=src python -m supermimic.scripts.diagnose_multirobot --robot m1_piper
  PYTHONPATH=src python -m supermimic.scripts.diagnose_multirobot --robot go2 --action-noise 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from supermimic.config import TrainConfig
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.rollout import imagine
from supermimic.training.loop import Trainer
from supermimic.world_model.heads import ReferenceFrame


def _reference_steps(trainer: Trainer, window):
    if trainer.lib is None:
        return [None] * window.a_exec.shape[0]
    k, n = window.frame.shape
    ref = trainer.lib.ref(window.frame.reshape(-1))
    result = []
    for j in range(k):
        sl = slice(j * n, (j + 1) * n)
        result.append(
            ReferenceFrame(
                None, None, None, None,
                ref.root_pos_w[sl], ref.root_quat_w[sl],
            )
        )
    return result


@torch.no_grad()
def _termination_report(trainer: Trainer, sample_windows: int) -> dict[str, object]:
    wm = trainer.wm
    window = trainer.buffer.sample_windows(sample_windows, 8)
    refs = _reference_steps(trainer, window)
    state = window.states[0]
    probability, labels, nse = [], [], []
    for j in range(8):
        state = wm.step(state, window.a_exec[j], window.xi)
        logits = wm.termination_logits(state, refs[j])
        probability.append(torch.sigmoid(logits))
        labels.append(window.done[j])
        nse.append(wm.spec.normalized_sq_error(state, window.states[j + 1]))
    p = torch.stack(probability).reshape(-1)
    y = torch.stack(labels).reshape(-1).bool()
    nse_t = torch.stack(nse)
    bce = F.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), y.float())
    qs = torch.quantile(p, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], device=p.device))
    return {
        "windows": int(sample_windows),
        "labels_terminal": int(y.sum()),
        "labels_nonterminal": int((~y).sum()),
        "pred_terminal_at_0_5": int((p >= 0.5).sum()),
        "accuracy_at_0_5": float(((p >= 0.5) == y).float().mean()),
        "bce": float(bce),
        "probability_quantiles_0_50_90_99_100": [float(x) for x in qs],
        "nse_1step": float(nse_t[0].mean()),
        "nse_8step": float(nse_t[-1].mean()),
        "label_status": "both_classes_present" if y.any() and (~y).any() else "one_class_only",
    }


def _imagination_report(trainer: Trainer, start, horizon: int) -> dict[str, float]:
    cfg = trainer.cfg.policy_trainer
    wm, policy, critic = trainer.wm, trainer.policy, trainer.critic
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    policy.zero_grad(set_to_none=True)
    try:
        rollout = imagine(
            wm, policy, trainer.task, start, horizon,
            future_frames=cfg.future_frames, obs_noise=False,
            step_fn=trainer.policy_trainer.step_fn, anchor_obs=cfg.anchor_obs,
        )
        objective = policy_objective(rollout, terminal_values=None, gamma=cfg.gamma)
        (-objective / horizon).backward()
        grad_norm = torch.sqrt(
            sum((p.grad.detach().square().sum() for p in policy.parameters() if p.grad is not None),
                torch.zeros((), device=trainer.device))
        )
        return {
            "horizon": horizon,
            "actor_grad_norm": float(grad_norm.detach()),
            "imag_return_undiscounted_soft_survival": float((rollout.survival[:-1] * rollout.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(rollout.rewards.mean().detach()),
            "survival_H_mean": float(rollout.survival[-1].mean().detach()),
            "survival_H_min": float(rollout.survival[-1].min().detach()),
            "termination_probability_mean": float(rollout.dhat[1:].mean().detach()),
        }
    finally:
        policy.zero_grad(set_to_none=True)
        wm.requires_grad_(True)
        critic.requires_grad_(True)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this MuJoCo-Warp diagnostic requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = TrainConfig()
    cfg.run_name = f"diag_{args.robot}_{int(time.time())}"
    cfg.log_root = args.output.parent / "runs"
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.env.robot_name = args.robot
    cfg.env.task_name = "velocity_command" if args.robot == "go2" else "motion_tracking"
    cfg.env.num_envs = args.num_envs
    cfg.env.replay_capacity = max(args.steps + 32, 256)
    cfg.env.episode_length_s = 20.0
    cfg.env.obs_noise = False
    cfg.env.reset_noise = False
    cfg.env.domain_randomization = args.domain_randomization
    cfg.env.action_noise_std = 0.0
    cfg.env.sampling_mode = "start"
    cfg.env.velocity_debug_objective = args.robot == "go2"  # diagnostic-only reward; no policy optimizer step
    if args.robot == "m1_piper":
        from supermimic.training.loop import M1_FIRST_PHASE_CLIPS, M1_FIRST_PHASE_MANIFEST

        cfg.data.clips = list(M1_FIRST_PHASE_CLIPS)
        cfg.data.manifest = M1_FIRST_PHASE_MANIFEST

    # Small test networks keep the run brief; the diagnostic measures the implemented
    # losses/gradient paths, not production-capacity throughput.
    cfg.world_model.hidden = (128, 128)
    cfg.world_model.term_hidden = (32, 32)
    cfg.policy.encoder_hidden = (128, 64)
    cfg.policy.decoder_hidden = (128, 64)
    cfg.critic.hidden = (64, 32)
    cfg.wm_trainer.window = 8
    cfg.wm_trainer.batch_size = args.window_batch
    cfg.wm_trainer.batches_per_update = 1
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.history = cfg.env.history_length
    cfg.policy_trainer.horizon = 32
    cfg.policy_trainer.batch_size = args.imag_batch
    cfg.policy_trainer.obs_noise = False
    cfg.policy_trainer.use_critic = False
    cfg.policy_trainer.compile = None

    trainer = Trainer(cfg)
    env = trainer.env
    try:
        obs, _ = env.reset()
        env_returns = torch.zeros(env.num_envs, device=trainer.device)
        real_episode_returns: list[float] = []
        real_episode_normalized_returns: list[float] = []
        real_episode_lengths: list[int] = []
        lengths = torch.zeros(env.num_envs, device=trainer.device)
        for _ in range(args.steps):
            with torch.no_grad():
                action = trainer.act(obs)
                if args.action_noise:
                    action = action + args.action_noise * torch.randn_like(action)
                action = action.clamp(-1.0, 1.0)
            obs, reward, terminated, timeout, _ = env.step(action)
            env_returns += reward
            lengths += 1
            done = terminated | timeout
            if done.any():
                ids = done.nonzero().flatten()
                finished_returns = env_returns[ids].detach().cpu().tolist()
                finished_lengths = lengths[ids].to(torch.int64).cpu().tolist()
                real_episode_returns.extend(finished_returns)
                real_episode_lengths.extend(finished_lengths)
                if args.robot == "m1_piper":
                    real_episode_normalized_returns.extend(
                        float(ret / (trainer.max_step_reward * max(length, 1)))
                        for ret, length in zip(finished_returns, finished_lengths, strict=True)
                    )
                env_returns[ids] = 0.0
                lengths[ids] = 0.0

        if trainer.buffer.size == 0 or int(trainer.buffer.window_mask(8).sum()) == 0:
            raise RuntimeError(
                f"replay buffer has no 8-step windows after {args.steps} collection steps; "
                "increase --steps or --num-envs"
            )
        before = _termination_report(trainer, args.diagnostic_windows)
        wm_updates = []
        for _ in range(args.wm_updates):
            wm_updates.append(trainer.wm_trainer.update(trainer.buffer))
        after = _termination_report(trainer, args.diagnostic_windows)
        starts = trainer.buffer.sample_starts(args.imag_batch, cfg.env.history_length)
        imagination = [_imagination_report(trainer, starts, h) for h in (8, 32)]
        result: dict[str, object] = {
            "git_commit": _git_commit(),
            "robot": args.robot,
            "seed": args.seed,
            "device": args.device,
            "envs": args.num_envs,
            "collection_control_steps": args.steps,
            "replay_rows": trainer.buffer.size,
            "sampleable_8step_windows": int(trainer.buffer.window_mask(8).sum()),
            "world_model_window": 8,
            "world_model_updates": wm_updates,
            "termination_before_warmup": before,
            "termination_after_warmup": after,
            "imagination": imagination,
            "real_simulation_episode_return_mean": (
                sum(real_episode_returns) / len(real_episode_returns) if real_episode_returns else None
            ),
            "real_simulation_episode_normalized_return_mean": (
                sum(real_episode_normalized_returns) / len(real_episode_normalized_returns)
                if real_episode_normalized_returns else None
            ),
            "real_simulation_episode_count": len(real_episode_returns),
            "real_simulation_episode_length_mean": (
                sum(real_episode_lengths) / len(real_episode_lengths) if real_episode_lengths else None
            ),
            "real_return_semantics": (
                "M1 physical tracking reward" if args.robot == "m1_piper"
                else "zero-weight collection placeholder; not a Go2 task return"
            ),
            "policy_optimizer_steps": 0,
            "go2_reward_note": "diagnostic velocity reward is enabled only to probe BPTT; it is not the selected training reward" if args.robot == "go2" else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    finally:
        if trainer.writer is not None:
            trainer.writer.flush()
            trainer.writer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("go2", "m1_piper"), default="m1_piper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--wm-updates", type=int, default=8)
    parser.add_argument("--window-batch", type=int, default=64)
    parser.add_argument("--diagnostic-windows", type=int, default=64)
    parser.add_argument("--imag-batch", type=int, default=32)
    parser.add_argument("--action-noise", type=float, default=0.05)
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=Path("logs/multirobot_diagnostics/latest.json"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
