"""Collect a short replay and diagnose WM termination/survival/BPTT for Go2 or M1.

This is a small acceptance diagnostic, not the normal training entry point: it performs a
few configured World Model updates, computes Actor gradients without stepping the Actor
optimizer, and writes a JSON report. Example:

  PYTHONPATH=src python -m supermimic.scripts.diagnose_multirobot --robot m1_piper
  PYTHONPATH=src python -m supermimic.scripts.diagnose_multirobot --robot go2 --num-envs 16 --action-noise 0
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
def _replay_termination_coverage(trainer: Trainer, window: int = 8) -> dict[str, float | int]:
    """Measure how often uniform valid windows can expose a positive done label."""
    buffer = trainer.buffer
    valid_windows = buffer.window_mask(window)
    positive_offsets = [buffer._rolled(buffer.done > 0.5, j) for j in range(window)]
    labels_per_window = torch.stack(positive_offsets).sum(0)
    sampleable_positive = valid_windows & (labels_per_window > 0)
    num_windows = int(valid_windows.sum())
    positive_window_count = int(sampleable_positive.sum())
    total_labels = num_windows * window
    positive_labels = int((labels_per_window * valid_windows).sum())
    return {
        "recorded_terminal_transition_rows": int((buffer.valid & ~buffer.terminal & (buffer.done > 0.5)).sum()),
        "recorded_terminal_state_rows": int((buffer.valid & buffer.terminal & (buffer.done > 0.5)).sum()),
        "recorded_timeout_terminal_state_rows": int((buffer.valid & buffer.terminal & (buffer.done <= 0.5)).sum()),
        "sampleable_windows": num_windows,
        "sampleable_windows_with_positive_label": positive_window_count,
        "sampleable_positive_window_fraction": positive_window_count / max(num_windows, 1),
        "positive_transition_labels_in_sampleable_windows": positive_labels,
        "positive_label_fraction_under_uniform_window_sampling": positive_labels / max(total_labels, 1),
    }


@torch.no_grad()
def _termination_report(trainer: Trainer, sample_windows: int) -> dict[str, object]:
    wm = trainer.wm
    window = trainer.buffer.sample_windows(sample_windows, 8)
    refs = _reference_steps(trainer, window)
    state = window.states[0]
    probability, replay_probability, survival, root_height, tilt = [], [], [], [], []
    labels, nse = [], []
    cumulative_survival = torch.ones(state.shape[0], device=state.device)
    for j in range(8):
        state = wm.step(state, window.a_exec[j], window.xi)
        logits = wm.termination_logits(state, refs[j])
        probability.append(torch.sigmoid(logits))
        real_logits = wm.termination_logits(window.states[j + 1], refs[j])
        replay_probability.append(torch.sigmoid(real_logits))
        cumulative_survival = cumulative_survival * (1.0 - torch.sigmoid(logits))
        survival.append(cumulative_survival)
        if refs[j] is None:
            feats = wm.heads.free_termination_features(state[..., :3], state[..., 3:7], wm.cfg.termination_base_height)
        else:
            feats = wm.heads.termination_features(state[..., :3], state[..., 3:7], refs[j])
        root_height.append(state[..., 2])
        tilt.append(feats[..., 1])
        labels.append(window.done[j])
        nse.append(wm.spec.normalized_sq_error(state, window.states[j + 1]))
    p_steps = torch.stack(probability)
    p = p_steps.reshape(-1)
    p_real_steps = torch.stack(replay_probability)
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
        "replay_state_probability_mean_by_step": [float(x) for x in p_real_steps.mean(dim=1)],
        "predicted_state_probability_mean_by_step": [float(x) for x in p_steps.mean(dim=1)],
        "predicted_survival_mean_by_step": [float(x) for x in torch.stack(survival).mean(dim=1)],
        "replay_root_height_mean_by_step_m": [float(x) for x in window.states[1:, :, 2].mean(dim=1)],
        "predicted_root_height_mean_by_step_m": [float(x) for x in torch.stack(root_height).mean(dim=1)],
        "predicted_tilt_feature_mean_by_step": [float(x) for x in torch.stack(tilt).mean(dim=1)],
        "steps_correspond_to": "step_00 through step_07 (first through eighth transition)",
        "nse_1step": float(nse_t[0].mean()),
        "nse_8step": float(nse_t[-1].mean()),
        "label_status": "both_classes_present" if y.any() and (~y).any() else "one_class_only",
    }


def _imagination_report(trainer: Trainer, start, horizon: int) -> dict[str, object]:
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
            "replay_state_termination_probability": float(
                torch.sigmoid(wm.term_head(rollout.term_features[0])).mean().detach()
            ),
            "termination_probability_by_step": [float(x) for x in rollout.dhat[1:].mean(dim=1).detach()],
            "survival_by_step": [float(x) for x in rollout.survival[1:].mean(dim=1).detach()],
            "root_height_mean_by_step_m": [float(x) for x in rollout.root_height[1:].mean(dim=1).detach()],
            "tilt_feature_mean_by_step": [float(x) for x in rollout.tilt[1:].mean(dim=1).detach()],
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
    cfg.env.action_noise_std = args.action_noise if args.robot == "m1_piper" else 0.0
    cfg.env.velocity_gaussian_noise_std = args.action_noise if args.robot == "go2" else 0.0
    cfg.env.velocity_ou_env_fraction = 0.0
    cfg.env.velocity_ou_noise_std = 0.0
    cfg.env.velocity_curriculum = False
    cfg.env.velocity_curriculum_fixed_stage = 0
    cfg.env.sampling_mode = "start"
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
        replay_termination_coverage = _replay_termination_coverage(trainer, 8)
        wm_updates = []
        for _ in range(args.wm_updates):
            wm_updates.append(trainer.wm_trainer.update(trainer.buffer))
        after = _termination_report(trainer, args.diagnostic_windows)
        starts = trainer.buffer.sample_starts(args.imag_batch, cfg.env.history_length)
        imagination = [_imagination_report(trainer, starts, h) for h in (8, 32)]
        unfinished = lengths > 0
        partial_returns = env_returns[unfinished].detach().cpu().tolist()
        partial_lengths = lengths[unfinished].to(torch.int64).cpu().tolist()
        partial_normalized_returns = (
            [
                float(ret / (trainer.max_step_reward * max(length, 1)))
                for ret, length in zip(partial_returns, partial_lengths, strict=True)
            ]
            if args.robot == "m1_piper" else []
        )
        result: dict[str, object] = {
            "git_commit": _git_commit(),
            "git_worktree_dirty": bool(subprocess.call(["git", "diff", "--quiet"]) != 0),
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
            "replay_termination_coverage": replay_termination_coverage,
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
            "real_simulation_partial_return_mean": (
                sum(partial_returns) / len(partial_returns) if partial_returns else None
            ),
            "real_simulation_partial_normalized_return_mean": (
                sum(partial_normalized_returns) / len(partial_normalized_returns)
                if partial_normalized_returns else None
            ),
            "real_simulation_partial_env_count": len(partial_returns),
            "real_simulation_partial_length_mean": (
                sum(partial_lengths) / len(partial_lengths) if partial_lengths else None
            ),
            "real_return_semantics": (
                "M1 physical tracking reward" if args.robot == "m1_piper"
                else "Go2 velocity reward core applied by the real RewardManager"
            ),
            "policy_optimizer_steps": 0,
            "go2_reward_note": "uses the training reward core; the diagnostic does not step the policy optimizer" if args.robot == "go2" else None,
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
    parser.add_argument("--num-envs", type=int, choices=(8, 16), default=16)
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
