# M1 L/M Independent Pilot Results

Date: 2026-11-02. Base commit `0f6826848691b2584c9b3414620ca1e2b0d8c604`; both runs used the modified local worktree, separate run names/checkpoints, 128 environments, privileged planner state, prior-mean CVAE, M1 batch 32/microbatch 8, WM batch 32/microbatch 8, WM window 8 and 1500 iterations. Observation/reset noise was off, action noise was 0, and domain randomization remained enabled. The configured AMP prior hash was `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`. No other training was interrupted. Each run used its own world model initialized from scratch, not a resume from the other task.

Run folders:

- L: `logs/m1_prior_actor/m1_prior_L_128_pilot_audit_20261102`
- M: `logs/m1_prior_actor/m1_prior_M_128_pilot_audit_20261102`

Both reached iteration 1500 normally in about 0.08 hours, saved `ckpt_latest.pt` and numbered checkpoints, and peaked at about 0.30 GiB PyTorch CUDA allocation. This is not a memory estimate for 2048 environments: these Pilots use small batches and differentiable Prior updates never opened. WM was not pretrained (`wm/pretrained_initialized=0`).

The Pilot console formatter at run time printed `nan` for skipped Prior updates because the metric keys did not exist; this meant “not computed”, not a nonfinite Prior loss. The current loop now displays `n/a` for absent/nonfinite diagnostic fields. The Pilot logs themselves are kept unchanged.

## Task L: pure velocity tracking

| Metric | Pilot result |
|---|---:|
| final real XY velocity RMSE | 0.225 m/s |
| best logged interval XY RMSE | 0.163 m/s |
| initial logged XY RMSE | 0.375 m/s |
| final yaw-rate absolute error | 0.200 rad/s |
| logged yaw-error range | 0.195–0.838 rad/s |
| final base vertical velocity | -0.032 m/s |
| final base height | 0.370 m |
| final logged L fall rate | 0 (last completed-episode window) |
| final WM training NSE1 / NSE8 | 0.035 / 0.335 |
| final replay-resampled open-loop NSE1 / NSE8 | 0.025 / 0.261 |
| final Actor reference error | 0.255 |
| Actor gate mean Piper error | 0.910 rad (limit 0.25) |
| Actor gate mean wheel-rate error | 2.782 rad/s (limit 0.75) |
| Prior / Actor / WM optimizer steps | 0 / 384 / 584 |

Velocity error improved transiently, but not monotonically or stably; yaw error remained material. Real world displacement reached 1.80 m over the recorded interval. Terminal survival was 1.0 and final height was not collapsed. The final interval action saturation was 0.025. The process began before the final extra command-component and wheel-rate interval tags were added, so this run does not contain the new `real/L/command_*` and `real/L/wheel_rate_abs_mean` series; rerun for those specific diagnostics. Its existing speed tags already use the repaired XY semantics.

WM replay NSE8 climbed late (peak about 0.394, final replay-resampled 0.261); training-window final NSE8 was 0.335. At final step, replay 8-step component errors included root position 0.0135 m, root orientation 0.0524 rad, root velocity 0.137 m/s, root angular velocity 0.470 rad/s, leg position 0.0499 rad, arm position 0.0653 rad, wheel position 0.122 rad and wheel velocity 1.244 rad/s. These are replay samples, not held-out validation. The Readiness Gate failed Piper and wheel tracking; WM quality thresholds were unconfigured, so `wm_ready=0`. Prior stayed frozen, as intended.

**Assessment:** the run proves the corrected L metric and actor warmup execute, but the observed velocity improvement is not yet reliable. The Actor cannot track its reference within the arm/wheel thresholds and the WM multi-step error worsens. It is not ready for joint Prior training or long training.

## Task M: fixed-world end-effector reach

| Metric | Pilot result |
|---|---:|
| initial logged EE distance | 0.428 m |
| best logged interval EE distance | 0.224 m |
| final EE distance | 1.281 m |
| final EE progress in interval | -1.003 m |
| final base displacement | 1.593 m (maximum 1.776 m) |
| final base height | 0.375 m |
| final logged M fall rate | 0 (last three completed episodes) |
| final success-hold rate | 0 (maximum logged 0.0026) |
| final WM training NSE1 / NSE8 | 0.020 / 0.154 |
| final replay-resampled open-loop NSE1 / NSE8 | 0.022 / 0.218 |
| final Actor arm tracking error | 0.114 loss component; real gate mean 0.546 rad (limit 0.25) |
| Actor gate mean wheel-rate error | 2.296 rad/s (limit 0.75) |
| Prior / Actor / WM optimizer steps | 0 / 384 / 584 |

The EE briefly approached the target but later moved substantially farther away as the base drift grew; success requires the true TCP to remain within 0.08 m for 0.5 s and was effectively zero. Final interval survival was 1.0 and joint-limit violation was 0. WM final 8-step replay component errors included root position 0.0090 m, root orientation 0.0442 rad, root velocity 0.0915 m/s, arm position 0.0517 rad, wheel position 0.113 rad and wheel velocity 0.960 rad/s. A one-step FK TCP discrepancy was 0.0025 m, but that does not validate multi-step goal gradients. Replay errors are not held out.

As in L, the physical Actor readiness gate failed arm and wheel tracking, WM quality limits were not configured, and the Prior remained frozen. The decrease in Actor tracking loss is not evidence of improved reach: the true fixed-world target metric regressed at the end.

The Pilot's success-hold tag was logged before the final safety audit added stable-base filtering. The logged M completed episodes showed no falls, but the follow-up must use the new metric that rejects low/tilted base states and fall-terminated episodes.

**Assessment:** M is not ready. The dominant observed issue is uncontrolled base displacement plus poor Piper/wheel reference tracking, followed by a WM horizon error that is too large to trust for planner updates.

## Pilot invocation and measured scope

The effective config saved with each run contains these common settings (substitute the task type and run name):

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
    --run-name m1_prior_<L_OR_M>_128_pilot_audit_20261102 \
    --log-root logs/m1_prior_actor \
    --device cuda:0 \
    --env.robot-name m1_piper \
    --env.task-name m1_prior_joint \
    --env.num-envs 128 \
    --env.history-length 5 \
    --env.no-obs-noise \
    --env.no-reset-noise \
    --env.action-noise-std 0 \
    --env.domain-randomization \
    --m1-joint-training.enabled \
    --m1-joint-training.task-type <L_OR_M> \
    --m1-joint-training.state-source privileged \
    --m1-joint-training.prior-mode prior_mean \
    --m1-joint-training.batch-size 32 \
    --m1-joint-training.micro-batch-size 8 \
    --m1-joint-training.freeze-prior \
    --wm-trainer.window 8 \
    --wm-trainer.batch-size 32 \
    --wm-trainer.micro-batch-size 8 \
    --wm-trainer.batches-per-update 1 \
    --schedule.m1-wm-only-iterations 200 \
    --schedule.total-iterations 320000 \
    --schedule.stop-at-iteration 1500 \
    --schedule.max-hours 0.5 \
    --schedule.log-interval 50 \
    --schedule.checkpoint-interval 500
```

The current L/M data sampler and target ranges were left unchanged. Run-specific TensorBoard and checkpoints are separate. The M console is retained at `console.log`; L stdout was captured in the run session but not saved as a console file.

The goals are not deployment-ready: planner state source is privileged. These 128-env runs are short diagnostic Pilots, not convergence, comparison-to-baseline, energy, or deployability evidence.

## Final stable-base metric Smoke

After the success filter change, an independent M-only 8-env / 20-iteration run in `/tmp/m1_reach_stability_smoke_final3` loaded the AMP prior strictly at step 10,000 with the expected SHA256, constructed the live M1 prior command, collected replay and emitted M task metrics through the final loop. It used no optimizer updates (`warmup_iterations=40`), so it verifies environment/metric wiring only. Peak allocated memory was 0.05 GiB and the final log displayed unavailable WM/Prior/Actor aggregates as `n/a` rather than as NaNs.

The first two attempts exposed and fixed type mistakes in the new stability metric path (Bool masked mean and Bool bitwise masking). The final retry completed at iteration 20 with exit code 0. These integration defects were found after the 128-env Pilots; their earlier success-hold tag is not retroactively changed.
