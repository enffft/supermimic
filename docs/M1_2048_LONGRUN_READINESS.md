# M1 2048-Environment / Two-Hour Readiness

Date: 2026-11-02. The requested 2048-env, 2-hour run was **not started**. Both independent 128-env Pilots ended at their 1500-iteration stop limit. A successful launch at the larger size is not justified by the evidence collected.

## Decision

| Scope | Decision | Evidence |
|---|---|---|
| Task L | **NOT READY** | XY RMSE reached 0.163 m/s transiently and ended at 0.225; yaw error ended at 0.200 rad/s. Actor readiness failed Piper (0.910 rad mean) and wheel (2.782 rad/s mean). Replay 8-step WM NSE ended at 0.261, training-window NSE8 at 0.335. |
| Task M | **NOT READY** | EE distance briefly reached 0.224 m but ended at 1.281 m, with -1.003 m interval progress and 1.593 m base drift. Success hold was zero at the end. Actor readiness failed arm (0.546 rad) and wheel (2.296 rad/s). Replay 8-step WM NSE ended at 0.218. |
| 2048 env | **NOT READY** | No held-out WM/action-gradient validation, no passing Actor gate, no stable task completion, and no 512/2048 memory preflight. 128-env peak was 0.30 GiB with microbatch 8 and no Prior BPTT; it does not scale linearly to certify memory. |
| 2-hour training | **NOT READY** | Both Pilots show the real task objective is not yet stable. Increasing duration would compound unvalidated WM/Actor behavior. |

The highest-priority blockers are (1) improve Piper and wheel reference tracking until the physical readiness gate passes, (2) collect held-out L and M transition sets and validate WM one-/8-/10-step errors on Actor actions, then (3) establish WM thresholds and compare predicted task-gradient direction with paired MuJoCo perturbations. For M, add a controlled base-drift versus arm-reach analysis. For L, include explicit lateral/yaw-only evaluation and report measured base displacement and wheel spin. Do not lower readiness thresholds just to open the Prior gate.

At the final resource check, `nvidia-smi` showed the RTX 4070 SUPER at 646 MiB used of 12,282 MiB and 28% utilization; the compute-app query returned no rows. Earlier in the session, ToDesk appeared as a 245 MiB video-session process and was left untouched. This snapshot does not replace a 2048-env peak-memory preflight.

## Resource and training gates

Before scaling, use separate new run names/checkpoints for L and M and a staged `128 → 512 → 2048` memory/performance check. Keep replay fields intact, report peak allocated and reserved CUDA memory, and reduce differentiable microbatch before reducing logical batch. The current Pilot reports allocated memory only (0.30 GiB) and did not exercise joint Prior BPTT, so it is not a useful upper bound for the intended 2048-env run.

First complete:

1. Fixed held-out replay suites for each task, including actor-generated actions, reporting one-, eight- and ten-step physical component errors and TCP FK error.
2. A readied tracking Actor evaluated in MuJoCo against frozen CVAE references, with survival, root/arm/wheel errors, saturation and per-task task metrics. Re-run M success-hold measurement with the newly added stable-base/fall-episode filter.
3. A paired action perturbation check for predicted versus MuJoCo changes in base velocity (L) and fixed-world EE progress (M). A finite nonzero gradient alone is insufficient.
4. New L/M 128-env Pilots with the final metric tags, particularly L command/wheel-rate and M base-drift/success-hold. These thresholds must be chosen from the held-out calibration, not copied from Go2 or inferred from the present replay window.
5. Only after those gates pass, run a bounded 512-env memory/performance check and then decide whether 2048 fits.

## CLI-checked launch templates (blocked)

The local `sm-train --help` confirms `--m1-joint-training.wm-readiness-max-nse-1`, `--m1-joint-training.wm-readiness-max-nse-horizon`, `--m1-joint-training.task-type`, batch/microbatch, `--schedule.max-hours` and `--schedule.stop-at-iteration`. The WM thresholds below are deliberately placeholders. The trainer will keep Prior frozen unless both task-specific values are configured and the Actor gate passes. Do not run these templates until the values have been derived from independent held-out L or M validation and the other gates above pass.

Replace `<L_HELDOUT_NSE1_LIMIT>` and `<L_HELDOUT_NSE8_LIMIT>` only after the held-out L validation. Then launch L as its own experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
    --run-name m1_prior_L_2048_after_readiness \
    --log-root logs/m1_prior_actor \
    --device cuda:0 \
    --env.robot-name m1_piper \
    --env.task-name m1_prior_joint \
    --env.num-envs 2048 \
    --env.history-length 5 \
    --env.no-obs-noise \
    --env.no-reset-noise \
    --env.action-noise-std 0 \
    --env.domain-randomization \
    --env.replay-capacity 1000 \
    --m1-joint-training.enabled \
    --m1-joint-training.task-type L \
    --m1-joint-training.state-source privileged \
    --m1-joint-training.prior-mode prior_mean \
    --m1-joint-training.batch-size 32 \
    --m1-joint-training.micro-batch-size 8 \
    --m1-joint-training.freeze-prior \
    --m1-joint-training.wm-readiness-max-nse-1 <L_HELDOUT_NSE1_LIMIT> \
    --m1-joint-training.wm-readiness-max-nse-horizon <L_HELDOUT_NSE8_LIMIT> \
    --wm-trainer.window 8 \
    --wm-trainer.batch-size 32 \
    --wm-trainer.micro-batch-size 8 \
    --wm-trainer.batches-per-update 1 \
    --schedule.m1-wm-only-iterations 200 \
    --schedule.warmup-iterations 40 \
    --schedule.total-iterations 320000 \
    --schedule.max-hours 2 \
    --schedule.log-interval 50 \
    --schedule.checkpoint-interval 500
```

Use a distinct M run and **M-specific** held-out limits. Never resume L's full joint checkpoint for M:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
    --run-name m1_prior_M_2048_after_readiness \
    --log-root logs/m1_prior_actor \
    --device cuda:0 \
    --env.robot-name m1_piper \
    --env.task-name m1_prior_joint \
    --env.num-envs 2048 \
    --env.history-length 5 \
    --env.no-obs-noise \
    --env.no-reset-noise \
    --env.action-noise-std 0 \
    --env.domain-randomization \
    --env.replay-capacity 1000 \
    --m1-joint-training.enabled \
    --m1-joint-training.task-type M \
    --m1-joint-training.state-source privileged \
    --m1-joint-training.prior-mode prior_mean \
    --m1-joint-training.batch-size 32 \
    --m1-joint-training.micro-batch-size 8 \
    --m1-joint-training.freeze-prior \
    --m1-joint-training.wm-readiness-max-nse-1 <M_HELDOUT_NSE1_LIMIT> \
    --m1-joint-training.wm-readiness-max-nse-horizon <M_HELDOUT_NSE8_LIMIT> \
    --wm-trainer.window 8 \
    --wm-trainer.batch-size 32 \
    --wm-trainer.micro-batch-size 8 \
    --wm-trainer.batches-per-update 1 \
    --schedule.m1-wm-only-iterations 200 \
    --schedule.warmup-iterations 40 \
    --schedule.total-iterations 320000 \
    --schedule.max-hours 2 \
    --schedule.log-interval 50 \
    --schedule.checkpoint-interval 500
```

These are templates, not currently runnable validated commands: shell angle-bracket placeholders must be replaced with calibrated thresholds, and the size preflight remains outstanding. The 128-env Pilot used WM window 8; no threshold is supplied here because replay NSE is not a held-out quality standard. Motion Prior state source remains privileged, so even a successful run would not establish a deployable proprioceptive controller.
