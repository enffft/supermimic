# M1 Prior/Actor Smoke Results

Date: 2026-11-02. Runs use `/home/yifeiy/miniconda3/envs/supermimic/bin/sm-train` on an RTX 4070 SUPER 12 GiB. All smoke artifacts are in `/tmp/sm_m1_prior_smokes`; no prior checkpoint, motion bank, or repository log was overwritten. These runs are pipeline checks, not convergence or task-efficacy evidence.

## Checks passed

- AMP prior checkpoint loaded strictly at step 10,000 with the expected SHA256 `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`.
- Real MuJoCo M1-Piper environment ran with State57, 360-D proprioception history, Goal8, and 22-D Mixed Action.
- The joint path collected real transitions, trained WM from Replay, and ran the Prior and Actor BPTT objectives in the intended order.
- Latest checkpoint is `supermimic_m1_prior_actor_joint_v2` with strict format version 2. Smoke E recorded 8 WM events/optimizer steps, 6 Prior steps, and 6 Actor steps; the 2 configured WM-only warmup events preceded the 6 alternating updates. The checkpoint's live Prior differs from its frozen initial Prior (max absolute parameter difference `2.15e-6`). Runtime checks and the CUDA ownership test ensure frozen Actor/WM parameters do not change during Planner updates, and frozen Prior/WM do not change during Actor updates.
- Full test suite: `75 passed`. Ruff for `src/supermimic/motion_prior` and `tests/test_m1_prior_joint.py`: passed. `git diff --check`: passed. The final full-suite run includes the decomposed smoothness metrics change.

## Latest measured runs

### Smoke E: short joint update with two WM warmup events

Configuration: 8 environments, M-only, 16 control iterations, `warmup_iterations=9`, `m1_wm_only_iterations=2`, batch/microbatch 2, WM window 8, 1 WM step/event, 1 Prior step/event, 1 Actor step/event, no observation/action/reset noise and no domain randomization. A randomly initialized small WM was used intentionally to test the complete initialization path. Result: 8 WM optimizer steps, then 6 Prior and 6 Actor optimizer steps. Joint checkpoint and TensorBoard data: `/tmp/sm_m1_prior_smokes/m1_prior_smoke_e_20261102`.

### Smoke F: smoothness component diagnostics

Configuration: 8 environments, M-only, 12 iterations, one WM-only event, then 3 alternating updates; constant LR to avoid the tiny-run cosine schedule collapsing the learning rates. Result checkpoint: `/tmp/sm_m1_prior_smokes/m1_prior_smoke_f_20261102/ckpt_latest.pt`.

The three joint samples varied sharply, which is material evidence that the objective and random WM are not calibrated:

| Metric | Three Planner updates / last real steps |
|---|---:|
| Planner task loss | 6.65, 2.87, 8.03 |
| Predicted EE progress (m) | -0.175, +0.002, -0.216 |
| Real EE distance (m), steps 10–12 | 0.359, 0.387, 0.418 |
| Real one-step EE progress (m), steps 10–12 | -0.068, -0.096, -0.126 |
| Real EE success | 0 throughout |
| WM normalized 8-step error, steps 10–12 | 0.335, 0.488, 0.731 |
| Actor tracking loss, updates 1–3 | 0.337, 0.235, 0.175 |

The reference smoothness regularizer was reduced from the earlier dimensionally incorrect jerk scaling, but its scale is still not calibrated. In Smoke F it ranged from `3.20` to `525.98` before its configured `0.01` weight. At the last update, the unit-normalized components were: root acceleration `0.120`, root jerk `190.745`, root angular acceleration `1.362`, root angular jerk `37.073`, joint acceleration `21.179`, joint jerk `138.497`, wheel acceleration `0.0001`, and wheel jerk `0.0001`. Root and joint jerk dominate. The prior gradient norm before clipping ranged from `10.2` to `268.3`; clipping is active. These figures need calibration against a held-out set of CVAE references and task trajectories before a longer run.

The measured real EE distance increased during this tiny random-policy run, and simulated EE success stayed zero. The imagined World Model's 8-step error also grew over the last three logged updates. The low one-step error does not validate 10-step planner gradients. `imag/survival` from the untrained termination head is not a meaningful safety estimate.

Real positive actuator mechanical power was logged from MuJoCo generalized actuator force × joint velocity. It is simulator mechanical power only; it is not a battery-energy estimate and was not optimized. Planner energy loss and energy weight remained zero. No motion energy improvement is demonstrated.

## Earlier staged checks

| Run | Purpose | Outcome |
|---|---|---|
| Smoke A | 8-env reference-only initialization | CVAE strict load, reset/history, reference and MuJoCo action loop ran; no optimizer update |
| Smoke B | Prior frozen, Actor-only tracking | 7 WM and 7 Actor updates; validates path execution, not tracking readiness |
| Smoke C | Planner-only | 4 WM and 4 Prior updates; before final v2 and smoothness diagnostics |
| Smoke D | Full alternation | 4 WM, 4 Prior, 4 Actor updates; before final jerk-scaling and checkpoint changes |
| Smoke E | WM warmup then joint | 8 WM, 6 Prior, 6 Actor updates; v2 checkpoint and ownership assertions |
| Smoke F | final loss breakdown | 4 WM, 3 Prior, 3 Actor updates; final code path with decomposed smoothness metrics |

The stdout summaries print legacy `imag_ret` fields as NaN for this new task; read the `prior/*`, `actor/*`, `imag/*`, `wm/*`, and `real/*` TensorBoard tags for this path. This is a logging presentation issue, not a NaN in the loss; all optimizer losses and gradients passed finite checks.

## Readiness decision

**Not ready for long training.** Missing evidence includes a validated multi-step World Model on L and M replay distributions, a reliable Actor-readiness pass under the default thresholds, smoothness/Planner-loss scale calibration, held-out goal and seed evaluation, and sustained positive real EE progress or L velocity-command tracking. Use the configured WM warmup and default measured Actor-readiness gate after those calibrations. Do not promote the short smoke checkpoint as a policy result.
