# M1 L/M Reward Audit and Fixes

Date: 2026-11-02
Audited base: `0f6826848691b2584c9b3414620ca1e2b0d8c604` (`main`).  128-env runs and tests used the local modified worktree. Existing unrelated G1 edits were retained.

## Findings and changes

Key implementation locations in the audited worktree:

| File | Location | Change |
|---|---:|---|
| `src/supermimic/motion_prior/velocity.py` | 10–55 | Shared `[vx,vy,wz]` error semantics and task-masked metric helper. |
| `src/supermimic/motion_prior/metrics.py` | entire file | Shared root tilt and stable-base predicate for real Reach success. |
| `src/supermimic/motion_prior/losses.py` | 23–64, 86–152, 179–240 | L/M objective, physical scales and jerk/acceleration losses; provisional smoothness coefficient. |
| `src/supermimic/envs/mdp/commands.py` | 1291–1498 | Episode-fixed task context, L metrics, M fixed target/TCP, stable-base success hold. |
| `src/supermimic/training/loop.py` | 940–993, 1200–1290, 1357–1374, 1792–1820 | Shared real metrics, task masks, fall-filtered hold success, WM diagnostics and console line. |
| `src/supermimic/world_model/wm_trainer.py` | `online_error_components` | Replay-resampled open-loop 1/k-step physical error breakdown. |
| `src/supermimic/motion_prior/trainer.py` | 101–171, 412–464 | Unit-aware Actor gate and WM readiness requiring configured task-specific quality thresholds. |
| `src/supermimic/config.py` | 180–217 | Explicit readiness fields and physical-unit limits. |
| `tests/test_m1_reward_audit.py` | entire file | Axis, gradient, fixed-goal, four-direction Reach, task-loss and masked metric regressions. |
| `tests/test_m1_prior_joint.py` | `test_m1_prior_actor_gradients_after_reward_fix` | L-task CVAE gradient and zero vertical-velocity gradient. |

### L velocity axes

The L command schema is `[vx_body_cmd, vy_body_cmd, wz_cmd]`, while root linear velocity is `[vx, vy, vz]`. Before this audit, both the planner and MuJoCo metrics subtracted all three command values from body linear velocity. The planner also separately tracked actual yaw rate, so a yaw-only command `[0,0,0.3]` simultaneously drove `vz` toward `0.3 m/s` and `wz` toward `0.3 rad/s`; the real metric likewise reported vertical-speed error instead of yaw error in that third linear component. This was an active task-objective bug, not just a naming ambiguity.

`motion_prior/velocity.py` now owns the shared semantics used by the planner and real metrics:

```text
body_xy_error = quat_inverse(root_q) * root_linear_velocity_world, first 2 components - [vx_cmd, vy_cmd]
yaw_rate_error = root_angular_velocity_body.z - wz_cmd
body_vertical_velocity = transformed root linear velocity.z   # diagnostic only
```

The planner retains Smooth L1 XY tracking, normalized by the prior `0.5` scale, and a separate Smooth L1 yaw-rate term with its prior `0.5` weight. It has no L-task gradient with respect to vertical velocity. Real logs now include `real/L/base_velocity_xy_rmse`, `real/L/yaw_rate_error`, and `real/L/base_vertical_velocity`; the legacy `base_velocity_error` alias is documented as XY-only. L/M values are masked by task, and empty task masks return no metric rather than a false zero. Current command sampling remains `vx=[0.15,0.50]`, `vy=[-0.20,0.20]`, `wz=[-0.40,0.40]`; it has no zero-vx pure lateral/yaw category. No training-distribution change was made.

The unit parity check uses actual body velocity `[0.32,-0.08,0.70]` and command `[0.30,-0.10,0.20]`: both the real path and planner helper yield XY residual `[0.02,0.02]`, XY RMSE `0.02828 m/s`, yaw residual `-0.01 rad/s`; vertical velocity remains `0.70 m/s` and is not part of tracking. The planner applies its normalized Smooth L1 transformation to those shared errors (linear contribution `0.000400`, yaw contribution `0.000025`, total `0.000425` with current weights). Pure yaw with matching yaw rate has zero L task loss for any `vz`.

### Stationary-policy and wheel-spin audit

The exact deterministic unit fixture uses a 0.2 s horizon, the same initial state, no reference motion, feasible joint positions, and command `[0.3,0,0]`. It compares these task/total losses:

| Case | L task | Feasibility | Total planner loss |
|---|---:|---:|---:|
| stationary | 0.0450 | 0.0000 | 0.0450 |
| correctly moving at 0.3 m/s | 0.0000 | 0.0000 | 0.0000 |
| moving at -0.3 m/s | 0.1800 | 0.0000 | 0.1800 |
| wheels at 5 rad/s, base stationary | 0.0450 | 0.0000 | 0.0450 |
| crouched to 0.20 m, base stationary | 0.0450 | 0.1600 | 0.2050 |

This verifies the objective does not count wheel rotation as base progress, moving correctly beats standing still, moving backwards is worse, and the configured low-height penalty makes the crouched state worse. The wheel case is deliberately below the planner's 10 rad/s wheel-rate cap; the metric still reports wheel rate separately for slip/spin diagnosis. This is a loss-shape unit fixture, not a MuJoCo-feasible trajectory comparison. No sampled MuJoCo replay comparison among stationary, prior, learned Actor, and Motion Bank replay was available in this run, so total-loss ordering on the real state/action distribution remains unverified.

### Smoothness scale

The finite-difference implementation starts each Future31 path with current State57 at `t=0`; position and velocity differences divide by `dt=0.02 s` once per derivative. Root translation uses metres, quaternion increments use relative rotation-vector angular velocity, joint positions use radians, and Future31 wheel values are wheel angular rates in rad/s paired with current wheel joint velocities. No wheel angle is substituted for wheel rate. The eight terms are normalized hinge penalties for root linear acceleration/jerk, root angular acceleration/jerk, joint acceleration/jerk and wheel acceleration/jerk. These thresholds are planner penalty thresholds, not certified hardware limits. Collision feasibility is not implemented as a differentiable guarantee.

Calibration used 180 AMP Motion Bank L trajectories and 180 M trajectories for empirical finite-difference quantiles, plus 100 L and 100 M windows passed through the frozen CVAE prior mean. The original kinematic database is not an actuator-limit certificate. Selected CVAE-reference absolute derivative quantiles across the sampled windows were:

| Quantity | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|
| root linear acceleration (m/s²) | 0.307 | 0.820 | 1.059 | 1.714 |
| root linear jerk (m/s³) | 26.57 | 71.94 | 91.41 | 157.94 |
| root angular acceleration (rad/s²) | 1.119 | 2.511 | 3.221 | 5.510 |
| root angular jerk (rad/s³) | 98.09 | 214.17 | 271.23 | 449.84 |
| joint absolute acceleration (rad/s²) | 1.90 | 11.55 | 16.86 | 30.70 |
| joint absolute jerk (rad/s³) | 145.39 | 742.64 | 1092.65 | 2257.93 |
| wheel absolute acceleration (rad/s²) | 0.119 | 0.613 | 0.982 | 1.972 |
| wheel absolute jerk (rad/s³) | 9.75 | 49.03 | 78.63 | 151.42 |

For context, raw AMP Motion Bank p99 values from the sampled L/M trajectory subsets were: L root linear acceleration/jerk `0.934 m/s² / 7.35 m/s³`, angular acceleration/jerk `1.09 rad/s² / 8.32 rad/s³`, and leg acceleration/jerk `27.97 rad/s² / 461.2 rad/s³`; M corresponding root values were `0.078 / 0.156`, angular values `0.127 / 0.226`, and leg values `0.303 / 21.37`. M arm p99 was `2.61 rad/s² / 192.6 rad/s³`, wheel p99 about `2.0 rad/s² / 161.5 rad/s³`. The frozen CVAE figures above pool sampled L/M windows and are not task-stratified; the value distributions are therefore descriptive, not a matched statistical comparison. The live Prior was never updated in the Pilots, so there is no post-training Prior distribution to report.

The current hinge thresholds are `12 m/s²` root acceleration, `75 m/s³` root jerk, `30 rad/s²` root angular acceleration, `600 rad/s³` angular jerk, `80 rad/s²` joint acceleration, `2000 rad/s³` joint jerk and `80 rad/s²` wheel acceleration / `4000 rad/s³` wheel jerk. Relative to these samples, several acceleration and wheel jerk thresholds sit far above the sampled p99 and will be inactive on ordinary CVAE output; root and joint jerk are the terms most likely to activate. They are inherited operating thresholds, not thresholds calibrated from actuator saturation or MuJoCo force. The scalar coefficient was reduced because its gradient scale was clearly excessive; the individual physical thresholds still need matched-data and dynamics validation.

On a repeated reset-state batch, command `[0.3,0,0.2]`, prior-mean reference, and direct CVAE reference graph, unweighted smoothness was `6.3344`. L task loss was `0.04536`; its direct decoder-bias task-gradient norm was `0.02264`, versus smoothness-gradient norm `126.64` before weighting. Weight sweep:

| Weight | Weighted smoothness | Smoothness/task gradient-norm ratio (direct-reference proxy) |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0.001 | 0.006334 | 5.59 |
| 0.003 | 0.019003 | 16.78 |
| 0.01 | 0.063344 | 55.94 |
| **0.0001 (selected provisional)** | **0.000633** | **0.56** |

This motivates lowering the default from `0.01` to `0.0001`. It is a scale correction and an initial calibration only: the gradient ratio is measured through the direct CVAE reference, not through a trusted 10-step Actor/WM task Jacobian. Planner updates were correctly withheld during the 128-env Pilots, so no measured learned-Prior task-gradient direction or gradient ablation exists yet. Smoothness should be reconsidered after held-out WM and executable-reference tests.

### M reach and actor terms

M planner evaluates differentiable-FK TCP against the fixed environment-local world target, not Actor output or predicted-reference TCP. Targets are sampled once at reset from offsets `x=[0.12,0.36] m`, `y=[-0.20,0.20] m`, `z=[-0.15,0.15] m`, with world height clamped to `[0.10,1.20] m`; the success radius is `0.08 m` held for `0.5 s`. Each rollout recomputes heading-relative Goal8 from the current predicted base while retaining the same fixed target in Replay. Real MuJoCo metrics use the same `piper_x_link7` FK TCP convention as the joint planner. These checks confirm data-path consistency, but they do not prove all target samples are collision-free/reachable.

The current task expression is:

```text
d_final/0.10 + 0.1 * mean_distance/0.10 - (d_initial - d_final)/0.10
```

Since `d_initial` is constant with respect to the rollout, the terminal-distance derivative is effectively `20 * d_final / metre` before the mean-path contribution. This is deliberate additive emphasis, not three independent signals; logs expose terminal error and progress separately. A synthetic same-start test with the goal 0.30 m from TCP yields stationary `M task=3.300`, versus an approaching reference with `0.15 m` progress and `0.15 m` terminal error `M task=0.2175`; its base drift penalty raises total from `0.2175` to `0.2314`, still below stationary. That fixture does not stand in for real dynamics.

`actor_tcp_weight=0.0` is explicitly marked reserved: `actor_losses()` does not implement TCP-reference tracking. Actor continues to optimize reference tracking plus existing stability/control regularizers; it does not receive hidden world-goal task reward. Planner energy remains zero and disabled; action effort remains an explicitly named optional surrogate. No true energy reduction claim is supported.

The Motion Bank M subset's initial EE-goal distance p50/p90/p95 was approximately `0.496/0.683/0.699 m`; this pilot's offset norm is bounded by about `0.437 m` before world-height clamping. This suggests the sampled offsets are not trivially millimetre-close and are within the scale of database reaches, but it does not certify collision-free or dynamically reachable targets for the reset posture.

The audit also found that the old real success hold counter accepted proximity alone. M1 Reach success now requires both EE distance `<0.08 m` and a stable base (`height >= m1_fall_height`, tilt `<= velocity_fall_tilt_deg`), using the same limits as M1 environment fall termination. Episode success is counted only for episodes that did not terminate from a fall. The old Pilot's `real/ee_success_hold_rate` predates this added stability filter; M recorded no falls in its logged completed episodes, but that tag must be regenerated in any follow-up.

## WM and readiness audit

The two 128-env Pilots each ran to iteration 1500, with randomized (not pretrained) WM, 200 WM-only iterations, then Actor tracking warmup. Each completed 584 WM update events/optimizer steps and 384 Actor optimizer steps; Prior optimizer steps stayed at zero. This is expected: the Actor physical readiness gate did not pass, and task-specific WM quality limits have intentionally not been invented. `freeze_prior=False` cannot bypass those gates. Current readiness errors are in physical units: root position metres, Piper joint position RMSE radians, wheel angular rate RMSE rad/s, plus survival, action saturation and minimum sample count.

WM logs now distinguish the training/replay-window NSE from replay-resampled open-loop diagnostics and report one-step physical component errors. These samples are still replay-resampled, not held out. No fixed held-out M/L state-action evaluation set or action-gradient finite-difference comparison against MuJoCo was run. Thus nonzero finite planner gradients in earlier random-WM Smokes only prove that Autograd connects; they do not certify gradient utility. New M1 console output reports update counts, readiness failures, task metrics and GPU peak memory; empty task metrics are omitted rather than printed as zeros.

## Tests executed

- `pytest -q tests/test_m1_prior_joint.py tests/test_m1_reward_audit.py`: **18 passed**.
- Full `pytest -q`: **89 passed**, 64 Torch/MuJoCo-Warp warnings.
- `ruff check src/supermimic/motion_prior tests/test_m1_prior_joint.py`: passed.
- `ruff check tests/test_m1_reward_audit.py`: passed.
- `git diff --check`: passed.

A broader direct `ruff check src/supermimic/training/loop.py` still reports six existing findings elsewhere in that shared file (import sorting, unused `world_size`/`extras`/`noqa`, and an already-integer cast). The new M1 logging helper has no Ruff finding. The shared loop already contained those findings before this M1 audit; they were left alone to avoid unrelated edits.

## Remaining uncertainty

The fixed-fixture objectives are correct by construction, but there is no validated executed moving-vs-stationary comparison, no successful real M reach, no Prior update in the new Pilots, and no held-out multi-step WM/gradient audit. The next useful experiment is held-out L and M WM evaluation plus a readied reference-tracking Actor, not a 2048-env run.
