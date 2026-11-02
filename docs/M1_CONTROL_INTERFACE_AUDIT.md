# M1 Motion Prior / Actor Control Interface Audit

Date: 2026-11-02
Repository base: `0bda727aaf5b422af792017ac3afb3190db92ee1` (`main`)
Working tree: contains this audit/fix plus the user's pre-existing Go2 edits; those edits were preserved.
Prior source: AMP-LOCO-M1 CVAE checkpoint SHA256 `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`.

## Action and state contract

M1 `State57` is `[root position(3), root quaternion wxyz(4), joint position(22), root linear velocity(3), root angular velocity(3), joint velocity(22)]`. The 22 action entries follow `M1_JOINTS`: 12 leg entries interleave three position controls and one wheel velocity control per leg, followed by six Piper position controls. The shared `mixed_m1_setpoints()` mapping clips the normalized action to `[-1,1]`; position entries map to default joint position plus scale and are then clamped to MuJoCo soft limits; wheel entries map to angular-velocity targets.

The previous Piper position scales were all `0.25 rad`; wheel targets were `±5 rad/s`. Seeded 32-environment MuJoCo audits decoded frozen `prior_mean` references for L and M and compared them to the actual default pose, action envelope and URDF soft limits.

| Task | Reference statistic | Before scale change | After scale change |
|---|---:|---:|---:|
| L | Position references outside setpoint envelope | 9.24% | 0% |
| L | Piper 2 outside envelope | 100% | 0% |
| L | Piper 4 outside envelope | 66.25% | 0% |
| L | Position references outside soft joint limits | 0% | 0% |
| M | Position references outside setpoint envelope | 0% | 0% |
| L/M | Wheel references outside `±5 rad/s` | 0% | 0% |

The L references required Piper 2 offsets of approximately `−0.322…−0.277 rad` from its default and Piper 4 offsets of `+0.244…+0.262 rad`. Their URDF soft limits permit these targets. Piper 2 and Piper 4 action scales are therefore `0.35 rad`; all other position scales remain `0.25 rad`, and wheel scales remain `5 rad/s`. Planner wheel-rate feasibility now uses the same `5 rad/s` setpoint bound.

Frozen-prior wheel-rate references were near zero in these particular reset/goal windows (per-wheel p5/p50/p95 ranges were approximately `−0.014…+0.016 rad/s` in L and `−0.016…+0.005 rad/s` in M). This is a property of the tested prior outputs, not a statement that the real robot can only turn its wheels that slowly.

## Real MuJoCo reference replay

Each audit executed the complete 10-frame plan in MuJoCo at `dt=0.02 s`, without observation/reset noise or domain randomization. The three controls were the trained Pilot Actor, a correctly inverted 22D Mixed Action derived from the 31D kinematic reference, and the same Actor with references projected into the action envelope. The inverse action applied joint-name mapping, default offsets, individual scales, clipping and wheel-rate semantics; it did not reinterpret Future31 as an Action.

| Task/control | Root position error (m) | Leg RMSE (rad) | Piper RMSE (rad) | Wheel-rate RMSE (rad/s) | Saturation | Falls |
|---|---:|---:|---:|---:|---:|---:|
| L Pilot Actor | 0.0315 | 0.1283 | 0.2269 | 3.402 | 0.91% | 0% |
| L inverse reference action | 0.0159 | 0.0438 | 0.1017 | 1.512 | 0% | 0% |
| M Pilot Actor | 0.0254 | 0.1285 | 0.1206 | 3.656 | 0.45% | 0% |
| M inverse reference action | 0.0164 | 0.0416 | 0.0746 | 1.555 | 0% | 0% |

This establishes that the joint mapping and actuators can follow the tested references over the short window. It also shows that the Pilot Actor had not yet learned wheel/reference tracking as accurately as direct feasible setpoints. It does not establish long-horizon task success. The L scale update makes the tested references representable without leaving soft limits. It changes the actuator contract, so prior M1 joint checkpoints carrying the old scale are not resume-compatible; new checkpoints use `m1_mixed_action_v2` and store the exact per-joint scales. G1/Go2 mappings are unchanged.

Reproduce with:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/audit_m1_reference_control.py \
  --actor-checkpoint logs/m1_prior_actor/m1_prior_L_128_pilot_audit_20261102/ckpt_latest.pt \
  --task-type L --num-envs 32 \
  --out logs/m1_prior_actor/control_audit_L_scale_v2_20261102.json
```

M uses the corresponding M Pilot checkpoint and `--task-type M`. Detailed per-joint JSON files are retained under `logs/m1_prior_actor/control_audit_*_scale_v2_20261102.json`.

## Velocity semantics

The L planner compares only body-frame `vx,vy` with command `vx,vy`, and compares yaw rate `wz` with the third command component. `vz` is not a command-tracking dimension. The real command metric uses the same XY/yaw decomposition. This keeps wheel rotation from counting as base translation and keeps the earlier `wz`/`vz` mix-up covered by regression tests.

## Validation

- Full test suite: `106 passed` (64 upstream deprecation/CUDA toolkit warnings).
- Targeted M1 joint/reward/control tests: `24 passed`.
- Ruff on changed M1 control/trainer/test files: passed.
- 512-env controlled-joint preflight: 30 iterations, WM/Prior/Actor optimizer counts `8/6/6`, peak training allocation `0.25 GiB`.
- 2048-env controlled-joint preflight: 30 iterations, counts `8/6/6`, peak training allocation `0.33 GiB`; no OOM/NaN. The strict WM quality Gate stayed false and the run is only a stage/throughput check.

The short preflights validate mechanics and memory, not learned control quality. Full Task L/M run outcomes are recorded separately.
