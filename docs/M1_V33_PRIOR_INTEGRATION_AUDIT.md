# M1 V3.3 Motion Prior integration audit

Date: 2026-11-03. Worktree branch: `fix/m1-v33-prior-integration`.

## Artifact and strict load

The selected Prior is
`/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt`.
Its SHA256 is `05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c`.
The old `cvae_v1_strict_fcl_2272` checkpoint remains available as the reproducible baseline (SHA256 `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`).

CPU loading verifies the artifact hash, schema, normalization hash, model state, and metadata before constructing the Prior. The recorded metadata is:

| Field | Value |
|---|---|
| feature version | `m1_cvae_motion_prior_v1_state53_goal8_future31` |
| condition | `History[4,53]+Goal[8]; L=Velocity, M/LM=Reach; no future condition` |
| target | `Future[10,31], 0.2 seconds at 0.02 seconds/frame` |
| architecture | latent 32, hidden 256, 3 layers |
| dataset hash | `95f9f0a39b05d5a0d736532d24f897b3b85b8cbf8f74977de620acc9b1a7146e` |
| split hash | `b1bb36beb45b141a77f28837df376690abc3e1d6b24dd810d5ee07da40fd2624` |
| normalization hash | `0dea5b4cac9a7297ad27e1e4d4192211a6f7af7cd138c73046b2ed7c80dd904f` |
| assurance | `mixed_kinematic_and_sampled_fcl`, `strict_fcl_only=False` |
| step | 9600 |

The input dimensions and normalization are compatible with SuperMimic. q6 standard deviations are about 0.370–0.373 and qd6 about 0.272; q6/qd6 are real motion features in this release. The selected input policy is `legacy_v1`, which uses the measured state and the checkpoint's original normalization unchanged. `train_constant_compat_v1` now rejects these nonconstant statistics and remains available for the tiny-std legacy checkpoint. No checkpoint weights or normalization buffers were edited.

Tests compare AMP's original `predict(State53, Goal8)` with SuperMimic `forward_features` on identical fixed tensors (`rtol=atol=1e-6`), and verify q6/qd6 changes survive conversion and normalization. The old checkpoint's constant projection remains numerically reproducible.

## Dataset and Reset alignment

`scripts/audit_m1_v33_dataset.py` reads the raw Motion Bank at
`/home/yifeiy/AMP-LOCO-M1/dataset/m1_kinematic_motion_bank_v3_3_variable_l/`.
It found 999 L, 1000 M and 1273 LM motions (3272 total), with 923,892 frames. All 999 L initial frames use the same `nominal_high/m1_standing_pose_v2_3/carry` group:

- root z: exactly 0.470 m; roll, pitch and yaw: 0 rad;
- front hip/knee: `+0.881283 / -1.618489` rad;
- rear hip/knee: `-0.881283 / +1.618489` rad;
- all abad and wheel positions: 0 rad; initial wheel rates are effectively 0;
- Piper joints: `[0, +0.5, -0.4, +0.3, 0, 0]` rad;
- initial root linear velocity is effectively zero.

The raw audit reports P1/P50/P99 and min/max over all 999 L starting rows. Root z is `0.470000 m` at every percentile and both extrema; roll, pitch and yaw are all 0. Every one of the twelve leg position joints and all six Piper joints has identical P1/P50/P99/min/max, equal to the source values above. All four wheel position angles are 0. Their tiny numerical initial angular rates are bounded by −0.000876 to +0.000723 rad/s across motions, with P50 effectively 0. Initial root world vx has P1/P50/P99 `−0.000009/0/+0.000013 m/s` and min/max `−0.000025/+0.000028 m/s`; root vy has P1/P50/P99 `−0.000005/0/+0.000002 m/s` and min/max `−0.000015/+0.000003 m/s`.

The new profile `dataset_v3_3_l_high_carry` encodes those generator defaults and is selected for L-only training. M retains `dataset_carry` and its 0.3893 m initial profile. The M motion trajectories remain a separate task and were not globally reset to L's high stance.

V3.3 L effective command percentiles are vx `[-0.08949, 0, 0.14412]`, vy `[-0.05026, 0, 0.05677]`, and wz `[-0.17163, 0, 0.18156]` (P1/P50/P99). The first L run samples vx `[-0.09, 0.145]`, vy/wz zero. Thus it covers the observed forward, stop and small reverse range without training on the OOD ±0.3 m/s evaluation commands.

## CPU prior check

`scripts/audit_m1_v33_cpu_prior.py` evaluates the old CVAE with its old v2 contract, the old CVAE with the A+B v3 contract, and the new V3.3 CVAE with the matching high-carry v3 profile. These are kinematic reference comparisons, not task success measurements. No `vx` or yaw-rate motion was added by hand.

For the high static L history:

| vx Goal | Old CVAE root dx / mean wheel rate | V3.3 CVAE root dx / mean wheel rate |
|---:|---:|---:|
| -0.08 m/s | -0.0039 m / +0.022 rad/s | -0.0099 m / -0.535 rad/s |
| 0 | -0.0016 m / +0.027 rad/s | approximately 0 m / approximately 0 rad/s |
| +0.12 m/s | +0.0019 m / +0.035 rad/s | +0.0155 m / +0.858 rad/s |

The new prior produces nonzero, direction-consistent references over the ID range. With a real L middle-history input, zero goal can still show about 3.9 mm of predicted root drift over 0.2 s, so context motion must be distinguished from goal response. All predicted frames are marked `reference_not_dynamically_validated`; some high-speed/history cases exceed the configured velocity envelope and are counted by the script.

For the new Prior, q6/qd6 compatibility projection count is zero. The old Prior continues to report its four-frame constant projection. V2 versus v3 does not change a pure Prior decode; it changes how those references are converted into simulator setpoints. Model, normalization/input policy, L pose and action envelope all change in the V3.3 comparison, so it is not a single-variable ablation.

## Physical interface smoke check

With no other training process present, `scripts/validate_m1_v33_physics.py` ran the real mjlab/MuJoCo Warp environment on CUDA:0 for 12 envs × 50 control steps (1.0 simulated second), no observation/reset/domain/action noise, seed 20261103. The high Reset was exactly 0.470 m. MuJoCo contact pairs included wheel-related collision geoms against the terrain. The observed means were:

| Fixed command | Signed forward displacement | Body vx | Mean wheel reference / measured rate | Root z | Roll / pitch |
|---|---:|---:|---:|---:|---:|
| stop | -0.0047 m | -0.0147 m/s | -0.091 / -0.058 rad/s | 0.444 m | 0.00003 / 0.0044 rad |
| +0.12 m/s | +0.0246 m | +0.0601 m/s | +1.120 / +0.897 rad/s | 0.446 m | 0.00010 / 0.0072 rad |
| -0.08 m/s | -0.0254 m | -0.0727 m/s | -0.818 / -0.590 rad/s | 0.446 m | -0.0036 / 0.0056 rad |

Leg and Piper reference RMSEs were about 0.047–0.049 rad and 0.057–0.064 rad; mean actuator-force magnitude was about 5.4–5.6 in the environment's units. This short run verifies positive/negative wheel and body motion directions, physical contacts, high Reset and target mapping. It is not a sustained stability or learned-policy result.

## Training

The first 30-minute run used [`scripts/run_m1_v33_l_2048_30min.sh`](../scripts/run_m1_v33_l_2048_30min.sh), and its Checkpoints remain preserved. After the CUDA-mapped RNG restoration fix, the run resumed through it 3000 in [`logs/m1_prior_actor/m1_v33_L_2048_2h_resume90m_rngfix`](../logs/m1_prior_actor/m1_v33_L_2048_2h_resume90m_rngfix). The same-seed MuJoCo evaluation and CPU WM action-response audit failed forward/reverse direction checks, so the continuation used an Actor-only phase with the Prior frozen at it 3000. The user stopped the run at it 5150 to focus on analysis; the latest complete Checkpoint is it 5000. It used about 86.8 minutes of cumulative training across all segments. Final physical evaluations, same-Actor Prior swaps, and WM response audits are in [`M1_V33_L_2048_TRAINING_RESULTS.md`](M1_V33_L_2048_TRAINING_RESULTS.md).
