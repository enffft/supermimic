# M1 Action22 Reference V1 Implementation

## Scope and repository state

This change implements the M1 `Future31 -> Action22 reference -> Actor -> World Model` path for `m1_prior_joint`. It does not train a new CVAE and does not claim task-level readiness.

The active checkout remains on `main` at `c01ba1e916a0a31c76ed03bde50e4183baad1503`; no branch switch, commit, or push was made. The checkout already contained uncommitted Go2 work before this task. Those Go2 files and changes were retained. Because the V3.3 Action Contract and prior-input fixes required by this task were only present on the existing M1 integration branch, their M1-specific history and this V1 implementation were three-way merged into the current `main` working tree. The merge had no content conflicts; shared config/loop files retain the pre-existing Go2 hunks. AMP-LOCO-M1 data, CVAE weights, robot SDK limits, and URDF files were not modified.

## Data and action contracts

- CVAE input is `History[4,53] + Goal[8]`, using the V3.3 `legacy_v1` normalization. Future is `Future31[10,31]` at 20 ms per frame. The q6/qd6 training-constant projection remains available only for legacy checkpoints whose saved standard deviations are genuinely tiny; selecting that policy for moving q6/qd6 dimensions fails validation.
- The loaded prior is `cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt`, SHA256 `05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c`. Its source metadata and compatibility policy are saved in V4 checkpoints. `prior_initial` is a separately frozen copy for anchoring and comparison.
- Verified checkpoint metadata: feature schema `m1_cvae_motion_prior_v1_state53_goal8_future31`; condition `History[4,53]+Goal[8]`; target `Future[10,31]` at `0.02 s/frame`; latent/hidden/layers `32/256/3`; step `9600`; dataset/split/normalization hashes `95f9f0a39b05d5a0d736532d24f897b3b85b8cbf8f74977de620acc9b1a7146e`, `b1bb36beb45b141a77f28837df376690abc3e1d6b24dd810d5ee07da40fd2624`, `0dea5b4cac9a7297ad27e1e4d4192211a6f7af7cd138c73046b2ed7c80dd904f`; assurance `mixed_kinematic_and_sampled_fcl`, `strict_fcl_only=false`. The q6/qd6 normalization has nontrivial standard deviations (about `0.370–0.373` and `0.272`), so V3.3 uses real state features rather than mean projection.
- `Future31[0:3]` is root-relative translation, `[3:9]` Rotation6D, `[9:21]` the 12 leg positions in AMP name order, `[21:27]` Piper positions 1–6, and `[27:31]` wheel rates ordered FAR/FBL/RAR/RBL. The single adapter maps positions by joint name, fills an M1 physical target, and calls the existing inverse V3 map.
- Action22 follows `M1_JOINTS`: FAR abad/hip/knee/wheel, FBL abad/hip/knee/wheel, RAR abad/hip/knee/wheel, RBL abad/hip/knee/wheel, then Piper joints 1–6. Each leg/Piper position value in Future31 is normalized through the saved asymmetric interval (negative action interpolates lower bound to default; positive action interpolates default to upper bound). The four wheel rates are divided by their saved per-wheel scale. The inverse map exposes unclipped actions for diagnostics/losses; only its execution copy is clamped to `[-1,1]`. `decode_future31` remains the independent state-reference decoder.
- M1 L uses the V3.3 `dataset_v3_3_l_high_carry` reset profile. The default command support is `vx=[-0.09,0.145] m/s`, `vy=[-0.05,0.055] m/s`, `wz=[-0.17,0.18] rad/s`; L sampling allocates 15% stop, 35% reverse, and 50% forward. This does not assert reachability across the whole range.

## Actor, rollouts, objectives, and phases

The V4 Actor schema is `m1_action_reference_actor_v1`: term-major proprio history `[B, history*72]` (history defaults to 5), `Goal8[B,8]`, and only the current `u_ref[k][B,22]` go in; it returns `Action22[B,22]`. It has no Future31 input, phase input, or privileged State57 input. Its final residual layer starts at zero and adds a bounded residual around the current reference.

The differentiable rollout uses the same real Replay start `State57`, the same `xi`, and the same action clamp for both paths. The direct path is `U_ref[k] -> WM`; the Actor path is `proprio/Goal/U_ref[k] -> Actor -> executed action -> WM`, with imagined proprio history updated each step. `Future31[k]` controls interval `k -> k+1`; cadence is 10, and reset/material Goal changes replan. The previous-action observation retains the raw Actor output; WM and Replay's `a_exec` use the final bounded command.

Default Actor objective:

```text
L_actor = tracking + 0.05 * BC + 1.0 * task + 1.0 * stability + 0.01 * control
tracking = 0.5 * root_position + 0.25 * root_rotation + 0.25 * leg_position
         + 1.0 * arm_position + 0.5 * wheel_velocity
```

Tracking terms are normalized by physical scales. Task L uses signed body-frame velocity/yaw errors; Task M uses the existing `piper_x_link7` FK, terminal/path distance, progress, and base drift. Stability includes minimum height, tilt, and soft joint bounds. Control is an action-rate / normalized joint-speed / acceleration / saturation proxy, not an actuator energy estimate.

Default Prior objective:

```text
L_prior = 1.0 * direct_task + 0.25 * actor_task + 0.5 * physical_dynamics
        + 1.0 * feasibility + 0.1 * out_of_envelope
        + 0.0001 * smoothness + 0.02 * grouped_anchor
```

The anchor logs root/leg/arm/wheel separately. Energy and the control-effort surrogate are disabled by default. Actor and Prior updates freeze the other model parameters without severing input gradients. Prior learning defaults to frozen (`freeze_prior=true`, `prior_updates_per_event=0`). Its gate requires a matching held-out MuJoCo/WM action-response report, matching WM hash and Action Contract, at least 128 held-out states, Actor readiness, and an explicit unfreeze setting. Uncalibrated thresholds and missing Actor/task-safety review fail closed.

Training progresses from `wm_warmup` to `actor_prior_frozen`; only externally evidenced calibration can enter `prior_calibration`, followed by `actor_adaptation` and explicit `validation`. M1 collection exploration is explicit, bounded, group-specific (leg 0.02, Piper 0.015, wheel 0.05 normalized-action std on 25% of environments, with decay), and applied before `env.step`. Internal M1 action noise is rejected in this path. Recorder retains raw policy action for the observation contract and writes the final clamped executed Action22 as `a_exec` for WM fitting.

## Checkpoints and compatibility

New training checkpoints use `supermimic_m1_action22_reference_v1`, format version 4. They save Actor input schema, adapter schema, prior path/hash/metadata/input policy, `prior_initial`, WM and hash, action map and pose profile, task sampler/command ranges, loss weights, optimizer states and step counters, training phase/gate evidence, exploration settings, and RNG state. Resume validates each saved contract and the loaded WM hash before continuing.

V2/V3 Future31-conditioned Actor checkpoints are rejected for V4 resume with an explicit schema error. The old Actor is retained under `LegacyM1FutureConditionedActor` solely for explicit read-only historical evaluation; it is not used by the new training path. The old 2048 launchers now stop immediately with an archived-script message so they cannot silently start incompatible resumes. No source files were deleted in this change because the legacy Actor still backs historical read-only evaluation and the audit tools remain useful.

## Validation performed

Using the `supermimic` Conda environment:

```bash
nice -n 10 env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/pytest -q
```

Result on the final `main` working tree: **135 passed, 64 warnings in 70.38 s**. The run covered the M1 Action22 adapter/round-trip, strict Actor signature, 3/10-step dual rollouts, gradient ownership, gate fail-closed behavior, V4 save/load mismatch cases, V3.3 prior compatibility, and G1/Go2 core regression, including the pre-existing local Go2 tests. No tests were skipped.

`python -m compileall -q src/supermimic scripts tests` and `git diff --check` also passed on the final working tree.

The V3.3 direct-reference physical smoke was run on `cuda:0` (3 environments, 10 control steps per command; 0.2 s per trace):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/validate_m1_v33_physics.py \
  --num-envs 3 --steps 10 --device cuda:0 --out /tmp/m1_action22_v1_physics_smoke.json
```

The loaded SHA matched the expected V3.3 SHA; `legacy_v1`, `dataset_v3_3_l_high_carry`, and Action Contract v3 were active. Reset was 0.470 m with the expected front hip/knee (`+0.8813/-1.6185 rad`), and the initial state had 8 detected wheel-ground contacts per environment. Stop / forward `+0.12` / backward `-0.08 m/s` produced signed body `vx` of `+0.0040 / +0.0226 / -0.0094 m/s` and signed forward displacement `+0.00053 / +0.00160 / -0.00025 m`; action and target saturation were zero, and the inverse/physical position-target map RMSE was below `5e-11 rad`. Wheel-rate RMSE remained about `1.40–1.49 rad/s`. This checks reset, signs, ordering, and setpoint conversion over a 0.2 s probe only. It does **not** establish stable walking, Actor success, WM action-response accuracy, or task readiness.

No training job was started. The 2048-env/30-minute-or-longer training request is explicitly excluded by the task instructions. The new fixed evaluator and held-out response audit were not run because no V4 training checkpoint exists. The response audit intentionally cannot pass without calibrated thresholds and a separate Actor/task-safety review.

## Safe, bounded commands for a later smoke run

These are prepared commands, not executed results. They use the CLI names checked with the installed `sm-train --help`, the V3.3 prior explicitly, and a unique run name. The short training command caps its wall time at 0.03 h and uses 8 environments; check GPU availability before running it.

```bash
cd /home/yifeiy/ws_supermimic
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name m1_action22_v1_L_smoke_20261103 \
  --log-root logs/m1_prior_actor --seed 20261103 --device cuda:0 \
  --env.robot-name m1_piper --env.task-name m1_prior_joint --env.num-envs 8 \
  --env.history-length 5 --env.m1-pose-profile dataset_v3_3_l_high_carry \
  --env.m1-action-contract v3 --env.m1-joint-task-type L \
  --env.m1-joint-vx-range -0.09 0.145 --env.m1-joint-vx-stop-fraction 0.15 \
  --env.m1-joint-vx-reverse-fraction 0.35 --env.m1-joint-vy-range 0 0 \
  --env.m1-joint-wz-range 0 0 --env.no-obs-noise --env.no-reset-noise \
  --env.no-domain-randomization --env.action-noise-std 0 \
  --m1-joint-training.enabled \
  --m1-joint-training.prior-checkpoint /home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_v3_3_mixed_3272_goal_l_holdout_20261103/checkpoint_best_val.pt \
  --m1-joint-training.expected-prior-sha256 05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c \
  --m1-joint-training.prior-input-compatibility legacy_v1 \
  --m1-joint-training.task-type L --m1-joint-training.state-source privileged \
  --m1-joint-training.prior-mode prior_mean \
  --m1-joint-training.training-mode staged_action22_v1 \
  --m1-joint-training.prior-updates-per-event 0 \
  --m1-joint-training.freeze-prior --m1-joint-training.exploration-enabled \
  --m1-joint-training.exploration-env-fraction 0.25 \
  --m1-joint-training.exploration-leg-std 0.02 \
  --m1-joint-training.exploration-arm-std 0.015 \
  --m1-joint-training.exploration-wheel-std 0.05 \
  --m1-joint-training.exploration-decay-end-fraction 0.75 \
  --m1-joint-training.batch-size 32 --m1-joint-training.micro-batch-size 8 \
  --m1-joint-training.actor-pretrain-updates 384 \
  --schedule.total-iterations 2000 --schedule.m1-wm-only-iterations 20 \
  --schedule.log-interval 10 --schedule.checkpoint-interval 100 --schedule.max-hours 0.03
```

After such a run creates a V4 checkpoint, fixed-seed evaluation and the held-out action-response audit can be invoked with:

```bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/evaluate_m1_prior_joint.py \
  --checkpoint logs/m1_prior_actor/m1_action22_v1_L_smoke_20261103/ckpt_latest.pt \
  --task-type L --prior-variant trained --num-envs 6 --steps 500 \
  --seed 20261103 --device cuda:0 --controller actor \
  --out logs/m1_prior_actor/m1_action22_v1_L_smoke_20261103/eval_actor.json

PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/audit_m1_action22_real_response.py \
  --checkpoint logs/m1_prior_actor/m1_action22_v1_L_smoke_20261103/ckpt_latest.pt \
  --state-count 128 --seed 20261103 --device cuda:0 \
  --out logs/m1_prior_actor/m1_action22_v1_L_smoke_20261103/action_response.json
```

The response audit remains fail-closed unless calibrated direction/magnitude thresholds are supplied and the saved Actor task/safety results are reviewed and included.

## Before a longer run

1. Run the complete CPU suite and compile check on the final working tree.
2. Confirm GPU memory/utilization is available without affecting existing G1/Go2 work.
3. Create a new V4 checkpoint, perform save/load equality and all strict mismatch rejection checks.
4. Run fixed-seed Actor and Prior-direct evaluations; report signed body velocity, displacement, fall/survival, root height/tilt, saturation, and wheel/leg/Piper tracking separately.
5. Calibrate and run held-out MuJoCo-vs-WM action perturbations from matched states. Do not unfreeze Prior until the action-response, Actor task, and real-task safety evidence matches the current WM hash and Action Contract.
