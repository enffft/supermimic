# SuperMimic Phase 1: G1, Go2, M1-Piper

## Scope and baseline

The local baseline at the start of this work was `cf4763a15aafb200cf39a38679fe73e6c0abe3dd`, ahead of the stated upstream baseline `1d9fabfd52d9e01a36111ea1a34fac2d9b28ad6e`. Existing user changes to `docs/4070_LAFAN1_Q1_TRAIN.md`, `scripts/play.py`, `.vscode/settings.json`, and `docs/SUPERMIMIC_CODE_WALKTHROUGH.md` were retained. No reference repository files were copied into this checkout.

The pinned source trees were cloned under `/tmp` and read at the requested revisions:

| Project | Revision | Relevant source read | Difference from local code | Phase 1 decision |
|---|---|---|---|---|
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) | `23c76cad5bf41110709c7f929b54c1da1be6b248` | `models/normalization.py`, `models/world_model.py`, `models/policy.py`, `rl/online_training.py`, `rl/imagination.py`, `config/rl_cfg.py` | It has actor/critic input normalization, WM normalization and a 6D body-orientation anchor. Local WM previously used fixed StateSpec scales only; G1 had a 9D position+orientation anchor implementation but disabled it by default. | Added optional WM input and delta RunningNorm and an orientation-only anchor option. Kept fixed scales, model widths and the old default G1 inputs. The paper's reported return increments are not treated as isolated feature effects or reproduced results. |
| [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) | `d9eee9b3f248e5300e636c433255db2c1c3e086a` | `robot/go2.py`, `supermimic/{train,world_model,state,imagine,buffer}.py`, `env/{env_cfg,mdp}.py` | The reference trains chase-and-dodge with perception and obstacle context; it is not a blind velocity-command task. Local Go2 already had a direct command path, but policy actions and imagined actions differed, reward allowed rewarding near-zero motion, and termination learning had produced bad imagination survival. | Borrowed the direct proprioception-plus-command input, online real-transition WM data, noise semantics, analytic height/tilt survival and staged actor training. Did not copy chase rewards, ball/obstacle goals, LiDAR, elevation scans or environment generators. Audited local URDF before setting actuator limits. |
| [emoPointer/supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation) | `810e5768c50117d2fec75f20f9e643a11cd0882b` | `algorithms/supermimic/{agent,runner}.py`, `tasks/{supermimic,lemon/supermimic}.py` | Reference separates task goal/reward/success behavior from the shared model-based learner. Local trainer had only Motion and a broad non-Motion branch. | Added a small `WorldPointReachTask` adapter through existing Replay, WM and imagination. Did not bring over grasp/contact approximations, multi-stage object tasks, reset machinery or an ensemble. |

## Implemented model and task contracts

### Shared World Model normalization

`StateSpec.canonical(s)` still applies the existing fixed feature scales. When `world_model.running_normalization=true`, the input normalizer then computes

```text
x_norm = (canonical(s) - mean_x) / max(sqrt(var_x), min_std)
```

The dynamics head predicts normalized pre-`out_scale` delta coordinates. The output is restored as

```text
delta = delta_norm * max(sqrt(var_delta), min_std) + mean_delta
```

and the original `StateSpec.integrate(s, delta)` applies `out_scale`, `dt`, and the existing integration rule once. `StateSpec.delta_target` inverts that integration to obtain exactly the pre-`out_scale` target for statistics. A new-transition callback updates moments once during real Replay writes; replay sampling, WM fitting and imagination do not update them. Means, variances and counts are module buffers in the WM checkpoint. The feature is off by default, so older checkpoints keep the original fixed-scale forward path and state-dict keys.

### G1 motion tracking

| Item | G1 contract |
|---|---|
| State | Existing 71D physical state for 29 joints; fixed StateSpec and integration unchanged. |
| Actor observation | Existing 930D proprioception history (`29*3+6`, ten frames). No global root XYZ is added. |
| Goal | Existing ten 68D motion frames. Default goal remains 680D. Optional orientation-only anchor appends 6D; legacy position+orientation anchor appends 9D. |
| Action | Existing 29D motion action. |
| Reward / termination | Existing tracking reward and learned reference termination remain unchanged. |
| Checkpoint behavior | Defaults preserve the old network dimensions. New normalization and anchor are opt-in and recorded in config/metadata. |

The real Goal and imagined Goal use the same anchor helper and mode. New orientation-only observations use `motion_anchor_ori_b`; the existing 9D `anchor_relative_obs` path remains intact.

### Go2 blind velocity command

| Item | Go2 contract |
|---|---|
| State | Existing 37D World Model state (12 joints), 8-step WM windows and 32-step default imagination. |
| Actor observation | 420D proprioception history plus 3D `[vx, vy, wz]` command, for 423D direct input. No measured base linear velocity is exposed to the Actor. |
| Goal / Replay | 3D body-frame velocity command stored per real transition. The WM dynamics do not take the command as input. The Critic still receives the command. |
| Action | 12D policy action; new Direct actors use `tanh` to stay in the physical `[-1,1]` support, then real collection adds configured Gaussian/OU noise and clips the executed action before mapping it to `default_joint_pos + action_scale * executed_action`. Replay records policy and executed actions. Imagination clamps to the same zero-noise deployment action range before WM dynamics. Action-rate uses policy actions; the torque proxy uses the executed action. Older checkpoints without `action_output` keep identity output when loaded. |
| Reward | Real and imagined paths share `velocity_reward_terms`; tracking remains bounded, command-direction feedback is signed and bounded for forward/backward/sideways commands, and the direction term is zero at a zero command. Yaw uses the same velocity-error tracking style. RewardManager and imagination apply `dt` once. Real torque is measured; imagination uses the configured PD/effort-limited proxy and is logged/documented as an approximation. |
| Termination | Real hard failure uses absolute base height and the same gravity-axis tilt limit. Imagination uses differentiable sigmoid probabilities from those same margins. Timeout remains separate; survival is the cumulative product of one-minus-soft-termination. |
| Noise | Gaussian standard deviation `0.18`; 30% OU environments; OU diffusion `0.60` and theta `2.0`. With `dX=-theta X dt + sigma sqrt(dt) noise`, the stationary OU standard deviation is `sigma/sqrt(2 theta) ~= 0.30`. |

The local Go2 URDF joint order matches `RobotSpec`: FL, FR, RL, RR with hip/thigh/calf per leg. URDF effort limits are `23.7 Nm` for hip/thigh and `35.55 Nm` for calf. The previous uniform `23.5 Nm` proxy clipped knee torque too low, so the model uses a named actuator version `go2_urdf_builtin_pd_kp40_kd1_effort_23p7_35p55_v2`. It keeps the local BuiltinPositionActuator PD controller and uses its same `kp=40`, `kd=1`, action scale `0.25`, and per-joint effort limits in the imagined torque proxy. The reference's DC motor parameters were not copied because this checkout uses a different URDF and controller.

The audited zero-action standing state settles near `0.283 m` root height from the `0.42 m` reset height. The absolute hard failure threshold remains `0.17 m`; `0.22 m` is only the soft low-height reward margin. Feet use the existing per-foot collision geometry and friction setup; no perception sensor is attached.

### M1-Piper world point reach

| Item | M1 contract |
|---|---|
| State | 57D physical state for all 22 joints (18 position-controlled plus four wheel joints). |
| Actor observation | Existing 720D, ten-frame proprioception history plus 3D body-relative target (`723D` direct input). No Motion Library or VelocityCommand curriculum is used. |
| Goal / coordinates | Episode target is sampled and stored as env-local inertial-world XYZ after MuJoCo has applied and forwarded the reset pose. Actor Goal is recomputed each step as `rotate_world_to_body(target_local_world - base_local_world, base_orientation)`. Replay stores the fixed target; imagination holds it fixed across the rollout and recomputes the body-relative Goal from each predicted state. |
| Action | Existing 22D `MixedM1Action`: position targets for 18 joints and velocity targets for four wheels. |
| TCP | The URDF has no calibrated TCP frame. Phase 1 uses the midpoint of fixed `piper_x_link7` and `piper_x_link8` finger frames as a gripper-center proxy. Differentiable FK matched the MuJoCo midpoint to max error `3.66e-7 m` in the initial-pose smoke check. This is not a calibrated grasp TCP. |
| Reward | Shared pure function for real reward and imagination: continuous negative TCP distance, signed bounded approach progress, near-target exponential reward, upright/base-height, soft joint limits, action-rate and excessive TCP-speed terms. The target remains fixed in env-local world coordinates. |
| Termination / success | Hard height/tilt failure and separate timeout. Evaluation success requires distance `<0.08 m` held for `0.5 s` without a fall. Targets mix near and farther front/side positions; farther targets are intended to require whole-body movement, and evaluation reports base displacement and arm motion separately. |

The sampled offsets are applied at the initial gripper-center proxy and rotated by the reset base orientation into env-local world coordinates. mjlab resets command terms before writing and forwarding the reset pose, so `WorldPointCommand.compute()` resamples only reset environments after that forward pass. Near offsets are `x=0.05..0.30 m`, `y=-0.20..0.20 m`, `z=-0.15..0.15 m`; far offsets are `x=1.00..1.40 m`, `y=-0.40..0.40 m`, `z=-0.10..0.25 m`. A 3000-sample arm-joint FK sweep with the base fixed measured maximum forward TCP position near `0.93 m` from base origin, so the forward far band extends beyond this sampled arm-only envelope. Policy success and actual base use still require real evaluation.

## Shared trainer, checkpoint and compatibility changes

`training/loop.py` now validates and routes the explicit task names `motion_tracking`, `velocity_command`, and `world_point_reach`. The same Replay, WM trainer, policy trainer, Critic, optimizers and imagination code remain shared. Actor BPTT freezes WM and Critic parameters while retaining state gradients through the WM, then restores their trainable state. Actor and Critic norm clipping reject nonfinite gradients. `PolicyTrainerCfg.critic_value_clip` optionally clips values used for Actor bootstrap and GAE; it defaults to `None` so task-specific return scales are not forced into one range. `init_wm_from` is mutually exclusive with `resume`; it loads only compatible WM weights and normalizer buffers, leaving Actor, Critic, optimizers, Replay and counters fresh. Metadata records robot, task, joint order, goal/action/state dimensions, Actor encoding, actuator version/model hash, termination mode, and the complete config including normalizer settings.

The old Go2 checkpoints found in `logs/` do not contain actuator-version metadata and were produced before the per-joint calf torque correction. They are therefore rejected for WM initialization and full resume. `scripts/play.py` reconstructs a legacy encoded Go2 Actor from the checkpoint's saved `PolicyCfg` and loads it strictly, while warning that it is being evaluated on changed physics. Its World Model is not considered compatible with the corrected actuator model. New Direct Go2 checkpoints use the default `direct` encoding and strict state-dict loads; no `strict=False` fallback is used.

This phase also leaves Actor/Critic running observation statistics and prioritized contact/fall windows out of the implementation. Policy inputs retain their existing fixed per-term scales for checkpoint compatibility; WM statistics are the optional normalization feature specified here. Replay still draws valid same-episode windows uniformly. A separate rare-event sampler is deferred until real contact/fall coverage and per-state WM errors justify its added sampling rules.

## Changed files and entry points

| Files | Key implementation points |
|---|---|
| `src/supermimic/config.py`, `src/supermimic/training/loop.py`, `src/supermimic/policy/nets.py` | Adds explicit task/robot validation, task Goal dimensions, Direct/Encoded policy selection, bounded Direct Actor outputs, WM initialization/resume checks, model metadata, and task-specific trainer construction. |
| `src/supermimic/world_model/normalization.py`, `world_model/world_model.py`, `world_model/state_spec.py` | `RunningNorm.update/normalize/denormalize`, `WorldModel.update_real_transition/step`, and `StateSpec.delta_target` implement real-transition-only statistics and unit-correct inverse scaling. |
| `src/supermimic/data/recorder.py`, `data/replay_buffer.py` | Replay stores policy and executed actions plus fixed task Goals; the Recorder callback updates optional WM statistics once per new physical transition. |
| `src/supermimic/world_model/heads.py`, `src/supermimic/envs/mdp/observations.py`, `envs/tracking_env_cfg.py` | Adds the orientation-only Anchor helper and its real observation path while retaining the existing 9D helper. |
| `src/supermimic/robots/spec.py`, `robots/mjlab_assets.py`, `envs/actions.py`, `tasks/action_contract.py` | Centralizes Go2 effort limits/model version and exposes policy/executed/physical action mappings. |
| `src/supermimic/tasks/velocity_reward.py`, `imagination/tasks.py`, `imagination/compiled.py`, `imagination/rollout.py` | Shared Go2 reward and analytic survival, clamped imagined deployment actions, and task-based imagined Goal/Reward dispatch. |
| `src/supermimic/envs/mdp/commands.py`, `envs/multirobot_env_cfg.py`, `tasks/world_point_reward.py`, `imagination/tasks.py` | Implements fixed env-local M1 target sampling after reset forwarding, body-relative Actor Goal, debug marker/path, differentiable FK reward and analytic task termination. |
| `src/supermimic/imagination/policy_trainer.py`, `world_model/wm_trainer.py` | Adds task-aware imagination, optional critic bootstrap clipping, and nonfinite-gradient rejection after effective gradient clipping. |
| `scripts/play.py`, `scripts/evaluate_go2_velocity.py`, `scripts/evaluate_m1_world_point.py` | Checkpoint-driven Play, fixed Go2 command evaluation, and M1 reach/success-hold evaluation. |
| `tests/test_action_contract.py`, `tests/test_world_model_normalization.py`, `tests/test_world_point_reach.py`, `tests/test_velocity_training.py` | Unit checks for action clipping/setpoints, normalized dynamics units/gradients, M1 Goal/FK/action gradients, and Go2 reward/termination. |

`src/supermimic/policy/nets.py` was not changed: its existing encoded and direct interfaces already support the selected task configurations, so G1/M1 Motion retain their current encoder and Go2/M1 point use the existing direct Decoder path.

## Verification completed

- Pinned reference source revisions were cloned and inspected under `/tmp`.
- Final full local test suite, including large-grid M1 reset-coordinate coverage, Direct Actor `tanh` gradient coverage, and G1 6D Anchor checks: `58 passed, 63 warnings` in `67.13 s`. Warnings were upstream PyTorch deprecations and the installed CUDA 12.2 driver warning; there were no test failures.
- Go2 zero-action physical check, 16 environments / 150 steps: stable root height `0.283167 m`, no falls, correct 12-joint order, replay populated, measured actuator force within the new URDF limits.
- Go2 trainer smoke, 8 environments / 18 iterations: 420D observation plus 3D Goal, Replay collected, WM and Actor updates ran; after the first updates the 32-step imagined survival diagnostic was about `0.94`. The initial 8-step WM NSE was `2.67`, which is expectedly too high for a meaningful policy result and supports retaining the WM-only warmup.
- M1 world point environment smoke, 8 environments: 720D observation plus 3D Goal, fixed target stayed unchanged for the episode, 22D action stepped, no initial termination, Replay recorded. FK vs MuJoCo gripper-center proxy max error was `3.66e-7 m` at the checked pose.
- Existing latest G1 checkpoint `logs/4070/lafan1_anchor_1024_8h_resume02_20261026/ckpt_latest.pt` was restored with strict Policy and WM state-dict loads: iteration `211050`, 53,988 Actor updates, 689D encoded goal with legacy 9D Anchor, and no RunningNorm. Its recorded 40-clip real benchmark is mean normalized return `0.4297`, step-weighted `0.4325`, one failure. Per-category means: walk `0.4779`, run `0.3513`, sprint `0.4560`, dance `0.4254`, jumps `0.4333`, fight `0.4118`, fall/get-up `0.3957`. The 4-hour earlier checkpoint scored `0.4135` with two failures, so the later 8-hour run improved modestly. In the latest benchmark, body-orientation reward was only about `1.0e-5` per step versus a `0.02` maximum for unit-weight terms; run and fall/get-up remain the clearest gaps. This is existing user training evidence, not a result from the new implementation.
- Historical pre-fix real Go2 evaluation in the workspace showed near-zero commanded motion: for `vx=0.3`, mean `vx ~= 0.0063 m/s`, RMSE `~=0.295 m/s`; for `vx=0.5`, mean `vx ~= 0.0063 m/s`, RMSE `~=0.495 m/s`. Historical diagnostics also showed a learned termination probability near `0.55` per step and vanishing 32-step survival. These are the reason for the reward/action/termination changes, not results from the new code.
- **Go2 bounded training:** `logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt`, 3072 environments, batch 1024, 15-minute wall-clock cap; stopped at iteration `1800`, with `704` WM updates and `504` Actor updates. Final WM NSE was about `0.010` one-step and `0.019` eight-step; 32-step imagination survival `0.990`; training fall rate `0`. The held-out real MuJoCo fixed-command evaluation found `vx=0.3`: mean `vx=0.0031 m/s`, RMSE `0.2975 m/s`, displacement `0.0335 m`, action saturation `14.5%`; `vx=0.5`: mean `vx=0.0031 m/s`, RMSE `0.4973 m/s`, displacement `0.0338 m`, saturation `21.3%`. Both commands had zero falls, timeout rate `100%`, and mean height about `0.293 m`. The Go2 model stayed standing and did **not** learn velocity tracking. The bounded checkpoint predates the final Direct-Actor `tanh` output mapping; its action mapping is tagged `identity` and cannot be resumed into the new mapping.
- **M1 bounded training:** `logs/m1_world_point_phase1_smoke_128_retry1_20261027/ckpt_latest.pt`, 128 environments, 1200 iterations, 474 total updates and 454 Actor updates. Final WM NSE was about `0.011` one-step / `0.042` eight-step, but action magnitude averaged `1.47` and imagined survival was `0.305`. This run used the previous unbounded Direct output. Real evaluation was rerun after correcting reset-time goal sampling: 128 episodes had mean initial TCP error `0.537 m`, mean episode error `0.821 m`, final error `0.993 m`, success rate `0%` at `<0.08 m` for `0.5 s`, falls `0%`, timeout `100%`, mean base displacement `0.543 m`, and arm motion `0.486 rad` (joint-space norm). Near targets averaged `0.221 m` initially; farther targets averaged `1.231 m`; neither group succeeded. FK-to-MuJoCo TCP proxy error was `5.87e-7 m`. This is a functioning training/evaluation path, but not a learned reaching policy.
- **G1 compatibility and 12-hour run diagnosis:** the latest existing checkpoint strictly restores with its legacy 9D Anchor and 689D goal. Its recorded 40-clip score was `0.4297` mean normalized return (`0.4325` step-weighted) with one failed clip, a modest increase from the prior checkpoint's `0.4135` and two failures. The saved run ended at global iteration `211050` of a `320000` cosine horizon, so the 12-hour total did not finish the planned schedule. Fast running and fall/get-up remain weak; body-orientation reward was near `1e-5` per step against `0.02` unit-weight maximum. WM NSE/survival were healthy, but training and imagination returns did not predict real tracking quality. This points to a task/reward/data-coverage plateau, not a failed optimizer or missing 12 hours of execution.
- The Direct Actor `tanh` mapping and post-forward M1 target resampling were added after the bounded training checkpoints above. They passed policy-gradient and environment reset-coordinate tests, but neither change has yet been validated by a new physical training run. For Go2, the next single targeted training is to initialize a fresh Actor/Critic from the corrected-physics WM in `ckpt_latest.pt` via `--init-wm-from` (default Direct `tanh`) and judge it again with the fixed `vx=0.3/0.5` evaluator. Do not resume that checkpoint into the new action mapping; its saved Direct Actor used identity output. For M1, use a fresh `world_point_reach` run with default bounded Direct output and corrected reset sampling rather than resuming its old unbounded Actor.

## Commands checked against the current Tyro CLI

The examples assume execution from the repository root in the `supermimic` Conda environment. Full training keeps the configured cosine schedule tied to `320000` iterations. Use `--schedule.max-hours` only as a wall-clock stop; it does not shorten the LR decay horizon.

### 1. G1 original Motion Tracking configuration

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name g1_lafan1_original \
  --env.robot-name g1 --env.task-name motion_tracking \
  --data.npz-dir data/npz/lafan1 \
  --schedule.total-iterations 320000
```

### 2. G1 with orientation-only anchor and WM RunningNorm

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name g1_lafan1_anchor6_norm \
  --env.robot-name g1 --env.task-name motion_tracking \
  --env.anchor-orientation-obs \
  --world-model.running-normalization \
  --data.npz-dir data/npz/lafan1 \
  --schedule.total-iterations 320000
```

This creates a new model with a 686D Motion Goal; it does not change the old 680D checkpoint architecture. `--policy-trainer.critic-value-clip` is available for task-specific use if measured bootstrap values become unstable; it is intentionally unset in the examples.

### G1 Play for the latest existing checkpoint

```bash
conda run -n supermimic python scripts/play.py \
  --ckpt logs/4070/lafan1_anchor_1024_8h_resume02_20261026/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --clips walk1_subject1 \
  --num-envs 1 --viewer native
```

The checkpoint stores its 9D Anchor setting, so Play rebuilds the matching 689D goal. Use another LAFAN1 clip name from `data/npz/lafan1` to inspect a different motion.

### G1 continue from the latest checkpoint for up to 12 more hours

The current checkpoint is at global iteration `211050`; Replay is not stored, so set warmup to `211090` to collect 40 fresh vectorized transitions. The original schedule reaches its LR floor at global step `250000` (`320000 * 0.78125`). The command below keeps that same global decay endpoint and raises the iteration cap so the 12-hour wall-clock limit, rather than the old `320000` cap, is expected to stop the run. At the previously recorded ~`4.7` iterations/second this should take about `12` hours and reach roughly `414k`; higher throughput can still hit the `450000` cap first.

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name lafan1_anchor_1024_resume12h_20261027 \
  --log-root logs/4070 --seed 0 \
  --resume logs/4070/lafan1_anchor_1024_8h_resume02_20261026/ckpt_latest.pt \
  --data.npz-dir data/npz/lafan1 \
  --env.robot-name g1 --env.task-name motion_tracking \
  --env.anchor-obs --env.num-envs 1024 --env.replay-capacity 256 \
  --policy-trainer.batch-size 1024 --policy-trainer.horizon 32 \
  --wm-trainer.window 8 --wm-trainer.batch-size 1024 \
  --wm-trainer.batches-per-update 10 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.policy-updates-per-wm-update 1 \
  --schedule.warmup-iterations 211090 \
  --schedule.total-iterations 450000 --schedule.max-hours 12 \
  --schedule.lr-schedule cosine --schedule.lr-final-ratio 0.03 \
  --schedule.lr-decay-fraction 0.5555555556 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 4000
```

`450000 * 0.5555555556` preserves the existing `250000`-step cosine decay endpoint. If you want only to finish the original schedule, set `total-iterations=320000` and `lr-decay-fraction=0.78125`; at the recorded throughput that run will end around `6.4` hours, before the 12-hour cap.

### 3. Go2 Direct Velocity on flat ground

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name go2_velocity_direct \
  --env.robot-name go2 --env.task-name velocity_command \
  --env.num-envs 3072 \
  --env.velocity-wm-batch-size 1024 --env.velocity-actor-batch-size 1024 \
  --schedule.total-iterations 320000 \
  --schedule.warmup-iterations 40 --schedule.wm-only-iterations 200 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.lr-schedule cosine --schedule.lr-decay-fraction 0.78125
```

### 4. Start a fresh Go2 learner from a compatible trained WM

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name go2_velocity_new_actor_from_wm \
  --env.robot-name go2 --env.task-name velocity_command \
  --env.num-envs 3072 \
  --env.velocity-wm-batch-size 1024 --env.velocity-actor-batch-size 1024 \
  --init-wm-from logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt \
  --schedule.total-iterations 320000 \
  --schedule.warmup-iterations 40 --schedule.updates-per-env-step 0.4
```

The source checkpoint must have the same Go2 URDF fingerprint, actuator version, StateSpec, WM widths, normalizer configuration, and control timestep. The pre-correction checkpoints in the workspace do not qualify.

### 5. Resume the Go2 learner

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name go2_velocity_resume \
  --env.robot-name go2 --env.task-name velocity_command \
  --env.num-envs 3072 \
  --env.velocity-wm-batch-size 1024 --env.velocity-actor-batch-size 1024 \
  --env.direct-action-output identity \
  --resume logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt \
  --schedule.total-iterations 320000
```

This checkpoint predates bounded Direct outputs, so the explicit identity mapping is required to match it. Resume restores optimizers, counters and task/action state. Omit the override when resuming a checkpoint saved with the default `tanh` mapping. Keep the environment and actuator model compatible.

### 6. Go2 fixed-command evaluation and interactive Play

Headless evaluation tests both required commands and reports real MuJoCo velocity, RMSE, base displacement, falls, height, action amplitude and saturation:

```bash
conda run -n supermimic python scripts/evaluate_go2_velocity.py \
  --checkpoint logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt \
  --num-envs 64 --episodes-per-command 64 \
  --output logs/go2_velocity_phase1_3072_b1024_20261027/fixed_velocity_eval.json
```

Interactive visualization for either fixed command:

```bash
conda run -n supermimic python scripts/play.py \
  --ckpt logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt \
  --vx 0.3 --vy 0 --yaw-rate 0

conda run -n supermimic python scripts/play.py \
  --ckpt logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt \
  --vx 0.5 --vy 0 --yaw-rate 0
```

### 7. M1 `world_point_reach` training

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name m1_world_point_reach_256 \
  --env.robot-name m1_piper --env.task-name world_point_reach \
  --env.num-envs 256 \
  --env.m1-world-point-wm-batch-size 256 \
  --env.m1-world-point-actor-batch-size 256 \
  --schedule.total-iterations 320000 \
  --schedule.m1-wm-only-iterations 200 \
  --schedule.updates-per-env-step 0.4
```

### 8. M1 real evaluation and Play

```bash
conda run -n supermimic python scripts/evaluate_m1_world_point.py \
  --checkpoint logs/m1_world_point_reach_256/ckpt_latest.pt \
  --num-envs 128 --episodes 256

conda run -n supermimic python scripts/play.py \
  --ckpt logs/m1_world_point_reach_256/ckpt_latest.pt --num-envs 1
```

### 9. M1 original six Motion Tracking clips

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name m1_motion_set_c_regression \
  --env.robot-name m1_piper --env.task-name motion_tracking \
  --data.m1-motion-set C --env.num-envs 2048 \
  --schedule.total-iterations 320000

conda run -n supermimic python scripts/play.py \
  --ckpt logs/m1_motion_set_c_regression/ckpt_latest.pt \
  --clips a_wheel_000000 a_wheel_000002 b_arm_000006 \
          c_assisted_000029 c_assisted_000039 c_assisted_000149 \
  --num-envs 6
```

## Known risks and next work

- A 15-minute Go2 run is a bounded integration and early learning check, not enough evidence to claim converged locomotion. Judge it by fixed-command real-state evaluation, not imagined return alone.
- Go2 imagined torque remains a PD/effort-limited approximation; only real MuJoCo evaluation measures the actual actuator response.
- The M1 point target uses a URDF gripper-center proxy until a calibrated TCP transform is supplied. The random FK sweep establishes geometric range, not stable closed-loop reachability.
- Blind locomotion on lightly rough continuous ground is a next curriculum step after flat command tracking. No heightmap, depth image, LiDAR, large-step or obstacle task is included here.
- G1 anchor and WM normalization are opt-in; a new G1 long run and comparisons against the original checkpoint are not part of this first code-validation pass.
