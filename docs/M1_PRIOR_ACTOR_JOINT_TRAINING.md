# M1 Piper Motion Prior × Actor Joint Training

Status: opt-in research path, code and gradient plumbing verified; **not ready for long training**. This implementation uses privileged MuJoCo state for the planner and does not establish deployable proprioceptive control.

## Architecture

`m1_prior_joint` adds a separate M1-only path. The existing G1, Go2, M1 motion-tracking, and M1 world-point tasks keep their prior configuration paths. Joint training is enabled only when both `--env.task-name m1_prior_joint` and `--m1-joint-training.enabled` are set.

```text
real MuJoCo transition → Replay → World Model update
                                      ↓ frozen parameters, differentiable inputs
history + fixed task context → CVAE prior → 10-frame reference
                                                    ↓
                                         Prior-conditioned Actor
                                                    ↓
                                          frozen World Model

then freeze Prior and World Model; train Actor against the fixed reference
```

The CVAE Prior and Actor have separate Adam optimizers. Each eligible event runs WM updates, then Prior updates, then Actor updates. The Prior objective uses task performance through the frozen Actor and WM; no Actor-state matching term is used as a planner objective. The Actor objective is reference tracking plus stability and control terms. Both objectives backpropagate through the frozen WM; parameter freezing does not detach its inputs.

| Module | Input | Output |
|---|---|---|
| AMP CVAE Prior | state history `[B,4,53]`, Goal8 `[B,8]` | Future31 `[B,10,31]` |
| SuperMimic adapter | physical State57 history `[B,4,57]`, L/M context | CVAE features and Goal8 |
| Prior-conditioned Actor | proprio history `[B,5,72]`, Goal8, Future31, phase | bounded Mixed Action `[B,22]` |
| World Model | physical State57, Mixed Action22, domain parameters | next physical State57 |

The 22 actions retain the M1 mixed-action mapping: 12 leg position controls, 4 wheel velocity controls, and 6 Piper position controls. The reference remains kinematics; it is not projected into actions.

## Data and coordinates

The adapters are device-native PyTorch code and preserve gradients. The SuperMimic physical state is `[root xyz, root quaternion wxyz, 22 joint positions, root linear velocity world, root angular velocity body, 22 joint velocities]` (57 values). State53 uses base height, yaw-heading-relative rotation6D, body-frame root linear velocity, body angular velocity, AMP-name-ordered leg/Piper positions and velocities, then four wheel rates. AMP rotation6D stores matrix columns 1 and 2 consecutively. Quaternion conversion is `wxyz` internally and `xyzw` only at the AMP frame boundary.

Goal8 is `[vx, vy, wz, ee_dx_heading, ee_dy_heading, ee_dz_heading, velocity_valid, ee_valid]`. L commands are stored with the real sampled command and remain episode-consistent. M stores the fixed environment-local world target in Replay; each imagined step recomputes heading-relative Goal8 from the current predicted base state. Future31 is root translation relative to the initial base yaw frame, root rotation6D, 12 leg positions, 6 Piper positions, and 4 wheel angular rates. Decoding uses joint names, including the interleaved wheel joints. The first reference state is at `t+0.02 s`; phase advances through all ten frames and replans every ten control steps.

The CVAE labels use `piper_x_link7`. The SuperMimic native point task uses the midpoint of `piper_x_link7` and `piper_x_link8`; on the checked M1 model their position FK agrees at tested arm poses, so the joint planner and CVAE can use link7. Orientation conventions are not thereby certified for other TCP definitions.

## Objectives and update boundaries

Planner loss is implemented in `src/supermimic/motion_prior/losses.py`:

```text
L_prior = L_task_L + L_task_M + λ_drift L_base_drift
        + λ_feasible L_feasibility
        + λ_smooth L_smoothness
        + λ_anchor L_pretrained_anchor
        + λ_energy L_energy
        + λ_effort L_control_effort_surrogate
```

L uses normalized body-velocity and yaw-rate command errors. M uses true fixed-world TCP distance, path distance, terminal progress, and a base XY/yaw drift penalty. Feasibility checks base height/tilt, joint limits, joint velocity, and a conservative wheel-rate cap. Smoothness checks root/joint/wheel acceleration and jerk separately, using `dt=0.02 s`; configured thresholds are regularization thresholds and are not claimed as experimentally validated actuator limits. The frozen initial CVAE output anchors the plan in normalized physical units. `L_supervised` is present as a logged zero-weight hook; Motion Bank sampling/reconstruction is not wired into v1.

There is no validated differentiable actuator-torque model yet. Therefore planner `energy_loss` is zero and `energy_weight` defaults to zero. An optional action-effort surrogate is separately named and also defaults to zero. Real MuJoCo generalized actuator force, joint velocity, positive mechanical power, and per-step mechanical work are logged as simulator diagnostics; these values are not battery energy or a validated deployment energy estimate. No differentiable collision guarantee is claimed.

Actor loss is the weighted sum of root position, SO(3) root orientation, leg position, arm position, and wheel-rate reference errors, plus base/joint stability and action-rate/joint-speed/joint-acceleration regularizers. Task reach reward is not an Actor objective. Planner and Actor critics are disabled; each uses a separate 10-step critic-free BPTT objective.

During a Prior update, Actor and WM parameters have `requires_grad=False`, their parameter gradients are cleared, and the computation graph still propagates through Actor and WM input Jacobians to Prior outputs. During an Actor update, Prior and WM parameters are frozen and the freshly generated reference is detached. Runtime assertions check finite positive gradient norms, parameter ownership, and that the selected optimizer changes a parameter. See `M1_PRIOR_ACTOR_GRADIENT_AUDIT.md` for the audit and finite-difference result.

The default phase schedule is WM warmup (200 update events), Actor tracking while the measured real-data readiness window is not met, then alternating Prior and Actor. `freeze-prior` defaults true. Readiness checks survival, root/arm/wheel tracking errors, action saturation, and a minimum sample count. The short Smoke deliberately used smaller gates and is not evidence that the default readiness gate has passed.

## Checkpoints and resume

The joint checkpoint has its own `supermimic_m1_prior_actor_joint_v2` format and stores WM, current and frozen-initial Prior, Actor, three optimizers, update counters, phase, task sampler configuration, RNG state, robot/model identity, prior source hash, schema, action and goal mapping, TCP convention, horizon, and loss coefficients. Loading is strict and rejects format/schema/source/config mismatches. Use `--m1-joint-training.resume`; do not combine it with `--init-wm-from` or legacy `--resume`. Replay contents and simulator state are not checkpointed, so a resumed run starts with an empty replay and must collect a fresh valid window.

## Current scope and launch configuration

Implemented tasks: L, M, and balanced L/M sampling. LM, 32-step replanning, dual critics, periodic Motion Bank supervised updates, and estimated/proprioceptive velocity are not implemented. `prior_state_source=privileged` is logged. `estimated` is reserved and explicitly errors instead of silently reading simulator truth. v1 also requires observation noise off and action noise zero so reconstructed Actor inputs and imagined actions match the sampled environment path.

The following is a **candidate configuration for a later controlled run**, not a recommendation to start long training now. Every flag name was checked against `sm-train --help`; long training is currently gated on the World Model and smoothness/task-scale calibration reported in `M1_PRIOR_ACTOR_SMOKE_RESULTS.md`.

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
    --run-name m1_prior_actor_balanced_v1 \
    --log-root logs/m1_prior_actor \
    --device cuda:0 \
    --env.robot-name m1_piper \
    --env.task-name m1_prior_joint \
    --env.num-envs 16 \
    --env.history-length 5 \
    --env.no-obs-noise \
    --env.action-noise-std 0 \
    --m1-joint-training.enabled \
    --m1-joint-training.task-type balanced \
    --m1-joint-training.state-source privileged \
    --m1-joint-training.prior-mode prior_mean \
    --m1-joint-training.batch-size 32 \
    --m1-joint-training.micro-batch-size 8 \
    --m1-joint-training.freeze-prior \
    --schedule.m1-wm-only-iterations 200 \
    --schedule.total-iterations 320000 \
    --schedule.max-hours 2
```

This command uses the checkpoint path and SHA256 configured in `M1JointTrainingCfg`. Run a readiness/pilot evaluation first; do not interpret a successful launch as convergence.
