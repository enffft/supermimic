# M1 Joint Training Phase and Gate Semantics

Date: 2026-11-02
Base revision: `0bda727aaf5b422af792017ac3afb3190db92ee1`

## Explicit modes

`M1AlternatingTrainer.update_event()` now has explicit, non-interchangeable semantics:

- `actor_only`: the Prior stays frozen regardless of readiness metrics.
- `auto_joint`: honors `freeze_prior`; when not manually frozen, it still requires configured strict WM and Actor quality Gates.
- `controlled_joint_experiment`: requires the explicit CLI flag `--m1-joint-training.no-freeze-prior`; it is labelled research mode and does not claim strict quality readiness.

Controlled mode first fits the WM only, then trains the Actor against a frozen Prior. It permits Prior updates only after configured warmups and real safety checks. The Task L/M production runs use 200 WM update events, 384 Actor optimizer steps, 100 sampled real-state diagnostics, survival `≥0.90`, base height `≥0.22 m`, tilt `≤0.90 rad`, action saturation `≤0.20`, and finite replay WM metrics. Prior learning rate is `1e-6`; Actor learning rate is `2e-4`. Prior updates are no more frequent than Actor updates.

No task-calibrated held-out WM NSE thresholds are configured yet. Therefore `schedule/wm_ready` remains false by design even when replay NSE values are finite and low. The controlled research gate is distinct and separately logged; it does not rewrite the strict WM result as a pass. Likewise `schedule/actor_ready` reports the stricter tracking thresholds, while the controlled research gate records its own safety/warmup reasons.

The stage and counters are saved in checkpoints and TensorBoard:

```text
schedule/training_phase
schedule/wm_ready
schedule/actor_ready
schedule/prior_unfrozen
schedule/controlled_joint_ready
schedule/controlled_joint_*_pass
wm/update_events
wm/optimizer_steps
prior/optimizer_steps
actor/optimizer_steps
```

Resume requires the same training mode, task sampler, prior source, loss coefficients, robot hash, Actor schema and Action contract. The M1 Action contract is now versioned `m1_mixed_action_v2`; old checkpoints with the earlier Piper scales are rejected for joint resume. No historical checkpoints were overwritten.

## Replay-consistent warmup exploration

M1 action noise in the environment remains zero. During WM-only warmup, an independently configured fraction of environments can receive small bounded action perturbations. The perturbed action is the action actually sent to MuJoCo and stored in Replay, so WM supervision uses the executed Action. Exploration is disabled by default and ends with WM-only warmup. The long-run command uses 25% of environments and standard deviation `0.10` only during that phase.

## Resource preflight and actual L transition

512- and 2048-env, 30-iteration tests both exercised the controlled stage transition with small logical batches and one WM optimizer step per update event. Each reached WM/Prior/Actor counts `8/6/6` without NaN or OOM. Their strict `wm_ready` flag remained false, as expected without calibrated held-out thresholds.

The 2048-env L run uses logical batch/microbatch `32/8`, one WM optimizer step per event, 10-step imagination, 40 schedule warmup iterations, 200 WM-only iterations, and a 30-minute wall-clock limit. Full training telemetry and final phase/counters are reported in `M1_L_2048_30MIN_RESULTS.md`.

The full suite passed `106` tests; M1-targeted tests passed `24`. The only full-suite warnings were upstream TorchScript deprecations and MuJoCo Warp's CUDA 12.2 graph-capture warning (graphs are disabled; execution continued normally).
