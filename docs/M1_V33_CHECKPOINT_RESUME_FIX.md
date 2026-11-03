# M1 V3.3 Checkpoint save/resume and fixed evaluation

## One serialized action contract

Checkpoint save and restore previously constructed the wheel mapping independently, differing only in the variable name inside the string (`action` vs `a`). Resume then rejected an otherwise matching v3 checkpoint. `serialize_m1_action_contract(robot_spec)` now owns the full mapping used by both save and load, including the pose profile, defaults, names/order, control modes, scales, absolute bounds, position mapping and wheel mapping.

The supported wheel string is `clip(action,-1,1)*wheel_scale_rad_s`. No duplicated wheel serialization remains in the M1 joint trainer.

## Strict resume behavior

`M1AlternatingTrainer.load_checkpoint` requires v3 format and checks robot/model identity, joint names, actor schema, Prior schema and full source provenance (SHA, normalization/dataset/split hashes, contracts, assurance and selected input policy), task sampler configuration, loss coefficients, training mode, action mapping and action scales. It then restores the World Model, Prior, frozen initial Prior, Actor, all optimizer states, real optimizer counters, stage and RNG states.

The CPU round-trip test constructs two small trainers, saves one, restores the second and compares all model tensors, optimizer states, counters and action mapping exactly. It also verifies rejection of v2 resume, changed Prior SHA, pose profile, joint bounds and input compatibility. A real 2048-environment resume then exposed a second bug: loading the full Checkpoint with `map_location=cuda` moved `rng_state.torch_cpu` onto the GPU, where `torch.set_rng_state` rejects it. Restore now explicitly converts Torch RNG byte tensors to CPU before applying them. The regression test reloads the saved Checkpoint with `map_location="cuda"` and restores that state; all 24 M1 integration tests pass.

## Fixed evaluation

`scripts/evaluate_m1_prior_joint.py` reads v2 and v3 checkpoint formats. For v3 it rebuilds the saved RobotSpec from `action_mapping`, loads the Prior from the saved source SHA and saved input compatibility, restores the saved default profile, action contract, bounds and task sampler ranges, and now defaults to the saved L/M task. An explicit task override that disagrees with the checkpoint is rejected. The fixed evaluator reports each live replan's first-frame and terminal root displacement, Prior wheel-reference direction, and root/wheel direction disagreement. Its `--prior-variant initial` option swaps only to the Checkpoint's saved `prior_initial` weights while retaining the same Actor for a reference-dependence diagnostic. OOD velocity cases are marked, with vector and per-axis XY RMSE reported separately.

`scripts/audit_m1_reference_control.py` uses the same action inversion and checkpoint reconstruction code. It reports the saved source assurance and categorized action-envelope, URDF, SDK and velocity violations, with unvalidated dynamics labeled explicitly.

## Goal sensitivity and update counters

Goal sensitivity now perturbs one L goal component at a time while holding the same history and other goal inputs fixed, and separately records M end-effector goal sensitivity and an L↔M task-switch sensitivity. Prior-unfrozen status is based on whether the Prior optimizer step counter actually increased in the event. WM, Actor and Prior optimizer steps are saved independently.

Training and fixed evaluation use the same XY definitions:

- vector XY RMSE: `sqrt(mean(ex² + ey²))`;
- per-axis XY RMSE: `sqrt((mean(ex²) + mean(ey²)) / 2)`.

Post-run review found that mjlab auto-resets an environment inside `env.step()` before returning, clearing that environment's ActionTerm buffers. The training loop now snapshots the Replay write cursor before the step and reads the exact executed `a_exec` from the transition row written before reset; physical setpoints are reconstructed from that action and the saved v3 contract. Previously logged terminal-step saturation counters may undercount those reset rows. Fixed-evaluation results and trained policy weights are unaffected by this telemetry-only fix.

## Verification

The full M1 test suite passed 48 tests in the `supermimic` Conda environment, including the CUDA-mapped Checkpoint round trip; 35 PyTorch TorchScript deprecation warnings remain. The fixed-evaluation task-resolution tests confirm checkpoint-first behavior and reject accidental task mismatch. Full training artifacts and post-run evaluations are recorded in [the L 2048 training report](M1_V33_L_2048_TRAINING_RESULTS.md).
