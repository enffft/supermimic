# M1 Prior/Actor Gradient Audit

## Intended graph and update ownership

| Update | Trainable parameters | Frozen parameters | Gradient path retained |
|---|---|---|---|
| World Model | WM | Prior and Actor are not in WM loss | Real Replay transitions only |
| Planner | CVAE Prior | Actor and WM | `Planner loss → predicted states → WM input Jacobians → Actor action → Actor reference Jacobian → Prior output` |
| Tracking Actor | Actor | Prior and WM | `Tracking loss → predicted states → WM input Jacobian → Actor parameters`; reference is detached |

The Prior and Actor update methods clear their own optimizer state plus frozen-network `.grad` fields before backpropagation. A context manager temporarily sets frozen module parameters to `requires_grad=False` and restores their previous flags in `finally`. It does not wrap the forward pass in `no_grad` and does not detach the Planner reference. Actor update generates a new reference after the Prior step and detaches that reference before tracking loss. The old graph is never reused across optimizer steps.

## Automated checks

`tests/test_m1_prior_joint.py::test_planner_actor_gradient_ownership_and_horizon3_finite_difference` uses a CUDA batch of 2, horizon 3, a small differentiable WM, M1 FK surrogate, and the actual pretrained CVAE. It:

1. Computes the Planner task-loss gradient with respect to a CVAE decoder parameter and compares it to a central finite difference.
2. Runs a Prior optimizer step and checks that Prior parameters change while Actor and WM parameters remain byte-equal.
3. Runs an Actor optimizer step and checks that Actor parameters change while Prior and WM parameters remain byte-equal.
4. Relies on the runtime optimizer assertions to reject missing/nonfinite/zero gradient norms and a no-op parameter update.

`tests/test_m1_prior_joint.py::test_pretrained_prior_strict_load_and_amp_cpu_gpu_parity` verifies the strict checkpoint hash, CPU AMP inference parity, and bounded CUDA parity. `test_state_goal_and_future_match_amp_contract` checks State53, Goal8, Future31, rotation, and wheel/joint-order mappings. `test_link7_matches_gripper_link7_link8_midpoint_for_position_fk` checks position FK equality for eight tested Piper configurations.

Full-suite result after the smoothness diagnostics refactor: **75 passed**. Ruff for the new Motion Prior package and its test file passed. Smoke E created a v2 joint checkpoint after six Prior and six Actor updates; its `prior_initial` and live Prior differ by a maximum absolute parameter value of `2.15e-6`. The optimizer update methods independently assert live parameter movement on each step.

## Boundaries of this audit

The finite-difference test validates one Planner task gradient coordinate in a small differentiable system; it does not validate the learned World Model Jacobian against MuJoCo over a distribution, prove task improvement, or establish useful gradients from an untrained Actor/WM pair. No test yet asserts goal-swap gradient direction across a held-out set, loss-ablation behavior, domain-randomized gradient accuracy, or long-horizon stability. These remain required before treating planner optimization as calibrated.

The real Smoke used a randomly initialized small WM, an untrained Actor, only 8 environments, 8 WM update events and 6 optimizer steps for each Prior and Actor. It proves the online plumbing executes and parameter ownership assertions pass. Its negative EE progress and large Prior gradient norms are evidence against promoting it to a training-quality result.
