# M1 V3.3 Task L — 2048 environment training results

## Run setup

Run name: `m1_v33_L_2048_30min`
Launch script: [`scripts/run_m1_v33_l_2048_30min.sh`](../scripts/run_m1_v33_l_2048_30min.sh)
Environment: Conda `supermimic`; device `cuda:0`; 2048 envs; privileged State57; history 5; 10-step Prior and imagination; fixed seed 20261103.
Prior: V3.3 checkpoint SHA256 `05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c`; `legacy_v1`; Action Contract v3; `dataset_v3_3_l_high_carry`.
Training mode: controlled joint experiment, 200 WM-only updates, Actor pretrain 384 updates, then Prior updates only after the safety/readiness checks. Commands use L vx `[-0.09, 0.145]` m/s and vy/wz fixed at zero. Observation, reset, domain and action noise are disabled.

## Runtime log

The launch strict-loaded the specified Prior at step 9600. At iteration 25, replay WM NSE1/8 was 0.0048/0.2627. It reached 0.0000/0.0003 by iteration 300 and 0.0000/0.0002 by iteration 450. These are online replay diagnostics, not held-out WM scores. The run keeps the configured readiness gate and does not call finite/replay-only scores deployment readiness.

| Iteration | Phase | WM / Prior / Actor optimizer steps | Replay WM NSE1/8 | XY vector RMSE | yaw RMSE | positive-command backward fraction |
|---:|---|---:|---:|---:|---:|---:|
| 25 | WM warmup | 40 / 0 / 0 | 0.0048 / 0.2627 | — | — | — |
| 300 | WM warmup | 1040 / 0 / 0 | 0.0000 / 0.0006 | — | — | — |
| 1000 | Actor warmup | 3840 / 0 / 184 | 0.0007 / 0.0043 | 0.1183 | 0.0675 | 0.4106 |
| 1250 | Actor warmup | 4840 / 0 / 284 | 0.0033 / 0.0253 | 0.1573 | 0.1355 | 0.4204 |
| 1500 | Actor warmup | 5840 / 0 / 384 | 0.0043 / 0.0247 | 0.0955 | 0.0667 | 0.0941 |
| 1525 | controlled research update | 5940 / 10 / 394 | 0.0038 / 0.0257 | 0.2021 | 0.2733 | 0.3613 |
| 1750 | controlled research update | 6840 / 100 / 484 | 0.0057 / 0.0285 | 0.0864 | 0.1214 | 0.0216 |
| 1875 | controlled research update | 7340 / 150 / 534 | 0.0059 / 0.0272 | 0.0870 | 0.1016 | 0.0667 |

This first segment stopped normally at its 30-minute budget. `schedule/controlled_joint_ready` became true at iteration 1525, but `schedule/actor_ready` still failed on wheel tracking at that point. The controlled research gate is a safety gate; it does not certify signed task velocity or held-out World Model quality. The run ended with Actor readiness passing, after the final rolling wheel-tracking mean dropped below the configured 0.75 rad/s threshold. WM NSE values are replay-batch diagnostics, not held-out estimates.

## Pre-training physical interface control

Before the training run, a separate 12-env, 50-step MuJoCo test applied the Prior's reference using the v3 inverse mapping. It measured signed body vx of +0.060 m/s for a +0.12 command and -0.073 m/s for a -0.08 command; stop drift was -0.015 m/s. Wheel references and measured rates had the same sign in the two moving cases. Average root z settled near 0.445–0.446 m with small roll/pitch. This demonstrates control direction and contacts for that 1-second smoke test, not actor performance.

## First-segment fixed-seed evaluation

The fixed evaluator used 64 environments, 500 steps (10 simulated seconds), seed `20261103`, zero observation/reset/action noise and no domain randomization. It restored each checkpoint's own task, pose profile, prior source SHA/input policy and action contract. The four commands below are in-distribution (ID); `±0.30 m/s` and `+0.20 rad/s` are separately marked out-of-distribution (OOD). Every group had zero falls in this 10-second evaluation. Full per-joint wheel, leg and Piper tracking, root height/tilt, action saturation and vector/per-axis RMSE are in `logs/m1_prior_actor/m1_v33_L_2048_30min/eval_it0500.json`, `eval_it1000.json`, `eval_it1500.json` and `eval_final.json`.

| Stage checkpoint | forward ID `+0.12`: signed dx / vector XY RMSE / backward fraction | backward ID `−0.06`: signed dx / vector XY RMSE | stop ID signed dx | forward OOD `+0.30`: signed dx / vector XY RMSE |
|---|---:|---:|---:|---:|
| it 500, before Actor updates | +0.007 m / 0.119 m/s / 0.383 | +0.004 m / 0.061 m/s | +0.005 m | +0.008 m / 0.299 m/s |
| it 1000, Actor warmup | −0.196 m / 0.159 m/s / 1.000 | −0.074 m / 0.090 m/s | −0.164 m | −0.199 m / 0.338 m/s |
| it 1500, frozen-Prior Actor warmup | +0.003 m / 0.129 m/s / 0.152 | −0.157 m / 0.116 m/s | −0.388 m | +0.086 m / 0.264 m/s |
| it 1875, after 150 Prior updates | +0.124 m / 0.090 m/s / 0.146 | −0.261 m / 0.053 m/s | −0.205 m | +0.686 m / 0.158 m/s |

The Prior-updated checkpoint improved signed forward displacement over the pre-update it 1500 checkpoint, while the positive-command backward fraction changed only from 15.2% to 14.6% in this fixed evaluation. At the trained ID forward command, mean displacement was only about 0.012 m/s over the 10-second episode, far below the commanded 0.12 m/s. The stop command drifted backward by 0.205 m, and ID backward motion exceeded its target magnitude. These are real physical-task outcomes, not evidence of successful tracking. A single before/after pair also combines 150 Prior and 150 Actor optimizer updates, so it is not an isolated causal Prior ablation.

At the 30-minute stop, `ckpt_latest.pt` was v3 at iteration 1875; its SHA256 was `d6b9272ff3f9cdbb80d83ee74f4473b1fe54956895476d1072de94e405675ae4`. Its saved counters were WM/Actor/Prior `7340/534/150`, and it records Prior SHA `05a858…ff10c`, `legacy_v1`, `dataset_v3_3_l_high_carry`, and the v3 asymmetric target bounds. This Checkpoint remains preserved as the resume source.

Stage Checkpoint SHA256 values used above: it 500 `3fe29de435c117efa700b73bb7502e3c31731acbc1edce8939bd90525a3dfe30`; it 1000 `220a5e8a6229cbfa1514573041efe681e173ce21e07d1ff7043464fb31cec2fb`; it 1500 `8fd46f25eb1544b1275db6acc44c58faaf964ea8f1b3808cf7f87d4af6d39652`; final it 1875 as above.

## Extended cumulative run and analysis stop

At the user's request, training was extended from the 30-minute final Checkpoint. The first resume attempt exposed that `torch.load(map_location=cuda)` moved the saved CPU RNG state to CUDA before `torch.set_rng_state`. The loader now moves Torch RNG byte tensors back to CPU before restoring them; the CUDA-mapped save/load regression test passes.

The controlled joint segment resumed from it 1875 and reached it 3000, accumulating 600 Prior updates and 984 Actor updates. An early CPU comparison at it 2000 used identical high-carry, real-L-middle and legacy-low histories for `prior_initial` and the updated Prior. On high carry, `+0.12 m/s` root dx over 0.2 s fell from `+0.01550` to `+0.00906 m`, mean four-wheel rate from `+0.858` to `+0.376 rad/s`; zero-Goal drift moved from about `+0.00004` to `−0.00313 m`. Each of the three tested in-support commands gained 2 joint-velocity-limit frame violations, with no URDF or SDK violations. The full output is in `logs/m1_prior_actor/m1_v33_L_2048_2h_resume90m_rngfix/prior_delta_it2000.json`; references remain marked not dynamically validated.

The same-seed it 3000 MuJoCo evaluation confirmed that the updated Prior/Actor was not ready for continued joint optimization. It used 64 environments, 500 steps, seed `20261103`, noise and randomization disabled, and had zero falls in every group:

| it 3000 fixed command | signed forward dx | vector XY RMSE | yaw RMSE | positive-command backward fraction |
|---|---:|---:|---:|---:|
| forward ID `+0.12 m/s` | −0.090 m | 0.138 m/s | 0.041 rad/s | 0.996 |
| backward ID `−0.06 m/s` | −0.337 m | 0.030 m/s | 0.037 rad/s | 0 |
| stop ID | −0.279 m | 0.062 m/s | 0.034 rad/s | 0 |
| forward OOD `+0.30 m/s` | +0.216 m | 0.254 m/s | 0.061 rad/s | 0.078 |

The fixed evaluation JSON contains full per-joint wheel/leg/Piper tracking, root, saturation and per-axis metrics at `logs/m1_prior_actor/m1_v33_L_2048_2h_resume90m_rngfix/eval_it3000.json`. Despite a forward-ID wheel RMSE of only `0.141 rad/s`, that command moved backward for 99.6% of samples; lower tracking error did not imply correct Task L behavior.

The evaluation was extended to measure the Prior plan at every replan using the live state/history. For `+0.12 m/s`, terminal root displacement over 0.2 s averaged a weak `+0.0015 m`, but the four wheel references averaged `−0.255/−0.106/−0.252/−0.090 rad/s`. The mean four-wheel reference sign was wrong in 98% of forward plans, and its direction disagreed with the root target in 94%. The prior reference is internally inconsistent on this trained command; the Actor follows the backward wheel targets and the robot moves backward. This is more precise than calling the entire Future31 root trajectory reversed.

To isolate runtime Prior dependence, the same it 3000 Actor, same seed, reset, action mapping and MuJoCo evaluation were run twice, changing only Prior weights between the fine-tuned `motion_prior` and saved `prior_initial`:

| Same it 3000 Actor | `+0.12` signed dx / XY RMSE / backward fraction | `+0.12` mean-four-wheel direction wrong / root-wheel disagreement | stop signed dx | `−0.06` signed dx |
|---|---:|---:|---:|---:|
| Fine-tuned Prior | `−0.090 m / 0.138 m/s / 0.996` | `0.980 / 0.940` | `−0.279 m` | `−0.337 m` |
| `prior_initial` swap | `+0.475 m / 0.078 m/s / 0.084` | `0.000 / 0.020` | `+0.077 m` | `−0.082 m` |

This same-Actor Prior swap is strong evidence that the it 3000 forward and stop failures are driven by the fine-tuned reference. It is a runtime reference intervention, not a retrained-policy causal ablation. It also shows `prior_initial` is not universally better: the it 3000 `−0.06` command is less accurate with the swapped Prior.

The CPU WM action-response audit at it 3000 replayed V3.3 reference actions through the saved WM using the saved asymmetric inverse contract. With the fine-tuned Prior, `vx=−0.08` generated WM-predicted `+0.0046 m` root displacement and `+0.064 m/s` final body vx; zero Goal produced `+0.0057 m` and `+0.071 m/s`; `+0.12` produced `+0.0073 m` and `+0.080 m/s`. Thus the it 3000 model predicted forward even for reverse and stop. The initial Prior's corresponding predictions were `−0.0019/−0.0004/+0.0022 m`. The replay NSE1/8 at it 3000 was `0.0015/0.0063`, which did not expose this action-response failure. Full output is at `logs/m1_prior_actor/m1_v33_L_2048_2h_resume90m_rngfix/wm_action_response_it3000.json`.

The joint segment was stopped after the it 3000 Checkpoint had been written (the process reached it 3025 and was interrupted during the next Prior backward pass). Its last complete Checkpoint is `ckpt_0003000.pt`, SHA256 `968feac5ccd01681a8c7133daa9b6095d3b934da4dcd7ade7810a696a90aae71`; no partial result replaced it. Because the WM failed the action-direction audit, the continuation used `--prior-updates-per-event 0`, retained the controlled safety gate and logged phase `actor_only_prior_frozen`.

The user then asked to stop training and focus on analysis. We stopped the process at it 5150; the active MuJoCo step raised `KeyboardInterrupt`, and no interrupted in-memory state was saved. The latest complete Checkpoint remains `logs/m1_prior_actor/m1_v33_L_2048_2h_actor_only_resume72m/ckpt_0005000.pt`, SHA256 `a17af8034c8b441db18368188b36e538808c935fcab848031ac09bcc2081039c`, with optimizer counters WM/Prior/Actor `19840/600/1784` and phase `actor_only_prior_frozen`. The run's last event was it 5150 at 0.5721 h. Across the three training segments, cumulative logged training time was approximately 1.446 h (86.8 minutes), below the previously planned two-hour total because of the user's stop instruction. No G1/Go2 process was changed.

Actor-only fixed-seed evaluations at it 5000 used the same command set and reset. With the trained Prior, forward-ID displacement recovered to `+0.126 m`, but vector XY RMSE was `0.106 m/s` and 33.0% of positive-command samples still had negative vx. The mean four-wheel reference sign was wrong in 52.4% of forward plans and disagreed with the root target in 40.4%. Stop drift remained `−0.115 m`; backward-ID displacement was `−0.242 m` for a `−0.06 m/s` target. All groups remained upright. Replay WM NSE1/8 fell to `0.00036/0.00120`, while the real signed velocity errors remained.

The it 5000 same-Actor swap to `prior_initial` made forward-ID displacement `+0.809 m`, reduced its backward fraction to 1.6%, and reduced vector XY RMSE slightly to `0.096 m/s`. Its forward wheel references were strongly positive (`2.096/1.910/2.058/1.946 rad/s`) and had no wrong-direction plans, but wheel tracking RMSE rose from `0.187` to `0.705 rad/s`. At the same time, the initial-Prior swap performed worse on the backward-ID command (`+0.018 m` displacement versus `−0.242 m` with the trained Prior) and on yaw-only. Thus the Actor learned to partly compensate for the trained Prior, while the initial Prior gives a much better forward target; neither variant solves all commands.

The actor-only CPU WM action-response audit at it 5000 still failed direction checks, but its error changed. For the *initial* Prior's `−0.08/0/+0.12` references, WM predicted root dx `−0.0148/−0.0139/−0.0100 m`; for the fine-tuned Prior it predicted `−0.0061/−0.0046/−0.0021 m`. It now predicts backward even for the positive reference. Replay NSE1/8 improved from `0.0015/0.0063` at it 3000 to `0.00036/0.00120` at it 5000, while counterfactual command response remained wrong. See `logs/m1_prior_actor/m1_v33_L_2048_2h_actor_only_resume72m/wm_action_response_it5000.json`.

## Root-cause evidence: gradient connectivity versus gradient quality

The phrase “Prior gradient reaches Actor” needs a distinction. During an Actor update, the Prior is a frozen teacher: `_one_micro_rollout(planner=False)` detaches the sampled Future31 reference, then computes Actor tracking loss through the differentiable WM rollout. Prior parameter gradients are intentionally absent in that phase. During a Prior update, the Actor and WM weights are frozen, but their forward operations remain differentiable, allowing planner loss to backpropagate through Actor actions and WM states into the Prior reference. This is the intended alternating optimization contract, not a missing gradient path.

The runtime guards verify that each Actor parameter receives a gradient and changes on each update; the same pattern guards Prior updates. In actor-only training, the Actor counter rose from 984 at it 3000 to 1784 at it 5000 while the Prior counter stayed at 600. Logged Actor gradient norms remained positive (for example, `0.0290` at it 5000). This rules out a disconnected Actor parameter path. It does not prove the gradient points toward better real MuJoCo task performance: Actor optimization minimizes reference tracking through the learned WM, while the sampled Prior reference is detached and treated as a target.

Evidence separates three problems: the Prior can produce incompatible root/wheel targets, the WM action response is not trustworthy for optimization, and the Actor gradient itself is connected. It does not support a global action-sign bug:

- **Prior target quality:** static high-carry Prior references have correct root-dx signs for `−0.08/0/+0.12` (`−0.0099/~0/+0.0155 m`). After 600 updates, forward root motion shrank to `+0.0055 m` and stop drift became `−0.0045 m`. In the it 3000 closed-loop `+0.12` evaluation, terminal root dx stayed weakly positive (`+0.0015 m` per 0.2 s), but the mean four-wheel reference sign was wrong in 98% of plans and disagreed with root direction in 94%. With the same it 3000 Actor, swapping only to `prior_initial` changed forward displacement from `−0.090` to `+0.475 m` and positive-command reverse fraction from `99.6%` to `8.4%`. This identifies the fine-tuned Prior target as the dominant direct cause of that forward-ID failure. The initial Prior is not universally better: at it 5000 its same-Actor backward-ID swap produces slight forward motion instead of the trained Prior's backward motion.
- The v3 inverse action mapping passed an earlier physical reference smoke test (`+0.12` command → `+0.060 m/s`; `−0.08` → `−0.073 m/s`) with wheel-ground contact. That test does not certify every learned Actor trajectory, but it argues against a global wheel-sign inversion.
- **World Model and its gradients:** at it 3000, WM predicted forward displacement for all three static `−0.08/0/+0.12` Prior plans (`+0.0046/+0.0057/+0.0073 m`). At it 5000, using the same test inputs, it predicted negative displacement for all (`−0.0148/−0.0139/−0.0100 m` for initial Prior references). The model's sign response changed while replay NSE1/8 improved from `0.0015/0.0063` to `0.00036/0.00120`. Therefore low replay error does not validate counterfactual action gradients; Prior/Actor optimization through this WM was not trustworthy.
- **Actor and actuator:** the v3 setpoint inverse passed a real MuJoCo sign smoke test (`+0.12 → +0.060 m/s`, `−0.08 → −0.073 m/s`) with wheel-ground contact. At it 3000 the Actor tracked the wrong-sign forward wheel reference with `0.141 rad/s` RMSE and moved backward for 99.6% of samples. At it 5000 it partly compensated: trained-Prior forward displacement became `+0.126 m`, but 33.0% of positive-command samples were still backward. Swapping only to initial Prior gave `+0.809 m` forward displacement despite *worse* wheel tracking RMSE (`0.705` vs `0.187 rad/s`). This directly shows that lower Actor reference loss can coincide with worse task motion.

Current diagnosis: **the Actor gradient is connected; the it 3000 forward failure is directly driven by fine-tuned Prior references whose wheel direction conflicts with their root trajectory.** The same-Actor Prior swap repairs forward and stop behavior at it 3000, strongly implicating the Prior update. WM response probes fail both before and after actor-only learning, with sign errors that change between checkpoints despite lower replay loss. This makes WM-mediated Prior gradients an untrustworthy optimization signal and likely contributor, but this experiment does not isolate their historical causality. The Actor partly compensates under the trained Prior, yet tracking loss still fails to rank task quality correctly. Action/actuator sign is not the primary cause. No result supports calling the full policy ready.
