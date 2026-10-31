# G1 344K Plateau Root-Cause Audit and CuboiLeo e3 Alignment

**审计日期：** 2026-10-28
**工作仓库：** `enffft/supermimic`, commit `aeef2c2b23f47499c6951635949053422f1695b5`
**主参考：** `CuboiLeo/SuperMimic-Reproduced`, inspected at `ba6105349069bf040f1fdad60873d13138d505a2`
**次要对照：** `Breezewrf/supermimic_mjlab`, inspected at `23c76cad5bf41110709c7f929b54c1da1be6b248`

本报告只使用已有日志、checkpoint 元数据、CPU 代码/数据检查和静态仓库比较。没有启动 G1 仿真、训练、benchmark、viewer 或 CUDA smoke test。审计期间 Go2 训练进程保持运行。

## A. 历史训练诊断

### A.1 G1 lineage

| 运行 | 全局迭代 | 环境/批大小 | 关键设置 | 可用结果 |
|---|---:|---|---|---|
| `logs/4070/lafan1_q1_4070_anchor_retry1` | 0–67,900 | 1024 / 1024 | `root_joint`, goal 689D, anchor 9D, RunningNorm=false | 40-clip benchmark mean `0.4135208755`, step-weighted `0.4143437505`, 2 failed |
| `logs/4070/lafan1_anchor_3072_8h_resume01` | 67,900–76,000 | 3072 / 1024 | 同一 G1 lineage | 无新的全 40 clip benchmark |
| `logs/4070/lafan1_anchor_1024_probe3m_retry1_20261026` | 76,000–76,525 | 1024 / 1024 | 同一 G1 lineage | 无新的全 40 clip benchmark |
| `logs/4070/lafan1_anchor_1024_8h_resume02_20261026` | 76,525–211,050 | 1024 / 1024 | 同一 G1 lineage | 40-clip mean `0.4297083083`, step-weighted `0.4324968950`, 1 failed |
| `logs/phase3_training/g1_resume_1024_8h_20261027` | 211,050–344,350 | 1024 / 1024 | 同一 G1 lineage, `schedule.total_iterations=500000` | 没有更晚的全 40 clip benchmark；训练回报末值约 `0.434` |

另有 `logs/4070/g1_lafan1_fullbody6_norm_1024_12h`，它使用 `full_body` goal、anchor orientation 和 RunningNorm=true，属于另一实验，不能混入 CuboiLeo e3 结论。

Resume 阶段的 TensorBoard 横轴和墙钟时间不是同一连续 event stream；本表按 checkpoint 的全局 iteration 对齐，未把重复的局部 TensorBoard step 当成新的训练进度。各目录未提供足够可靠的统一墙钟小时数，因此没有伪造 wall-clock 字段。

### A.2 平台期证据

CPU EventAccumulator 提取物：[`history_summary.json`](artifacts/g1_344k_audit/history_summary.json)、[`history_metrics.csv`](artifacts/g1_344k_audit/history_metrics.csv)。图表：

- [`train_episode_norm_return.png`](artifacts/g1_344k_audit/train_episode_norm_return.png)
- [`wm_nse_1step.png`](artifacts/g1_344k_audit/wm_nse_1step.png)
- [`wm_nse_8step.png`](artifacts/g1_344k_audit/wm_nse_8step.png)
- [`policy_imag_return_undisc.png`](artifacts/g1_344k_audit/policy_imag_return_undisc.png)
- [`policy_imag_survival_H.png`](artifacts/g1_344k_audit/policy_imag_survival_H.png)
- [`policy_policy_grad_norm.png`](artifacts/g1_344k_audit/policy_policy_grad_norm.png)
- [`policy_critic_loss.png`](artifacts/g1_344k_audit/policy_critic_loss.png)
- [`policy_actor_lr.png`](artifacts/g1_344k_audit/policy_actor_lr.png)
- [`policy_action_abs_mean.png`](artifacts/g1_344k_audit/policy_action_abs_mean.png)

| 阶段 | 训练 normalized return | WM NSE 1-step | WM NSE 8-step | imagined return | survival | actor grad median |
|---|---:|---:|---:|---:|---:|---:|
| 0–67.9k | `0.374 → 0.429` | `0.2003 → 0.0126` | `1.173 → 0.0305` | `0.089 → 1.739` | 约 `0 → 0.999` | `0.0728` |
| 67.9k–211.05k | `0.429 → 0.454`, median `0.440` | `0.0126 → 0.00963` | `0.0305 → 0.0218` | `1.739 → 1.835` | 约 `0.9997` | `0.0692` |
| 211.05k–344.35k | `0.454 → 0.434`, median `0.443` | `0.00888 → 0.00754` | `0.0218 → 0.0122` | `1.835 → 1.829` | 约 `0.99985` | `0.0705` |

结论：实际行为最迟在约 67.9k 后进入平台，67.9k–211k 只有小幅波动，211k–344k 没有恢复性提升。World Model 在平台期后仍继续改善，特别是 8-step NSE；imagined return 和 survival 已经接近稳定高值，却没有转化为真实全身跟踪提升。这是 real/imagined objective 脱钩的证据，不能用低 WM NSE 证明整个训练链路正确。

Actor 梯度没有长期归零：344.35k 末值约 `0.0436`，中位数约 `0.0705`。因此“Actor 完全没有更新”不是日志支持的结论。action absolute mean 长期约 `1.9–2.0`，没有收缩到小动作。Critic loss 的正常中位数约 `0.0016–0.0019`，但存在严重离群峰：约 140850 (`1.79e6`)、255650 (`3.63e4`)、262450 (`4.75e10`)、300750 (`1.50e6`)。这些峰需要后续检查 batch/target/数值稳定性，不能被均值掩盖；它们尚未被证明是平台期的唯一根因。

### A.3 学习率和 checkpoint

从 checkpoint optimizer state 读取的实际 actor/critic/WM lr（这三者在这些 G1 runs 中相同）：

| checkpoint iteration | 实际 lr | 初始 lr 比例 | 说明 |
|---:|---:|---:|---|
| 67,900 | `1.66781625e-4` | `0.8339` | 早期 resume |
| 211,050 | `1.7390226e-5` | `0.08695` | 已显著衰减 |
| 344,350 | `1.2640707e-5` | `0.06320` | 此运行实际使用 `total_iterations=500000` |

最新 344k resume 的配置明确是 `schedule.total_iterations=500000`，不是参考 e3 的 320k。对于当前默认 `lr_final_ratio=0.03`、`lr_decay_fraction=0.78125` 的 500k schedule，344k 尚未到 floor；如果使用正确的 320k schedule，衰减段约在 250k 结束，之后接近 `0.03 * initial_lr = 6e-6`。因此旧实验的 500k schedule 是确认的配置偏差，也使后段学习率日程与 e3 不可比；它不能单独解释 67.9k 起的 body orientation 崩塌。

没有发现 Resume 后重新 warmup 的证据。checkpoint 会恢复 optimizer、全局 iteration、sampler 状态、RNG 和可选 command/action state；旧 replay buffer 不在 checkpoint 中，211k resume 后 replay 从空 buffer 重新积累。这是旧实验的 P1/P2 解释因素，Scratch 实验不继承该问题。

## B. 与 CuboiLeo e3 的静态比较

严重性：**P0** 确认的数学/数据/训练链路错误；**P1** 高可信度性能问题；**P2** 实验配置/可比性差异；**P3** 无害或隔离的实现差异。未完成 GPU 验证的项目不会标为 confirmed bug。

| Module | Reference implementation | Current implementation | Difference / expected consequence | Severity / evidence |
|---|---|---|---|---|
| G1 reward formulas | root/body position, orientation, linear/angular velocity and regularizers use the same Gaussian coefficients and dt scaling | same formulas and coefficients | No static formula mismatch found | P3; source diff + CPU golden state |
| Quaternion/body order | WXYZ, q/-q-equivalent `quat_box_minus`, 14 tracked bodies | same convention and order | No evidence for a quaternion sign or body-order bug | P3; CPU quaternion max component error `2.98e-7` |
| Goal/actor interface | 10 history frames, 93D proprio/frame, 68D motion frame, 10 future frames, 9D anchor, 689D goal | same encoded G1 path | G1 does not use Go2/M1 direct actor path | P3; static config/network comparison |
| Goal mode | `root_joint` + position anchor; no anchor orientation | same target for e3 lineage | `full_body`/orientation experiment is separate and excluded | P3/P2 if mixed; logs/config |
| Running normalization | e3 fixed state behavior, WM RunningNorm=false | G1 default/old run false | Breezewrf RunningNorm features are not mixed into this manifest | P3 for G1; static config |
| Actor/Critic architecture | encoded motion encoder, matching MLP dimensions, latent 32 and critic dimensions | matching dimensions; shared code has gated direct Go2/M1 paths | No G1 architecture drift demonstrated | P3; source comparison |
| World Model | state 71D, action 29D, xi 33D, window 8, horizon 32, explicit integration | same G1 dimensions and settings | shared generalized code is gated for G1 | P3; config/source comparison |
| Real/imagined action contract | action semantics must match physical actor | code paths are structurally aligned, but live real-vs-imagined orientation remains untested | Could explain body orientation if live observation/action field differs despite static matches | P1 candidate; `PENDING_GPU_VALIDATION` |
| Replay terminal labels | reference samples successor-row `done`; current samples source-row `done` | recorder stores `done_next` on source transition row, making source alignment semantically coherent | Difference can change termination-head labels and replay comparison; not proven to cause plateau | P1/PENDING; source audit |
| Replay resume | reference behavior depends on replay state | replay tensor is not restored; sampler bin state is restored | Resume begins with a fresh replay and temporarily short windows | P1/P2 for old resume; irrelevant to fresh Scratch |
| Reset sampler | return mode, bin frames 50, uniform/hard/memory 0.5/0.4/0.1, temperature 25, gamma .99 | matching settings | No static sampler mismatch found | P3; source/config comparison |
| Domain randomization | friction, COM, encoder bias and reset perturbation ranges/frequency | G1 ranges and asset settings appear equal | No confirmed extra difficulty | P2 pending live trajectory check |
| G1 MJCF/control | same G1 asset, dt .005, decimation 4, solver iterations 10, line-search iterations 20, action scale | same asset/control settings | No static model mismatch found | P3; source/config comparison |
| Learning schedule | e3 uses complete 320k cosine plan | latest old resume used 500k | Later LR and resumed experiment are not e3-compatible | P2 confirmed from config/checkpoint |
| Environment/batch | reference 4096 env and corresponding batch | resource-limited target uses 1024 env and batch | Algorithmic settings can align, throughput/noise statistics cannot be claimed identical | P2 confirmed resource limitation |

The shared additions for Go2/M1 (generic dimensions, direct actors, optional normalization and diagnostics) are gated away from the encoded G1 path in the inspected code. No evidence justifies changing Go2/M1 defaults or rewriting the G1 architecture during the current Go2 run.

## C. Reward anomaly analysis

### C.1 Observed terms

The 211,050 checkpoint full benchmark reported the following mean per executed step:

| positive term | measured | theoretical max | measured / max |
|---|---:|---:|---:|
| `motion_global_root_pos` | `0.00808052` | `0.010` | `0.808 |
| `motion_global_root_ori` | `0.00320556` | `0.010` | `0.321 |
| `motion_body_pos` | `0.01223548` | `0.020` | `0.612 |
| `motion_body_ori` | `0.0000102261` | `0.020` | `0.000511 |
| `motion_body_lin_vel` | `0.01244611` | `0.020` | `0.622 |
| `motion_body_ang_vel` | `0.00787165` | `0.020` | `0.394 |

Positive terms sum to about `0.04385` of the `0.10` theoretical tracking maximum. Regularizers add approximately `-0.000600`, consistent with the reported benchmark around `0.43` after normalization. The largest positive deficit against the theoretical maximum is body orientation (`~0.01999`), followed by body angular velocity (`~0.01213`), body position (`~0.00776`) and linear velocity (`~0.00755`). Root position is comparatively healthy; root orientation is materially behind.

CuboiLeo's documented e3 24k reward-term means were approximately root position `0.0064`, root orientation `0.0095`, body position `0.0191`, body orientation `0.0149`, body linear velocity `0.0178`, and body angular velocity `0.0168`. These are not the same checkpoint/protocol, but they show why `motion_body_ori≈1e-5` is a three-order-of-magnitude anomaly rather than a normal “train longer” difference.

### C.2 What is confirmed

- Static reward formula comparison found no coefficient, sigma, dt, q/-q or body-order mismatch.
- CPU G1 FK using `data/npz/lafan1/walk1_subject1.npz` matches stored body positions with max error `1.34e-7` and mean error `3.40e-8`.
- Quaternion comparison after sign minimization has max-component error `2.98e-7`, mean `1.58e-7`.
- Feeding an exact reference state to the same analytic reward yields approximately the theoretical maxima: root terms `0.01`, body position/orientation and velocity terms `0.02`, total `0.10`.

Thus the static reward implementation can express the expected optimum. The observation `~1e-5` is confirmed, but a reward math bug is not confirmed.

### C.3 Highest-priority unresolved checks

1. Compare one real MuJoCo state and one imagined state field-by-field at the same reference frame: anchor frame, root/body pose, velocity, quaternion order, history ordering, and action clipping.
2. Verify live RewardManager body indices and reference body tensors against the CPU golden frame; the NPZ/FK check does not exercise the live GPU environment.
3. Verify real and imagined rewards use the same anchor timestamp and reanchoring transform.
4. Inspect orientation error distributions, not only means, and correlate them with action saturation and failed clips.
5. Reproduce a minimal real-vs-imagined transition after GPU becomes available.

These are `PENDING_GPU_VALIDATION`; they must not be silently promoted to confirmed bugs. Do not widen sigmas, increase reward weights, or reduce action noise to hide the anomaly.

## D. E3 Alignment Manifest for the new Scratch run

### D.1 Intended algorithmic configuration

- **Robot/task:** G1, `motion_tracking`, existing G1 MJCF/asset.
- **Data:** `data/npz/lafan1`, all 40 clips, no `data.clips` filter. The current manifest contains: `dance1_subject1`, `dance1_subject2`, `dance1_subject3`, `dance2_subject1`–`dance2_subject5`, `fallAndGetUp1_subject1`, `fallAndGetUp1_subject4`, `fallAndGetUp1_subject5`, `fallAndGetUp2_subject2`, `fallAndGetUp2_subject3`, `fallAndGetUp3_subject1`, `fight1_subject2`, `fight1_subject3`, `fight1_subject5`, `fightAndSports1_subject1`, `fightAndSports1_subject4`, `jumps1_subject1`, `jumps1_subject2`, `jumps1_subject5`, `run1_subject2`, `run1_subject5`, `run2_subject1`, `run2_subject4`, `sprint1_subject2`, `sprint1_subject4`, `walk1_subject1`, `walk1_subject2`, `walk1_subject5`, `walk2_subject1`, `walk2_subject3`, `walk2_subject4`, `walk3_subject1`–`walk3_subject5`, `walk4_subject1`.
- **Goal:** `root_joint`, 68D motion frame × 10 future/history interface as implemented, 9D position anchor, total actor goal 689D; `anchor_orientation_obs=false`.
- **Observation:** 10-frame history, 93D proprioception/frame, reference goal future frames 10, encoded motion actor; RunningNorm=false.
- **World Model:** state 71D, action 29D, xi 33D, window 8, imagination horizon 32, explicit integration, UTD 0.4, 10 WM batches per update, one policy update per WM update.
- **Sampler/replay:** return mode, replay capacity 256 in the proposed low-memory run, bin frames 50, uniform/hard/memory ratio 0.5/0.4/0.1, softmax temperature 25, gamma .99. The reference uses a larger replay/batch budget where applicable; capacity 256 is a resource choice and should be recorded as a deviation if changed.
- **Seed:** 0.
- **Initialization:** random Actor, Critic and World Model; no `--resume`, no `--init-wm-from`, no old G1 checkpoint.
- **Learning rate:** initial actor/critic/WM lr defaults `2e-4`; cosine schedule, `total_iterations=320000`, `lr_final_ratio=0.03`, `lr_decay_fraction=250000/320000` (`0.78125`). `21000` is a stop point only, not the schedule length.
- **Resource setting:** 1024 environments, WM batch 1024, policy batch 1024. This aligns the algorithm and observation/reward path but not the reference e3 compute configuration, which used 4096 environments.
- **Checkpointing:** regular interval 4000; exact 21000 final checkpoint is saved by normal loop exit after the stop condition.
- **Evaluation:** all 40 clips, first frame, no observation/action noise, no reset noise, no domain randomization, full-clip execution and identical termination definition. Normalized return is `sum(reward)/(0.1 * executed_steps)`; report mean, step-weighted mean, failed clips, per-clip score and each reward term.

### D.2 Remaining deviations

1. 1024 environments/batches versus e3's 4096; this is required by the RTX 4070 SUPER 12GB resource limit.
2. The exact e3 hardware/throughput and any undocumented launcher defaults cannot be reproduced from this machine.
3. Replay capacity 256 in the proposed command is lower than the generic default 1000 and must be treated as a resource choice; if RAM allows, use the reference capacity instead.
4. A new optional `stop_at_iteration` field is prepared but not applied to the active worktree while Go2 is running.
5. Full 344,350-iteration G1 benchmark is absent; no GPU evaluation was fabricated.

The Breezewrf full-body 268D/2686D-style and RunningNorm experiment is deliberately excluded from this e3 manifest.

## E. Prepared changes, checks and commands

### E.1 CPU checks already completed

- `pytest tests/test_phase3_regularization.py`: 3 passed.
- Reference analytic-head tests: 5 skipped because the optional assets/dependencies are unavailable.
- `python3 -m compileall -q src/supermimic scripts/eval_benchmark.py`: passed.
- CPU FK/quaternion/reward golden-state checks: passed as reported in section C.

No check initialized CUDA or a MuJoCo training environment.

### E.2 Isolated stop patch

Candidate patch: [`g1_21k_stop_at_iteration.patch`](artifacts/g1_344k_audit/g1_21k_stop_at_iteration.patch).

It adds `ScheduleCfg.stop_at_iteration: int | None` and checks it after normal periodic checkpoint logic. The loop's existing post-loop `ckpt_latest.pt` save therefore preserves the exact final global iteration. It leaves `total_iterations` and the cosine LR schedule unchanged, uses absolute global iteration on resume, and does not alter Go2/M1 defaults.

The patch dry-run and AST syntax check passed on a temporary copy. It is **not applied** to the active worktree because the Go2 process is still using the installed/current code path. After Go2 finishes, apply and verify the patch in an isolated review step before training. The currently installed `sm-train --help` does not expose `--schedule.stop-at-iteration`; do not run the command below until the patch is applied and CLI parsing is rechecked.

### E.3 G1 21k Scratch command (prepared, not executed)

```bash
# Run only after Go2 has finished, the stop patch is applied, and --help shows the option.
export CUDA_VISIBLE_DEVICES=0
/home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name g1_e3_scratch_1024_21k_20261028 \
  --log-root logs/g1_e3_scratch \
  --device cuda:0 \
  --env.robot-name g1 \
  --env.task-name motion_tracking \
  --env.num-envs 1024 \
  --env.anchor-obs True \
  --env.no-anchor-orientation-obs \
  --env.motion-goal-schema root_joint \
  --env.history-length 10 \
  --env.future-frames 10 \
  --env.replay-capacity 256 \
  --env.sampling-mode return \
  --data.npz-dir data/npz/lafan1 \
  --world-model.no-running-normalization \
  --policy.goal-encoding encoded \
  --policy.action-output identity \
  --wm-trainer.window 8 \
  --wm-trainer.batch-size 1024 \
  --wm-trainer.batches-per-update 10 \
  --policy-trainer.batch-size 1024 \
  --policy-trainer.horizon 32 \
  --schedule.total-iterations 320000 \
  --schedule.stop-at-iteration 21000 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.policy-updates-per-wm-update 1 \
  --schedule.lr-schedule cosine \
  --schedule.lr-final-ratio 0.03 \
  --schedule.lr-decay-fraction 0.78125 \
  --schedule.warmup-iterations 40 \
  --schedule.checkpoint-interval 4000 \
  --seed 0
```

The command intentionally contains neither `--resume` nor `--init-wm-from`. `data.npz-dir` with no clip filter selects all 40 current LAFAN1 files. Do not set `--schedule.total-iterations 21000`; that would change the cosine schedule and reach its decay floor too early.

### E.4 Safe resume after the 21k stop

Use the same e3-compatible settings and preserve the 320k schedule. The stop option must be omitted or set above the desired absolute target. The checkpoint path is the exact `ckpt_latest.pt` written by the 21k run:

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name g1_e3_scratch_resume_to_320k_20261028 \
  --log-root logs/g1_e3_scratch \
  --device cuda:0 \
  --resume logs/g1_e3_scratch/g1_e3_scratch_1024_21k_20261028/ckpt_latest.pt \
  --env.robot-name g1 \
  --env.task-name motion_tracking \
  --env.num-envs 1024 \
  --env.anchor-obs True \
  --env.no-anchor-orientation-obs \
  --env.motion-goal-schema root_joint \
  --data.npz-dir data/npz/lafan1 \
  --world-model.no-running-normalization \
  --policy.goal-encoding encoded \
  --policy.action-output identity \
  --wm-trainer.window 8 \
  --wm-trainer.batch-size 1024 \
  --wm-trainer.batches-per-update 10 \
  --policy-trainer.batch-size 1024 \
  --policy-trainer.horizon 32 \
  --schedule.total-iterations 320000 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.policy-updates-per-wm-update 1 \
  --schedule.lr-schedule cosine \
  --schedule.lr-final-ratio 0.03 \
  --schedule.lr-decay-fraction 0.78125 \
  --schedule.checkpoint-interval 4000 \
  --seed 0
```

A resume cannot restore the old replay tensor because replay is not serialized; record that limitation in the run metadata. Optimizer state, iteration, scheduler parameters and sampler state are restored by the checkpoint path.

### E.5 Benchmark commands (prepared, not executed)

Run only after GPU is free, one checkpoint at a time:

```bash
for it in 0004000 0008000 0012000 0016000 0020000 0021000; do
  /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/eval_benchmark.py \
    --ckpt "logs/g1_e3_scratch/g1_e3_scratch_1024_21k_20261028/ckpt_${it}.pt" \
    --npz-dir data/npz/lafan1 \
    --num-envs 1024 \
    --device cuda:0 \
    --out "logs/g1_e3_scratch/benchmark_${it}.json"
done
```

The exact `ckpt_0021000.pt` name depends on the final-save implementation; if it is only present as `ckpt_latest.pt`, evaluate that file and record the actual checkpoint iteration from its metadata. This loop is a template and was not run in this audit.

### E.6 TensorBoard and CPU log parsing

```bash
tensorboard --logdir logs/g1_e3_scratch
python3 - <<'PY'
import csv
p = 'docs/artifacts/g1_344k_audit/history_metrics.csv'
with open(p, newline='') as f:
    rows = list(csv.DictReader(f))
print('rows:', len(rows), 'first:', rows[0] if rows else None)
PY
```

## F. New Scratch result at 272650 iterations

The later from-scratch run `logs/g1_e3_scratch/g1_e3_scratch_1024_16h_20261028` was initialized without `--resume` or `--init-wm-from`, and its saved configuration confirms `goal_dim=689`, `root_joint`, position+orientation Anchor 9D, RunningNorm=false, 1024 environments, and 1024 WM/Actor batches. It reached iteration `272650` after 16 hours. The deterministic 40-clip benchmark is [`benchmark_272650.json`](/home/yifeiy/ws_supermimic/logs/g1_e3_scratch/g1_e3_scratch_1024_16h_20261028/benchmark_272650.json): mean normalized return `0.40223`, step-weighted `0.40508`, and 2 failed clips.

This rules out insufficient iteration count as the primary explanation. CuboiLeo e3 reports `0.609` at 8k, `0.792` at 16k, and `0.824` at 24k, while this run remains near `0.40` after more than eleven times 24k iterations. The scratch run's real body-orientation reward is already only `0.00120` at 8k, `0.000615` at 16k, and `0.000319` at 24k; at 272650 it is `0.000163`. The final benchmark term is `1.18e-5`, versus the reference e3 value `0.0149` at 24k. Continuing this same run is therefore unlikely to recover the reference behavior.

The TensorBoard evidence points to an optimization/distribution failure rather than a dead Actor: the Actor gradient remains nonzero (about `0.03` at the end), but imagined survival is `0.9998`, imagined return is about `1.82`, and imagined body-orientation reward is about `5.83e-4` while real body-orientation reward remains `1.63e-4`. At the same time, imagined action magnitude grows from `0.62` at 8k to `2.32` at the end. The policy is optimizing a favorable world-model rollout that does not predict the real orientation behavior.

The strongest confirmed experimental difference is effective batch/resource scale: this machine uses 1024 environments and 1024-policy/WM batches, whereas CuboiLeo e3 used 4096 environments and the documented early runs used batch 4096. This changes gradient noise and replay coverage and is a P1/P2 candidate, but it is not by itself a proven implementation bug. The 320k cosine schedule is also correct for the scratch run; its learning rate is still approximately `1.96e-4` at 24k, so the early failure cannot be explained by the later floor at `6e-6`.

The static comparison still finds no confirmed G1 reward, FK, quaternion-order, body-index, goal-layout, or World Model dimension error. The remaining highest-priority validation is a paired real/imagination transition audit that compares clean action, executed action, physical setpoint, state history, anchor frame and body orientation field by field. Until that check and a controlled batch-size/action-contract comparison are done, the responsible conclusion is: **not an iteration shortage; the current training distribution or real/imagination contract is failing, with 1024-vs-4096 batch scale the leading confirmed difference and action growth a strong symptom.**

## G. 12 GB GPU batch optimization

The trainers now support logical batches larger than the GPU micro-batch. Set `batch_size=4096` and `micro_batch_size=1024` to run four 1024-sample forward/backward passes, accumulate gradients, clip once, and perform one optimizer step. WM batches and Actor/Critic rollout batches use the same logical-batch semantics; leaving `micro_batch_size=None` preserves the previous behavior. The new path is CPU-tested and exposed by `sm-train --help`; peak GPU memory still requires a short validation run.

For the 12 GB card, the intended first configuration is:

```text
1024 environments
logical WM batch 4096, micro-batch 1024
logical Actor batch 4096, micro-batch 1024
UTD 0.4, policy updates per WM update 1
```

Do not increase `updates_per_env_step` to compensate for micro-batching: four micro-batches form one optimizer update. If the detached Critic payload still exceeds available memory, use micro-batch 512 and accumulation eight; the logical batch remains 4096.

## Decision

- **Ready for Scratch training now:** **NO** — Go2 PID 1948552 is still occupying the GPU, and the stop patch is intentionally unapplied.
- **Ready after Go2 ends and patch/CLI verification:** **YES**, subject to the listed GPU validation items.
- **Confirmed bugs:** no P0 reward/FK/quaternion bug found; latest old resume's 500k LR schedule is a confirmed P2 configuration deviation.
- **High-priority unresolved issues:** real-vs-imagined observation/action alignment, live body/reference indexing and reanchoring, replay terminal-label semantics, critic-loss outliers, and full benchmark coverage.
- **Blocking evidence gaps:** no later full 40-clip benchmark than 211,050; no live GPU validation of the body-orientation anomaly.
