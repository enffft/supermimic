# M1-Piper Whole-body Motion Tracking：实现与短训记录

## 状态

6 条动作数据已加载，M1 观察、奖励一致性与物理接口验收通过。4096 和 3072 env 在 MuJoCo Warp 凸碰撞/CCD 临时数组分配时 OOM。2048 env 首次试跑因 Actor 在 WM 拟合前启动，iteration 50 时 `H=32` imagination survival 只有 `1.57e-9`；随后增加 300 次 M1 WM-only warmup。第二次 2048 env 短训运行到 iteration 1050 后正常停止、保存并加载 checkpoint 完成逐条评测。六条动作都能播放结束，但严格跟踪 success rate 为 0%，这轮只验证训练与评测链路，不代表策略已经学会动作。

M1 4096、3072 OOM 日志分别在 `logs/m1_piper_motion_C_4096_15m/diagnostic_stop.json`、`logs/m1_piper_motion_C_3072_15m/diagnostic_stop.json` 和 `logs/m1_piper_motion_C_3072_15m_r3/diagnostic_stop.json`。这些 OOM 均发生在物理仿真碰撞阶段，没有做 WM 或 Actor optimizer 更新。`logs/m1_piper_motion_C_2048_15m/diagnostic_stop.json` 记录了首次 2048 试跑的近零 survival；第二次 2048 短训和评测摘要见 `logs/m1_piper_motion_C_2048_wm300_15m/run_summary.json`。原先物理检查过的 M1 容量 `nconmax=256, njmax=1024` 已恢复。

## 动作集、Goal 和状态接口

`training/loop.py::M1_MOTION_SETS` 使用经过筛选的 6 条 clip：

- `a_wheel_000000`
- `a_wheel_000002`
- `b_arm_000006`
- `c_assisted_000029`
- `c_assisted_000039`
- `c_assisted_000149`

A/B/C 分别是前 2、前 3、全部 6 条。`configs/m1_piper_first6.csv` 是默认 manifest；传入未被所选 motion set 接纳的 clip 会报错，避免用不同 Motion Bank 配置误恢复 checkpoint。

M1 `RobotSpec` 是 22 维关节动作：16 个腿关节加 6 个 Piper 臂关节。四个 `*_FOOT_JOINT` 为轮速度目标，其余 18 个用关节位置目标，落在 `envs/actions.py::MixedM1Action`；缺失的轮角位置不进入 M1 AMP/Goal 关节位置项。M1 AMP 维度 18。

训练默认 `anchor_obs=True`：每帧 Goal 有 18 维参考关节位置、18 维速度和 body/root 特征共 46 维；10 帧为 460 维，加 9 维 anchor-relative pose，最终 Goal `(N,469)`。Policy observation 为 10 帧、每帧 72 维，展平后 `(N,720)`。实际启动日志验证了 action `(N,22)`、policy observation 720 和 Goal 469。

## 真实与想象奖励

`envs/multirobot_env_cfg.py::make_multirobot_env_cfg` 安装 mjlab 的解析 tracking reward，`world_model/heads.py::AnalyticHeads.reward` 通过 FK 调用共享 tracking reward term；这使真实和想象端使用同一误差定义。M1 默认权重和标准差如下：

| 项 | 权重 | std | 实际核 |
|---|---:|---:|---|
| global anchor/root position | 0.5 | 0.3 m | `exp(-||e||²/std²)` |
| global anchor/root orientation | 0.5 | 0.4 rad | `exp(-e_quat²/std²)` |
| relative tracked-body position | 1.0 | 0.3 m | `exp(-mean_body ||e||²/std²)` |
| relative tracked-body orientation | 1.0 | 0.4 rad | `exp(-mean_body e_quat²/std²)` |
| global tracked-body linear velocity | 1.0 | 1.0 m/s | `exp(-mean_body ||e||²/std²)` |
| global tracked-body angular velocity | 1.0 | 3.14 rad/s | `exp(-mean_body ||e||²/std²)` |
| action-rate L2 | -0.1 | — | mjlab `action_rate_l2` |
| joint limit | -2.0 | — | 默认 `joint_limit_weight` |

真实环境与解析 reward parity 测试最大绝对误差约 `2.0e-4`；观察 parity 最大绝对误差约 `2.98e-8`。Wheel joint 的位置奖励不纳入不存在/无效的参考角度项。

## 共享训练核心

M1 复用同一套 `ReplayBuffer`、World Model、8-step autoregressive loss、Actor/Critic、32-step imagination、BPTT、Lambda Return、optimizer/checkpoint 流程。与 Go2 不同，M1 不启用 Go2 velocity Goal、velocity reward、OU exploration 或 A/B/C 速度课程。M1 先执行配置项 `schedule.m1_wm_only_iterations=300` 个 WM 更新事件，再开始 Actor/Critic BPTT；该预热是为避免首次试跑中随机初始化 WM 导致近零 imagined survival。动作采样仍由 M1 `MixedM1Action` 的 `action_noise_std` 管理。

`MultiClipMotionCommand` 默认按 return-based reset sampling 在 Motion Library 的 1 秒 bins 上采样：每步保存 reward 和 bin；episode 结束后计算 discounted return-to-go 并更新 `bin_return` EMA；`_sample_frames` 混合 uniform、hard-bin softmax 和 memory 分布。采样器状态、episode return 记忆和随机数状态随 checkpoint 保存。

M1 诊断记录总的 root position/orientation、arm joint、TCP 误差和每条 motion 的 return/success。评测脚本 `src/supermimic/scripts/evaluate_m1.py` 固定每个环境到一个选定 clip，关闭噪声与 DR，逐动作报告完成率及误差；它只做推理，不更新权重。

## 显存问题和资源调整

前两次 M1 启动观察到：

- 4096 env：Warp convex narrowphase 分配约 1.86 GB CCD 临时缓冲失败。
- 3072 env：初始化/forward 阶段仍 OOM；另一次只为测容量而把 M1 `nconmax` 降到 64，立即被模型报告的 216 个初始接触拒绝。将容量暂时设为最低可启动值 216 后，第一次 simulation step 仍在约 589 MB `multiccd_face2` 分配时 OOM，随后已恢复 `nconmax=256,njmax=1024`。

训练配置现在将 `env.num_envs` 设为可选值：未显式传参时，`Trainer` 为 M1 默认选择 2048，为 Go2/G1 默认选择 4096；CLI 显式传入的数值保持不变。M1 与 Go2 的资源表现不同：M1 22 关节、更多凸碰撞 geometry，模型的初始 contact 数量级也明显更高；Warp 会按 world 数量扩展 contact/CCD 数据。以上错误发生在物理仿真碰撞阶段，不是 Replay Buffer、World Model、Actor 的 batch 或 loss 导致。M1 当前默认使用 2048 并行环境，不降低 contact 容量、不禁用碰撞几何。

## 训练命令

在 `supermimic` Conda 环境、仓库根目录运行（2048 env，最多 15 分钟）：

```bash
PYTHONPATH=src python -m supermimic.scripts.train \
  --run-name m1_piper_motion_C_2048_wm300_15m \
  --env.robot-name m1_piper --env.task-name motion_tracking \
  --env.num-envs 2048 --env.replay-capacity 256 \
  --data.m1-motion-set C \
  --wm-trainer.window 8 --wm-trainer.batch-size 128 \
  --wm-trainer.batches-per-update 1 --wm-trainer.compile None \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 128 \
  --policy-trainer.compile None \
  --schedule.total-iterations 2000 --schedule.warmup-iterations 40 \
  --schedule.m1-wm-only-iterations 300 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 250 \
  --schedule.max-hours 0.25
```

训练结束后，对应 checkpoint 可逐条评测：

```bash
PYTHONPATH=src python -m supermimic.scripts.evaluate_m1 \
  --checkpoint logs/m1_piper_motion_C_2048_wm300_15m/ckpt_latest.pt \
  --motion-set C --output logs/m1_piper_motion_C_2048_wm300_15m/eval_C.json
```

## 2048 env 短训与逐动作评测

第二次短训参数、日志和 checkpoint 在 `logs/m1_piper_motion_C_2048_wm300_15m/`。M1 默认 env 数量现为 2048；显式传入 `--env.num-envs` 时保留用户值。

| 指标 | iteration 1050 |
|---|---:|
| WM 更新 / WM-only 更新 | 404 / 300 |
| Actor 更新 | 104 |
| 1-step / 8-step NSE | 0.0306 / 0.1248 |
| Term BCE / 预测终止率均值 | 0.0829 / 0.0795 |
| H=32 imagined survival | 0.1368 |
| imagined return / Actor grad norm | 1.062 / 0.0220 |
| 归一化真实 episode return | 0.820 |
| checkpoint | `ckpt_latest.pt`（194 MB，iteration 1050） |

`max_hours=0.25` 按每 50 iterations 检查一次，实际训练 loop 计时为 0.2708 h（约 16.2 分钟，含启动约 15 分钟以上）；正常输出 `max_hours reached` 并退出，无 OOM/NaN。

`evaluate_m1.py` 逐条固定 clip 的结果（评测无训练更新）：

| Motion | 完成 | Root pos err (m) | Arm joint err | TCP err (m) | Success |
|---|---:|---:|---:|---:|---:|
| a_wheel_000000 | 是 | 0.173 | 0.178 | 0.392 | 否 |
| a_wheel_000002 | 是 | 0.162 | 0.180 | 0.337 | 否 |
| b_arm_000006 | 是 | 0.093 | 0.541 | 0.521 | 否 |
| c_assisted_000029 | 是 | 0.093 | 0.277 | 0.358 | 否 |
| c_assisted_000039 | 是 | 0.098 | 0.502 | 0.490 | 否 |
| c_assisted_000149 | 是 | 0.095 | 0.374 | 0.457 | 否 |

六条全部完成播放，但 success rate `0.0`。平均 root position error 为 0.119 m，arm joint error 0.342，TCP error 0.426 m。最新逐条 JSON 在 `eval_C.json`；本轮 aggregate 和训练指标另存于 `run_summary.json`。

## 验收记录和限制

- M1 6 条选定 motion、joint order、wheel masking、实际/想象 observation parity、reward parity、Go2 reward parity/OU/replay 和物理目标检查通过。
- `tests/test_multirobot_acceptance.py -k 'not m1_eight_step_wm_backward_and_32_step_actor_gradient'`：6 passed，1 deselected；`tests/test_velocity_training.py`：4 passed。
- M1 八步 WM backward/32 步 Actor gradient 集成测试没有单独用低环境数启动训练；4096/3072 OOM 尝试没有 optimizer update。2048 短训实际完成了 104 次 Actor 更新，但只验证了有限的短程梯度链路。
- 4096 和 3072 均未完成训练；2048 首次试跑在 iteration 50 因近零 survival 停止且没有 checkpoint。第二次 2048 短训完成 300 次 WM-only warmup、104 次 Actor 更新，保存并载入 checkpoint 做了逐条评测；六条播放完成但 success rate 0.0，不得描述为 M1 已学会这些动作。

## 下一步

下一步先检查 M1 termination 概率在训练数据中的标签覆盖，并改进 imagination 后段的 tracking 误差；再基于逐条评测结果安排更长预算。不能只凭 aggregate return 增加训练，也不能把本次短训描述成成功策略。
