# M1-Piper v2 任务、奖励与顺序训练修订

日期：2026-11-05。状态：实现和短程验证；**尚未启动正式训练**。用户指定下一轮 PPO 3.5 小时 → SuperMimic 3.5 小时，等待明确的“开始训练”。所有验证使用 Conda `supermimic`，不使用 venv。

## 1. 这轮实验回答什么

仍然是无 motion prior 的 reward-only M1-Piper。沿用现有 mjlab 环境和下载的 `external/go2_velocity` 所采用的 mjlab/RSL-RL PPO；没有重新实现 PPO、GAE、rollout 或优化器。两者共享物理模型、动作映射、命令接口和真实环境奖励。SuperMimic Actor 通过冻结参数但保留输入梯度的 world model 做 BPTT。

这轮按用户要求，为 SuperMimic **单独增加了 H=32 的想象终点距离项**。因此这轮是相同真实任务、相同训练时间下的系统比较，不能标成“两个算法优化目标完全相同”的严格消融实验。论文表格应单列这个差异；之后可再做 SuperMimic 关闭该项的消融。

## 2. 新任务

| 项目 | v1（上轮） | v2（下一轮） |
|---|---|---|
| 环境 | 2048，固定 1024 L / 1024 M | 保持相同 |
| L 命令 | 机体坐标 vx/vy/wz | 保持相同 |
| M 目标水平位置 | TCP 初始朝向附近的小盒子 | 采样时机体中心水平半径 3 m 的圆盘 |
| 水平分布 | x/y 区间均匀 | 圆盘面积均匀：r=3√u，θ=2πv |
| 高度 | TCP 相对高度偏移，原任务约 0.66–0.96 m | 本环境地面以上 z∼U[0.03,1.10] m |
| 目标保持 | 每 episode 固定 | 每个 goal 固定为世界点，机体移动后不跟随机体 |
| 到达后的行为 | 等 episode 结束 | 稳定到达 8 cm 内后，本步计分，立即换点，不重置机器人 |
| episode 上限 | 10 s | 两算法共同改为 30 s，给远点移动留时间 |
| SM 想象 | 8×0.02=0.16 s | 32×0.02=0.64 s |
| 预算 | 2.5 h / 算法 | 3.5 h / 算法，PPO 完成后再跑 SM |

采样原点是每次换点时的底盘位置。内部状态和目标使用“世界方向、相对于本环境 origin 的坐标”；viewer 绘制时加回环境 origin。z 是相对本环境平地的高度，不是离机体的高度。只拒绝已落在当前 TCP 成功球内的点，避免采样即免费成功；**没有用 IK 筛掉低点、背后点或边界难点**。因此这轮主动包含需要底盘与机械臂共同运动的困难目标，不能保证所有点在当前关节边界和碰撞约束下可达。

L 命令维持 vx∈[-0.09,0.145] m/s，15% 停止、35% 后退、50% 前进；vy∈[-0.05,0.055] m/s；wz∈[-0.17,0.18] rad/s。M 的速度目标为零但不启用 L 速度跟踪奖励；M 以世界 TCP 点为任务输入，底盘可以移动。这里仍是 L/M 两类任务，不是同一环境同时跟踪速度和点，也不包含抓取、夹爪或 TCP 姿态任务。

成功条件：M、TCP 距离 <0.08 m、机体高度 ≥0.20 m、倾斜 <70°。按用户“到达立刻换点”的要求，使用首次满足条件的控制步，不再要求停留 0.5 s。

## 3. PPO 和 SuperMimic 的全部共享奖励

下表是**奖励率**，真实仿真和想象均只乘一次 dt=0.02 s。`isL`、`isM` 是任务门控；未标任务的项两类均开启。距离 d 是 TCP 到当前世界目标的欧氏距离。

| 项目 | 两算法共享的公式 / 权重 | 这轮变化 |
|---|---|---|
| L 平移速度 | 3 exp(-‖v_xy-v_cmd‖²/0.15²) | 无 |
| L 偏航速度 | exp(-(wz-wz_cmd)²/0.25²) | 无 |
| M 粗到点 | 2 exp(-d²/σ²) | σ: 0.50 → 1.50 m |
| M 精到点 | exp(-d²/0.08²) | 无 |
| M 接近进度 | 0.25 tanh((d_prev-d)/(dt×0.5)) | 无 |
| M 近点低速 | exp(-d²/0.08²) exp(-‖v_TCP‖²/0.10²) | 无；到达即换点，已不要求保持 |
| 直立 | 0.5 exp(-(1+g_body,z)/0.1) | 无 |
| 底盘高度 | 0.5 exp(-((h-h_default)/0.1)²)，h_default≈0.47 m | 无 |
| 竖直速度 | -0.2 vz² | 无 |
| 横滚/俯仰角速度 | -0.05‖w_xy‖² | 无 |
| 腿部姿态 | -0.02 Σ(q_leg-q_default)² | 无 |
| L 手臂收纳姿态 | -0.01 Σ(q_arm-q_default)² | 无 |
| 动作变化 | -0.01 mean((a-a_prev)²) | 无 |
| 关节位置越界 | -Σ violation²，排除轮子角度 | 无 |
| 关节速度 | -1e-5 mean(qd²) | 无 |
| 关节加速度 | -1e-7 mean(qdd²) | 无 |
| 控制幅度 | -0.005 mean(a²)，是动作代理项而非电机能耗 | 无 |
| M 底盘速度 | -0.1(‖v_xy‖²+wz²) | 无 |
| **M 距离相关时间成本** | **-0.5 d/(d+0.5)** | **新增，两算法共享** |
| **M 首次到达事件** | **每个完成 goal +12**，实现为奖励率 12/dt | **新增，两算法共享** |

v1 实际启用前 18 项。新增两个项的通用 dataclass 默认权重均为 0；只有 v2 preset 启用，所以旧 checkpoint、旧任务评测继续按原奖励和目标约定恢复。

时间成本的单位是 reward/s：d=0、0.1、0.5、1、3 m 时分别是 0、-0.0833、-0.25、-0.3333、-0.4286。每多花一个控制步都会产生一次成本，距离越远，每秒成本越大，并有界于 -0.5；这不是一个所有距离相同的常数时间惩罚。它是第一轮参数，而不是已经调优的最佳权重。

粗距离核扩大到 1.5 m，是为了给 3 m 目标提供有效过程信号：d=3 m 时该项从约 4.6e-16 提高到约 0.0366。进度项仍提供接近/远离的有符号信号。

加入到达事件奖励的原因：如果到达就跳到另一个远点，策略可能选择停在阈值外，反复收集近点稠密奖励。原有 M 正奖励率保守上界约 5.25/s；γ=0.99、dt=0.02 时，相同目标无限等待的正奖励上界约 10.5。一次到达 +12 超过这个保守界，减少“宁可不完成”的目标切换诱因。这只是尺度依据，实际仍应检查成功率、到达时间、跌倒和是否在阈值附近徘徊。

## 4. SuperMimic 专属终点项

```
J_actor = J_shared_imagined_return_and_bootstrap
          - gamma^32 × 1.0 × distance(TCP_predicted_32, current_world_target)
```

只对 M 样本启用，距离采用数值平滑的欧氏范数 `sqrt(||delta||²+1e-8)-1e-4`。此项每条想象轨迹只算一次，不乘 dt，不放入真实环境奖励、不用于 PPO。日志为 `imag_r/terminal_distance`，是带符号的未折扣轨迹终点奖励；折扣在 Actor objective 处应用。γ=0.99 时 γ^32≈0.725。

如果想象中在发生物理跌倒之前已经稳定完成当前 goal，则这一 goal 的剩余终点距离成本为零。想象在首次完成处截断当前 goal 的回报和 value bootstrap，成功奖励只拿一次；不会继续对旧点领奖，也不会凭空采样下一个随机目标来反传。真实机器人不结束 episode，下一步直接观察新点。因此 SM 的想象近似是“当前 goal 的局部规划”，PPO 观察到的真实 MDP 则包含换点后的继续执行，这也是需要在比较中记录的设计差异。

当前 `imag_dhat_mean` 和想象诊断的结束概率包含“成功完成当前 goal”，不能直接当作跌倒概率；真实跌倒另看实仿真 fall 指标。

终点负距离成本不乘物理存活概率，避免简单通过预测死亡把负成本抹掉。Critic 继续拟合共享逐步奖励的 GAE；显式终点项直接加入 Actor 目标，不混成每步环境奖励。WM 和 Critic 在 Actor 更新时参数冻结，状态/动作的计算图仍保留。

恢复 32 步只修正了想象长度，不意味着 world model 的 32 步预测已准确，更不等同于完整复现原论文。模型误差、可达性和梯度方向仍可能限制效果。

## 5. 数据、观测和评测边界

一次成功控制步按如下顺序处理：物理推进 → 使用旧 target 计算 reward → command 换到新 target → 新观测；Replay 写入该步动作真正使用的旧 context，并把新 context 保存为下一步快照。换点不会被记录成跌倒，也不会重置历史观测。

评测沿用 checkpoint 保存的 v1/v2 环境。v1 保留原来的 episode 终点距离和 0.5 s hold 成功率；v2 新增每 goal 的初始距离、完成时间、成功/跌倒/超时、结束距离和目标坐标。评测结束时尚未完成的 goal 标成 censored，不悄悄算失败，也不计入已结束 goal 成功率；同时报告每个 M 环境每分钟成功次数，避免只看已结束样本造成误读。v2 的 `M_hold_success_rate` 为 null，不能与旧 hold 率混用。

默认正式后评测：balanced、固定前进 L(vx=0.12)、纯 M，64 env、60 s、seed=123，无训练噪声或 DR。两算法使用相同评测约定；v2 报告脚本拒绝目标协议、时长、seed 或任务类型不一致的数据，不与 v1 的小目标结果自动混比。

Play 沿用红球=目标、青球=TCP、黄色连线，并在换点时打印目标坐标与误差。显式传入 `--m1-target-world` 表示用户固定点，关闭自动换点；它是固定点演示，不是随机连续换点评测。

## 6. 训练配置与预算

两者使用 2048 env、固定均分 L/M、seed=2026、相同 Action22/Goal8、10 帧历史、相同仿真频率、关节/PD/噪声/DR。PPO 的 rollout 长度维持 32、逻辑 minibatch 1024、5 epochs，使用上游算法。SuperMimic WM/Actor 逻辑 batch 维持 128，WM 每事件 4 fits；Actor microbatch 保持 32，累加后逻辑 batch 仍为 128。WM microbatch 32，禁用 Torch 编译的多进程内存峰值。

每个算法各有 3.5 小时训练预算，包含采样、优化、训练期间存盘；不包含环境初始化、训练后评测。停止发生在完整更新边界，可能略超时，不能保证秒级同时结束。PPO 与 SM 真实样本量、更新数仍会不同，必须随结果报告。

SM 保留原来的 300 个 WM-only 更新事件和 500 Actor ramp 事件；这没有随着 H=32 自动放大。WM 精度是否满足长想象，应在训练结果中单独检查，不能仅看 loss 或本次链路 smoke。

## 7. 顺序启动与恢复

当前只准备好了代码，没有启动以下正式命令。正式启动由用户确认后进行。

M1 默认先 PPO，再 SuperMimic，各 3.5 小时，然后自动评测两者并汇总：

```bash
cd /home/yifeiy/ws_supermimic
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_comparison.py
# 以上只打印计划。明确要开始后才加：
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_comparison.py --launch --detach
```

中断后保持原 tag、预算和 env 数恢复：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_comparison.py --launch --detach --resume
```

新 Go2 启动器默认用新 tag `2p5h_seed2026_v2`，3072 env、PPO 2.5h → 导出 → SM 2.5h → 导出：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_go2_comparison.py --algorithm both
# 同样仅在准备开始时：
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_go2_comparison.py --algorithm both --launch --detach
# 中断恢复：
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_go2_comparison.py --algorithm both --launch --detach --resume
```

两种队列共用 GPU 锁及现有 RAM/watchdog，只启动一个训练子进程。`--detach` 独立进程组、日志重定向，终端关闭后继续。原子写入阶段状态；重启跳过已完成训练/评测/导出，失败阶段可以重试。恢复训练使用原生模型和优化器 checkpoint，按已保存训练用时补足总预算，不再无条件从零训练 2.5/3.5h。PPO 原生文件改成临时文件后原子替换；SM 每 5 分钟同时更新 wall checkpoint 和 `ckpt_latest.pt`，而非只更新历史 wall 文件。PPO 每 10 更新存盘。

`--resume` 要求本轮队列已有 manifest，以校验原始预算和参数；不会自动收编缺少 manifest 的历史目录。底层 worker 仍可显式传入原生 checkpoint 做人工续训。

恢复不是逐 bit 的仿真回滚：物理场景/Replay 重新初始化，未保存的进展会丢失，预算只累计检查点保存的训练用时，故崩溃前最后一个存盘区间的计算可能重做。旧 checkpoint 没有新时间字段时无法自动重建精确已耗时；本修订主要保证新一轮可恢复，不能把历史中断无损还原。

旧 Go2 中断目录的日志停在 compiler 初始化附近，没有完整异常栈，不能断言唯一根因。新比较 worker 使用 eager，避免初始化编译进程带来的资源峰值；同时修复了旧脚本拒绝已有 checkpoint、缺少后台模式和阶段恢复的问题。

队列日志/状态：

- M1：`logs/m1_comparison/v2_lm50_2048_3p5h_seed2026/{launcher.log,status.json}`。
- Go2：`logs/go2_comparison/queue_2p5h_seed2026_v2_both/{launcher.log,status.json}`。
- 训练 checkpoint 分别仍在 `logs/m1_ppo/<tag>/`、`logs/m1_reward/<tag>/`、`logs/go2_comparison/{ppo,supermimic}_<tag>/`。
- M1 自动结果：各训练目录 `evaluation_*.json`，队列目录 `comparison.json` / `comparison.md`。

## 8. 已完成验证

验证清单与最终数值见本节末尾追加记录。短程验证只验证实现和资源可运行，不代表新任务已经学会，也不能承诺 3.5 小时就能稳定完成全范围目标。

- CPU 契约测试：目标面积/高度分布、旧 checkpoint 默认行为、PPO/SM 环境一致性、距离时间成本、终点梯度方向、M/L 门控、想象成功一次和旧目标 bootstrap 截断、失败阶段隔离及恢复。
- M1 H32 4-env smoke：真实 WM 和 Actor 更新；真实/想象奖励最大差 0，Goal8 差 0，TCP FK / MuJoCo 差 2.6822e-7 m；Actor 更新 3 次。
- M1 连续换点实仿真 probe：首次到达只给一次 +12；robot 未 reset；旧目标保留在奖励和 Replay 行中，新目标进入下一帧 Goal8 / recorder 快照；非零 env origin 正确。
- Go2 顺序实测：小规模上游 PPO → 临时 rl_sar 路径导出 → SuperMimic → 临时路径导出，全部完成，未覆盖正式部署策略。
- M1 顺序实测：小规模 PPO → H32 SuperMimic → 两者共 6 个短评测 → 匹配报告，全部完成。

证据目录：`logs/m1_reward/v2_h32_validation_20261105/`、`logs/m1_comparison/v2_contract_validation_20261105/`、`logs/m1_comparison/v2_sequence_validation_20261105/`、`logs/go2_comparison/queue_validation_v2_20261105_both/`。本地 logs 不提交为训练成果；这些是短测 artifact。


### 最终验证记录

- **27 项针对性测试全部通过**；相关新文件 `ruff --select F` 与 `git diff --check` 通过。仅有现有 TorchScript 弃用提示。
- **2048 env / H32 / Actor logical batch 128 / microbatch 32 / WM 4 fits**：14 个真实向量步、1 次 WM 事件和 1 次 Actor 更新；固定 1024 L / 1024 M；梯度有限。进程峰值 RSS **4864 MiB（约 4.75 GiB）**，PyTorch CUDA peak allocated **618 MiB**（不含驱动/框架上下文），本次 Actor 更新约 **12.9 s**。这是缩短预热的一次链路验证，不代表已收敛。
- Go2 最新完整顺序短测：`validation_v4_20261105`，SM 实际 **2 次 WM 事件、1 次 Actor 更新、H32**，两次导出均完成。测试导出位置 `/tmp/go2_sequence_rl_sar_v4_20261105`，未改正式 rl_sar 策略。
- 实测原生恢复：Go2 PPO、M1 PPO 均从已完成 1 次更新续到 **2 次 / 每 env 64 步**；Go2 SM 从第 10 步续到 **22**；M1 SM 从第 24 步续到 **26**。训练耗时元数据继续累计。测试数据见 `logs/m1_comparison/v2_contract_validation_20261105/resume_validation.json`。
- 后台恢复实测：对已完成的 Go2 短测队列执行 `--detach --resume`，四阶段全部跳过，无重复训练或导出。
- 新评测协议重新跑通双方 6 个 case，并用独立测试 checkpoint 缩短 episode 验证边界：**12 个已结束 goal、4 个截止时仍活跃的 censored goal**；v2 不输出旧 hold 成功率。
- 额外排查并修复 Go2 smoke 的 batch 覆盖：Trainer 会从 env 的 `velocity_*_batch_size` 恢复逻辑 batch，因此 smoke 也需同时改这两个字段；最终短测实际 batch=4。正式 Go2 batch 仍沿用参考配置，未被 smoke 修改。

精简验证记录：`docs/results/m1_v2_validation_20261105.json`。所有短测已完成，正式命令只打印过计划，未启动 3.5h 或 2.5h 作业。
