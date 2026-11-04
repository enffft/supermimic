# M1 Action22 P0/P1 修补与训练退化诊断（2026-11-04）

## 结论摘要

这次证据支持一个比“Actor 梯度没传过去”更具体的解释：Actor 的优化确实获得了非零梯度，但旧训练没有 WM 对真实动作响应的准入保护；Actor 的动作偏离 Prior 后，闭环性能会随 checkpoint 和任务大幅波动。固定种子残差扫描显示，最终 Actor 在 L 前进任务中的全量残差明显破坏了 Prior Direct 的运动表现；M 任务则出现部分目标上的改善，但稳定保持仍为零。因而“Actor 修正过大/方向不可靠”已由真实仿真支持；WM 动作 Jacobian 是否是主要成因，还要等配对有限差分结果，不能仅凭 Replay NSE 归因。

已修复 M 任务重复终端权重和提前终止获利漏洞，增加物理容差归一化、边界平滑项、独立梯度诊断、bounded-residual Actor V2/V5 契约，以及严格 fail-closed 的 Actor WM Gate。Prior 在所有现有实验中保持冻结。

**本轮没有启动新训练。** 本机只有旧 V4 checkpoint 可用于动作响应审计；它与新 Actor V2 的 Gate 契约不兼容，且此次有限差分仅采 16 个物理状态，远低于 Gate 的 128 状态覆盖要求。因此训练脚本会拒绝 Actor WM Task BPTT。L-only 短训标记为 `not_run`，没有把 Gate 绕过或把旧 checkpoint 假称为 V5。

## 工作树与来源

- 仓库：`/home/yifeiy/ws_supermimic`
- 分支：`main`（未切换分支、未建 worktree）
- 基线 HEAD：`4ba2991de8ec8ff8bff88a13c51a87674c72d430`
- 本次不执行 reset、stash、clean、commit 或 push。
- 保留原有 Go2/G1 未提交成果；本次只在已有共享 `config.py`、`training/loop.py` 中添加 M1 专属改动，没有触碰 Go2/G1 代码路径。
- 旧一小时实验的五个 V4 Actor 快照和 V3.3 Prior 二进制均在本机；SHA256、评估设置和逐场景数据见 `results/m1_action22_p0p1_20261104/alpha_sweep/summary.json` 与同目录 JSON。

## 修补内容

### M 任务目标与损失尺度

`src/supermimic/motion_prior/losses.py` 现在按物理容差计算 L 速度误差和 M TCP 误差。初始配置默认使用 L `vx/vy=0.05 m/s`、`wz=0.10 rad/s`、M TCP 距离 `0.08 m`、TCP 速度 `0.05 m/s`；这些是可配置的首轮标度，不代表已经通过训练验证的最佳容差。

M 目标包含非终端过程距离、只计算一次的终端距离、接近目标时低 TCP 速度、底盘漂移与底盘速度/角速度约束。旧目标把 terminal distance、path distance 和 progress 对同一目标重复加权；当前合成状态下，末帧距离的梯度相对普通过程帧从约 `201:1` 降为约 `9:1`。倾角/高度、关节限位等 dense feasibility 项继续保留。10 帧 rollout 约 0.2 秒，**不能替代真实 0.5 秒稳定保持评估**。

### 固定时域终止代价

`src/supermimic/motion_prior/rollout.py` 仍保留离散 hard alive mask 作为想象轨迹诊断，同时保存可微 cumulative survival probability。Task/Tracking 在 hard mask 后不会被清空：失败分支沿用最后安全状态，剩余固定时域仍承担未完成任务成本；另对失去的剩余控制时域计成本，单位是配置的任务容差，而不是任意放大的常数。WM 预测终止、解析几何安全项和真实仿真 timeout 仍分别记录。

CPU 合成反例（初始 TCP 到目标 0.30 m、安全静止）：

| 情况 | 新总损失 | 说明 |
|---|---:|---|
| 安全存活 10 帧 | 6.50 | 过程与终端目标成本保留 |
| 第 3 帧后失败 | 约 7.20 | 吸收状态保留目标成本，另计失去时域 |
| 第 1 帧失败 | 7.50 | 不能通过清空 mask 获得更低损失 |

原反例 `3.3` 安全静止优于 `1.0` 立即失败的顺序已反转为 `6.5 < 7.5`。低于安全高度的状态还额外承担解析几何 safety cost。测试同时覆盖 L 大速度误差、M 远/近目标、低/高失败概率、有限动作梯度与无 NaN/Inf。

### Loss 与梯度诊断

物理容差缩放后，离线静止合成例子的原始 task 数值为：L `vx=0.12 m/s` 时 `0.95`，M 目标距 TCP `0.30 m` 时 `6.50`。数值比约 `6.84`；这不是参数梯度比。对未来 TCP 轨迹变量的离线梯度，末帧/首帧约 `9:1`。

`actor_loss_gradient_diagnostics()` 通过独立 `autograd.grad` 记录加权 loss 对共享 Actor 参数和 Action22 输出的梯度范数，并比较 L/M、Task/Tracking、Task/BC 的余弦；它不写入 optimizer `.grad`。配置已经接入训练事件日志。历史 V4 checkpoint 未保存 Replay，因此无法从训练数据重算当时各项的共享参数梯度；本轮也没有 Actor BPTT 训练，所以没有真实 V5 Actor 梯度观测。不能把合成梯度或单元测试值冒充为历史训练梯度。

但历史事件日志有**总 Actor 参数梯度范数**，足以判断梯度是否断开：70 个记录点的 `actor/grad_norm` 均非零，范围 `2.72–19.99`，中位数 `9.33`。旧一小时训练实际为 WM `1584` 次、Actor `1384` 次、Prior `0` 次更新。与此同时，训练 `actor/task` 从首个记录 `3.585` 降到最终 `1.699`，`actor/total` 从 `4.281` 降到 `1.955`，但 tracking error 从 `0.0118` 升到 `0.1449`、BC loss 从 `0.00020` 升到 `0.2544`；L 真实评估也随 checkpoint 恶化。Replay 上 8-step NSE 的中位数是 `0.838`（范围 `0.252–2.939`，最终 `0.628`），仍不是独立 held-out 动作响应验证。这排除“Actor 梯度完全没传到网络”的说法，反而提示它获得了有效梯度、逐渐离开参考动作，却在优化一个无法可靠预测真实 L 运动的 WM Task 目标。总梯度不能分辨 L/M 或 action-space 梯度夹角；该分解现在会在新的、具备 Gate 证据的训练中记录。原始值见 `historical_training_gradient_and_loss.json`。

### Actor bounded residual 与 checkpoint 契约

新训练使用 `m1_action_reference_actor_v2_bounded_residual`，归一化 Action22 的零残差初始化精确复制 Prior，并把腿、Piper、轮组残差分别限制在 `0.05 / 0.05 / 0.10`。这些是明确可调的候选信任域，不是已经验证的机器人安全上限。动作契约的非对称映射把归一化 Action22 转成真实关节设定点/轮速；限幅仍由原 V3 Action Contract 和机器人限制共同决定。新 checkpoint 格式为 V5，旧 V4 只读评估，拒绝静默恢复进 V5 训练。

按当前 L high-carry V3 RobotSpec，把该 normalized residual cap 通过真实 asymmetric setpoint mapping 换算成**单次位置/速度目标变化上限**（不代表关节实际运动）：腿关节各自约 `0.020–0.045 rad`（全组最大 `0.0451 rad`）；Piper 约 `0.014–0.114 rad`（全组最大 `0.1144 rad`，最大来自关节 2）；每个轮子的速度目标上限为 `0.10 × 5 = 0.50 rad/s`。这些物理界限已在单元测试中用 `mixed_m1_setpoints()` 验证。

历史 V4 alpha 消融是外部评估混合：`clip(u_ref + alpha * (a_actor-u_ref), -1, 1)`，不是 V5 bounded residual 训练。它报告腿/臂/轮的归一化 residual RMS；该旧评估未逐关节记录换算后的物理目标变化，因此不能把归一化 residual RMS 直接称为角度或 rad/s。

### Actor WM Gate 与 Prior 回滚

`M1AlternatingTrainer` 对 Actor BPTT 使用 `reference_only / tracking_diagnostic / wm_task_gated` 模式。默认 `wm_task_gated` fail-closed；Gate 要求真实配对动作响应与 task-loss 方向证据，同时匹配当前 WM SHA、V5 checkpoint、RobotSpec、V3 Action Contract、物理参数、任务分布和 Reset 契约，并至少覆盖 128 个 held-out 物理状态。Replay NSE 达标不放行。

Gate 通过时，WM 在 Actor 更新期间冻结。Gate 未通过时，Actor 保持 reference-copy 基线、仍可做 WM 训练和诊断，不做风险性 Task BPTT。Prior 默认继续冻结，优化器更新数为 0。Prior Calibration 另有冻结 WM 快照、完整 Prior/optimizer 备份、结果验证和哈希确认回滚；本轮没有进入 Calibration。

训练 loop 会在新 run 开始时保存 `ckpt_initial_actor.pt`，并照常保存 `ckpt_latest.pt`。新增 `scripts/select_m1_action22_best_real_checkpoint.py` 作为分阶段训练后的真实评估选优/回滚工具：要求两份 V5 checkpoint 和同 seed、同初始状态 hash、同固定目标组的 Actor 评估；按跌倒/底盘安全先筛，再比较 L 速度指标或 M 末端距离/保持指标。接受或回滚时复制完整 checkpoint，连同 Actor 与 optimizer 状态一起恢复；输入文件只读。selector 的合成契约/回滚单测已通过，但本轮没有真实 V5 checkpoint，因此没有生成真实 `ckpt_best_real.pt`。

### Prior 首帧平滑

Future31 参考平滑现在包括真实 root 线/角速度与首帧参考速度、真实关节 `qdot` 与首段参考关节速度的衔接。已有 wheel 初始速度连续性保留。该改动是未来 Prior 校准的边界修补，不能解释为本轮 Actor 退化的直接原因。

## V4 Actor Residual Alpha 真实仿真消融

评估程序：`scripts/run_m1_action22_residual_alpha_sweep.py`；逐项子进程执行，避免单进程堆积模拟器状态。实验固定 seed `20261104`、32 MuJoCo env、300 steps（6 秒）、无 DR/观测噪声/动作噪声；覆盖 15/30/45/60 分钟和最终 checkpoint，L/M 各 `alpha={0,0.25,0.5,0.75,1}`，另有最终 Prior Direct。共 52 个评估文件、260 个场景指标行。

每个 alpha 单独重置同一 seed 和任务条件；Alpha=0 与 Prior Direct 的额外检查记录到初始状态、Xi、首个 reference action 和首个执行 action 的 SHA256，Actor Alpha=0 与 Prior Direct 哈希相同。该等价性复核的后续聚合指标最大差异约 L `0.0032`、M `0.0201`；它们来自分开的模拟运行，不能声称之后的状态相同。主 alpha 扫描的逐案文件生成于加入 hash 输出前，因此没有逐案初始状态 hash；这一限制已保留在结果说明中。

最终快照的 ID 结果（位移为 6 秒窗口的平均有符号位移；fall rate 是已结束 episode 中跌倒比例，episode 数同时列出）：

| 任务/命令 | Alpha | 有符号位移 | vx RMSE | EE 进展 / 末端距离 | 已结束 episode / fall rate | 0.5 秒保持 |
|---|---:|---:|---:|---:|---:|---:|
| L 前进 `+0.12 m/s` | 0 | +0.249 m | 0.079 m/s | — | 0 / —（6 秒内无 episode 结束） | — |
| L 前进 `+0.12 m/s` | 0.50 | +0.123 m | 0.137 m/s | — | 0 / — | — |
| L 前进 `+0.12 m/s` | 0.75 | +0.021 m | 0.162 m/s | — | 2 / 1.00 | — |
| L 前进 `+0.12 m/s` | 1.00 | +0.028 m | 0.183 m/s | — | 9 / 1.00 | — |
| L 后退 `-0.06 m/s` | 0 | -0.024 m | 0.098 m/s | — | 0 / — | — |
| L 后退 `-0.06 m/s` | 0.50 | +0.086 m | 0.134 m/s | — | 0 / — | — |
| L 后退 `-0.06 m/s` | 1.00 | -0.042 m | 0.128 m/s | — | 2 / 1.00 | — |
| M 前方 | 0 | -0.113 m | — | -0.113 m / 0.243 m | 0 / — | 0 |
| M 前方 | 0.75 | +0.070 m | — | +0.070 m / 0.227 m | 0 / — | 0 |
| M 前方 | 1.00 | +0.051 m | — | +0.059 m / 0.201 m | 7 / 1.00 | 0 |
| M 前方+侧向 | 0 | -0.109 m | — | -0.109 m / 0.341 m | 0 / — | 0 |
| M 前方+侧向 | 0.50 | -0.108 m | — | +0.009 m / 0.325 m | 0 / — | 0 |
| M 前方+侧向 | 0.75 | +0.075 m | — | +0.102 m / 0.215 m | 0 / — | 0 |
| M 前方+侧向 | 1.00 | +0.072 m | — | +0.095 m / 0.143 m | 3 / 1.00 | 0 |
| M 向上 | 0 | -0.123 m | — | -0.123 m / 0.298 m | 0 / — | 0 |
| M 向上 | 0.75 | -0.116 m | — | +0.001 m / 0.227 m | 0 / — | 0 |
| M 向上 | 1.00 | -0.026 m | — | -0.101 m / 0.539 m | 22 / 1.00 | 0 |

解释：最终 L 前进上，Alpha=0 的 Prior Direct 基线比 Alpha=1 Actor 更稳定，且位移/速度误差均更好；Alpha 增大伴随腿/臂/轮 residual RMS 增大，方向表现并不稳定。后退基线本身较差，Actor 不能稳定修正。M 的 Alpha=0 Prior Direct 稳定但远离目标；部分 checkpoint/目标在中等 Alpha 下获得进展，另一些 alpha/快照反而退化。Alpha=1 在最终 M 前方和组合目标下末端更近，但所有已经终止的 episode 都是跌倒，且无任何 0.5 秒保持成功。因而证据支持“L 上 Actor 修正损害控制、M 上修正有任务相关的局部收益但稳定性不足”，不支持一个普适的最佳 Alpha。

同一套 32-env / 300-step 扫描下，15 分钟快照与最终快照对照如下；它与旧报告中的 64-env / 500-step 评估口径不同，不混作同一成功率序列：

| 场景 | 快照 | Alpha 0 位移 / vx RMSE | Alpha 0.5 位移 / vx RMSE | Alpha 1 位移 / vx RMSE | Alpha 1 已结束 episode / fall rate |
|---|---|---|---|---:|
| L 前进 ID | 15m | +0.248 m / 0.087 m/s | +0.209 m / 0.130 m/s | +0.285 m / 0.153 m/s | 1 / 1.00 |
| L 前进 ID | Final | +0.249 m / 0.079 m/s | +0.123 m / 0.137 m/s | +0.028 m / 0.183 m/s | 9 / 1.00 |
| L 后退 ID | 15m | -0.020 m / 0.104 m/s | +0.048 m / 0.141 m/s | -0.037 m / 0.172 m/s | 2 / 1.00 |
| L 后退 ID | Final | -0.024 m / 0.098 m/s | +0.086 m / 0.134 m/s | -0.042 m / 0.128 m/s | 2 / 1.00 |
| M 前方+侧向 | 15m | EE progress -0.097 m / end 0.357 m | -0.298 m / 0.821 m | -0.407 m / 0.912 m | 0 / — |
| M 前方+侧向 | Final | -0.109 m / 0.341 m | +0.009 m / 0.325 m | +0.095 m / 0.143 m | 3 / 1.00 |

15m 时 M 正 residual 对该组合目标明显有害，最终时 Alpha 0.5–1 有局部末端进展却伴随跌倒 episode；同一个训练快照不是所有 alpha 的表现单调缩放。因此训练恶化既包括后期 L residual 变得更破坏，也有 M 目标相关的修正收益/稳定性代价。

详表、其余 alpha、stop/yaw/OOD 场景、关节/轮跟踪、动作饱和与 residual 组 RMS 均见 `results/m1_action22_p0p1_20261104/actor_residual_alpha_metrics.csv` 和 `alpha_sweep/*.json`。episode_count=0 时 fall rate 未定义；表格不把它写成零跌倒概率。Alpha=0 与 Prior Direct 等价性 hash 检查详见 `alpha0_equivalence_probe/*.json`。

## World Model ↔ MuJoCo 动作响应

配对审计脚本：`scripts/audit_m1_wm_finite_difference.py`。设计覆盖 L 正/反/停/减速，M 前方/侧向/向上；共同及独立四轮、前髋、沿 TCP 目标梯度的 Piper 正/反扰动；在匹配初态、匹配物理上下文下比较真实 MuJoCo 与 WM 的 body 速度、root 位移、轮速、腿/臂关节、TCP、倾角/高度和 task-loss 差分，时域为 1/2/4/8/10 步。低信号样本单独分桶，终止后的 autoreset 状态丢弃。

旧 V4 最终 checkpoint 的 `epsilon=0.02`、16 状态输出响应结果见 `wm_response_v4_16states_eps002.json` 和压缩后的 `wm_response_aggregate.csv`。这里动作状态响应统计有效；第一版文件中的 task-loss 差分曾把正/负分支先求均值，造成差值为零，已定位并修复，**该文件中的 task-gradient 子项不作证据**。修复后的 `epsilon=0.02` task-loss 方向复核使用 4 个状态/场景，见 `wm_task_response_v4_4states_eps002.json` 与 `wm_task_direction_aggregate.csv`。

动作状态响应有明确的 horizon 依赖：L 四个命令场景合并后，wheel common 扰动的实际轮速方向与 WM 在 1/2/4/8/10 步的平均 cosine 约为 `+0.96/+0.98/+0.97/+0.98/+0.99`；但 root 位移在 8/10 步 cosine 约 `-0.77/-0.77`，10 步 body velocity cosine `-0.83`，root 位移 8/10 步 median relative response error 约 `1.63/1.58`。也就是说，WM 短时能预测轮子变化，却把累积的 L 机体运动响应预测反向。M 的目标方向 Piper 扰动在 2/4/8/10 步 TCP cosine 约 `+0.97/+0.99/+0.99/+0.99`，相对响应误差中位数约 `0.24–0.37`，局部臂响应比 L 底盘响应可信。

修复后的任务损失方向差分也发现 L 方向不一致：4 个 L 前进状态在 2/4/8/10 步均 `0/4` 同向；L 后退从 4 步开始也是 `0/4`。M 侧向到达是 `8/8` 同向（两种臂方向 × 4 个状态），M 前方在 4–10 步为 `6/8`，M 向上在 4 步为 `0/8`。这些小样本只用于定位：它们不能证明总体失败概率或 Gate 通过；但与长 horizon L 位移反向响应一致，足以说明旧 WM 不应驱动 L 的风险性 Task BPTT。16-state 审计所有 7 个场景均有 16 个有效起始状态，但按 horizon/direction/output 仍有 320 个动作桶样本不足，低信号也很多；整个审计未通过 Gate。V4 还会因 checkpoint schema/状态数门槛被硬拒绝。

未执行：0.01/0.04 epsilon 敏感度、128 状态 Gate 覆盖、特意构造的临近跌倒/接触转换 held-out 状态。这些结果不能从 Replay NSE 或当前 16 状态数据推断。

## 测试、资源与短训状态

| 检查 | 状态 |
|---|---|
| 原始离线反例重放 | 已运行；见 `offline_probes_after.json` |
| 新 M/终止排序/梯度边界/残差/FD helper/真实 checkpoint selector 单元测试 | 最终 focused suite 46 passed |
| 固定 seed MuJoCo residual-alpha sweep | 已完成，52 案例 / 260 场景行 |
| WM/真实 MuJoCo paired response | 已完成；16-state 输出响应 + 修复后 4-state task-loss 方向复核，Gate 未通过 |
| Actor `tracking_diagnostic` / Task BPTT 短训 | `not_run`；真实物理 Gate 未通过 |
| Prior 更新 / Prior Calibration | `not_run`；Prior 始终冻结，更新 0 次 |
| WM Replay 扩容/梯度累积数据利用率实验 | `not_run`；本轮没有训练，不用未验证的更大 Replay/Batch 改变诊断条件 |
| epsilon 敏感度（0.01 / 0.04） | `not_run` |
| 2048 / L-M 混合训练 | 本轮明确不执行 |

全套仓库测试在 GPU 可用的真实环境设置下通过：`162 passed, 68 warnings, 183.69s`，包含 G1/Go2 回归覆盖。此后又增加一个首帧速度连续性单元测试；最终 M1 focused suite 为 `46 passed`，覆盖该新增测试。使用 `CUDA_VISIBLE_DEVICES=''` 的第一次全套运行有 3 个测试因测试本身显式创建 `cuda:0` 环境而报“无 CUDA”；可见 GPU 后重跑全部通过。`compileall`、gated shell script `bash -n`、`git diff --check` 均通过；用当前 `sm-train --help` 校验 gated 短训脚本的 42 个 CLI option，未发现无效参数。将旧 V4/16-state audit 交给训练脚本时，它实际拒绝启动 Actor BPTT，证明 fail-closed 生效。

电脑重启后检查到唯一计算任务是本轮有限差分审计；显存约 0.8 GB、CPU 约一个核，内存可用约 5.6 GB，当时无 G1/Go2 训练任务被触碰。alpha 扫描已逐项子进程化、可以根据已落盘 JSON 恢复；本轮没有后台训练进程。

## 因果诊断：已确认、部分支持、未验证

| 假设 | 当前证据 | 结论 |
|---|---|---|
| Actor 没有收到梯度 | 旧训练有非零 Actor 梯度范数；旧 checkpoint Actor 输出与 Prior 不同；Alpha 扫描的 Actor residual 改变真实运动 | 不符合“完全没有梯度”的解释。梯度是否指向真实改善另论 |
| 大 residual 破坏 L 的可用运动 | 最终正向任务 Alpha=0 位移 `+0.249 m`、vx RMSE `0.079`；Alpha=1 位移 `+0.028 m`、RMSE `0.183`，终止 episode 中跌倒率 1.0 | 由同 seed 真实闭环消融支持；不同 Alpha 的全轨迹不相同，非单步 Jacobian 因果识别 |
| Actor 修正幅度越小越好 | 15/30/45/60/final、多种 L/M 场景呈非单调变化；M 中间 Alpha 有时改善，其他快照/目标又恶化 | 不成立为普遍规律；目标与训练阶段交互显著 |
| Prior 是唯一问题 | L 正向 Alpha=0 有前进但速度仍低于命令；后退 Alpha=0 运动弱；M Alpha=0 远离目标 | Prior 自身不完整，但无法解释 L 前进中 Actor residual 造成的额外退化 |
| WM 的 task/action gradient 错误 | 16-state V4 物理 FD 中，L wheel response 的 1–4 步轮速同向、8–10 步 root motion 反向；4-state task FD 中 L forward/2–10 步 `0/4` 同向，backward/4–10 步 `0/4`。M arm→TCP 多数方向同向 | 有直接物理证据表明旧 WM 的长时域 L 机体响应/任务差分不可靠，足以阻止 Actor Task BPTT；状态数和分层覆盖不足，不能单独证明它解释了全部训练退化 |
| M 任务已学会 reach | 中等/全量 residual 部分场景降低距离，但 stable_hold 成功率全为 0，且 Alpha=1 出现跌倒 | 未学会稳定 reach |

## 下一步（最多三项）

1. 从新鲜的 V5/WM checkpoint 开始，采集覆盖接触转换、起步/减速与安全边界的真实配对状态，完成至少 128 个 held-out 状态、多个 epsilon 的 MuJoCo/WM 动作响应审计；只有同 SHA、同物理契约证据通过后才放行 WM Task BPTT。
2. 在 Gate 未通过期间只保留 Prior Direct/reference-copy，对 L 做固定 seed tracking-only / bounded-residual BC sanity；真实评估以 `ckpt_best_real` 接受，若比初始 reference-copy 恶化就连网络与 optimizer 一起回滚。
3. M 先独立固定 Reset 与可达目标分布，观察 TCP path/terminal/near-goal-speed、底盘稳定与 0.5 秒 hold；不要从短时 EE progress 单指标扩大 mixed 训练。
