# M1 Motion Prior / Actor / World Model 静态审计

审计日期：2026-11-02。审计基线：SuperMimic `cf9abeb`，AMP-LOCO-M1 `fa2be1e`。

本次目标是解释已有失败实验并核对实现，不修改训练行为、不启动训练或仿真步进。实际源仓库为 `~/AMP-LOCO-M1`，不是用户口述的 `~/AMO-LOCO-M1`。全部数值检查使用 Conda `supermimic`、CPU；没有使用 `.venv`。

## 1. 判断与证据边界

**目前不应把效果差主要归因于训练时间短，更不应先继续调奖励或延长训练。** 已发现可复现的输入归一化失效、训练数据与动作目标范围不兼容，以及预训练目标对“当前状态不满足指令”的覆盖缺失。它们会先破坏 reference，再让 Actor 通过尚未验证方向正确性的 WM 梯度追踪它。M 的已有实验甚至没有更新过 Prior。

用户希望的分工是合理的：Prior 根据当前状态和任务生成短时参考；Actor 在正则约束下跟踪；WM 用真实交互拟合动力学，提供 Actor 和 Prior 的可微更新路径；后训练让 Prior 的轨迹更可执行、更接近目标。当前代码已有这个计算图，但还没有充分保障“Prior 输出本身是一条好的物理轨迹”。

当前 Prior 更接近一个通过离线/在线学习得到的短时规划网络：推理时一次 forward 给出 reference，没有在线候选轨迹筛选或 MPC 求解迭代。它每 10 个控制步重规划一次，恰好执行完全部 0.2 s 视野；不是每步滚动重新求解。Actor 仍逐步闭环控制，但“预测未来”本身不等于“选出了当前最优未来”。

| 优先级 | 发现 | 证据与影响 |
|---|---|---|
| P0 | Piper 第 6 关节输入标准差为 `1e-6`，实际可控 | CPU 直接复现：前进指令下，`q6=0.0001 rad` 把终点参考从前进 6.29 mm 变成后退 53.74 mm |
| P0 | 预训练参考姿态与控制目标范围明显不兼容 | L 抽样中手臂第 2/3/4 关节目标全部超出 actor setpoint 范围；M 中相应比例约 85%/86%/100% |
| P1 | L 数据中的目标速度完全等于当前速度 | 全部 211,148 个 L 窗口均如此；不能据重建精度推断从静止启动、纠偏、换向能力 |
| P1 | 数据主要不提供持续主动滚动的轮速参考 | L 有效轮速标签中位数约 `7.9e-6 rad/s`；M 轮速监督有效比例为 0 |
| P1 | 优化的是 WM 执行轨迹任务误差，缺少 Prior 轨迹与执行轨迹的一致性约束 | Prior 可以变成驱动 actor 的信号，而不再代表可跟踪的目标轨迹 |
| P1 | WM 数值预测误差小，不代表动作梯度方向可靠 | 缺少独立数据、动作扰动方向、真实动力学 Jacobian 的验证；Actor/Prior rollout 比 WM 训练窗口长 |
| P1 | 过程／终点目标未形成完整 MPC 目标 | M 的进度项与终点项重复；L 无独立终点项；只有 0.2 s 视野，无终端价值和保持损失 |
| P1 | 实验不符合所述混合任务设定 | 最近是 L-only 与 M-only 各 2048 环境；M 实验 Prior 更新数为 0 |
| P2 | 指标口径和采样存在误导风险 | XY RMSE 差一个 `sqrt(2)`；goal sensitivity 实际切换任务；controlled 模式的 prior_unfrozen 标志错误 |

“确认”指代码或离线数值可直接验证。“可能解释实际倒退”不等于已经完成因果定位：原运行没有保存 Replay/逐步完整状态，无法还原发生倒退时的 q6、qd6、参考、动作和接触。本文不把合成状态反例冒充真实实验回放。

## 2. 已有实验实际上做了什么

核对 Git 历史、运行 config、checkpoint、TensorBoard 和已有分析：

- `0f68268` 引入联合训练；`0bda727` 调整任务损失与门控；`cf9abeb` 校准控制接口并记录两次 2048 环境实验。
- L run：`logs/m1_prior_actor/m1_prior_L_2048_30min_ctrl_v2_20261102`，2150 iteration，约 30.6 min；最终 checkpoint 的 WM/Actor/Prior 更新数为 **844/644/260**，Prior 到约 1550 iteration 才开始更新。
- L 最后日志平均指令 `vx=+0.329 m/s`，实际 body `vx=-0.044 m/s`，XY RMSE `0.407 m/s`；存活率 1，说明存活并不代表任务成功。
- M run：`m1_prior_M_2048_30min_ctrl_v2_20261102`，1450 iteration 用户停止，约 18.6 min；运行末段计数 **564/364/0**，未达到 384 Actor 更新的解冻条件。保存的 latest 实际只有 iteration 1000、计数 **384/184/0**。
- M 最后 EE 距离约 `0.783 m`，进度 `-0.504 m`。它证明当前 frozen-prior tracking 没有完成 Reach，不能用来判断“Prior 后训练没有用”。

TensorBoard 对部分 schedule 计数做区间平均，因此最后标量会出现 `834.5/634.5/250.5`，不能当作最终整数更新计数；应以 checkpoint 和阶段记录为准。

两次配置明确是 `task_type=L` 和 `task_type=M`，不是一个 1024 L + 1024 M 的联合实验。当前 `balanced` 在每次 reset 时以 Bernoulli 概率采样，`.5` 只保证采样概率，不保证实时有且只有 1024 个环境属于各任务。不同任务的 episode 长度还可能改变占用比例。Replay 随机抽取有效状态，也没有保证每个学习 batch 中一半 L、一半 M。

相关代码：[任务采样](../src/supermimic/envs/mdp/commands.py)、[Replay](../src/supermimic/data/replay_buffer.py)、[门控](../src/supermimic/motion_prior/trainer.py)。已有实验报告：[L](M1_L_2048_30MIN_RESULTS.md)、[M](M1_M_2048_30MIN_RESULTS.md)。

## 3. 最关键的反例：极小标准差让命令被关节扰动淹没

当前 prior 源 checkpoint：

`/home/yifeiy/AMP-LOCO-M1/logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt`

SHA256：`1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`。

State53 每帧中的 q6、qd6 分别位于 index 30、48。对所有 492,125 个缓存窗口检查，历史 q6/qd6 都为零或约 `1e-16` 级浮点余量。源 `motion_prior/cvae_data.py` 的标准差下限是 `1e-6`；当前 `cvae.py` 直接用 checkpoint 的标准差归一化，没有处理这种“数据中固定、仿真中可动”的维度。

CPU 反例设置：四帧相同历史，根高度 0.334 m、四元数单位姿态、关节为当前默认角、速度零；分别给 `vx=+0.3` 和 `vx=-0.3 m/s`；只改变 q6。数值是 **Prior 的 0.2 s 终点 root-x 位移参考**，不是仿真位移。

| q6 / rad | 最大标准化输入 | +0.3 指令的终点 dx / m | −0.3 指令的终点 dx / m |
|---:|---:|---:|---:|
| 0 | 4.57 | +0.006289 | −0.006384 |
| +0.00001 | 10 | −0.002160 | −0.015893 |
| +0.0001 | 100 | **−0.053740** | −0.058168 |
| +0.001 | 1,000 | −0.057277 | −0.057702 |
| +0.01 | 10,000 | −0.057396 | −0.057438 |
| −0.01 | 10,000 | +0.013171 | +0.013125 |

这不是简单的全局正负号反了：正 q6 使模型偏向后退，负 q6 使它偏向前移，而两种情况下前进／后退命令的差别都趋近消失。`qd6` 的独立扰动也有同类现象。q6=+0.01 时，输出轮速四轮均为负，约 −0.26 至 −0.33 rad/s。

**L 最终 checkpoint 没有修好这个问题**：q6=+0.0001 时 +0.3 指令仍给 −0.05270 m 的终点参考；q6=+0.01 时仍为 −0.05599 m。

运行中进入这种输入分布是现实风险：WM warmup 对全部 22 个动作加 std=0.1 的探索，q6 位置目标的标准差因此约 0.025 rad；且 q6 没有被锁定。离线调用 L 最终 Actor，在上述默认合成历史下输出 q6 目标约 −0.126 至 −0.130 rad（不同 phase）。这些是 actor setpoint，不是测得的关节角，不能据此声称真实日志中的 q6 为正。

处理方向：明确六轴机械臂的实际活动维度；使用有物理意义的归一化尺度、数据扰动和部署分布覆盖；若暂时固定某轴，数据、状态输入、控制和奖励必须同步约定。**不能只改旧 checkpoint 的 std 就认为已修复**，这会改变网络所见分布，需要重新校准/训练并验证同状态不同命令的输出与扰动鲁棒性。在线把 q6 偷换成零也会隐藏真实物理状态，最多是明确受限的消融诊断。

代码：[在线归一化](../src/supermimic/motion_prior/cvae.py)、[输入适配](../src/supermimic/motion_prior/features.py)、[探索](../src/supermimic/training/loop.py)。

## 4. 运动学定义：主要接口一致，但数值和姿态分布有问题

按原始 motion 文件 SHA256 从新的 combined bank 找回旧训练数据，L/M/LM 各取 3 条、每条 14 帧，核对原始 Frames→State57→State53、Future31 解码，以及源 URDF FK 与当前可微 FK。

- 源 Frames 四元数 xyzw，运行 State57 wxyz：适配正确。
- 源线速度 body frame，运行 root 线速度 world frame：适配有正确旋转；角速度按 body frame 对齐。
- State57 的 root-v / root-w / qd 起始 index 为 29/32/35；当前实现没有误用其他机器人偏移。
- 18 个非轮关节、4 个轮速按名字映射；rotation6d 使用前两列逐列拼接，与预训练定义一致。
- `dt=0.02 s`、4 帧历史、10 帧未来一致；抽查 root position、非轮关节位置解码误差为 0。
- 抽查 TCP 和四个 wheel link 的源／运行 FK 最大坐标误差约 **3.79e-7 m**。TCP 为 `piper_x_link7`，当前 gripper 两个末端原点重合，未发现足以解释失败的 TCP 偏置。
- 当前默认姿态下 `axis × world_up` 给出的四轮正滚动方向均为 `[+1,0,0]`；轮子动作使用正的比例 `5 rad/s`，没有发现统一反号。轮半径为 `0.0975 m`。

这里还发现一个数值敏感性：latest-frame heading-relative rotation 的理论零维度 index 161 同样 std=`1e-6`。源 NumPy 与 Torch 特征约 `1e-7` 的舍入差，归一化后最大达到 0.107；抽查 prior 输出最大差到 0.00276（混合输出单位，不能直接称作毫米）。**给两个实现完全相同的特征时，网络输出误差为 0**。所以不是权重转换或架构不一致，而是归一化也放大了正常数值误差。

这些检查是抽样运动学一致性验证，不证明接触力、摩擦、惯量和执行器动力学等价。checkpoint loader 校验 schema、归一化和数据 hash，但没有强制比较源预训练机器人 URDF fingerprint 与运行模型 fingerprint，后续应补充这个契约。

## 5. 参考数据与当前控制能力不兼容

动作是固定默认姿态附近的有限范围位置目标，而不是任意关节位置或相对上一目标的增量。当前机械臂默认：

`[0, 2.55, -1.50, -0.80, 0, 0] rad`

L 数据抽样 2048 个窗口、每窗 10 帧，手臂参考恒为：

`[0, 0.50, -0.40, +0.30, 0, 0] rad`

| 关节 | Actor 可输出位置目标 / rad | L 数据参考 / rad |
|---|---:|---:|
| Piper 2 | [2.20, 2.90] | 0.50 |
| Piper 3 | [−1.75, −1.25] | −0.40 |
| Piper 4 | [−1.15, −0.45] | +0.30 |

这些不是硬关节限位，真实角度可能因接触、负载或惯性越界；但 Actor **不能直接给出这些参考的持续位置 setpoint**。此前根据“reset 状态下 prior 的输出”扩充个别 action scale，只能说明那个输出局部可跟踪，不能证明数据本身可跟踪；reset 又恰好处于预训练姿态分布之外。

同样的 2048 窗口抽样中，M 参考 Piper 2/3/4 的 setpoint 越界比例约 **85.1%/85.6%/99.98%**；LM 为 **93.1%/91.3%/100%**。部分腿关节也有明显越界。详细分关节比例见 JSON。

Reach 目标通过初始 TCP 周围偏移采样，只裁剪世界高度，没有针对当前动作范围做可达性/碰撞筛选。本次没有计算全部目标的逆运动学，因此不声称所有 Reach 目标不可达；但数据参考与控制范围的冲突已经确定。

应先统一机器人工作姿态、可输出目标范围和数据分布。扩大 scale 会改变控制增益意义、探索和可行域，不能盲目放大；另一选择是基于 reference 的有界残差控制，但那需要明确 reference 可行性与防失控规则。本次没有更改控制接口。

代码：[默认姿态和 scales](../src/supermimic/robots/spec.py)、[动作映射](../src/supermimic/tasks/action_contract.py)。

## 6. 数据教会了什么，没教会什么

### 6.1 速度命令与当前状态完全共线

对 **全部 211,148 个 L 窗口**，Goal8 的 `[vx,vy,wz]` 与历史最新帧的 body `[vx,vy,wz]` 逐项最大误差均为 0。源 labeler 可以使用 effective command，也可以退回 reference velocity；这个统计本身不能证明标签实现错误或未来泄漏，却足以说明缺少“当前速度与目标不同”的监督约束。

模型可以依赖当前速度和历史继续外推，把 command 当冗余输入，依然得到不错的重建 loss。部署时从静止要求 +0.3 m/s，或已经倒退时要求前进，就进入了预训练未覆盖的状态—命令组合。

全 L 缓存命令范围：vx ±0.30、vy ±0.15 m/s、wz ±0.40 rad/s。运行采样 vx=[0.15,0.50]，若均匀采样，约 **57% 的 vx 命令超过训练最大值**；vy=[−0.20,0.20] 也超出数据覆盖。

应生成有明确意图的启动、停止、换向、偏差恢复样本，并验证同一历史下正向／停止／反向指令的区别。不能只把原轨迹随机改标签而保持不相容的未来 target。

### 6.2 轮速很小，不自动等于轮速单位错误

每任务 2048 窗口的抽样统计：

| 任务 | 轮速监督有效比例 | 有效标签绝对轮速中位数 / rad/s | 有效标签绝对轮速 P99 / rad/s |
|---|---:|---:|---:|
| L | 54.4% | 0.0000079 | 0.0454 |
| M | 0% | 无有效监督 | 无有效监督 |
| LM | 53.1% | 0.0000070 | 0.0382 |

作为量级比较，腿姿态基本不变、无滑动纯滚动达到 0.3 m/s，需要约 `0.3/0.0975 = 3.08 rad/s`。

源 `wheel_kinematics_from_contacts` 用轮心世界速度沿 URDF 滚动方向投影，再除半径，并只在稳定接触区赋有效值。这个公式本身合理。腿在移动而支撑轮心近似世界静止时，基座前进可以同时有接近零的支撑轮速；因此不能把上述统计直接定性为标签单位/符号 bug。

但它说明这个 bank 不能直接等同于“轮子持续滚动推进”的 reference bank。M 的 zero wheel 值还被全部 mask，模型未被直接监督在 M 下输出零轮速；在线 Actor 却对所有输出轮速做 tracking，未使用监督置信度。需要明确滚动／迈步／混合的目标行为，补充对应数据和轮速有效性语义。

### 6.3 M / LM 任务语义混合

预训练采样约 L 50%、M 25%、LM 25%，但 Goal8 把 M/LM 都编码为 `[0,0,0,ee_goal_xyz,0,1]`。也就是同一 Reach 标志下，有的参考基座静止，有的参考基座移动；在线 M 又额外要求小基座漂移。历史可能部分区分二者，但输入没有显式“允许底盘移动/固定底盘”的意图位。

旧 2272 条 bank 没有源 motion_phase，所有有效窗口都参与训练，不能从成功率推断每个窗口都在主动接近目标。高位目标有大量 near-hold 窗口。预训练文档本身也只证明有限的目标使用，未证明闭环完成任务。

### 6.4 新 bank 没有自动进入旧 checkpoint

目前源仓库 combined bank 是 3272 条，但运行仍加载旧 **2272 条 V2.5 bank** 训练出的 CVAE。旧路径已删除、数据保留在合并 bank 中，不影响现有权重加载，但配置复现和样本追溯应更新。新旧 motion ID 不能简单一一对应；本次使用原始文件 hash 对齐。

## 7. Actor 为什么可能“更会跟踪，却更不会完成任务”

Actor 当前只优化 reference tracking 加稳定性和控制正则，既不直接优化速度指令，也不直接优化世界 Reach 目标。这种分工在 reference 可靠时成立；当 reference 已经不响应目标时，跟得更好也可能让任务更差。

当前 tracking：root 位置、root 旋转、腿关节位置、臂关节位置、轮速。没有 root 线速度 tracking、非轮关节速度 tracking、TCP reference tracking；`actor_tcp_weight` 只是保留字段，没有实现。

量级例子：机器人不动、理想 reference 以 0.3 m/s 走 10 帧，其他误差为零，root-position 项对总 loss 的贡献约 **0.000924**。而默认手臂与上述 L 数据姿态的差异，在当前 arm tracking 权重下约 **0.276**，大约 300 倍。这只是可复算的损失尺度示例，并不声称所有在线 reference 都等于原始 L 姿态；也不能用 loss 数值之比代替参数梯度之比。但它清楚展示了短视野位置跟踪的弱移动信号。

轮速归一化除以 10，再乘 0.5；假设 reference 要求 3.08 rad/s，实际为零，四轮一致时这项约 0.047。若 reference 本来接近零，则它会鼓励低轮速，不能帮助实现滚动前进。

M 固定种子已有评估也符合这种脱节：iteration 500→1000 时 arm RMSE 约 1.207→0.314 rad，EE 距离却从 0.437→1.032 m。它不是“跟踪改善证明任务改善”的证据。

代码：[tracking 定义](../src/supermimic/motion_prior/reference.py)、[actor loss](../src/supermimic/motion_prior/losses.py)。

## 8. 梯度连通正确，但还没保障轨迹语义和梯度可信

当前训练路径是：

`Prior(history, goal) → reference → Actor → action → WM → imagined states → task loss`

Prior 更新时冻结 Actor/WM 的参数，但保留对输入的梯度；Actor 更新时 detach reference、冻结 WM；WM 只用真实 Replay 拟合。静态代码没有发现这个主路径被误 detach。已有有限差分测试使用小 WM，能验证 autograd 所算的是这张图的导数，不能证明这个导数符合真实物理。

主要缺口：

1. **Planner task loss 在 WM-executed states 上，reference 本身只受 smoothness 和 anchor。** 没有显式约束 reference 与 WM 执行轨迹接近，也没有对 decoded reference 加终点任务约束。于是 Prior 可以学到让 Actor 采取有利动作的 reference 编码，而不一定是一条可解释、可执行的最优运动轨迹。要保持用户设想的分工，应同时约束 reference 任务质量、可跟踪性和执行结果；一致性项的权重/stop-gradient 需避免退化为“Prior 迎合差 Actor 的静止轨迹”。
2. **WM 是未知动力学的学习模型，不是动力学可行性证明。** 当前低高度、倾斜、关节限位和速度 hinge 作用于 imagined states；没有 self-collision、接触支撑、无滑动、摩擦锥、真实力矩/功率等约束。FCL 通过只是运动学无碰撞，不能替代这些物理条件。
3. **8-step WM 训练/诊断，10-step Actor/Prior 展开。** 任务相关误差与动作 Jacobian 应在实际使用的 10-step horizon 上验证。
4. **NSE 只是在训练 Replay 上的平均归一化误差。** 它不会直接约束 `d next_state / d action` 的方向。严格门控的 NSE 阈值为 None，最近运行走的是 controlled research gate，不是质量通过。
5. **真实数据多不等于模型充分拟合。** L 收集约 440 万 transition，但只做 844 次 WM optimizer step。当前每次逻辑 batch32、窗口8，累计约 216,064 个抽样 transition 使用次数（可重复），还要覆盖变化中的 Actor 和 DR。不能仅以 2048 环境或墙钟时间推断样本利用足够。
6. **termination 预测仅作为诊断。** 联合 rollout loss 没有 survival weighting/显式终止代价或终端价值。速度/高度软约束不足以代表完整失败后果。

已有 TensorBoard 最后记录的 8-step、Replay 重采样物理误差：

| 指标 | L | M |
|---|---:|---:|
| root 位置 RMSE / m | 0.00569 | 0.01257 |
| root 线速度 RMSE / m/s | 0.0663 | 0.1581 |
| root 角速度 RMSE / rad/s | 0.2157 | 0.3039 |
| arm 关节位置 RMSE / rad | 0.0250 | 0.0408 |
| wheel 角速度 RMSE / rad/s | 0.4832 | 0.6991 |

这些不证明 WM 完全失效，却足以反对“总 NSE 小，所以任务梯度一定可靠”。需要单独保留按 episode/策略阶段划分的真实验证数据，并在相同初始状态下比较正负动作扰动引起的真实和预测速度/TCP 变化，检查梯度符号、方向一致率、局部改进是否转移。这个物理验证需要之后的受控实验；本次静态审计没有运行它。

## 9. 过程、终点、稀疏与稠密奖励的准确情况

joint task 的环境 reward 是 `data_collection_zero`；实际学习目标在 imagined losses 中，不是通过环境稀疏 reward 训练 critic。这个实现可以工作，但要用它实际拥有的梯度机制讨论奖励。

### 9.1 L：只有短时平均速度误差

小误差区间内，默认 L task loss 近似：

`mean_t[0.5*(ex²+ey²) + 0.25*ewz²]`。

它是有效的连续过程目标，不会把前进误写成后退；没有独立终点速度、位移、稳定巡航价值。L 是速度任务，不必生硬套用位置终点，但可约束 horizon 末速度跟随、加速度收敛，并为后续巡航提供更长视野或终端价值。

### 9.2 M：有终点项，但 progress 与终点重复

记 `d0` 为当前 TCP 距离，`dH` 为最后一帧距离，`mean(d)` 为 10 帧平均距离，单位米：

`L_M = dH/0.10 + 0.1*mean(d)/0.10 - (d0-dH)/0.10`

整理后：

`L_M = 20*dH + mean(d) - 10*d0`。

对于一次采样 rollout，d0 与 Prior/Actor 待更新参数无关。因此 progress 没增加独立的过程引导，而是把终点距离的权重又加了一遍。H=10 时，一个非终点距离的系数是 0.1，终点距离的系数是 20.1，比例为 **201:1**。这是距离变量的系数比较，不等于网络最终梯度比，因为早期动作也影响终点。

在 M 距离 0.3 m、尚无进展时，task 数值约 3.3；L 速度误差 0.3 m/s 时约 0.045。混合训练中两者分别取任务内均值再相加，不会自动按物理量校准，M 有明显主导共享 Prior 参数的风险。应记录分任务梯度范数/夹角，而不是只比较总 loss；microbatch 中任务数量不固定还会增加权重波动。

这意味着当前问题不是“没有终点奖励”，而是终点重复加权，过程偏弱；同时每次只看 0.2 s，通常不足以到达当前 0.2–0.4 m 偏移目标。MPC 可以在短 horizon 使用终点距离，但不能据此保证最终可达、能够刹停或长期保持。没有终端速度、保持、值函数或 terminal set，因此可能偏好短期冲向目标，也可能受 anchor/短视野限制而停滞；实际哪种需要实验区分。

成功半径 0.08 m、保持 0.5 s 是统计指标，**不在当前 planner loss 中**。0.5 s 的保持也长于整个 imagination horizon。基座漂移惩罚相对每次 rollout 起点，而非整段 episode 的起点，允许每个短窗漂移很少但长期累计很大。

### 9.3 稀疏成功与可微规划的结合

直接增加 `1[d < radius]` 几乎处处对状态梯度为零，在纯 pathwise BPTT 中不能替代连续引导。可采用连续的距离/姿态误差、进度势函数、目标区内速度/保持项，再把稀疏成功用于 critic/终端价值学习、候选轨迹排序或独立评估。若使用逐步势函数 shaping，应注意无折扣求和会望远镜式退化为端点差，不应再当独立过程项重复计权。

应明确两个对象：reference trajectory 的过程/终点质量，以及经 Actor+WM 执行后的过程/终点质量。推荐形式是物理尺度归一化的 running cost 加 terminal cost，再加入执行一致性、动态可行性、跨 plan 连续性和信任域；具体权重需用真实状态分布下的梯度与消融标定，不能从单个 reset 状态定出通用系数。

### 9.4 平滑与能量也有缺口

smoothness 是超过阈值才惩罚的 acceleration/jerk hinge，非全程平滑；对 root/非轮关节虽拼入当前位置，却没有把当前真实线速度/关节速度作为首个速度点，因此不惩罚起始速度不连续。没有重规划前后轨迹重叠的一致性项。

L 最后 raw smoothness 249.1 乘 `1e-4` 约 0.0249，与 task 0.118 相比不算离谱，但这不能证明梯度平衡。anchor 还会把模型拉向“同一 OOD 状态下的原始 prior 输出”，不保证是正确参考。

`energy` 当前恒为 0；effort surrogate 默认关闭；`supervised_weight` 没接入 total，返回 supervised 恒为 0。把这些配置数字调大不会自动获得对应训练约束。

## 10. 指标应先修正的地方

- 训练 XY RMSE 使用 `sqrt(mean(ex²+ey²))`；固定评估用 `sqrt(mean(ex²+ey²)/2)`。后者低 `sqrt(2)`，静止面对 0.3 m/s 指令会显示约 0.212，不能把它与训练 0.3 直接比较。
- fixed evaluation 的 base displacement 是各时刻距初始位置的平均距离，不是有符号前进位移，更不是最终位移；wheel abs mean 也不能证明轮子向前转。
- `_goal_sensitivity` 实际把 L/M 标志翻转，不是同一任务内将 forward 改成 backward，无法替代命令跟随测试。
- `schedule/prior_unfrozen` 只判断 `stage == joint_alternating`；controlled 模式已经更新 Prior 时仍可能报 0，应看实际 optimizer 计数。
- 日志详细追踪诊断按每 10 步采集，与每 10 步重新规划对齐，在这些运行中主要测 phase9，不能代表整个参考执行过程。

建议记录每个任务、每个 phase 的有符号实际速度、轮速、prior root delta/轮速、actor setpoint、q6/qd6、标准化输入最大值和越界率；对一小组环境留存逐步快照，才能做倒退的因果追溯。

## 11. 建议修复与验证顺序

具体实现选择、损失设计和验收方法见 [修复方案](M1_REMEDIATION_PLAN_20261102.md)。方案尚未实施。

1. **先修数据—状态—控制契约。** 决定手臂工作姿态和可控范围；解决 tiny-std 维度；把 checkpoint/URDF/数据版本和字段契约统一。先离线验证小扰动不改变任务方向、同状态不同命令确实改变轨迹。
2. **补足意图与动态可执行的数据覆盖。** L 补启动/停止/逆向纠偏、明确滚动模式；M 明确底盘是否允许移动、Reach/Hold 阶段和目标可达性，补齐轮速监督。保留 kinematic bank 的结构与多样性，但不要把无碰撞当成动力学可行。
3. **先验证 Actor 能跟踪可信且可达的参考。** 将任务相关速度/TCP tracking 和物理正则做合理尺度标定；在 reference 损坏前就能检测、拒绝或回退。此阶段不应靠放松 gate 来证明联合训练成立。
4. **验证 WM 对任务方向的局部预测。** 使用独立 episode 和动作扰动验证 1/5/10-step root-v、wheel、TCP 误差及梯度方向，覆盖 Actor 后续分布和 DR；根据样本利用率调整模型容量、更新频率与覆盖，而非仅扩大环境数。
5. **再解冻 Prior。** 明确 reference 与 execution 两套目标及一致性；避免重复终点权重，加入保持/末速度与跨计划连续性，用受约束的更新维持参考语义。分别监控 L/M 梯度，不用一个总 loss 掩盖退化。
6. **最后进行真正的混合任务实验。** 若要求严格 1024/1024，就固定环境任务分组；Replay 学习 batch 也应分层采样，并分别验收 L/M。联合模型不能只看平均成功率。

后续验收应至少包括：从静止前进、后退、停止及反向纠偏；M 从当前姿态到可达目标并保持；更换任务标志确实改变行为；更新 Prior 后 reference 的任务误差、执行一致性和真实任务指标共同改善。不要把“训练不 NaN”“存活率高”“tracking loss 下降”当作替代标准。

## 12. 本次新增的复核材料

- [CPU 审计脚本](../scripts/audit_m1_prior_static.py)：读取 checkpoint、缓存和原始数据，不调用 optimizer 或 `mj_step`。
- [数值证据 JSON](artifacts/m1_static_audit_20261102.json)：原始/最终 prior 的正负 q6/qd6 扫描、全缓存命令范围和恒定维度、抽样 setpoint 越界、轮速监督、9 条原始 motion 的接口/FK 对齐、最后 WM 分量日志。

复现命令（要求原始 artifacts 和现有 `/tmp` cache 仍在；脚本不会重建数据或启动训练）：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/yifeiy/miniconda3/envs/supermimic/bin/python \
  scripts/audit_m1_prior_static.py \
  --output docs/artifacts/m1_static_audit_20261102.json
```

原有 Go2 改动未调整。此次新增仅审计文档、独立复核脚本及结果；没有修复训练代码，也没有启动训练。
