# M1-Piper：任务、奖励、网络、训练结果与退化机制

2026-11-06。本报告以 `v2_lm50_2048_3p5h_seed2026` 为主，两种算法各约3.5小时、reward-only、没有Motion Prior。历史v1与motion-prior实验只提供背景，不能混入v2正式统计。新图重绘已有原始记录，没有重新训练或追加GPU评测。

## 组会展示版

1. 共享Actor学习两类任务：固定1024个L环境跟踪底盘速度，1024个M环境跟踪世界TCP点。不是在同一个环境同时完成两种目标，也没有抓取物体。
2. v2目标为3m圆盘、离地.03–1.10m、8cm稳定首次到达即换点；episode30s。SM恢复H32，并多一个独有终点距离项。
3. 等时长PPO采到35.00M真实transition，SM6.10M，相差5.736倍。PPO纯M成功334/469≈71.2%；SM0/128，主要低位停滞。
4. 局部真实物理检查发现WM动作响应反号；冻结快照检查发现上一动作反馈可放大饱和、Critic梯度可主导。它们支持复合退化机制，尚不能唯一归因。

推荐图：[M01配置](figures/M01_m1_training_config.png)、[M02正式结果](figures/M02_m1_final_results.png)、[M03分桶](figures/M03_m1_reach_success.png)、[M04退化](figures/M04_m1_training_failure.png)、[M05方向审计](figures/M05_m1_wm_gradient_audit.png)、[M06任务奖励](figures/M06_m1_task_rewards.png)、[M07梯度与反馈](figures/M07_m1_gradient_and_feedback.png)、[A02机制假设](figures/A02_m1_failure_mechanism.png)。

## 1. 版本边界与证据

| 项目 | v1历史结果 | v2本轮主结果 |
|---|---|---|
| 时间 | 各约2.5h | 各约3.5h |
| M目标 | 初始TCP附近相对小盒子 | 以换点时底盘为圆心3m面积均匀圆盘 |
| 高度 | 约.66–.96m，取决于初态TCP | 地面相对.03–1.10m |
| Episode | 10s | 30s |
| 成功 | 8cm、保持.5s的历史评测口径 | 8cm内且稳定的首次控制步，立刻换点 |
| SM horizon | H8 | H32 |
| SM辅助终点 | 未启用 | 权重1.0，仅Actor想象M分支 |

旧Action22 motion-prior v4/v5审计还有Prior Direct、残差Actor、Gate等机制；本次reward-only没有加载Prior，也没有走那个残差Actor或WM Gate。不能用“另一个链路有Gate/回滚”推断这次也有。

正式权重：PPO `logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/model_533.pt` 与其 `policy_inference.pt`；SM `logs/m1_reward/v2_lm50_2048_3p5h_seed2026/ckpt_latest.pt`。实际模型SHA256、网络形状与配置见 [manifest](data/experiment_manifest.json)。原始评测保存在本目录data及仓库 [m1_v2结果目录](../results/m1_v2_20261106/)。

本机审计源码与保存config可交叉验证，但没有完整保存每个历史时刻所有未提交Python文件。因此“本次读取源码”与“训练时源码已不可变封存”是两种证据强度，详见 [来源限制](EXPERIMENT_PROVENANCE.md)。

## 2. 任务与坐标协议

### 2.1 Task L：机体速度跟随

输入 $u=[v_x,v_y,\omega_z]$，分别m/s、m/s、rad/s。vx范围[−.09,.145]，采样分支15%将vx置零、35%从负区间采样、50%从正区间采样；vy∈[−.05,.055]，wz∈[−.17,.18]。**15%是vx=0，不代表三个分量同时为零的完整停止命令**，因为vy/wz仍独立采样。该点比历史文字中的“15%停止”更精确。

L命令在episode内固定，command重采样周期设置为1e9s，实际在reset时重新采样。它没有Go2的A/B/C速度课程。L的Piper臂有carry正则，但不要求TCP到点、抓取、臂姿态跟踪或非零载荷操作。

### 2.2 Task M：固定世界点到达

每次采样：$r=3\sqrt U,\theta=2\pi V$，水平位置为当时底盘中心加$r(\cos\theta,\sin\theta)$，高度 $z\sim U[.03,1.10]$。点一经选定保持world方向不动，底盘移动后点不跟随。拒绝落入当前TCP成功球的点；未做IK/碰撞/可达性筛选。

环境并行平铺时，state和command存储“相对于该env origin、方向仍为world”的坐标，viewer再加origin。误把绝对场景位置和env-local-world坐标混合会造成系统性偏差；本轮契约/FK检查与PPO成功结果支持当前实现的基本一致性，但没有证明全部动力学接口无误。

8cm内且高度≥.20m、倾角<70°时计一次成功。本步reward仍基于旧goal，随后command采新goal，新观测使用新点；机器人与历史不reset。Replay保留产生该动作时的旧context和下一帧新context，避免把换目标误记成动力学跳变。episode30s独立超时，一个episode可以完成多个goal。

M不启用L速度跟踪：速度命令为0，M底盘速度惩罚仍存在。底盘可以为接近远点移动，但要在速度成本与到点奖励之间权衡。任务不包含TCP朝向、夹爪开合、物体接触或抓取，因此“Manipulation成功”在本文只表示TCP point reach。

### 2.3 Goal8与部署信息需求

$$g=[v_x,v_y,\omega_z,\ \Delta p_x^h,\Delta p_y^h,\Delta p_z^h,\ is_L,is_M],\quad
\Delta p^h=R_{heading}^{-1}(p_{target}^{world}-p_{root}^{world}).$$

L的相对目标xyz为0；M的速度分量为0。task one-hot为二选一，50:50环境配额不是同时loco-manipulation。历史里Replay context实际为7维 `[task, vx,vy,wz, target_world_xyz]`，Goal8是根据当前/预测state逐步计算的Actor输入，不能把两者维数混淆。

M虽然不把真实线速度直接送给Actor，**相对世界目标的计算仍需root位置与heading**。实机必须提供定位/目标相对感知；只靠编码器与IMU、没有位置来源不能直接完成同一world-point任务。Go2速度命令部署不需要这一位置输入。历史观测不能解决所有全局定位歧义。

TCP由可微FK取Piper两个固定指帧link7/link8原点的中点，属于夹爪中心代理。URDF没有独立标定TCP工具帧，不能直接当作实机末端标定点。实时command使用link7；本模型中两指帧固定在共同原点，已有一致性检查支持这一路径，但更换工具/模型后必须重验。

源码： [m1_reward.py](../../src/supermimic/tasks/m1_reward.py) 的 `sample_m1_disk_targets/m1_actor_goal/m1_goal_reached`；[commands.py](../../src/supermimic/envs/mdp/commands.py) 的 `M1PriorJointCommand`；[world_point_reward.py](../../src/supermimic/tasks/world_point_reward.py) 的 `tcp_position_world`；[StateIO](../../src/supermimic/envs/state_io.py)。

## 3. 训练配置、数据量与硬件成本

| 项目 | PPO | SuperMimic |
|---|---|---|
| 仿真 / 环境 | mjlab / MuJoCo Warp；2048 | 相同 |
| L/M | 固定1024/1024，共享Actor | 相同；Replay采样按任务平衡 |
| seed / prior | 2026 / 无 | 2026 / 无 |
| timestep / control | .005s×4 = .02s，50Hz | 相同 |
| 真实奖励/动作 | v2共享RewardCfg，v3/high-carry | 相同 |
| 实际学习墙钟 | 12601.320s | checkpoint12602.049s |
| worker含初始化 | 12604.695s | 12605.419s |
| 每环境真实步 | 17088 | 2979 |
| 真实transitions | 34,996,224 | 6,100,992 |
| 完整更新 | 534 PPO updates | 1176 WM事件，876 Actor updates |
| 每update真实rollout | 2048×32=65,536 | 不适用 |
| PPO minibatch/epoch | 1024；64minibatches×5epochs | 不适用 |
| WM窗口/batch/micro | 不适用 | 8；128；32；4fits/事件 |
| WM优化器step | 不适用 | 4704 |
| Actor H / batch / micro | 不适用（真实rollout32） | H32；128；32 |
| Replay | 不适用（独立rollout storage） | 256控制步/env=5.12s |
| Actor/Critic基础LR | 2e−4，adaptive | 2e−4/2e−4，constant基础LR |
| Actor ramp | 不适用 | 前500次实际Actor更新渐增 |
| 预热 | 上游on-policy采样 | 前40向量步填充窗口，WM-only300事件 |
| Critic | 真实GAE；value coef1，clipped value loss | 想象λ-return，5epochs×4minibatches |
| 探索 | 可学Gaussian初始.3，叠加环境.03 | 确定性Actor，环境噪声.03 |
| DR | 摩擦[.3,1.2]、bias±.01rad、COM各轴±.025m | 相同 |
| 观测/reset噪声 | 关闭 | 关闭，想象输入噪声也关闭 |
| Compile | 上游PPO正常执行 | WM/Actor torch.compile=None，避免16GiB主机编译压力 |
| Episode | 30s；h<.20m或倾角>70°终止 | 相同 |
| 硬件 | RTX4070 SUPER 12GiB / 16GiB RAM | 相同 |

PPO每完整update有5×64=320次优化step，534×320=170,880；SM876次Actor step、每次另20次Critic optimizer step，不能相互等价。SM每逻辑Actor更新生成128×32=4096预测transition，但它们仍源于拟合模型，不是新增真实信息。

等时长真实样本差为5.73615倍。SM计算消耗包括22D混合控制动力学、32步FK/网络计算与反传、4次WM fit和多次Critic拟合；小microbatch还增大调度开销。WM-only阶段约.22h后Actor才开始，Actor ramp也占了有限更新预算。更多计算不自动意味着覆盖更广或导数更准确。

## 4. Actor、Action22与Critic的真实结构

### 4.1 输入与共享Actor

每帧72D = 编码器相对位置22 + 关节速度22 + body omega3 + projected gravity3 + previous policy action22。**四轮角度在位置通道中置0**，保留维数但隐藏无界累积轮角；wheel qd仍可观测。10帧=720D，再加Goal8=728D，term-major展开。固定scale：q1、qd .2、omega .5、重力1、动作1、Goal1。10帧最旧至最新相隔.18s，常称.2s历史窗口。

两种Actor均 **728→2048→1024→512→256→22**，隐藏SiLU，末层tanh；4,252,950个线性参数。direct模式没有独立Motion Encoder、latent32不参与该计算。PPO另有22个log_std参数；std日志均值.30134→.47892。PPO采样action、tanh均值与带噪执行action需要区分，见Go2分析；部署取均值。

### 4.2 Action22是混合控制

顺序是四条腿各 `[abad, hip, knee, wheel]` 形成前16维，再6维Piper。腿为索引0/1/2、4/5/6、8/9/10、12/13/14；轮为3/7/11/15；臂为16–21。

- 12腿+6臂：归一化动作按默认姿态到绝对关节上下界作非对称位置映射，再软限位裁剪。不能用`default+固定scale`替代v3映射。
- 4轮：`clip(a,-1,1)×5 rad/s`，是轮速度目标，不是位置目标，也不是直接机体速度。
- 正负轮指令的机体响应由真实运动学和接触决定，不能仅看轮关节轴号给所有轮强行统一反号。

默认high-carry根高约.47m，臂默认关节 `[0,.5,-.4,.3,0,0]`；训练动作包络在 [保存task contract](data/m1_ppo_manipulation.json) 中给出。v3的静态动作bounds虽与历史数据包络相关，但本次没有加载运动学数据集或motion prior做学习。

### 4.3 两个Critic

都读 **95D** = canonical State57 + Goal8 + normalized ξ26 + zero root_error4，特权状态包含真实线速度、姿态、关节和物理随机化参数。网络95→256→128→1，SiLU，57,601个参数。

PPO用真实rollout和真实终止/timeout构造GAE训练Critic，策略梯度是基于真实advantage的log概率梯度。SM用想象状态、解析reward、软终止构造λ-return，再用终点 $\nabla_s V$ 反传Actor。网络结构相同不意味着它们学到的梯度质量相同。

## 5. World Model的State57、结构和误差尺度

State57布局精确为：

| 切片 | 含义 | 维数 |
|---|---|---|
| 0:3 | env-local world root位置 | 3 |
| 3:7 | root quaternion wxyz | 4 |
| 7:29 | 22关节位置，含轮角 | 22 |
| 29:32 | world root线速度 | 3 |
| 32:35 | body root角速度 | 3 |
| 35:57 | 22关节速度 | 22 |

canonical57 = root_z1 + heading去除后的rot6d6 + heading线速度3 + body omega3 + q22 + qd22；轮角在特征/误差中wrap为角度。ξ26=摩擦1+encoder bias22+COM3，按固定offset/scale归一化。

动力学 **105→2048→2048→2048→56**，隐藏SiLU，预测 `[dv_h3,dw_b3,dqd22,dp_h3,dtheta_b3,dq22]`，显式积分：

$$v'=v+R_h\Delta v_h,\quad\omega'=\omega+\Delta\omega_b,\quad\dot q'=\dot q+\Delta\dot q,$$
$$p'=p+\Delta t v'+R_h\Delta p_h,\quad Q'=Q\otimes\exp(\Delta t\omega'+\Delta\theta_b),\quad q'=q+\Delta t\dot q'+\Delta q.$$

输出仍包含自由位置修正，因此是带积分先验的学习动力学，不是严格刚体动力学约束。零输出时作匀速外推。归一化误差尺度依次为root position .05m、root rotvec .1rad、root linvel .5m/s、root omega1rad/s、q .1rad、qd2rad/s，NSE为56维归一化误差平方均值，不是百分比误差或R²。

保存学习式term head为4→64→64→1，WM总参数8,729,081（含head）；本轮 `M1RewardTask` 继承 `WorldPointReachTask` 的解析高度/倾角termination，想象Actor使用该解析logit而非学习头。成功事件也会截断当前goal的想象生存，故imagined termination不等于跌倒概率。

具体软终止公式（源码温度T=.015）为：

$$p_h=\sigma((.20-h)/T),\qquad p_{tilt}=\sigma((|1+g^b_z|-(1-\cos70^\circ))/T),$$
$$\hat d_{fall}=1-(1-p_h)(1-p_{tilt}).$$

这里同一个数值温度分别用于高度和无量纲倾角特征，不是有统一物理单位的距离。想象logit还把概率裁至$[10^{-6},1-10^{-6}]$。**软终止与真实硬阈值并不等价**：仅作公式推算，假设未来32步均直立、h=.219m且未成功，则$p_h\approx.220$，终端生存约$(1-.220)^{32}\approx3.5\times10^{-4}$。这不是实际轨迹统计，但说明低位状态即使仍高于真实.20m跌倒线，H32后段信号也可能被强烈衰减；软终止本身又产生远离边界的导数。它可能影响恢复、终值与目标信用分配，需要用相同起点记录逐步高度/生存权重、比较温度或吸收失败语义，不能据此认定它已经是饱和的触发原因。

8步训练=.16s，32步Actor想象=.64s。短窗误差低不能自动保证四倍时域、不同动作分布、接触切换和task Jacobian可靠。源码与Go2相同，reward专门调用 [M1RewardTask](../../src/supermimic/imagination/tasks.py) 和 [m1_reward.py](../../src/supermimic/tasks/m1_reward.py)。

## 6. 全部20项共享真实奖励

符号：$d=\|p^{TCP}-p^{goal}\|$；$g^b$为投影单位重力，$h$为root高度；动作无量纲，关节q/qd为rad、rad/s。除成功事件外，表中全是奖励率，真实RewardManager与想象task均只乘一次dt=.02。表内w是配置值，M时间项公式自行带负号。


| 名称 / 源码名 | 权重w | 分支 | 加权奖励率 | 单位 / 最大量级 | 性质与局部最优风险 | 一致性 |
|---|---|---|---|---|---|---|
| L线速度 / `l_linear` | 3 | L | $w e^{-\Vertv^b_{xy}-u_{xy}\Vert^2/0.15^2}$ | m/s误差平方；[0,3]/s | 稠密可微；只要求速度，不要求抬腿 | 共享真实/想象解析核；事件/终止处理见后文 |
| L偏航 / `l_yaw` | 1 | L | $w e^{-(\omega_z-u_z)^2/0.25^2}$ | rad/s误差；[0,1]/s | 稠密可微；侧移/转向能力仍需实测 | 共享真实/想象解析核；事件/终止处理见后文 |
| 粗到点 / `m_reach_coarse` | 2 | M | $w e^{-d^2/1.5^2}$ | 距离m；[0,2]/s | 稠密可微；远端梯度仍弱；不保证可达 | 共享真实/想象解析核；事件/终止处理见后文 |
| 精到点 / `m_reach_fine` | 1 | M | $w e^{-d^2/0.08^2}$ | 距离m；[0,1]/s | 远离8cm时接近零 | 共享真实/想象解析核；事件/终止处理见后文 |
| 接近进度 / `m_progress` | 0.25 | M | $w\tanh((d_{prev}-d)/(0.02\cdot0.5))$ | 有限差分接近速度按0.5m/s缩放；[−0.25,0.25]/s | 有符号可微；不是保证策略不绕路的潜在函数差 | 共享真实/想象解析核；事件/终止处理见后文 |
| 近点低速 / `m_hold` | 1 | M | $w e^{-d^2/.08^2}e^{-\Vert(p^{TCP}_{next}-p^{TCP}_{prev})/.02\Vert^2/.10^2}$ | TCP速度m/s；[0,1]/s | 稠密可微；成功马上换点使作用窗口短；不等于保持0.5s | 共享真实/想象解析核；事件/终止处理见后文 |
| 距离相关时间成本 / `m_distance_time` | 0.5 | M | $-w d/(d+0.5)$ | 距离m；(−0.5,0]/s | 稠密可微；不是常数时间罚，远处导数衰减 | 共享真实/想象解析核；事件/终止处理见后文 |
| 首次成功 / `m_success` | 12 | M | $(w/\Delta t)\mathbf1[d<.08\land h\ge .20\land tilt<70^\circ]$ | 率最高600/s；乘dt后每goal +12 | 布尔事件；几乎处处零路径导数；真实成功换点防重复 | 共享真实/想象解析核；事件/终止处理见后文 |
| 直立 / `upright` | 0.5 | L/M | $w e^{-(1+g^b_z)/.1}$ | 单位重力；[0,0.5]/s | 可微；倾斜后奖励及恢复导数均减小 | 共享真实/想象解析核；事件/终止处理见后文 |
| 底盘高度 / `height` | 0.5 | L/M | $w e^{-((h-.47)/.10)^2}$ | h单位m；[0,0.5]/s | 可微；h≈.219时位于核尾部，恢复导数很弱 | 共享真实/想象解析核；事件/终止处理见后文 |
| 竖直速度 / `vertical_velocity` | -0.2 | L/M | $w(v^b_z)^2$ | (m/s)²；最大0，无固定负下界 | 可微；静止低姿态不被此项惩罚 | 共享真实/想象解析核；事件/终止处理见后文 |
| 横滚俯仰角速度 / `roll_pitch_rate` | -0.05 | L/M | $w(\omega_x^2+\omega_y^2)$ | (rad/s)²；最大0 | 可微；静态倾斜不产生此项罚 | 共享真实/想象解析核；事件/终止处理见后文 |
| 腿默认姿态 / `leg_pose` | -0.02 | L/M | $w\sum_{j\in leg}(q_j-q^0_j)^2$ | 12腿关节rad²；最大0 | 可微；未包含轮子或臂 | 共享真实/想象解析核；事件/终止处理见后文 |
| 臂收纳 / `l_arm_carry` | -0.01 | L | $w\sum_{j=16}^{21}(q_j-q^0_j)^2$ | 6臂关节rad²；最大0 | 只对L，M不惩罚离开收纳姿态 | 共享真实/想象解析核；事件/终止处理见后文 |
| 动作变化 / `action_rate` | -0.01 | L/M | $w\operatorname{mean}_{22}(a-a_{prev})^2$ | 无量纲；有界动作率≥−0.04 | 可微；恒定±1也为0 | 共享真实/想象解析核；事件/终止处理见后文 |
| 位置软越限 / `joint_limit` | -1 | L/M | $w\sum_{j\notin wheel}([q^{min}_j-q_j]_+^2+[q_j-q^{max}_j]_+^2)$ | 18位置关节rad²；最大0 | 轮角度排除；界内动作饱和不受约束 | 共享真实/想象解析核；事件/终止处理见后文 |
| 关节速度 / `joint_velocity` | -1e-05 | L/M | $w\operatorname{mean}_{22}\dot q^2$ | (rad/s)²；最大0 | 含轮子；不同物理动作组没有单独归一化 | 共享真实/想象解析核；事件/终止处理见后文 |
| 关节加速度 / `joint_acceleration` | -1e-07 | L/M | $w\operatorname{mean}_{22}((\dot q_{next}-\dot q_{prev})/.02)^2$ | (rad/s²)²；最大0 | 真实与想象都用控制周期差分 | 共享真实/想象解析核；事件/终止处理见后文 |
| 控制幅度代理 / `control_effort` | -0.005 | L/M | $w\operatorname{mean}_{22}a^2$ | 无量纲；有界动作率≥−0.005 | 非测量力矩/功率；全饱和每步仅−0.0001 | 共享真实/想象解析核；事件/终止处理见后文 |
| M底盘速度 / `m_base_speed` | -0.1 | M | $w(\Vertv^b_{xy}\Vert^2+\omega_z^2)$ | m/s和rad/s平方直接混合；最大0 | M命令为0；移动会付成本，与远点快速到达有权衡 | 共享真实/想象解析核；事件/终止处理见后文 |



这20项是源码 `M1_REWARD_NAMES` 的有效集合，没有另加未启用的足端reward、接触reward、实际能耗或抓取成功reward。M关节速度正则含轮子，位置限位/腿姿态则排除轮子。`m_base_speed`实现是速度误差平方，但M的速度命令为0，因此化简成实际速度平方。

真实与想象共用`m1_reward_terms`，相同预测state输入时公式一致；真实输入是MuJoCo state，想象输入是WM state，因此行为与梯度仍可不同。真实稀疏成功后换点，想象只截断当前goal，不继续采新随机点。事件前向值相同并不等于反向梯度有同等学习能力。

### 6.1 每步量级与局部最优

L正任务率上界4、共享upright+height上界1；M粗/精/低速/进度正率上界4.25，共享正率1，连续项正率保守上界5.25/s。全饱和控制幅度成本为−.005/s，即每步−.0001；其他速度类惩罚在停住后也会变小。这个尺度不足以仅凭项名阻止恒定边界动作。

成功事件实现为奖励率12/.02=600，实际一次+12，并非每步连续获得+12。原设计用它超过近点等待的保守折扣正回报上界5.25×.02/(1−.99)=10.5，以缓解到点后换远点使策略故意停在阈值外的诱因。这是尺度动机，不能代替实际成功/停留行为验证。

时间罚 $-.5d/(d+.5)$ 对d=0/.1/.5/1/3m分别为0/−.0833/−.25/−.3333/−.4286每秒。导数 $-.25/(d+.5)^2$，d=2m时幅度.04 reward/(s·m)。它随距离变化，但远处梯度衰减，也不显式除以“合理预计到达时间”；是否符合所有距离任务的效率偏好仍需消融。

高度奖励的导数为 $-2(h-.47)/.1^2\times .5e^{-((h-.47)/.1)^2}$。h=.219m时奖励率约.00092、导数约.046 reward/(s·m)；到正常高度附近才更有恢复信号。动作rate只管变化，恒定±1时严格为0；真实最终该项均值约−3.17e−10/step。因此“有动作平滑/高度reward”不能证明能从低位饱和恢复。

粗距离σ1.5使d=3m的率从旧σ.5时约4.64e−16提高至.0366，确实改善远距离前向信号；但模型把动作方向导错时，有梯度也可能在真实物理中朝反方向更新。

### 6.2 SuperMimic独有终点项

$$J_{actor}=J_{shared}-\gamma^{32}w_d\,I_M\,I_{not\ completed}\,\tilde d_{32},\quad w_d=1,$$
$$\tilde d=\sqrt{\|p^{TCP}_{32}-p^{goal}\|^2+10^{-8}}-10^{-4}.$$

单条轨迹仅一次，不乘dt；Actor实际loss仍为$-J/32$。M mask来自context task，L无此项。若想象中在硬几何失败之前已成功，detach的completed mask清除此goal终点成本；success控制步的+12只拿一次，随后当前goal reward与bootstrap截断。过程reward用survival，终点负距离不乘survival，以免预测“死亡”抹掉负成本。

代价是：物理失败之后若模型仍滚动，终点可能优化已不可执行的假想状态。更一致的未来方案应使用失败吸收态保留未完成成本，再另加失败成本；不能简单用survival乘负距离，否则可能鼓励用死亡逃避任务。

Critic仍拟合共享逐步reward的λ-return，**不含这个额外终点项**。这可以被定义为Actor辅助规划启发式，但Critic与Actor总目标不完全一致；也没有对新随机goal后续价值自举。PPO真实MDP包含成功后继续换点执行，SM想象近似当前goal局部规划。因此本轮是等时间系统比较，不能说“除了算法，优化目标和MDP处理完全相同”。

源码确切位置：`M1RewardTask.terminal_reward`、`imagine`成功mask、`PolicyTrainer.update`中`J += gamma**horizon*terminal.mean()`；[objective.py](../../src/supermimic/imagination/objective.py) 是共享过程与Critic公式，终点项在其外部加入。没有在PPO奖励里隐式增加终点项。

## 7. 正式独立评测：结论与分母

6组：双方balanced、纯L固定(.12,0,0)、纯M随机点。每组64env、60s、seed123，确定性Actor、无DR/训练噪声；不是训练回报图。双方混合M成功率PPO172/242=71.07%、SM0/64；主纯M统计如下。


| 指标 | PPO | SuperMimic |
|---|---|---|
| L实际vx m/s | 0.164993 | -0.000658 |
| L XY RMSE m/s | 0.050466 | 0.136749 |
| L yaw RMSE rad/s | 0.079386 | 0.086212 |
| L跌倒率/结束episode | 0.00% | 0.00% |
| M目标成功率 | 71.22% | 0.00% |
| M成功goal数 | 334 | 0 |
| M结束goal数 | 469 | 128 |
| M活跃删失goal数 | 27 | 0 |
| M每env每min成功数 | 5.218750 | 0.000000 |
| M成功goal平均耗时s | 2.966525 | 不适用/无样本 |
| M时间平均TCP距离m | 0.583338 | 2.022180 |
| M跌倒率 | 25.19% | 0.00% |
| M结束episode数 | 135 | 128 |



PPO纯M469个结束goal包括334成功、34跌倒结束、101episode超时结束；27活跃goal为censored，不并入成功率。纯M episode结束135次，其中34跌倒，分母与goal不同；SM128goal全部超时。成功goal均值耗时仅在成功子集中计算，不代表无条件expected time-to-goal；失败/删失必须同时报告。PPO成功时间中位2.88s、P90=5.22s。

L实际SM vx≈−.000658m/s，96.18%时间|vx|<.02，明显反向低于−.02的时间约1.96%。把“vx负号很多”写成持续大幅后退不准确。PPO .12命令实速.16499，约37.5%相对超调，速度跟踪仍需优化。未补测L固定负向/侧移/纯yaw，不能宣称全面学会locomotion。


### 按目标世界高度m分桶

| 目标世界高度m | PPO 成功/结束goal | SM 成功/结束goal |
|---|---|---|
| [0.03, 0.3) | 40/121 = 33.06% | 0/41 = 0.00% |
| [0.3, 0.8) | 183/224 = 81.70% | 0/58 = 0.00% |
| [0.8, 1.1) | 111/124 = 89.52% | 0/29 = 0.00% |


### 按初始TCP距离m分桶

| 初始TCP距离m | PPO 成功/结束goal | SM 成功/结束goal |
|---|---|---|
| [0, 0.5) | 11/12 = 91.67% | 0/5 = 0.00% |
| [0.5, 1.5) | 66/94 = 70.21% | 0/28 = 0.00% |
| [1.5, 2.5) | 150/208 = 72.12% | 0/52 = 0.00% |
| [2.5, +∞) | 107/155 = 69.03% | 0/43 = 0.00% |



高度/初始距离桶只看结束goal，存在结束样本选择偏差；不同算法成功换点后目标序列与物理状态已不同。远距>2.5m成功107/155支持PPO不仅解决臂旁小目标；低位33.1%提示动作包络、碰撞和全身协调困难，但没有逐目标IK/碰撞可达性检验，不能把失败都叫“不可达”。

## 8. 已确认的退化：动作、姿态和历史反馈

4env、5s、无噪声真实小诊断统计1–5s，不能替代64env正式评测：


| 策略/任务 | root平均高度m | 平均∣a∣ | ∣a∣>.95 | 平均1−a² |
|---|---|---|---|---|
| PPO L | 0.4868 | 0.699581 | 26.93% | 0.427041 |
| PPO M | 0.3191 | 0.484458 | 14.74% | 0.658275 |
| SuperMimic L | 0.2193 | 0.999876 | 100.00% | 0.000247 |
| SuperMimic M | 0.2194 | 0.999899 | 100.00% | 0.000202 |



SM root≈.219m，仅比.20m终止线高约2cm；未测机身接触力，不能断言机身已贴地。goal反事实保持历史不变，L从+.12换−.12，SM动作差1.76e−6，M左右切点差7.76e−7；PPO约.119/.522。SM在该区域几乎失去指令敏感性。

`tanh`导数 $1-a^2$ 在实际SM动作下约.0002，使目标和正则的更新信号难以穿过饱和末层。43份checkpoint参数全部有限，排除参数NaN作为这次表现的直接解释；不代表网络梯度可靠。固定名义物理输入、零上一动作历史下，最终平均|a|=.8522，38.6%超过.95；因此它不是对任意输入永远恒定±1的函数。

纯CPU反事实：物理观测和goal固定，只把Actor输出反馈到10帧上一动作历史，64次前向、无物理运动和优化。60m快照|a|从.07885→.09808，90m .04757→.03664，100m .68792→.96562，120m .62255→.98202，final .85224→.99848，final100%动作>.95。说明**上一动作反馈本身足以放大已有策略的饱和**；不证明它是训练最初触发因素，也不应未经消融删掉历史、破坏部署契约。

末层权重Frobenius范数.0469→.8837，约18.8倍；最终平均bias仅.0117。饱和并不只是bias被推大，前层增益/激活和反馈也参与。训练日志在it1800约03:32首次imagined mean|a|>.95，it1900约03:41超过.99；这是训练中逐渐退化，不是导出后突然改变行为。

## 9. WM预测与真实物理动作导数

最终WM，12条分支=6对±动作方向，名义初态围绕最终Actor初始动作执行32步常动作；同真实state和context重放WM。±.03主测、±.01复核，固定诊断目标[1.5,0,.8]。不是闭环策略轨迹，也不是整个Replay的泛化评测。


| H | 时域s | 底盘XY误差m | TCP误差m | vx RMSE m/s | 全状态NSE |
|---|---|---|---|---|---|
| 1 | 0.02 | 0.0080 | 0.0177 | 0.1327 | 2.348 |
| 8 | 0.16 | 0.0496 | 0.1659 | 0.6686 | 2.953 |
| 16 | 0.32 | 0.1575 | 0.3824 | 0.9452 | 4.751 |
| 32 | 0.64 | 0.3192 | 0.3069 | 0.3004 | 10.354 |



TCP H32误差≈.307m，相比8cm成功半径大约3.8倍；但误差非单调，H16比H32更大，不能称每步固定指数增加。NSE10.35是局部反事实分布结果，不能与Replay平均NSE .117直接视为同分布指标。

已在两个epsilon下保留的反号例子：四小腿同向H8的∂vx/∂α真实约−.455、WM+.467；同方向∂distance真实+.0369、WM−.0322；四轮前后差分H32的∂distance真实−.0194、WM+.0898。见 [M05](figures/M05_m1_wm_gradient_audit.png)。WM自身有限差分与autograd大体接近，支持“学到的局部物理偏导不正确”，而不是仅仅autograd程序不能求导。

接触物理不光滑，小量差分有数值噪声；只挑经过两种扰动幅度复核、信号可见的例子。不能外推为所有状态/动作方向反号，不能由此给某个轮轴全局翻符号。应扩展到正常高位、起步、轮速正反、臂运动、接触转换和临界失败分层状态，分别验证state预测与task/action Jacobian。

Replay256只覆盖每env5.12s，远短于30sepisode。固定预热300WM事件无法防止Actor分布改变后模型失准；训练早期低NSE可能仅来自近静止数据。SM最终Replay NSE8约.117，H32对照却没有直接被训练监控约束。更多轮数只有在覆盖和模型导数改善时才可能有益。

## 10. Critic、过程和终点梯度的冻结分解

6份快照、相同4个名义真实起点（L±.12、M左右点），物理zero-action settle1s，再冻结WM/Critic，分别对Actor参数计算 $J/32$ 的三项梯度。不是当时历史Replay起点，没有做optimizer更新，不能还原训练每一步的因果。


| 快照min | 过程梯度L2 | Critic梯度L2 | 终点梯度L2 | 总梯度L2 |
|---|---|---|---|---|
| 15 | 0.007319 | 0.095478 | 0.000775 | 0.088379 |
| 60 | 0.310444 | 0.403246 | 0.009197 | 0.184021 |
| 90 | 0.039701 | 0.115188 | 0.002534 | 0.125793 |
| 105 | 0.586034 | 0.195352 | 0.047288 | 0.386209 |
| 120 | 0.519293 | 0.312441 | 0.051165 | 0.823513 |
| 210 | 0.057495 | 2.966088 | 0.004159 | 3.014520 |



15m和最终Critic梯度局部主导；105m过程项更强，终点项与总梯度近乎反向；120m三项相加有抵消。最终critic≈2.966、过程≈.0575、终点≈.00416。**不能把失败简单归因于终点权重1太大**，也不能从这张表证明Critic是唯一错误源。Critic没有真实value-gradient真值校验；范数只描述信号大小，未证明方向错。

Actor目标包含 $\nabla_{s_H}V(s_H)\,\partial s_H/\partial\theta$，一个数值拟合良好的Critic也可能在局部或分布外有很大、不可靠导数。想象Critic目标又自举当前V，可能自洽地强化模型偏差。末端survival也参与导数；120m的生存导数对总梯度投影约25%，其他若干快照较小，不能默认生存项主导所有阶段。

梯度分项范数不能当因果百分比；相反方向抵消时投影比可超过100%。现有证据尚未执行关闭Critic终值、stop-gradient终值或目标网络等训练消融。直接归咎终点距离、Critic、L/M冲突中的任何单一因素都超出证据。

## 11. 为什么PPO能学到一部分，而SM越来越差

PPO额外Gaussian探索显著更强，真实样本多5.74倍，真实advantage能直接利用+12事件结果；不需要知道“哪个动作怎样连续改变下一状态”。SM小幅噪声围绕确定性策略采样，在3m远点分布下较难获得接近成功数据；对硬阈值直接反传几乎处处为0，主要依赖距离/进度和Critic梯度。

即使reward相同，PPO依赖采样回报的概率排序，SM依赖代理模型/价值函数对动作和state的微分。前向reward可以正确而导数错误，甚至在想象中越来越高、真实任务越来越差。真实WM反号、上一动作反馈放大、弱恢复正则、近乎饱和tanh可以组成一个退化闭环，但覆盖恶化与每项贡献仍需因果消融。

M1有轮地接触、腿高/身体姿态、6D臂和全局点目标耦合；H32仅.64s，与米级到达相比仍短。Critic必须跨时域赋值。Go2的稠密速度目标、12D位置控制与阶段课程更容易在短窗内形成可用反馈，这是一种任务结构解释，尚没有同维数/同探索的因果对照。

PPO成功并不证明reward最优或SM无实现错误：它的低位目标跌倒、速度超调、单seed和未验证可达性都是实际限制。SM零跌倒也不证明稳定性更好，因为低位停住能躲过现有终止条件。

## 12. 建议的下一轮验证顺序

1. **保持H32，先做物理可信度校准。** 固定独立校准状态集，覆盖正常/低位/接触转换；H1/8/16/32分别报告base、TCP、wheel误差和多epsilon动作方向一致率。不要只凭Replay NSE和固定预热次数放行Actor。
2. **加入真实评估闭环与Actor输出约束诊断。** 按动作组记录logit、饱和率、goal sensitivity、root height；保留真实表现最好的checkpoint和optimizer回滚点。比较较小LR/退火、输出变化约束、EMA评测策略，但任何候选都需配对测试。
3. **定位Critic影响。** 同初始权重和真实Replay，分别比较完整终值、弱终值、无终值或停止其state梯度；H32保持不变。无Critic会短视，若更稳定也不代表永久删除更好。独立验证真实return排序与V局部变化方向。
4. **修正数据覆盖。** 使用物理验证Reset Bank、normal/recovery/near-goal分层Replay；若用PPO轨迹预训练WM，明确计入额外数据/计算预算，标为新实验。
5. **再做reward与目标课程。** 在真实可达性筛选后由近及远、低中高分桶；检验M底盘速度成本、hold与快速换点冲突；用吸收失败代价统一终点语义。原始全分布评测必须保留，避免只在容易目标上报高成功。
6. **分开L-only/M-only定位，再回混合。** 只有单任务也失败才优先共同模型问题；若单任务成功而混合失败，再看共享参数梯度冲突。最终再做多seed、等时间与等真实样本两套公平比较。

以上为后续方案，本次没有修改或启动任何训练。视频由用户录制，固定世界点与L任务的准确命令见 [PLAY_COMMANDS.md](PLAY_COMMANDS.md)。
