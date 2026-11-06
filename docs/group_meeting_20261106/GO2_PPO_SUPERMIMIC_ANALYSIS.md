# Go2：PPO 与 SuperMimic 的训练、奖励、网络及真实评测

2026-11-06。本文的主比较是 **seed=2026、从零各训练约2.5小时**；旧对角步态模型作为训练历史单独分析。本次只读取已有权重、配置、TensorBoard和评测结果，没有训练、更新策略、修改原有研究代码或生成新评测数据。图表由本目录脚本从保存的原始JSON自动生成。

## 组会展示版

1. 相同Go2物理模型、3072环境、50Hz、真实奖励与Actor均值网络，从零训练约2.5h。
2. 本轮SM九条固定指令均未跌倒；PPO前进与前进混合指令可用，纯侧移与原地转向明显失败，后退响应很弱。
3. 零跌倒不代表漂亮步态：新SM低速仍拖地；旧明显对角步态来自约4.55h完整权重链。
4. 结论限于单训练seed和名义平地；最终课程PPO停在B、SM到C，因此不是相同命令经历的严格算法消融。

推荐图： [G01配置](figures/G01_go2_training_config.png)、[G02完整结果](figures/G02_go2_velocity_results.png)、[G03全部指令误差](figures/G03_go2_velocity_comparison.png)、[G04步态历史](figures/G04_go2_gait_history.png)、[G05奖励](figures/G05_go2_reward_summary.png)、[G06公平比较步态指标](figures/G06_go2_fair_gait.png)。播放命令见 [PLAY_COMMANDS.md](PLAY_COMMANDS.md)。

## 1. 证据与实验身份

实际工作区 `/home/yifeiy/ws_supermimic`，分支main，审计起点HEAD `f3f29e86073593277244b1d2d307c2b28b8411cc`。工作区存在先前未提交训练实现，不能把HEAD当作这些实验的完整源码快照。保存的config、checkpoint参数形状、原始评测和本次审计源码快照共同构成证据，历史完整训练源码并未全部封存。详见 [实验证据索引](EXPERIMENT_PROVENANCE.md)。

主要模型：

- PPO：`logs/go2_comparison/ppo_2p5h_seed2026/model_758.pt`，推理包位于 `~/rl_sar/policy/go2/ppo_velocity/policy_inference.pt`。
- SM：`logs/go2_comparison/supermimic_2p5h_seed2026/ckpt_latest.pt`，it18600；不是旧续训权重。
- 旧模型：`logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_00{22000,24000,34000}.pt`。这些路径的实际文件、SHA256、参数形状见 [manifest](data/experiment_manifest.json)。

PPO复用了 `giangdao1402/go2_velocity` 的mjlab/RSL-RL训练链，固定上游SHA `23a364b1aafc454b5c281b9f115de434e6db2280`；本地只适配环境、Actor接口、导出与预算，不另写PPO、GAE或rollout。

## 2. 实际训练配置与计数

| 项目 | PPO | SuperMimic |
|---|---|---|
| 仿真 | mjlab / MuJoCo Warp，同Go2模型 | 相同 |
| 并行环境 / seed | 3072 / 2026 | 3072 / 2026 |
| 物理 / 控制 | 0.005s，decimation4，50Hz | 相同 |
| Episode | 10s；h<0.17m或倾角>70°跌倒 | 相同 |
| 学习循环墙钟 | 9004.077s | 9004.034s |
| 每环境真实步 | 24,288 | 18,600 |
| 真实transition | 74,612,736 | 57,139,200 |
| 更新 | 759完整PPO updates | 7424 WM事件；7224 Actor优化器更新 |
| WM优化器step | 不适用 | 7424×4=29,696 |
| Actor逻辑batch | 每minibatch1024 | 1024条H32想象起点，micro512 |
| Rollout | 每环境32真实步；总98,304 | H32；每逻辑更新32,768预测transition |
| Epoch | 5×96 minibatches/update | WM4 fits/事件；Critic5×4/Actor update |
| Actor优化器 | Adam，基础LR2e−4，adaptive-KL | Adam(β=.9,.95)，基础LR2e−4 |
| Critic优化器 | 同PPO优化器基础LR；Value loss系数1 | Adam，基础LR2e−4，5epochs×4minibatches |
| LR调度 | adaptive，最终日志约1e−5 | cosine，ramp1000 Actor updates；并非2.5h内完整衰减 |
| γ / λ | .99 / .95 | .99 / .95 |
| Replay | 不适用（PPO rollout storage） | 每环境256控制步≈5.12s |
| WM batch/window | 不适用 | 1024窗口，micro512，8步自回归 |
| 环境探索 | 75%环境Gaussian .1；25% OU σ=.3、θ=2 | 相同环境噪声 |
| 额外策略随机性 | 可学习Gaussian，初始std .3 | 不适用，确定性Actor |
| 观测/reset噪声 | 关闭 | 关闭；imagination输入噪声也关闭 |
| DR | 足摩擦U[.3,1.2]每reset；encoder bias±.01rad每reset；root COM各轴±.025m启动采样 | 相同 |
| Curriculum | A起始，最终B | A起始，最终C |
| 硬件 | 本机RTX4070 SUPER 12GiB，16GiB RAM | 相同 |

这里“Actor更新”需谨慎：PPO每个完整update包含5×96=480次minibatch优化；759×480=364,320次算法优化step，但每步也更新Critic和策略分布，不能把它与7224次H32路径梯度更新直接相除比较优劣。SM每次真实向量步产生3072条transition，iteration就是这种向量步；PPO iteration是32步rollout加更新。想象transition不是新真实样本。

PPO在等墙钟预算内的真实样本量是SM约1.306倍。时间包括学习循环内编译、仿真、更新和存盘，初始化/评测另计。PPO完成文件的9004.077s与最后一次progress的9004.048s来自停止与存盘时间点差异，不应把几毫秒差异当成实验偏差。

## 3. 速度任务与课程机制

命令为机体系目标 $u=[v_x^{cmd},v_y^{cmd},\omega_z^{cmd}]$。实际线速度保存在world-frame state中，reward先旋转到body-frame，再比较XY；偏航使用body z角速度。命令第三维是yaw rate，绝不是竖直速度。

`VelocityCommand._sample_stage_a/_sample_stage_b/_sample_stage_c/_resample_command` 和 `_evaluate_window` 在 [commands.py](../../src/supermimic/envs/mdp/commands.py)。实际保存配置使用：

| 阶段 | 命令分布 | 保留与重采样 |
|---|---|---|
| A | 前进vx∈[.25,.60]；vy=wz=0；5%静止 | 5s重采样 |
| B | vx∈[−.2,.7]、vy∈[−.15,.15]、wz∈[−.4,.4] | 35%继续采A，其余B |
| C | vx∈[−.4,1]、vy∈[−.3,.3]、wz∈[−.8,.8] | 25%A、25%B、50%C |

每256个结束episode评一次窗口，连续2个通过才晋级。A要求至少128个moving episode、存在静止样本、速度RMSE<.18m/s、静止速度<.12m/s、跌倒率<10%。B还要求至少64个yaw episode、速度RMSE<.20、yaw RMSE<.20rad/s，4方向桶均至少2个episode，3幅值桶覆盖至少2/3。评估使用该窗口episode统计，并非固定测试集。实际命令分布随晋级而改变，双方相同规则不代表看过相同数据。

### Actor没有真实机体线速度，为什么仍能跟踪

Actor读取编码器位置/速度、IMU角速度、投影重力、上一动作与目标命令。训练时真实机体速度进入reward和privileged Critic；SM的WM也读取真实速度。因此训练有正确的速度监督信号，部署Actor无需直接读取该量。

历史观测能反映近期关节运动、机体姿态及控制响应，帮助网络从部分可观测信息推断运动状态；目标命令也能学习前馈的动作-速度关系。但本体感觉不能在所有接触/打滑状态下唯一确定world平移速度，不能把“能跟踪”解释为网络必然准确估计了机体速度。滑地、外力、时延下的鲁棒性必须另测。

**本轮Go2不是10帧，而是5帧。** 每帧42D：12维$q-q^0$、12维qd、3维body omega、3维重力、12维上一动作。5帧=210D，再拼当前Goal3=213D。展平为term-major：一个观测项内历史从旧到新，再拼下一个项。5次50Hz采样的最旧到最新相隔.08s，通常称5帧/.1s窗口。M1才使用10帧。

## 4. 奖励完整审计

公式以下一状态为主；$g^b=R(q)^{-1}(0,0,-1)$、$h$为root高度，$[x]_+=\max(x,0)$。全部表内权重是奖励率权重，最终一次且仅一次乘 $\Delta t=.02$。PPO和SM真实环境使用相同`RewardManager`适配器和纯函数核，不是复制两套公式。

源码： [velocity_reward.py](../../src/supermimic/tasks/velocity_reward.py) 的 `velocity_reward_terms`、`go2_velocity_reward_term`；[go2_gait.py](../../src/supermimic/tasks/go2_gait.py) 的 `go2_gait_reward_terms`；[multirobot_env_cfg.py](../../src/supermimic/envs/multirobot_env_cfg.py)。当前目录的 [源码快照](data/source_snapshot/src/supermimic/) 固定本次审计内容；通用 `envs/mdp/rewards.py` 还有motion-tracking奖励，并不是这个velocity任务额外启用的奖励。


| 中文 / 源码名 | 实际权重w | 加权奖励率公式 | 归一化、单位与尺度 | 真实与想象 | 行为影响 |
|---|---|---|---|---|---|
| XY速度跟踪 / `track_lin_vel_xy` | 3 | $w[\exp(-\Vertv^b_{xy}-u_{xy}\Vert^2/0.09)-\exp(-\Vertu_{xy}\Vert^2/0.09)]$ | m/s平方按0.09归一化；率≤3 | 两算法真实奖励同核；SM想象按预测状态计算 | 直接要求真实机体跟踪；低抬脚仍可高回报 |
| 命令方向 / `command_direction` | 0.5 | $w\frac{\Vertu_{xy}\Vert}{\Vertu_{xy}\Vert+0.1}\tanh[\frac{v^b_{xy}\cdot u_{xy}}{\max(\Vertu_{xy}\Vert,10^{-4})\,0.3}]$ | 有符号、绝对值<0.5；0.1/0.3单位m/s | 两算法真实奖励同核；SM想象按预测状态计算 | 鼓励沿目标方向运动；纯yaw时为0 |
| 偏航跟踪 / `track_yaw_rate` | 1.2 | $w[\exp(-(\omega^b_z-u_z)^2/0.09)-\exp(-u_z^2/0.09)]$ | 角速度误差按0.09 (rad/s)²归一化；率≤1.2 | 两算法真实奖励同核；SM想象按预测状态计算 | 同一0.09用于XY与yaw；不要误写成不同σ |
| 直立 / `upright` | 0.5 | $w\exp(-\Vertg^b_{xy}\Vert^2/0.16)$ | 重力为单位向量；[0,0.5] | 两算法真实奖励同核；SM想象按预测状态计算 | 倾斜约束；并非姿态硬限制 |
| 竖直速度 / `base_lin_vel_z` | -1 | $w(v^w_z)^2$ | (m/s)²；无固定负下界，最大0 | 两算法真实奖励同核；SM想象按预测状态计算 | 抑制跳动；也可能使保守步态有利 |
| Roll/Pitch角速度 / `base_ang_vel_xy` | -0.05 | $w\Vert\omega^b_{xy}\Vert^2$ | (rad/s)²；最大0 | 两算法真实奖励同核；SM想象按预测状态计算 | 抑制机体快速摆动 |
| 动作变化 / `action_rate` | -0.03 | $w\sum_{j=1}^{12}(a_j-a_{j,prev})^2$ | 归一化动作平方和；[-1,1]内率≥−1.44 | 两算法真实奖励同核；SM想象按预测状态计算 | 平滑动作，但不约束恒定边界动作 |
| 关节力矩 / `joint_torque` | -1e-05 | $w\sum_j\tau_j^2$ | (N·m)²；最大0；力矩受执行器限幅 | 两算法真实奖励同核；SM想象按预测状态计算，用物理代理 | 真实MuJoCo测量；想象使用限幅PD代理，不能宣称逐值相等 |
| 关节加速度 / `joint_acceleration` | -2.5e-07 | $w\sum_j\ddot q_j^2$ | (rad/s²)²；最大0 | 两算法真实奖励同核；SM想象按预测状态计算，用物理代理 | 真实joint_acc；想象(下一qd−当前qd)/0.02，时间尺度不同 |
| 偏离默认姿态 / `joint_deviation` | -0.02 | $w\sum_j(q_j-q^0_j)^2$ | rad²平方和；最大0 | 两算法真实奖励同核；SM想象按预测状态计算 | 压制极端关节姿态，也有默认姿态偏置 |
| 软限位 / `joint_limit` | -1 | $w\sum_j([q^{min}_j-q_j]_+^2+[q_j-q^{max}_j]_+^2)$ | rad²；最大0；界内为0 | 两算法真实奖励同核；SM想象按预测状态计算 | 不限制软限位内的极限动作 |
| 低机身高度 / `base_height_low` | -5 | $w[0.22-h]_+^2$ | m²；最大0；h=0.20时率−0.002 | 两算法真实奖励同核；SM想象按预测状态计算 | 0.22m以上无高度引导，不是追踪默认高度 |
| 足端抬高 / `feet_clearance` | 0 | $w\,m\,\operatorname{mean}_i[e^{-((h_i-.05)/.025)^2}\tanh((\sqrt{\Vertv_{i,xy}\Vert^2+10^{-6}}-.001)/.5)]$ | 当前恒为0；高度m、速度m/s | 两算法真实奖励同核；SM想象按预测状态计算 | 明确关闭；不能将旧较好步态归功于抬脚奖励 |
| 对角协调 / `diagonal_gait` | -0.15 | $w\,m[\frac12((s_{FL}-s_{RR})^2+(s_{FR}-s_{RL})^2)+(\frac{\sum_i s_i-2}{2})^2]$ | 无量纲；当前负权重；率在[−0.15,0] | 两算法真实奖励同核；SM想象按预测状态计算 | 软几何协调，无相位时钟，无固定步频 |
| 足滑 / `feet_slip` | -0.2 | $w\operatorname{mean}_i[\sigma((.015-h_i)/.005)\Vertv^w_{i,xy}\Vert^2]$ | (m/s)²；最大0；平地几何近地权重 | 两算法真实奖励同核；SM想象按预测状态计算 | 权重较小且依赖正确WM速度导数；存在不等于足滑消失 |



足端公式里 $h_i=p^{foot}_{i,z}-.022$ 是足底离平地高度，足端来自可微FK；$s_i=\sigma((h_i-.025)/.008)$ 是软腾空程度。移动门控：

$$m=\operatorname{clip}\left(\frac{\sqrt{\|u_{xy}\|^2+(.25u_z)^2+10^{-8}}-10^{-4}-.03}{.12},0,1\right).$$

没有相位时钟、空中时间计时或接触力。滑移项静止也开启；对角与可选抬脚项由m门控。纯yaw可开启门控但不保证满幅；yaw=.4时等效速度约.1m/s，门控约.58。

### 静止基线的真实含义

从tracking reward中减去的是给定命令下静止所得的常数，不会改变固定命令单步reward对速度的导数；因此不能声称它在数学上直接“增大了前进梯度”。它改变绝对回报、Critic目标、不同命令比较及生存加权回报，在可终止任务和自举值函数里可能影响行为。方向项才额外提供沿命令方向的有符号引导。不要把静止基线当成充分的防停滞保障。

### 真实与想象中的力矩/加速度不完全相同

真实reward用MuJoCo `qfrc_actuator` 和 `joint_acc`；SM想象没有执行器内部状态，按相同动作映射重构PD力矩并限幅，关节加速度用控制周期有限差分。PD代理为：

$$q^{sp}=\operatorname{clip}_{soft}(q^0+.25a^{exec}),\quad
\hat\tau=\operatorname{clip}_{effort}(40(q^{sp}-q)-1\dot q).$$

这是共享目标下的可微代理，而非保证每个状态逐项一致。实际reward还以clean/policy action计算动作变化，不能把执行噪声后的动作混进历史项。足端核在真实/想象完全共享，但想象输入是WM预测，错误仍会传播。

### 为什么会拖地，对角和足滑为什么不够

速度任务对足端离地没有必然要求；低幅步态如果能前进且姿态稳定，也能得到高回报。动作变化、加速度和姿态偏离成本可能偏好小幅运动。当前抬脚权重明确为0。

对角项惩罚同组摆腿不一致和偏离两足支撑，但不要求周期性交替、特定步频或每次抬升5cm。足端轻微离地就可能改变软支撑分数，且.15相对速度率3较小。足滑项也只是系数.2的均值速度平方，奖励权衡允许非零滑移；接触状态是高度代理，WM速度和梯度误差也会干扰它。

跌倒定义只检查高度/倾角，能存活的拖地步态不违规；跌倒时四脚峰值高也不代表优良摆腿。因此必须同时报告速度、滑移、四足高度、支撑比例和跌倒，不能只选其中一项。

## 5. 网络与优化机制：实际维度


| 机器人 | Actor输入 | Critic输入 | WM输入 | WM输出 | Actor参数（不含std） |
|---|---|---|---|---|---|
| Go2 | 5×42+3=213 | 37+3+16+4=60 | 37+12+16=65 | 36 | 3195660 |
| M1-Piper | 10×72+8=728 | 57+8+26+4=95 | 57+22+26=105 | 56 | 4252950 |



### PPO

`AlignedActor`复用SM `Policy`中的均值MLP：**213→2048→1024→512→256→12**，隐藏SiLU，末端tanh，固定缩放位置1、qd .2、omega .5、重力1、动作1、命令1；关闭running observation normalization。参数3,195,660，Gaussian分布另有12个可学习log_std参数。原始std配置.3，日志均值从.2998到.3986。

策略均值 $\mu=\tanh(f_\theta(o,g))$；PPO训练从未截断Gaussian采样 $a\sim\mathcal N(\mu,\mathrm{diag}(\sigma^2))$，样本可超出[-1,1]。环境再叠加执行噪声并限幅映射PD目标，历史记录的是传给环境的policy action（PPO时是采样值），不是策略均值，也不是带环境噪声的执行动作。评估/导出使用均值。因此均值饱和率、采样值超界率、执行限幅率是三个指标，不能混写。

Critic输入 **60D**：canonical State37 + Goal3 + normalized ξ16 + zero root_error4；网络60→256→128→1，SiLU，48,641参数。真实速度和物理参数为特权输入，部署不需要Critic。

PPO配置：clip=.2、γ=.99、λ=.95、entropy=.01、Value loss系数1、clipped value loss、全局advantage normalization、5epochs、96minibatches、max_grad_norm1、desired KL .01、自适应LR。优化器更新目标来自真实rollout的GAE，不需对MuJoCo反传。

### SuperMimic Actor / WM / Critic

Actor同213D直接Goal路径；`Policy.encoders`在direct模式为空，即使config保留`encoder_hidden`和latent32，也没有一个实际参与计算的独立Motion Encoder。12D输出映射为 `q_target=q_default+0.25*a`，关节顺序FL/FR/RL/RR；Kp40/Kd1，力矩限幅每腿23.7/23.7/35.55N·m。

State37布局：root xyz3、quat wxyz4、q12、world线速度3、body角速度3、qd12。canonical State37去除全局XY与yaw，使用root_z1、局部rot6d6、heading线速度3、body omega3、q12、qd12。ξ16为摩擦1、encoder bias12、COM xyz3。固定input/error/output scale，不使用running normalization。

WM动力学 **65→2048→2048→2048→36**，SiLU，最后层零初始化；预测速度增量与积分残差，而非直接输出无约束下一状态。保留学习式Termination Head **4→64→64→1**，输入高度/倾角误差及其绝对值。WM总线性参数8,606,181（含该head）。本轮任务的Actor想象终止改由 `VelocityCommandTask.termination_logits` 的解析高度/倾角soft gate计算，保存的`termination_mode=analytic_height_tilt`可复核；**不能把学习头BCE准确率当成此任务Actor终止模型可靠性**。

Critic同60→256→128→1。`PolicyTrainer.update`每次在H32预测上构造固定λ-return target，用detach后的feature拟合；Actor阶段冻结WM和Critic参数，保留它们对state/action的导数。

### H32究竟怎样更新Actor

$$c_0=1,\qquad c_{t+1}=c_t(1-\hat d_{t+1}),$$
$$J=\mathbb E\left[\sum_{t=0}^{31}\gamma^t c_t r_{t+1}+\gamma^{32}c_{32}V_\psi(\hat s_{32},g,\xi)\right],\qquad L_\pi=-J/32.$$

步长.02s，想象.64s；WM仅用8步/.16s窗口自回归拟合。生存权重从高度/倾角soft boundary产生，reward为解析核。Actor梯度经过每步观测重构、动作、WM积分、FK/reward和末端Critic传播。梯度norm clip1与逐元素clip.1限制单次梯度，不等于PPO probability-ratio clipping。

Critic目标：

$$R_{32}=V_{32},\qquad R_t=r_{t+1}+\gamma(1-\hat d_{t+1})[(1-\lambda)V_{t+1}+\lambda R_{t+1}],$$
$$L_V=\mathbb E[(V_t-\operatorname{stopgrad}(R_t))^2].$$

使用当前Critic自举，无独立EMA target network。低Value loss只说明对构造目标拟合好，不能证明其state梯度对真实回报正确。

源码函数：`Policy/Decoder/Critic`，`WorldModel.step/window_loss`，`StateSpec.canonical/integrate`，`WorldModelTrainer.update`，`imagine`，`policy_objective/gae_targets`，`PolicyTrainer.update`。分别在 [nets.py](../../src/supermimic/policy/nets.py)、[world_model](../../src/supermimic/world_model/)、[imagination](../../src/supermimic/imagination/)。架构图见 [A01](figures/A01_algorithm_architecture.png) 和 [A04](figures/A04_network_dimensions.png)。

## 6. 九指令正式评测

数据来自 `logs/final_evaluation_20261104/{ppo,supermimic}_velocity_{evaluation,extra}.json`，已原样复制到本目录data并记录SHA256；不是从报告截图抄数。各指令64个结束episode，seed2026、确定性Actor、10s/episode，无DR/观测/执行/reset噪声，固定命令关闭课程。64环境名义初态相同，不是64个独立扰动试验。


| 命令vx/vy/wz | PPO实际vx/vy/wz | SM实际vx/vy/wz | PPO XY/yaw RMSE | SM XY/yaw RMSE | 跌倒PPO/SM | 结束episode PPO/SM |
|---|---|---|---|---|---|---|
| [0.0, 0.0, 0.0] | (0.038, -0.006, -0.022) | (0.000, 0.001, 0.004) | 0.0759/0.0373 | 0.0135/0.0143 | 0.00%/0.00% | 64/64 |
| [0.3, 0.0, 0.0] | (0.343, -0.004, -0.014) | (0.282, 0.003, -0.009) | 0.0659/0.0365 | 0.0410/0.0176 | 0.00%/0.00% | 64/64 |
| [0.5, 0.0, 0.0] | (0.545, -0.020, -0.001) | (0.458, -0.001, -0.019) | 0.0856/0.0414 | 0.0714/0.0283 | 0.00%/0.00% | 64/64 |
| [0.0, 0.3, 0.0] | (0.136, 0.350, -0.029) | (-0.024, 0.310, -0.008) | 0.2771/0.1716 | 0.0460/0.0179 | 100.00%/0.00% | 64/64 |
| [0.0, 0.0, 0.4] | (0.029, -0.030, 0.118) | (-0.035, 0.006, 0.392) | 0.1213/0.3475 | 0.0388/0.0425 | 100.00%/0.00% | 64/64 |
| [-0.15, 0.0, 0.0] | (-0.010, 0.003, -0.031) | (-0.168, 0.003, 0.000) | 0.1459/0.0586 | 0.0339/0.0264 | 0.00%/0.00% | 64/64 |
| [0.0, 0.1, 0.0] | (0.082, 0.147, -0.042) | (-0.021, 0.100, -0.010) | 0.2381/0.1294 | 0.0323/0.0208 | 100.00%/0.00% | 64/64 |
| [0.0, 0.0, 0.2] | (0.026, -0.004, 0.091) | (-0.011, 0.009, 0.182) | 0.0763/0.1574 | 0.0239/0.0409 | 40.62%/0.00% | 64/64 |
| [0.3, 0.1, 0.2] | (0.344, 0.090, 0.186) | (0.278, 0.104, 0.197) | 0.0726/0.0681 | 0.0445/0.0246 | 0.00%/0.00% | 64/64 |



XY RMSE是每个episode内 $\sqrt{\mathrm{mean}(\|v_{xy}-u_{xy}\|^2)}$ 再平均；yaw同理。均值含起步瞬态，跌倒episode比存活episode轨迹短。不能把跌倒前均值恰好接近命令当作成功。

PPO在.3m/s前进时实速约.343，存在超调；−.15指令时仅约−.010，后退基本未学会。纯侧移.1仍跌倒，说明失败不只是.3超出最终B课程vy范围；纯yaw .2有40.6%跌倒，.4全跌倒，前进混合指令却可以稳定。这说明覆盖与协调具有明显命令依赖性。

SM九指令均未跌倒，.5指令仍欠速；这是一组名义场景结果，未测试速度突变、扰动、斜坡、台阶、延迟与真实硬件。单训练seed不能给出算法显著性结论。

## 7. 旧步态权重的真实训练链

旧目录名字中的2p5h只是该段计划预算，不是完整模型训练成本。原始TensorBoard `time/hours`与checkpoint初始化来源给出：

1. WM来源：it1800，.251819h；后续只导入WM，来源Actor没有导入，因此不计入Actor权重链。
2. 基础Actor：it17000，2.004372h；从预训练WM开始，Actor/Critic新初始化，最初不含新增步态塑形。
3. 对角/足滑续训：完整恢复Actor、Critic、WM、optimizer和课程；Replay重建。


| 模型 | Actor链h | 完整链h | vx m/s | 足滑RMS m/s | 对角支撑 | FL/FR/RL/RR峰高cm |
|---|---|---|---|---|---|---|
| 新从零 | 2.50112 | 2.50112 | 0.2810 | 0.4795 | 0.00% | 0.42/0.66/6.34/0.53 |
| 旧it22000 | 2.69972 | 2.95154 | 0.2769 | 0.4114 | 0.00% | 0.80/0.31/2.91/2.62 |
| 旧it24000 | 2.96039 | 3.21221 | 0.3053 | 0.4367 | 0.00% | 0.59/0.58/3.16/3.21 |
| 旧it34000 | 4.29410 | 4.54592 | 0.2895 | 0.1674 | 48.57% | 5.14/3.44/3.47/5.04 |



it34600才人工停止，完整实际消耗约4.62836h，但最后600步未保存，不能算作it34000权重所学时间。旧最终保存Actor链4.29410h，连WM来源完整4.54592h。

匹配复测使用同seed2026、各命令16episode、10s、无噪声，比较新2.50h和旧完整2.95h、旧Actor2.96h、旧完整4.55h；不是拿旧不同口径评测值混表。低速.3m/s：新模型足滑.4795m/s，对角支撑0%；旧约3h也仍拖地；旧4.55h足滑.1674，对角支撑48.6%，四足峰值约5.14/3.44/3.47/5.04cm。纯yaw下旧最终并非所有指标都改善，所以不能宣称全命令步态最优。

旧seed0、新seed2026；旧有WM预训练、分阶段reward、恢复课程，且旧想象obs_noise=true、新false。额外预算与训练路径均为混杂因素。**没有证据证明“暂停/续训”本身提升学习**；要验证需同总预算、同seed、同初始化、同reward schedule下连续与恢复两支配对实验。

## 8. 阶段结论与下一轮最有价值的实验

这轮支持：SM能在Go2这类12D位置控制和稠密速度目标上工作，当前等时间结果的命令覆盖优于本轮PPO；速度成功与高质量Trot是两个目标。较好的旧对角步态需要更多训练和历史塑形路径，不能用于宣称等时长获胜。

优先补多训练seed、固定课程和等真实样本预算对照；保持不同公平性问题分别作图。步态改进应同时约束足滑、足高、频率/对称和速度，避免只放大一个奖励。先在正常姿态/起步/接触切换上校验WM动作响应，再判断是否需要新增时序步态信号。本任务没有实施这些修改。
