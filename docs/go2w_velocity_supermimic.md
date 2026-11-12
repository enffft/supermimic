# Go2WVelocitySuperMimic：开发交付与后续验证

本任务为代码实现。此次没有启动训练、MuJoCo/MjWarp、CUDA、torch.compile，
没有加载 checkpoint、安装依赖或运行 pytest。所有控制、奖励数值均是初始参数，
不代表动力学、性能或实机验证。现有训练进程、日志与数据没有被操作。

## 架构审计与兼容边界

正式实现为 `src/unisupermimic`，没有使用 migration 中的旧训练器。
已检查 networks/geometry/learner/replay/driver、MDP protocol、Go2 的 config、
contract、MDP、reward、manager、sampler、curriculum、speed_curriculum、terrain、
heightfield、env、registry、checkpoint 及当前 Hydra owners。

- 原 State 为六元组 `(p,R,v_world,w_world,q,qd)`；序列化 R 为 wxyz quaternion。
  原 unflatten 根据总维度平分 q/qd，不能解析 Go2W 的不同 q/qd 宽度。
- 原 WM 默认输入 `13+3*nj+context+terrain`、输出 `12+2*nj`；Replay 的
  s/s1、a/a_prev、err_floor 和所有 learner 解包点也有相同的关节宽度假设。
  现在通过可选 StateLayout 统一计算，Go2/G1 默认回退到原等宽布局。
- 原 direct Policy 限定 5×42→12，按观测项组织历史，固定缩放并用 SiLU/tanh。
  现在 DirectPolicySpec 定义分组、缩放、历史、命令和动作宽度；Go2 默认 spec
  保持原数值、初始化顺序及 state_dict。G1 encoded 分支不变。
- 原 Go2 Action 使用具名 Entity 位置目标 API。UniLab 的 velocity target API
  只是写 actuator ctrl，没有把 torque motor 变成速度伺服器。因此 Go2W
  独立实现每个物理子步反馈的 effort Action；没有改原 Go2 Action。
- 原 H8 WM rollout 与 H32 imagination 均调用 `_wm_step(state,...)`，随后重新
  `_terrain_scan(state)`。Go2W 沿用这一路径，没有缓存首帧高程或读取真实未来位姿。
- Replay 保留执行动作、前动作、命令、context、episode 边界及地形/命令元数据。
  Go2W 不使用未实现的随机化，context 宽度为 0；Actor 从不接收该通道。
- 复用 Go2VelocityEnv 中不依赖关节宽度的 transition/reset 快照流程；Go2W
  自己拥有模型、MDP、action、command、reward、factory 和 checkpoint contract。
- 没有修改原 Go2/G1 配置、地形分布或模型，没有自动跨机器人参数迁移。
  本次开始时已有的 Go2 楼梯修改保持在工作区，未纳入此次提交。

## 模型来源和物理映射

来源：[Unitree 官方 MuJoCo Go2W](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/go2w)。
固定版本 `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`。
BSD-3-Clause 原文保存在 assets/robots/go2w/LICENSE；源码与网格 SHA256、
改动说明及轮网格范围保存在 provenance.json。

保留官方 Go2W 的独立轮体（每轮 0.98 kg）、惯量、关节轴和碰撞网格，
并保留官方腿部结构与惯量。没有用普通 Go2 加圆柱轮来替代模型。
删除非必要的机身装饰网格；官方每轮有两个重叠 mesh geom，
其中展示 geom 改为无碰撞/无质量，保留 foot-class mesh 的实际碰撞与材质参数。
这项处理和模型整体动力学尚未通过仿真验证。

轮半径约 0.086 m，网格轴沿 Y。左轮保留官方 X 轴 180° 的几何变换，
关节轴四轮均为 `0 1 0`、连续旋转。15 rad/s 对应理想轮缘速度约 1.29 m/s。
没有把绝对轮角置零或从模拟器中删除。

**逻辑顺序（Actor/WM qd/控制/元数据）：**

1. FL hip, thigh, calf；FR hip, thigh, calf；RL hip, thigh, calf；RR hip, thigh, calf。
2. FL wheel, FR wheel, RL wheel, RR wheel。

**实际 MJCF qpos/qvel：** FL leg3+wheel、FR leg3+wheel、RL leg3+wheel、RR leg3+wheel。
**实际 motor：** FR leg3、FL leg3、RR leg3、RL leg3、FR/FL/RR/RL wheel，
保留官方 actuator 顺序；每个 actuator 改为与目标 joint 同名，gear 显式为 1。
所有读取索引和控制映射均按名字解析，未假设 qpos 或 actuator 连续排列。
XML-only 审计拒绝重复/缺失 motor、错误轴、错误 gear、错误力矩范围及改变的轮网格。

## 控制与纯本体 Actor

控制周期 .02 s，物理周期 .005 s，每个子步重新计算：

```
a = clamp(a, -1, 1)
q_target = clamp(default_leg_q + .25 * a_leg, leg_soft_limits)
tau_leg = clamp(40 * (q_target - q_leg) - 1 * qd_leg, effort_leg)
qd_target_wheel = 15 * a_wheel
tau_wheel = clamp(.5 * (qd_target_wheel - qd_wheel), effort_wheel)
```

腿部限幅遵循该官方 MJCF：hip/thigh 23.7 Nm，calf 45.43 Nm；轮子 15 Nm。
全部参数位于 `env.actions.mixed.control`，可降低限幅，不能超过官方模型范围。
前后 thigh 位置限位不同，soft limits 均取真实范围的 90%。
无控制延迟；collector 可添加动作噪声，裁剪后动作进入控制、Replay、奖励及下一帧历史。
reset 后上一执行动作与步号清零，不把末步动作写入新 episode 历史。

单帧顺序为腿位置偏差12、腿速度12、轮速4、IMU 角速度3、projected gravity3、
上一实际执行动作16，共 50D。固定缩放分别为 1/.2/.05/.5/1/1。
5 帧按项组织历史，再单独拼接命令3：输入 253D、输出16D tanh。
无真实线速度、高程、地形标签、接触真值、视觉或 velocity encoder。
仿真中 IMU 角速度和重力由机身姿态/速度投影构造，尚未验证真实 IMU 标定/噪声。

## World Model 与地形

`go2w_compact_state_v1`：p3、quat4、v3、w3、q_leg12、qd_leg12、qd_wheel4，共41。
增量：dp3、dR3、dv3、dw3、dq_leg12、dqd_leg12、dqd_wheel4，共40。
没有轮子绝对角度输入、输出或损失；物理模拟器仍使用完整23D qpos/22D qvel。

WM 的 invariant_features 为高度1、heading-relative R6、body v3/w3、q12/qd16，
共41；再加动作16、context0、高程77，输入134，输出40。
state_error、normalizer、err_floor、Replay s/s1 和 learner 所有解包点使用同一布局。
结构化积分保留原 `_drift()` 的旧/新速度 .375/.625 加权，根位置在 yaw 局部系
计算漂移后回到世界系，姿态/速度增量在机身系变换。关节漂移只选择前12个腿速度，
四个轮速仍由 WM 预测。学习的位置残差与32步 BPTT 链路保留。

沿用世界坐标下的不可变 collision-aligned HeightField，11×7 yaw 对齐扫描。
每次 `_wm_step` 使用当前预测 p/R 重新采样；H8 与 H32 共用该入口。
`sensor_grad=true` 不 detach；heightfield 用 Torch 分片线性三角插值。
reset 使用同一编译地图的世界坐标 origin，所有环境共享地图但通过各自世界 XY 查询。
越出地图沿用 contains 检查及终止；高度惩罚/跌倒阈值使用预测位置处的相对地形高度。
本次未初始化模型，因此地图绑定、子步读取和 reset 的端到端行为均未运行验证。

## Reward shaping

独立 reward_core 将数学定义供真实 transition 和 imagination 共用，最终只乘一次 dt。
数学设计参考下列实现，未复制外部 reward manager 或接触依赖逻辑：

- [LocoWheeledLegged](https://github.com/zhaozijie2022/LocoWheeledLegged)，
  Go2W locomotion_env_cfg.py blob `591cc1d241ffb4d58bd34159e558198f883bdfa8`：
  腿/轮加速度分开、动作变化、力矩约束、停止命令。
- [DreamWaQ_Go2W](https://github.com/ShengqianChen/DreamWaQ_Go2W)，
  go2w_config.py blob `bad67922a3908a209860745a89ccc4c6d29a54b8`、
  go2w_robot.py blob `c0201db7464b01c84c8a80812882f8cb1002d149`：
  腿位置/轮速度混合 PD、轮位置排除、指数速度跟踪。
- Uni 当前 Go2 的零速度 baseline、地形相对高度和软终止。

| 项目 | 初始权重/尺度 |
|---|---|
| vx/vy/yaw tracking | 2.5/1.5/2.0，sigma=.25/.25/.30 |
| 静止机身运动奖励 | .5，命令平滑门控尺度 .05，xy速度尺度 .15 |
| upright | .4，projected-gravity xy 指数尺度 .4 |
| 世界竖直速度/机身xy角速度 | -.5/-.08 |
| 地形相对低高度 | -5，仅惩罚低于 .22m |
| 腿默认位置偏差/软限位 | -.03/-1 |
| 腿/轮动作变化 | -.01/-.003 |
| 腿/轮力矩平方 | -2e-5/-1e-5 |
| 腿/轮控制周期速度差分加速度平方 | -2.5e-7/-2.5e-9 |

三轴先转换到机身坐标系，tracking 使用 `exp(-error²/sigma²)`。
每轴减去站立 baseline，并在该轴零命令附近平滑恢复其最大静止奖励；
额外 stand_still 用**完整三维命令幅值**门控，避免非零命令奖励停下。
固定命令下 baseline/恢复项与状态无关，不改变 tracking 本身的速度最优点。
默认正奖励率上界为6.9，`max_step_reward=.138`。无 rolling/slip/engagement、
接触或强制步态奖励。upright 与软跌倒阈值沿用 Go2，斜坡 pitch 的实际影响待验证。

真实奖励读取最后物理子步的 noiseless jointactuatorfrc；想象用预测 endpoint
重新计算同一饱和 PD。两者采用同一力矩定义、gear 和限幅，但**时序不完全一致**：
真实力矩由末次积分前状态生成，proxy 用积分后状态，无子步/actuator 状态可用。
因此不能声称真实/想象力矩数值完全相同；需要后续采集比较。
其余状态/动作/加速度/高度奖励共用同一函数，不依赖真实未来状态或接触。

## 地形与课程配置

所有阶段使用同一生成器、模型和地图，不另设训练器。

| 配置 | 阶段与领域比例（flat/rough/up/down/stairs-up/stairs-down） |
|---|---|
| go2w_velocity/flat | A：1/0/0/0/0/0；低速五帧盲控 |
| go2w_velocity/mixed | B：.50/.20/.15/.15/0/0；平地+粗糙+坡面 |
| go2w_velocity/terrain 或 curriculum | C：.35/.20/.10/.10/.125/.125 |

row0 完全平地；level1..4 粗糙幅度1..3cm、坡度.05..20、台阶2..5cm，踏面.40m。
不同地形方向继承原实际几何和 approach spawn；下坡/下楼梯用 inverted 地形。
flat 领域始终位于 row0 中心；非平地领域从入口开始。
复用固定领域配额、六种 flat command mode、episode 成功/失败统计、Replay 起点
mode 权重与 TerrainCurriculum。非平地推广依据前进通过、跟踪误差和无跌倒的
连续成功（默认3次），失败回退，并保留15%较低难度回收。
A 阶段速度三级范围通过 `sampler.set_speed_level()` 显式调整，默认最低级。
不会基于训练时间自动宣称通过，也不默认扩大到20cm台阶。

| 领域 | vx | vy | wz |
|---|---|---|---|
| flat（C） | [-.25,.60] | [-.15,.15] | [-.35,.35] |
| rough | [-.20,.70] | [-.15,.15] | [-.35,.35] |
| slope_up/down | [.10,.70] | [-.10,.10] | [-.25,.25] |
| stairs_up/down | [.15,.60] | [-.06,.06] | [-.15,.15] |

范围是一般/组合命令包络；sampler 另保留停止及纯轴命令，因此纯横移/纯转向的
vx 可以为0。困难楼梯沿用原 sampler 的.65横移/转向缩放，默认80%向前命令。
普通轮不具备麦克纳姆式横向滚动能力，横移必须依靠腿协调，未增加人为强制奖励。

A/B/C 切换只改 command sampler 分布，普通 resume 仍严格拒绝变更。
需要未来显式调用现有 `learner.load(path, allow_command_distribution_change=True)`，
然后 `learner.train()`；该 API 持续使用同一训练器，保留旧 Replay 与真实命令标签，
但保留新阶段固定领域配额，对新分配环境的 mastery 清零。当前通用 CLI 的
`algo.resume=true` 是精确协议恢复，不自动授权课程分布变更。
这条阶段恢复路径也未执行验证。新日志目录应单独选择，不覆盖旧 run。

## Checkpoint

独立 task/schema 与 state layout 版本；记录逻辑/实际关节及 actuator 顺序、
轮关节名、模型/轮网格 SHA256、混合控制协议和全部参数、reward、terrain、
heightfield fingerprint、观测分组/尺度、时步、terminations、events。
共享 learner 进一步严格检查 direct/history、网络宽度、wm_structured、
terrain_training/conditioning/sensor_grad。12D Go2 checkpoint 无法匹配 Go2W contract；
structured 不匹配会明确拒绝。原 Go2 actor 导入/flat-WM扩展路径拒绝 Go2W，
没有实现未经验证的跨机器人转换。Go2W 的 v1 learner 要求 direct/history5、
structured WM 和可微 terrain conditioning；默认 H8/H32。

## 本次检查和未验证内容

执行的验证仅为 diff 空白检查、supermimic conda Python 的 compileall、git status，
以及极小的 Python AST/YAML/XML 静态引用检查。新增 tensor 单元测试未执行。
确切结果见此次最终交付报告，不包含仿真/训练成功声明。

仍待验证：MuJoCo编译模型与惯量/碰撞、公开 API 的子步状态刷新与具名映射、
速度方向与电机饱和、reset/地形坐标、H8/H32闭环梯度与精度、课程恢复行为、
盲走复杂地形通过率、奖励权重和实机安全参数。

## 下一次允许启动时的建议（本次不执行）

1. 先执行新增 CPU tensor tests，再单环境无学习验证：检查16个具名驱动器、
   qpos/qd映射、轮轴正负方向、leg/wheel限幅及每子步状态刷新。
2. 检查 reset 在六领域与所有level的接触/机身高度；确认轮mesh无重复碰撞，
   前进轮速符号正确，地形扫描与碰撞高度匹配。
3. 小批真实 transition 比较真实力矩与 endpoint proxy，分别统计腿/轮误差和
   饱和比例；检查 Replay a/a_prev 及 reset 后观测历史，无动作噪声协议错位。
4. 先在A验证停止、前后、转向，观察三轴body速度RMSE、奖励分项、跌倒率、
   相对高度、wheel target/actual rad/s、力矩饱和、轮速加速度及动作变化。
5. 比较 held-out H1/H8/H32 根位置/姿态/body速度/腿q与qd/轮速误差；在斜平面
   检查每步scan随预测XY/yaw变化，排查 world/body混用、轮排序、yaw与地图偏移。
   对比 rollout_scan 与真实同位姿scan区分位置漂移和地图误差。
6. 确认真实/想象奖励及有限差分梯度方向合理后，再扩大B/C覆盖与训练时长；
   阶段推进依据真实通过与held-out WM指标，不能仅看想象reward上涨。
   未来资源许可后可覆盖 `algo.num_envs=128`、buffer_steps=1000，批量/网络沿用
   当前配置能力；先评估 MuJoCo CPU负载，再决定长期运行规模，不改全局默认。
