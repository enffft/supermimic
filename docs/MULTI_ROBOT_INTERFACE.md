# SuperMimic 多机器人接口说明

本文根据当前工作区源码说明 G1、Go2、M1-Piper 如何使用同一套采集、World Model 和 imagination/BPTT 训练链路。Go2 当前 velocity reward 已接入并用于一次 15 分钟有界短训练；结果见[验收记录](MULTI_ROBOT_VALIDATION.md#go2-15-分钟短训练)。

## 1. 代码结构与接入路径

~~~mermaid
flowchart LR
  C[TrainConfig] --> T[training/Trainer]
  T --> E[make_env / mjlab managers]
  E --> A[Action term]
  E --> O[policy + goal observations]
  E --> R[ReplayRecorder]
  R --> B[ReplayBuffer]
  B --> W[WorldModelTrainer: 8-step]
  B --> P[PolicyTrainer: starts]
  P --> I[imagine: 32-step]
  I --> D[Policy / Critic]
  S[RobotSpec + MuJoCo model] --> E
  S --> W
~~~

| 职责 | 源码路径、函数 | 做什么 |
|---|---|---|
| 配置 | [config.py](../src/supermimic/config.py#L14) EnvCfg、ScheduleCfg | robot/task、环境数、历史长度、UTD、更新日程 |
| 机器人契约 | [robots/spec.py](../src/supermimic/robots/spec.py#L9) RobotSpec、get_robot_spec() | policy 关节顺序、默认姿态、控制模式/scale、模型路径、tracked bodies |
| 仿真模型/执行器 | [robots/mjlab_assets.py](../src/supermimic/robots/mjlab_assets.py#L64) load_robot_spec_model()、get_robot_entity_cfg() | 编译模型并按名称校验关节，生成 mjlab EntityCfg |
| 环境分发 | [envs/tracking_env_cfg.py](../src/supermimic/envs/tracking_env_cfg.py#L97) make_env_cfg()、make_env() | G1 motion tracking 保留原环境；其他当前任务分发到多机器人环境 |
| 新环境拼装 | [envs/multirobot_env_cfg.py](../src/supermimic/envs/multirobot_env_cfg.py#L37) make_multirobot_env_cfg() | observation/action/command/reward/termination/DR/recorder |
| 训练接线 | [training/loop.py](../src/supermimic/training/loop.py#L30) Trainer.__init__() | 按 robot/task 设置维数，构造 env、FK heads、WM、Actor、Critic、task |
| 在线数据 | [data/recorder.py](../src/supermimic/data/recorder.py#L29) ReplayRecorder | 将真实 transition 和终态写入 ReplayBuffer |
| 任务语义 | [imagination/tasks.py](../src/supermimic/imagination/tasks.py#L12) MotionTrackingTask、VelocityCommandTask | 让 motion reference 与固定速度命令共用 imagination 代码 |

主循环仍按 Collect → World Model update → Policy/Critic update 执行，见 [training/loop.py](../src/supermimic/training/loop.py#L216)。

## 2. Robot State、Observation、Goal 和 Action 尺寸

状态由浮动基座 MuJoCo qpos/qvel 拼接：

~~~text
state = [root_pos(3), root_quat_wxyz(4), joint_pos(n),
         root_lin_vel_world(3), root_ang_vel_body(3), joint_vel(n)]
state_dim = 13 + 2*n
delta_dim = 12 + 2*n
one_frame_obs = joint_pos_error(n) + joint_vel(n) + base_ang_vel(3)
                + projected_gravity(3) + previous_action(n)
obs_dim = 3*n + 6; Actor history = 10 frames
~~~

| Robot | 编译模型 nq/nv | 关节数 | State / delta | Action | 单帧 Obs / 10 帧 Obs | Goal |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 36 / 35 | 29 | 71 / 70 | 29 | 93 / 930 | 10×68=680 |
| Go2 | 19 / 18 | 12 | 37 / 36 | 12 | 42 / 420 | 机体坐标 [vx,vy,yaw_rate]，3 |
| M1-Piper | 29 / 28 | 22 | 57 / 56 | 22 | 72 / 720 | 默认 10×46+9=469 |

尺寸属性在 [robots/spec.py](../src/supermimic/robots/spec.py#L46)。M1 在 Trainer 默认启用 9 维 anchor observation；直接构造环境且 anchor=false 时 Goal 是 460 维。定义在 [training/loop.py](../src/supermimic/training/loop.py#L42)。

M1 的单帧 observation 维数仍按 22 个关节计算，但 wheel joint position error 被 mask；wheel velocity 和上一时刻 wheel action 仍是 policy 输入。该规则在环境观测组通过 joint_pos_rel_masked 配置（[envs/multirobot_env_cfg.py](../src/supermimic/envs/multirobot_env_cfg.py#L50)）。

### G1

- 保持原 G1 asset、Motion Library、环境 reward 和 reference termination 路径；分发逻辑见 [envs/tracking_env_cfg.py](../src/supermimic/envs/tracking_env_cfg.py#L97)。
- 维持 29 关节、state 71、xi 33、Goal 680 的既有尺寸。

### Go2

- 模型：/home/yifeiy/ws_parkour/lidar_locomotion/legged_gym/resources/robots/go2/urdf/go2.urdf。关节顺序为 FL、FR、RL、RR，每条腿依次 hip/thigh/calf；默认角度依次为 (0.1,0.8,-1.5),(-0.1,0.8,-1.5),(0.1,1.0,-1.5),(-0.1,1.0,-1.5)，root 高度 0.42 m（[robots/spec.py](../src/supermimic/robots/spec.py#L67)、[robots/spec.py](../src/supermimic/robots/spec.py#L127)）。
- 12 路 position action，scale 0.25；Kp=40、Kd=1、effort limit=23.5。控制 dt=0.02 s，由 4 个 0.005 s physics substep 组成（[robots/mjlab_assets.py](../src/supermimic/robots/mjlab_assets.py#L117)、[envs/multirobot_env_cfg.py](../src/supermimic/envs/multirobot_env_cfg.py#L179)）。
- 训练时 low-height termination 为 `root_z < 0.17 m`，tilt 为 `abs(1+g_local_z) > 1-cos(70°)`。16 env 零动作 8 秒测试的稳态 root z 是 `0.28271 m`；四足均接触、无腿部碰撞和力矩饱和。此结果验证静止支撑，不代表已测准真实跌倒边界（详见 [验收记录](MULTI_ROBOT_VALIDATION.md#p0-cgo2-站立与真实-termination-阈值)）。
- Goal 为 body-frame 速度命令 `(vx,vy,yaw_rate)`，reset 时采样，并在固定的 10 秒 resample 周期内保持不变。当前配置默认启用 A/B/C command curriculum，从 A 开始；A 主要采样 `vx=[0.15,0.55] m/s`，15% 为静止命令，B/C 的范围和保留比例可在 `EnvCfg` 配置（[config.py](../src/supermimic/config.py#L46)、[envs/mdp/commands.py](../src/supermimic/envs/mdp/commands.py#L535)）。该课程逻辑在当前验收中没有做长程晋级验证。Go2 不使用 Motion Library、LiDAR 或地形输入。
- Go2 使用 11 项 velocity reward core，并通过 `RewardManager` 使用 `weight=1`、再由 manager 乘控制 `dt`；imagination 使用同一个计算模块，但 MuJoCo measured torque/joint acceleration 缺失时会用可微 proxy（[tasks/velocity_reward.py](../src/supermimic/tasks/velocity_reward.py#L78)、[envs/multirobot_env_cfg.py](../src/supermimic/envs/multirobot_env_cfg.py#L169)）。这套当前 reward 已用于 3072-env、15 分钟短训练；该运行记录工程链路和短期行为，不代表长程收敛或课程晋级验收。

### M1-Piper

- 模型：/home/yifeiy/AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf。策略顺序为 FAR、FBL、RAR、RBL 四条腿各 ABAD/HIP/KNEE/FOOT，然后 piper_x_joint1..6（[robots/spec.py](../src/supermimic/robots/spec.py#L75)、[robots/spec.py](../src/supermimic/robots/spec.py#L138)）。
- 混合 action 同序 22 维：12 个非 FOOT 腿位置控制，4 个 FOOT 轮速控制，6 个 Piper 位置控制。leg Kp/Kd=250/9，轮 damping=0.5，arm Kp/Kd=32/1；position scale=0.25，wheel velocity scale=5.0（[robots/spec.py](../src/supermimic/robots/spec.py#L140)、[robots/mjlab_assets.py](../src/supermimic/robots/mjlab_assets.py#L133)、[envs/actions.py](../src/supermimic/envs/actions.py#L59)）。
- Full Dynamics State 是 22 关节，但 motion source 只有 18 个 AMP 关节有效参考（12 腿关节+6 arm）。每帧 Goal 为 18 关节位置误差与速度（36）+ gravity（3）+ body angular velocity（3）+ heading-frame linear velocity（3）+ height（1）=46；10 帧 460，默认再拼 anchor pose 的 pos3+rot6d6 得 469。M1 reference 不包含可普遍使用的 wheel 角度/速度目标。
- TCP tracked body 是 piper_x_link7。模型加载保留 fixed link；MuJoCo 不解 DAE，加载时将有 STL 对应的 mesh 指向 STL，无 STL 对应的固定夹爪碰撞 mesh 被省略，供应商原 URDF 保持只读（[robots/mjlab_assets.py](../src/supermimic/robots/mjlab_assets.py#L23)）。

## 3. 环境控制一步与数据采集

一个 Trainer iteration 是一个向量化控制步：

1. Trainer.act() 将 (E, history_obs) 和 (E, goal) 输入 Actor，得到 (E,A) clean action。
2. env.step(action) 的 action term 处理噪声/clip，并转换为关节 actuator target。M1 的 MixedM1Action 按 joint name 拆位置目标和轮速目标；Go2 使用 position action。
3. mjlab 运行 4 个 MuJoCo substeps（dt=0.005 s），完成 0.02 s 控制步；先算 termination/reward。auto-reset 后调用 sim.forward，再推进命令/reference 并计算新 observation。
4. Recorder 将 `(s_t,a_clean,a_exec,xi,r,done_next,frame/clip/goal)` 写进动作源状态行；真正 terminated 时再追加没有 outgoing action 的 terminal successor state。Timeout/clip-end 也会结束 episode，但 `done_next` 只标真实 termination，不把 truncation 当跌倒样本。
5. 下一步 observation 进入下一轮；WM update 的 `sample_windows()` 从 transition 源行读取 `done_next`，使标签对应 `s_{t+1}`，不提前一帧。
6. warmup 后由 UTD 累加器决定是否执行 World Model 和 Actor/Critic 更新。

StateIO.read() 返回 (E,13+2n) 并扣除每个 env 的 world origin；SimContext.read() 返回 xi=[foot friction(1), encoder bias(n), COM offset xyz(3)]，shape (E,n+4)（[envs/state_io.py](../src/supermimic/envs/state_io.py#L12)、[envs/sim_context.py](../src/supermimic/envs/sim_context.py#L11)）。

clean action 和 executed action 分开存：后者是噪声/clip 后的归一化 action，WM 用它学习实际执行的转移；不是物理单位下的 target。M1 action adapter 具体拆分和写入目标见 [envs/actions.py](../src/supermimic/envs/actions.py#L93)。

## 4. Replay Buffer 与 Motion Library

ReplayBuffer 是每个并行 env 自己的环形在线经验，shape 前缀 (E,T)，其中 T=capacity（[data/replay_buffer.py](../src/supermimic/data/replay_buffer.py#L40)）。

| 张量 | Shape | 作用 |
|---|---|---|
| s | (E,T,S) | 完整仿真状态 |
| a,a_exec | (E,T,A) | clean 与 noisy/clipped executed action |
| xi | (E,T,X) | dynamics 上下文 |
| r,done | (E,T) | 真实 reward 与本行 action 是否导致真实 termination 的 `done_next` |
| frame,clip,ep_id | (E,T) | motion 索引和 episode 连续性 |
| terminal,ep_start,valid | (E,T) bool | 终态、episode 起点、有效槽位 |
| goal | 可选 (E,T,G) | 保存 reference-free 命令任务快照 |
| ptr,ep_counter | (E,) | 每个 env 的写指针/episode 计数 |

真实 termination 时 Recorder 先写最后一次 action transition，再追加无出边的 terminal successor state；timeout/clip-end 只追加 terminal-state row，不把 `done_next` 置正。8-step window 可以以 terminal state 作为最后 target，但不会穿过 episode 边界（[data/recorder.py](../src/supermimic/data/recorder.py#L80)）。`sample_windows()` 的 done 标签现在从 source action rows 读取，与预测的 `states[1:]` 对齐（[data/replay_buffer.py](../src/supermimic/data/replay_buffer.py#L130)）。Go2 将 3 维 command 存进 Replay。Motion Goal 可根据 clip/frame 恢复，因而 G1/M1 的 replay goal 不分配。

MotionLibrary 则是静态离线参考库：按全局 frame 查 root/joint state、tracked-body FK、future Goal 和 clip/bin 边界，用于 reset、tracking 和 return-based sampling（[envs/mdp/motion_library.py](../src/supermimic/envs/mdp/motion_library.py#L81)）。它表示“目标运动”，Replay 表示“仿真器实际经历”。Go2 无 MotionLibrary。

M1 转换入口 [data/m1_motion_convert.py](../src/supermimic/data/m1_motion_convert.py#L45)，输出 data/npz/m1_piper/*.npz。转换器做 xyzw→wxyz、body-frame root linear velocity→world、名称映射、50 Hz 重采样和 FK；保存 joint_valid/wheel_reference_valid、category、源文件和 hash。没有有效 wheel sidecar 时 mask=false，不把推断 wheel 值作为目标。

`joint_valid` / `wheel_reference_valid` 的用途要区分：MotionLibrary 会把 NPZ joint state 按 `joint_names` 重排，并检查选入 Goal 的 `goal_joint_names` 全部有效；它不会用 `wheel_reference_valid` 从实时 tracking reward 中逐帧剔除 wheel joint。M1 当前 Goal 明确只使用 18 个 AMP 非轮关节，四轮仍在完整 22 维动力学状态和 22 维 action 中。不要据此声称有 wheel angle tracking reward mask。

## 5. 连续 8-step Window 和 Imagination Starts

### sample_windows(n,k)

window_mask(k) 只接受 `t..t+k` 均有效且属于同一 episode 的窗口；terminal state 只能出现在最后 target，不允许窗口跨到 reset 后的 episode；起点还需避开即将被环形写指针覆盖的区域。合格窗口起点有放回抽样，再取连续 offset `0..k`（[data/replay_buffer.py](../src/supermimic/data/replay_buffer.py#L103)、[data/replay_buffer.py](../src/supermimic/data/replay_buffer.py#L130)）。

- states: (k+1,n,S)，起始状态加 k 个真实后续状态。
- a_exec、a_clean、r、frame: (k,n,...)，与 k 次转移对齐。
- done: (k,n)，由每步 source action row 的 `done_next` 组成；`done[j,n]` 是 `a_exec[j,n]` 是否令 `states[j+1,n]` 终止。
- xi: (n,X)，同一窗口内共用。

默认 k=8。sample_starts(n,hist) 从有效非终止行抽 s_t，取过去 hist 个同 episode state；若 episode 刚开始则用第一行 backfill，起点以前 action 置零。返回 state (n,S)、hist state/action (hist,n,S/A)、xi/frame、prev clean action 和可选 Goal（[data/replay_buffer.py](../src/supermimic/data/replay_buffer.py#L145)）。

### 32-step Imagination

imagine() 用 start state/history 初始化 Actor history。每一步重新 flatten 当前预测 history，调用 Actor 得新 action，再将 action 送入 World Model 得到新 state，并产生新的 observation。Motion reference 按 frame 推进；Go2 的 replay command 在 H 步保持不变（[imagination/rollout.py](../src/supermimic/imagination/rollout.py#L71)、[imagination/rollout.py](../src/supermimic/imagination/rollout.py#L131)）。

## 6. World Model 和目标函数实现

### State Canonicalization、Dynamics、积分

StateSpec.canonical(s) 归一化 quaternion；wheel joint angle wrap 到 [-π,π]；从输入去掉绝对 root x/y/yaw，只输入 root z、heading-relative root orientation rot6d、heading-frame linear velocity、body angular velocity、joint position/velocity，再按 in_scale 归一化（[world_model/state_spec.py](../src/supermimic/world_model/state_spec.py#L94)）。

DynamicsMLP 拼接 [canonical(s), a_exec, normalize_xi(xi)]，默认三层 (2048,2048,2048)，输出 delta_dim；最后线性层零初始化。积分将输出乘 out_scale 后拆为 [dv_heading(3),dw_body(3),dqd(n),dp_heading(3),dtheta_body(3),dq(n)]：先更新速度，再以 dt=0.02 显式积分 root position/quaternion 和 joint position；quaternion 每步归一化（[world_model/dynamics.py](../src/supermimic/world_model/dynamics.py#L31)、[world_model/state_spec.py](../src/supermimic/world_model/state_spec.py#L117)、[world_model/world_model.py](../src/supermimic/world_model/world_model.py#L63)）。

### World Model 8-step loss

WorldModelTrainer.update() 抽真实 replay 的 8-step window，使用 a_exec。window_loss() 从 states[0] 开始 autoregressive rollout：预测 s_next 对真实 states[j+1] 算归一化误差，再将预测状态送进下一步，而非用真值重置。每步状态误差平方后对状态维求和、batch 求均值，累计后除以 k；termination head 对相同下一状态的 done[j] 算 BCE，累计后除以 k。总损失是 state loss 加权 termination BCE（[world_model/world_model.py](../src/supermimic/world_model/world_model.py#L81)、[world_model/world_model.py](../src/supermimic/world_model/world_model.py#L96)）。

loss.backward() 沿 8 步展开图反传到 dynamics 和 termination head，然后 all-reduce、gradient clipping、Adam 更新（[world_model/wm_trainer.py](../src/supermimic/world_model/wm_trainer.py#L56)）。Motion Library 为 None 的 Go2 仍使用 reference-free base-height/upright termination features。

### Observation、Reward、Termination

- analytic_observation() 由预测 state、前一 action、encoder bias 和 default pose 计算 joint-pos error、joint velocity、base angular velocity、gravity 和 previous action，shape (N,3n+6)（[world_model/heads.py](../src/supermimic/world_model/heads.py#L30)）。
- Motion reward 经可微 FK 得到 tracked body transforms/velocities，再调用 `tracking_reward_terms()`；Go2 当前有首版 `VelocityCommandTask.reward()`，imagined path 使用 PD torque/finite-difference acceleration proxy，但它尚未成为团队确认的正式 blind-locomotion objective（[world_model/heads.py](../src/supermimic/world_model/heads.py#L109)、[imagination/tasks.py](../src/supermimic/imagination/tasks.py#L46)、[tasks/velocity_reward.py](../src/supermimic/tasks/velocity_reward.py#L78)）。
- TerminationHead 接收两个 feature 并输出一个 logit。Motion task feature 是 root 高度/tilt 相对 reference 的误差；Go2 feature 是 base_height-z 和 upright tilt。sigmoid(logit) 是预测终止概率（[world_model/heads.py](../src/supermimic/world_model/heads.py#L142)、[world_model/heads.py](../src/supermimic/world_model/heads.py#L151)、[world_model/world_model.py](../src/supermimic/world_model/world_model.py#L69)）。真实 done 标签来自环境 termination function（[envs/mdp/terminations.py](../src/supermimic/envs/mdp/terminations.py#L16)、[envs/mdp/terminations.py](../src/supermimic/envs/mdp/terminations.py#L34)）。

### Soft survival、Eq. (2)、Eq. (3) 和冻结梯度

Rollout 初始化 c_0=1，逐步用 c_{t+1}=c_t(1-sigmoid(logit_{t+1})) 更新 soft survival。Eq. (2) 中 r_{t+1} 乘的是更新前的 c_t；对状态 s_{t+1} 的终止预测会降低 c_{t+1}，从下一项 reward 开始降低后续奖励权重，也降低末端 bootstrap 权重。它不是硬阈值截断，因此 survival 路径可微。

policy_objective() 计算 batch mean of sum_t gamma^t c_t r_{t+1}，可选加 gamma^H c_H V_H。PolicyTrainer 最小化 -J/H，对应论文 Eq. (2) 的具体实现（[imagination/objective.py](../src/supermimic/imagination/objective.py#L10)、[imagination/policy_trainer.py](../src/supermimic/imagination/policy_trainer.py#L76)）。

Critic 输入 canonical state、Goal、normalized xi、4 维 root error，输出 scalar。lambda return 从 R_H=V_H 反向递推，乘 gamma(1-dhat) 并混合 (1-lambda)V_{t+1} 与 lambda R_{t+1}，对应 Eq. (3)（[imagination/objective.py](../src/supermimic/imagination/objective.py#L24)、[policy/nets.py](../src/supermimic/policy/nets.py#L93)）。

Actor update 期间 WM 和 Critic 参数 requires_grad=False，但 forward 不包 no_grad()：权重不更新，计算仍对输入 action/state 求导，故梯度通过冻结 dynamics/reward/FK 回传 Actor。可选的末端 critic value 也可对 s_H 求导。WM/Critic 梯度状态在 finally 恢复；Critic 自己更新时使用 detach 的 rollout/target（[imagination/policy_trainer.py](../src/supermimic/imagination/policy_trainer.py#L69)）。

## 7. 更新规模与 Return-based Reset Sampling

- 每个 Trainer iteration 采集一个 vector env 控制步，即 E 条真实 transition；默认 G1 `E=4096`、Go2 `E=3072`、M1-Piper `E=2048`，CLI 可显式覆盖。一个控制步不是 4 条 transition；底下 4 个 physics substep 合成一条控制 transition。
- 默认 UTD=0.4，40 步 warmup 后平均每 2.5 个 collection iteration 触发一次 WM update。一次默认 WM update 为 10 batches × 2048 windows × 8 steps，即抽 20,480 windows/163,840 transition positions；从 Replay 有放回采样且窗口可重叠，不代表新增物理样本。
- 每次 WM update 后默认 1 次 Policy update：4096 starts × H=32，即 131,072 次 imagined dynamics step，不是真实仿真 transition。默认值见 [config.py](../src/supermimic/config.py#L43)、[world_model/wm_trainer.py](../src/supermimic/world_model/wm_trainer.py#L16)、[imagination/policy_trainer.py](../src/supermimic/imagination/policy_trainer.py#L20)。

Go2 与 M1 当前还配置各 200 次 WM-only update event，再开始 Actor/Critic 更新；Go2 Actor LR 另按实际 Actor optimizer update 逐步 ramp。该配置让 WM 先接触 Replay，但没有正类 failure transitions 时不能让 TerminationHead 获得正类监督；相关安全诊断见 [MULTI_ROBOT_VALIDATION.md](MULTI_ROBOT_VALIDATION.md#p1-c-wm-8-steptermination-survival-与-actor-bptt)。

Return sampler 仅供 motion task：将每步 reward 归到当前 reference 所属 50-frame bin；episode 结束后反向算 discounted return-to-go，对 bin return 做 EMA 更新。timeout 可用 critic bootstrap；没有 bootstrap 时丢弃截断点附近样本。采样时 softmax(-temperature * bin_return / g_max) 使低回报 bin 更容易选，再与 uniform、历史 memory 分布按默认 0.5/0.4/0.1 混合；选中 bin 后在 bin 内抽 frame，并限制在同 clip 范围（[envs/mdp/commands.py](../src/supermimic/envs/mdp/commands.py#L196)、[envs/mdp/commands.py](../src/supermimic/envs/mdp/commands.py#L251)、[envs/mdp/commands.py](../src/supermimic/envs/mdp/commands.py#L399)）。Go2 无 Motion Library，不走该 reset sampler。

## 8. 添加其他机器人或任务

### 新机器人

1. 在 get_robot_spec() 注册 joint names/order、默认 pose、控制 mode/scale、Kp/Kd、root/anchor/tracked bodies、model path 和 AMP joint subset。
2. 在 load_robot_spec_model()、get_robot_entity_cfg() 的 asset builder 中加载模型；按名称检查 compiled nq=7+n、nv=6+n。新机器人不能只加 Spec：当前 asset builder 的具体加载逻辑还需增加分支。
3. 混合执行器要实现相应 mjlab ActionTerm，明确 action 顺序、噪声、clip、scale 与目标单位；单一 position action 可复用 NoisyJointPositionAction。
4. 配置 observation、foot geom、COM body、wheel 特例和 FK tracked bodies；做 16-env physics smoke、joint order/FK 对照、Replay、WM 8-step 与 H=8/32 梯度检查后再扩规模。

### 新任务

1. motion reference 可复用 MotionTrackingTask；固定命令可复用 VelocityCommandTask。在 Trainer.__init__() 配置 task、Goal dimension 和 network interface。
2. 在 multirobot_env_cfg.py 配置 command、policy/goal observation、真实 reward 和 termination label。Replay 中要么记录 reference-free Goal 快照，要么确保能用 frame 重建 motion Goal。
3. 给 imagination 的一步函数实现可微 reward/termination features，并明确每一步 Goal 的推进规则、Critic 输入和失败标签。
4. Go2 的首版 reward core 虽已接入训练链路，当前 `Trainer.run()` 在完成配置的 WM-only 更新后仍会调用 Actor/Critic 更新；源码没有“正式 reward 已确认”运行时 guard。团队确定正式 reward 前，不要把这种配置下的训练称为正式 blind-locomotion 训练。M1 可继续使用现有 motion tracking objective。

下一阶段代码入口：Go2 在 multirobot_env_cfg.py velocity rewards/terminations、imagination/tasks.py::VelocityCommandTask.reward()、imagination/compiled.py::VelocityImagStep；M1 在 multirobot_env_cfg.py motion reward terms、world_model/heads.py::AnalyticHeads.reward()、envs/mdp/rewards.py::tracking_reward_terms()、envs/mdp/motion_library.py::goal_features()。

## 9. 本轮训练前接口验收更新

### 关节顺序和显式映射

- `load_robot_spec_model()` 按 MuJoCo `jnt_qposadr` 排列 scalar joints，检查完整顺序等于 RobotSpec，并检查 free base 后的 qpos/qvel 地址连续。只比较 name set 不再算通过。
- M1 的运行时 mjlab model 会把 joint 名写作 `robot/FAR_ABAD_JOINT`；`StateIO` 去掉实体 namespace 后建立 joint-name → qpos/dof address 映射，再按 RobotSpec 顺序读写。不会把原始 MuJoCo 地址顺序误当成 Actor 顺序。
- Trainer 检查运行时 articulation joint order；M1 action term 用 `find_joints(..., preserve_order=True)` 并校验顺序；Go2 position action 使用 RobotSpec 的 `actuator_names` 并要求保序。
- M1 的 wheel joints 索引为 3、7、11、15。持久化验收给一个 wheel 和一个 Piper 关节不同非零 action，检查只有对应 velocity/position target 改变。

### M1 observation 和 imagination

`analytic_observation(..., wheel_indices)` 将轮角从 joint-position error 特征中 mask 掉，但不裁剪动力学 state；joint velocity 与 clean previous action 不变。`ImagStep` 和 rollout 从 `StateSpec.wheel_indices` 传值。encoder bias 仍来自同一 `xi`，mask 发生在 `q + bias - default_joint_pos` 后，与 `joint_pos_rel_masked()` 对齐。关闭 observation corruption 后，M1 单帧与 720D term-major history 的持久化测试都通过数值精度比较；G1/Go2 的空 wheel index 分支保持原行为。

### M1 reward reanchor 时间点

环境在控制步结束时先算 RewardManager，再更新命令/reference 缓存。M1 imagined reward 现在可以使用本步 action 执行前的 anchor state 来重建 body-relative reference，和真实 `MotionCommand` 缓存语义一致；由 `reward_anchor_at_start` 显式控制，默认为关闭，因此 G1 原路径不改。真实/解析项的剩余小差异来自 mjlab reward 计算时 link 派生量有一个 physics substep 的缓存延迟。

### 第一阶段 M1 数据白名单

`configs/m1_piper_first6.csv` 保留本机 PhysX tracking pass 的六条 clip。`Trainer` 默认只读取这六条，并拒绝配置其它 clip 或用 packed library 绕过 manifest。NPZ 文件在本地只读验收，没有删除、重转换或覆盖 Motion Bank。
