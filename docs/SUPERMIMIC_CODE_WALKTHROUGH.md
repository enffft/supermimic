# SuperMimic 源码走读：从仿真采集到可微想象训练

本文以仓库当前实现为准，对照论文 Figure 2、§3.1–3.3、Appendix A–D。关键论文位置：Figure 2 与 §3.1–3.3（印刷页 3–5）、Appendix A（页 12–13）、Appendix C（页 14）、Appendix D.1–D.2 与 Table A.5（页 15–16）。

源码行号按本次阅读时的工作树记录。每节都把公式落到实际函数、张量和控制流上；“论文说”与“代码做”不完全相同的地方会直接指出。

## 1. 真实仿真采集一个向量化控制步

调用入口是 [`Trainer.run()`](../src/supermimic/training/loop.py#L146)：第 150 行先 reset 得到观察；每轮第 162–163 行对整批观测调用 `act()`，再执行 `env.step(action)`。默认 `num_envs=4096`，所以一个“控制步”是 4096 个环境各推进一次，而不是单个机器人推进一次（[config.py:L23-L35](../src/supermimic/config.py#L23)）。`act()` 在 [`loop.py:L142-L144`](../src/supermimic/training/loop.py#L142) 的 `no_grad` 下调用策略，输入为 `obs["policy"]` 和 `obs["goal"]`。

逐环境的数据流如下：

1. 策略读取 10 帧本体感觉历史 `(N, 930)` 和未来参考目标 `(N, 680)`，输出 clean/raw action `(N, 29)`。其中 `N=4096`；观测各项为关节位置 29、关节速度 29、机身角速度 3、投影重力 3、上一步动作 29，共 93 维，十帧历史展平为 930 维；目标每帧 68 维，默认十帧为 680 维（[tracking_env_cfg.py:L95-L127](../src/supermimic/envs/tracking_env_cfg.py#L95)、[heads.py:L20-L39](../src/supermimic/world_model/heads.py#L20)、[motion_library.py:L17-L20](../src/supermimic/envs/mdp/motion_library.py#L17)）。
2. `env.step()` 的 action manager 调用 `NoisyJointPositionAction.process_actions()`：保存 clean action；加逐维高斯噪声 `a_exec = a + 0.1·randn`（默认配置；论文 Table A.5 写 `N(0, 0.12)`）；再乘 G1 action scale、加 default-pose offset，必要时 clip，成为关节位置控制 setpoint（[actions.py:L21-L43](../src/supermimic/envs/actions.py#L21)、[tracking_env_cfg.py:L130-L139](../src/supermimic/envs/tracking_env_cfg.py#L130)、[config.py:L24-L35](../src/supermimic/config.py#L24)）。模型训练用的 `a_exec` 是加噪后的 normalized action，而不是 actuator setpoint。
3. 仿真配置为 `timestep=0.005`、`decimation=4`，因此每个控制步执行 4 个物理子步，总控制间隔 `dt=0.02s`（50 Hz；[tracking_env_cfg.py:L237-L239](../src/supermimic/envs/tracking_env_cfg.py#L237)）。mjlab 的 step 调度会在子步 apply action、写入仿真、推进物理并更新场景；之后计算 termination 和 reward。
4. termination 检查 root/pelvis 高度偏差和倾斜偏差；clip end 与 episode 时长上限是 timeout（[terminations.py:L16-L31](../src/supermimic/envs/mdp/terminations.py#L16)、[tracking_env_cfg.py:L215-L221](../src/supermimic/envs/tracking_env_cfg.py#L215)）。Reward Manager 按 `dt=0.02` 缩放奖励。
5. Recorder 在非终止环境保存该步 transition；终止/timeout 环境先在 reset 前写最后一条 transition 和一个只含终态的额外 row，之后才 reset。普通 row 保存 transition 起点状态 `prev_s`，clean action、executed action、context、奖励和目标帧；reset 后 `record_post_reset()` 重新记住新 episode 起点（[recorder.py:L55-L120](../src/supermimic/data/recorder.py#L55)）。`Trainer.run()` 通过累加 `rew` 和 `term|tout` 计算日志指标，实际 Replay Buffer 写入由 Recorder 内部完成（[loop.py:L159-L177](../src/supermimic/training/loop.py#L159)）。

机器人状态是相对每个并行环境 origin 的 `[qpos(36), qvel(35)]`，合为 `(N,71)`；`StateIO.read()` 按 root position 3、root quaternion 4、关节位置 29、root 线速度 3、root body-frame 角速度 3、关节速度 29 拼接（[state_io.py:L7-L30](../src/supermimic/envs/state_io.py#L7)）。sim context `xi` 是摩擦 1、encoder bias 29、躯干 COM offset 3，共 `(N,33)`（[sim_context.py:L8-L40](../src/supermimic/envs/sim_context.py#L8)）。

## 2. Replay Buffer 存储的张量和 Shape

[`ReplayBuffer.__init__()`](../src/supermimic/data/replay_buffer.py#L39) 使用 `[env,time]` 布局，默认容量 `T=1000`、并行环境数 `E=4096`。注意 ring buffer 是每个环境各留 1000 行，并非全局总共 1000 条。

| 字段 | 分配 Shape | 类型 / 含义 |
|---|---:|---|
| `s` | `(E,T,71)` | 每个控制 transition 的起点仿真状态 |
| `a` | `(E,T,29)` | clean policy action |
| `a_exec` | `(E,T,29)` | 加噪后实际送入 action scale/offset 的 action；WM 用它训练 |
| `xi` | `(E,T,33)` | episode 的仿真 context |
| `r`, `done` | `(E,T)` | 奖励；目标状态是否为硬 termination（float 存储） |
| `frame`, `clip`, `ep_id` | `(E,T)` | 参考帧、clip 索引、episode 索引（整数） |
| `terminal`, `ep_start`, `valid` | `(E,T)` | 终态 row、episode 起点、有效 row 标记（bool） |
| `ptr`, `ep_counter` | `(E,)` | 每环境 ring 写指针和 episode 计数 |

见 [`replay_buffer.py:L39-L60`](../src/supermimic/data/replay_buffer.py#L39) 和实际写入 [`ReplayBuffer.add()`](../src/supermimic/data/replay_buffer.py#L64)。额外终态 row 的 `s` 是最后动作抵达的状态，动作和奖励置零；`terminal=True`，并带该终态的 termination label（[recorder.py:L85-L108](../src/supermimic/data/recorder.py#L85)）。timeout 也有 terminal row，但 `done` 来自 `reset_terminated`，因此纯 timeout 的模型终止标签是 0。

采样时的 `WindowBatch` 形状定义为 `states=(k+1,n,71)`、action `(k,n,29)`、`xi=(n,33)`、`reward/done/frame=(k,n)`；`StartBatch` 则包含 state、context、frame、previous action 和 history（[replay_buffer.py:L18-L36](../src/supermimic/data/replay_buffer.py#L18)）。Buffer 不存观察历史；想象时从记录状态和 clean action 重建观察。

## 3. Motion Library 与 Replay Buffer 的根本区别

两者不是同一类数据集：

- **Motion Library 是静态参考轨迹库**：按全局 frame 拼接所有 reference clip，主要保存参考状态 `(F,71)`；full 模式另存 14 个 tracked bodies 的位置/四元数/线速度/角速度和 goal feature `(F,68)`。compact 模式仅存 71 维 reference state，其余通过 FK 现算（[motion_library.py:L1-L20](../src/supermimic/envs/mdp/motion_library.py#L1)、[motion_library.py:L100-L168](../src/supermimic/envs/mdp/motion_library.py#L100)）。它回答“目标动作是什么”，提供 frame、clip、bin 查询及未来目标窗口（[motion_library.py:L217-L242](../src/supermimic/envs/mdp/motion_library.py#L217)）。
- **Replay Buffer 是在线仿真经验**：每个 row 是 policy 在随机化物理环境里执行动作后产生的状态转移和 reward，包括偏离参考的真实 robot state、动作噪声、context、done 等；它回答“当前控制器在仿真中实际发生了什么”。它是按环境写入、旧数据环形覆盖的动态缓存（[replay_buffer.py:L1-L8](../src/supermimic/data/replay_buffer.py#L1)）。

换句话说，Motion Library 的 frame 是目标和 reset 来源；Replay Buffer 的 frame 是把一条真实 transition 对齐回该目标轨迹的索引。Motion Library 没有 policy action/reward/context，Replay Buffer 也不代替完整 reference clip。

## 4. `sample_windows` 如何抽出连续 8 步

[`ReplayBuffer.sample_windows(n,k)`](../src/supermimic/data/replay_buffer.py#L96) 先用 `window_mask(k)` 找到可用起始 `(env,t)`：起点及其后 `k` 个状态 row 都有效、`ep_id` 相同，不能在中间穿过 terminal row；同时排除即将被 write pointer 覆盖的数据（[replay_buffer.py:L96-L112](../src/supermimic/data/replay_buffer.py#L96)）。合格起点经 `torch.multinomial(..., replacement=True)` 有放回抽样，因此 batch 可重复。

之后用偏移 `0…k` gather 状态，用 `0…k-1` gather transition 字段（[replay_buffer.py:L123-L137](../src/supermimic/data/replay_buffer.py#L123)）。默认 `k=8,n=2048` 得到：

- `states`: `(9,2048,71)`，即 `s_t…s_{t+8}`；
- `a_exec`, `a_clean`: `(8,2048,29)`，即连续 8 个 transition 的动作；
- `reward`, `done`, `frame`: `(8,2048)`；`xi`: `(2048,33)`，取窗口起点 context，episode 内保持不变。

若 episode 恰在第 8 步结束，最后一个状态可以是额外 terminal row：这让训练包含真实终态及对应的 `done` 正例，但不会把不同 episode 拼成一个窗口。WM 只把 `a_exec` 用作 rollout action。

## 5. World Model：State Canonicalization、Dynamics MLP、积分

调用链是 [`WorldModel.step()`](../src/supermimic/world_model/world_model.py#L50) → [`StateSpec.canonical()`](../src/supermimic/world_model/state_spec.py#L78) → [`DynamicsMLP.forward()`](../src/supermimic/world_model/dynamics.py#L31) → `StateSpec.integrate()`（[state_spec.py:L91-L107](../src/supermimic/world_model/state_spec.py#L91)）。

原始状态 `s=(…,71)` 分成 root position、root quaternion、joint position、root world linear velocity、root body angular velocity、joint velocity。Canonical feature 仍是 71 维，但去掉全局平移和 heading 依赖：只用 root 高度 1、去 heading 后的 root orientation 6D、heading frame 线速度 3、body-frame 角速度 3、joint position 29、joint velocity 29；再按 `in_scale` 除尺度（[state_spec.py:L74-L89](../src/supermimic/world_model/state_spec.py#L74)）。

网络输入拼接 `(canonical state 71, action 29, normalized xi 33)`，总计 **133 维**；3 层各 2048 个 SiLU hidden units，输出 70 维 delta，最后一层零初始化（[dynamics.py:L9-L39](../src/supermimic/world_model/dynamics.py#L9)、[world_model.py:L16-L24](../src/supermimic/world_model/world_model.py#L16)）。`xi` 用 `(xi-offset)/scale` 归一化，涵盖足底摩擦、编码器偏置和躯干 COM 偏移（[state_spec.py:L57-L70](../src/supermimic/world_model/state_spec.py#L57)）。

70 维输出依次代表 heading frame 的 `dv(3)`、body frame `dw(3)`、关节 `dqd(29)`、heading frame `dp(3)`、body-frame 姿态修正 `dθ(3)`、关节位置修正 `dq(29)`。先乘 `out_scale`，再做显式积分，默认 `dt=0.02s`：

- `v' = v + R_heading·dv`，`w'=w+dw`，`qd'=qd+dqd`；
- `p'=p + dt·v' + R_heading·dp`；
- `q'=normalize(q ⊗ exp(dt·w' + dθ))`；
- `qj'=qj + dt·qd' + dq`。

因此零初始化的输出仍让 root/joint 位置按旧速度前进，是常速度先验，不是状态完全静止。积分细节在 [`StateSpec.integrate()`](../src/supermimic/world_model/state_spec.py#L93)。

## 6. World Model 的 8-step Loss 与反向传播

[`WorldModelTrainer.update()`](../src/supermimic/world_model/wm_trainer.py#L52) 每个 WM update 重复 10 次：各自抽一批 2048 个 8-step windows，建 termination 参考 root pose，调用 `window_loss()`，清梯度、`loss.backward()`、跨卡平均梯度、全局范数裁剪到 1，再做 Adam step（[wm_trainer.py:L27-L67](../src/supermimic/world_model/wm_trainer.py#L27)）。

[`WorldModel.window_loss()`](../src/supermimic/world_model/world_model.py#L77) 从真实 `states[0]` 开始；第 `j` 步用预测状态递归地作为下一步输入，而不是每步重置到真值，动作是 replay 的 `a_exec[j]`。对预测 `ŝ_{j+1}` 和真实 `s_{j+1}`，先计算 70 维归一化误差：位置差、四元数相对旋转向量、root 线/角速度差、关节位置/速度差，再按误差尺度 `(0.05,0.1,0.5,1.0,0.1,2.0)` 分组归一（[state_spec.py:L49-L54](../src/supermimic/world_model/state_spec.py#L49)、[state_spec.py:L111-L125](../src/supermimic/world_model/state_spec.py#L111)）。

对每一步，代码先将平方误差沿 70 个状态维度求和，再对 2048 窗口求均值；8 步求平均。再加上每一步 termination BCE 的 batch mean，并按 `term_loss_weight=1` 加权（[world_model.py:L65-L72](../src/supermimic/world_model/world_model.py#L65)、[world_model.py:L84-L113](../src/supermimic/world_model/world_model.py#L84)）。因此这里是跨步**平均**，与论文 Eq. (1) 的时间求和只差整体 `1/k` 尺度；状态误差另有显式分量归一化。

论文 Eq. (1) 同时列出 state、reward、observation 和 termination 项，但说明解析 head 不需要 loss。当前仓库的 reward 和 observation 都是解析 head；`window_loss()` 实际优化的只有 state prediction loss 与 learned termination BCE，Replay Buffer 中保存的 `reward` 目前没有送进这段 loss。`nse_1step`/`nse_kstep` 是归一化平方误差的诊断量，不是额外损失。

## 7. `sample_starts` 如何初始化 32-step Imagination

[`sample_starts(n,hist)`](../src/supermimic/data/replay_buffer.py#L139) 从 `valid & ~terminal` 的真实仿真状态行均匀有放回采样；排除会被 writer 覆盖的历史区间。每个起点 `(env,t)` 取它之前至当前的 `hist` 个状态：默认 `hist=10`，即 `(10,n,71)`。若历史早于 episode 起点或无效，用最早可用的 episode 内状态回填；每个历史状态对应的前一 clean action 不存在时填 0（[replay_buffer.py:L140-L167](../src/supermimic/data/replay_buffer.py#L140)）。

默认 policy batch 为 `n=4096`。返回 `state=(4096,71)`、`xi=(4096,33)`、`frame=(4096,)`、`prev_action=(4096,29)`、`hist_states=(10,4096,71)` 和 `hist_actions=(10,4096,29)`。`PolicyTrainer.update()` 将 `xi` 归一化后调用 `imagine(..., horizon=32)`（[policy_trainer.py:L18-L25](../src/supermimic/imagination/policy_trainer.py#L18)、[policy_trainer.py:L60-L68](../src/supermimic/imagination/policy_trainer.py#L60)）。

`imagine()` 先用十个历史 state/action 通过解析 observation head 重建历史观察，初始参考 frame、context 与上一 clean action 来自 Replay Buffer。它预先 gather `(H+1,n)` 的参考帧，超出 clip 末尾就 clamp 到末帧（[rollout.py:L68-L96](../src/supermimic/imagination/rollout.py#L68)）。

## 8. Actor 如何每一步重算动作

[`imagine()` 的循环](../src/supermimic/imagination/rollout.py#L98) 每步都重新执行完整 actor，而不是把第一步 action 重复 32 次：

1. 将最近 10 个 93 维 noisy observations 按 term-major 历史顺序展平为 `(n,930)`；
2. 取当前参考 frame 的未来 10 帧 goal `(n,680)`；
3. `policy(obs_hist, goal)` 编码 goal 得到 `z∈R^32`，再将 `z` 与历史观察拼接，经 decoder 得到 `a_t∈R^29`（[nets.py:L51-L82](../src/supermimic/policy/nets.py#L51)）；
4. 用这个动作推进一小步预测模型，获得 clean next observation；在 observation 外部重新采样并加均匀传感器噪声；滑动历史窗口，设置 `a_prev=a_t`，下一循环由新预测状态和新目标重新生成动作。

默认策略网络是 motion goal encoder 的 MLP `(2048,1024,512,256)`→32 维 tanh latent，以及 shared decoder `(2048,1024,512,256)`→29 维动作（[nets.py:L18-L30](../src/supermimic/policy/nets.py#L18)）。想象中的 action 不再加训练用的 Gaussian action noise；论文也只把 action noise 用于模型拟合，想象 policy 是确定性映射。每一步重新加的是 observation noise，见 [`rollout.py:L17-L20,L54-L57,L98-L109`](../src/supermimic/imagination/rollout.py#L17)。

论文 Appendix B 描述多个 goal interface 共享 decoder/world model 的训练；当前 [`Policy`](../src/supermimic/policy/nets.py#L72) 的 `ModuleDict` 只注册 `motion` encoder，`forward()` 虽保留 interface 参数，但仓库现状没有 velocity 等其他 interface 的 encoder/Trainer。

## 9. Reward、Observation 和 Termination Head

本仓库采用“学 dynamics + 学 termination + 解析 observation/reward”的结构（[world_model.py:L1-L3,L40-L48](../src/supermimic/world_model/world_model.py#L40)）。

- **Observation：解析、无噪声**。`analytic_observation(s,prev_action,xi,q0)` 输出 `(…,93)`：`joint_pos + encoder_bias - q0` (29)、joint velocity (29)、root body angular velocity (3)、root 姿态下的 projected gravity (3)、previous clean action (29)（[heads.py:L30-L39](../src/supermimic/world_model/heads.py#L30)）。噪声在模型外添加：real env 用 Table A.1 对应的 uniform corruption；imagination 中 `rand*2-1` 乘半宽 `[0.01×29,0.5×29,0.2×3,0.05×3,0×29]`（[rollout.py:L17-L20,L54-L57](../src/supermimic/imagination/rollout.py#L17)）。
- **Reward：解析、可微**。`ImagStep` 先从预测 state 做 G1 FK，再调用 `AnalyticHeads.reward()` 和 `tracking_reward_terms()`（[compiled.py:L37-L48](../src/supermimic/imagination/compiled.py#L37)、[heads.py:L101-L131](../src/supermimic/world_model/heads.py#L101)）。包含 6 个 tracking 指数项：root anchor 全局位置/姿态、anchor 对齐后的 tracked-body 位置/姿态、body 线速度/角速度；加 action-rate 惩罚与 joint-limit 惩罚。默认权重依次为 `0.5,0.5,1,1,1,1,-0.1,-2`，对应 std 为 `0.3,0.4,0.3,0.4,1,3.14`，最后每项乘 `dt=0.02`（[rewards.py:L23-L43,L93-L127](../src/supermimic/envs/mdp/rewards.py#L23)）。实现没有 self-collision penalty。
- **Termination：learned logits**。先用预测 root position/quaternion 与当前 reference root 算两个 feature：高度偏差和重力方向在 pelvis frame 的倾斜差；按 `0.25m`、`0.8` 阈值缩放，同时拼上 absolute value，形成 4 维输入，经 `(64,64)` MLP 输出 logit（[heads.py:L133-L143](../src/supermimic/world_model/heads.py#L133)、[world_model.py:L26-L37](../src/supermimic/world_model/world_model.py#L26)）。WM 训练用 `reset_terminated` 的硬 label 做 BCE；想象时 sigmoid 成软概率。timeout/clip-end 不是该 head 的正例；想象中的 learned termination 只覆盖 root height/tilt，参考帧在 clip 末尾 clamp。

## 10. Soft Survival Weight 如何影响奖励

想象开始设 `c_0=1`、`d̂_0=0`。每步由 termination logit 得到 `d̂_{t+1}=sigmoid(logit)`，更新 `c_{t+1}=c_t(1-d̂_{t+1})`（[rollout.py:L73-L77,L98-L115](../src/supermimic/imagination/rollout.py#L73)）。`c_t` 代表截至 t 仍存活的软权重：预测死亡概率越高，后续奖励和 bootstrap 越小，但不会像硬阈值那样截断梯度。

Actor 的第 `t` 项即时奖励乘的是到达当前状态前的 `c_t`；Critic bootstrap 乘 `c_H`。Critic 的 backward target 也在每步乘 `(1-d̂_{t+1})`（[objective.py:L10-L20,L23-L35](../src/supermimic/imagination/objective.py#L10)）。

## 11. Actor 目标函数对应论文 Eq. (2)

[`policy_objective()`](../src/supermimic/imagination/objective.py#L10) 计算 batch mean：

```text
J = mean_n [ sum_(t=0..H-1) gamma^t * c_t * r_(t+1)
             + gamma^H * c_H * V_H ]
```

代码中的 `disc=gamma**arange(H)` 与 `survival[:-1]*rewards` 对应前半项；当 `use_critic=True`，`terminal_values` 是 critic 在 `s_H,g_H,xi` 上的估值，补上尾项。默认 `gamma=.99`，H=32（[objective.py:L10-L20](../src/supermimic/imagination/objective.py#L10)、[policy_trainer.py:L18-L35](../src/supermimic/imagination/policy_trainer.py#L18)）。Critic 实际还接收 canonical state 和 4 维 root drift feature，属于代码增加的输入细节（[nets.py:L85-L96](../src/supermimic/policy/nets.py#L85)）。

训练器令 `loss=-J/H`，对 actor 反传后用 Adam 最小化；负号使其等价于提升 Eq. (2)，除以 H 只缩放梯度（[policy_trainer.py:L65-L81](../src/supermimic/imagination/policy_trainer.py#L65)）。

## 12. Critic Lambda Return 对应论文 Eq. (3)

[`gae_targets()`](../src/supermimic/imagination/objective.py#L23) 从 `R_H=V_H` 开始，倒序执行：

```text
R_t = r_(t+1) + gamma * (1-dhat_(t+1))
      * ((1-lambda)*V_(t+1) + lambda*R_(t+1))
```

与论文 Eq. (3) 的 λ-return 完全对应；代码传入 `dhat_next=ro.dhat[1:]`。`PolicyTrainer.update()` 在 `no_grad` 下用 rollout 的全部 `H+1` 个状态计算 values，再算一次固定 targets；将 `t=0…H-1` 的 canonical state、goal、xi 和 root error 展平，拟合 MSE。默认 `λ=.95`、5 epochs、每 epoch 4 个 shuffle minibatches；每次 policy update 有 `4096×32=131072` 个 value target，故每个 critic minibatch 32768 条（[policy_trainer.py:L97-L125](../src/supermimic/imagination/policy_trainer.py#L97)）。这与公式中 `E_t[(V_t-R̂_t)^2]` 的实现相符；`V_H` 只作 bootstrap，不进入 `t<H` 的 critic loss 样本。

## 13. 冻结 World Model 时梯度如何传回 Actor

Actor 阶段在 [`PolicyTrainer.update()`](../src/supermimic/imagination/policy_trainer.py#L60) 对 `wm.requires_grad_(False)`，并冻结 critic 参数（第 65–68 行），但**没有**把 `imagine()` 放到 `torch.no_grad()`、也没有对预测 state/action 做 detach。于是 WM 的权重不积累梯度，仍会保留 `ŝ_{t+1}=fω(ŝ_t,a_t,xi)` 对输入 `ŝ_t` 和 `a_t` 的 autograd Jacobian；state 递归保留计算图，预测 FK、解析 Reward、termination sigmoid/survival 和 Critic 终值也把导数传回 action。每个 action 又是 `πθ(obs_hist,goal)`，因此 BPTT 穿越 32 个动作和预测步，最终得到 `∂J/∂θ`。`loss.backward()` 后只对 policy 梯度做 all-reduce、裁剪、step，然后恢复 WM 与 critic 参数可训练（[policy_trainer.py:L65-L83](../src/supermimic/imagination/policy_trainer.py#L65)）。

重点是 `requires_grad_(False)` 冻结的是参数叶节点，不是函数关于输入的求导；若这里使用 `no_grad` 或 detach，Dynamics 的 Jacobian 就无法把奖励梯度传到 Actor。WM 仍在下一次 `WorldModelTrainer.update()` 中正常训练。

## 14. 每次更新的真实样本量、窗口数和想象步数

默认值见 [`config.py`](../src/supermimic/config.py#L23) 与 [`WMTrainerCfg` / `PolicyTrainerCfg`](../src/supermimic/world_model/wm_trainer.py#L16)：

| 计量口径 | 默认值 | 含义 |
|---|---:|---|
| 一个真实向量化控制步 | 4096 transitions | 每个环境推进 0.02 s；episode 结束的环境还会多写终态 row |
| update-to-data | 0.4 WM updates / env step | warmup 40 个控制步后累积小数 UTD；平均 2.5 个向量化控制步触发一个 update |
| 一个 WM update | 10 × 2048 = 20,480 windows | 每个 window 的 `k=8`，单次 batch 的 8-step rollout 有 9 个状态 |
| 一个 policy update / WM update | 4096 starts × H=32 | 每次 policy update 为 4096 条 rollout，每条 32 步，即 131,072 个 imagined transitions |
| policy updates per WM update | 1 | 默认值；由 schedule 控制 |

主循环的 UTD 累加器和更新调用在 [`loop.py:L179-L196`](../src/supermimic/training/loop.py#L179)，具体 batch 数分别在 [`wm_trainer.py:L16-L21`](../src/supermimic/world_model/wm_trainer.py#L16) 和 [`policy_trainer.py:L18-L35`](../src/supermimic/imagination/policy_trainer.py#L18)。因此源码并非每个仿真控制步都立刻做一次模型和策略更新；论文 Algorithm A.1 的逐轮描述在这里以 UTD=0.4 实现。

## 15. Return-based Reset Sampling

实现位于 [`MultiClipMotionCommand`](../src/supermimic/envs/mdp/commands.py#L192)：整条 motion library 按每 clip 50 帧分 bin（50Hz 下约 1 秒），`MotionLibrary.bin_of_frame()` 将 frame 映射到全局 bin（[motion_library.py:L156-L168,L219-L221](../src/supermimic/envs/mdp/motion_library.py#L156)）。每个 bin 持有 `bin_return=G_b`（初始为 `g_max=10`）和 `bin_memory`（初始均匀）。

Episode 内每步将 simulator reward 和当前 reference bin 记入 episode log。结束时对每个时间点计算 discounted return-to-go `R_t=Σ_{s≥t}γ^(s-t)r_s`，将该 bin 被访问时间点的 RTG 求均值，以 `ema_alpha=.05` 更新 `bin_return`（[commands.py:L194-L245](../src/supermimic/envs/mdp/commands.py#L194)）。reset 抽样时计算 `hard=softmax(-25*bin_return/g_max)`；`bin_memory` 按 `memory_alpha=.01` 对 hard 分布做 EMA，随后将三种分布按配置混合：50% 所有 bin 均匀、40% 当前 hard 分布、10% memory 分布。先按混合概率抽 bin，再在 bin 内均匀抽 frame，并将 frame 限制在 clip 至少还剩两步的区间（[commands.py:L249-L274](../src/supermimic/envs/mdp/commands.py#L249)、[commands.py:L410-L420](../src/supermimic/envs/mdp/commands.py#L410)）。低回报 bin 的 `-25G_b/g_max` 较大，softmax 权重更高；10% memory 项保留过去难点的采样份额。

代码的 timeout 处理也有边界：`bootstrap_fn` 默认是 `None`；对时间长度截断但非自然 clip end 的 episode，采样器会丢弃离截断点不足 100 步的 RTG 样本，以免无 bootstrap 时把截断尾部误当完整回报（[commands.py:L79,L223-L234](../src/supermimic/envs/mdp/commands.py#L79)）。

## 阅读时最重要的实现差异小结

1. 论文 Eq. (1) 写出 state/reward/observation/termination 项；本实现 reward 与 observation 为解析函数，没有独立学习损失，实际 WM loss 是归一化 state rollout error + termination BCE。
2. 论文说 action noise 用于仿真采集、Replay 记录 noisy action；代码明确同时记录 clean `a` 和 noisy `a_exec`，WM 用 `a_exec`，历史 observation / action-rate reward 用 clean action。
3. 论文 Algorithm A.1 概述每轮更新；代码默认 UTD=0.4，平均每 2.5 个向量化控制步更新一次，每次 WM 更新含 10 个独立 2048-window optimizer steps。
4. Actor 更新冻结模型参数但保留模型输入梯度；BPTT 才能把奖励关于未来状态的导数穿过 Dynamics 传回当前动作和 policy。
