# 阶段一：复用 mjlab PPO，统一 Go2 任务

日期：2026-11-04。正式对比尚未训练；已完成训练链路短测。

## 开源来源与复用范围

先检索 GitHub，选择 [giangdao1402/go2_velocity](https://github.com/giangdao1402/go2_velocity)，下载到 `external/go2_velocity`，固定版本 `23a364b1aafc454b5c281b9f115de434e6db2280`。也查看了 [Kaweees/unitree_mjlab](https://github.com/Kaweees/unitree_mjlab) 和 [mujocolab/mjlab](https://github.com/mujocolab/mjlab)。选择的项目本身就是自定义 Go2 + mjlab 速度任务 + RSL-RL PPO，适合做环境适配。

外部项目已加入本机 `.git/info/exclude`，不作为本仓库提交内容。没有安装它的 uv 环境，也没有使用 `.venv`。解释器统一为 `/home/yifeiy/miniconda3/envs/supermimic/bin/python`。

其他机器复用时需要先下载同一版本：

```bash
git clone git@github.com:giangdao1402/go2_velocity.git external/go2_velocity
git -C external/go2_velocity checkout 23a364b1aafc454b5c281b9f115de434e6db2280
```

上游配置还使用合并式 ActorCritic，当前安装的 mjlab/RSL-RL 使用分离 Actor/Critic。`src/supermimic/baselines/go2_ppo.py` 读取上游 `rl_cfg.py` 中的 PPO 参数，转换配置接口，然后直接运行安装版本的 `MjlabOnPolicyRunner`、`RslRlVecEnvWrapper`、`MLPModel` 和 `PPO`。rollout、GAE、PPO loss、优化器、存储和 TensorBoard 全部走上游实现，没有另写 PPO。基准环境调用本仓库同一个 `make_env_cfg`，不复制另一套奖励。

## 对齐项目

| 项目 | 两种算法共用设置 |
|---|---|
| 机器人与环境 | 同一个 Go2 URDF、同一个 mjlab 环境工厂，3072 envs |
| 频率 | 物理步长 0.005 s，decimation=4，策略 50 Hz |
| episode | 10 s，跌倒高度 0.17 m、倾角 70° |
| Actor 输入 | 5 帧 proprio；42×5=210 维，另加当前 body-frame 命令 3 维 |
| proprio | q−q_default、qd、body omega、projected gravity、previous clean action；按 term 展开，每个 term 内从旧到新 |
| 固定缩放 | q×1，qd×0.2，omega×0.5，gravity×1，action×1，command×1；没有 PPO running normalization |
| Actor 均值网络 | 213→2048→1024→512→256→12，SiLU，tanh 输出 |
| Critic 输入/宽度 | canonical state37 + goal3 + normalized xi16 + zero root_error4；256/128 |
| 动作/PD | 训练关节顺序 FL/FR/RL/RR，各关节 scale=0.25，Kp=40、Kd=1 |
| 初始化 | 两者从头训练，seed=2026；不加载旧 Actor 或 WM 权重 |
| 环境探索/随机化 | 同一 DR 和执行噪声；Gaussian 0.1，25% env 用 OU，sigma=0.3、theta=2；obs/reset noise 关闭 |
| 命令课程 | 同一 A/B/C 采样与晋级规则，从 A 开始，5 s 重采样 |
| horizon | PPO 真实 rollout 32；SuperMimic 想象 horizon 32 |
| 逻辑 minibatch | 1024；SuperMimic microbatch 512 是梯度累积细节 |
| discount / GAE | 0.99 / 0.95 |
| 初始 Actor 学习率 | 2e-4 |

PPO 的 rollout batch=3072×32=98,304，所以是 **96 个 minibatch，每个 1024 条**。不能机械地同时保留上游的 4 个 minibatch，否则每批会变为 24,576 条。PPO 使用上游 5 epochs、clip=0.2、entropy_coef=0.01、adaptive KL schedule；SuperMimic 保留 WM 更新、想象梯度、warmup 和自己的 LR schedule。这些属于算法差异。

额外修正了新的 SuperMimic 对比入口：旧配置中 `env.obs_noise=false`，但 `policy_trainer.obs_noise=true`，意味着想象 Actor 输入仍会加噪声。现在显式让两者一致，新的无观测噪声对比也关闭想象输入噪声；没有修改历史 checkpoint 或共享训练器。

“两种算法配置相同”在这里指相同任务、观测、奖励和可比的网络/采样设置；真实 rollout 与想象 rollout、优化次数、PPO 的额外随机策略采样不可能完全相同。PPO Gaussian 初始 std=0.3；确定性均值使用与 SuperMimic 相同的 tanh 网络。评测和部署均取确定性均值。

## 奖励核

所有权重是乘 dt 前的 rate，RewardManager 只乘一次 dt。PPO 直接调用 `tasks/velocity_reward.py`，与 SuperMimic 实际环境一致；SuperMimic 想象训练也使用同一个纯函数奖励核。

| 项 | 权重 |
|---|---:|
| 平移跟踪 / yaw 跟踪 / 方向 | 3.0 / 1.2 / 0.5 |
| upright / vz / omega_xy | 0.5 / −1 / −0.05 |
| action rate / torque / joint acceleration | −0.03 / −1e-5 / −2.5e-7 |
| joint deviation / joint limit / low height | −0.02 / −1 / −5 |
| 抬脚 / 对角支撑 / 滑脚 | **0 / −0.15 / −0.20** |

tracking_sigma_sq=0.09，zero_speed_baseline=true，min_safe_height=0.22。抬脚奖励继续关闭。

## 时间与公平性

主比较采用用户指定的 **从头训练，各 2.5 h 墙钟时间**。计时包含学习循环中的仿真、编译、模型更新与存盘，初始化单独发生。PPO 在完整 update 后检查预算；SuperMimic 在 log_interval=50 的边界检查，所以都有少量超时，必须报告实际时间。

同时报告真实环境转移数，作为采样效率的补充。PPO iteration 包含 32 个真实环境步；SuperMimic iteration 是一次真实环境步，不能直接对齐 iteration 数。32 步想象轨迹也不能计入真实样本数。

旧 SuperMimic `ckpt_0034000.pt` 是续训和历史奖励/课程产生的结果，仅用作部署验证，不能贴上“从头训练 2.5h”的标签。命令课程共用晋级规则，但由于策略表现不同，两者晋级时间可能不同；组会应同时展示最终阶段/命令覆盖，必要时补充固定课程的控制实验。

## 已完成验证

- 64 env ×16 steps ×2 updates：2048 条转移，训练/保存/导出成功。
- **3072 env ×32 steps ×2 updates**：196,608 条真实转移，约 24.13 s；最后 value loss≈0.0065、surrogate≈−0.0016，均为有限值；这些只证明链路可运行，不能证明策略已学会行走。
- 结果位于 `logs/go2_ppo/smoke_3072_20261104/{comparison_config.json,progress.json,completion.json,model_1.pt,actor_latest.pt}`。
- 对齐、Actor 均值/历史转换、实际 PPO checkpoint 导出、SDK 关节映射、资源读取/越限清理及已有步态测试合计 **18 项通过**，日志 `logs/go2_comparison/contract_tests.log`。

新代码只在 `baselines`、新增脚本、测试和三阶段文档中；既有 M1 修改不属于本任务。
