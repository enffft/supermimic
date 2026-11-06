# M1-Piper 纯 Reward SuperMimic 基线

## 目的和实现边界

新增任务名 `m1_reward`，用于回答：不使用运动学数据集、Motion Prior、reference-copy 或预训练 Actor，仅靠真实交互数据训练 World Model，并让 Actor 通过 WM 的梯度优化任务奖励，2.5 小时能学到什么。

复用现有 `Trainer → ReplayRecorder/ReplayBuffer → WorldModelTrainer → PolicyTrainer → imagine → Actor/Critic`，没有重写强化学习算法。原有 `m1_prior_joint` 的零采样奖励、Prior/Action22 Actor 及训练阶段保留；Go2 奖励也不启用这些新项。新启动脚本不会串接 Go2/PPO 训练。

默认从零初始化三个网络，`resume=None`、`init_wm_from=None`、`m1_joint_training.enabled=False`。配置树仍包含旧任务的 Prior 路径字段，但本任务不读取该文件。短验收已确认 checkpoint 无 `motion_prior`、`m1_joint_trainer is None`。

## 参考工作

1. [RoboDuet 论文，arXiv:2403.17367v2](https://arxiv.org/html/2403.17367v2)、[官方实现](https://github.com/locomanip-duet/RoboDuet)。已阅读论文的策略/奖励/两阶段训练和消融部分，以及 `go1_gym/envs/rewards/rewards.py`，参考提交 `a7e1528215c048199f90cb69ceb7749a1d745f28`。借鉴速度指数核、末端跟踪、默认姿态和动作平滑正则。它使用两个协同策略和两阶段训练，论文消融训练 45,000 次迭代、3 个种子；本实验是单 Actor、单阶段 SuperMimic 基线，不能称为 RoboDuet 复现。
2. [Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking With a Wheeled-Quadrupedal-Manipulator，IEEE RA-L](https://doi.org/10.1109/LRA.2024.3519856)。已通过 Crossref 核对题目、作者和 DOI。相关公开实现参考 [whichisavailable/locomanip](https://github.com/whichisavailable/locomanip)，提交 `9c0a22676bdfaf9a3c2b1f65cd31010c4644b2bf`；阅读 README、flat 配置和 `mdp/rewards.py`。该仓库自述将原轮足任务迁移为四足版本，属于第三方参考，不能当作原论文官方轮足实现。其末端粗/细跟踪、任务门控和正则分工有参考价值；没有直接照搬四足抬脚、静态支撑足滑动或 trot 接触奖励到 M1 轮子上。

这些工作提供设计参考，并不证明下表权重对 M1 最优。阅读材料保存在 `logs/m1_reward_sources/`，不引入 Isaac Gym/Isaac Lab 运行依赖，不复制其训练框架。

## 命令、动作和环境分配

使用现有 `M1PriorJointCommand` 作为命令采样器，名字保留是为了复用接口；它本身不加载 CVAE。

| 接口 | 定义 |
|---|---|
| Replay 固定上下文 7D | `[task_type, vx, vy, wz, target_x, target_y, target_z]`；`0=L, 1=M` |
| Actor Goal8 | `[vx, vy, wz, heading_relative_target_xyz, is_L, is_M]`，与已有 `task_goal8` 一致 |
| L 命令 | 机体坐标速度 `vx,vy`，机体角速度 `wz`；第三维不是竖直速度 |
| M 命令 | 固定的 env-local 世界坐标目标点；每个仿真环境去掉各自 origin 后表达 |
| M 目标更新 | 真实/想象每一步根据当前机体 pose 重新计算 heading-relative 位置，世界目标不漂移 |
| 默认配额 | 2048 环境固定 1024 L、1024 M；reset 不改变任务标签 |
| Replay 更新采样 | balanced 模式下 WM/Actor 每个 micro-batch 各半 L/M；纯 L/M 模式普通采样 |
| Actor 输入输出 | 每帧 72D proprioception，10 帧 720D + Goal8；22D `tanh` 动作 |
| 物理动作 | v3 mixed contract：12 腿 + 6 臂位置目标，4 个轮速度目标 |
| reset pose | `dataset_v3_3_l_high_carry`，与既有 fixed-quota M1 链路一致 |
| TCP | `piper_x_link7/link8` 中点，沿用已有世界点任务代理；不是外部标定工具坐标 |

速度/目标范围沿用已有 M1 命令分布：`vx=[-0.09,0.145] m/s`、`vy=[-0.05,0.055] m/s`、`wz=[-0.17,0.18] rad/s`；vx 独立采样 15% 停止、35% 后退、50% 前进。M 的初始 TCP heading-frame 偏移为 `x=[0.12,0.36]`、`y=[-0.20,0.20]`、`z=[-0.15,0.15] m`，采样后固定在世界系。原有命令的目标高度限制仍生效。命令范围含低速、反向和横向请求，不能只用总 return 判断前进能力。

## Reward v1

唯一公式来源：`tasks/m1_reward.py::m1_reward_terms`。真实端 RewardManager 与想象端调用同一函数，两端分别只乘一次 `dt=0.02`。下表为每秒 reward rate，不能再次手动乘 dt。L/M 分支由显式命令标志选择，不由跟踪误差改变任务类别。

| 项 | 权重 / 核 | 目的 |
|---|---|---|
| L 水平速度 | `3 exp(-||v_xy-c_xy||² / 0.15²)` | 使用物理机体速度；不硬编码四轮正负号 |
| L yaw rate | `1 exp(-(wz-cwz)² / 0.25²)` | 独立跟踪转向 |
| M 粗距离 | `2 exp(-d² / 0.50²)` | 较远目标仍有连续梯度 |
| M 细距离 | `1 exp(-d² / 0.08²)` | 近目标精度 |
| M 有符号进度 | `0.25 tanh((d_prev-d_next)/(dt*0.5))` | 接近为正，远离为负；不是只奖励正进度 |
| M 到点低速 | `1 exp(-d²/0.08²) exp(-tcp_speed²/0.10²)` | 避免高速扫过目标；不是一次性成功脉冲 |
| upright / height | 各 `+0.5` 平滑核 | 直立和默认机体高度 |
| 竖直速度 / roll-pitch 角速度 | `-0.2 vz²` / `-0.05 ||w_xy||²` | 抑制跳动、晃动 |
| 腿默认姿态 / L 臂 carry | `-0.02` / `-0.01`，位置误差平方和 | 控制不必要姿态漂移；M 不受臂 carry 惩罚 |
| 动作变化 | `-0.01 mean((a-a_prev)²)` | 平滑控制 |
| 软关节限位 | `-1`，越界距离平方和 | 排除轮角位置的限位误惩罚 |
| 关节速度 / 加速度 | `-1e-5 mean(qd²)` / `-1e-7 mean(qdd²)` | 真实/想象均用控制周期有限差分 |
| 控制量 surrogate | `-0.005 mean(a²)` | 小权重控制努力；不是实际力矩、功率或能量 |
| M 底盘速度 | `-0.1 (||v_xy||²+wz²)` | 鼓励近点操作时稳定，仍允许需要的机体协同 |

任务项最大值约为 L=4、M=4.25，加上公共姿态项约 1。初始权重让任务跟踪处于主要量级，正则以有限偏差为代价约束解。实际平衡仍需要检查 TensorBoard 的 `m1/reward_real/*` 和 `m1/reward_imag/*`，不能仅凭权重绝对值下结论。

到点保持的训练奖励是连续的；评测成功定义沿用命令阈值：TCP 距离 `<0.08m`、机体没有触发高度/倾倒失败，连续 0.5 秒。评测另外报告真实终点距离。训练目标仍为 SuperMimic 的折扣累计奖励与 Critic `V_H` bootstrap，没有另加 MPC 式有限时域终端距离约束，也不保证轨迹终点必达；这是明确保留的基线定义。

## 默认训练配置和资源

物理 `0.005s × decimation4 = 50Hz`，episode 10 秒。WM logical batch 128 / micro 32、每事件 4 次拟合，Actor logical batch 128 / micro 32；UTD 0.4。先执行 300 个 WM-only 更新事件，Actor LR 再用 500 次更新 ramp。Actor imagination H=8，即 0.16 秒；Critic 提供更长价值预测。网络保留仓库默认宽度，学习率 `2e-4`，本次按墙钟预算采用 constant LR。

H=8 是保守的实验起点：先前审计发现旧 M1 WM 长时域底盘动作响应可反向，重用该旧 WM 会污染本实验，因此默认从零训练，且没有直接用 H=32。新 WM 的 300 次预热和短 horizon 都不等于真实梯度校准已通过。若新策略反向行驶，需要先对同 checkpoint 检查真实 MuJoCo/WM 的动作响应、探索覆盖及执行方向，再判断纯 reward 的能力边界。

为本机 16GiB 内存环境关闭 `torch.compile`，Replay capacity=256；避免同时生成大量 compiler worker。obs/reset noise 关闭，执行高斯噪声 0.03，DR 开启；真实和想象 Actor 的观测噪声配置一致。

启动器复用 Go2 比较任务的 watchdog 和同一互斥锁，发现其他 GPU 计算/训练进程或可用 RAM <4GiB 时拒绝启动；识别并保留 ToDesk 桌面视频会话，不将其误判为训练任务。运行中每 2 秒检查可用内存、进程组 RSS 和新增 swap；越界只停止本脚本自己的进程组。日志写入 `logs/m1_reward/<tag>/train.log`。不会清理无关训练/桌面进程。

2.5 小时预算由原 `Trainer.run()` 控制，从环境/网络初始化和首次 reset 后开始计算，包含学习、仿真、日志和后续 checkpoint IO；按 log interval 检查，允许少量超时。额外的 worker 总耗时另写 `completion.json`。同名目录已有 checkpoint 时拒绝覆盖，应换 `--tag`。

## 命令

只打印计划（默认，不训练）：

```bash
cd /home/yifeiy/ws_supermimic
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_reward.py
```

正式 2048 环境、各半 L/M、2.5 小时：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_reward.py --launch
```

独立 L/M 消融保持相同预算；注意这两条应分别运行：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_reward.py --task-type L --tag L_2048_2p5h_seed2026 --launch
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_reward.py --task-type M --tag M_2048_2p5h_seed2026 --launch
```

短验收，不是正式训练：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_reward.py --smoke --tag smoke_new --launch
```

续训到新的目录，保留 checkpoint 的物理/奖励/优化器配置，`--hours` 表示本次追加时长：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_reward.py \
  --resume logs/m1_reward/OLD_TAG/ckpt_latest.pt --tag NEW_TAG --hours 0.1666666667 --launch
```

续训恢复 WM/Actor/Critic 与优化器、预热/Actor 更新计数；Replay 和仿真 episode 重新开始。`--num-envs/--task-type/--seed` 在续训时由 checkpoint 配置继承，不用于重新定义实验。

评测已经训练好的模型，默认 16 环境、20 秒、固定 seed，关闭噪声和 DR：

```bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/evaluate_m1_reward.py \
  --checkpoint logs/m1_reward/lm50_2048_2p5h_seed2026/ckpt_latest.pt \
  --out results/m1_reward/lm50_seed123.json
```

固定前进命令或固定世界点，可分别加 `--task-type L --vx 0.12 --vy 0 --yaw-rate 0`，或 `--task-type M --target-world X Y Z`。`X Y Z` 是实际需要的 env-local 世界坐标，并非相对 TCP 偏移。评测使用 pre-reset 终止状态和原命令，完成 episode 的终点/成功率不包含未完成 episode；没有样本的指标写 `null`。

播放（仅使用训练后的 checkpoint；native/viser 沿用已有 mjlab viewer）：

```bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt logs/m1_reward/lm50_2048_2p5h_seed2026/ckpt_latest.pt \
  --num-envs 1 --m1-task L --vx 0.12 --vy 0 --yaw-rate 0
```

M 模式改为 `--m1-task M`，默认每 episode 采样目标；可加 `--m1-target-world X Y Z`。2026-11-05 已补充默认目标显示：红球为目标、蓝球为奖励 TCP、黄箭头为误差，终端打印 XYZ 和距离；可用 `--no-m1-debug-vis` 关闭。播放保留 saved v3 动作/姿态定义，关闭 Replay；不会使用默认 legacy M1 动作映射。目标范围和完整训练设计分析见 [M1 训练设计审计](M1_PIPER_TRAINING_DESIGN_AUDIT_20261105.md)。

## 已完成验收与未完成项

- CPU 语义/回归检查 39 项通过（含新奖励/接口/资源 9 项、原 Go2 比较/步态、M1 pose/control-mode、velocity Goal），见 `logs/m1_reward/contract_tests.log`。
- 4 环境 24 步，96 transitions，5 个 WM 更新事件、3 次 Actor 更新；Actor/Critic/WM checkpoint 保存成功，无 Prior。
- `logs/m1_reward/smoke_final_20261104/smoke_validation.json`：真实/想象逐项奖励最大误差 0、Goal8 最大误差 0、TCP FK/MuJoCo 最大误差 `2.98e-7m`；2 L + 2 M。
- 已从保存的 Actor 运行独立真实仿真评测，短测输出 `eval_smoke.json` 位于 `smoke_validated_20261104`。这轮仅 0.5 秒，没有完整 episode，因此终点/成功率为 `null`，不能解释为训练成果。
- `compileall`、CLI help、默认 plan-only 和 `git diff --check` 已检查。
- 已按用户追加要求完成 2048 环境、两段合计约 21 分钟测试，以及初始/训练后两份真实仿真评测；详见 `M1_REWARD_ONLY_2048_PROBE_20261104.md`。正式 2.5 小时训练于当天 18:01 接续启动，20:32 结束；随后完成独立分项和中间 checkpoint 评测，见 [正式策略评估报告](FINAL_POLICY_EVALUATION_20261104.md)。最终策略严重退化，尚不能部署；尚未进行实机或多 seed 效果验证。

## 如何用结果指导 Prior

先用相同命令分布、真实仿真指标比较，L 看速度 XY/yaw RMSE、正向命令时负 vx 的比例和跌倒率；M 同时看全过程平均距离、终止前距离及 0.5 秒保持成功率。不要用 L/M 混合平均 return 代替这些指标。

如果 L-only/M-only 有效而 mixed 明显退化，优先研究任务冲突和分阶段训练；如果两个版本都反向或 WM 想象改善但真实指标不改善，优先校准 WM 动作梯度。若纯 reward 能完成基本动作而数据 Prior 路径仍差，重点回看 Prior 的数据/动作契约和残差控制。若纯 reward 也很差，单凭该实验不能推断“运动学 Prior 没价值”：探索不足、长程规划、Critic 估值及 WM 导数都可能是共同瓶颈。

## 正式训练接续队列（2026-11-04）

用户授权在 Go2 SuperMimic 完成后立即开始 M1-Piper 纯 reward 正式训练。已于约 17:51 建立独立后台等待进程，PID 446186；每 2 s 检查指定 Go2 启动器的完成状态，待自动导出完成、启动器退出且共享训练锁释放，再调用现有 `scripts/run_m1_reward.py`。若 Go2 失败，队列会记录错误，不将失败当作完成。

M1 参数为 2048 envs、2.5 h、seed=2026、balanced（1024 L + 1024 M），从头训练，不恢复短测权重、不加载 motion prior。仍使用原启动器的资源检查与看门狗，不叠加 GPU 训练。目标目录为 `logs/m1_reward/lm50_2048_2p5h_seed2026/`。

接续状态在 `logs/training_queue/go2_to_m1_20261104/status.json`，队列日志 `watcher.log`、M1 启动器日志 `m1_launcher.log` 位于同目录。M1 训练日志为目标目录下 `train.log`。队列独立于交互终端；等待期间不会占用 GPU。

实际接续：Go2 训练和导出成功，启动器于 18:01:28 记录 completed；队列于同一秒启动 M1。M1 启动器 PID 449476、训练进程 PID 449478，GPU 上仅有该训练进程。初检到 it 100，已完成 24 次 WM 更新事件，单步/八步 NSE 约 0.0134/0.0523；仍在 300 次 WM 预热中，尚未开始 Actor 更新。已检查保存配置与上述参数一致；进程 RSS 约 4.1 GiB、可用 RAM 约 6.4 GiB，未触发资源保护。预计约 20:32 结束，以实际预算检查与存盘耗时为准。

最终已正常结束：5400 步、11,059,200 条真实转移、2144 次 WM 更新事件、1844 次 Actor 更新，含初始化实际 9038.099 s。后续评测表明 balanced L/M 跌倒率约 99.6%，M 保持成功率 0；相同 seed 的15/60分钟/最终 checkpoint 对比显示从静止到反向再到失稳的退化。训练进程正常结束不代表策略性能通过，详情和改进建议见正式评估报告。
