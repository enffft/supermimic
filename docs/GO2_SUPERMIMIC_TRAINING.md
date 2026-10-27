# Go2 Blind Velocity Tracking：实现与短训记录

## 2026-10-26 后续试验：直接速度目标、3072 环境与 Play 评估

本节记录本轮最新实现和评估；下面“4096 环境短训”是此前的独立试验，不能当作本轮 checkpoint 的结果。

### 本轮配置与 checkpoint 初始化

- Go2 Actor 使用直接目标输入：420D proprioception 与 3D `[v_x, v_y, yaw_rate]` 分别按既有 scale 处理后拼接，Decoder 输入为 423D；不经过 motion encoder。G1 和 M1 仍使用原来的 encoded 路径。
- 仅从 `logs/go2_blindlocomotion_resume_15m_b3072_20261026/ckpt_latest.pt` 严格载入兼容的 World Model。源 checkpoint SHA-256：`e1985fa7ba3fa67b10646a5b30c070d10803efb54f60d48ae405c87d8cc1d01e`。Actor、Critic 和优化器均重新初始化；Replay Buffer 重新采集。新 World Model 从在线更新开始继续训练。
- Go2 探索参数：Gaussian 标准差 `0.18`；`30%` 环境使用 OU 噪声，OU `sigma=0.60`、`theta=2.0`。
- 平面速度跟踪权重改为 `3.0`，yaw-rate 跟踪权重改为 `1.2`，其它奖励项、Termination、课程和 32 步 imagination 保持现有实现。
- 训练使用 `3072` 个环境、Replay 容量 `256/env`、World Model `8` 步窗口和 batch `128`、Actor batch `128`、imagination horizon `32`、真实数据 warmup `40` 个向量化步。配置中的 WM-only 数值是 `200`，但由于初始化了预训练 WM，本次运行的有效 WM-only 更新数为 `0`；WM 与 Actor 在 warmup 后在线交替更新。

### 15 分钟训练结果

运行目录和 checkpoint：`logs/go2_direct_pretrained_3072_15m_20261026/`。训练在约 15 分钟上限正常停止于 iteration `1850`，完成 `724` 次 Actor/WM 更新，未发生 OOM。保存的 `ckpt_latest.pt` 同时包含重新训练后的 World Model、Direct Actor、Critic、优化器和训练状态。实际运行配置及预训练源校验值保存在该目录的 `config.json`。

这次短训**没有学会前进**。终止时课程仍在 Stage 0，训练日志的平面速度 RMSE 约 `0.370 m/s`、平均机身高度约 `0.223 m`、Actor grad norm 约 `0.034`、32 步 survival 为 `1.0`。最后一个日志区间动作裁剪率为 `9.45%`，此前峰值约 `70.9%`；最新跌倒统计窗为 `13/15` 个 episode（样本数较少）。World Model 8 步 NSE 约 `0.0136`。这些数据表明 rollout survival 和动力学拟合不能代替真实速度跟踪验收。因此本 checkpoint 用于查看和诊断，不应视为可用的 Go2 locomotion policy。

### 固定命令仿真评估

使用最终 checkpoint，在无动作噪声、观测噪声和 domain randomization 的 16 个评估环境中，对每个命令运行 500 个控制步（10 秒）。完整逐项数据保存在 `logs/go2_direct_pretrained_3072_15m_20261026/fixed_velocity_eval.json`。

| 固定前向命令 | 实测平均机身前向速度 | 最后 100 步平均速度 | 平面速度 RMSE | 平均/最低机身高度 | 跌倒/超时 |
|---:|---:|---:|---:|---:|---:|
| `0.3 m/s` | `0.0063 m/s` | `-0.00006 m/s` | `0.2953 m/s` | `0.2267 / 0.2213 m` | `0 / 0` |
| `0.5 m/s` | `0.0063 m/s` | `-0.00006 m/s` | `0.4946 m/s` | `0.2255 / 0.2201 m` | `0 / 0` |

评估结果显示机器人基本原地蹲伏，速度命令没有转化为前进；零跌倒只说明这 10 秒没有触发跌倒事件。Play 窗口已用同一 checkpoint、单环境、`vx=0.3 m/s` 启动，MuJoCo 原生 viewer 保持打开供人工观察。

在仓库根目录的 `supermimic` 环境可以复现同一可视化：

```bash
PYTHONPATH=src python scripts/play.py \
  --ckpt logs/go2_direct_pretrained_3072_15m_20261026/ckpt_latest.pt \
  --vx 0.3 --vy 0 --yaw-rate 0 --num-envs 1 --device cuda:0 \
  --viewer native --actual-velocity-arrow --velocity-arrow
```

Play 启动日志确认使用 Go2 URDF、12 个关节、420D observation 和 3D goal。Play 期间动作/观测/reset 噪声、domain randomization 和 Replay 记录关闭。当前窗口由本轮 Play 进程持有；关闭 viewer 窗口或按 `Esc` 即结束该评估进程。

### 本轮验证

- Direct Actor 结构、目标梯度和旧 encoded checkpoint 兼容：10 项 CPU 测试通过；World Model/关节顺序/G1 FK 与解析头回归：13 项通过。
- Go2 奖励逐项真实/解析一致性、Replay 执行动作和 OU 探索诊断通过；16 环境、200 控制步诊断测得 Gaussian 噪声标准差约 `0.180`、OU 稳态标准差约 `0.300`、OU 环境占比约 `0.281`、动作裁剪率约 `0.00038`、真实跌倒率 `0`。此诊断在最终 3072 环境训练前完成，训练中的裁剪率后来明显升高。
- `scripts/play.py` 能加载新的 Direct Go2 checkpoint，也能加载旧 encoded Go2、G1、M1 checkpoint。以上结构/加载验证不等于策略效果通过；固定速度评估明确未通过前进跟踪。

## 状态

Go2 4096 环境短训运行到 900 次向量化迭代后停止。虽然真实仿真跌倒率为 0，想象 rollout 的 32 步末存活率降到 `3.59e-4`，触发本任务给出的停止条件。最新 checkpoint 是迭代 750；没有在迭代 800–900 保存 Actor checkpoint。该实验说明采集与 World Model 链路可以运行，不能证明策略学会了速度跟踪。

原始运行日志和配置在 `logs/go2_blind_short_4096/`；停止诊断在 `logs/go2_blind_short_4096/diagnostic_stop.json`。RTX 4070 SUPER 上这次 Go2 仿真以 4096 环境启动成功；没有记录训练时的峰值显存，因此不报告峰值显存数字。

## 训练链路

`src/supermimic/scripts/train.py::main` 用 tyro 解析 `TrainConfig`，创建 `training/loop.py::Trainer`。Go2 选 `robot_name=go2`、`task_name=velocity_command`：

1. `Trainer.__init__` 从 `RobotSpec` 配置 12 个关节、3 维速度 Goal、Go2 状态归一化和网络输入输出维度。
2. `envs/multirobot_env_cfg.py::make_multirobot_env_cfg` 建立 mjlab/MuJoCo Warp 仿真、`VelocityCommand`、NoisyJointPositionAction、11 个共享奖励项、失败终止条件和 Replay Recorder。
3. `training/loop.py::Trainer.run` 每次迭代先采集一个 4096 环境向量化控制步；更新达到 warmup 门槛后按 UTD=.4 抽取 8 步 Replay Window 并更新 World Model。
4. Go2 再执行 300 个 WM-only 更新事件。此后 `PolicyTrainer.update` 从 Replay Buffer 抽 start states，执行 32 步 differentiable imagination；Actor/Critic 沿用共享 BPTT / Lambda Return 更新。
5. `Trainer.save` 定期保存网络、优化器、随机数状态、课程状态和 OU 噪声状态。`scripts/diagnose_multirobot.py` 可从实际仿真与 checkpoint 生成诊断信息。

主要实现文件：`src/supermimic/training/loop.py`、`src/supermimic/imagination/tasks.py`、`src/supermimic/imagination/rollout.py`、`src/supermimic/imagination/policy_trainer.py`、`src/supermimic/tasks/velocity_reward.py`、`src/supermimic/envs/mdp/commands.py`、`src/supermimic/envs/actions.py`。

## Goal、动作和探索

Actor Goal 为机身坐标下的 `[v_x, v_y, yaw_rate]`，形状 `(N,3)`。Stage A 采样前向速度 `0.15–0.55 m/s`，其中 15% 为静止命令；B 扩展到倒车、侧向和 yaw；C 扩展到 `v_x[-0.4,1.0]`、`v_y[-0.3,0.3]`、`yaw[-0.8,0.8]`。

Go2 动作是 12 个关节位置目标。`NoisyJointPositionAction.process_actions` 保留干净动作 `a`，以 `a_exec=clip(a+noise,-1,1)` 执行并写入 Replay Buffer；25% 环境使用 OU 噪声，其余使用 Gaussian 噪声。OU 状态按 `x←x-θx·dt+σ√dt·N(0,1)` 更新，episode reset 时重置；World Model 训练用执行动作 `a_exec`，Actor observation 和动作惩罚仍用干净动作。

## Reward 实现与系数

真实环境的 11 个 RewardTerm 和想象任务都调用 `tasks/velocity_reward.py::velocity_reward_terms`。真实端传 MuJoCo 测得的 actuator force 与 joint acceleration；想象端从 Actor 动作和预测状态构造 PD 力矩 proxy、并用相邻预测关节速度差分计算加速度。Reward core 返回每秒 reward rate；真实 RewardManager 和 `VelocityCommandTask.reward_terms` 各自只乘一次 `dt=0.02`。

令 `v^b_xy` 为机身坐标平面速度、`ω_z` 为机身 yaw rate、`g^b_xy` 为投影重力水平分量，单步奖励为 `dt` 乘以以下加权和：

| 项 | 实现（括号内为源码系数） |
|---|---|
| 平面速度跟踪 | `2 exp(-||v^b_xy-v_cmd||²/0.25)` |
| yaw 跟踪 | `1 exp(-(ω_z-yaw_cmd)²/0.25)` |
| upright | `0.5 exp(-||g^b_xy||²/0.16)` |
| 垂直线速度 | `-1.0 (v^w_z)²` |
| 横滚/俯仰角速度 | `-0.05 ||ω^b_xy||²` |
| 动作变化 | `-0.03 ||a-a_prev||²` |
| 力矩 | `-1e-5 ||τ||²` |
| 关节加速度 | `-2.5e-7 ||q̈||²` |
| 关节偏离默认姿势 | `-0.02 ||q-q0||²` |
| 关节软限位 | `-1.0` 乘超限量平方和 |
| 低机身高度 | `-5.0 relu(0.22-z)²` |

实现细节见 `src/supermimic/tasks/velocity_reward.py::velocity_reward_terms`、`velocity_reward_total` 和 `go2_velocity_reward_term`。奖励函数通过同一仿真步 cache 一次状态/命令快照，避免 RewardManager 逐项调用时输入不一致。

## Termination、课程与记录

`envs/mdp/terminations.py::free_root_too_low` 在 root 高度低于 0.17 m 时失败；`free_root_tilted` 使用 70° 倾斜阈值。WM 以 `[0.42-z, upright tilt feature]` 构造终止特征。两者都由 `config.EnvCfg` 提供阈值，避免真实失败定义与 WM Termination Head 不一致。

`VelocityCommand.record_real_step` 只从真实 MuJoCo 状态累计 moving/yaw/static RMSE、跌倒、长度和高度。每 256 个 episode 形成一个评估窗：Stage A 要求至少 128 个 moving episode、静止速度 RMSE `<0.12`、移动速度 RMSE `<0.18`、跌倒率 `<0.10`；连续两个窗口通过才升 Stage B。Stage B 还要求至少 64 个 yaw episode、yaw RMSE `<0.20`、4 个方向 bin 全覆盖和至少 2/3 速度幅值 bin 覆盖；连续两个窗口通过升 Stage C。状态及窗口写入 checkpoint。

## 短训配置、结果与限制

本次关键设置：4096 env、Replay 容量 256/env、WM window 8、WM batch 128、每次 WM update 一个 batch、Imagination horizon 32、Actor batch 128、warmup 40 iterations、WM-only 300 update events、Actor ramp 1000 updates、UTD=.4、total iterations 1000、`max_hours=.25`。启动配置完整保存在 `logs/go2_blind_short_4096/config.json`。

| 迭代 | 记录 |
|---:|---|
| 500 | 真实 episode return `26.58`；此时为早期行为，不能视作已学会 |
| 900 | WM NSE 1-step `0.01884`，8-step `0.03750`；term BCE `0.25196`，term label ratio `0`，term probability mean `0.22272` |
| 900 | 真实跌倒率 `0`，curriculum Stage `0`；imagined `d_hat` 均值 `0.21980`，H 步末 survival `0.0003585` |
| 900 | Actor ramp scalar `0.0054`；checkpoint 仍为 iteration 750 |

900 次向量步约为 3.69M 个真实环境 transition；运行约 7.1 分钟后被低 imagined survival 停止，未等满 15 分钟，也没有继续同一失败实验。对应约 8.7k environment transitions/s 是按总 transition / 墙钟时间估算。TensorBoard event 包含完整时间序列。

## 验收记录

- Go2 16-env 站立、真实奖励/解析奖励一致性、OU 和 Replay action 检查通过；Go2 reward parity 最大绝对误差约 `1.9e-9`。
- `tests/test_multirobot_acceptance.py -k 'not m1_eight_step_wm_backward_and_32_step_actor_gradient'`：6 passed，1 deselected。
- `tests/test_velocity_training.py`：4 passed。
- `compileall` 通过。
- 4096-env Go2 训练成功启动并运行到 iteration 900；低 imagined survival 是明确停止条件。Actor 仅开始 ramp，未完成有意义的策略学习验证。

## 重现命令

在 `supermimic` Conda 环境、仓库根目录运行：

```bash
PYTHONPATH=src python -m supermimic.scripts.train \
  --run-name go2_blind_short_4096 \
  --env.robot-name go2 --env.task-name velocity_command \
  --env.num-envs 4096 --env.replay-capacity 256 \
  --wm-trainer.window 8 --wm-trainer.batch-size 128 \
  --wm-trainer.batches-per-update 1 --wm-trainer.compile None \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 128 \
  --policy-trainer.compile None \
  --schedule.total-iterations 1000 --schedule.warmup-iterations 40 \
  --schedule.wm-only-iterations 300 --schedule.actor-ramp-iterations 1000 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 250 \
  --schedule.max-hours 0.25
```

不要从迭代 750 checkpoint 直接忽略停止原因继续同一配置。下一轮 Go2 训练前，应先校准/改进 Termination Head 的分布外概率，使 Stage A 真实 episode 中的失败标签、预测终止概率和 imagined survival 相容，再用 4096 env 复测；本次结果不支持长训。
