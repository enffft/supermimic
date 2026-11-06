# Go2 速度跟踪：抬腿与对角步态奖励

2026-11-02。本次只实现奖励与 CPU 验证，没有启动训练、Play 或 GPU 仿真。

当前选择：**关闭抬脚高度奖励，开启对角协调和滑脚惩罚**。下述默认值及续训命令已同步；抬脚奖励实现保留供显式启用。

## 对应的已有策略

`logs/phase3_training/go2_velocity_deployable_v2_20261102/` 是现有两小时速度跟踪训练，Actor 使用 5 帧本体历史、直接速度目标和 tanh 动作。已有 `eval_latest.json` 中，前进命令 0.3/0.5 m/s 的平均实测速度分别约 0.310/0.506 m/s，64 episode 跌倒率均为 0。旧评估没有足端抬升指标，因此不能从这些数据推断原步态的实际抬脚高度。

原有奖励没有直接约束抬脚高度和四条腿的配合，低抬腿步态也可以得到较好的速度跟踪奖励。这是本次改动的动机，尚未通过新训练验证效果。

## 参考与实现

主要参考主机上 `ws_parkour/Unitree-Go2-Adaptive-RL/legged_gym/envs/base/Go2.py` 的 `_reward_foot_clearance`、`_reward_foot_slide` 和 `_reward_foot_mirror`，以及 `envs/go2/go2_config.py` 中对应配置：抬腿项按足端运动加权，并用滑脚及对角腿配合项约束步态。原项目的 body-frame 高度、接触力和系数不能直接用于本仓库的世界模型 BPTT。

新增 `kinematics/go2_fk.py`，按当前 Go2 URDF 计算 FL/FR/RL/RR 四个足端碰撞球中心的位置和世界速度，包含球心偏置、0.022 m 半径、root 旋转及角速度。只读取现有 37D 状态，无需新增 World Model 输出、Actor 输入或 Replay 字段。

`tasks/go2_gait.py` 实现以下奖励率；真实 RewardManager 和想象任务沿用同一个函数，各自只乘一次 `dt=0.02`。

| 项 | 默认系数 | 含义 |
|---|---:|---|
| `feet_clearance` | `0`（关闭） | 可选的足底离地高度奖励；当前不参与优化 |
| `diagonal_gait` | `-0.15` | 惩罚 FL/RR、FR/RL 两组内部离地状态不一致，以及偏离双足支撑的状态；鼓励对角腿配合 |
| `feet_slip` | `-0.20` | 近地面足端的世界水平速度平方均值，惩罚支撑脚滑动 |

关闭的抬腿项保留 **5 cm** 目标和 0.025 m 高斯宽度参数；其系数为 `0` 时，这些参数不影响奖励。只有显式把 `--env.go2-gait.feet-clearance` 设为非零才会启用。

移动门控使用 `sqrt(vx_cmd² + vy_cmd² + (0.25*yaw_cmd)²)`，在约 0.03–0.15 m/s 间逐渐开启。纯转向同样可触发步态激励，零命令关闭抬腿及对角协调项，滑脚惩罚仍有效。

接触判断是足底高度的平滑几何近似；没有使用仿真的硬接触事件或不可微的腾空计时。对角项是软协调约束，没有指定固定频率或保证严格交替，相位和频率由动力学与速度跟踪共同决定。当前只适用于 `z=0` 平地的 `go2/velocity_command` 任务，Chase/G1/M1 奖励未接入此项。

## 后续续训命令（本次未执行）

在仓库根目录使用 Conda `supermimic` 环境。以下保留已有训练的重要设置，从现有 Actor/WM/Critic checkpoint 续训，另写新目录。`--resume` 不会自动把旧 JSON 的所有配置应用到新命令，因此显式保留历史长度、网络、探索和课程设置。恢复后 Critic 需要适应新的奖励。

```bash
conda activate supermimic
PYTHONPATH=src python -m supermimic.scripts.train \
  --run-name go2_velocity_diagonal_slip --log-root logs/phase3_training \
  --resume logs/phase3_training/go2_velocity_deployable_v2_20261102/ckpt_latest.pt \
  --env.robot-name go2 --env.task-name velocity_command \
  --env.num-envs 3072 --env.replay-capacity 256 --env.history-length 5 \
  --env.no-obs-noise --env.no-reset-noise \
  --env.velocity-ou-noise-std 0.3 \
  --env.velocity-stage-a-vx-range 0.25 0.6 \
  --env.velocity-stage-a-static-fraction 0.05 --env.velocity-resample-seconds 5 \
  --env.go2-gait.feet-clearance 0 \
  --env.go2-gait.diagonal-gait -0.15 --env.go2-gait.feet-slip -0.20 \
  --env.go2-gait.clearance-target 0.05 --env.go2-gait.clearance-sigma 0.025 \
  --world-model.hidden 2048 2048 2048 \
  --policy.encoder-hidden 2048 1024 512 256 \
  --policy.decoder-hidden 2048 1024 512 256 \
  --wm-trainer.batches-per-update 4 \
  --schedule.total-iterations 320000 --schedule.max-hours 2 \
  --schedule.checkpoint-interval 4000 --schedule.log-interval 50
```

把 `feet-clearance`、`diagonal-gait`、`feet-slip` 三个系数均设为 `0`，可关闭新增奖励并跳过足端计算，用作原奖励基线。CLI 参数同时传到真实和想象奖励，配置随新 checkpoint 保存。Play 和固定速度评估读取 checkpoint 中的步态配置；缺少这些字段的旧 checkpoint 默认关闭新增项。

## 观测效果

TensorBoard 自动增加 `go2/reward_real/{feet_clearance,diagonal_gait,feet_slip}` 和对应的 `go2/reward_imag/*`。当前 `feet_clearance` 为零；需要同时看速度跟踪、跌倒率和动作饱和率。

`scripts/evaluate_go2_velocity.py` 额外输出：

- `mean_episode_peak_foot_clearance_m`：FL/FR/RL/RR 每条腿的 episode 最大足底离地高度，再对 episode 求均值。
- `stance_slip_rms_m_s`：足底高度小于 1.5 cm 时的滑脚 RMS 速度。
- `diagonal_support_fraction`：恰好一组对角腿支撑时的控制步比例。

这三项排除每个 episode 开始的 1 秒落地稳定阶段。它们用于新旧 checkpoint 的同命令对照，不代表已验证新策略改善。

## 验证

使用 `/home/yifeiy/miniconda3/envs/supermimic/bin/python`，仅运行 CPU 检查：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_go2_gait.py tests/test_velocity_training.py \
  tests/test_velocity_goal_path.py tests/test_phase3_regularization.py \
  tests/test_action_contract.py
```

结果 **26 passed**。覆盖随机姿态/速度下与 MuJoCo URDF 的 `mj_forward` 足端位置及速度一致性、目标高度偏好、对角支撑偏好、滑脚方向、静止/倒退/侧移/转向及零速梯度、真实/想象共享奖励的 dt 缩放、CLI 参数到真实奖励配置的传递。`mj_forward` 只计算 CPU 运动学，没有进行仿真步进或策略训练。

上述续训命令已用同一 Conda 环境的 `tyro.cli(TrainConfig, args=...)` 静态解析通过，确认 checkpoint 文件存在，没有构造 Trainer。新增模块 lint、修改文件的 Python 编译及 `git diff --check` 均通过。新增评估指标尚未运行 GPU 仿真验证。
