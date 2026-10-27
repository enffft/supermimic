# Phase 3：Go2/M1 正则化、World Model 初始化与长程训练准备

本地 Git HEAD 是 `e9fee106b63cf05712c40780d62574cbe5f4d87d`，与审计基线一致。本地工作区开始时该 HEAD 上已有未提交的 `docs/4070_LAFAN1_Q1_TRAIN.md` 和 `.vscode/`；本阶段没有覆盖它们。完成本阶段修改后，新增/修改文件为：

- `src/supermimic/tasks/velocity_reward.py`
- `src/supermimic/tasks/chase_reward.py`
- `src/supermimic/tasks/world_point_reward.py`
- `src/supermimic/imagination/tasks.py`
- `src/supermimic/envs/tracking_env_cfg.py`
- `src/supermimic/envs/multirobot_env_cfg.py`
- `src/supermimic/envs/mdp/commands.py`
- `src/supermimic/config.py`
- `src/supermimic/training/loop.py`
- `tests/test_phase3_regularization.py`

Go2 Chase 新增 `joint_torque=-2.0e-5` 和 `joint_acceleration=-2.5e-7`，系数来自 `Zhengtianjin/go2-supermimic` 的 Reward Rate 约定。真实路径读取 `robot.data.qfrc_actuator` 和 `robot.data.joint_acc`，并强制检查它们是按 RobotSpec 排列的 12 维；想象路径使用实际执行动作的 clip 后值、Go2 的 `q0+0.25*a_exec`、`Kp=40/Kd=1` 和 HIP/THIGH/CALF 限幅。想象加速度使用控制周期 `0.02 s` 的速度差分。奖励率由环境统一乘一次 dt。Chase 原有任务项和 6D Goal 未改变。

M1 World Point 新增 `joint_velocity=-1e-6`、`joint_acceleration=-1e-7`，系数来自 `M1_perception_rl_loco`；加速度真实和想象都使用 0.02 秒状态差分，MuJoCo 加速度保留为诊断数据。新增 `leg_action_magnitude=-0.002`，只作用于 action 顺序中的 12 个腿部位置通道 `(0,1,2,4,5,6,8,9,10,12,13,14)`；轮子是 `(3,7,11,15)`，Piper 是 `(16..21)`。三项都可由 `--env.*` 覆盖，腿部幅度项设为 `0` 即可关闭。M1 的 Kp/Kd、力矩限幅、轮速配置、Piper 配置、URDF、默认姿态和 FK 没有改动。新增奖励仍保持可微，Actor BPTT 阶段冻结 World Model 参数。

`--init-wm-from` 现在是机器人无关的严格检查：Go2 的 `velocity_command`/`chase_ball` 之间允许复用，M1 只允许 `world_point_reach`；检查 robot、任务族、URDF SHA256、执行器版本、关节顺序、动作维度、StateSpec、控制 dt、World Model hidden widths、终止语义和归一化设置。它只加载 `wm` 和相关元数据，不加载旧 Actor、Critic、优化器或 Replay。M1 预训练 WM 会跳过机械的 200 次 WM-only 预热，但 Replay 未有可采样的完整 8-step 窗口前不会进行 WM/Actor 更新；之后主循环仍按 `updates_per_env_step=0.4` 持续在线更新 WM。无论 `record_transitions` 是否开启，奖励都会使用本地状态差分；Replay 开启时优先采用其 episode 内前一状态，避免跨 episode 加速度尖峰。

关键 TensorBoard 路径包括：`go2/chase/tag_rate`、`go2/chase/mean_ball_distance`、`go2/reward_real/joint_torque`、`go2/reward_real/joint_acceleration`、`go2/reward_imag/joint_torque`、`go2/reward_imag/joint_acceleration`、`go2/joint_torque_rms`、`go2/joint_acceleration_rms`、`go2/torque_proxy_rms_error`、`go2/acceleration_fd_rms_error`、`go2/action_clip_fraction`，以及 M1 的 `m1/reward_real|reward_imag/{joint_velocity,joint_acceleration,leg_action_magnitude}`、三组动作均值/饱和率和腿部执行器 RMS/限幅率。`go2/chase_timeout_success_fraction` 已改为真实 `chase_tag` 事件率，不再用 `1-fall_rate` 冒充成功。WM 另记录 `wm/pretrained_initialized`、`wm/pretrained_source_iteration`、`wm/optimizer_steps`、`wm/updates_per_env_step_actual`。

本机 `supermimic` conda 环境验证结果：

- `pytest -q tests/test_phase3_regularization.py tests/test_chase_ball.py tests/test_world_point_reach.py tests/test_velocity_training.py`：**22 passed**。
- 完整 `pytest -q`：**68 passed**；仅有既有 `mujoco_warp` CUDA Graph/弃用警告。
- Go2 `8 env × 10 iter`：预训练 Chase WM 严格加载，Replay/WM/Actor 更新均有限，checkpoint 写入成功。
- Go2 `128 env × 100 iter`：40 次 WM 更新、40 次 Actor 更新，日志中 `wm nse1=0.0050`、`wm nse8=0.0010`，无 NaN。
- M1 `8 env × 10 iter`：预训练 M1 WM 严格加载，奖励项包含三项新增正则，Replay/WM/Actor 更新均有限。
- M1 `128 env × 100 iter`：40 次 WM 更新、40 次 Actor 更新，`wm nse1=0.0016`、`wm nse8=0.0042`，无 NaN。
- 对两个 128×100 checkpoint 做了 WM 参数前后对照：Go2 最大参数变化 `0.00297`，M1 最大参数变化 `0.00316`，且参数均有限，证明初始化后 WM 仍在线更新。
- 零动作 M1 50 控制步：平均 base height `0.3665 m`，末步 `0.3641 m`，实际力矩 RMS `10.58 Nm`，观测到的 180 Nm 限幅率为 `0`。这支持基础站立通过；尚未把随机 Actor 的对照独立评估写成正式门禁。
- 当前机器 CUDA 驱动为 12.2，`mujoco_warp` 因此关闭 CUDA Graph；上述 smoke 仍成功完成。未宣称策略已经训练成功。

本地可用的严格初始化来源及 SHA256：Go2 Chase `logs/phase2_training/go2_chase_initwm_3072_b1024_20261027/ckpt_latest.pt`，SHA256 `f11b686b355f1f703383554bc72f92b80e6c571baa13caa0902458e3854ec98e`；M1 World Point `logs/phase2_training/m1_world_point_near_128_15m_20261027/ckpt_latest.pt`，SHA256 `845422a1e942d55dcb71f24434d6b18595c50865d16e9982608cdc6c9afe8bfb`。两者都已在 smoke 中严格加载。

以下命令使用主机上的 `supermimic` conda 环境和仓库已有 `sm-train` 入口。`--schedule.total-iterations 320000` 是足够大的迭代上限；`--schedule.max-hours 4.5` 是最长墙钟时间，不保证最短运行时间。默认的 32-step imagination、8-step WM window、`updates_per_env_step=0.4`、10 个 WM batches/update 和 cosine 学习率不在命令中重复改写，命令中的 reward 系数、batch、env 数、初始化来源、日志目录和 checkpoint 间隔是显式设置的。

Go2 正式训练：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate supermimic
sm-train \
  --run-name phase3_go2_chase_reg_3072_4h_20261027 \
  --log-root logs/phase3_training \
  --device cuda:0 \
  --env.robot-name go2 --env.task-name chase_ball --env.num-envs 3072 \
  --env.velocity-wm-batch-size 1024 --env.velocity-actor-batch-size 1024 \
  --init-wm-from logs/phase2_training/go2_chase_initwm_3072_b1024_20261027/ckpt_latest.pt \
  --env.chase-joint-torque-weight -2e-5 \
  --env.chase-joint-acceleration-weight -2.5e-7 \
  --schedule.total-iterations 320000 --schedule.max-hours 4.5 \
  --schedule.checkpoint-interval 4000 --schedule.log-interval 50
```

中途恢复、Go2 64/128 episode Stage 0、Stage 1/2 评估和 Play：

```bash
conda activate supermimic
sm-train --run-name phase3_go2_chase_reg_3072_4h_20261027 --log-root logs/phase3_training \
  --device cuda:0 --env.robot-name go2 --env.task-name chase_ball --env.num-envs 3072 \
  --resume logs/phase3_training/phase3_go2_chase_reg_3072_4h_20261027/ckpt_latest.pt \
  --schedule.total-iterations 320000 --schedule.max-hours 4.5
python scripts/evaluate_go2_chase.py --checkpoint logs/phase3_training/phase3_go2_chase_reg_3072_4h_20261027/ckpt_latest.pt --num-envs 64 --episodes 64 --target-stage 0 --out logs/phase3_training/go2_stage0_64.json
python scripts/evaluate_go2_chase.py --checkpoint logs/phase3_training/phase3_go2_chase_reg_3072_4h_20261027/ckpt_latest.pt --num-envs 128 --episodes 128 --target-stage 0 --out logs/phase3_training/go2_stage0_128.json
python scripts/evaluate_go2_chase.py --checkpoint logs/phase3_training/phase3_go2_chase_reg_3072_4h_20261027/ckpt_latest.pt --num-envs 128 --episodes 128 --target-stage 1 --out logs/phase3_training/go2_stage1_128.json
python scripts/evaluate_go2_chase.py --checkpoint logs/phase3_training/phase3_go2_chase_reg_3072_4h_20261027/ckpt_latest.pt --num-envs 128 --episodes 128 --target-stage 2 --out logs/phase3_training/go2_stage2_128.json
python scripts/play.py --ckpt logs/phase3_training/phase3_go2_chase_reg_3072_4h_20261027/ckpt_latest.pt --device cuda:0 --num-envs 1 --viewer native
```

M1 近目标正式训练：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate supermimic
sm-train \
  --run-name phase3_m1_world_point_near_reg_128_4h_20261027 \
  --log-root logs/phase3_training \
  --device cuda:0 \
  --env.robot-name m1_piper --env.task-name world_point_reach --env.num-envs 128 \
  --env.m1-world-point-wm-batch-size 256 --env.m1-world-point-actor-batch-size 256 \
  --env.m1-fixed-target-stage 0 \
  --init-wm-from logs/phase2_training/m1_world_point_near_128_15m_20261027/ckpt_latest.pt \
  --env.m1-joint-velocity-weight -1e-6 \
  --env.m1-joint-acceleration-weight -1e-7 \
  --env.m1-leg-action-magnitude-weight -0.002 \
  --schedule.total-iterations 320000 --schedule.max-hours 4.5 \
  --schedule.checkpoint-interval 4000 --schedule.log-interval 50
```

M1 Resume、近/中/远目标评估和 Play：

```bash
conda activate supermimic
sm-train --run-name phase3_m1_world_point_near_reg_128_4h_20261027 --log-root logs/phase3_training \
  --device cuda:0 --env.robot-name m1_piper --env.task-name world_point_reach --env.num-envs 128 \
  --resume logs/phase3_training/phase3_m1_world_point_near_reg_128_4h_20261027/ckpt_latest.pt \
  --schedule.total-iterations 320000 --schedule.max-hours 4.5
python scripts/evaluate_m1_world_point.py --checkpoint logs/phase3_training/phase3_m1_world_point_near_reg_128_4h_20261027/ckpt_latest.pt --num-envs 128 --episodes 128 --target-stage 0 --out logs/phase3_training/m1_near_128.json
python scripts/evaluate_m1_world_point.py --checkpoint logs/phase3_training/phase3_m1_world_point_near_reg_128_4h_20261027/ckpt_latest.pt --num-envs 128 --episodes 128 --target-stage 2 --out logs/phase3_training/m1_far_128.json
python scripts/play.py --ckpt logs/phase3_training/phase3_m1_world_point_near_reg_128_4h_20261027/ckpt_latest.pt --device cuda:0 --num-envs 1 --viewer native
```

仍未解决的问题是 4 小时正式训练尚未运行，策略成功率、长期动作饱和趋势和真实/想象力矩代理误差需要由用户执行正式训练后根据 TensorBoard 与 MuJoCo 评估决定；当前也没有把随机 Actor 对照纳入自动门禁。Go2 的当前初始化检查要求旧 checkpoint 的 RobotSpec/URDF/执行器元数据严格一致，缺失或不兼容时会明确拒绝，不会静默 `strict=False`。

交付状态：

```text
[Go2]
- Reward 正则化：完成
- Real/Imagine 对齐：通过
- WM 权重初始化：通过
- WM 持续更新：通过
- Smoke：通过
- 长训命令：已给出（由用户手动执行）

[M1]
- Reward 正则化：完成
- 动作幅度约束：完成
- 执行器配置未改动：确认
- 默认姿态零动作测试：通过
- WM 权重初始化：通过
- WM 持续更新：通过
- Smoke：通过
- 长训命令：已给出（由用户手动执行）
```
