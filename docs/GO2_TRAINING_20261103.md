# Go2 对角步态与滑脚约束续训：3072 环境、2.5 小时

更新：用户于2026-11-03 11:31要求停止，进程已退出；实际约2.37小时，评测使用最后完整保存的iteration34000。最终评测与Play命令见 [最终评测](GO2_FINAL_EVALUATION_20261103.md)。下文保留原始启动配置与预计时间。

2026-11-03。按用户要求启动后台训练。当前是启动记录，最终效果需要训练结束后的固定命令评估和步态指标对照。

## 训练选择

从 `logs/phase3_training/go2_velocity_deployable_v2_20261102/ckpt_latest.pt` 的 iteration 17000 续训，恢复 Actor、Critic、World Model、优化器和课程状态。源课程已达到 Stage C，保留自动课程；Replay 重新采集。

源 checkpoint SHA256：`e21a0ae14ffe58ffbf7aaa9bea4cc34fbce190f40371986f95e07e2767b1cd3b`。

选择依据为 Git `3638d9b` 的速度目标改进、原稳定训练的实际 `config.json` 与 `eval_latest.json`，以及已实现的 [步态奖励](GO2_GAIT_REWARD.md)。原策略在固定命令前进 0.3/0.5 m/s 时，平均实测速度约 0.310/0.506 m/s，跌倒率为 0；因此从已会行走的策略改进步态。

| 设置 | 值 |
|---|---|
| 环境数 | 3072 |
| 停止条件 | 本次训练循环运行 2.5 小时后自动保存并停止，每 50 iteration 检查；环境初始化时间另计 |
| 任务 | Go2 `velocity_command`，Direct Actor，tanh 动作 |
| 本体历史 | 5 帧，沿用源网络结构 |
| 步态系数 | 抬脚 `0`；对角协调 `-0.15`；滑脚 `-0.20` |
| WM | 2048×3，8-step window，每事件 4 个更新 batch |
| Actor | 2048/1024/512/256，32-step imagination |
| 学习 batch | WM/Actor 逻辑 batch1024，microbatch512 累积，降低峰值显存需求 |
| 探索 | Gaussian std0.1；25% 环境 OU，sigma0.3、theta2 |
| 噪声与随机化 | 观测/reset 噪声关闭，动作探索与 domain randomization 保留 |
| 学习率计划 | 保留原 320000 iteration 的 cosine schedule，从恢复 iteration 继续 |
| 保存频率 | 每 2000 iteration；达到时间上限时另存 `ckpt_latest.pt` |

新增步态奖励同时用于真实环境与 WM imagination。对角项是几何软约束，不保证固定频率和严格交替；实际改善需结合速度误差、跌倒、滑脚和对角支撑比例判断。

## 运行材料

运行目录：`logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/`。

- `console.log`：后台训练日志。
- `command.sh`：完整执行命令，使用 `/home/yifeiy/miniconda3/envs/supermimic/bin/python`。
- `config.json`：Trainer 实际生效配置；`planned_config.json` 为解析后的启动配置。
- `launch_metadata.json`：源 checkpoint 哈希、Git 基线及启动约定。
- `source_snapshot.tar.gz`、`source_changes.patch`：训练时 Python 源码与工作区差异；步态实现当时尚未提交。
- `tb/`：TensorBoard。
- `launcher.pid`、`started_at.txt`：后台进程与启动时间。
- `exit_code`、`finished_at.txt`：正常退出或程序报错时由启动脚本写入。

后台进程独立于交互会话。尚不能把进程已启动视作训练质量或步态效果通过。

## 启动前验证

在 Conda `supermimic`、CPU 下运行 Go2 gait、velocity training、velocity goal path、phase3 regularization 与 action contract 测试：**26 passed**。CLI 解析及实际配置核对通过，恢复日志确认 iteration17000。没有使用 `.venv`。

## 首批更新核对

2026-11-03 09:09:13（Asia/Shanghai）启动后台进程：launcher PID2240428，训练 Python PID2240431。首次编译后已推进到 iteration17200，约 1.8～1.9 iteration/s，采样显存约 5.3 GB。WM、Actor、Critic 更新指标为有限值，真实与想象的对角/滑脚项均非零，抬脚项均为零；详见运行目录 `startup_verification.json`。

episode 评估窗和课程状态来自恢复的 checkpoint，首批日志的速度 RMSE、跌倒率不能直接当作本次训练的新评估结果。未完成 episode 时 `train_ret nan (n=0)` 表示该区间没有 episode return 样本，需结合优化损失区分数值故障。

预计北京时间约 11:40 达到本次时间上限，实际停止时间取决于日志检查间隔。最终效果待训练结束验证。
