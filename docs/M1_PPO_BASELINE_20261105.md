# M1-Piper PPO：沿用纯 Reward SuperMimic 任务

## 实验定义

本轮用于比较：相同 M1-Piper 任务、相同 Actor 和 2.5 小时墙钟预算下，直接由真实交互训练的 PPO 表现如何。PPO 复用已经跑通 Go2 的 `mjlab.MjlabOnPolicyRunner → RSL-RL PPO`，没有重写 PPO、GAE、rollout 或优化器循环。参考开源仓库仍为 [giangdao1402/go2_velocity](https://github.com/giangdao1402/go2_velocity)，本机下载版本 `23a364b1aafc454b5c281b9f115de434e6db2280`；安装包版本和实际 SHA 写入每次运行的 `comparison_config.json`。

SuperMimic 参照为 `logs/m1_reward/lm50_2048_2p5h_seed2026/ckpt_latest.pt`，只读取其中的配置，PPO 不继承任何已训练网络参数。启动时检查当前机器人 XML SHA256、默认姿态和完整动作映射与参照一致。

| 项目 | 本次 PPO / 已有 SuperMimic |
|---|---|
| 任务 | `m1_piper / m1_reward`，不使用运动数据或 Motion Prior |
| 环境、seed、预算 | 2048、2026、各 2.5 h，从零训练 |
| L / M 配额 | 固定 1024 / 1024；reset 保持任务类别 |
| 命令 | 原有 Command7 和 Actor Goal8：机体速度 / 固定世界目标点 |
| Actor | 10 帧 720D proprioception + Goal8，原有固定缩放和 2048/1024/512/256 解码网络，22D tanh 确定性输出 |
| Critic | canonical state57 + Goal8 + normalized xi26 + zero root error4；256/128 SiLU |
| 动作 | 原有 v3：18 个关节位置目标 + 4 个轮速度目标，原有关节顺序、姿态、绝对位置边界和轮速映射 |
| 物理 | 原有 0.005s × decimation4、10s episode、倾倒/高度终止 |
| 奖励 | 原有 `m1_reward_terms` 全部 18 项，RewardManager 中各项只乘一次 dt，无新奖励或权重调整 |
| 随机化 | 原有 DR 开启、obs/reset noise 关闭、执行噪声 0.03 |

PPO 不创建/训练 WM，Actor 由真实 rollout 的 PPO 梯度更新。保留原 Recorder 以获得 reset 之前的物理状态和命令，保证 TCP 进度、速度、加速度等奖励的计算与 SuperMimic 相同；其未用于学习的 WM Replay capacity 从 256 缩至 2，节约显存，PPO 使用上游自己的 rollout storage。

PPO 使用原 Go2 对比链路的参数：rollout 32 步，2048 × 32 = 65,536 transitions/update，64 个 logical minibatches、每个 1024 样本、5 epochs、初始 LR 2e-4、adaptive KL 调整、gamma .99、GAE lambda .95、clip .2、entropy .01、初始 Gaussian std .3。高斯探索是 PPO 算法的一部分，叠加原环境执行噪声；训练中的随机动作可超出 tanh 均值范围，由原有执行层裁剪，确定性评测仅执行均值。原动作历史和正则仍读取原始策略动作。这一探索分布与 SuperMimic 不相同，不能宣称训练轨迹逐条一致。

相同 seed 不保证两种算法采到相同随机轨迹。PPO 每次更新使用的真实样本数、优化 epochs 和 SuperMimic WM/BPTT 更新不可直接等同；主要控制墙钟预算，并记录真实 transitions、steps/env 和更新数。预算从环境/网络初始化完成后开始，到下一次完整 PPO 更新结束时停止，包含训练期间保存/日志开销，允许一个更新的超时；后续评测不计入 2.5 小时。

## 入口与产物

仅打印计划：

```bash
cd /home/yifeiy/ws_supermimic
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_ppo.py
```

正式训练并自动评测、出图：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_m1_ppo.py --launch
```

默认输出目录 `logs/m1_ppo/lm50_2048_2p5h_seed2026/`。同目录非空则拒绝覆盖；再次从零训练需指定新 `--tag`。这不是 SuperMimic/PPO 连跑队列，只启动 M1 PPO。支持原任务的 `--task-type L|M` 独立实验，但此时默认自动 balanced 对比不适合作为混合任务训练对照，应使用 `--skip-eval` 再单独评测。

上游日志的 `0/100000` 和 ETA 按 iteration 上限推算，不代表本次实际时长；本任务另有 2.5 h 墙钟 hook，到期会在完整更新结束后保存退出。

- `train.log`：上游 PPO 日志；每 50 次更新及预算到期保存 `model_*.pt`，含可恢复的上游模型和优化器状态。本入口尚未实现 resume 参数。
- `progress.json`：最近保存时的更新数、真实样本量和墙钟时间。
- `actor_latest.pt`：原始 728D 输入的 TorchScript 确定性 Actor。
- `policy_inference.pt`：转换为仓库现有 play/evaluate 可读取的推理格式；标记 `source_algorithm=PPO`、`inference_only=True`，没有 WM 或 Prior。
- `validation.json`：当前真实奖励积分一致性、固定 L/M 配额、实际训练 Actor 与两种导出的动作误差。
- `completion.json`：训练完成标志；`launcher_status.json` 覆盖训练、评测、报告和失败状态。
- `evaluation_{balanced,forward,manipulation}.json`：同原 SuperMimic 的 64 envs / 30s / seed123 评测；forward 固定 `(0.12,0,0)`，其余沿用原命令分布。评测关闭噪声/DR，终点及成功率只统计完成的 episode，保留 pre-reset 状态。
- `comparison.md` / `comparison.png`：与 `logs/final_evaluation_20261104/m1_*.json` 的已有 SuperMimic 结果自动汇总。短测预算不等时明确标注不可比较算法能力。

查看进度：

```bash
cat logs/m1_ppo/lm50_2048_2p5h_seed2026/launcher_status.json
cat logs/m1_ppo/lm50_2048_2p5h_seed2026/progress.json
tail -n 40 logs/m1_ppo/lm50_2048_2p5h_seed2026/train.log
```

完成后播放 PPO 的前进命令：

```bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt logs/m1_ppo/lm50_2048_2p5h_seed2026/policy_inference.pt \
  --num-envs 1 --m1-task L --vx 0.12 --vy 0 --yaw-rate 0
```

M 任务改为 `--m1-task M`，需要固定世界点可加原有 `--m1-target-world X Y Z`。

## 资源约束和验收

只使用 Conda `supermimic`。与 Go2/M1 原任务共用 GPU 互斥锁及 `run_guarded`：启动前检查其他训练/评测及 GPU 任务，可用 RAM <4GiB 拒绝启动；运行时可用 RAM <1.5GiB、任务进程组 RSS >8GiB 或新增 swap >1GiB 时，只终止自身任务进程组。顺序训练、评测，不开启并行重任务或编译 worker。

已通过配置/奖励/Actor/导出契约及原 M1 reward-only 回归检查 12 项。32 envs / 两次真实 PPO 更新已完成；2048 envs / 一次真实 PPO 更新完成 65,536 transitions，预算终止、保存和导出通过，1024 L + 1024 M。该短测真实 reward dt 积分最大误差 `2.98e-8`，实际训练 Actor / 推理 checkpoint / TorchScript 最大动作误差均为 0；后续复用原脚本进行三项完整评测以验收链路。

三项 64 envs / 30s 评测均已完成，`probe2048_20261105/launcher_status.json` 为 completed；对比表与 PNG 也已生成。短测仅一次 PPO 更新，基本保持静止，M 到点成功率为 0，不作为正式效果结论。

这些检查证明训练与评测链路可运行，不证明正式训练能达到目标。结果需同时看速度 RMSE、倒退比例、L/M 跌倒率和 M 到点保持成功率；静止不跌倒不能代替任务成功。单 seed 结果也不能唯一归因此前 WM/BPTT 的退化机制。

## 正式启动记录

2026-11-05 14:50:57（Asia/Shanghai）已启动后台 launcher，PID 1384754；训练 worker PID 1384764。已看到第一次完整 PPO 更新、65,536 条 transitions 和 `model_0.pt`。训练 worker 使用约 5.7 GiB GPU 显存，主机可用 RAM 约 6.7 GiB，资源保护保持开启。预计约 17:21 完成训练，随后顺序评测和生成报告，以实际状态文件为准。

后台入口日志：`logs/m1_ppo/launcher_lm50_2048_2p5h_seed2026.log`；PID/启动时间及完整命令：同名 `.json`。后台进程独立于当前交互终端。
