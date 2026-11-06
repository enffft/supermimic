# 阶段三：正式对比启动、资源保护与统一评测

日期：2026-11-04。最初仅完成代码、短测和部署检查；随后用户启动了正式顺序对比。PPO 已完成 2.5 h，SuperMimic 首次启动失败后已按用户要求单独补跑，详见文末记录。

## 资源事件与后续限制

本机物理内存约 16 GiB。过程中观察到 M1 扫描进程 RSS 约 9.3 GiB，桌面和多项其他程序同时运行，多进程进入 D 状态，符合严重内存/磁盘等待现象；后来主机 uptime 重新从几分钟开始。没有足够内核日志将卡死唯一归因到某一进程，不能仅凭 GPU 占用低就认定安全。

目前已结束本任务先前启动的短测/编译/验证进程，不会自动继续大规模测试。新启动器 `scripts/run_go2_comparison.py` 使用 `baselines/resources.py`：

- 两种算法**顺序**执行；通过文件锁避免本启动器并发跑两份训练。
- 每个训练开始前检查 GPU compute 进程及训练/评测 Python 进程，存在则拒绝启动；这也覆盖扫描父进程在两个 GPU 子任务之间的空隙。可用 RAM 不足 4 GiB 也拒绝。
- 每 2 s 检查本任务整个进程组 RSS、系统可用 RAM 和新增 swap。
- RAM 可用低于 1.5 GiB、本任务 RSS 高于 8 GiB，或新增 swap 超过 1 GiB 时，终止本任务进程组；8 s 仍未退出则 SIGKILL，保留已有 checkpoint 和日志。
- Ctrl+C / SIGTERM 会清理该启动器正在运行的训练子进程组。不会全局 pkill Python，也不会清掉其他任务。

这些检查可以缓解再次卡死的风险，但不能阻止别的程序在检查间隙突然申请大量内存。正式跑之前应先结束不需要的扫描/评测和占内存应用。短测的约 24 s 测速发生在资源事件之前，不作为稳定长训吞吐保证。

## 先查看命令，不训练

```bash
cd /home/yifeiy/ws_supermimic
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_go2_comparison.py
```

默认仅打印计划。两者读取相同参考配置，但 `resume=None`、`init_wm_from=None`，SuperMimic 重新初始化 WM/Actor/Critic，PPO 也重新初始化；均 seed=2026、3072 envs、2.5 h。从 A 课程开始。旧 checkpoint 不提供任何训练权重。

## 将来正式从头各训练 2.5 h

```bash
cd /home/yifeiy/ws_supermimic
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_go2_comparison.py --algorithm both --launch
```

先 PPO 2.5 h，再 SuperMimic 2.5 h；每种算法训练完成后自动导出到 `~/rl_sar/policy/go2/{ppo_velocity,supermimic_velocity}`。需要只跑 PPO 时：

```bash
/home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/run_go2_comparison.py --algorithm ppo --launch
```

日志在 `logs/go2_comparison/{supermimic,ppo}_2p5h_seed2026/train.log`。终端关闭后继续运行可自行在上面命令前加 `nohup` 并重定向日志。重新做实验使用 `--tag 新名字`；同目录已有 checkpoint 会拒绝覆盖，也不会隐式续训。

计时包括学习循环中的仿真、编译、更新和存盘。PPO 在 update 结束边界停止；SuperMimic 在 50 次真实环境步的日志边界停止，因此要报告实际用时，不声称两者严格等于 9000.000 秒。每种算法的 `progress.json` 保存真实转移数和最新 checkpoint，PPO 的 `comparison_config.json` 还保存上游 SHA、参考配置和 runner 参数。

## 训练后统一评测

复用已有评测脚本；两种算法的导出包都含相同推理 schema，避免另写一套 PPO 指标计算。以下命令在正式训练/导出完成后运行：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/evaluate_go2_velocity.py \
  --checkpoint /home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt \
  --num-envs 64 --episodes-per-command 64 \
  --output logs/go2_comparison/supermimic_evaluation.json

PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/evaluate_go2_velocity.py \
  --checkpoint /home/yifeiy/rl_sar/policy/go2/ppo_velocity/policy_inference.pt \
  --num-envs 64 --episodes-per-command 64 \
  --output logs/go2_comparison/ppo_evaluation.json
```

两条评测也顺序运行。固定同一组命令、同一 seed，关闭 DR 和执行噪声。输出跟踪 RMSE、fall rate、动作饱和、足底滑移、对角支撑等。当前脚本固定五类命令：站立、0.3/0.5 m/s 前进、0.3 m/s 侧向、0.4 rad/s 转向；它是名义平地比较，不能替代反向/混合速度、扰动和实机测试。

组会建议同时给出：实际训练墙钟时间、真实转移数、最终课程阶段、跟踪误差、跌倒率和同指令视频。单 seed 的结果只是本次实验观察；不要把旧续训模型和新 PPO 当成严格算法胜负，也不要把真实/想象 rollout 数混为一谈。

## 三阶段交付索引

- [阶段一：来源、对齐、短测](GO2_COMPARISON_STAGE1_PPO_20261104.md)
- [阶段二：导出协议、C++ 与 sim2sim、play/实机命令](GO2_COMPARISON_STAGE2_DEPLOY_20261104.md)
- 本文：资源保护、正式命令和统一评测。

本任务没有提交或推送代码。下载的外部仓库仅本地保留；已有 M1 等工作区修改未回退、未打包提交。

## 2026-11-04 正式运行及 SuperMimic 补跑

PPO 已完成：`ppo_2p5h_seed2026/completion.json` 记录实际训练 9004.077 s、759 次更新、每环境 24,288 步，共 74,612,736 条真实转移；最终 checkpoint 为 `model_758.pt`，已导出至 `~/rl_sar/policy/go2/ppo_velocity/`。训练日志的误差不作为独立评测结果。

顺序启动中的第一次 SuperMimic 在编译初始化阶段退出，没有留下 checkpoint；现有日志不足以确定原因，不能将编译资源压力认定为已证实根因。原目录已归档为 `supermimic_2p5h_seed2026_failed_init_20261104_153114`。

用户要求接续后，于北京时间 15:31 左右仅重新启动 SuperMimic。由于没有可恢复 checkpoint，本次从头训练，仍为 seed=2026、3072 envs、2.5 h，奖励和算法配置不变，PPO 不重跑。启动器 PID 416016、训练 PID 416019；独立会话运行，终端关闭不会中断。预计约 18:02 结束，具体以预算检查及导出耗时为准。

本次启动器固定 OMP/MKL/OpenBLAS、TorchInductor 编译及构建线程数为 1，以限制并发资源开销；仍启用原资源保护。新增 `launcher_status.json` 记录训练、导出、完成或失败状态，子进程/保护器异常写入 `train.log`，导出也受资源保护并检查退出码。对应 CPU 测试 7 项通过。

补跑初检已到 it 500、184 次 WM 更新，单步/八步 NSE 分别约 0.0076/0.0139，处于 200 次 WM 预热阶段；预热期间 `imag_ret nan` 表示尚无 Actor 想象回报统计，不能据此判定数值发散。进程 RSS 约 2.6 GiB，GPU 显存约 1.3 GiB，主机可用内存约 8 GiB。这里只确认训练链路已实际更新，尚不能判断最终策略质量。

随后确认已到 it 650、244 次更新事件，200 次 WM 预热完成，Actor 已参与后续 44 次更新；`imag_ret=0.147`、`surv=0.971`，均为有限值。首次 Actor 编译使 it 500–550 区间耗时约 137 s，之后恢复约 2.1 it/s；进程 RSS 约 3.3 GiB。该检查确认 WM 与 Actor 两条更新链路均已跑通，不代表独立评测性能。

日志为 `logs/go2_comparison/supermimic_2p5h_seed2026/train.log`，PID/归档路径另存于 `logs/go2_comparison/supermimic_restart_latest.json`。成功训练后启动器自动导出至 `~/rl_sar/policy/go2/supermimic_velocity/`；补跑完成前，该目录仍可能是旧模型。

补跑最终于 18:01 左右达到 2.5 h 预算：it 18600、7424 次 WM 更新事件，最终日志 `imag_ret=1.973`、`surv=0.998`（训练内指标）。checkpoint 已保存，自动导出成功，`launcher_status.json` 于 18:01:28 标记 completed，部署目录现已更新为本次模型。尚未进行本轮独立性能评测。按用户新增安排，接续队列在该启动器退出后于同一秒启动 M1-Piper 纯 reward 训练；两任务未重叠，详情见 `M1_REWARD_ONLY_BASELINE_20261104.md`。

晚间已完成正式模型独立评测和部署复验，最新结果见 [正式策略评估报告](FINAL_POLICY_EVALUATION_20261104.md)。Go2 两种算法均完成九个指令的名义场景评测；SuperMimic 全部未跌倒，PPO 的纯侧移/原地转向失败。两份导出模型均通过 C++ 一致性检查，原生 MuJoCo 复现相同的成功/失败结论。M1 2.5小时训练已完成，但最终性能严重退化。
