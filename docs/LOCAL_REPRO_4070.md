# SuperMimic 本机复现记录：RTX 4070 SUPER

记录日期：2026-10-24（Asia/Shanghai）。仓库算法源码没有修改；本轮新增的仓库变更只有本文件和算法对照文档。运行命令、环境快照、测试输出、训练日志、资源采样和结果文件保存在 `logs/4070_audit/`、`logs/4070/`、`data/`。

## 阶段结论

- RTX 4070 SUPER 12 GB 上已经实际运行原仓库的 G1 MuJoCo Warp GPU 环境、World Model、H=32 imagination/BPTT、Critic、checkpoint、40 条 LAFAN1 评测和 native viewer。
- CUDA 13.0 锁定 wheel 与本机 R535 驱动不兼容。为保留驱动和原算法，单独创建 `supermimic` Conda 环境，使用 PyTorch 2.11.0+cu126 与 Warp 1.18.0 的 CUDA 12 wheel。原始 `pyproject.toml`、`uv.lock`、mjlab commit 和算法代码保持不动。
- `pytest tests -q`：24 passed、56 warnings、0 failed。
- 40 条 LAFAN1 G1 CSV 全部转换为 NPZ，0 失败，总时长 2.45056 h。
- 500 次迭代 H=32 短训中，World Model 1/8-step 误差下降、Actor/Critic 都完成更新、H=32 soft survival 变为非零；真实训练回报仍约 0.39–0.44。40 条 benchmark 全部失败，因此策略短训尚未达到可用跟踪水平。
- 完整网络、H=32 的最大资源试跑为 1024 env / 2048 Actor batch / 2048 WM batch，单次 WM batch，峰值 8,738 MiB，无 OOM。包含默认 10 个 WM batch/update 的候选配置已验证为 1024 env / 1024 Actor batch / 1024 WM batch，峰值 6,175 MiB。长期训练建议先用后者并留出显存余量。
- Native viewer 窗口 `MuJoCo : mjlab scene` 已打开，正在播放 `walk1_subject1` 与参考 Ghost。训练结果差，画面用于验证播放链路，不代表跟踪质量。
- 正式 320,000 iteration 全数据训练尚未启动。按 1024 env 试跑观测到约 5.1 个向量化迭代/秒粗估，训练主体约 17.4 小时；该估算会随系统负载、编译和随机化设置变化。

## Phase 0：硬件与系统审计

| 项目 | 实测 |
|---|---|
| Ubuntu / kernel / 架构 | Ubuntu 20.04.6 LTS / `5.15.0-139-generic` / `x86_64` |
| GPU | NVIDIA GeForce RTX 4070 SUPER，12,282 MiB（12 GB 型号） |
| NVIDIA driver | 535.230.02；`nvidia-smi` 报告 CUDA 12.2 |
| 系统 CUDA Toolkit | `/usr/local/cuda-12.8`，NVCC 12.8.61 |
| GCC / G++ | 9.4.0 / 9.4.0 |
| RAM / swap | 15 GiB / 19 GiB；审计时可用 RAM 约 7.4 GiB |
| 磁盘 | 根分区 916 GiB，总可用约 389 GiB |
| Miniconda | `/home/yifeiy/miniconda3`，Conda 25.9.1 |
| ROS 污染 | 继承的 `PYTHONPATH` 含 ROS Noetic、rl_sar、Livox；`LD_LIBRARY_PATH` 含 ROS 与 CUDA 12.8 |
| 初始 GPU 占用 | 约 603 MiB，桌面/Xorg/GNOME；没有发现训练任务 |

NVIDIA R535 不满足 CUDA 13 驱动要求，不能运行锁定的 `torch 2.14.1+cu130`。没有安装或升级系统驱动。无 `nvcc` 并不妨碍 PyTorch wheel 执行 CUDA；本机也实际存在 CUDA 12.8 Toolkit。兼容栈的 Torch CUDA 运算、反向传播、Warp GPU kernel 和 MuJoCo Warp 环境都已在 GPU 上运行。

MuJoCo Warp/Warp 日志报告运行时 driver 12.2，低于 CUDA Graph 所需 12.4，因此禁用 CUDA Graphs。仿真功能正常，性能可能低于仓库作者在更新驱动/GPU 上的测量。Torch Inductor 还输出 `Not enough SMs to use max_autotune_gemm mode` 警告；编译成功。

审计原始输出见 `logs/4070_audit/nvidia-smi-q.txt`、`os-release.txt`、`uname.txt`、`nvcc-version.txt`、`gcc-version.txt`。

## Phase 1–2：仓库与隔离 Python 环境

- 仓库：`https://github.com/CuboiLeo/SuperMimic-Reproduced`
- Git commit：`d9be385b3eeb13a638fb33fb200d3bfd9be9839f`
- mjlab 固定 commit：`8bb2bc9e187797689ec82c78c35019afd2fd0376`，mjlab 1.6.0。
- 原始 `pyproject.toml` 与 `uv.lock` 未改，工作树算法源码未改。
- 唯一 Python 运行环境为 Conda `supermimic`，Python 3.12.15，解释器 `/home/yifeiy/miniconda3/envs/supermimic/bin/python`。没有创建第二个 uv `.venv`，其他 Conda/Isaac Gym/ROS 环境保留。
- 最终核心版本：PyTorch `2.11.0+cu126`（Torch CUDA runtime 12.6）、Warp `1.18.0+cu12`、MuJoCo `3.16.0`、mujoco-warp `3.16.0`、mjlab `1.6.0`。GPU 为 sm_89，Torch 与 Warp 均能访问。
- 上游 `uv.lock` 锁定 PyTorch `2.14.1+cu130` 和标准 Warp 1.18.0 CUDA 13 wheel。兼容环境在同一个 Conda Python 中安装 cu126 Torch、NVIDIA 官方 cu12 Warp wheel、锁定 mjlab 与其余项目依赖；这是 wheel/runtime 兼容偏离，未改变 Dynamics、Actor、Critic、Reward、Termination 或 BPTT 源码。
- 运行日志中的模型参数量为 World Model 8.82M、Actor 8.89M、Critic 0.235M；与仓库原始网络宽度一致。
- 项目运行/测试需用 `env -u PYTHONPATH -u LD_LIBRARY_PATH`，防止旧 ROS/CUDA 路径污染。没有卸载旧环境、改全局代理、重装 Toolkit 或驱动。
- 完整版本快照为 `logs/4070_audit/conda-environment.yml`、`conda-list.txt`、`pip-freeze.txt`、`runtime-versions.txt`。

Tensor 运算 probe 成功：512×512 CUDA 矩阵乘法及 backward 的 loss/gradient 都有限；Warp kernel 把 `[1,2,3,4]` 映射成 `[2,4,6,8]`。日志见 `logs/4070_audit/gpu_runtime_check.txt`。早期把 Warp kernel 定义在 stdin heredoc 的 probe 因 Warp 需要 inspect 文件源码而报错；将相同 kernel 放入 `gpu_runtime_check.py` 后成功，未涉及仓库代码。

## Phase 1 算法代码审计

完整代码、张量形状、Figure 2 / Algorithm A.1 / Eq. 1–3 映射在 [`SUPERMIMIC_ALGORITHM_MAP.md`](SUPERMIMIC_ALGORITHM_MAP.md)。仓库确实实现完整链路：G1 真实仿真采样、8-step 自回归 World Model、解析 FK/reward/observation、学习 termination、冻结 World Model 权重但保留输入梯度的 Actor BPTT、Critic Lambda return 和 return-based reset sampling。没有用“模块存在”代替运行验证；下一节列出真实测试结果。

审计要点：状态 71 维，动作 29 维，仿真 context 33 维；Actor 输入由 930 维历史 observation 和 680 维未来运动目标构成；Dynamics 输入 133 维，输出 70 维增量并显式积分；Termination head 预测概率，Reward/Observation 由 Torch FK 和解析公式生成；默认 World Model 隐层 `(2048,2048,2048)`，Actor 隐层保持原始配置；Imagination 默认 32 步。

## Phase 3：LAFAN1 数据

- 数据源：`lvhaidong/LAFAN1_Retargeting_Dataset`，下载 revision `ce1572906efe6157840e8474d5a0d7aa87481e74`，只拉取 `g1/*.csv`，没有下载 BONES-SEED。
- `data/raw/lafan1/g1/`：40 CSV；`data/npz/lafan1/`：40 NPZ、`manifest.csv`。
- 转换：30 Hz → 50 Hz，4 workers；0 失败。Manifest 汇总 441,140 frames、2.45056 h。
- 全量检查每个 NPZ：关键数组有限、FPS=50、关节角/速度 shape 为 `(T,29)`、刚体位置/姿态 shape 为 `(T,30,3/4)`，body quaternion 范数误差小于 `1e-5`。
- 抽查 `dance2_subject5`、`walk3_subject4`、`walk1_subject5`：输出最大 quaternion 范数误差 `1.19e-7`，关节角处于弧度量级，根位置处于米量级，G1 关节顺序 29 DOF。
- 数据检查结果：`logs/4070_audit/data_validation.txt`；清单：`data/npz/lafan1/manifest.csv`。

## Phase 4：单元测试与 GPU 环境

执行 `python -m pytest tests -q`（隔离 ROS 环境），结果：**24 passed, 56 warnings in 177.15s，0 failed**。测试覆盖 `test_g1_fk.py`、`test_analytic_heads.py`、`test_world_model.py`、`test_imagination.py`、`test_env_smoke.py`、`test_motion_convert.py`、`test_motion_library_compact.py` 等模块；完整输出 `logs/4070_audit/pytest_full.log`。

- FK 与 MuJoCo 对照测试通过；解析奖励/观测头测试通过。
- World Model shape、one-step、8-step loss、backward、action-to-predicted-state gradient 测试通过；测试内模型拟合用例通过。
- Imagination actor→WM→reward 梯度、soft survival、critic Lambda return 测试通过。代码冻结的是 WM 参数，rollout 不在 `no_grad` 中，因此保留动作输入 Jacobian。
- GPU 环境 smoke、运动库 compact/replay buffer 和 checkpoint 恢复单测通过。
- warning 主要为 TorchScript deprecation、当前 driver 对 CUDA Graph 的限制和 Warp 类型提示；没有导致测试失败。

## Phase 5：RTX 4070 显存扫描

### Level A：H=8 链路冒烟

用 64 env、policy/WM batch 64、H=8、replay capacity 256、每次 WM update 1 batch 跑 150 iterations，模型宽度不变。结果保存于 `logs/4070/smokeA_defaultwidth_4070/`：更新 44 次；WM loss 从约 64.65 降到 47.07，末期 online NSE 约 0.159/1.517（1/8-step）；train norm return 约 0.426，imag return 约 0.096，H survival 约 0.009；Critic loss 有限（约 2.8e-4），Actor gradient 非零但很小（约 2e-6）。checkpoint 保存成功，GPU 采样峰值约 1,277 MiB。该 H=8 配置只用于快速验证链路。

### Level B：恢复 H=32

最初 64 env、batch64 的一次更新完整执行了 H=32：WM loss 257.386，1-step NSE 0.929、8-step 7.948，Critic loss 0.1808，imag return 0.114，H survival 0，Actor gradient 约 1e-8，峰值 1,435 MiB。它验证 tensor/反向执行而未证明未训练模型能产生有效策略梯度。随后 500-iteration 训练数据在 Phase 6 报告。

网络宽度保持原始值；所有 Level B 均使用 H=32，环境逐级增加。每档只执行一轮 optimizer 更新，适用于资源验证，不把 12 iteration 结果当作策略训练指标。显存为 `nvidia-smi` 每秒采样的总占用，含桌面背景进程；环境容量 12,282 MiB。

| num_envs | Actor batch | WM batch | WM batches/update | 峰值显存 | 结果 |
|---:|---:|---:|---:|---:|---|
| 64 | 64 | 64 | 1 | 1,435 MiB | 完整 H=32 update、checkpoint 成功 |
| 128 | 128 | 128 | 1 | 1,636 MiB | 成功 |
| 256 | 256 | 256 | 1 | 2,040 MiB | 成功 |
| 512 | 512 | 512 | 1 | 3,069 MiB | 成功 |
| 1,024 | 512 | 512 | 1 | 3,192 MiB | 成功 |
| 1,024 | 1,024 | 1,024 | 1 | 5,146 MiB | 成功 |
| 1,024 | 2,048 | 2,048 | 1 | 8,738 MiB | 成功，无 OOM |
| 1,024 | 1,024 | 1,024 | **10** | **6,175 MiB** | 默认 WM update 批次数，H=32 和 checkpoint 成功 |

资源采样 CSV 和每次 CLI 输出分别以 `gpu_h32_*.csv`、`smokeB_h32_*.log` 保存。Level B policy/WM 编译首轮可占 80–90 秒；后续短训 64 env 的稳定吞吐约 6.5–7.0 vector iterations/s。1024 env/1024 batches、默认 10 个 WM batch/update 的一次更新记录 `logs/4070_audit/levelC_default_update.log`，时间分项 sim/WM/policy 为 3.0/1.6/3.6 秒（42 iter 的该次测量窗口）。所有探测都没有 OOM。

建议区分两个实测边界：

1. **已验证的较大单步 batch：** 1,024 env / 2,048 policy batch / 2,048 WM batch / 1 WM batch per update，H=32，峰值 8.74 GiB。它是短资源试跑，不适合直接作为长期配置。
2. **长期训练候选：** 1,024 env / 1,024 policy batch / 1,024 WM batch / 10 WM batches per update，H=32，峰值 6.18 GiB。它保留完整网络、reward、termination、Horizon 和默认 WM 更新批次数，并为完整数据集载入与桌面留有余量。

1024/1024/10 是本轮实际验证过的最大稳定长训候选配置；没有尝试 3,072/4,096 batch，也没有从 12 GiB 试跑推断这些配置可行。仓库默认的 Actor batch 4096 不建议直接放到该卡上。Torch compile 警告没有造成异常；本机 driver 使 Warp CUDA Graph 关闭。没有为了显存降低网络宽度，也没有对 H=32 的正式候选缩短 horizon。

## Phase 6：500 iteration、完整 H=32 短训

运行 `walk1_subject1`，64 env、policy/WM batch 64、H=32、默认 10 WM batches/update、40 iteration warmup、UTD=0.4，500 次向量化迭代。命令和逐秒 GPU 采样分别见 `logs/4070_audit/commands.sh`、`gpu_short32_500.csv`；训练 TensorBoard 事件在 `logs/4070/short32_walk1_4070/tb/`。

| iteration | WM loss | online NSE 1-step | online NSE 8-step | Actor grad norm | Critic loss | imagination return | H survival | real train norm return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 49.99 | 0.170 | 1.087 | 2.39e-6 | 0.00571 | 0.107 | 5.1e-8 | 0.439 |
| 250 | — | — | 0.119 | — | — | 0.647 | 0.535 | 0.408 |
| 500 | 4.50 | 0.023 | 0.105 | 0.0196 | 0.0178 | 0.671 | 0.268 | 0.424 |

WM state loss 从 49.41 降到 4.45；termination BCE 从 0.580 降到 0.0506；终点 termination 概率均值约 0.168。Actor 梯度有限且非零，动作绝对值均值从 0.00029 增至 0.153；Critic value/target 约 0.293/0.301，loss 有限。过程未观察到 NaN/Inf。真实训练回报约 0.39–0.44 波动，imagination return 不能代表真实跟踪改善。

GPU 采样峰值 1,458 MiB。TensorBoard 含完整曲线 tag（WM state/termination loss、NSE、gradient norm、Actor objective/gradient norm、Critic loss、real episode return、timing 等）。checkpoint `ckpt_0000250.pt`、`ckpt_0000500.pt`、`ckpt_latest.pt` 均写出，各约 206 MiB。

### Checkpoint resume

从 `short32_walk1_4070/ckpt_latest.pt` 恢复，日志明确显示 iteration 500 和 184 次更新已载入。checkpoint 不包含 replay 数据，因此先采集 10 个 vectorized step，补满连续 8-step 窗口，再运行一次 H=32 Actor/Critic + WM 更新，进到 iteration 512、185 次更新并保存新的 `ckpt_latest.pt`。恢复日志 `logs/4070_audit/resume_short32.log`；新产物在 `logs/4070/resume_short32_4070/`。

## Phase 7：benchmark 与 viewer

评测 `short32_walk1_4070/ckpt_latest.pt`，加载 40 个 NPZ、32 个真实 MuJoCo Warp env。输出：**40 clips、40 failed、mean Norm Return 0.3869、step-weighted 0.3879**。失败集中在约 41–45 steps，符合 500 步单 clip 短训尚未获得稳定策略的情况。结果写在 `logs/4070/short32_walk1_4070/benchmark_40clips.json`，控制台全量日志为 `logs/4070_audit/eval_benchmark.log`。

`scripts/play.py --viewer native --clips walk1_subject1 --num-envs 1` 已运行。`DISPLAY=:1`，窗口树确认 `MuJoCo : mjlab scene` 1280×720 窗口存在；屏幕截图 `logs/4070_audit/viewer_desktop.png` 中可见实心策略机器人与半透明蓝色参考 ghost 同时播放。进程保持运行，便于本机直接查看。纯 SSH 时可设 `--viewer viser` 并通过安全的 SSH 本地端口转发访问 viewer 提示的端口（默认 8080），例如本地 `ssh -L 8080:127.0.0.1:8080 user@host`。

## Phase 8：算法文档

[`SUPERMIMIC_ALGORITHM_MAP.md`](SUPERMIMIC_ALGORITHM_MAP.md) 覆盖 Figure 2、Algorithm A.1 与 Eq. 1–3，列出 71/29/33/133 维 state/action/context/model input、显式状态积分、8-step autoregressive loss、解析 reward/observation、32-step 图中 rollout、soft survival、Actor BPTT 梯度路径、Critic Lambda return、reset sampler 和每个训练 iteration 的真实/想象样本计数。

## Phase 9：正式训练建议

当前可复用的长期候选命令如下。它未在本轮启动：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name lafan1_full_4070_1024 --log-root logs/4070 \
  --env.num-envs 1024 --env.replay-capacity 256 \
  --policy-trainer.batch-size 1024 --policy-trainer.horizon 32 \
  --wm-trainer.batch-size 1024 --wm-trainer.batches-per-update 10 \
  --schedule.total-iterations 320000 --schedule.warmup-iterations 40 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 4000
```

不指定 `--data.clips` 时使用 40 条 LAFAN1 NPZ。此配置保留原始 World Model、Actor、Critic、Reward、Termination 和 32 步 BPTT；偏离论文硬件配置的是 1,024 个 env 和 1,024 个训练 batch（论文默认 4,096 env、Actor batch 4,096、WM batch 2,048），并且 PyTorch/Warp wheel 采用与旧驱动兼容的 CUDA 12 构建。WM 的默认 batch/update 数仍为 10。按一次 1,024-env 更新测量约 5.1 vector iterations/s 粗估，320,000 iteration 约 17.4 小时；建议正式跑前确认桌面可用显存，并根据开始后真实日志重新估算完成时间。

训练数据范围也是明确偏离：本次按任务要求只用 2.45056 h 的 LAFAN1 G1 重定向数据，不下载 BONES-SEED。论文主实验使用 264.56 h BONES-SEED；论文报告的 LAFAN1 对照在 H100 上约 1.2 h 达到 Norm Return 0.82。硬件、数据规模、训练时长均不同，因此这轮 500 iteration 结果不能与 0.82 目标作等条件比较。[原论文 arXiv:2610.28480](https://arxiv.org/abs/2610.28480)。

### 当前验收表

| 验收项 | 结果 |
|---|---|
| 环境部署 / CUDA GPU tensor | 通过；cu126 Torch 前向/反向正常 |
| Warp GPU kernel | 通过；输出 `[2,4,6,8]` |
| MuJoCo 3.16 / Warp / mjlab GPU env | 通过；G1 XML `nq=36,nv=35,nbody=31`，env smoke 通过 |
| LAFAN1 CSV→NPZ | 通过；40/40、0 失败、2.45056h |
| 全部 pytest | 通过；24 passed、56 warnings |
| 8-step WM loss 与梯度 | 单测通过；短训 online NSE 0.170/1.087 → 0.023/0.105 |
| 32-step BPTT / Actor | 通过运行和单测；500 iter actor grad norm 0.0196 |
| Critic | 通过更新；loss 有限，500 iter 为 0.0178 |
| checkpoint 写入/恢复 | 通过；恢复后继续更新并保存 |
| benchmark | 链路通过，短训模型质量失败：40/40 failed、mean 0.3869 |
| 可视化 | Native MuJoCo viewer 已开窗，ghost + policy 播放 |
| RTX 4070 稳定候选 | 1024 env / Actor 1024 / WM 1024 / WM 10 batches / H32，峰值 6,175 MiB |
| 论文目标 Norm Return 0.82 | 未达到；本轮只验证系统链路，未进行长程训练 |

## 实验记录索引

- 机器审计/版本：`logs/4070_audit/`。
- 完整测试：`logs/4070_audit/pytest_full.log`。
- Level A：`logs/4070_audit/smokeA.log`、`logs/4070/smokeA_defaultwidth_4070/`。
- Level B/H32 扫描：`logs/4070_audit/smokeB_h32_*.log`、`gpu_h32_*.csv`、`logs/4070/smokeB_h32_*/`。
- 默认 WM 更新规模验证：`logs/4070_audit/levelC_default_update.log`、`gpu_levelC_default_update.csv`、`logs/4070/levelC_default_update_4070/`。
- 500-step H32 train / resume：`logs/4070_audit/short32_500.log`、`resume_short32.log`、TensorBoard、checkpoints。
- benchmark/viewer：`logs/4070_audit/eval_benchmark.log`、`play_native.log` 和评测 JSON。
- 可重放命令：`logs/4070_audit/commands.sh`。
