# RTX 4070 SUPER：LAFAN1 Q1 训练启动与恢复记录

**状态：15:04 的首次启动命令因换行无效；用户随后用普通终端成功运行 `lafan1_q1_4070_anchor_retry1`。训练于 2026-10-24 19:15 左右按 `max_hours` 正常结束，完成 67,900 iteration；2026-10-25 完成全 40 条确定性 MuJoCo Warp 评测。**首次失败目录保留，不复用。

审计时间：2026-10-24 14:53（Asia/Shanghai）。

启动尝试复核时间：2026-10-24 15:08（Asia/Shanghai）。

## 1. 启动前状态

| 项目 | 检查结果 |
|---|---|
| Git | 14:53 预检时 `main` HEAD 为 `58d9898fe617bd861d19fe1d750f7da5b4fcc27f`；本报告结果快照基于 `2b1d77b` |
| 算法源码 | 相对基线 `d9be385`，`src/supermimic/`、`pyproject.toml`、`uv.lock` 无差异 |
| Conda | `/home/yifeiy/miniconda3/envs/supermimic/bin/python`，Python 3.12.15；Torch 2.11.0+cu126、Warp 1.18.0+cu12、MuJoCo/mujoco-warp 3.16.0、mjlab 1.6.0；CUDA 可用 |
| 数据 | `data/npz/lafan1/` 有 40 NPZ，manifest 40 行，总时长 2.45056 h；不传 `--data.clips` 时会按文件名字典序加载全部 40 条 |
| 训练/Viewer | 没有 `sm-train`、`scripts/play.py` 或 benchmark 进程；旧 Viewer 已退出；tmux 3.0a 已安装，目前无 tmux session |
| GPU | RTX 4070 SUPER 12,282 MiB；快照占用 853 MiB，可用约 11,429 MiB；没有项目 Python/CUDA 进程，剩余为桌面图形任务 |
| 新 run 目录 | 14:53 预检时不存在；15:04 的无效尝试后来创建了该目录，现保留，不复用 |
| 旧 checkpoint | iteration 500、512 的 `ckpt_latest.pt` 均约 215 MB，字段完整，但 `anchor_obs=false` 且只训练过 `walk1_subject1`，不能作为本次训练初始化 |

原始快照：`logs/4070_audit/q1_preflight_nvidia_smi.txt`、`q1_preflight_processes.txt`、`q1_preflight_git_status.txt`、`q1_preflight_git_commit.txt`、`q1_env_check.txt`。数据完整性检查见 `data_validation.txt`。

### CLI 与 anchor 输入验证

当前 Conda 环境的 `sm-train --help` 确认了 `--data.npz-dir`、`--env.anchor-obs`、`--seed`、指定的 env/batch/horizon/schedule 参数和 `--resume`。随后用 Tyro 对本次完整参数解析，没有构造 Trainer 或启动训练：得到 `clips=None`（全量数据）、1024 env、Actor/WM batch 1024、H=32、10 WM batches/update、320000 iteration、4 小时上限、seed 0、`anchor_obs=true`。输出见 `q1_sm_train_help.txt`、`q1_config_parse.txt`。

`anchor_obs` 为每帧目标加入 9 维 torso-anchor-relative pose，goal 维度从 680 变成 689。`Trainer.__init__` 会同步设置 Policy 的 `goal_dim` 和 `PolicyTrainer.anchor_obs`；World Model state/action、Dynamics 隐层宽度、Reward、Termination 和 H=32 不变。CPU shape smoke 得到 Actor `(2,29)`、Critic `(2,)` 有限输出。预期参数量：World Model 8.82M、Actor 8.9096M、Critic 0.2373M；Actor/Critic 输入层因新增 9 维而增加参数，隐藏层宽度不变。结果见 `q1_anchor_shape_check.txt`、`q1_anchor_model_param_estimate.txt`。

**论文偏离：**原始观测设置和此前 500 iteration baseline 均为 `anchor_obs=false`。本次设为 `true`，增加 9 维相对锚点目标，是对论文观测配置的明确偏离。该选项改变 Actor/Critic 输入层形状，不改变核心 World Model 或 BPTT 算法。评测和 Viewer 会读取 checkpoint 中的配置。

### 用户启动尝试结果（无效，保留现场）

用户在 15:04:44 创建了 `lafan1_q1_4070_anchor` tmux session，但检查时 tmux server/session 和训练进程都已退出。目录内只有 `config.json` 与 88 字节 TensorBoard event，没有 `console.log`、checkpoint 或训练指标。实际写出的配置使用默认值：`num_envs=4096`、`replay_capacity=1000`、`anchor_obs=false`、Policy batch 4096、WM batch 2048、`max_hours=null`；仅 seed 和 total iterations 与计划相同。因此不能把它作为 Q1 训练、恢复来源或评测结果，也无法从已消失的 pane 还原精确 shell 错误。没有证据表明发生了 OOM。15:11 GPU 快照为 843 / 12,282 MiB；进程表只有 Xorg、GNOME、浏览器/WebKit 和 ToDesk 图形进程，没有训练计算进程。

失败尝试的配置快照与摘要保存在 `logs/4070_audit/q1_failed_launch_config.json` 和 `logs/4070_audit/q1_failed_launch_summary.txt`。原目录 `logs/4070/lafan1_q1_4070_anchor/` 不删除、不覆盖。

## 2. 本次计划配置

| 参数 | 计划值 |
|---|---:|
| 固定 seed | 0（与前轮 smoke 一致，命令中显式设置） |
| LAFAN1 clips | 全部 40 条，按文件名字典序 |
| `env.anchor_obs` | `true` |
| 环境数 / replay capacity | 1024 / 256 |
| Policy batch / imagination horizon | 1024 / 32 |
| WM batch / batches per update | 1024 / 10 |
| WM window | 8（源码默认） |
| 总迭代 / 单进程 wall-time 上限 | 320000 / 4 小时 |
| warmup / log interval / checkpoint interval | 40 / 50 / 4000 |
| UTD / policy updates per WM update | 0.4 / 1（源码默认） |

`max_hours` 在每个 50 iteration 日志点检查；超限后正常退出 loop，随后写最终 `ckpt_latest.pt` 并 flush TensorBoard。编号 checkpoint 每 4000 iteration 保存一次，若达到则包含 20000、40000、60000、80000 等里程碑。

## 3. 普通终端前台启动命令与核验

上次命令把真实换行放进 `bash -lc '…'` 的引号中，Shell 会把它们当成命令分隔符；Shell 引号中的换行不会自动续接命令。另外，`--wm-trainer.batches-per-update` 被拆成了两行。实际配置快照显示 `sm-train` 只收到前面的 run name/log root/seed，其他字段使用默认值。tmux 本身不是训练失败原因：当内层命令结束后，tmux 的最后一个窗口关闭，session/server 也随之消失。具体退出错误没有保存到日志，无法确认是 CLI 后续命令报错还是进程在启动中另行退出。

以下命令在当前普通终端前台运行：日志同时显示在屏幕和文件中；按 Ctrl+C 可中断。每个续行反斜杠都在 Shell 顶层，不再嵌套到引号里。启动使用新目录 `retry1`，不会覆盖失败现场：

```bash
cd /home/yifeiy/ws_supermimic
RUN=lafan1_q1_4070_anchor_retry1
OUT="logs/4070/$RUN"

test -s data/npz/lafan1/manifest.csv || { echo "LAFAN1 manifest missing"; exit 1; }
if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing run directory: $OUT"
  exit 1
fi
mkdir -p "$OUT"
set -o pipefail
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name "$RUN" --log-root logs/4070 --seed 0 \
  --data.npz-dir data/npz/lafan1 --env.anchor-obs \
  --env.num-envs 1024 --env.replay-capacity 256 \
  --policy-trainer.batch-size 1024 --policy-trainer.horizon 32 \
  --wm-trainer.batch-size 1024 --wm-trainer.batches-per-update 10 \
  --schedule.total-iterations 320000 --schedule.max-hours 4 \
  --schedule.warmup-iterations 40 --schedule.log-interval 50 \
  --schedule.checkpoint-interval 4000 \
  2>&1 | tee "$OUT/console.log"
```

用户已在普通终端执行该命令；助手只做只读核验，没有启动或中断训练。2026-10-24 15:17 的实际快照如下：

| 项目 | 实际结果 |
|---|---|
| 进程 / 终端 | PID 1876086，普通前台进程；无 tmux session |
| 加载数据 | 40 clips，441,140 frames，2.451 h @50Hz，8,842 bins，0.57 GB GPU |
| 生效配置 | 1024 env，replay 256，`anchor_obs=true`，Actor/WM batch 1024，H=32，WM window=8，batches/update=10，max_hours=4，seed=0 |
| 参数量 | World Model 8.82M、Policy 8.91M、Critic 0.237M |
| 进度 / 速度 | 550 iteration，204 updates；日志约 4.3 it/s，累计训练时间约 0.0343 h |
| 学习指标（step 550） | WM NSE 1-step 0.0303、8-step 0.0902，WM grad norm 13.03；Actor grad norm 0.138；Critic loss 0.00865；各值有限 |
| 回报（step 550） | 真实采集 `episode_norm_return` 0.384；imagination return 0.673，H-step survival 0.276。均为早期训练指标，不是 40 clip 最终评测 |
| GPU | 6,013 / 12,282 MiB，GPU 利用率约 93%；未见 OOM |
| Checkpoint | 尚无；当前仅 550 iteration，首个周期 checkpoint 计划在 4,000 iteration |
| 异常检查 | 没有 traceback、NaN/Inf、CUDA OOM；有一条非致命提示：driver 12.2 导致 CUDA Graphs disabled，仿真与训练继续正常运行 |

该快照说明启动参数、40 条数据、MuJoCo Warp GPU 仿真、WM 8-step 训练和 Actor 32-step BPTT/critic 更新均已进入运行链路。训练仍处于早期，尚不能据此判断多轨迹跟踪已学会。前台终端需要保持打开；如果在 4,000 iteration checkpoint 前手动中断，可能没有可恢复的本轮 checkpoint。查看实时情况可在另一个终端运行 `tail -f logs/4070/lafan1_q1_4070_anchor_retry1/console.log`、`nvidia-smi`。

启动日志已确认：`MotionLibrary(full): 40 clips, 441140 frames (2.451 h @50Hz)`；参数量 `8.82M / 8.91M / 0.237M`；设备 `cuda:0`；console log 与 TensorBoard event 正在 run 目录生成。

## 4. 训练监控与预算估算

训练结束于 iteration 67,900、27,144 updates、`time/hours=4.00259`，平均约 4.71 vectorized iterations/s。达到 4 小时上限时未达到 80,000 iteration（完整 320,000 计划的 1/4），实际完成约 21.2%。按本轮吞吐估算：

| 目标 iteration | 估算耗时 |
|---:|---:|
| 20,000 | 1.18 h |
| 40,000 | 2.36 h |
| 60,000 | 3.54 h |
| 80,000 | 4.72 h |
| 160,000 | 9.44 h |
| 320,000 | 18.88 h |

本轮实际到 67,900 iteration，和上述估算接近。评测前确认训练进程已退出，GPU 回到桌面空闲占用。

查看进度、显存和 TensorBoard：

```bash
tail -f logs/4070/lafan1_q1_4070_anchor_retry1/console.log
nvidia-smi
nvidia-smi --loop=2

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/tensorboard \
  --logdir logs/4070/lafan1_q1_4070_anchor_retry1/tb --host 127.0.0.1 --port 6006
```

TensorBoard 重点看 WM loss/NSE/gradient、`policy/policy_grad_norm`、`policy/critic_loss`、imagined return/survival、`train/episode_norm_return`、`train/episode_len` 和 `time/*`。Imagination Return 上升不等于真实跟踪改善。

首选让 `max_hours=4` 正常结束，它会执行最终 checkpoint 保存。前台运行时按 Ctrl+C 会发送 SIGINT；这不会经过 loop 的最终保存路径，可能丢失最近一个编号 checkpoint 之后最多约 4000 iterations。需提前确认编号 checkpoint 已完整落盘。中断后可在普通终端检查：

```bash
ls -lh logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_*.pt
```

如果手动停止，优先从最后一个完整的 `ckpt_NNNNNNNN.pt` 恢复；不要在 `torch.save` 正写文件时强行结束进程。只有确认训练卡死且必须立即停止时，才从另一终端对本项目的训练 PID 发送 `kill -INT <PID>`，不要用 `kill -9`。

## 5. Checkpoint 与恢复

源码保存并恢复 World Model、Actor、Critic 参数，三个 optimizer 状态，iteration、更新次数和 Motion Sampling 的 `bin_return/bin_memory`。不保存 Replay Buffer、环境/episode 状态、RNG 状态、UTD 小数累积器或 wall time。因此这是状态恢复，不是逐步完全一致的重放；启动进程时 `torch.manual_seed(seed)` 会重新初始化随机流。

恢复时 replay 为空。loop 按全局 iteration 判断 warmup；若只保留 `warmup_iterations=40`，checkpoint iteration 已大于 40，下一步就会尝试更新，窗口数据不足。若 checkpoint iteration 为 K，应设置 `warmup_iterations=K+40`，先采集 40 个新向量化环境步。总迭代数仍保持 320000，cosine LR 继续按全程 schedule。每次 resume 使用新 run name 和目录，避免覆盖旧 checkpoint。

恢复时使用相同 `data/npz/lafan1`、不传 clips/limit/manifest（全 40 条并按文件名排序）、相同 seed、网络结构和 `--env.anchor-obs`。本次训练必须从本次 anchor checkpoint 恢复；iteration 500/512 的旧 checkpoint 是 `anchor_obs=false`，不兼容本次输入形状。

### 恢复命令模板（尚未执行）

```bash
cd /home/yifeiy/ws_supermimic
PY=/home/yifeiy/miniconda3/envs/supermimic/bin/python
CKPT=logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt
RUN=lafan1_q1_4070_anchor_retry1_resume_01
OUT="logs/4070/$RUN"
CKPT_ITER=$(env -u PYTHONPATH -u LD_LIBRARY_PATH "$PY" -c \
  'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["iteration"])' "$CKPT")
WARMUP_ITER=$((CKPT_ITER + 40))

test -s "$CKPT" || { echo "Checkpoint missing: $CKPT"; exit 1; }
if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing run directory: $OUT"
  exit 1
fi
mkdir -p "$OUT"

set -o pipefail
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name "$RUN" --log-root logs/4070 --resume "$CKPT" --seed 0 \
  --data.npz-dir data/npz/lafan1 --env.anchor-obs \
  --env.num-envs 1024 --env.replay-capacity 256 \
  --policy-trainer.batch-size 1024 --policy-trainer.horizon 32 \
  --wm-trainer.batch-size 1024 --wm-trainer.batches-per-update 10 \
  --schedule.total-iterations 320000 --schedule.max-hours 4 \
  --schedule.warmup-iterations "$WARMUP_ITER" \
  --schedule.log-interval 50 --schedule.checkpoint-interval 4000 \
  2>&1 | tee "$OUT/console.log"
```

后续恢复轮次将 `CKPT` 改为上一轮完整 checkpoint，`RUN` 改成新的唯一目录。若要在约 160,000 iteration 暂停，等待 `ckpt_000160000.pt` 完整出现再停止；恢复时仍保留 `total_iterations=320000`。按本轮平均速度，从 0 到 160k 约 9.44 小时；从 67,900 继续到 160k 还需约 5.43 小时，实际以新一轮日志为准。可分成一个 4 小时恢复轮次和一个约 1.5 小时恢复轮次。

## 6. 全量评测结果与 Viewer

训练结束、GPU 空闲后，以 `ckpt_latest.pt`、32 个环境运行全 40 条 deterministic benchmark。脚本关闭 observation/action noise、reset noise 和 domain randomization，所有轨迹均从第一帧开始。命令与控制台输出保存在 `eval_benchmark_40clips.log`，逐轨迹结果保存在 `benchmark_40clips.json`。

```bash
set -o pipefail
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/eval_benchmark.py \
  --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 32 \
  --out logs/4070/lafan1_q1_4070_anchor_retry1/benchmark_40clips.json \
  2>&1 | tee logs/4070/lafan1_q1_4070_anchor_retry1/eval_benchmark_40clips.log
```

### 实测结果（2026-10-25）

| 指标 | 本轮 `retry1` | 用户提供的 500 iteration baseline |
|---|---:|---:|
| Mean Normalized Return | **0.4135** | 0.3869 |
| step-weighted Normalized Return | **0.4143** | 未提供 |
| Failed Clips | **2 / 40** | 40 / 40 |
| 平均评测 episode 长度 | **10,928.3 steps（约218.6秒 @50Hz）** | 未提供 |

Mean Norm Return 比 baseline 高 `0.0266`（约 `+6.9%`），失败数从 40 降至 2。该比较需注意本轮启用了 `anchor_obs=true`，baseline 为 `false`，且 baseline 来自 500 iteration；因此差异不应只归因于训练时长。仓库输出的 `step_weighted_norm_return` 是按每条 clip 实际执行步数加权，不是按完整参考帧数加权的另一种独立统计量。

两条失败轨迹：

- `fallAndGetUp1_subject4`：第 7,102 / 8,411 步触发失败，Norm Return 0.3886。
- `fallAndGetUp2_subject3`：第 5,496 / 8,196 步触发失败，Norm Return 0.4336。

Return 最好：`walk4_subject1` 0.5279、`walk3_subject3` 0.5234、`walk1_subject1` 0.5039。最差：`run1_subject5` 0.3043、`run2_subject4` 0.3300、`run2_subject1` 0.3301。按类别均值，walk 12 条为 0.4648；run 4 条为 0.3306；sprint 2 条为 0.4144；dance 8 条为 0.4115；jumps 3 条为 0.4064；fight 5 条为 0.3981；fall/get-up 6 条为 0.3851（其中两条失败）。当前主要短板是 run 类动作回报偏低，以及 fall/get-up 稳定性。

### 训练过程信号与解释

在训练日志中 WM NSE 从 step 50 的 1-step `0.2003` / 8-step `1.1725` 降到 step 67,900 的 `0.0126` / `0.0305`；Actor 梯度范数从 `6.0e-6` 到 `0.0817`，Critic loss 从 `0.0172` 到 `0.00255`，均为有限值。训练采集 Norm Return 从 `0.3737` 到 `0.4287`。Imagination Return 从 `0.0893` 升至 `1.739`，Survival 从接近 0 升至 `0.999`；这表示模型内目标明显上升，但不能单独证明真实跟踪能力，需与上述真实仿真评测一起解释。

训练尾部日志在 iteration 67,900 打印 `[INFO] max_hours reached, stopping`；无 traceback、NaN/Inf 或 OOM。周期 checkpoint 保存到 `ckpt_00064000.pt`，最终 checkpoint 为 `ckpt_latest.pt`（iteration 67,900，约 206 MiB）；GPU 已释放。训练中采集到的一次显存快照约 6,013 MiB；日志未记录 allocator peak，因此不把该快照称为精确峰值。此轮保持原始 WM/Policy 网络规模、8-step WM、32-step BPTT 和 Reward/Termination，配置偏离是 `anchor_obs=true`。训练和评测时看到 CUDA Graphs disabled（驱动 12.2）；Warp 仿真仍运行。

### Viewer 回放（未执行）

`play.py` 是交互式 Viewer，会持续占用一个终端/窗口直到手动退出。本轮考虑到此前用户不希望遗留可视化窗口，没有自动启动 Viewer。需要观看时，在普通终端串行运行下列命令，每个窗口看完按 `Ctrl+C` 或关闭窗口，再运行下一条：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips walk1_subject1 --num-envs 1 --viewer native
env -u PYTHONPATH -u LD_LIBRARY_PATH /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips dance1_subject1 --num-envs 1 --viewer native
env -u PYTHONPATH -u LD_LIBRARY_PATH /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips jumps1_subject1 --num-envs 1 --viewer native
```

## 7. 当前报告状态

本轮 4 小时训练、周期与最终 checkpoint 保存、TensorBoard 日志和全 40 条真实仿真 benchmark 均已完成。Q1 checkpoint 在 67,900 iteration，Mean Normalized Return 0.4135、step-weighted 0.4143、失败 2/40。相对用户提供的 500 iteration baseline 有改善，但 `anchor_obs` 不同，且本轮低于计划的 80k，也不等同于完整论文结果。Viewer 回放尚未执行；短板集中在 run 和 fall/get-up 动作。建议保持本轮结构和算法，恢复训练到至少 160k，并在中途定期做同协议 benchmark；不建议仅依据 imagination return 判断是否继续。

预检输出见 `logs/4070_audit/q1_*`；此前链路记录见 [`LOCAL_REPRO_4070.md`](LOCAL_REPRO_4070.md)，算法细节见 [`SUPERMIMIC_ALGORITHM_MAP.md`](SUPERMIMIC_ALGORITHM_MAP.md)。

## 8. 下一轮 2048 / 3072 环境续训建议

建议先用 `2048 env + 15 分钟资源探测`，稳定后再跑 12 小时；3072 env / 16 小时属于更激进、尚未验证的配置。保持完整 40 条 LAFAN1、`anchor_obs=true`、WM/Actor batch 1024、WM window 8、H=32、网络架构和 Reward/Termination 不变。只改 `num_envs` 与该轮 `max_hours`，并保持 `updates_per_env_step=0.4`。按当前代码，UTD 是每个 vectorized env iteration 计数；增加并行环境会令每次更新前收集的数据更多，因此这是采样规模变化，不是完全相同的数据/更新比例。

ReplayBuffer 的张量形状是 `[num_envs, replay_capacity]`。按本项目 71-d state、29-d action、33-d xi，capacity=256 时 replay 数据约为 2048 env 的 341 MiB、3072 env 的 512 MiB；Mujoco Warp 环境数组还会随环境数增长。1024 env 曾观测到约 6.0 GB GPU 占用，但没有记录精确 peak，因此不把 3072 标为已验证可稳定运行。

下面是普通终端前台命令。先使用 2048/0.25h 探测；成功退出并生成 `ckpt_latest.pt` 后，把 CKPT 指向探测 checkpoint、改 RUN 和 HOURS，再运行正式 2048/12h。想尝试 3072/16h 时，先另做 3072/0.25h 探测。不要复用 run 目录。

```bash
cd /home/yifeiy/ws_supermimic
PY=/home/yifeiy/miniconda3/envs/supermimic/bin/python
CKPT=logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt
ENV_COUNT=2048
HOURS=0.25
RUN=lafan1_q1_resume_2048_probe15m
OUT="logs/4070/$RUN"

test -s "$CKPT" || { echo "Checkpoint missing: $CKPT"; exit 1; }
test -s data/npz/lafan1/manifest.csv || { echo "LAFAN1 manifest missing"; exit 1; }
if [ -e "$OUT" ]; then
  echo "Refusing to reuse existing run directory: $OUT"
  exit 1
fi
CKPT_ITER=$(env -u PYTHONPATH -u LD_LIBRARY_PATH "$PY" -c \
  'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["iteration"])' "$CKPT")
WARMUP_ITER=$((CKPT_ITER + 40))
mkdir -p "$OUT"
set -o pipefail
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/sm-train \
  --run-name "$RUN" --log-root logs/4070 --resume "$CKPT" --seed 0 \
  --data.npz-dir data/npz/lafan1 --env.anchor-obs \
  --env.num-envs "$ENV_COUNT" --env.replay-capacity 256 \
  --policy-trainer.batch-size 1024 --policy-trainer.horizon 32 \
  --wm-trainer.window 8 --wm-trainer.batch-size 1024 --wm-trainer.batches-per-update 10 \
  --schedule.updates-per-env-step 0.4 --schedule.policy-updates-per-wm-update 1 \
  --schedule.total-iterations 320000 --schedule.max-hours "$HOURS" \
  --schedule.warmup-iterations "$WARMUP_ITER" \
  --schedule.log-interval 50 --schedule.checkpoint-interval 4000 \
  2>&1 | tee "$OUT/console.log"
```

探测稳定后正式 2048/12h：`CKPT=logs/4070/lafan1_q1_resume_2048_probe15m/ckpt_latest.pt`、`HOURS=12`、`RUN=lafan1_q1_resume_2048_12h`，其余命令不变。3072/16h 则先将探测轮次设为 `ENV_COUNT=3072`、`RUN=lafan1_q1_resume_3072_probe15m`；通过显存观察后，正式轮次改为 CKPT 指向该探测输出、`HOURS=16`、`RUN=lafan1_q1_resume_3072_16h`。

`max_hours` 是每次进程的运行上限，恢复时计时器会重置；若 12/16 小时指含已完成首轮在内的总墙钟时间，应将下一轮 `HOURS` 分别设为约 8/12。训练末尾 1024-env 日志的 `sim/wm/pi` 约为 `2.0/4.1/4.3 s` 每 50 iteration。假设只有仿真部分随环境数线性增长，2048 env 约 4.0 it/s、3072 env 约 3.5 it/s；从 67,900 起，12h 粗估分别到约 242k / 218k，16h 粗估约 300k / 268k。该线性模型未实测，不能作为吞吐保证。总 iteration 仍受 320,000 上限约束，达到上限会早于 max_hours 正常结束并保存最终 checkpoint。
