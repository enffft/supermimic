# RTX 4070 SUPER：LAFAN1 Q1 训练启动与恢复记录

**状态：启动前检查完成；用户于 2026-10-24 15:04 尝试启动，但命令换行导致配置参数未传入，进程已退出。本次没有形成有效训练。**失败目录已保留；重试必须使用新的 run 名称。

审计时间：2026-10-24 14:53（Asia/Shanghai）。

启动尝试复核时间：2026-10-24 15:08（Asia/Shanghai）。

## 1. 启动前状态

| 项目 | 检查结果 |
|---|---|
| Git | 14:53 预检时 `main` HEAD 为 `58d9898fe617bd861d19fe1d750f7da5b4fcc27f`；本次复核 HEAD 为 `fa033b8e3b7c5e91bf3ef98a62d40e4d049361be`，仅本报告有未提交更新 |
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

## 3. 普通终端前台启动命令（尚未执行）

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

本次助手没有启动训练。正式启动后，从第二个终端检查 `config.json` 是否确为 1024 env、anchor=true、Actor/WM batch=1024、H=32、max_hours=4；另用 `pgrep -af sm-train` 查进程、`tail -f logs/4070/lafan1_q1_4070_anchor_retry1/console.log` 看日志、`nvidia-smi` 看显存。

启动日志应实际确认：`MotionLibrary(full): 40 clips, 441140 frames (2.451 h @50Hz)`；参数量约 `8.82M / 8.910M / 0.237M`；GPU 为 `cuda:0`；console log 和 TensorBoard event 位于该 run 目录。以上是启动核对项，不是已发生结果。

## 4. 训练监控与预算估算

之前单 clip、`anchor_obs=false` 的 42 iteration 资源验证测得约 5.1 vectorized iterations/s，峰值占用 6,175 MiB。它不是全 40 clips + anchor 的实际吞吐。本次启动后需用真实日志更新估算。

按 5.1 it/s 粗估：

| 目标 iteration | 估算耗时 |
|---:|---:|
| 20,000 | 1.09 h |
| 40,000 | 2.18 h |
| 60,000 | 3.27 h |
| 80,000 | 4.36 h |
| 160,000 | 8.71 h |
| 320,000 | 17.43 h |

因此 4 小时按旧吞吐约为 73,400 iterations，**不能保证达到 80,000**。首次编译、40 clip 数据和新增目标特征可能改变速度；等编译完成后用连续日志区间重算。

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

后续 session 将 `CKPT` 改为上一 session 的完整 checkpoint，`RUN` 改成新的唯一目录。若要在约 160,000 iteration 暂停，等待 `ckpt_000160000.pt` 完整出现再停止；恢复时仍保留 `total_iterations=320000`。按当前估算从 0 到 160k 约 8.7 小时；实际以日志为准。

## 6. 训练后评测与 Viewer（待执行）

训练结束后再评测全 40 条，避免与主训练争 GPU：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/eval_benchmark.py \
  --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 32 \
  --out logs/4070/lafan1_q1_4070_anchor_retry1/benchmark_40clips.json
```

记录 Mean Normalized Return、Frame-weighted Normalized Return、Failed Clips、JSON 中逐条轨迹的 steps/failure，并汇总平均 episode length。比较此前短训 baseline：**Mean Norm Return 0.3869，Failed Clips 40/40**。当前没有 Q1 checkpoint 或评测结果。

训练后串行回放 walk、dance、jump（每次运行一个）：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips walk1_subject1 --num-envs 1 --viewer native
env -u PYTHONPATH -u LD_LIBRARY_PATH /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips dance1_subject1 --num-envs 1 --viewer native
env -u PYTHONPATH -u LD_LIBRARY_PATH /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/4070/lafan1_q1_4070_anchor_retry1/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips jumps1_subject1 --num-envs 1 --viewer native
```

## 7. 当前报告状态

已完成：Git/环境/GPU/数据/进程/checkpoint 预检，CLI help 与配置解析，anchor goal shape smoke，启动/恢复命令整理。用户的 15:04 启动尝试创建过短暂 tmux session 并写出错误配置，但没有有效训练、PID/session 留存、完整 console、Q1 checkpoint、阶段性训练指标、benchmark 或 Q1 Viewer 回放。已整理 retry1 安全重试命令，尚未执行。以上结果不得用吞吐估计代替。

预检输出见 `logs/4070_audit/q1_*`；此前链路记录见 [`LOCAL_REPRO_4070.md`](LOCAL_REPRO_4070.md)，算法细节见 [`SUPERMIMIC_ALGORITHM_MAP.md`](SUPERMIMIC_ALGORITHM_MAP.md)。
