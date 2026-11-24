# Go2W Motion Prior 与训练阶段

速度跟踪是 Go2W 的任务目标。Actor 在 SuperMimic 的 H8 imagination 中同时优化任务回报、Critic bootstrap 和逐帧运动状态模仿。101 条运动学参考（23 条 Qualified、78 条动态 Candidate）用于提供前期可探索的关节运动。Qualified 和 Candidate 的基础权重分别为 1.0 和 0.5。Actor 使用 50D 本体历史 × 5 加 7D goal，部署时根据当前速度目标和本体历史控制。 [Motion Bank 数据说明](go2w_motion_generation.md)

## 参考跟踪与真实采集

精确命令优先。连续命令在相同活跃轴和方向内选择最近的名义速度，随后只在选中的轨迹内匹配相位。启动、恒速、减速分别预计算连续 H8 起点。匹配通过腿位置、腿速度和有效轮速衡量状态，预测第一步比较匹配相位的下一帧。启动时 Actor 持有固定的目标速度。停车时参考选择使用当前运动速度，静止时使用站立参考。缺少匹配的三轴命令由任务奖励训练。

模仿使用按 0.25 rad、3 rad/s 和 5 rad/s 归一化的 Smooth L1，三项权重为 1.0、0.25、0.50。参考接地有效位掩码应用于轮速，imagination 存活概率用于加权。Actor 更新时冻结 WM 和 Critic 参数，模仿梯度经 WM 状态预测传给 Actor。Critic 目标是任务奖励。初始 `lambda_prior=0.60`，约 60 分钟内逐渐降至 0.21，Stage 2 内延续淡出到零。

| 时期 | 普通 Actor | RSI Actor | 连续参考 PD | RSI 短程 Actor |
|---|---:|---:|---:|---:|
| WM-only | 40% | 10% | 0% | 50% |
| Actor 学习初期 | 50% | 20% | 0% | 30% |
| Stage 1 后期 | 75% | 15% | 0% | 10% |
| Stage 2 淡出完成 | 100% | 0% | 0% | 0% |

采集来源以并发环境配额分派，episode 内保持来源。RSI 在 reset 时一次写入四轮支撑状态。短程采集持续 8 或 16 个控制步，由 Actor 动作加时间相关扰动，普通探索噪声在这些环境静默。到期触发 timeout，真实的终止前状态、实际执行动作和真实后继状态进入 Replay。WM Loss 和动力学 normalizer 只读取这些真实物理转移。参考 PD 的执行诊断见训练准备报告。 [训练准备与验证报告](validation/go2w_motion_prior/training_readiness.md)

WM 使用 Recent Replay 和有界 Historical Window Bank，目标配额为 50% 近期、30% 普通历史、20% 受保护成功经验。WM-only 时增采参考附近的真实窗口。受保护项采用普通 Actor 完成的成功平地命令和严格越障结果，失败窗口仍留在普通池。Actor 使用 Recent Replay 的真实历史栈，按方向和速度分箱均衡，最多 20% 复习已成功的自主状态。Terrain 阶段平地 Actor 起点目标占 40%。

## 课程与评测

- `flat_prior` 从纯平地开始。初始 vx、vy、wz 范围分别为 `[-0.15, 0.25]`、`[-0.10, 0.10]`、`[-0.20, 0.20]`，用十档向 `[-0.35, 0.75]`、`[-0.35, 0.35]`、`[-0.80, 0.80]` 扩张。普通 Actor 的单轴跟踪表现推进对应技能，速度等级与地形等级分别记录。15 分钟运行基础评测，30 分钟运行完整评测。
- `flat_autonomous` 在 1800 秒内淡出参考来源与模仿，增加耦合、三轴随机命令和 2–5 秒命令重采样。按 30 分钟完整物理评测检查分方向、切换和跌倒率。
- `terrain` 保留平地训练并引入八地形课程。切换时按配置更新配额、显式闭合 Replay 边界，再批量 reset。坡面、台阶与楼梯的向前命令、楼梯 18 秒时限和 ±3°、±5°、±10° 初始 yaw 扰动由地形课程配置。模型、优化器、normalizer 和经验池继续使用。

阶段由两个不同训练迭代的连续达标评测推进。初始和收尾评测用于报告。新版平地命令独立运行统一 4 秒窗口，切换使用每段 2 秒。提前触及平地边界的评测记录为无效，物理跌倒单独统计。历史协议仍可通过 `--protocol motion_prior` 调用。阈值、实际通过数及运行证据见准备报告。 [训练准备与验证报告](validation/go2w_motion_prior/training_readiness.md)

## 十小时训练与恢复

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate go2_unilab
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic-motion-prior
export PYTHONPATH=.:src:../UniLab/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run=runs/go2w_motion_prior_long

python tools/train_go2w.py --mode formal --profile motion_prior --fresh \
  --envs 4096 --seconds 36000 --max-stage terrain --output "$run"
```

预算按训练循环计时。环境初始化、预采集、独立物理评测和 checkpoint 写盘各自发生在预算外。输出目录必须尚未创建。

```bash
run=runs/go2w_motion_prior_long
tail -F "$run/train_log.csv"
python -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["stage"]); print(x["flat_skill_levels"]); print(x["domains"])' "$run/progress.json"
kill -TERM "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$run/launch.json")"
```

在前台终端也可按 Ctrl-C，在 iteration 边界停止并保存 `last.pt`。恢复命令的 `--seconds` 指额外循环时间。以下按十小时总预算算出剩余时间，并写入独立目录。

```bash
run=runs/go2w_motion_prior_long
remaining=$(python -c 'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=True,mmap=True); print(max(0,36000-x["wall_time"]))' "$run/last.pt")
python tools/train_go2w.py --mode formal --profile motion_prior \
  --resume "$run/last.pt" --envs 4096 --seconds "$remaining" \
  --max-stage terrain --output runs/go2w_motion_prior_long_resumed
```

每小时更新完整恢复文件，阶段切换后及安全停止时也写入 `last.pt`。恢复会加载模型、优化器、课程、Replay、阶段和学习用 RNG，然后批量重新采样物理环境并闭合原 episode 边界。阶段最佳推理权重同时保存在 `best_<stage>.pt`。完成预算后可使用以下方式独立评测和播放。

```bash
run=runs/go2w_motion_prior_long
for checkpoint in best last; do
  python -m tools.evaluate_go2w --protocol motion_prior_flat \
    --checkpoint "$run/$checkpoint.pt" --output "$run/${checkpoint}_flat.json" \
    --switches --seeds 2026 2027 2028
done
python tools/play_go2w.py --checkpoint "$run/best.pt" --vx 0 --vy 0.15
python tools/play_go2w.py --checkpoint "$run/best.pt" --vx 0 --wz -0.40
python tools/play_go2w.py --checkpoint "$run/best.pt" --terrain single_step_up --level 1 --vx 0.30
```

## 首轮信息和结果

正式运行使用 `08f6158`，按用户要求在 **1 小时 57 分 54 秒**安全停止。最终为 5,518 次迭代和 56,766,464 条真实 transition，其中 262,144 条来自循环前的预采集。纯平地环境始终为 4096 个，训练循环平均约 7,988 transitions/s，峰值 CUDA 分配约 4,077 MiB。按最终 100% 平地约束，阶段状态为 `flat_prior`，`ready_for_next_stage=false`。

| 累计时间 | 迭代 | 真实 transition | 平地通过率 | vx RMSE | vy RMSE | wz RMSE | 跌倒率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 初始 | 0 | 0 | 0.0% | 0.429 | 0.522 | 1.259 | 100.0% |
| 约 30 min | 1,599 | 16,633,856 | 0.0% | 0.476 | 0.517 | 2.552 | 100.0% |
| 约 60 min | 2,969 | 30,662,656 | 23.3% | 0.171 | 0.135 | 0.349 | 19.2% |
| 约 90 min | 4,280 | 44,089,344 | **50.0%** | 0.107 | 0.104 | 0.270 | **8.3%** |
| 停止 | 5,518 | 56,766,464 | 40.0% | 0.158 | 0.193 | 0.616 | 19.2% |

三个速度 RMSE 单位为 m/s、m/s、rad/s。评测从普通站姿开始，仅由 Actor 控制。

| 方向 | 最佳 checkpoint | 停止 checkpoint |
|---|---:|---:|
| 静止 | 100% | 100% |
| 前进 | 60% | 80% |
| 后退 | 100% | 75% |
| 左侧移 | 0% | 0% |
| 右侧移 | 0% | 0% |
| 正向原地转 | 100% | 20% |
| 负向原地转 | 80% | 60% |
| 前进＋转向 | 50% | 100% |
| 前进＋侧移 | 25% | 0% |
| 侧移＋转向 | 0% | 0% |
| 命令切换 | 33.3% | 16.7% |

最佳 checkpoint 对左右方向均产生了一些横向速度。高速侧移会跌倒，低速右移响应也不足。停止 checkpoint 的左移出现过冲并跌倒，右移动作接近短时停滞。90 分钟时原地旋转有效，随后的 checkpoint 对旋转跟踪明显回退。出现的进展在普通 reset 上测得。先验和默认姿态奖励调整的各自贡献仍需对照，当前结果支持完成首轮实验并保留在 `flat_prior`。

最终 World Model 在 2,196 个普通 Actor 真实留出窗口上，H1/H8 线速度 RMSE 为 0.029/0.098 m/s，角速度为 0.290/0.521 rad/s，腿位置为 0.014/0.092 rad，腿速度为 1.114/1.769 rad/s。23 个参考 PD 窗口的 H8 角速度 RMSE 为 0.836 rad/s，23 个扰动 Actor 窗口的 H8 线速度 RMSE 为 0.133 m/s。完整逐类别结果记录在运行目录。采集源累计计数曾在 float32 的 2²⁴ 饱和，修复采用 int64。饱和前的日志及并发配额显示普通 Actor、RSI Actor、参考 PD 约为 70%／20%／10%。

恢复 checkpoint、真实 rollout 视频、评测 JSON 和完整 `training_report.md` 保存在 `runs/go2w_motion_prior/`。

[完整训练与评测报告](validation/go2w_motion_prior/training_report.md) · [学习曲线](validation/go2w_motion_prior/learning_curves.png)
