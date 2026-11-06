# Go2 两份 SuperMimic 策略的来源与步态差异

2026-11-06。用户观察当前 play 命令步态拖地。本次关闭残留的 play PID 3845238，确认 GPU 进程退出；没有训练、更新模型或覆盖部署包。

后续已查明旧最终权重的 Actor 链累计约 4.29h，连 WM 来源约 4.55h；累计约 3h 的中间权重也尚未形成最终低速对角步态。详见 [时长与同口径复测](M1_SATURATION_AND_GO2_TRAINING_HISTORY_20261106.md)，不能把中断恢复本身视为效果改善的已证实原因。

## 已确认：当前部署目录和之前的对角续训策略不是同一份权重

- 当前 `~/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt` 来自 `logs/go2_comparison/supermimic_2p5h_seed2026/ckpt_latest.pt`，是 PPO 比较实验从头训练约 2.5 小时的策略。今天逐张量比较 Actor 权重，导出与源模型完全一致。
- 之前对角步态改善的策略为 `logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt`。该轮从 iteration17000 的稳定策略继续训练，用户于约 2.37 小时停止，评测使用保存的 iteration34000；它不是从零训练同预算模型。
- 旧权重仍存在，并已通过当前导出加载函数及 actuator contract 检查；未覆盖或替换。

## 原有正式评测已经显示步态差距

以下取两次已有名义平地评测的 `vx=0.3 m/s` 行，均为 seed2026、64 个 10 秒 episode；没有在今天重新评测。

| 指标 | 旧对角续训策略 | 新从头训练比较策略 |
|---|---:|---:|
| 实际 vx (m/s) | 0.289 | 0.282 |
| 足滑 RMS (m/s) | 0.167 | 0.479 |
| 几何对角双足支撑比例 | 48.5% | 0% |
| FL/FR/RL/RR 每 episode 峰值离地高度均值 (cm) | 5.14 / 3.44 / 3.47 / 5.04 | 0.40 / 0.61 / 6.31 / 0.51 |

足滑/支撑使用几何高度代理，非接触力真值；峰值不是每一步平均高度。这些指标支持用户观察到的拖地和不对称步态。速度跟随与零跌倒不足以验收步态。

两份 checkpoint 的步态参数均为：抬脚 `0`、对角 `-0.15`、滑脚 `-0.20`，网络/WM/Actor 主要配置相同。另有实际配置差异：旧策略想象 Actor 输入噪声开启，新比较策略关闭；没有消融实验，不能把它单独认定为根因。

## 解释与后续判断

最直接的来源差异是旧策略已有稳定训练历史，之后专门续训塑造步态；新比较策略从随机网络和 A 阶段课程开始。相同奖励和相近单段时长不会保证同一局部最优。

对角奖励是瞬时足端高度的软协调项，未指定交替相位或步频；滑脚是软成本；显式抬脚奖励按用户要求关闭。速度收益可能与不对称、低抬脚的运动同时存在。目前结果说明新策略尚未被这些软约束塑造成旧策略的步态，不能据此证明哪一个单独设计因素导致退化。应同时验收滑脚、对角支撑、四足抬升和跟踪，保留最佳 checkpoint，而非只根据训练时间和零跌倒选择最终模型。

旧模型可用于查看较好步态，但与从零训练的 PPO 做算法比较时必须注明额外训练历史。当前部署包保持原样，避免混淆两种实验来源。

Ctrl+C 后出现的 `GLXBadContext` 属于退出阶段的图形上下文清理错误；cached module 日志是内核缓存加载信息，不是步态退化证据。本次残留进程已终止。

## 旧对角续训策略 play

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt \
  --num-envs 1 --viewer native --vx 0.3 --vy 0 --yaw-rate 0 \
  --velocity-arrow --actual-velocity-arrow
```

将 `--vx 0.3` 改为 `--vx 0.5` 可以查看中速步态。退出优先按 Esc / 关闭窗口。

证据：[旧评测报告](GO2_FINAL_EVALUATION_20261103.md)、[新比较策略评测报告](FINAL_POLICY_EVALUATION_20261104.md)，以及这两份报告所引用的原始 JSON。当前部署来源见 `~/rl_sar/policy/go2/supermimic_velocity/manifest.json`。
