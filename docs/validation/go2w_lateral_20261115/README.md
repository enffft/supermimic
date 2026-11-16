# Go2W 横移诊断证据

对应[分析报告](../../go2w_lateral_diagnosis_20261115.md)。本目录可直接读取，不含权重或长训 Replay。

- `physics/summary.json`：36 组设置、432 次短程首回合物理诊断；确定性命令、探索探测、奖励分解、腿运动和轮胎间隙。
- `go2_transfer/summary.json`：96 次 Go2 腿策略零样本移植探测，四轮速度目标置零，全部跌倒；无有效速度样本的统计为 null。
- `config_analysis.json`：奖励解析、819,200 次命令采样、局部 FK 与物理探索尺度。
- `model_audit.json`：本地模型质量、轮胎参数、编译模型选项核对。
- `ppo_first_update_kl.json` / `ppo_first_update_kl_clip.json`：两次相同原生 PPO 首轮更新诊断，第二次补充 ratio；损失与原正式训练第一轮相同。仅在内存更新，没有保存新模型。
- `target_spectrum.json`：腿目标频谱。右移中有跌倒的组包含 reset 后数据，不用于报告主要频谱结论；报告的左移组全程存活。
- `github_sources.json`：本次经 GitHub API 读取的公开源码固定版本链接；源码缓存保留本地。
- `provenance.json`：源码快照、checkpoint SHA256、WM 只读检查及源缓存摘要。
- `protocol/`：本次诊断和图表脚本。
- `sha256.json`：本目录文件摘要清单。

完整物理轨迹在 `/home/yifeiy/ws_go2_unilab/research/go2w_lateral_20261115/physics/trajectories.npz`。所有几何离地统计是轮胎碰撞网格最低点与地面的间隙，不是接触力。主诊断使用同一 reset seed 的 12 个合法位置/朝向扰动，不能解释成多训练 seed 验证。

从仓库根目录复现主物理诊断（为输出另取一个不存在的目录）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/yifeiy/ws_go2_unilab/runs/go2w_ppo_2h_20261115/source_snapshot/src \
/home/yifeiy/miniconda3/envs/go2_unilab/bin/python tools/diagnose_go2w_lateral.py \
  --initial /home/yifeiy/ws_go2_unilab/runs/go2w_ppo_2h_20261115/bootstrap/actor.pt \
  --ppo /home/yifeiy/ws_go2_unilab/runs/go2w_ppo_2h_20261115/training/last.pt \
  --supermimic /home/yifeiy/ws_go2_unilab/runs/go2w_omni_v3_2h_20261114/training/last.pt \
  --output /tmp/go2w_lateral_new_run
```

同样设置 `PYTHONPATH` 和解释器后，运行 `tools/analyze_go2w_lateral_config.py --ppo <training/ppo_resume.pt> --output <new.json>` 可复算配置检查；`tools/diagnose_go2w_ppo_kl.py --actor <bootstrap/actor.pt> --output <new.json>` 可复现一轮内存更新诊断。跨机器人零样本探测在物理诊断命令增加 `--go2-from <Go2 checkpoint> --transfer-only`。

外部论文原文：[DreamWaQ](https://arxiv.org/html/2301.10602)、[Learning robust autonomous navigation and locomotion for wheeled-legged robots](https://arxiv.org/html/2405.01792v1)。外部源码与论文支持方法比较，不构成当前配置达到横移目标的独立验证。
