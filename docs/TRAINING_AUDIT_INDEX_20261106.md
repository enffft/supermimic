# M1-Piper / Go2 训练结果与审计材料索引

本批记录于 2026-11-06 汇总。包含昨晚 M1 v2 两条训练链的正式评测、动作饱和/WM/值函数梯度分析，以及 Go2 旧续训链累计时长和约 3 小时模型对照。新增诊断只做冻结策略评测，没有训练或修改权重。

## 当前结论与结果

1. [M1 饱和深入分析 + Go2 训练历史与约 3h 比较](M1_SATURATION_AND_GO2_TRAINING_HISTORY_20261106.md)：时钟追溯、四份 Go2 同口径复测、M1 快照/上一动作反馈/三部分梯度、改进顺序。
2. [昨晚 M1 v2 PPO / SuperMimic 正式评测](M1_V2_RESULTS_AND_FAILURE_ANALYSIS_20261106.md)：双方各 3.5h、6 组评测、局部 WM 物理响应、四份主要策略的 play 命令。
3. [Go2 新比较模型与旧对角续训权重的来源差异](GO2_POLICY_PROVENANCE_AND_GAIT_20261106.md)：解释当前 play 拖地，保留旧策略命令。

## 图与数据

- [Go2 时间/步态对比图](figures/GO2_TRAINING_HISTORY_MATCHED_20261106.png)。
- [M1 饱和与梯度分解图](figures/M1_SATURATION_GRADIENT_DIAGNOSTICS_20261106.png)。
- [M1 v2 正式结果与物理诊断图](figures/M1_V2_RESULTS_AND_DIAGNOSTICS_20261106.png)。
- [M1 v2 训练曲线](figures/M1_V2_SUPERMIMIC_TRAINING_20261106.png)。
- [本次新增原始结果、时钟、诊断脚本、实际源码节选](results/training_audit_20261106/)。
- [昨晚双方正式评测、完成记录与物理响应诊断](results/m1_v2_20261106/)。

JSON 标注 checkpoint 路径/SHA256、评测分母和诊断边界；完整训练日志/权重及部署包保留在实验主机，不上传。审计脚本使用主机训练工作区，源码哈希和节选用于追溯；本次提交不是完整训练代码发行版。

## 任务与历史背景（按文档日期阅读）

- [M1 v2：H32、过程/终端奖励、大范围目标和顺序训练设计](M1_V2_AND_SEQUENTIAL_TRAINING_20261105.md)。
- [M1 v1 训练设计审计](M1_PIPER_TRAINING_DESIGN_AUDIT_20261105.md)，分析的是旧 H8/近距离任务，不能与 v2 混用。
- [M1 PPO 基线](M1_PPO_BASELINE_20261105.md)、[M1 reward-only 基线](M1_REWARD_ONLY_BASELINE_20261104.md)、[2048 环境短测](M1_REWARD_ONLY_2048_PROBE_20261104.md)。
- [Go2 步态奖励](GO2_GAIT_REWARD.md)、[旧续训设置](GO2_TRAINING_20261103.md)、[旧中期进展](GO2_PROGRESS_20261103.md)、[旧最终评测](GO2_FINAL_EVALUATION_20261103.md)。
- [Go2 复用 PPO 链路](GO2_COMPARISON_STAGE1_PPO_20261104.md)、[部署对齐](GO2_COMPARISON_STAGE2_DEPLOY_20261104.md)、[从头训练比较与恢复记录](GO2_COMPARISON_STAGE3_RUNBOOK_20261104.md)。部署目录后来更新过，实际模型来源以 manifest 和本次审计为准。
- [此前发布的正式评测](FINAL_POLICY_EVALUATION_20261104.md)、[M1 v1 PPO/SM 评测](M1_PPO_SUPERMIMIC_EVALUATION_20261105.md)。
