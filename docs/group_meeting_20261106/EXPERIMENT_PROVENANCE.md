# 实验证据索引与来源等级

2026-11-06。本次输出不代表训练代码发行版。Git main / HEAD `f3f29e86073593277244b1d2d307c2b28b8411cc`，原有dirty实现保留；未切分支、建worktree、reset/stash/clean、修改训练实现或权重、commit/push。只新增本组会目录。

## 1. 怎样从图追到原始数字

1. [figure_registry.json](data/figure_registry.json) 给出每张图的输入JSON与绘图脚本。
2. [numeric_evidence.csv](data/numeric_evidence.csv) 给出汇总指标、配置与诊断曲线的JSON Pointer和完整精度数值；目标分桶从原始goals列表重新统计，不含censored。
3. [experiment_manifest.json](data/experiment_manifest.json) 将data快照映射回本机原始路径、SHA256，并列出checkpoint、网络参数形状、版本和审计起点状态。
4. [evidence_verification.json](data/evidence_verification.json) 保存CPU导出一致性、训练配置共同字段比较、原始TensorBoard时钟复核和前次审计源码哈希一致性。
5. [source_snapshot](data/source_snapshot/) 固定本次读取的相关本机源码；[upstream_runtime](data/upstream_runtime/) 固定本机安装版本PPO关键实现。它们不是重新构建出的历史完整源码。

原始JSON中的null保留为无样本/不适用，不擅自填0。训练曲线只保留有限标量，不把warmup的`nan(n=0)`解释成权重NaN；原始TensorBoard路径和hash仍在manifest。

## 2. 模型身份与直接checkpoint核验


| 身份 | 本机checkpoint | SHA256 | iteration |
|---|---|---|---|
| go2_ppo | logs/go2_comparison/ppo_2p5h_seed2026/model_758.pt | 1a1a0cde9e04d32c909b519088864ba984396b6c71fc548d440573736a1be0fb | 758 |
| go2_supermimic | logs/go2_comparison/supermimic_2p5h_seed2026/ckpt_latest.pt | 816923703b1998cf1faf669ca43781798006b1339c8cbc5ac30f03a71f853565 | 18600 |
| m1_ppo | logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/model_533.pt | b66319817f8039360d6360634182fb9e729b11f2f4a4ecb7261ab1f6ab1039a5 | 533 |
| m1_supermimic | logs/m1_reward/v2_lm50_2048_3p5h_seed2026/ckpt_latest.pt | d49bcd84e3c85a377e9524ab5cfbf9de17de9192bcd937d308cc4684450b5eef | 2979 |
| go2_old_22000 | logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0022000.pt | 8a2e766248fd78329810a855fece1e7d454d04429109fd48290c58734f803a04 | 22000 |
| go2_old_24000 | logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0024000.pt | c58d90e82ed6e3f56510dd281e341c20a5b4c33aa62c5fc0d60f2f2cf03fe26b | 24000 |
| go2_old_34000 | logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt | 0b3d8e4bb26173d13a6ce0b99d5eecf43005927a8ab2e01947c4d0311009487a | 34000 |
| go2_ppo_inference | /home/yifeiy/rl_sar/policy/go2/ppo_velocity/policy_inference.pt | 5b103bca280e0a1cece3bb9715291e73eebfcc8efb6ae4ee6270c6438dd80b10 | 推理包 |
| go2_sm_inference | /home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt | 3bcae044d39d61553374d21caf7df39d7bd96d9ada6cfa237d7d05dd188bff0c | 推理包 |
| m1_ppo_inference | /home/yifeiy/ws_supermimic/logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/policy_inference.pt | 227e43edd1d4e9352fed1a910d01c585c4f1b9ee7b41763bc3d8e7fc91c9cec9 | 推理包 |

Go2两种部署包和M1 PPO推理包已在CPU逐张量核对来源Actor，完全相同；64组随机输入的动作最大误差均0。检查的是导出和身份，没有运行新的真实仿真，也不证明实机闭环安全或新viewer可用。

## 3. 数字—条件—来源映射


| 数据族 | 主要data文件 | 评估条件 / checkpoint / 口径 | 证据级别 |
|---|---|---|---|
| Go2等时长配置、样本、更新 | go2_*config、*training_progress、*training_completion、experiment_manifest | 从零seed2026；3072环境；学习循环约2.5h；checkpoint形状/metadata | 本次直接读取配置、checkpoint、日志 |
| Go2九命令正式结果 | go2_{ppo,supermimic}_{evaluation,extra} | 各指令64结束episode，10s、seed2026、deterministic/noDR/noise；指向对应部署包 | 本次读取原始评测JSON；不重跑 |
| Go2历史步态 | go2_new_cold_2p5h、go2_old_chain_2p95h、go2_old_actor_2p96h、go2_old_final_chain_4p55h | 4权重×4命令，各16episode、10s、seed2026；排除起步1s足端指标 | 本次读原始JSON与ckpt哈希 |
| Go2累计训练时间 | go2_lineage_clocks、evidence_verification.old_clocks | time/hours原始TB；WM仅导入、Actor链不含source Actor；保存与未保存段分开 | 本次独立复读原始TB并匹配checkpoint来源 |
| M1正式结果 | m1_{ppo,supermimic}_{balanced,forward,manipulation} | v2，各64env60s，seed123；共享真实reward、SM独有endpoint；M成功换点、删失单列 | 本次读原始JSON、config与ckpt |
| M1目标高度/初始距离桶 | m1_*_manipulation.goals | 仅已结束goal；生成脚本排除censored；成功后随机轨迹分叉 | 从原始逐goal记录本次重新汇总 |
| M1最终姿态/指令敏感性 | m1_policy_posture_probe | 4env5s balanced，统计1–5s；L固定.12；冻结策略 | 已有真实MuJoCo probe，本次只读 |
| M1模型H1/8/16/32误差 | m1_wm_response_probe、*_eps0.01 | 一个名义姿态，12分支常动作，epsilon.03/.01，目标[1.5,0,.8]；不是闭环或全分布 | 已有真实FD/WM probe，本次只读 |
| M1全快照、动作反馈 | m1_fixed_input_snapshots、m1_action_history_feedback | 43快照固定合成物理输入；反馈64次，无物理/优化；非历史Replay | 已有CPU反事实JSON；本次重绘 |
| M1分项梯度 | m1_gradient_components | 6快照、4相同起点、H32、frozenWM/Critic，J/H，裁剪前；不是历史训练贡献百分比 | 已有冻结梯度诊断，本次只读 |
| M1训练时序/正则实际值 | training_curves、m1_selected_training_scalars、m1_reward_scalar_audit | 原始TB的真实/想象标签分别保留；重复step不强行合并 | 本次读TB并保留已有导出 |
| Reward公式/默认正则 | source_snapshot/.../tasks/{velocity_reward,go2_gait,m1_reward}.py + 保存config | 显式字段从config读取；固定系数从审计源码和历史报告交叉核对；全部dt一次 | 审计源码与前次审计10文件哈希相同；非完整训练时源码封存 |
| 开源机制 | external_sources/index.json、各repo/provenance.json | 6仓库固定SHA，README及选定实现；未安装、运行或复现实验 | 本次联网核验公开说明与代码 |


## 4. 本次新增复核结果


| 检查 | 结果 |
|---|---|
| 共同环境配置字段 | Go2无差异；M1仅PPO未使用WM的Replay容量2、SM256（共同键比较） |
| 最终课程 | checkpoint：Go2 PPO stage1(B)、SM stage2(C) |
| 导出包 | Go2 PPO/SM与M1 PPO：Actor tensor exact equality，动作误差0 |
| 原始时钟 | WM .251819h；base Actor2.004372h；gait22000/24000/34000/34600分别.695349/.956017/2.289729/2.372166h |
| 前次审计源hash | objective/policy_trainer/rollout/tasks/nets/m1_reward/go2_gait/loop/evaluate_go2/training.m1_reward共10项均相同 |
| 硬件与运行环境 | RTX4070 SUPER 12GiB；Conda supermimic；包版本见manifest；本任务未用venv或启动GPU训练/评测 |


## 5. 仍依赖历史记录与未验证边界

- 旧训练具体人工启动/中断背景、旧端到端部署C++/sim2sim误差和硬件部署边界，引用既有报告；本次没有重建C++、重新跑sim2sim或实机。当前导出身份另做了CPU核验。
- Go2固定reward系数与历史实现用保存配置、历史报告和当前源码核对；缺少每次训练完整未提交源码归档，不能断言全部训练文件已通过本次HEAD完整复现。M1本次reward源码hash与当天前次审计一致，仍不等于训练启动时全仓库快照。
- 当前源码所存学习式term head存在，但本轮Actor实际用解析高度/倾角终止；通用论文描述与任务实现分开。
- WM反号是局部物理证据；Critic方向没有真实梯度真值；反馈闭环因果图尚无完整消融。Replay分布恶化、共享多任务冲突和低目标不可达未证明。
- 单训练seed、无DR独立名义评测不能给出真实硬件或随机地形鲁棒性结论。没有新长程训练、录像或PPT文件。
- 论文细节的页码/原实验硬件来自仓库PAPER_SPEC与既有报告，本次未重新下载核对论文全文；公开复现的分钟数也仅为作者报告，不是本机复测。

## 6. 必读历史材料索引

- [FINAL_POLICY_EVALUATION_20261104](../FINAL_POLICY_EVALUATION_20261104.md)
- [GO2_POLICY_PROVENANCE_AND_GAIT_20261106](../GO2_POLICY_PROVENANCE_AND_GAIT_20261106.md)
- [GO2_FINAL_EVALUATION_20261103](../GO2_FINAL_EVALUATION_20261103.md)
- [GO2_TRAINING_20261103](../GO2_TRAINING_20261103.md)
- [GO2_COMPARISON_STAGE1_PPO_20261104](../GO2_COMPARISON_STAGE1_PPO_20261104.md)
- [GO2_COMPARISON_STAGE2_DEPLOY_20261104](../GO2_COMPARISON_STAGE2_DEPLOY_20261104.md)
- [GO2_COMPARISON_STAGE3_RUNBOOK_20261104](../GO2_COMPARISON_STAGE3_RUNBOOK_20261104.md)
- [GO2_GAIT_REWARD](../GO2_GAIT_REWARD.md)
- [M1_PPO_BASELINE_20261105](../M1_PPO_BASELINE_20261105.md)
- [M1_PPO_SUPERMIMIC_EVALUATION_20261105](../M1_PPO_SUPERMIMIC_EVALUATION_20261105.md)
- [M1_V2_AND_SEQUENTIAL_TRAINING_20261105](../M1_V2_AND_SEQUENTIAL_TRAINING_20261105.md)
- [M1_V2_RESULTS_AND_FAILURE_ANALYSIS_20261106](../M1_V2_RESULTS_AND_FAILURE_ANALYSIS_20261106.md)
- [M1_SATURATION_AND_GO2_TRAINING_HISTORY_20261106](../M1_SATURATION_AND_GO2_TRAINING_HISTORY_20261106.md)
- [M1_PIPER_TRAINING_DESIGN_AUDIT_20261105](../M1_PIPER_TRAINING_DESIGN_AUDIT_20261105.md)
- [M1_REWARD_ONLY_BASELINE_20261104](../M1_REWARD_ONLY_BASELINE_20261104.md)
- [M1_ACTION22_P0_P1_REMEDIATION_20261104](../M1_ACTION22_P0_P1_REMEDIATION_20261104.md)
- [TRAINING_AUDIT_INDEX_20261106](../TRAINING_AUDIT_INDEX_20261106.md)
- [SUPERMIMIC_ALGORITHM_MAP](../SUPERMIMIC_ALGORITHM_MAP.md)
- [SUPERMIMIC_CODE_WALKTHROUGH](../SUPERMIMIC_CODE_WALKTHROUGH.md)
- [PAPER_SPEC](../PAPER_SPEC.md)
