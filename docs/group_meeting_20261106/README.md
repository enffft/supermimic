# SuperMimic / PPO 组会素材包 · 2026-11-06

**按最新要求，直接使用[一页简版对比](QUICK_COMPARISON.md)：Go2比速度，M1比L跟随误差与M到点成功率。只需要两张现成图；下面的长报告与其他图片均作备查。**

已完成 **17张16:9高清中文图（3840×2160 PNG + SVG）**、三份详细分析、20页组会提纲与12个追问回答、实验证据索引及play命令。按用户后续要求不录视频、不做PPT，不创建视频占位。

建议先读 [组会提纲](MEETING_OUTLINE.md)，按下表挑图；个人研究阅读再看三份分析。视频自行录制时使用 [Play命令](PLAY_COMMANDS.md)。

## 1. 阅读入口

| 文件 | 用途 |
|---|---|
| [MEETING_OUTLINE.md](MEETING_OUTLINE.md) | 20页建议标题、核心点、图片/演示、讲解结论、现场追问；12问参考回答 |
| [GO2_PPO_SUPERMIMIC_ANALYSIS.md](GO2_PPO_SUPERMIMIC_ANALYSIS.md) | Go2任务/课程、完整15项reward、网络、真实九指令结果、旧训练链与步态发展 |
| [M1_PIPER_PPO_SUPERMIMIC_ANALYSIS.md](M1_PIPER_PPO_SUPERMIMIC_ANALYSIS.md) | v2任务/Goal8/Action22、完整20项reward、endpoint、正式统计、WM/Actor/Critic退化 |
| [ALGORITHM_COMPARISON_AND_REFLECTION.md](ALGORITHM_COMPARISON_AND_REFLECTION.md) | PPO与BPTT数学差异、事实/假设分级、6个开源项目核验、优先验证实验 |
| [EXPERIMENT_PROVENANCE.md](EXPERIMENT_PROVENANCE.md) | 数字的原始来源、checkpoint SHA、评估条件、证据等级与未验证边界 |
| [PLAY_COMMANDS.md](PLAY_COMMANDS.md) | Go2两算法与旧步态；M1纯L、同点M、远/低/随机点的命令 |

## 2. 图片索引

全部白底科研配色，PPO深蓝、SM青绿；使用系统Noto CJK字体。SVG字体已转路径，换电脑无需安装相同中文字体。PNG可直接插入PPT；图表带实验版本与口径脚注。每张图对应的真实输入文件见 [figure_registry.json](data/figure_registry.json)。

| 编号 | PNG | SVG | 建议用途 |
|---|---|---|---|
| G01 | [训练配置](figures/G01_go2_training_config.png) | [矢量](figures/G01_go2_training_config.svg) | Go2从零2.5h配置与样本量 |
| G02 | [九指令结果表](figures/G02_go2_velocity_results.png) | [矢量](figures/G02_go2_velocity_results.svg) | 全部命令/实速/误差/跌倒 |
| G03 | [全部指令误差图](figures/G03_go2_velocity_comparison.png) | [矢量](figures/G03_go2_velocity_comparison.svg) | 不省略失败命令 |
| G04 | [旧训练链步态发展](figures/G04_go2_gait_history.png) | [矢量](figures/G04_go2_gait_history.svg) | 2.5h与旧约3h/4.55h分开解释 |
| G05 | [奖励汇总](figures/G05_go2_reward_summary.png) | [矢量](figures/G05_go2_reward_summary.svg) | 抬脚关闭、对角与滑脚开启 |
| G06 | [等时长步态指标](figures/G06_go2_fair_gait.png) | [矢量](figures/G06_go2_fair_gait.svg) | 与G04不同的公平算法比较 |
| M01 | [v2训练配置](figures/M01_m1_training_config.png) | [矢量](figures/M01_m1_training_config.svg) | 各3.5h、真实样本差5.736倍 |
| M02 | [正式结果](figures/M02_m1_final_results.png) | [矢量](figures/M02_m1_final_results.svg) | 分开L/M与goal/episode分母 |
| M03 | [成功与高度分桶](figures/M03_m1_reach_success.png) | [矢量](figures/M03_m1_reach_success.svg) | 低位目标困难，删失不混算 |
| M04 | [训练退化与WM误差](figures/M04_m1_training_failure.png) | [矢量](figures/M04_m1_training_failure.svg) | 日志/固定输入/真实姿态/局部模型四种证据 |
| M05 | [WM物理梯度反号](figures/M05_m1_wm_gradient_audit.png) | [矢量](figures/M05_m1_wm_gradient_audit.svg) | 已复核的三个局部反号例子 |
| M06 | [任务与终点奖励](figures/M06_m1_task_rewards.png) | [矢量](figures/M06_m1_task_rewards.svg) | 共享真实reward与SM额外目标 |
| M07 | [梯度分解与历史反馈](figures/M07_m1_gradient_and_feedback.png) | [矢量](figures/M07_m1_gradient_and_feedback.svg) | Critic信号与动作历史放大机制 |
| A01 | [算法架构](figures/A01_algorithm_architecture.png) | [矢量](figures/A01_algorithm_architecture.svg) | 真实GAE/PPO vs Replay/WM/BPTT |
| A02 | [M1失败机制假设](figures/A02_m1_failure_mechanism.png) | [矢量](figures/A02_m1_failure_mechanism.svg) | 虚线表示因果消融尚未完成 |
| A03 | [四条训练链诊断曲线](figures/A03_training_curves.png) | [矢量](figures/A03_training_curves.svg) | 分开训练reward与Replay误差，不跨目标比绝对return |
| A04 | [真实网络维度](figures/A04_network_dimensions.png) | [矢量](figures/A04_network_dimensions.svg) | Go2 213D、M1 728D，控制语义不同 |

## 3. 可复现脚本

已有data快照足够重画图片，不需要加载checkpoint或使用GPU：

```bash
cd /home/yifeiy/ws_supermimic
PY=/home/yifeiy/miniconda3/envs/supermimic/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" docs/group_meeting_20261106/scripts/plot_tables.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" docs/group_meeting_20261106/scripts/plot_training_curves.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" docs/group_meeting_20261106/scripts/plot_architecture.py
"$PY" docs/group_meeting_20261106/scripts/assemble_documents.py
"$PY" docs/group_meeting_20261106/scripts/validate_deliverables.py
```

`style.py`统一字体/颜色/输出尺寸；`build_documents.py`从JSON生成完整reward与诊断表；`assemble_documents.py`用保留的模板重建三份文档、索引与数值来源表。

`prepare_data.py`读取本机日志和checkpoint重提取证据；`verify_evidence.py`在CPU核对导出与TB历史时钟。它们不是重画图片的必需步骤；别的机器没有这些本机权重路径时无需运行。`fetch_sources.py`下载公开仓库的有界文本证据，不安装依赖、不训练；网络版本可能随时间改变，所以本批固定SHA与原文件另存于`external_sources/`。

## 4. 证据等级与展示注意

直接从本机checkpoint/config/TensorBoard复核：网络维度/参数量、H32、endpoint权重、真实步数/样本、Actor和WM更新、最终课程阶段、旧训练累计时钟；从原始评测JSON复核：九指令表现、M1成功/失败/删失与分桶；CPU直接确认两份Go2与M1 PPO推理包Actor完全等于来源，动作误差0。

C++/sim2sim历史检查、人工启动过程与论文细节引用已有报告，本次不重跑。当前源码和前次审计的10个关键源码hash相同，仍缺每次训练完整dirty源码封存。外部项目经验有本次联网文件证据，性能数字未在本机复现。

材料可用于组会。需要你自行完成录像和PPT排版；没有新的实机/扰动/多seed鲁棒性证据，也没有完整因果消融。旧4.55h模型、M1 v1/v2、PPO真实样本优势、SM额外endpoint与统计分母均已显式区分。

本任务仅新增此目录，保留此前未提交训练工作；没有修改训练代码/奖励/权重，没有启动训练、GPU评测或viewer，没有commit或push。

## 5. 最终交付检查

[检查记录](data/deliverable_validation.json) 已通过：17组PNG/SVG的尺寸和路径字体、60,811条数值JSON Pointer、本地文档链接、17种play场景的参数接口及checkpoint存在性；原始证据、源码和权重SHA256以及Git分支/HEAD/原有工作区状态一致。检查脚本仅做CPU文件核对与`--print`命令生成，没有启动viewer；图片另经总览与重点单图目视检查。此检查保证材料与留存证据一致，不替代新仿真、多seed或因果消融。
