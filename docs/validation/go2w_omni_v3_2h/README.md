# Go2W v3 两小时实验依据

对应[阶段报告](../../go2w_results_20261115.md)。数据来自 2026-11-14 的本地实验，2026-11-15 整理发布。

|路径|用途|
|---|---|
|`evaluation/bootstrap.json`、`evaluation/final.json`|初始化与最终 660 次物理评测|
|`evaluation/iteration_*.json`|中途检查点评测，含单独标记的扩展命令评测|
|`evaluation/*_extended.json`|252 次平地补测，±0.35 m/s 横移超出训练范围|
|`evaluation/comparison.csv`|初始化与最终逐场景对照，CSV 行尾规范为 LF|
|`evaluation/wm_*.json`|共同物理留出集与初始化留出集上的 WM 精度|
|`training/args.json`、`training/task_contract.json`、`config/`|实际训练参数、任务协议与 Hydra 配置快照|
|`training/summary.json`、`training/train_log.csv`|训练量和训练内指标|
|`continuity_verification.json`、`training/world_model_*.json`|WM 加载、持续优化与发布证据|
|`earlier/`|前期覆盖、关节范围和失败宽范围实验依据|
|`protocol/`|标准、扩展命令和 WM 留出评测脚本快照|
|`source_manifest.json`、`provenance.json`|实际训练源码摘要和来源记录|
|`final_verification.json`、`snapshot_tests.log`|实验阶段验证结果|
|`training.png`、`physics_comparison.png`|训练曲线及物理结果对照|
|`manifest.json`|发布附件的 SHA256 清单|
|`publication_verification.json`|本次发布前的数字、附件和相对链接核对|

模型权重、Replay、物理窗口二进制及完整训练源快照仍在本地，不包含在此证据包中。JSON 内的绝对路径是原始实验来源，跨机器不可直接运行。本文及主报告的相对链接可在 GitHub 阅读。

`protocol/evaluate_go2w_extended.py` 为大命令补测版本，设置和结果见 `*_extended.json`。本次提交用于报告归档，不替代实验实现的单独代码审查。
