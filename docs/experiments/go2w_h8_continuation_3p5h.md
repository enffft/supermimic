# Go2W H8 额外3.5小时续训结果

后续同条件复核发现：中途19796轮的平地84.4%、单级下台阶80.0%、连续下楼73.6%，均明显优于下表最终模型；18401轮平地为98.7%。最后约30分钟出现能力退化。完整分轴、分级评测和视频见[累计7.5小时物理评测复核](go2w_h8_7p5h_evaluation.md)。以下自动汇总仍保留为最终21190轮的真实结果。

状态：已完成额外12600秒有效训练循环。
父检查点：`/home/yifeiy/ws_go2_unilab/runs/go2w_unified_h8_4h_20261116/last.pt`；续训目录：`/home/yifeiy/ws_go2_unilab/runs/go2w_unified_h8_resume_3p5h_20261117`。
恢复已有Actor、Critic、WM、三套优化器、Replay、历史池和课程；环境重采样，非逐步完全复现。
H8、wm_window8、4096env、1024buffer、WM/Actor batch2048、奖励及采样设置保持不变。
方向检查保留诊断，依照用户明确授权继续训练。原始WM资产SHA未变。

本次有效循环：12601.150秒；累计：27001.999秒。
本次iteration：9629；新增真实转移：98603008。
新增更新：{'wm': 96290, 'actor': 9629, 'critic': 192580}；GPU峰值：4062.03466796875MiB。
最终14项课程等级：[10, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 6, 1, 1]。

评测为冻结Actor的真实物理847条固定条件，每条最多10秒；11个确定性初始yaw并非独立随机实验。

|任务|续训前通过率|续训后通过率|续训后跌倒率|
|---|---:|---:|---:|
|flat|20.8%|33.8%|48.1%|
|rough|0.0%|20.0%|11.8%|
|slope_up|0.0%|100.0%|0.0%|
|slope_down|0.0%|100.0%|0.0%|
|single_step_up|0.0%|0.0%|100.0%|
|single_step_down|14.5%|0.0%|100.0%|
|multi_stairs_up|0.0%|0.0%|99.1%|
|multi_stairs_down|0.0%|0.0%|99.1%|

分轴速度、RMSE和每级台阶明细见[完整物理评测](../validation/go2w_unified_h8_continuation/final_11yaw.json)。
[平地视频](../validation/go2w_unified_h8_continuation/flat.mp4) · [台阶视频](../validation/go2w_unified_h8_continuation/stairs.mp4)。
训练时间完成与运动能力达标分开判断；通过率来自物理评测，不以loss或想象奖励替代。
