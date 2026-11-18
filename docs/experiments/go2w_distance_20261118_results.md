# Go2W 距离课程续训：实际结果

训练状态：完成额外12600秒有效训练。
运行目录：`/home/yifeiy/ws_go2_unilab/runs/go2w_distance_resume_3p5h_20261118`；父断点：`/home/yifeiy/ws_go2_unilab/runs/go2w_unified_h8_resume_3p5h_20261117/last.pt`。
这是显式课程/采样迁移：保留三网络、归一化、三优化器、Replay和历史池，重置课程及采样统计。不是逐步等价恢复。
H8/window8、4096env、1024Replay、2048batch、单Actor及21项reward不变。
[实现及验证报告](go2w_distance_curriculum_20261118.md) · [reward公式与限制](go2w_h8_reward_analysis.md)

新增有效循环 12600.358s；新增轮数 9473；新增转移 97001472。
实际更新 {'wm': 94730, 'actor': 9473, 'critic': 189460}；峰值GPU 4063.24951171875MiB。

冻结Actor、真实物理847条固定条件，每条首次episode最多10秒；11个固定yaw不代表11次独立训练。
课程基座越线与严格四轮/10帧物理成功分别统计。训练完成不代表能力通过。

|任务|迁移前成功率|续训后成功率|续训后跌倒率|
|---|---:|---:|---:|
|flat|33.8%|59.7%|0.0%|
|rough|20.0%|50.0%|0.0%|
|slope_up|100.0%|100.0%|0.0%|
|slope_down|100.0%|100.0%|0.0%|
|single_step_up|0.0%|8.2%|12.7%|
|single_step_down|0.0%|74.5%|25.5%|
|multi_stairs_up|0.0%|0.0%|7.3%|
|multi_stairs_down|0.0%|67.3%|32.7%|

## 平地分轴跟踪

|命令|成功/条件|平均vx,vy,wz|平均RMSE|
|---|---:|---|---|
|flat_stop|11/11|[0.0017, -0.0161, 0.0028]|[0.0276, 0.0227, 0.0544]|
|flat_forward|11/11|[0.2748, 0.0008, 0.0016]|[0.0415, 0.019, 0.0492]|
|flat_backward|11/11|[-0.1961, -0.0563, 0.0835]|[0.0638, 0.0612, 0.1341]|
|flat_left|2/11|[0.0034, 0.1183, -0.0523]|[0.045, 0.1071, 0.1564]|
|flat_right|0/11|[-0.1735, -0.0603, 0.0576]|[0.1861, 0.1452, 0.1243]|
|flat_yaw_left|0/11|[-0.0206, -0.0311, 0.229]|[0.0565, 0.0408, 0.1797]|
|flat_yaw_right|11/11|[0.0155, 0.0402, -0.3704]|[0.046, 0.0511, 0.113]|

## 各级严格越障

|任务|Level|高度cm|成功/条件|跌倒/条件|
|---|---:|---:|---:|---:|
|single_step_up|1|3|7/11|0/11|
|single_step_up|2|4|2/11|1/11|
|single_step_up|3|5|0/11|0/11|
|single_step_up|4|7|0/11|1/11|
|single_step_up|5|9|0/11|4/11|
|single_step_up|6|12|0/11|2/11|
|single_step_up|7|15|0/11|1/11|
|single_step_up|8|18|0/11|2/11|
|single_step_up|9|20|0/11|1/11|
|single_step_up|10|22|0/11|2/11|
|single_step_down|1|3|11/11|0/11|
|single_step_down|2|4|11/11|0/11|
|single_step_down|3|5|11/11|0/11|
|single_step_down|4|7|11/11|0/11|
|single_step_down|5|9|11/11|0/11|
|single_step_down|6|12|10/11|1/11|
|single_step_down|7|15|10/11|1/11|
|single_step_down|8|18|1/11|10/11|
|single_step_down|9|20|1/11|10/11|
|single_step_down|10|22|5/11|6/11|
|multi_stairs_up|1|3|0/11|0/11|
|multi_stairs_up|2|3|0/11|0/11|
|multi_stairs_up|3|3|0/11|0/11|
|multi_stairs_up|4|3|0/11|1/11|
|multi_stairs_up|5|5|0/11|0/11|
|multi_stairs_up|6|6|0/11|0/11|
|multi_stairs_up|7|8|0/11|4/11|
|multi_stairs_up|8|10|0/11|1/11|
|multi_stairs_up|9|12|0/11|0/11|
|multi_stairs_up|10|15|0/11|2/11|
|multi_stairs_down|1|3|11/11|0/11|
|multi_stairs_down|2|3|11/11|0/11|
|multi_stairs_down|3|3|11/11|0/11|
|multi_stairs_down|4|3|11/11|0/11|
|multi_stairs_down|5|5|11/11|0/11|
|multi_stairs_down|6|6|9/11|2/11|
|multi_stairs_down|7|8|7/11|4/11|
|multi_stairs_down|8|10|1/11|10/11|
|multi_stairs_down|9|12|0/11|11/11|
|multi_stairs_down|10|15|2/11|9/11|

## 逐项物理reward

加权rate，未乘dt=.02；按各域首次episode存活转移池化。短时跌倒可能有较高rate，须结合上表时长和成功率判断。
没有改动奖励权重；公式和功能见上述reward报告。完整每episode贡献及各域数据见JSON。

|reward|之前flat|之后flat|之前单级上|之后单级上|之后单级下|之后连续上|之后连续下|
|---|---:|---:|---:|---:|---:|---:|---:|
|track_vx|2.38943|2.41774|0.88413|0.624358|0.967815|0.646004|0.878727|
|track_vy|1.9576|2.32956|2.23193|2.60969|2.59454|2.69067|2.70093|
|track_yaw|1.66234|1.98584|1.63869|2.01335|2.02816|2.03125|1.98933|
|stand_still|0.0115079|0.0330145|0|0|0|0|0|
|terrain_alignment|0.933354|0.954806|0|0|0|0|0|
|base_lin_vel_z|-0.0122313|-0.00126095|-0.00545877|-0.000767248|-0.00376099|-0.000670151|-0.00308594|
|base_ang_vel_xy|-0.289067|-0.0195406|-0.910483|-0.0904408|-0.247782|-0.0736401|-0.177659|
|base_height_low|-0.0122338|0|-0.00905912|-0.0013902|-0.00155747|-0.0010927|-0.00215394|
|leg_joint_deviation|-0.0909206|-0.0135903|-0.00624628|-0.00223714|-0.003119|-0.00178101|-0.00279771|
|leg_joint_limit|-0.0162611|0|-0.0179613|-0.00337966|-0.00625361|-0.00147433|-0.00332949|
|leg_action_rate|-0.0126086|-0.00799009|-0.010854|-0.0071959|-0.0096|-0.00851137|-0.0115471|
|wheel_action_rate|-0.00107161|-0.00272427|-0.001074|-0.00259008|-0.00181749|-0.00292332|-0.0022811|
|leg_torque|-0.0369372|-0.0234592|-0.0433409|-0.0192345|-0.0226531|-0.0189107|-0.0216717|
|wheel_torque|-0.00170883|-0.000748584|-0.00263095|-0.000927211|-0.0010724|-0.000903037|-0.000867848|
|leg_joint_acceleration|-0.0505706|-0.0138331|-0.100642|-0.0421834|-0.0529845|-0.0487176|-0.0504204|
|wheel_joint_acceleration|-0.000618047|-0.000266986|-0.000959595|-0.000395729|-0.000447132|-0.000404848|-0.000407133|
|tracking_error_cost|-0.344089|-0.0698077|-0.736461|-0.19769|-0.2384|-0.167489|-0.172251|
|side_drift|-0.000720271|-0.00109619|0|0|0|0|0|
|wheel_drive_consistency|0.0053009|0.0126976|0|0|0|0|0|
|stair_progress|0|0|0.0311487|-0.125133|0.0387312|-0.0903833|-0.0454641|
|stair_success|0|0|0.00141358|0.0219585|0.407758|0.00042303|0.200245|

[完整数据](../validation/go2w_distance_20261118_results/final_11yaw_rewards.json) · [平地视频](../validation/go2w_distance_20261118_results/flat.mp4) · [台阶视频](../validation/go2w_distance_20261118_results/stairs.mp4)

播放最终模型：

```bash
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic-unified
PYTHONPATH=src /home/yifeiy/miniconda3/envs/go2_unilab/bin/python tools/play_go2w.py --checkpoint /home/yifeiy/ws_go2_unilab/runs/go2w_distance_resume_3p5h_20261118/evaluation/final.pt --demo
```

候选29333轮与最终模型同条件847条补充比较见[能力复评与reward诊断](go2w_distance_20261118_review.md)。候选平地87.0%，最终59.7%；最终下单级74.5%，候选58.2%，不存在全任务统一占优。
