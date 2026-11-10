# Go2 Blind Actor：2048 环境三小时训练评估

训练从 `longrun_B/warmstart/ckpt_002000.pt` 接续，2048环境、H8 WM、H32 Actor、盲Actor 213→12，先300轮Critic预热、500轮Actor学习率渐增；达到三小时wall预算后正常停止。最终第25307轮，累计129,701,888条真实训练转移（其中先前WM预热10,371,072条），累计WM更新101,228次、Actor更新23,007次、Critic更新466,140次。此次进程wall 10,802.5秒；训练进度数字含上轮预热。未触发策略崩溃保护。

最终固定评估：每场景3个seed（2426、2427、2428），命令前进0.35–0.45m/s，每场景最长20秒；穿越要求入口方向6.8m且全程横偏不超过0.6m，跌倒由硬终止判定。平地15次无跌倒、全部穿越，XY RMSE 0.0396m/s。每格三个样本，不能解释为可靠泛化成功率。

|地形|Level1 通过/跌倒/XY误差|Level2|Level3|Level4|
|---|---|---|---|---|
|粗糙|3/0/0.042|3/0/0.051|3/0/0.059|1/0/0.068|
|上坡|3/0/0.041|3/0/0.042|3/0/0.044|3/0/0.045|
|下坡|3/0/0.042|3/0/0.040|3/0/0.041|3/0/0.046|
|上楼|3/0/0.045|3/0/0.051|3/0/0.052|3/0/0.057|
|下楼|3/0/0.043|3/0/0.051|3/0/0.055|3/0/0.054|

误差单位m/s；Level4台阶高度8cm、宽40cm。粗糙Level4另两次未满足严格走廊条件，不能称跌倒。统计与原始逐episode数据在 `results/actor_training_B/final_policy_evaluation/fixed_evaluation.json`。

另外对真实plane flat做每个10秒单seed测试：后退-0.3m/s的XY RMSE为0.283m/s、世界x仅-0.205m；侧移0.25m/s的XY RMSE为0.248m/s、世界y仅0.036m；原地转向0.5rad/s的yaw RMSE为0.397rad/s。这些测试均未跌倒，但跟踪差，符合本轮以正向命令训练的范围；不证明三维速度跟踪能力。其他地形/多seed的后退侧移转向尚未测试。

Reward保持对角步态-0.15、滑足-0.20，足部抬高0；使用地形相对平滑接触代理。Level4各地形足部滑移代理均值约0.207–0.246m/s，不能称零滑足，且该指标不是实测接触力。final WM独立同一批留出物理轨迹预测通过，动作梯度平均方向一致率89.31%、最弱78.47%，仍未达到所有探针≥80%的旧门槛。不要据此认定WM梯度完全可靠。

播放/测试入口：`tools/play_go2_terrain.py`。实测 stairs Level4 `vx=0.4`、20秒headless无跌倒，x推进8.20m、XY RMSE0.0494m/s；MuJoCo图形viewer实测可启动与正常退出。

```bash
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic
OMP_NUM_THREADS=4 /home/yifeiy/miniconda3/envs/go2_unilab/bin/python tools/play_go2_terrain.py --terrain stairs --level 4 --vx 0.4 --seconds 20
```

支持 `flat`、`rough`、`slope`、`slope_down`、`stairs`、`stairs_down`，等级0–4；命令`--vx --vy --wz`，窗口需可用桌面DISPLAY，`--headless`可仅跑物理并打印指标。checkpoint默认为 `results/actor_training_B/last.pt`，也可用 `--checkpoint`指定。平地以原plane物理模型执行，非flat使用公开UniLab生成地形。文件不覆盖旧checkpoint。
