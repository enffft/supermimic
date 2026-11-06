# 组会简版：只比较任务表现

Go2：PPO、SuperMimic从零各训练约2.5小时；M1：v2 reward-only各训练约3.5小时。都是已有正式仿真评测，本次未训练或重跑评测。

## Go2：目标速度与实际速度

| 指令 | PPO实际 | SuperMimic实际 |
|---|---:|---:|
| 前进0.30 m/s | 0.343 | 0.282 |
| 前进0.50 m/s | 0.545 | 0.458 |
| 后退−0.15 m/s | -0.010 | -0.168 |
| 侧移0.10 m/s | 0.147 | 0.100 |
| 侧移0.30 m/s | 0.350 | 0.310 |
| 转向0.20 rad/s | 0.091 | 0.182 |
| 转向0.40 rad/s | 0.118 | 0.392 |

结论：两者都能前进；PPO略超速，SuperMimic对后退、侧移和转向跟随更完整。PPO纯侧移0.10/0.30及转向0.40的评测跌倒率100%，转向0.20为40.6%；所以这些平均速度不能单独解释成跟随成功。SuperMimic上述指令均无跌倒。

每指令64个已结束episode；单训练seed、名义平地。这里只比较本次从零2.5h，旧4.55h对角步态模型不混入。

## M1：两类任务

| 任务 / 指标 | PPO | SuperMimic |
|---|---:|---:|
| L速度任务：目标vx=0.120 m/s，实际vx | 0.165 m/s | −0.001 m/s |
| L速度任务：XY速度RMSE，越小越准 | 0.050 m/s | 0.137 m/s |
| M操作到点任务：目标成功率 | **71.2%（334/469）** | **0%（0/128）** |
| M操作到点任务：平均TCP距离，越小越好 | 0.583 m | 2.022 m |

L现有评测未定义成功阈值，因此用跟随误差表达准确性，不人为换算成百分比。M是TCP进入8cm范围的目标成功率，无抓取；分母为已结束goal，PPO另有27个活跃未结束goal未计入。PPO M任务也有25.2%的episode跌倒率，仍有改进空间。

结论：这轮M1 PPO学到了速度跟随和部分操作到点，SuperMimic基本停滞。两者等墙钟但PPO真实样本为SM的5.736倍；这是当前系统结果，不是所有变量完全一致的算法消融。

## 展示素材和播放

只需两张现成图：[Go2速度表](figures/G02_go2_velocity_results.png)、[M1两任务结果](figures/M02_m1_final_results.png)。其余长报告和图片作为备查，不需要放进组会。

[全部play命令](PLAY_COMMANDS.md)。在仓库根目录逐条运行，每次关闭一个窗口再开下一个：

```bash
cd /home/yifeiy/ws_supermimic
# Go2
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_ppo_forward
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_sm_forward
# M1：L速度任务
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_L
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_L
# M1：M操作任务，同一个目标[1.5,0,0.8]m
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_M
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_M
```

原始来源：data/go2_results.json、data/m1_{ppo,supermimic}_{forward,manipulation}.json；更详细的条件见[证据索引](EXPERIMENT_PROVENANCE.md)。视频由你自行录制。
