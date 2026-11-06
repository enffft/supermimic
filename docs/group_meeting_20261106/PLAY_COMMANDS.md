# 组会录屏用 Play 命令

按用户后续要求，本次不录制视频，也不创建空视频占位。下面全部为已有真实checkpoint推理，不训练、不覆盖权重、不运行实机控制。只在有图形桌面的终端逐条运行，关闭一个viewer后再启动下一个。

## 最方便的六条主命令

```bash
cd /home/yifeiy/ws_supermimic
# Go2 从零各2.5h，前进0.3m/s；显示命令箭头与实际速度箭头
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_ppo_forward
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_sm_forward

# M1 v2 各3.5h，纯L，固定vx=0.12m/s
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_L
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_L

# M1 v2 各3.5h，纯M，相同固定环境局部世界点[1.5,0,0.8]m
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_M
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_M
```

脚本封装的就是现有`scripts/play.py`，只设置checkpoint和现成参数；不新写仿真或控制逻辑。加`--print`只打印完整命令，完全不启动viewer：

```bash
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_M --print
```

## 可以直接复制的原始命令

Go2 PPO：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt /home/yifeiy/rl_sar/policy/go2/ppo_velocity/policy_inference.pt \
  --num-envs 1 --viewer native --vx 0.3 --vy 0 --yaw-rate 0 \
  --velocity-arrow --actual-velocity-arrow
```

Go2 SuperMimic，从零2.5h：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt /home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt \
  --num-envs 1 --viewer native --vx 0.3 --vy 0 --yaw-rate 0 \
  --velocity-arrow --actual-velocity-arrow
```

M1-Piper PPO，v2 3.5h，相同固定目标：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/policy_inference.pt \
  --num-envs 1 --viewer native --m1-task M --m1-target-world 1.5 0 0.8
```

M1-Piper SuperMimic，v2 3.5h/H32，相同固定目标：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src OMP_NUM_THREADS=1 /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt logs/m1_reward/v2_lm50_2048_3p5h_seed2026/ckpt_latest.pt \
  --num-envs 1 --viewer native --m1-task M --m1-target-world 1.5 0 0.8
```

M1两条原命令若看L任务，将最后两项替换为`--m1-task L --vx 0.12 --vy 0 --yaw-rate 0`。M任务红球=目标、青球=TCP、黄线=误差，终端定期显示距离和是否到达。固定目标演示关闭成功后换点；它与正式随机连续goal成功率不是同一评测协议。

## 补充与失败案例

```bash
# 原地yaw=0.4rad/s：PPO正式评测失败，SM能跟随；不要只录前进
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_ppo_yaw
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_sm_yaw

# 纯侧移vy=0.3m/s
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_ppo_side
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_sm_side

# 旧SM漂亮对角步态：Actor链4.294h / 完整链4.546h，单独标注
bash docs/group_meeting_20261106/scripts/play_demo.sh go2_old_trot

# M远目标[2.5,0,0.8]；这是示例坐标，不预先承诺必成功
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_far
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_far

# M低目标[1.5,0,0.15]；这是困难示例，不能代替全部低目标统计
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_low
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_low

# M训练分布随机连续换点，双方成功后目标序列会分叉
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_ppo_random
bash docs/group_meeting_20261106/scripts/play_demo.sh m1_sm_random
```

## 录制口径与界面限制

Go2现有play固定seed=0；M1使用`evaluation_args`默认seed123。各机器人内两算法配对名义初态、关闭DR和噪声；不是说Go2 play录像和seed2026正式评测逐帧相同。现有play没有`--seed`参数，本任务不擅自新增参数。M随机目标即使同seed，成功换点/episode结束不同后也不保证每个后续目标配对；严谨对照优先使用同一个固定世界点。

建议录1920×1080/30fps，左右片段按相同仿真时间和倍速剪辑，统一机位和尺度，角标写“算法 / 训练预算 / 命令或目标”。Go2旧4.55h单独出镜，不混入等时长主比较。M标注“TCP到点，无抓取”；固定目标仅演示行为，不作为随机goal成功率证据。

原生viewer优先用窗口关闭/Esc退出。此前Ctrl+C后的GLXBadContext属于viewer销毁错误，不能据此解释策略步态。若无桌面，可将原始命令`--viewer native`改成`--viewer viser`，用浏览器访问输出地址；本次未启动viewer或验证远程浏览器连通。现有可视化能显示目标/TCP/速度箭头，但没有统一算法和训练时长字幕，应由录屏剪辑添加。

命令已经通过脚本语法、case参数生成、checkpoint存在性、play的help接口检查；本次按用户要求不运行viewer，不将“命令检查”写成“新视频已验证”。
