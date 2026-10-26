# SuperMimic 多机器人接口验证记录

验证基于当前工作区和本机 mjlab/MuJoCo 环境；没有启动正式训练。Python 环境是 Conda supermimic。

~~~bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m compileall -q src/supermimic
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m pytest -q tests/test_world_model.py tests/test_imagination.py tests/test_env_smoke.py
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m supermimic.data.m1_motion_convert
~~~

没有使用 uv，也没有重装 Python、CUDA、Warp、mjlab 或 MuJoCo。

## Implemented

- RobotSpec、非 G1 模型/执行器、Go2 velocity-command 配置、M1 mixed action 与 Motion Bank converter 已接入同一 Trainer/Replay/World Model/PolicyTrainer。
- G1 motion tracking 仍由原环境配置与 Motion Library 驱动；训练主循环顺序保持 collect → WM → Actor/Critic。
- Go2 不加载 Motion Library；默认禁止 Go2 policy optimizer 更新，除非显式启用仅用于 smoke 的 diagnostic objective。
- M1 转换输出放在 data/npz/m1_piper/；源 Motion Bank/URDF 不修改。

## Tested

### 编译与 G1 回归

- 全量 compileall：通过。
- pytest -q tests/test_world_model.py tests/test_imagination.py tests/test_env_smoke.py：**10 passed**，102.09 s。包括 G1 仿真/训练 smoke、WM replay overfit 和 imagination 测试。
- warning 包括 Warp driver 12.2 低于 Toolkit 12.4 导致 CUDA graphs 关闭，以及 TorchScript/deprecation、数值转换提示；没有测试失败。

### 编译模型和维度

| Robot | Compiled nq/nv | 控制关节数 | State/delta | xi |
|---|---:|---:|---:|---:|
| G1 | 36/35 | 29 | 71/70 | 33 |
| Go2 | 19/18 | 12 | 37/36 | 16 |
| M1-Piper | 29/28 | 22 | 57/56 | 26 |

三模型关节名均与各 RobotSpec 集合一致，StateSpec/Action 维数吻合。World Model 张量接口：Go2 [B,37]+[B,12]+[B,16]→[B,37]；M1 [B,57]+[B,22]+[B,26]→[B,57]。

### M1 Motion Bank

- 将本机 Motion Bank 的 14 条 clip 转成 50 Hz NPZ。
- 旧 clip 的 wheel reference-valid fraction 为 0；有效 sidecar clip 为 1。缺少 wheel 参考的 clip 未被当成有监督 wheel motion。
- 示例 a_wheel_000000.npz：body position (145,14,3)、joint validity (145,22)、dt=0.02 s，含 TCP body piper_x_link7。

### 16 环境 physics/action smoke

| 测试 | 实测 |
|---|---|
| Go2 零动作 | 16 env × 300 控制步（每步 4×0.005 s），policy obs (16,420)、Goal (16,3)、State (16,37)、Replay 4800 行。root z 从 0.42 m 稳定到约 0.283 m，过程最低 env-mean 约 0.250 m；无 termination，状态有限。这只检查仿真与采集链路，不是速度跟踪测试。 |
| Go2 domain randomization | 16 env 开启 DR，reset 后一步通过；obs (16,420)、Goal (16,3)、encoder bias (16,12)。 |
| M1 motion env | 16 env × 50 控制步，单 clip；obs (16,720)、Goal (16,460)（此直接 env smoke 显式关闭 anchor）、State (16,57)、xi (16,26)、Replay 800 行；无 termination，状态有限。Trainer 的默认 M1 Goal 开 anchor 后是 469。 |
| M1 domain randomization | 16 env 开 DR，reset 后一步通过；encoder bias (16,22)。 |
| M1 wheel/arm action | 同一 clip 上 50 步对照：零 action 平均前向位移 0.001116 m；四轮正 action 0.1（target 0.5 rad/s）为 0.037886 m，wheel qvel 约 [0.455,0.454,0.509,0.511] rad/s。单个 Piper 关节从参考起点移动约 0.3673 rad。证明所测动作通道在执行，不说明站立或 tracking 稳定。 |

### Differentiable FK

当前模型在典型姿态下与 MuJoCo mj_forward 对照：

- Go2 tracked-body 最大位置差：9.824e-8 m。
- M1 tracked-body 最大位置差：7.734e-8 m。
- M1 piper_x_link7 转换参考与本地 FK 最大位置差：1.192e-7 m。

### 8-step WM 和 BPTT

Go2 和 M1 各用 16 env × 50 控制步得到 800 条在线 replay transition；WM update 使用 8 个窗口、每窗 8 步。Actor 分别检查 H=8 和 H=32。使用小网络/小 batch 的 smoke 参数，不是正式训练配置。

| Task | WM 一次更新 loss | Actor grad norm | 附加检查 |
|---|---|---|---|
| M1-Piper | total 3.6662；state 3.1273；term BCE 0.5388 | H=8 1.042e-6；H=32 1.217e-7，有限且非零 | anchor Goal 469；H survival 约 0.01349 / 3.32e-8 |
| Go2 | total 41.51；state 40.74；term BCE 0.774 | H=8 2.02e-8；H=32 2.92e-8，有限且非零 | policy stage 中 WM 权重保持不变；只启用诊断目标；H survival 很小 |

这些结果证明 loss、shape、冻结参数和反向图可运行。随机/未充分训练的动力学及 termination head 会让梯度或 survival 很快衰减；不能据此声称策略已学会运动。

## Not Yet Implemented

- **Go2 正式 blind-locomotion reward 尚未确定。** collection reward 为零权重；VelocityCommandTask.reward() 是 smoke-only 诊断目标，默认 PolicyTrainer.update() 阻止 Go2 policy 更新。需要先讨论线速度/偏航误差、能耗、平滑、站立行为和命令分布。
- **Go2 termination 阈值仍是临时规则。** 当前基于 default_base_height - root_height_threshold 与 upright tilt；需按真实跌倒/趴伏情况复核。
- **M1 reward 还未针对 wheel/leg/arm/TCP 做权重标定。** motion reward 和 FK 路径可复用，但应确认哪些 AMP joints/bodies 参与、wheel mask 的处理、TCP 权重及肢体平衡权重。
- 没有运行正式训练，也没有 curriculum、复杂地形、LiDAR、抓取或 AMP 结果。
- 多机器人 physics/WM/BPTT 验证由临时小型 smoke harness 执行，没有新增持久化测试文件。开头列出可重跑的现有 pytest、compileall 和 converter 命令。

## 下一阶段直接修改的代码入口

- Go2：src/supermimic/envs/multirobot_env_cfg.py velocity 分支的 rewards/terminations；src/supermimic/imagination/tasks.py::VelocityCommandTask.reward()；src/supermimic/imagination/compiled.py::VelocityImagStep。
- M1：src/supermimic/envs/multirobot_env_cfg.py motion reward terms；src/supermimic/world_model/heads.py::AnalyticHeads.reward()；src/supermimic/envs/mdp/rewards.py::tracking_reward_terms()；src/supermimic/envs/mdp/motion_library.py::goal_features()。

Reward/termination 定稿后先增加对应小验证，再跑短 smoke；Go2 policy update 在此之前保持关闭。
