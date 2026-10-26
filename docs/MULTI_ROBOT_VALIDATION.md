# 多机器人训练前验收记录

本轮针对 Go2 velocity-command 数据采集与 M1-Piper motion-tracking 训练正确性做了短程检查，没有启动正式训练，也没有重新转换或生成 Motion Bank。环境使用 Conda `supermimic`。

## 验收结论

| 状态 | 项目 | 结论 |
|---|---|---|
| 已修复 | M1 真实/想象 proprioception | 四个 wheel angle 在两条路径都 mask；wheel velocity、wheel action、encoder bias 保留。720D history 对齐误差为 `2.98e-8`。 |
| 已修复 | qpos/qvel、Actor 和 NPZ 关节索引 | 编译模型按 qpos address 检查顺序；StateIO 将 MuJoCo 的 `robot/name` 映射到裸 joint name 并按 RobotSpec 顺序读写；Action 和 MotionLibrary 也按名称对齐。 |
| 已修复 | M1 第一阶段数据误选 | Trainer 默认只允许 manifest 中 PhysX 回放通过的 6 条 clip，拒绝其它 clip 和绕过 manifest 的 packed library。原 14 条 NPZ 保留。 |
| 已验证 | M1 Motion Library mask 使用边界 | Goal 只包含 18 个 AMP 非轮关节；MotionLibrary 检查这些 Goal 关节 `joint_valid` 全有效。它不使用 `wheel_reference_valid` 做训练时 wheel reward mask，因此不宣称轮子位置跟踪已经 mask。 |
| 已验证 | M1 real/analytic reward | 8 个 Reward Term 的逐项误差已测；最大绝对误差 `2.003e-4`，最大平均相对误差 `0.518%`。剩余误差主要来自 mjlab 在 reward 计算时使用上一个 physics substep 的派生量。 |
| 已验证 | Go2 零动作站立与终止规则 | 16 env × 400 控制步（8 秒）；四个足端碰撞体在所有环境都接地，没有膝/小腿碰地、没有 reset、没有扭矩饱和。基准根高稳定在 `0.28271 m`。真实 termination 与 World Model 特征阈值一致。 |
| 仍未验证 | 真实失败姿态下 Go2 `0.17 m` 根高阈值的宽严程度 | 已验证站立不误终止、合成低根高/倾倒姿态会触发；没有扫描真实下蹲、趴伏和跌倒边界，暂不改高度或力矩配置。 |
| 仍未验证 | Termination Head 正类学习 | 短诊断窗口中 Go2 与 M1 都没有真实 positive termination 样本。少量更新后概率仍接近/高于 0.5，不能声称 termination head 已学好；特征到环境标签的阈值映射测试通过。 |
| 留待下一轮 | Go2 正式 Reward、M1 多轨迹 Tracking Reward 设计 | 不改奖励权重、不启用 Go2 policy update。 |

## Git 与环境

- 本轮基于已审查提交 `7a2b626a26b4033c8ab65576d2430487dbbb2606` 修改。
- Python：`/home/yifeiy/miniconda3/envs/supermimic/bin/python`；没有使用 uv，也没有重装依赖。
- Go2/M1 短诊断在本机 RTX 4070 SUPER / MuJoCo-Warp CUDA 上执行。驱动 12.2 低于 CUDA Toolkit 12.9，因此 Warp 提示关闭 CUDA Graphs；不影响下述测试通过，但数值来自当前本机仿真构建。

## 修改的源码和持久化验收

| 路径 | 改动/验收内容 |
|---|---|
| `src/supermimic/world_model/heads.py::analytic_observation` | 增加显式 `wheel_indices`；只 mask joint-position error，状态、速度和 action 维度不裁剪。增加 M1 reward reanchor 时使用控制步起点 anchor 的选项。 |
| `src/supermimic/imagination/compiled.py::ImagStep`、`src/supermimic/imagination/rollout.py::imagine` | 从 StateSpec 传递 wheel index；M1 reward 使用动作执行前的预测状态 anchor，与真实 MotionCommand 的 reanchor 时间点相同。G1 默认路径不变。 |
| `src/supermimic/envs/state_io.py::StateIO` | 通过名字将 MuJoCo 编译名 `robot/FAR_ABAD_JOINT` 转为 articulation/RobotSpec 名；按 name-to-qpos/dof address 读写。验证 floating base 位于地址 0。 |
| `src/supermimic/robots/mjlab_assets.py::load_robot_spec_model`、`src/supermimic/training/loop.py::Trainer.__init__` | 编译模型按 qpos 地址检查完整 joint order，运行环境的 mjlab articulation order 必须等于 RobotSpec order。 |
| `src/supermimic/envs/mdp/motion_library.py::MotionLibrary.__init__` | NPZ 的 `joint_names` 映射到 RobotSpec 顺序；若 Goal joint 被 `joint_valid` 标为无效则拒绝加载。wheel reference mask 不被假装成已应用。 |
| `src/supermimic/envs/multirobot_env_cfg.py` | Go2 actuator name order 显式保序；向量化 MuJoCo contact capacity 最小值提高到可容纳 M1 8-env 模型。 |
| `src/supermimic/training/loop.py`、`configs/m1_piper_first6.csv` | M1 默认锁定六条通过回放的 clips；拒绝未经批准名称和 packed library 绕过路径。 |
| `src/supermimic/scripts/diagnose_multirobot.py` | 短采集、8-step WM 更新前后统计、termination 概率/BCE、H=8/H=32 survival、Actor 梯度和仿真回报 JSON 诊断。Actor optimizer 不执行。 |
| `tests/test_multirobot_acceptance.py` | 持久化 joint order、M1 source/sidecar integrity、观测/action 顺序、reward、Go2 站立/termination、8-step WM backward 和 32-step BPTT 检查。 |

## 验收命令和结果

以下命令在仓库根目录运行：

~~~bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m compileall -q src/supermimic tests/test_multirobot_acceptance.py
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m pytest -q tests/test_multirobot_acceptance.py
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m pytest -q tests/test_analytic_heads.py tests/test_world_model.py
~~~

结果：多机器人验收 **6 passed**；原 G1 analytic-head / World Model 回归 **6 passed**。未运行 M1 converter 命令。

### MuJoCo、Actor、FK、Motion Reference 的顺序

- G1、Go2、M1 的编译模型 scalar joint 按 `jnt_qposadr` 排序后都与 RobotSpec 顺序一致；测试还检查 qpos 地址为 `7..7+n-1`、qvel 地址为 `6..6+n-1`。
- M1 22 关节策略顺序为四条腿各 ABAD/HIP/KNEE/FOOT，再接 Piper 6 关节；wheel index 是 `(3, 7, 11, 15)`。
- M1 动作测试同时给 wheel index 3 与 arm index 16 非零 action，只有相应 velocity/position target 被改变，倍率分别为 5.0 与 0.25。
- 本地 mjlab model 给 M1 joint 加 `robot/` 命名空间。旧 StateIO 直接集合比较会把 22 个关节全部判为缺失；本轮改为显式 namespace/name → address 映射。M1 16-env ReplayRecorder 初始化和读写现在通过。

### M1 第一阶段六条 Motion Bank clip

清单 `configs/m1_piper_first6.csv` 与本机 `physx_replay_results.json` 的 `physx_reference_tracking_pass=true` 项一致：

~~~text
a_wheel_000000
a_wheel_000002
b_arm_000006
c_assisted_000029
c_assisted_000039
c_assisted_000149
~~~

剩余 8 条 keep=0，原始 14 个 NPZ 没有删除。对这六条逐条核对了 JSON `Frames` 为 finite `(T,79)`、source/sidecar SHA256、AMP joint order、NPZ 22-joint order、body order、50 Hz 帧数与时间戳、root position 插值、xyzw→wxyz 的 Slerp，以及 body-frame AMP root linear velocity 旋转到 world frame。六条选中源文件都能找到 sidecar；wheel angles/rates 与 validity mask 相符。Motion Library 的 tracking Goal 仍只选 18 个非轮 AMP joints，四个 wheel DOF 仍保留在完整动力学状态中。

### M1 真实与想象 observation

在 16 个真实 M1 MuJoCo 环境、关闭 observation noise 的条件下，对默认状态、非零 wheel angle、超过 `2π` 的 wheel angle、非零 wheel velocity、Piper joint 非零状态逐帧对比。DR encoder bias 保持开启，analytic 路径读取同一个 ReplayRecorder `xi`；每步 action 也使用 env 当前 clean action。

| 对比项 | 最大逐元素绝对误差 |
|---|---:|
| 单帧 72D proprioception | `2.9802e-8` |
| 10 帧 term-major Flatten 后 720D policy observation | `2.9802e-8` |

wheel joint position error 被 mask 后仍占原 4 个维度；wheel velocity 和上一动作都保留。G1/Go2 未提供 wheel index，沿用原 observation 行为。

### M1 Reward Term parity

16 env、12 个小动作步，对比 RewardManager 实际 term 与 analytic FK reward。均值是已加权、乘过 `dt` 的每步值；absolute/relative error 汇总这些真实样本。

| Term | Real mean | Analytic mean | Mean abs error | Max abs error | Mean relative error |
|---|---:|---:|---:|---:|---:|
| `motion_global_root_pos` | 0.00995883 | 0.00995748 | 1.343e-6 | 3.782e-6 | 0.0135% |
| `motion_global_root_ori` | 0.00999577 | 0.00999554 | 2.300e-7 | 5.988e-7 | 0.0023% |
| `motion_body_pos` | 0.01962059 | 0.01959072 | 2.987e-5 | 7.473e-5 | 0.1522% |
| `motion_body_ori` | 0.01942233 | 0.01937821 | 4.412e-5 | 1.095e-4 | 0.2272% |
| `motion_body_lin_vel` | 0.01736636 | 0.01728011 | 8.989e-5 | 2.003e-4 | 0.5176% |
| `motion_body_ang_vel` | 0.01915711 | 0.01912968 | 3.033e-5 | 7.514e-5 | 0.1583% |
| `action_rate_l2` | -1.2003e-5 | -1.2003e-5 | 7.769e-13 | 1.819e-12 | 6.47e-6% |
| `joint_limit` | 0 | 0 | 0 | 0 | 0 |

八个 term 的 body order、reference、权重和 dt 尺度一致。mjlab 在 manager 计算 reward 时复用前一 physics substep 的派生 body transforms/velocities；analytic FK 从读到的当前 qpos/qvel 重算，因此仍有不超过 `2.01e-4` 的 velocity term 差异。M1 imagined body reward 的 reanchor 使用该控制步起点 state，和 MotionCommand 的 target cache 时间点一致；G1 默认 reward anchor 实现保持不变。

### Go2 零动作站立、接触与 Termination

测试命令包含在 `test_go2_zero_action_stand_and_termination_features` 中：16 env × 400 控制步，`dt=0.02 s`，总计 8 秒。零 action 配置为无 action noise、无 observation noise、无 domain randomization。

| 指标 | 实测 |
|---|---:|
| 配置初始 root z → 稳态 root z | `0.42 m` → `0.28271 m` |
| 最后 100 步 root z 标准差 | `5.99e-6 m` |
| 最终倾斜 feature `abs(1+g_local_z)` 最大值 | `2.967e-4` |
| 最后 100 步 root angular velocity 最大绝对值 | `7.44e-5 rad/s` |
| 全程 actuator force/torque 最大值 | `22.412 / 23.5`，无饱和样本 |
| 最小 soft joint-limit margin | `0.641 rad` |
| 地面接触 | 16/16 env 的 FL/FR/RL/RR 四足都接触 |
| 地面接触中的 thigh/calf/hip geom | 0 |
| termination/reset | 0 |

Go2 稳态根高显著低于配置初始高度，但四足有接触、无膝/小腿碰地、关节仍远离软限位、姿态与速度稳定，不支持把它判为趴伏或膝部异常碰撞。真实低根高条件为 `root_z < 0.17 m`；它比观察到的稳态再低约 0.113 m。真实 tilt 条件 `abs(1+g_local_z)>0.8` 与 World Model 的第二个 feature 阈值完全同形。测试还写入 `z=0.16 m` 和 90° 倾倒状态，真实 termination 与 analytic feature 布尔结果一致。

这验证的是阈值实现和健康站立基准，不足以定标实际下蹲/跌倒边界；所以没有修改 `default_base_height`、Kp/Kd 或 23.5 N·m effort limit。

### WM 8-step、Termination、Survival、Actor BPTT

复跑命令与 JSON 日志：

~~~bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m supermimic.scripts.diagnose_multirobot --robot m1_piper --num-envs 16 --steps 160 --wm-updates 4 --window-batch 16 --diagnostic-windows 32 --imag-batch 16 --action-noise 0.03 --seed 41 --output logs/multirobot_diagnostics/m1_acceptance.json
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m supermimic.scripts.diagnose_multirobot --robot go2 --num-envs 16 --steps 160 --wm-updates 4 --window-batch 16 --diagnostic-windows 32 --imag-batch 16 --action-noise 0 --no-domain-randomization --seed 41 --output logs/multirobot_diagnostics/go2_acceptance.json
~~~

诊断使用 128-wide 小型 WM、每次一个 16-window batch、共 4 次 WM 更新。每次 policy 只反传测 gradient，Actor optimizer step 数为 0。它不代表正式网络或训练收敛。

| 指标 | M1-Piper | Go2 |
|---|---:|---:|
| 收集向量化控制步 / Replay 行 | 160 / 2565 | 160 / 2560 |
| 可采 8-step windows | 2397 | 2432 |
| WM 8-step NSE，warmup 前 → 后 | 1-step `0.00734→0.00424`；8-step `0.08276→0.05470` | 1-step `0.06695→0.00605`；8-step `0.95845→0.60988` |
| Term BCE，warmup 前 → 后 | `0.71523→0.71148` | `0.79953→0.79526` |
| termination label / probability | 256 nonterminal / 0 terminal；平均概率约 `0.51086→0.50909` | 256 nonterminal / 0 terminal；平均概率约 `0.55038→0.54859` |
| warmup 后 `p>=0.5` 准确率 | 0%（全预测 terminal） | 0%（全预测 terminal） |
| H=8 survival / Actor grad norm | `3.373e-3 / 2.027e-5` | `1.720e-3 / 2.189e-9` |
| H=32 survival / Actor grad norm | `1.295e-10 / 5.068e-6` | `8.389e-12 / 5.520e-10` |
| Real simulation return | 5 个 episode；raw `13.631`，normalized `0.9138` | 无正式 Go2 reward；collection placeholder 为零权重，不能报告为任务回报 |

诊断说明了两个不同点：StateSpec/环境标签测试没有发现阈值或 tilt feature 语义错误；当前短 warmup 下 TerminationHead 明显还没学会，且这批样本没有 terminal positive class，无法验证正类识别。模型的 open-loop NSE 经少量更新有改善，但 soft survival 在 H=32 几乎归零，Actor 梯度仍很小。不能据此关闭 survival、删除 TerminationHead 或改变论文 BPTT 公式。后续若讨论调度，应单独提出可配置 WM warmup，并增加可控失败样本来覆盖正类。

## G1 回归与未完成设计

- `tests/test_analytic_heads.py tests/test_world_model.py`：**6 passed**；G1 reward、observation、termination/anchor observation 和 WM replay 测试通过。G1 的默认奖励 reanchor 与训练流程未改。
- Go2 的正式 blind-locomotion Reward 仍待团队选择；velocity debug objective 只用于 imagination/BPTT 检查，policy update 未执行。
- M1 当前仍沿用既有 tracking term 和权重。本轮只对齐接口与数值语义，没有定稿 wheel/leg/arm/TCP 多轨迹奖励权重。
- 所有 GPU smoke/诊断均为短时验收；没有长程训练、课程学习、复杂地形、LiDAR 或抓取实验。
