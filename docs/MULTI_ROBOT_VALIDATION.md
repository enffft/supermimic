# 多机器人训练前验收记录

本轮按训练前工程验收范围检查 Go2 Blind Locomotion 与 M1-Piper Motion Tracking；之后按用户要求补跑了一次 15 分钟 Go2 短训练。没有启动数小时长程训练、没有重新生成/转换 Motion Bank，也没有改 Reward 权重。回归验收使用最多 16 个环境；Go2 短训练使用 3072 个环境。

## 验收结论

| 状态 | 项目 | 结论 |
|---|---|---|
| 已修复并验证 | M1 真实/想象 observation | wheel angle position error 在两条路径都 mask；velocity、上一动作、完整动力学状态保留。单帧与 720D history 最大误差均 `2.98e-8`。 |
| 已修复并验证 | MuJoCo/MJLab/Actor/NPZ joint order | 编译 qpos/qvel address、运行时 articulation 和 RobotSpec 顺序通过名称逐项对齐；M1 wheel index `(3,7,11,15)`；wheel 与 Piper 两个独立 action 只改对应 target。 |
| 已修复并验证 | M1 第一阶段 Motion Bank | manifest 仅列 PhysX reference-tracking 通过的 6 条；原 14 条 NPZ 均保留。转换 hash、79D AMP、sidecar mask、50Hz 重采样、Root frame/quaternion 和 joint order 已核验。 |
| 已验证 | M1 MotionLibrary validity 边界 | tracking Goal 只含 18 个有效 AMP 非轮关节。`joint_valid` 会按 Goal 检查；`wheel_reference_valid` 没有用于实时 wheel reward mask，因此不宣称轮角跟踪已按有效性 mask。 |
| 已验证 | M1 real/analytic reward parity | 8 个启用的 Reward Term 逐项比对；最大绝对误差 `1.996e-4`，最大平均相对误差 `0.514%`。 |
| 已验证 | Go2 零动作站立及阈值实现 | 16 env × 400 control steps；站稳高度 `0.28271 m`，四足接触，无 hip/thigh/calf 地面接触、无 reset、无力矩饱和。低高度与倾斜 termination 的真实布尔规则和解析 feature 语义一致。 |
| 仍未验证 | Go2 真实跌倒阈值边界 | `0.17 m` 低高度和 70° 倾斜的阈值实现已通过合成状态检查，但尚未扫描真实下蹲/趴伏/跌倒的稳定边界；不改初始高度、Kp/Kd 或 effort limit。 |
| 已修复并在短训练中产生自然正类 | Replay done 标签与 Go2 Termination Head | `sample_windows()` 将 transition row 的 `done_next` 对齐到该动作预测的 successor state；Go2 短训练中真实跌倒自然产生 terminal labels。Head 的概率与样本正类比例接近，但正类稀少，召回率和校准仍未充分验证。 |
| 已启动并完成短训练 | Go2 当前 velocity reward | 使用现有 11 项 velocity reward core 完成一次 15 分钟有界训练；训练结果和失败样本统计见下文。没有在本轮修改 reward 权重。 |
| 已完成短训练；策略验收未通过 | M1-Piper Motion Tracking | 后续 15 分钟运行在 2048 env 下完成并保存 checkpoint；六条 clip 均播完，但 success rate 为 0，不能称为已收敛。 |
| 已修复并可见性验收 | M1 Play STL visual frame | 修正 quadruped STL 的重复 -90° visual roll；Play 使用匹配的 Piper-X Link STL。真实关节/碰撞/action 未改。 |
| Play 验收未通过，待修 | Go2 视觉与前进表现 | 用户观察到 Go2 关节错位、机体悬浮且不前进。Go2 使用 DAE visuals，本轮 M1 STL 修复不覆盖 Go2；仍需单独核对 Go2 URDF、mesh frame 和策略命令响应。 |
| 留待下一轮 | M1 多轨迹 Tracking Reward | M1 继续使用现有 8 项 tracking terms，本轮不重新设计权重。 |

## 环境、代码基准与日志

- 源码与测试提交：`1b4b280f3317bdcc0c16d370cad01f102f15572f`（基于 `060a0442ca7a9775fea44713cad5d141956e100d`）。测试与短诊断是在提交前对同一组源码改动运行；诊断 JSON 因此记为旧 HEAD 且 `git_worktree_dirty=true`。
- Python 环境：`/home/yifeiy/miniconda3/envs/supermimic/bin/python`。没有重装依赖。
- GPU：RTX 4070 SUPER。运行时 Warp 报告驱动 12.2、CUDA Toolkit 12.9 不满足 CUDA Graphs 条件，因此关闭 CUDA Graphs；下列结果来自当前 MuJoCo-Warp 构建。
- 可复看本机日志：`logs/multirobot_diagnostics/acceptance.log`、`g1_regression.log`、`go2_stand_pose.log`、`m1_16env_256step.json`、`go2_16env_256step.json` 及其同名 `.log`。这些诊断产物保存在本机 logs 目录，不属于仓库交付文件。

## 修改的源码与持久化测试

| 源码/测试 | 本轮修正或验收内容 |
|---|---|
| `src/supermimic/world_model/heads.py::analytic_observation`、`src/supermimic/imagination/compiled.py::ImagStep`、`src/supermimic/imagination/rollout.py::imagine` | 明确传递 wheel indices；只屏蔽观测中的 wheel position error，不裁剪状态、wheel velocity 或 action；保留 G1/Go2 空 wheel-index 行为。M1 reward 使用 transition 起点 anchor 重建 reference。 |
| `src/supermimic/envs/state_io.py::StateIO`、`src/supermimic/robots/mjlab_assets.py::load_robot_spec_model`、`src/supermimic/training/loop.py::Trainer.__init__` | MuJoCo joint name→qpos/dof address 显式映射；按 qpos address 校验完整顺序，校验运行时 articulation order 与 RobotSpec 一致。 |
| `src/supermimic/envs/mdp/motion_library.py::MotionLibrary.__init__`、`src/supermimic/training/loop.py::M1_MOTION_SETS` | 按 NPZ `joint_names` 重排；校验 Goal joint validity；训练入口将第一阶段选择限制在 manifest clip，拒绝不在集合内的 clip 和 packed library 绕过。 |
| `src/supermimic/data/replay_buffer.py::ReplayBuffer.sample_windows` | 使用动作源行的 `done_next` 标注该动作预测的 successor；不再从 successor 行读入前一转移的标签。 |
| `src/supermimic/scripts/diagnose_multirobot.py` | 短采集、Replay 正类覆盖率、8-step WM 误差/BCE/概率分布、WM warmup 前后、H=8/32 survival/Actor grad/imagined return、仿真 return。Actor optimizer step 固定为 0；CLI 环境数限制为 8 或 16。 |
| `tests/test_multirobot_acceptance.py`、`tests/test_replay_buffer.py` | 持久化编译/运行时关节顺序、M1 数据完整性/观测和奖励 parity、Go2 站立/termination、WM backward、32-step Actor BPTT；另用 CPU 测试锁定 Replay terminal label 对齐。 |
| `src/supermimic/robots/mjlab_assets.py::_convert_visual_mesh/_m1_visual_rpy/_m1_stl_counterpart`、`tests/test_multirobot_acceptance.py::test_m1_visual_meshes_follow_link_frames_and_piper_stls_are_visible` | STL 转换保持 link-local 坐标；M1 `zg_m1` 重复 visual roll 归零；Piper-X 可用 STL 加入 visible geometry 并挂在 URDF 对应 link body。 |

## 复现命令与结果

以下命令在仓库根目录运行：

~~~bash
PYTHONPATH=src conda run --no-capture-output -n supermimic python -m pytest -q tests/test_replay_buffer.py
PYTHONPATH=src conda run --no-capture-output -n supermimic python -m pytest -q -s tests/test_multirobot_acceptance.py
PYTHONPATH=src conda run --no-capture-output -n supermimic python -m pytest -q tests/test_analytic_heads.py tests/test_world_model.py
PYTHONPATH=src conda run --no-capture-output -n supermimic python -m pytest -q tests/test_velocity_training.py tests/test_replay_buffer.py
~~~

结果：Replay done 对齐 **1 passed**；Go2/M1 acceptance **7 passed**；原 G1 analytic-head / World Model 回归 **6 passed**；velocity training + replay checks **7 passed**。GPU 测试产生 Warp CUDA Graphs 和 TorchScript deprecation warnings，不影响断言结果。

短 WM/Termination 诊断命令（两者各 16 env、256 control steps、8 次小批 WM 更新；没有策略优化）：

~~~bash
PYTHONPATH=src conda run --no-capture-output -n supermimic python -m supermimic.scripts.diagnose_multirobot --robot m1_piper --num-envs 16 --steps 256 --wm-updates 8 --window-batch 64 --diagnostic-windows 512 --imag-batch 32 --output logs/multirobot_diagnostics/m1_16env_256step.json
PYTHONPATH=src conda run --no-capture-output -n supermimic python -m supermimic.scripts.diagnose_multirobot --robot go2 --num-envs 16 --steps 256 --wm-updates 8 --window-batch 64 --diagnostic-windows 512 --imag-batch 32 --action-noise 0.05 --output logs/multirobot_diagnostics/go2_16env_256step.json
~~~

脚本使用 128-wide 诊断网络，采集到的回报是当前真实环境的有限 horizon 回报，不代表正式训练表现。`world_model_updates` 对每次更新记录 loss、Term BCE、label ratio、NSE 与 grad norm；报告还区分 Replay 内真实正类数、可采样正类窗口和均匀窗口抽样下的正类比例。

## P0-B：joint order 与 M1 六条轨迹

- G1/Go2/M1 编译 scalar joints 按 MuJoCo `jnt_qposadr` 排列后与 RobotSpec 完全相同；qpos addresses 为 `7..7+n-1`，qvel/dof addresses 为 `6..6+n-1`。M1 runtime mjlab 使用 `robot/FAR_ABAD_JOINT` 这类 entity namespace，StateIO 去掉 namespace 后按名字映射，未把集合相同误当作顺序相同。
- M1 策略 22 维顺序为 FAR/FBL/RAR/RBL，每腿 ABAD/HIP/KNEE/FOOT，随后 `piper_x_joint1..6`；wheel indices `(3,7,11,15)`。测试分别给 wheel index 3 和 Piper index 16 非零 action，只有对应 velocity/position target 改变。Go2 与 G1 编译 order 也通过同一逐项检查。
- manifest `configs/m1_piper_first6.csv` 的 keep=1 恰为下列六条，源于本机 `physx_replay_results.json` 的 `physx_reference_tracking_pass=true`：

~~~text
a_wheel_000000
a_wheel_000002
b_arm_000006
c_assisted_000029
c_assisted_000039
c_assisted_000149
~~~

验收对每条读入 79D finite AMP source frames、source/sidecar SHA256、AMP 与 22D joint order、wheel sidecar 名称和 mask、body order、50Hz `frame_dt=0.02` 与重采样时间戳；检查 root position 插值、xyzw→wxyz Slerp，以及 body-frame linear velocity→world-frame 转换。其余 8 条 keep=0，原 NPZ 未删除或重写。wheel reference 有效性仅保留真实 mask；训练 Goal 仍只使用 18 个非轮 AMP joints。

## P0-A：M1 real/imagined observation parity

16 个 MuJoCo env 关闭 observation corruption，但保留同一 ReplayRecorder dynamics context 中的 encoder bias；analytic observation 使用真实 `action_manager.action` 作为上一动作。覆盖默认关节状态、非零 wheel angle、wheel angle 超过 `2π`、非零 wheel velocity、非零 Piper 关节位置：

| 比较 | 最大逐元素绝对误差 |
|---|---:|
| 单帧 72D proprioception | `2.9802e-8` |
| 10 帧 term-major flatten 的 720D policy observation | `2.9802e-8` |

Wheel position mask 在 `q + encoder_bias - q_default` 后应用，因此完整 World Model wheel angle 仍存在，只有该观测特征置零。Wheel velocity 与上一动作依然被输入 Actor。测试实现见 `test_m1_real_and_analytic_observation_parity_and_action_order`。

## P1-B：M1 每项 Reward parity

16 个 env 执行 12 个小动作步，analytic 端使用同一 reference、transition 起点 anchor、当前/上一 action；Real 列取 RewardManager 的已加权且乘过 dt 的 term。每项均值及误差如下：

| Reward Term | Real mean | Analytic mean | Mean abs error | Max abs error | Mean relative error |
|---|---:|---:|---:|---:|---:|
| `motion_global_root_pos` | 0.00995938 | 0.00995806 | 1.325e-6 | 3.721e-6 | 0.0133% |
| `motion_global_root_ori` | 0.00999578 | 0.00999555 | 2.294e-7 | 6.231e-7 | 0.0023% |
| `motion_body_pos` | 0.01962504 | 0.01959534 | 2.969e-5 | 7.496e-5 | 0.1513% |
| `motion_body_ori` | 0.01942753 | 0.01938365 | 4.388e-5 | 1.096e-4 | 0.2259% |
| `motion_body_lin_vel` | 0.01740428 | 0.01731832 | 8.948e-5 | 1.996e-4 | 0.5141% |
| `motion_body_ang_vel` | 0.01917004 | 0.01914296 | 3.025e-5 | 7.509e-5 | 0.1578% |
| `action_rate_l2` | -1.2003e-5 | -1.2003e-5 | 7.769e-13 | 1.819e-12 | 6.47e-6% |
| `joint_limit` | 0 | 0 | 0 | 0 | 0 |

Tracked body order、reference frame、root/anchor pose、quaternion convention、velocity terms、action rate、joint limit 与 dt/weight 均通过逐项比较。可见最大误差来自 velocity term；mjlab RewardManager 在最终 MuJoCo forward 前读取的 derived body velocity 会有 substep 缓存时差，解析端则由当前 state 重算 FK。Go2 的 reward 原型有 real manager 接线，但尚未做“正式 Reward”验收；不把它当成已定稿目标。

## P0-C：Go2 站立与真实 termination 阈值

测试 `test_go2_zero_action_stand_and_termination_features`：16 env × 400 control steps = 8 秒；控制周期 0.02 秒、4 个 0.005 秒 physics substeps。零动作、无 observation/reset noise、无 domain randomization。

| 指标 | 实测 |
|---|---:|
| 默认起始 root z → 最后 100 步平均 root z | `0.42 → 0.2827106 m` |
| 最后 100 步 root z 标准差 | `5.995e-6 m` |
| 最终倾斜 feature 最大值 `abs(1+g_local_z)` | `2.967e-4` |
| 最后 100 步 root angular velocity 最大绝对值 | `7.37e-5 rad/s` |
| 全程 actuator force 最大绝对值 / effort limit | `22.412 / 23.5`，饱和比例 `0` |
| 四个足端地面接触 | 16/16 env 均有 FL/FR/RL/RR 接触 |
| 其他 ground contact | 无 hip/thigh/calf 接触 |
| 最小 soft joint-limit margin / 最大关节偏离 default | `0.6407 rad / 0.1594 rad` |
| reset/termination | 0 |

四足支撑、root angular velocity/tilt 稳定、膝小腿无地面碰撞，关节和默认姿态偏差小于 `0.16 rad`；结果支持这是当前 URDF+PD+接触模型下的稳定名义站姿。其根高比配置初始值低约 `0.1373 m`，但不是因 actuator 饱和或膝部异常碰撞造成。此项是静态仿真姿态判断，没有做视觉人工检查。

真实 low-height termination 使用 `root_z < 0.17 m`；解析 feature 为 `default_base_height - root_z = 0.42 - root_z`，因此同一边界对应 feature `>0.25 m`。Tilt termination 和第二个 feature 都使用 `abs(1+g_local_z)`，角度参数为 70°，阈值 `1-cos(70°)≈0.6580`。测试真实稳定姿态不终止，并把一组环境设为 `z=0.16 m`、另一组设为 90° 倾倒，检查真实终止布尔值与对应解析条件相同。TerminationHead 本身仍是学得的 sigmoid logit，不是硬编码阈值函数。

没有扫描实际跌倒/趴伏姿态，因此 `0.17 m` 是否过宽仍未定标；不据此调高阈值，也不改 default height、Kp/Kd 或 effort limit。

## P1-C：WM 8-step、Termination、Survival 与 Actor BPTT

诊断均采集 `16×256=4096` 个真实控制 transition；M1 另追加 16 个没有 outgoing action 的 timeout terminal-state rows。M1 所有 16 个结束 episode 都是 reference clip end/timeout，`terminated=0`；Go2 在这 256 步内没有 reset。没有真实跌倒/侧翻 positive transition。Replay 中、8-step 可采窗口中、均匀抽样 batch 中都没有正标签：

| 诊断量 | M1-Piper | Go2 |
|---|---:|---:|
| Replay terminal transition rows / terminal-state rows | `0 / 0`（另有 16 timeout state rows） | `0 / 0` |
| Replay 中可采 8-step windows | 3856 | 3968 |
| 窗口中 positive label 数 | 0 / 4096 label positions | 0 / 4096 label positions |
| 8-step WM NSE，warmup 前 → 后 | 1-step `0.01361→0.01390`；8-step `0.15294→0.15681` | 1-step `0.05296→0.04055`；8-step `0.34482→0.51010` |
| Term BCE，warmup 前 → 后 | `0.66157→0.65347` | `0.69498→0.68504` |
| Term probability quantile `[min,p50,p90,p99,max]`，前 → 后 | `[.4823,.4840,.4842,.4846,.4855]` → `[.4783,.4798,.4800,.4804,.4811]` | `[.4959,.5010,.5011,.5038,.5084]` → `[.4920,.4959,.4961,.4983,.5030]` |
| H=8 survival / Actor grad norm / imagined return | `5.365e-3 / 6.060e-7 / 0.1932` | `4.173e-3 / 1.341e-5 / 0.1129` |
| H=32 survival / Actor grad norm / imagined return | `8.351e-10 / 1.601e-7 / 0.1942` | `3.033e-10 / 3.367e-6 / 0.1133` |
| Real simulation return | 16 完成 episodes：raw `16.765`，normalized `0.9300` | 无完成 episode；未结束 16 env 的 256-step 原型 reward 累计均值 `14.387`，不是正式任务回报 |

解释：负标签代表“这次转移没有真实 termination”，即存活；timeout/clip end 会结束仿真 episode，但不会把 `reset_terminated=0` 错标成跌倒。16-env 短诊断全负类是预期数据事实，不是“机器人全死了”。无需人为构造失败样本；后面的 Go2 训练中真实低高度/倾斜触发 termination 后自然写入正标签。短诊断里初始 Term Head 概率接近 0.5 会使 `(1-d)` 在 32 步连乘而把 survival 压到约 `1e-10`；正式 Go2 短训练的 Head 后来收敛到很低的概率，指标见下一节。Replay 的 off-by-one 标签读取已修复并有 `tests/test_replay_buffer.py` 覆盖。

16-env 诊断不能验收正类区分能力；Go2 短训练只提供了有限的自然跌倒样本，尚不足以证明 Head 对不同跌倒模式的召回率。

## Go2 15 分钟短训练

按用户要求使用仓库当前的 Go2 velocity reward 启动训练；训练受 `max_hours=0.25` 限制，未进行数小时长程训练。最终运行设置为 3072 个真实仿真环境、32 步 imagination、每次 Actor 更新抽取 1024 个起点、200 次 WM-only 更新后开始 Actor/Critic 更新。batch size 只控制一次 imagination 更新的起点数，没有改 reward、WM 损失、32-step horizon 或 BPTT。

~~~bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src conda run --no-capture-output -n supermimic \
python -m supermimic.scripts.train \
  --run-name go2_blindlocomotion_validation_15m_b1024_20261026 \
  --env.robot-name go2 --env.num-envs 3072 --env.task-name velocity_command \
  --policy-trainer.batch-size 1024 \
  --schedule.max-hours 0.25 --schedule.log-interval 50
~~~

最终运行在 1600 个向量控制步、624 次总更新后按时间上限正常退出并保存 checkpoint。真实 transition 数为 `1600 × 3072 = 4,915,200`；其中 200 次 WM-only 更新，之后 424 次 Actor/Critic 更新。每个 Actor 更新使用 1024 个起点并沿模型展开 32 步。训练没有合成或注入失败样本。

| 指标 | 结果 |
|---|---:|
| 8-step WM NSE，it 50 → it 1600 | `0.5046 → 0.00541` |
| 最后记录的 1-step WM NSE | `0.00442` |
| Termination label ratio / probability mean（it 1600） | `1.617e-5 / 1.903e-5` |
| Termination BCE（it 1600） | `2.5485e-4` |
| 32-step mean survival（it 1600） | `0.99943` |
| Imagined return（it 1600） | `1.736` |
| Actor grad norm（it 1600） | `0.0252` |
| Real episode return | `26.62`（it 1000，3072 个 500-step timeout episodes）；`26.54`（it 1500，2724 个 timeout episodes）；`26.44`（it 1550，348 个 timeout episodes） |
| 自然失败样本 | it 1050 有 348 个提前终止 episode，平均长度 `14.40` 步、raw return `0.376`、fall rate `1.0`；后续 it 1500/1550 评估 fall rate 为 `0.0` |

这轮训练说明真实失败会自然进入 Replay：早期一批 episode 在约 14 个控制步后跌倒，后续成批 episode 则走完 500 步。WM 采到的 terminal-label 比例约 `1.6e-5`，预测概率均值约 `1.9e-5`；H=32 survival 因此接近 1。该短训练说明标签链路能处理自然正类，不能据低 BCE 单独断言 Head 已学会所有跌倒形态。Actor grad norm 非零；it 1600 的 actor ramp scale 为 `0.424`，`policy_grad_norm` 是 clip 前的 Actor 参数梯度范数。

第一次使用默认 imagination batch `4096` 时，切入 Actor 的 BPTT 更新发生 CUDA OOM，没有 checkpoint（日志：`logs/multirobot_diagnostics/go2_blindlocomotion_validation_15m_20261026.log`）。第二次将 batch 改为 `2048` 后越过 Actor 编译，但在 it 850 的 MuJoCo-Warp CCD 临时分配 `75,497,472 bytes` 失败，也没有 checkpoint（日志：`logs/multirobot_diagnostics/go2_blindlocomotion_validation_15m_b2048_20261026.log`）。最终使用 batch `1024` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 完成上述训练。checkpoint 为本机 `logs/go2_blindlocomotion_validation_15m_b1024_20261026/ckpt_latest.pt`；运行日志为 `logs/multirobot_diagnostics/go2_blindlocomotion_validation_15m_b1024_20261026.log`。较早一次 M1-Piper 2048-env 运行在 MuJoCo-Warp CCD 临时分配 `465,567,744 bytes` 时 OOM（日志：`logs/multirobot_diagnostics/m1_piper_validation_15m_20261026.log`）；后续有界 M1 训练和策略验收见本文件的“M1 15 分钟短训练与 Play mesh follow-up”。

## G1 回归、限制与待定项

- `tests/test_analytic_heads.py tests/test_world_model.py`：**6 passed**；原 G1 analytic reward/observation 与 WM replay regression 通过。
- `tests/test_multirobot_acceptance.py`：**7 passed**；额外 CPU Replay done-alignment 测试：**1 passed**。
- WM loss 和 32-step Actor 梯度在多机器人 acceptance 中做 backward smoke；诊断期间 `policy_optimizer_steps=0`。
- Go2 当前 11 项 velocity reward 已用于上述短训练；15 分钟结果不代表长程收敛或课程晋级通过。M1 多轨迹 reward 设计留待下一轮。
- 没有进行数小时长程训练、完整课程晋级验收、复杂地形/LiDAR 测试或抓取实验。

## M1 15 分钟短训练与 Play mesh follow-up

按用户指定先完成 M1 的 15 分钟短训练，再用 Play 检查 M1 与 Go2。M1 配置是 2048 env、Motion Set C 的六条白名单 clip、32-step imagination、batch 128、300 次 WM-only 更新后开始 Actor/Critic；时间上限触发时在 iteration 1000 保存 checkpoint。实际产生约 `1000×2048=2,048,000` 个真实控制 transition，共 384 次更新（300 次 WM-only、84 次 Actor/Critic）。本机产物：

- checkpoint：`logs/m1_piper_motion_C_2048_safe_15m_20261026/ckpt_latest.pt`
- 训练日志：`logs/multirobot_diagnostics/m1_piper_motion_C_2048_safe_15m_20261026.log`
- Play 数值评估：`logs/m1_piper_motion_C_2048_safe_15m_20261026/eval_C.json`
- Play 截图：`/tmp/supermimic_play_m1_stl_fixed.png`

该 checkpoint 的 8-step WM NSE 为 `0.1343`、Term BCE `0.0932`、H=32 survival `0.1021`、imagined return `0.9943`、Actor grad norm `0.0297`。六条 reference clip 均完成播放，但 success rate 为 `0.0`；root position error 范围 `0.104–0.175 m`，Piper arm joint error `0.253–0.602 rad`，TCP error `0.365–0.549 m`。因此策略尚未达到 tracking 成功标准，Play 中的动作姿态偏差不能归咎为 STL 位姿故障，也不能把该短训练称为训练收敛。

Play mesh 修复在 `robots/mjlab_assets.py`，不需要改 `scripts/play.py` 的动作或训练逻辑：

1. 超大 `BASE_LINK.STL` 转 OBJ 时，Blender STL 导出使用保留坐标的 `Y/Z` 轴定义；缓存版本改为 `v4`，旧缓存不再复用。新 OBJ 顶点包围盒和原 STL 的三轴边界最大差 `6.2e-5 m`（网格 decimation 误差）。DAE 仍沿用原转换和缓存。
2. M1 四足 `meshes/zg_m1/*.STL` 的原始长轴、负 Z 膝关节位移和 collision link frame 一致；源 URDF 上重复的 `roll=-90°` 会把腿部视觉 mesh 转向侧面。Play/FK 编译模型只对这些视觉 STL 将该 roll 设为 0，不改 collision geom、关节坐标、RobotSpec 或 action。
3. Piper-X 集成 URDF 的视觉引用以 DAE 为主，但同目录提供 `base_link.STL`、`Link1..Link6.STL`。对存在对应 STL 的 visual mesh，加载器使用对应 STL；gripper DAE 无同目录 STL 对应时仍保留 Blender DAE 转换。`test_m1_visual_meshes_follow_link_frames_and_piper_stls_are_visible` 检查编译后 mesh geom 处于 visible group 且挂在对应 link body。对照 URDF 递归 FK 与编译模型，43 个 body frame 的最大位置差约 `2.23e-16 m`。

定点验收命令：

~~~bash
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m pytest -q tests/test_multirobot_acceptance.py::test_m1_visual_meshes_follow_link_frames_and_piper_stls_are_visible
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py --ckpt logs/m1_piper_motion_C_2048_safe_15m_20261026/ckpt_latest.pt --num-envs 1 --viewer native --ghost False
~~~

结果：视觉资源测试 **1 passed**；Play 在 1 env 下启动成功，实际加载 M1-Piper URDF，22 actions、720D observation，并从第一条 clip 开始。截图中 M1 quadruped STL 和 Piper-X `base_link/Link1..Link6` 可见；STL 是无纹理网格，显示颜色由 URDF rgba 决定。该视觉检查与短训练策略成功率分开记录。

同日 Go2 checkpoint 也用 `play.py` 运行过（`logs/multirobot_diagnostics/play_go2_15m_checkpoint_after_m1.log`，屏幕记录 `/tmp/supermimic_play_go2_after_m1.png`）。用户指出画面中 Go2 关节错位、机体悬浮且没有前进；本轮改动只处理 M1 STL，未宣称 Go2 Play 通过。
