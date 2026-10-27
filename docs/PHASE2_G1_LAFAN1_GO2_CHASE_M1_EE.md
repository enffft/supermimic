# Phase 2 源码审计与训练改进记录

日期：2026-10-27
本地起始 HEAD：`b7a8637906fdd42ca6ed0e108a19f88deeeb58bf` (`feat: add G1 Go2 M1 phase-one training support`)
远端审查基准：同一 commit。代码按本地仓库实现审查；参考仓库仅克隆到 `/tmp`，没有修改。

起始工作区另有用户未提交的 `docs/4070_LAFAN1_Q1_TRAIN.md` 修改，以及未跟踪的 `.vscode/`、`docs/SUPERMIMIC_CODE_WALKTHROUGH.md`。本轮保留这些文件，不纳入本轮实现提交。

## 结论摘要

| 任务 | 本轮代码实现 | 真实仿真验证 | 当前能否称为训练成功 |
|---|---|---|---|
| G1 LAFAN1 | 加入与 Breezewrf 一致的 268D 全身参考输入；保留旧 68D 配置；RunningNorm 与 6D Orientation Anchor 可选；增加可选 WM 预热 | 40 条数据完整载入；1024 环境 smoke 与旧 checkpoint 回归通过；新 1024-env、12h scratch run 已启动，实时全 40 clip 评估 watcher 已挂载 | 否。12h 训练尚未完成首个全 40 clip checkpoint 评估 |
| Go2 Chase Ball | 新增无 Motion Library、无感知传感器的 `chase_ball` 任务、6D 目标、Replay Goal、奖励、tag/fall 分离、评估和可视化 | 3072 env、有界 15 分钟 WM 初始化训练结束；真实 MuJoCo 已评估 64 episodes | 否。tag `1/64`，平均接近速度 `.0136 m/s`，仍不是可靠追球 |
| M1 World Point | 沿用既有 `world_point_reach`，统一 TCP proxy；调整饱和过早的接近项；加入近/中/远目标课程与 TCP 误差时间曲线评估 | 128 env、有界 15 分钟 near-stage 训练和 128 episode 真实评估结束；FK 与 MuJoCo TCP 差 `5.56e-7 m` | 否。episode 平均 `.22 s` 即因 base height 跌倒终止，success 为 0% |

## 参考源码对照

| 来源 | 参考项目做法 | 本仓库原状 | 本轮采用与拒绝 |
|---|---|---|---|
| Breezewrf `supermimic_mjlab` `23c76cad5bf41110709c7f929b54c1da1be6b248` | Actor/Critic 输入归一化、World Model 输入/增量归一化、相对 Anchor 姿态、268D full-body motion features | 68D/frame root+joints Goal、固定 StateSpec 尺度，G1 默认无 Anchor | 采用可选 RunningNorm、6D Orientation Anchor 和原样复现其 268D/frame 全身特征；旧 68D/689D 结构仍保留。未直接迁入其新 73D root schema，也不把报告中的组合收益当作我们模型的预期收益 |
| `go2-supermimic` `d9eee9b3f248e5300e636c433255db2c1c3e086a` | Chase 方向接近、朝向、目标 tag；目标状态由任务管理，Replay 训练动力学；另有 Dodge、障碍、地形与传感器任务 | 本仓库只有 Go2 velocity command，没有 Chase Ball | 只实现平地低速球目标、6D 相对位置/速度、方向接近和 tag；拒绝横杆、躲避、投掷物、LiDAR、高程扫描、障碍物和追球速度命令。想象中的球采用固定速度匀速外推，真实目标也在 episode 内保持该速度；没有假称学习球动力学 |
| `supermimic_manipulation` `810e5768c50117d2fec75f20f9e643a11cd0882b` | 无参考的 Goal Conditioned Actor、状态重建、可微 FK 和任务适配器 | Phase 1 已有 M1 `world_point_reach`、固定 env-local 目标和 TCP proxy | 复用既有任务链路和 22D Mixed Action，只调整目标课程、距离进度项与评估。没有引入抓取、接触近似、OmniReset、物体动力学或新的训练器 |

## G1：40 条 LAFAN1 与身体朝向低奖励

### 源码与数据审计

- `data/npz/lafan1` 下 40 条轨迹全部可读，共 `441,140` 帧、约 `2.451 h @ 50 Hz`。复位采样按片段帧 bin 进行，长短轨迹样本数按时长变化，没有发现某一条轨迹独占 Replay 或复位采样的问题。
- 原 `root_joint` Goal 每帧 68D：关节位置/速度、重力投影、角速度、平面线速度、机体高度。它没有逐身体参考姿态；6D Anchor 提供 pelvis 朝向相对误差，但不能替代手臂、腿等身体参考输入。
- 参考工程精确的 full-body 每帧 schema 为 268D，字段顺序是 `joint_pos(29), joint_vel(29), body_pos(14×3), body_orientation_6D(14×6), body_linear_velocity(14×3), body_angular_velocity(14×3)`。本实现采用 root-yaw frame、pelvis XY 锚定、body orientation 前两列、位置除以 0.3、关节速度除以 5、角速度除以 3.14；数据均能由现有 14 个跟踪身体和 NPZ 无损构造。10 帧 + 6D Orientation Anchor 为 2686D Goal。
- 当前 40-clip baseline：`logs/4070/lafan1_anchor_1024_8h_resume02_20261026/benchmark_40clips.json`，mean normalized return `0.429708`，step-weighted `0.432496`，39/40 成功。类别结果：walk `.4779`、run `.3513`、sprint `.4560`、dance `.4254`、jumps `.4333`、fight `.4118`、fall/get-up `.3957`。
- 同一 benchmark 记录的逐步 RewardManager term 均值为 root position `.008081`、root orientation `.003206`、body position `.012235`、body orientation `.00001023`、body linear velocity `.012446`、body angular velocity `.007872`；Action rate 为 `-.000540`、joint limit 为 `-.000060`。这些数值解释低 body-orientation 项，但不是单独的姿态误差指标。
- 同一旧 checkpoint 的确定性 `walk1_subject1` 回归评估得分 `.4970`、未失败，证明旧 9D Anchor / 689D Goal checkpoint 仍按旧结构加载。该单 clip 结果不与 full-body 新模型作 40-clip 对比。

### Body Orientation 低奖励原因

旧 checkpoint 的 40-clip 评估中 `motion_body_ori` 平均为 `1.0226e-5`/step，而该项理论最大值为 `0.02`/step。奖励实现为：

```text
0.02 × exp(-mean_geodesic_error² / 0.4²)
```

旧策略在 `walk1_subject1` 上的重锚定身体朝向误差均值约 `0.950 rad`，root 朝向误差约 `0.103 rad`。相同状态下可微 FK 与 MuJoCo body orientation 的最大差约 `1.3e-6 rad`；四元数顺序、body list 顺序、世界坐标系和重锚定实现一致。因此证据指向：身体姿态本身跟踪误差大，`0.4 rad` 的窄指数尺度进一步压低该项；不是日志多乘了 `dt` 或 FK 坐标系错位。没有改大该奖励系数。

### 采用的 G1 改进和兼容性

- `MotionLibrary` 增加可选 `full_body` Goal schema，默认仍为 `root_joint`；旧网络维数、旧 9D Anchor 与旧评估接口不变。
- full-body checkpoint 使用 2686D Goal + encoded-motion Actor，不能直接 Resume 689D/9D Anchor checkpoint。原 checkpoint 继续严格恢复；新结构从头训练，不使用 `strict=False` 或未经验证的参数切片迁移。
- RunningNorm 作用于现有固定尺度后的 World Model canonical input 和 state-delta target；网络预测标准化 delta，积分前反归一化回原始 delta 单位。均值/方差只由新采集的真实 Transition 更新；Replay 采样和 Imagination 不更新，反向梯度穿过固定统计值。旧配置缺少字段时仍禁用。
- `ScheduleCfg.g1_motion_wm_only_updates` 是 opt-in（默认 0）。新结构的早期 smoke 在仅 8 个更新时出现 32-step survival=0，因此正式新跑命令先预拟合 WM 200 次更新，再进入 Actor BPTT；旧 Resume 默认不改变历史调度。
- 新 full-body 1024-env smoke：40 clips；Goal 2686D；RunningNorm input/delta 统计各累计 61,440 个真实状态；60 iteration、8 次 WM / 8 次 Actor 更新。末次 1/8-step NSE 为约 `.211 / 3.17`，32-step survival 为 0。新结构的 8-env、50-iteration checkpoint 在单条 `walk1_subject1` 评估于第 46 步失败，得分 `.4308`。这些只说明执行链路通过，且早期模型尚未适合评估，不能当作性能改进结果。

## Go2：纯 Chase Ball

- Task/命令名为 `robot_name=go2, task_name=chase_ball`。Actor 输入为 420D（42D×10 历史）+ 6D Goal `[球相对位置 xyz, 球速度 xyz]`，仅使用 Go2 joint/IMU/action history；不含 base linear velocity、速度命令、LiDAR 或高度图。World Model 仍只预测 Go2 37D 状态，Replay 固定保存 env-local 球世界坐标和速度。
- 球位置以 0.70–1.10 m 前向低角度起步，Stage 0 仅 35% 球为移动目标、速度不超过 0.05 m/s；Stage 1/2 才增加距离、角度和速度，按真实 tag 成功率晋级。球是可视化/逻辑目标，不是动态刚体，也没有接触求解。
- 真实/想象共用方向接近、距离 shaping、tag、朝向和稳定项；真实 tag 是一次性 reward 并作为成功 timeout 结束 episode，不记为摔倒或 WM failure label。Imagination 对预测球位置按当前速度确定性外推，以可微 tag 概率结束 survival。RewardManager 和 imagination 均只乘一次 `dt`。
- 从旧 Go2 `velocity_command` checkpoint 初始化 WM 已用 1 iteration 完整训练入口核实：robot/URDF hash、joint order、actuator version、StateSpec、dt、终止阈值、WM 宽度和归一化语义相同时 strict-load WM；Actor、Critic、Replay 与任务课程均新建。旧 Go2 checkpoint 的失败速度表现不能算新 Chase 结果。
- Go2 有界 WM 初始化训练已实际结束，checkpoint 为 `logs/phase2_training/go2_chase_initwm_3072_b1024_20261027/ckpt_latest.pt`；`1750` iteration、`684` WM / `684` Actor updates、`5,376,000` vectorized real transitions，墙钟预算 `.25 h`。Replay 不随 checkpoint 保存。训练期间 8-step WM NSE 约 `.012–.019`、Imagination Survival 约 `.95–.97`，但真实 Chase 不是由这些指标判定。
- 最终 64-episode Stage 0 真实评估在 `logs/phase2_training/go2_chase_initwm_3072_b1024_20261027/eval_stage0_64ep.json`：初始球距 `.713 m`、整段平均距 `.648 m`、最终距 `.614 m`、tag `1/64=1.56%`（平均 tag 时间 `9.42 s`）、机体位移 `.145 m`、平均接近速度 `.0136 m/s`、跌倒 `3.13%`、timeout `95.31%`、动作饱和 `18.49%`、NSE 1/8 `.00538/.01138`、Imagination Survival `.875`。相比 80-step smoke 有微弱正向接近，但不是可靠追球策略。
- 训练 TensorBoard 的末段真实单项奖励揭示一个明确的目标量级问题：approach 约 `-0.00017` 至 `0.00007`，facing 约 `+0.0090`，upright 约 `+0.0068`/step。即使机器几乎不前进，也能持续取得朝向和站立的正奖励。若继续做一次修正，优先把 facing/upright 改成零基线的偏差惩罚，或让朝向奖励只在正向接近时生效；不要再堆传感器或复杂任务。按要求本轮没有启动第二组 Chase 长训。

## M1-Piper：World-frame TCP Position Reach

- 沿用 `WorldPointCommand`、`WorldPointReachTask`、57D M1 WM 状态、Direct 720D proprioceptive history + 3D body-relative target Actor 和 22D Mixed Action。
- 固定目标为 episode 内 env-local world 坐标；Actor 每一步将同一世界目标按当前（或 imagined predicted）base pose 转为 body frame，Reward 用固定目标与 TCP 的世界坐标误差。
- 统一 TCP proxy 为 `midpoint(piper_x_link7, piper_x_link8)`。它是 Piper URDF 两指 frame 中点，不宣称等同外部标定 TCP。命令采样、Reward、Evaluator、Play 共用该定义。
- `tcp_approach` 改为 `0.25 * tanh((d_prev-d_next)/0.05)`，避免旧速度式 tanh 在约 5mm 单步位移时已饱和；保留负距离、近目标、稳定、关节范围、动作率和超速项。stage 0 只采近处机械臂目标；真实 success hold 达阈值后才提升到近/中、再到近/远混合。
- 16-env/80-iteration smoke 更新 6 WM / 5 Actor。真实 16-episode Stage 0 评估：初始误差均值 `.269 m`，平均轨迹误差 `.450 m`，最终误差 `.455 m`，success `0%`，base displacement `.0023 m`，arm joint motion `.342`，跌倒 `0%`，timeout `100%`。误差曲线 `.0/.5/1/2/3/5/7.5/10 s` 分别为 `.269/.439/.476/.463/.455/.453/.455/.455 m`；评估文件为 `logs/phase2_smoke/phase2_m1_near_smoke_16env/eval_near_16ep.json`。FK proxy 与 MuJoCo 两指中点均值差 `2.9e-7 m`。真实 RewardManager 的 `tcp_distance` 约 `-.0134`/step，`upright` 约 `+.0050`/step，`tcp_approach` 与 `tcp_near` 接近 0；距离项已经大于直立正奖励。当前证据支持目标方向动作还没学出来，而非盲目增大距离项。最直接现象是运动使 TCP 先远离目标，底盘几乎没动。

## 修改文件

- G1：`src/supermimic/envs/mdp/motion_library.py`、`src/supermimic/envs/tracking_env_cfg.py`、`src/supermimic/training/loop.py`、`src/supermimic/config.py`。
- Go2 Chase：`src/supermimic/tasks/chase_reward.py`、`src/supermimic/envs/mdp/commands.py`、`src/supermimic/envs/mdp/observations.py`、`src/supermimic/envs/mdp/terminations.py`、`src/supermimic/envs/multirobot_env_cfg.py`、`src/supermimic/imagination/{tasks,compiled,rollout,policy_trainer}.py`、`scripts/evaluate_go2_chase.py`、`scripts/play.py`。
- M1：`src/supermimic/tasks/world_point_reward.py`、`src/supermimic/envs/mdp/commands.py`、`scripts/evaluate_m1_world_point.py`。
- G1 benchmark 的历史结构读取和真实评估 Best checkpoint 选择：`src/supermimic/eval/benchmark.py`、`scripts/eval_benchmark.py`。
- 覆盖：`tests/test_chase_ball.py` 和已有 action/WM/world-point/multirobot tests。

## 验证结果

正确环境为 `conda run -n supermimic ...`（该环境使用 Torch 2.11.0+cu126，RTX 4070 SUPER 可用）。系统 Python 缺 PyTorch/mjlab，仓库 `.venv` 使用 CUDA 13 Torch，而本机驱动为 12.2，不可用于 GPU 验证。运行测试时需用 `conda run -n supermimic env PYTHONPATH=src python -m pytest`，避免调用系统 pytest。

- Phase 2 修改后的阶段测试：64 passed。`compileall` 和 `git diff --check` 通过。
- `--help` 已核对训练、G1 benchmark、Go2 Chase evaluator、M1 evaluator 和 Play 参数。
- G1 旧 checkpoint 用旧 `root_joint`/9D Anchor strict restore 成功；新输入维数独立训练。
- Go2 WM 任务间初始化 strict-load smoke 成功，未加载 Actor/Critic。
- M1 FK/目标坐标和 Go2/M1 Play 环境均经过 GPU smoke。

## 训练命令

以下正式训练命令保留完整 `320000` iteration cosine 周期（衰减到 250000 后保持 floor），用 `max-hours` 限制墙钟时长。全 40 clip 是默认行为，不要传 `--data.clips`。

### 1. G1 40 条 full-body 新训练（scratch）

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name g1_lafan1_fullbody6_norm_1024_12h \
  --log-root logs/4070 --device cuda:0 \
  --env.robot-name g1 --env.task-name motion_tracking --env.num-envs 1024 \
  --env.motion-goal-schema full_body --env.anchor-orientation-obs \
  --env.replay-capacity 256 --world-model.running-normalization \
  --wm-trainer.window 8 --wm-trainer.batch-size 1024 \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 1024 \
  --schedule.total-iterations 320000 --schedule.lr-decay-fraction 0.78125 \
  --schedule.g1-motion-wm-only-updates 200 \
  --schedule.checkpoint-interval 20000 --schedule.max-hours 12
```

这是独立的新模型，不能添加旧 checkpoint 的 `--resume`。另开终端在每个 checkpoint 完成后运行全 40 条确定性评估：

```bash
conda run -n supermimic python scripts/eval_benchmark.py \
  --ckpt logs/4070/g1_lafan1_fullbody6_norm_1024_12h/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 1024 \
  --out logs/4070/g1_lafan1_fullbody6_norm_1024_12h/benchmark_latest.json \
  --best-checkpoint logs/4070/g1_lafan1_fullbody6_norm_1024_12h/ckpt_best_eval.pt \
  --best-metrics logs/4070/g1_lafan1_fullbody6_norm_1024_12h/best_eval.json
```

### 2. G1 strict Resume 与确定性全 40 评估

Resume 必须保持相同 `motion-goal-schema=full_body`、6D Anchor、RunningNorm、网络宽度和机器人参数。Replay 不随 checkpoint 保存，建议将 `warmup-iterations` 设为 checkpoint `iteration + 40` 重新收集窗口：

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name g1_lafan1_fullbody6_norm_1024_resume \
  --log-root logs/4070 --device cuda:0 \
  --env.robot-name g1 --env.task-name motion_tracking --env.num-envs 1024 \
  --env.motion-goal-schema full_body --env.anchor-orientation-obs \
  --env.replay-capacity 256 --world-model.running-normalization \
  --wm-trainer.window 8 --wm-trainer.batch-size 1024 \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 1024 \
  --schedule.total-iterations 320000 --schedule.lr-decay-fraction 0.78125 \
  --schedule.g1-motion-wm-only-updates 200 --schedule.warmup-iterations 120040 \
  --resume logs/4070/g1_lafan1_fullbody6_norm_1024_12h/ckpt_latest.pt
```

`120040` 只适用于从 iteration `120000` checkpoint 恢复；按实际 global iteration 改成 `iteration + 40`。评估命令同上，确定性策略直接输出 Actor action。

旧 G1 架构的全 40 评估：

```bash
conda run -n supermimic python scripts/eval_benchmark.py \
  --ckpt logs/4070/lafan1_anchor_1024_8h_resume02_20261026/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 1024 \
  --out logs/4070/lafan1_anchor_1024_8h_resume02_20261026/benchmark_40clips_regression.json
```

### 3. Go2 Chase Ball 从零训练

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name go2_chase_scratch_3072_b1024 \
  --log-root logs --device cuda:0 \
  --env.robot-name go2 --env.task-name chase_ball --env.num-envs 3072 \
  --env.velocity-policy-encoding direct --env.direct-action-output tanh \
  --env.chase-initial-stage 0 --env.chase-auto-curriculum \
  --wm-trainer.window 8 --wm-trainer.batch-size 1024 \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 1024 \
  --schedule.total-iterations 320000 --schedule.lr-decay-fraction 0.78125 \
  --schedule.wm-only-iterations 200 --schedule.checkpoint-interval 4000
```

### 4. Go2 Chase Ball 初始化兼容旧 WM

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name go2_chase_initwm_3072_b1024 \
  --log-root logs --device cuda:0 \
  --env.robot-name go2 --env.task-name chase_ball --env.num-envs 3072 \
  --env.velocity-policy-encoding direct --env.direct-action-output tanh \
  --env.chase-initial-stage 0 --env.chase-auto-curriculum \
  --wm-trainer.window 8 --wm-trainer.batch-size 1024 \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 1024 \
  --schedule.total-iterations 320000 --schedule.lr-decay-fraction 0.78125 \
  --schedule.checkpoint-interval 4000 \
  --init-wm-from logs/go2_velocity_phase1_3072_b1024_20261027/ckpt_latest.pt
```

此时只加载兼容 WM，Actor/Critic/Replay/课程从头开始。第一次启动默认 `max-hours=None`，可加 `--schedule.max-hours 0.25` 做有界验证。真实评估与 Play：

```bash
conda run -n supermimic python scripts/evaluate_go2_chase.py \
  --checkpoint logs/go2_chase_initwm_3072_b1024/ckpt_latest.pt \
  --num-envs 128 --episodes 128 --target-stage 0 \
  --out logs/go2_chase_initwm_3072_b1024/eval_stage0.json

conda run -n supermimic python scripts/play.py \
  --ckpt logs/go2_chase_initwm_3072_b1024/ckpt_latest.pt \
  --num-envs 1 --viewer native --chase-debug-vis
```

### 5. M1 近到远课程训练、评估与 Play

```bash
conda run -n supermimic python -m supermimic.scripts.train \
  --run-name m1_world_point_curriculum_128 \
  --log-root logs --device cuda:0 \
  --env.robot-name m1_piper --env.task-name world_point_reach --env.num-envs 128 \
  --env.m1-target-curriculum --env.m1-initial-target-stage 0 \
  --env.m1-world-point-wm-batch-size 256 --env.m1-world-point-actor-batch-size 256 \
  --policy-trainer.horizon 32 \
  --schedule.total-iterations 320000 --schedule.lr-decay-fraction 0.78125 \
  --schedule.m1-wm-only-iterations 200 --schedule.checkpoint-interval 4000

conda run -n supermimic python scripts/evaluate_m1_world_point.py \
  --checkpoint logs/m1_world_point_curriculum_128/ckpt_latest.pt \
  --num-envs 128 --episodes 256 --target-stage 0 \
  --out logs/m1_world_point_curriculum_128/eval_near.json

conda run -n supermimic python scripts/evaluate_m1_world_point.py \
  --checkpoint logs/m1_world_point_curriculum_128/ckpt_latest.pt \
  --num-envs 128 --episodes 256 --target-stage 2 \
  --out logs/m1_world_point_curriculum_128/eval_far.json

conda run -n supermimic python scripts/play.py \
  --ckpt logs/m1_world_point_curriculum_128/ckpt_latest.pt \
  --num-envs 1 --viewer native --ghost True
```

## 实际训练进程与未解决项

- Go2 15 分钟上限初始化训练已结束，真实评估为 1/64 tag 和 `.0136 m/s` 平均接近速度：有少量朝向目标位移，但远未达到可靠追逐。新 checkpoint 可续训，Replay 将重新 warmup。
- 新 G1 的 12 小时正式训练已于本轮启动：`logs/4070/g1_lafan1_fullbody6_norm_1024_12h`，1024 env、full-body 2686D Goal、6D Anchor、RunningNorm，scratch 初始化。启动后首分钟检查为 64 个 WM updates、actor updates `0`（WM-only 预热中），NSE 1/8 为 `.0290/.0646`；这些早期数据不是质量结果。一个独立的本地 watcher 在每个 20,000-iteration checkpoint 后用 128 env 对全 40 clips 评估，写 `benchmark_*.json` 并更新真实评估最佳 checkpoint。旧 `.4297` 只作基线。
- M1 128-env、near stage 的 15 分钟真实训练已完成：iteration `3300`，总更新 `1304`（WM-only `200`，Actor 更新 `1104`）；WM NSE 1/8 为 `.0068/.0214`，但末段 32-step imagination survival 约 `.002`。真实 MuJoCo 128 episode：初始 TCP 距离 `.2721 m`、最终 `.2615 m`，成功 `0%`；平均 episode 只运行 `.22 s`，跌倒率 `100%`、timeout `0%`，终止 base height `.1897 m`、tilt `1.08°`，base displacement `.00275 m`。FK 与 MuJoCo TCP 误差 `5.56e-7 m`，所以这不是 FK/坐标误差。训练期 Actor `mean(abs(action))=.994`，更符合 22D action 饱和并迅速触发绝对 base-height termination 的证据。已保存 `logs/phase2_training/m1_world_point_near_128_15m_20261027/eval_near_128ep.json`；暂不扩 env 或长训。下一次 M1 训练应先限制饱和混合动作造成的物理 setpoint 幅度，再跑短真实检查。
- Go2 在旧速度任务中没有产生速度跟踪，但其 WM 在跨任务物理语义通过严格检查后可以初始化 Chase；其新 Chase 结果必须看真实位移、接近速度与 tag，不看 imagined return 单项。
- CUDA driver 12.2 使 MuJoCo Warp 关闭 CUDA Graph。运行期间显存与吞吐仍需以实际设备监控；不能仅凭 1024 env 初始化成功断言 12h 性能足够。

## 四种完成状态

1. **代码已实现**：G1 Full-body Goal、Chase Ball 任务与可视化、M1 TCP progress 和距离曲线评估。
2. **代码已验证**：65 项 pytest、CLI help、compileall、1024-env G1 smoke、Go2/M1 环境/训练 smoke、旧 G1 strict checkpoint 回归。
3. **真实仿真已评估**：G1 旧 checkpoint 单 clip；Go2 Chase 有界训练后评估 64 episode；M1 15 分钟近目标训练后评估 128 episode，终止时间、base height 和 tilt 均有记录。
4. **任务已训练成功**：本轮没有任何一个新任务达到该状态。G1 新架构 40-clip 指标、Go2 可靠追球和 M1 近/远 TCP 到达仍待验证。
