# M1-Piper Motion Prior、Motion Bank 与 SuperMimic 接口总览

更新时间：2026-11-02

本文整理 AMP-LOCO-M1 已生成的 M1-Piper Motion Bank 与 Conditional VAE Motion Prior，以及当前 SuperMimic M1-Piper 环境、网络和动作接口。它是跨仓库接口参考，不表示 CVAE 已经集成进 SuperMimic、已经通过 PhysX 训练，或已具备真实抓取能力。

## 1. 版本与来源

审计时 SuperMimic 本地代码基线为 `3638d9b8bf3e57ce26e49cc4ca8957de70c9cf08`。当时本地工作区有未提交修改：`src/supermimic/config.py`、`src/supermimic/envs/mdp/commands.py`、`src/supermimic/imagination/policy_trainer.py`、`src/supermimic/policy/nets.py`、`src/supermimic/training/loop.py`，以及 `.vscode/`、`docs/G1_PURE_PROPRIO_REMEDIATION.md`、`tests/test_g1_actor_contract.py` 未跟踪内容。本文的现状描述应理解为基于该 HEAD 加审计时本地 M1 相关改动的快照；本次文档提交不包含这些文件。

AMP Motion Bank 和 CVAE 位于同机的另一个仓库 `/home/yifeiy/AMP-LOCO-M1`，不是 SuperMimic 仓库的子目录。CVAE 相关源码/正式报告最后核对版本为 AMP `main` 的 `a5d90615ffbf842d626a0f66ec920c2880252352`；CVAE 模型主体首次提交于 `97ccc1cade024fdc638a30298586b779376f395f`。关键 AMP 文档：

- [Motion Bank v2.5 Final 数据库报告](https://github.com/enffft/AMP_LOCO_M1/blob/main/docs/M1_MOTION_BANK_V2_5_FINAL_DATABASE.md)
- [M 类 EE 分布与全库审计摘要](https://github.com/enffft/AMP_LOCO_M1/blob/main/docs/M1_MOTION_BANK_V2_5_FINAL_DATABASE.md#m-ee-分布)
- [Conditional VAE Motion Prior v1 训练与评价报告](https://github.com/enffft/AMP_LOCO_M1/blob/main/docs/M1_CVAE_MOTION_PRIOR_V1_TRAINING.md)
- [SuperMimic 接入静态审计与奖励建议](https://github.com/enffft/AMP_LOCO_M1/blob/main/docs/M1_SUPERMIMIC_CVAE_INTEGRATION_AUDIT.md)

## 2. Motion Bank v2.5 Final

正式数据源只使用 AMP 仓库中的：

```text
/home/yifeiy/AMP-LOCO-M1/dataset/m1_kinematic_motion_bank_v2_5_final/
```

关键文件：

```text
dataset/m1_kinematic_motion_bank_v2_5_final/manifest.json
dataset/m1_kinematic_motion_bank_v2_5_final/motions/<motion_id>.txt
dataset/m1_kinematic_motion_bank_v2_5_final/motions/<motion_id>.npz
dataset/m1_kinematic_motion_bank_v2_5_final/reports/merge_validation.json
dataset/m1_kinematic_motion_bank_v2_5_final/reports/collision_report.json
dataset/m1_kinematic_motion_bank_v2_5_final/reports/collision_audit.jsonl
dataset/m1_kinematic_motion_bank_v2_5_final/reports/audit_provenance.json
dataset/m1_kinematic_motion_bank_v2_5_final/reports/ee_distribution_summary.json
dataset/m1_kinematic_motion_bank_v2_5_final/reports/m_ee_distribution_by_source.csv
dataset/m1_kinematic_motion_bank_v2_5_final/reports/m_ee_trajectory_distribution_by_source.csv
```

| 任务类 | 数量 | 含义 |
|---|---:|---|
| L | 999 | Velocity-conditioned locomotion |
| M | 1,000 | 固定 EE Goal、Base XY 静止的 Reach |
| LM | 273 | 固定 EE Goal、Base 同时运动的 Reach |
| **总计** | **2,272** | **521,661 帧** |

每帧采样间隔 `0.02 s`，原始 AMP Frame 为 `79-D`。2,272 条 TXT/NPZ 成对存在；旧合格 M 280 条 + 新 M 720 条；新增 M 重命名为不冲突的 `m_001000` 至 `m_001719`。合并记录检查了配对、Manifest、SHA-256、finite 数值和审计状态。旧 M 的文件与新增 M 均保留 provenance。Motion Bank 的 Full-FCL/结构/运动学审查属于**来源参考轨迹**；`physx_validated=false`。这不证明 PhysX 中能跟踪，也不证明 CVAE 生成轨迹无碰撞。

原始 `79-D Frames` 布局：

| 索引 | 内容 |
|---|---|
| `0:3` | Base 世界位置，m |
| `3:7` | Base quaternion，xyzw |
| `7:19` | 12 个 Leg joint position，rad |
| `19:25` | 6 个 Piper joint position，rad |
| `25:37` | 四个 wheel center 的 Base 坐标 |
| `37:40` | Piper TCP/EE 在 Base 坐标的位置 |
| `40:43` | Base 机体系线速度，m/s |
| `43:46` | Base 机体系角速度，rad/s |
| `46:64` | 12 Leg + 6 Arm joint velocity，rad/s |
| `64:76` | 四个 wheel center 速度 |
| `76:79` | EE/TCP 速度 |

轮角速度由 NPZ sidecar 的有效字段提供，不是原始 Frame 中 `64:76` 的 wheel-center 线速度。EE actual/reference 与 task EE goal 是不同概念：goal 是任务条件；reference 是期望路径；actual 是按运动学重建的结果。M/LM Goal 为固定世界目标，目标不是“当前 EE 位置”。

### 2.1 M 类目标分布

1,000 个 M 目标使用世界系 XYZ、单位 m，均值 `(0.6645, -0.0295, 0.4497)`，标准差 `(0.1615, 0.1234, 0.2571)`；范围为 X `0.2529–0.9474`、Y `-0.4455–0.3139`、Z `0.0408–0.8985`。全部 M 的目标在每条轨迹内最大漂移为 0。EE 终点误差 P50/P90/P95/最大分别为 `0 / 0.1599 / 0.1780 / 1.3615 mm`；最大值来自 `m_001318`，仍低于库使用的 3 mm endpoint goal 审计阈值。

新旧 M 目标分布不同：旧 280 条目标高度均值约 `0.6104 m`，新增 720 条约 `0.3872 m`。因此训练和汇报应保留来源分组，不能只看全库 aggregate。

## 3. CVAE 网络与张量协议

模型版本：`m1_cvae_motion_prior_v1`。它与 SuperMimic 的 Actor、Critic、World Model 完全不同，也没有覆盖历史 `MotionPriorMLP`。历史 MLP 是直接监督回归 baseline；新的 CVAE 有 Posterior、条件 Prior 和 Decoder。

```text
State history:  [B, 4, 53]
Task goal:      [B, 8]
Condition:      [B, 220]      # flatten(4×53) + Goal8
Future output:  [B, 10, 31]   # 10×20 ms = 0.2 s
```

### 3.1 State 53

每帧特征顺序与单位：

```text
Root height                          1  m
Heading-relative Root rotation 6D   6  unitless
Body linear + angular velocity      6  m/s, rad/s
12 leg + 6 Piper positions          18  rad
12 leg + 6 Piper velocities         18  rad/s
4 wheel angular rates                4  rad/s
---------------------------------------
Total                               53
```

4D 历史来自**当前及过去观测**。状态中的实际 Base 速度是 state observation，不是未来速度命令。

### 3.2 Goal 8

```text
[vx, vy, wz, ee_dx_heading, ee_dy_heading, ee_dz_heading,
 velocity_valid, ee_valid]
```

- L：Velocity task，使用 `[vx,vy,wz]` 和 `velocity_valid=1`，EE 坐标清零、`ee_valid=0`。
- M/LM：Reach task，使用当前 Base **yaw-only heading frame** 中的 EE goal delta，`velocity_valid=0, ee_valid=1`。
- M/LM Goal 固定于世界坐标；每一时刻相对当前 Base heading 的 delta 会随 Base 位置/朝向变化。
- LM 的未来 Base Velocity Command **没有输入模型**。相同状态和相同 Reach Goal 允许多种 Base/Arm 协调方式。
- 无效条件值置零，Mask 区分无效和有效的零命令。

### 3.3 Future 31

```text
Current-root-heading-relative Base translation  3  m
Current-root-heading-relative rotation 6D       6
12 leg joint positions                          12  rad
6 Piper joint positions                          6  rad
4 wheel angular rates                            4  rad/s
---------------------------------------------------
Total                                            31
```

Rotation 6D 是旋转矩阵前两列按列拼接。输出是短时运动学参考；没有 torque、gripper action、接触状态，也不含目标之后的 Ground Truth。wheel 的四通道次序为 `FAR, FBL, RAR, RBL`。

### 3.4 CVAE 结构

```text
Posterior Encoder: q(z | history, goal, future) → μq, logvarq
Conditional Prior: p(z | history, goal)         → μp, logvarp
Decoder: p(future | history, goal, z)            → [10,31]
```

隐变量维度 32，MLP hidden width 256、3 层，总参数 769,206。训练时从 Posterior 重参数化采样并最小化重建项及 `KL[q(z|history,goal,future) || p(z|history,goal)]`。推理只从 Conditional Prior 取均值或采样；不能使用 Ground-truth Future。推理 API：

```python
prior = load_cvae_checkpoint(path, device="cpu")
reference = prior.predict(state_history, task_goal, mode="prior_mean")
samples = prior.sample(state_history, task_goal, num_samples=4, seed=20261102)
```

输出分别为 `[B,10,31]` 与 `[B,4,10,31]`。`seed` 用于可重复 latent sampling。推理代码与训练 Dataset 解耦；使用保存的 normalization 和 schema 校验。

## 4. 训练划分与实际训练结果

严格准入 Adapter 接受 2,272/2,272 运动：`full_fcl_valid`、`structural_valid`、`kinematic_valid`、label 与适用的 endpoint goal 均有效。它没有通过伪造旧 G2/AMP 状态字段让 Loader 接受数据，也没有修改源 TXT/NPZ。原库没有强制 `motion_phase`；没有因 Carry/Reach 等旧阶段标签筛除当前 M/LM 窗口。

history 4、horizon 10、stride 1 共 `492,125` 个窗口：L `211,148`、M `188,889`、LM `92,088`。运动级去重/分组后按约 80/10/10 划分，同一 motion/hash group 不跨 split：

| split | motion 数 L/M/LM | 窗口数 |
|---|---|---:|
| Train | 799 / 800 / 219 | 394,791 |
| Val | 100 / 101 / 27 | 48,395 |
| Test | 100 / 99 / 27 | 48,939 |

正式训练使用 CPU、10,000 steps、batch 128、AdamW、LR `3e-4`、beta target `0.01`、20% KL warmup，约 95.32 秒 optimizer time。最佳验证 checkpoint 为 step 10,000，validation weighted normalized reconstruction `0.136923`。

同一冻结 in-distribution Test 的 weighted normalized MSE：

| 方法 | Test MSE |
|---|---:|
| CVAE prior mean | **0.248128** |
| CVAE prior random sample | 0.250130 |
| deterministic MLP，同 Goal8 | 0.258216 |
| Historical `MotionPriorMLP`，重新训练于 Goal14 | 0.261233 |
| Constant velocity | 0.457189 |
| Last-frame hold | 1.874914 |
| Posterior reconstruction（oracle，看到 future，不可部署） | 0.176085 |

prior mean 的分项指标：Root translation RMSE `0.466 mm`、Root rotation 平均角误差 `0.0762°`、Leg q RMSE `0.00751 rad`、Piper q RMSE `0.01164 rad`、有效 wheel rate RMSE `0.01020 rad/s`。这些是与运动学数据的短时 reference 预测误差，不是闭环控制/任务成功率。

独立 held-out Goal 诊断的 correct/shuffled normalized reference MSE 为 `0.003461/0.003507`，预测 0.2 s EE Goal progress 为 `25.38/22.86 mm`，真实参考 progress `29.27 mm`。Goal 使用有效但幅度小；四个 latent sample 的 Piper 关节均值分离约 `0.0114 rad`、FK EE 均值分离 `4.25 mm`，尚不能证明具有清晰语义的多模态动作。预测有少数 joint limit violation，joint acceleration p95 `22.0 rad/s²`、maximum `723.7 rad/s²`；预测没有通过 FCL 筛查。

## 5. AMP 仓库中的具体文件路径

这些路径相对于 AMP 根目录 `/home/yifeiy/AMP-LOCO-M1`。Motion Bank 源数据及正式 Markdown 报告已推送在 AMP GitHub；训练运行目录 `/logs` 被 AMP `.gitignore` 排除，**二进制 checkpoint 与运行 JSON 不随本次 SuperMimic 文档提交上传**。

| 内容 | AMP 路径 | 备注 |
|---|---|---|
| 最终原始库 | `dataset/m1_kinematic_motion_bank_v2_5_final/` | TXT/NPZ、Manifest、Full-FCL/结构/运动学审计 |
| M 类 EE 汇总 | `dataset/m1_kinematic_motion_bank_v2_5_final/reports/m_source_ee_distribution_summary.json` | M 来源分组目标分布 |
| M 类逐条/路径 CSV | `.../reports/m_ee_distribution_by_source.csv`、`.../reports/m_ee_trajectory_distribution_by_source.csv` | 目标、终点误差、归一化路径采样 |
| M 类分布图 | `docs/assets/m1_v2_5_final_m_ee_distribution.svg` | 报告图 |
| CVAE 模型代码 | `motion_prior/cvae_model.py`、`motion_prior/cvae_features.py`、`motion_prior/cvae_inference.py` | 结构、schema、CPU-safe inference |
| 训练数据 Adapter | `motion_prior/strict_fcl_adapter.py`、`motion_prior/cvae_data.py` | 只读严格审查 Adapter 与窗口数据 |
| 训练入口/评价 | `motion_prior/train_cvae.py`、`motion_prior/evaluate_cvae.py` | 正式训练与评价源码 |
| 主实验目录 | `logs/motion_prior/cvae_v1_strict_fcl_2272/` | 当前主 CVAE run |
| 最佳 checkpoint | `logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt` | SHA-256 `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`；本机文件，Git ignored |
| 主实验 schema/划分/归一化 | 主实验目录下 `config.json`、`dataset_audit.json`、`dataset_distribution.json`、`split_manifest.json`、`normalization.json` | split 和 train-only normalization |
| 主实验评估 | 主实验目录下 `evaluation_metrics.json`、`goal_sensitivity.json`、`per_category_metrics.json`、`latent_diversity.json`、`trajectory_quality.json`、`metrics.json` | Test、Goal、类别、latent、质量和学习曲线数据 |
| 推理/轨迹示例 | 主实验目录下 `reference_only_dry_run.json`、`inference_examples/`、`trajectory_examples/` | 三类 prior-only dry run 与 GT/Mean/Samples/FK 对比 |
| held-out Goal 独立实验 | `logs/motion_prior/cvae_v1_strict_fcl_2272_heldout_goal_diagnostic/` | 独立 held-out goal split、checkpoint、评价 JSON |
| 汇总训练报告 | `docs/M1_CVAE_MOTION_PRIOR_V1_TRAINING.md` | 结果、限制、复现命令与 checkpoint 路径 |
| 接入审计报告 | `docs/M1_SUPERMIMIC_CVAE_INTEGRATION_AUDIT.md` | 坐标/action gap、奖励建议和实施门槛 |

若在 SuperMimic 主机上需要加载 checkpoint，必须先将 AMP 主机的文件安全复制到该主机上的明确路径，再校验上述 SHA-256；不能只凭文档链接假定模型权重存在于本仓库。Source bank 当前也没有自动转换成 SuperMimic Motion Library 格式。

## 6. SuperMimic 当前 M1-Piper 机器人与网络契约

### 6.1 RobotSpec、状态、Actor observation

`src/supermimic/robots/spec.py` 定义的 M1 为 22 个标量关节：

```text
FAR_ABAD, FAR_HIP, FAR_KNEE, FAR_FOOT,
FBL_ABAD, FBL_HIP, FBL_KNEE, FBL_FOOT,
RAR_ABAD, RAR_HIP, RAR_KNEE, RAR_FOOT,
RBL_ABAD, RBL_HIP, RBL_KNEE, RBL_FOOT,
piper_x_joint1 ... piper_x_joint6
```

Root body `BASE_LINK`；默认 Base height `0.3892858 m`；control dt `0.02 s`。16 个腿动作中四个 `*_FOOT_JOINT` 是 wheel velocity，其余腿关节和 Piper 六轴为 position target。位置 action scale `0.25 rad`，wheel scale `5 rad/s`；RobotSpec 设置腿非轮 Kp/Kd `250/9`、Piper `32/1`、wheel Kd `0.5`。这些是 SuperMimic action/control convention，不是 CVAE 输出缩放。

SuperMimic 的状态为：

```text
[root_pos_world(3), root_quat_wxyz(4), joint_pos(22),
 root_lin_vel_world(3), root_ang_vel_body(3), joint_vel(22)]
state_dim = 13 + 2×22 = 57
delta_dim = 12 + 2×22 = 56
```

SuperMimic state quaternion 内部为 `wxyz`；AMP motion file/feature API 的原始 quaternion 为 `xyzw`。`StateIO` 把 root position 转成 env-origin-free 局部位置。`StateSpec.canonical()` 输出 57-D heading-invariant dynamics feature；4 个 wheel joint position 在网络 state 中保留，但 wheel angle 在 canonical 特征中 wrap。World Model 训练 8-step autoregressive physical transition；policy imagination 默认 horizon 32。

Actor 单帧 observation `72-D = 3×22+6`：joint-position error (wheel position mask)、22 joint velocity、body angular velocity3、projected gravity3、previous action22。默认 10 帧历史展平为 `720-D`。当前 actor 不直接使用 Base linear velocity；CVAE 53-D State 要用它，因此部署/Actor-facing adapter 必须明确线速度的状态估计来源，不能暗中使用仿真 privileged state。

### 6.2 Action and Goal

- Policy 输出 `22-D` normalized M1 Mixed Action。执行器把腿/Piper 位置目标变换到默认角度附近，再按关节限位 clamp；四 wheel 通道转成 velocity target。具体对应关系必须以关节名称为键，不可以把 CVAE `[10,31]` 直接 reshape 成22维动作。
- `motion_tracking` 默认使用六条已批准 reference：`a_wheel_000000`, `a_wheel_000002`, `b_arm_000006`, `c_assisted_000029`, `c_assisted_000039`, `c_assisted_000149`，清单为 `configs/m1_piper_first6.csv`。单帧 motion goal 46-D；10帧加9-D Anchor，Goal `469-D`；Actor history `720-D`。该旧 tracking 数据和 final 2,272 bank/CVAE 不是同一协议。
- 当前 M1 converter 的默认旧源库是 `/home/yifeiy/AMP-LOCO-M1/dataset/m1_motion_bank_v1`，转换输出到 SuperMimic 工作区 `data/npz/m1_piper/`；它不会自动读取 `dataset/m1_kinematic_motion_bank_v2_5_final/`。要使用新库必须新增经过审查的转换/任务适配，不能只修改目录名。
- `world_point_reach` 的 Actor/Critic Goal 为 `3-D`，语义为固定 env-local world target 相对当前 base 的 body-frame delta；当前每步用完整 base quaternion 变换。M1 task 白名单目前为 `motion_tracking` 与 `world_point_reach`；M1 不允许选择 Go2 `velocity_command` 或 `chase_ball`。
- `WorldPointCommand` 用 `midpoint(piper_x_link7,piper_x_link8)` 为夹爪中心 TCP proxy；CVAE 标签使用配置的 `piper_x_link7`。在两边统一 TCP 或验证刚体偏移之前，Goal/reward/FK 可能不一致。

### 6.3 World Point Reach reward and reset

当前原生 world-point reward 包括：

```text
tcp_distance       = -1.5 * d_next
tcp_approach       = 0.25 * tanh((d_prev-d_next)/0.05)
tcp_near           = 0.6 * exp(-(d_next/0.10)^2)
upright            = 0.25 * exp(-||gravity_local_xy||^2/0.16)
base_height_low    = -4.0 * relu(fall_height+0.04-root_z)^2
joint_limit        = -0.3 * squared soft-limit violations
action_rate        = -0.005 * ||a-a_prev||^2
tcp_speed          = -0.02 * relu(speed-0.8)^2
joint_velocity     = -1e-6 * ||qd||^2
joint_acceleration = -1e-7 * ||qdd||^2
leg_action_magnitude = -0.002 * ||leg action||^2
```

Reward core 用同一可微 Piper FK 计算真实和 imagination terms，适配边界各乘一次 `dt`。Fall height 配置为 `0.20 m`，Root tilt 终止阈值按配置角度；Reach success distance 默认 `0.08 m`。真实 command 记录连续 `0.5 s` 在 success radius 内的 episode success/curriculum 状态；成功当前不在 `WORLD_POINT_REWARD_NAMES` 中作为单独的一次性 RewardTerm，也不作为一个 task termination。M1 near/far 目标课程可在 `EnvCfg` 修改，审计时的本地配置 Stage 0 以近目标为主。有关课程参数以实际启动的 config/checkpoint metadata 为准。

该任务现在没有 object pose、object-target error、grasp/release state 或 gripper actuator。接入真正 manipulation 时需先新增物体/接触/夹爪状态和可验证 success predicate，不能将 point Reach 奖励改名为抓取或放置奖励。

## 7. CVAE → SuperMimic 的适配协议

CVAE 可以用于冻结的短时 reference shaping 或单独 reference-conditioning 实验；不是 torque/controller，不取代 Actor、Critic、World Model。优先从 M-only、prior mean、Reference-only 开始。

| 必须适配的项 | 正确处理 |
|---|---|
| State history | 按 4 帧和 53-D 字段名从 StateIO/state-history 重建；特别验证 `wxyz↔xyzw`、root linear velocity 的 frame/unit、joint order、wheel angular velocity 来源 |
| Goal | 从 env-local fixed target 生成当前 Base yaw-only heading delta；CVAE Reach Mask=`[0,1]`（velocity-valid,ee-valid）；与 SuperMimic 完整 body-frame 3-D Goal 区分 |
| TCP | 将 `piper_x_link7` 与两指 frame midpoint 的 FK 差校到容差内，统一 target、reward、plot；差值显著则显式补固定工具变换 |
| Future Root | 31-D root xyz/rot6D 相对当前 root heading frame；转回 env-local/world 后和 WM predicted state 对齐时间索引 |
| q/action order | CVAE 18 position 为12非轮腿+6臂；SuperMimic action有交错的16腿槽位+6臂，四轮速度另行填入；仅在明确 action-residual 实验时按 scale/q0反变换 |
| Future horizon | Prior 只提供未来10步/0.2秒；SuperMimic imagination 是32步/0.64秒。先只在10步应用 shaping，不复制末帧填满其余时间 |
| Gradient/device | 当前独立 loader 是 `inference_mode()` CPU API。固定 detached reference 仍可对 imagined state 的误差回传 Actor；若要重规划并对输入求导，需开发 frozen-weight/device-native differentiable call |
| World Model | 只用实际执行动作和仿真物理 state transition 训练 WM。CVAE future 不可以当作真实 next-state label |
| Goal multimodality | LM 未包含未来底盘速度意图。先 M，再独立 LM ablation；固定 latent/common random numbers，不能宣称现有样本已学出语义 mode |

### 建议接入奖励

1. **保留 task reward**：目标距离变化/近目标势、upright、root height、关节限位、action-rate、速度/加速度项保持为 task objective。成功 radius 内稳定保持0.5秒才登记成功，success 与跌倒/timeout 分开统计；避免重复叠加两份距离进度奖励。
2. **CVAE reference tracking**：在 imagination 的前10步加小权重、逐渐退火的分项误差 `Root position + Root rotation + leg q + Piper q + wheel rate`。首版主要跟踪 Piper 和整体姿态；M 对 Base XY/yaw 保持低漂移，wheel rate 偏向静止。Reference 仅为 shaping，Actor 可以为平衡偏离它。真实原生 return 与 imagination shaped return 单独记录。
3. **执行安全**：继续显式看 fall/tilt、soft joint limit、动作饱和、joint qd/qdd、TCP 速度；预测 trajectory 的 FCL/碰撞先作为离线诊断，不表示 source bank 的 Full-FCL PASS 可传递给预测结果。
4. **LM**：不要施加 M 的“底盘不动”惩罚；用不同 Reach Goal/latent 检验 Base-Arm 合作与 goal progress。缺乏未来运动意图造成的 ambiguity 要在结论中保留。
5. **Object manipulation future**：只有加入对象与夹爪后，再增加 object-goal potential、contact/grasp stability、lift/transport、release/support、物体低速 hold 和 hard success predicate。

manipulation SuperMimic repo 提供有效 reset category、中途 reset、解析 task reward、真实 hard predicate 与 imagination 可微 success surrogate 的设计参考；Dex-SuperMimic 展示 reference/residual 控制用于不同灵巧手任务的思路。两者都不是 M1 的 action/奖励数值模板：

- [emoPointer/supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation)
- [Manipulation task contracts](https://raw.githubusercontent.com/emoPointer/supermimic_manipulation/main/docs/tasks.md)
- [SuperMimic manipulation adaptation notes](https://raw.githubusercontent.com/emoPointer/supermimic_manipulation/main/docs/supermimic.md)
- [CumulusAlpha/Dex-SuperMimic](https://github.com/CumulusAlpha/Dex-SuperMimic)

## 8. 当前已知训练/评价状态与限制

- SuperMimic M1 motion tracking 仅用六条已有 approved clips，不是 final 2,272 条库；历史 2048-env、iteration 1050 短训 checkpoint 可以完成 clip playback，但 6 条的严格 tracking success 为 `0%`。这验证了训练/评估链路，不是策略学会动作的证据。
- M1 World Point 的历史 2026-10-27 试跑记录了 episode 平均约 `0.22 s` 因 Base height 跌倒、128-episode success `0%`；后续 2026-11-02 文档中的 16-env smoke/16-episode Stage-0 evaluation success 仍为 `0%`，末端最终误差大于初始误差，Base 几乎没移动。该结果来自所记录的旧 run；当前源文件有未提交改动，必须用明确 commit/config 做新的冻结评估才能代表最新代码。
- CVAE 在冻结 test 上优于 hold、constant-velocity 和 deterministic MLP 的总 normalized MSE，但只预测运动学 reference，不输出动作，不保证真实任务 progress。
- Goal-conditioned effect 和 latent 多样性目前有限；LM 小于 M，且只给 Reach Goal、没有未来 base command。
- Piper TCP 定义两边不同；yaw-heading 与 full body-frame 目标也不同；SuperMimic policy 本身不观测 root linear velocity。
- M1 当前只做末端 Reach；未集成操纵物体、抓取/释放、真实夹爪控制、SuperMimic Actor 或 torque policy。

审计时文档中记录的 SuperMimic 侧历史产物路径：

```text
logs/m1_piper_motion_C_2048_wm300_15m/ckpt_latest.pt
logs/m1_piper_motion_C_2048_wm300_15m/run_summary.json
logs/m1_piper_motion_C_2048_wm300_15m/eval_C.json
logs/phase2_smoke/phase2_m1_near_smoke_16env/eval_near_16ep.json
```

这些是旧短训/评估记录，不是 CVAE checkpoint，也不是新的 M1 成功策略。不同记录对应不同代码/配置版本；Resume 或对比前应检查 checkpoint 的 config、机器人模型 hash、motion set、task 与 state/goal dimensions。

## 9. 推送到仓库的内容与不包含内容

本文件作为 SuperMimic 仓库内的协议/结果索引提交。AMP 2,272 motion source bank 位于独立 AMP 仓库，主库数据在 AMP Git 中；CVAE checkpoint 和 `logs/motion_prior/...` 评价产物是 AMP 工作站上的本地运行文件，当前被 AMP `/logs` ignore，不包含在这次 SuperMimic 文档提交中。任何实际接入前应先传输 checkpoint 并验证 SHA-256，记录目标机器路径和 model/schema hash。

本文件记录审计时的仓库状态，不修改动作、Actor、Critic、World Model、reward 或训练进程；不表示训练结果、代码变更或 SuperMimic 的本地未提交内容已经合并。
