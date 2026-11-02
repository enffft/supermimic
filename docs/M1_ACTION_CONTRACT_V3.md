# M1 Mixed Action Contract v3

## 目的

Contract v2 对位置关节使用 `q_target = q_default + action * scale`。M1 prior joint 过去以 Manipulation 姿态重置，但 AMP 的 L/M/LM Motion Bank 的 Piper 首帧统一为 Carry 姿态。比如 Piper q2 的默认值从 `2.55` 变为 `0.50 rad`，q3 从 `-1.50` 变为 `-0.40 rad`，q4 从 `-0.80` 变为 `0.30 rad`。旧 ±0.25/0.35 rad 偏移根本覆盖不了训练参考。

v3 把 reset pose 与执行目标区间一起版本化。`m1_prior_joint` 默认 `dataset_carry`；M1 旧 world-point / motion 路径仍可用 `legacy_manipulation` 与 v2。G1/Go2 不变。

## 22D 顺序与映射

仍按 `RobotSpec.joint_names` 排列：12 个腿位置目标、4 个交错轮速目标、6 个 Piper 位置目标。所有维度用名字构建位置/轮速 mask，不假设动作向量前若干维连续属于同一类别。

先裁剪实际执行动作 `a` 到 `[-1,1]`。位置关节：

```text
a < 0: q_target = q0 + a * (q0 - q_min)
a ≥ 0: q_target = q0 + a * (q_max - q0)
```

因此 `-1 / 0 / +1` 分别得到区间下界、默认角、上界。随后仍受 MuJoCo soft limits 限制。该分段函数在 0 两侧导数不同；零点 torch.where 采用正支导数，输出连续且有限。Wheel 维度继续为 `wheel_target_rad_s = a * 5.0`，其角速度语义完全独立。

v3 当前 Piper 工作区（rad）：

| Joint | v2 旧可命令区间 | v3 区间 |
|---|---:|---:|
| q1 | [-0.25, 0.25] | [-0.53457, 0.61986] |
| q2 | [2.20, 2.90] | [0.15700, 2.84280] |
| q3 | [-1.75, -1.25] | [-1.88899, -0.30207] |
| q4 | [-1.15, -0.45] | [-0.41415, 0.60037] |
| q5 | [-0.25, 0.25] | [-0.36002, 0.29504] |
| q6 | [-0.25, 0.25] | [-0.05, 0.05] |

腿部 v3 每关节上下界见审计 JSON。它们由 AMP Cache 的位置参考分位数外加 `0.05 rad` 起步，再与 URDF/SDK 物理区间和 mjlab soft limits 取交集。q6 因运动数据恒定，使用窄而保守的工作区；不能将此范围理解为已验证 q6 运动。

## 三种边界来源

Piper SDK 软件限制来自 `M1_RL_GRASP/src/rl_sar/piper/piper_backend.py`。URDF 物理限制来自 `AMP-LOCO-M1/resources/robots/zg_m1_piper_x/zg_m1_piper_x.urdf`。Setpoint envelope 是 Actor 能直接命令的更窄工作区域。代码采用这些范围的安全交集并加上 mjlab 的 soft-limit 裁剪；没有放宽任何硬限位。SDK 表示部署软件设置，不是对实时固件或硬件安全回路的验证。

## 覆盖与数据残差

系统抽样 L/M/LM 未来帧中，单个位置坐标的 v2/v3 覆盖率分别为：

| Task | v2 | v3 |
|---|---:|---:|
| L | 77.54% | 99.20% |
| M | 78.47% | 98.66% |
| LM | 74.65% | 98.11% |

同时要求一帧 18 个位置关节全部处于区间内，v2/v3 分别为：L `0.00% / 88.16%`、M `0.0037% / 89.78%`、LM `0.00% / 81.36%`。抽样没有发现 URDF 硬限位、SDK 限位或 wheel ±5 rad/s 越界；soft-limit 越界仍存在（M FBL knee 最大约 4.66%，LM Piper q2 最大约 4.82%）。这部分保留为参考数据残差，不扩大执行范围来追求 100%。

审计 JSON 保存每个关节的 P1/P5/P50/P95/P99、min/max、硬限位越界、soft-limit 越界、v2/v3 超界和 v3 接近饱和比例。

## Checkpoint 兼容与验证边界

新的 M1 joint checkpoint 标记 `m1_mixed_action_v3`、姿态 profile、Action bounds、模型/URDF 指纹和 CVAE 输入策略。v2 联合 checkpoint 恢复会报错；不能将旧 Actor/WM 默认为 v3 物理动作语义。AMP CVAE 源权重单独加载且未修改。

CPU 测试检查了 22D 顺序、零/端点映射、wheel 速度映射、梯度和限位。静态 MuJoCo `mj_forward` 检查 Carry 与 Legacy Reset 均为有限状态、初始接触 0，Root 高度未变。尚未做动态平衡、Piper 跟踪、力矩/速度和真实 mjlab action setpoint 对拍；当前不可据此声称部署安全或参考全可达。
