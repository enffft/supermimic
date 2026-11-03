# M1 A+B 阶段实施记录（2026-11-02）

## 范围与隔离

本次只处理 M1 Reset/Action 数据契约和冻结 CVAE 的输入数值鲁棒性。开发位于独立 worktree `/home/yifeiy/ws_supermimic_m1_ab_fix`、分支 `fix/m1-action-normalization-contract-v3`，基线 `c01ba1e916a0a31c76ed03bde50e4183baad1503`。原工作区 `/home/yifeiy/ws_supermimic` 未切换分支、未 stash/reset；其中原有 Go2 修改保持原样。

当前 G1 训练 PID `1937328` 在本次开始前后均存活，命令仍是 `g1_pure_proprio_scratch_1024_14h_20261102`。结束检查时进程运行约 28 分钟；GPU 显存约 5919/12282 MiB、利用率 97%。这些是只读观察值。本次没有终止或修改该进程，没有改 G1 代码、配置、checkpoint 或日志。

所有 Python 测试和 CVAE 推理使用 `supermimic` Conda 环境、`CUDA_VISIBLE_DEVICES=''`、至多 2 个线程并以 `nice -n 15` 运行；`supermimic.__file__` 指向本 worktree。没有初始化 CUDA Context，没有执行 GPU 仿真或训练，也没有重生成 Motion Bank。CVAE 源 checkpoint SHA256 为 `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`，文件未改。

## 阶段 A：Pose Profile 与 Action Contract v3

- 新增 `legacy_manipulation` 与 `dataset_carry` 姿态 Profile。M1 `m1_prior_joint` 默认使用 Carry；L、M、LM Motion Bank 的首帧 Piper 均为 `[0, 0.50, -0.40, 0.30, 0, 0]`（统计样本分别 999、1000、273 条轨迹）。所以 L/M 共享 Carry Reset 与参考一致。旧 M1 任务默认仍用 Manipulation/v2；腿关节默认、Root 高度 `0.3892858 m`、轮初速度均未改变。
- M1 新控制映射版本为 `m1_mixed_action_v3`。仅位置控制关节使用逐关节非对称映射：动作 `-1/0/+1` 映射到下界/默认位置/上界；轮子仍是独立的 `a*5 rad/s` 速度命令。22 维排列仍从关节名建立，四轮交错位置不变。Contract v2 仍可显式构造，G1/Go2 路径没有改其 Action 语义。
- 新位置工作区以 AMP Bank 系统抽样窗口 P1/P99 加 `0.05 rad` 裕量形成，并裁剪到 URDF 与 Piper SDK 软件范围的交集及 MuJoCo 0.9 soft-limit 范围。q6 在 Bank 中无有效运动样本，故设为保守 `[-0.05, 0.05] rad`，这是控制边界，不表示 Prior 学会了 q6 动作。每关节完整范围见 [Action Contract v3 审计 JSON](artifacts/m1_action_contract_v3_audit.json)。
- SDK 文件 SHA256：`9ab737b7c911fe0b23c5c95b7603d28354ace9616f62c76e4cdd70f7e45cc307`；URDF SHA256：`25bebd86d7595da81f0718f8f0185c8b5f0fd70efca0c8d05b24a3d14b624c1f`。SDK 是软件限制，不等同实时固件保护；Piper q4 的交集由 URDF `±1.57 rad` 收紧，q5/q6 也分别受 SDK/URDF 交集限制。腿部只有 URDF 限位来源。
- 对已有 CVAE Window Cache 以 stride 35 抽样：L 6033 窗口、M 5397、LM 2631；总计分别检查 60330、53970、26310 个未来帧。旧 Manipulation/v2 的位置坐标覆盖率为 L 77.54%、M 78.47%、LM 74.65%；同一批数据的 v3 为 99.20%、98.66%、98.11%。逐帧要求 18 个位置关节全部在边界内时，旧 v2 覆盖率约 0%、0.0037%、0%；新 v3 为 88.16%、89.78%、81.36%。这里“坐标覆盖率”是单关节坐标平均，“整帧覆盖率”是 18 个位置关节同时满足；精确逐关节数据都在 JSON 中。
- 抽样参考没有超过 URDF 硬限位或 Piper SDK 范围，轮速没有超过 `±5 rad/s`。但存在超出 MuJoCo soft limits / 所选安全工作区的参考：最差 M 轨迹中 FBL Knee 的 soft-limit 越界约 4.66%；LM 中 Piper q2 约 4.82%。不扩大物理限位覆盖这些样本，已作为数据/参考边界残差保留。整帧 v3 覆盖未达到 100%，所以不宣称参考全部可执行。
- CPU MuJoCo 模型编译检查得到 `nq=29,nv=28,nbody=44`。Legacy/v2 Manipulation 与 Carry/v3 均在 Root 高度 `0.3892858 m`、四轮零速下完成 `mj_forward`；qpos 有限、初始接触数为 0。TCP `piper_x_link7` 的静态位置从 Manipulation 的 `[0.828, 0.000, 0.488] m` 变为 Carry 的 `[0.433, 0.000, 0.725] m`。M 的目标仍按 Reset 后 TCP 采样局部偏移，并在 episode 内固定世界目标。静态 FK/碰撞检查不代表动态稳定或执行器可达性验证。
- M1 prior joint checkpoint 升为 v3，记录 Action Contract、Pose Profile、位置边界和 CVAE 输入兼容策略；M1 joint v2 checkpoint 恢复会显式拒绝。旧 M1 的 Actor/WM 对应物理动作语义不同，不应当作 v3 完整兼容模型 Resume。原始 CVAE 权重仍可单独严格加载。

## 阶段 B：CVAE 输入兼容策略

CVAE 保留原始 `x_std`，未改权重或归一化 buffer。增加两种运行时策略：

- `legacy_v1`：逐位复现过去的 State53/Goal8 归一化输入。
- `train_constant_compat_v1`：只将训练集恒定为零的 Piper q6/qd6 条件输入（4 帧各一维）投影回 checkpoint `x_mean`。物理 State57、Actor 观测、WM 状态、传感器值都不改。该模式是旧模型 OOD 兼容，不会让模型获得 q6/qd6 条件运动能力。

Heading-relative Rotation6D 的最新帧 `R_headingᵀR` 第 1 行第 0 列由同一帧 yaw 定义解析上严格等于 0。State feature adapter 现在显式写成精确 0，避免 float32 的约 `1e-7` 舍入除以 `1e-6` std。只处理这个解析恒定量，没有把所有低方差特征一概清零。

CPU 对拍结果和扰动扫描详见 [CVAE 输入兼容审计 JSON](artifacts/m1_cvae_input_compat_audit.json) 与 [CVAE 报告](M1_CVAE_INPUT_COMPATIBILITY_AUDIT.md)：AMP 原始 `L_forward.npz` 与 AMP/SuperMimic legacy 推理最大差值均为 0；`vx=-0.3 m/s` 的首帧轮速复现为约 `[-3.0773,-3.0739,-3.0741,-3.0769] rad/s`。q6 `+0.0001 rad` 在 legacy 下造成 100 的 z-score；兼容模式将其输入偏差投影掉，输出与 q6=0 相同。更大扫描中 legacy 最高 z-score 到 10000；兼容模式对应 q6/qd6 输出与基准参考相同。兼容模式总体输入最大 z-score 仍约 20.72，主要来自其他输入特征，不能据此声称全输入都在分布内。

固定同一 Carry 历史，把速度 Goal 从 `-0.3` 扫至 `+0.3 m/s`，预测 0.2 秒根部 x 位移相对零 Goal 的最大变化只有 `0.001527 m`，两种输入策略数值一致。因此保留结论 `PRETRAINED_GOAL_CONDITIONING_LIMITATION`：当前冻结 CVAE 在这组历史上的速度 Goal 响应很弱。兼容层没有修复此训练数据/模型能力限制，也未对 CVAE 输出人工加速度或轮速。

## 验证、未完成项与复现

- 新增 A/B 测试：`15 passed`。
- 原有 `test_m1_prior_joint.py` 与 `test_m1_reward_audit.py`：`16 passed, 2 skipped`。
- `compileall`、新测试/脚本 Ruff、`git diff --check` 通过。修改前后对本次 touched Python 文件进行 Ruff 诊断差分后，没有新增诊断；仓库既有 lint 诊断仍存在。
- 未完成：真实 mjlab 环境的 Carry Reset 动态站立、Position Actuator 的实际参考跟踪、Actor/WM v3 Action 对拍、跌倒/负载下的关节/力矩/速度验证。这些需要 G1 空闲后的短 CPU MuJoCo 环境验证，再决定是否安排 GPU 训练。
- CPU 审计复现：在 worktree 设置 `PYTHONPATH=src`、`CUDA_VISIBLE_DEVICES=''`、线程环境变量为 2，并使用 `supermimic` Python 执行 `scripts/audit_m1_action_contract_v3.py --stride 35` 和 `scripts/audit_m1_cvae_input_compat.py`。原始 Motion Bank、AMP checkpoint、G1 进程均未更改。
