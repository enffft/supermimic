# M1 CVAE 输入兼容审计

## 已证实的数值问题

原始 AMP checkpoint 的条件输入是 4×53 State + 8D Goal。预训练归一化中，Piper q6、qd6 的 std 为 `1e-6`，训练集这两个物理量几乎恒零。legacy 输入 `q6=+1e-4 rad` 会产生 `z=100`；`q6=+0.01 rad` 会产生 `z=10000`。qd6 在同量级也会放大。模型于是可能因极小状态噪声进入训练分布外区域。

另外，最新 heading-relative rotation6d 的一个分量在数学上严格为 0，但经过 float32 四元数变换可能出现极小舍入残差。该分量 std 同样是 `1e-6`。State53 转换现在按几何定义显式置零这一项（flattened condition index 161）；NumPy/AMP 推理对拍证实此约定与源特征一致。没有对所有小 std 特征做通用 clamp。

## 两个版本化运行策略

- `legacy_v1` 完全复现旧的特征和归一化路径，作为对照。
- `train_constant_compat_v1` 只在 CVAE 条件张量内，将 4 帧 q6 和 4 帧 qd6 输入替换为 checkpoint 训练均值。State57、Actor 输入、WM 状态和物理测量不变，原始 `x_mean/x_std`、网络权重及 checkpoint 都不改。该选项默认在 M1 joint 配置中显式选择，可配置关闭。

这是冻结旧模型的分布兼容处理。非零 q6/qd6 信息不会进入 CVAE 条件，因此模型没有学会依赖这些轴规划；被投影输入应记录为 OOD/兼容动作。超出验证范围的在线部署没有安全保证。

## CPU 对照结果

使用原始 checkpoint（SHA256 `1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d`）及 AMP 的 `L_forward.npz`：

- AMP 推理与保存的参考最大绝对差：`0`。
- SuperMimic `legacy_v1` 与 AMP 推理最大绝对差：`0`。
- Carry/q6=0 样本下兼容模式与源输出最大绝对差：`0`。
- `vx=-0.3 m/s` 首帧四轮参考约 `[-3.0773, -3.0739, -3.0741, -3.0769] rad/s`；所有输出有限。
- q6 `+0.0001 rad` 时 legacy 最大输入 z-score `100`；compat 投影后该轴 z-score 为 0，预测恢复到 q6=0 的输出。扰动扫到 `±0.01 rad` 时 legacy z-score 最大为 `10000`。qd6 扫描具有同样的机制。Compat 的完整条件仍有最大 z-score 约 `20.72`，来源不是被投影的 q6/qd6，因此只说明修复了已知 tiny-std 放大点。
- 同一 Carry history 的 Goal vx 扫描 `[-0.3,-0.2,0,0.2,0.3]`，`|root_dx(0.2s)-root_dx(vx=0)|` 最大仅 `1.527 mm`，legacy/compat 相同。报告状态为 `PRETRAINED_GOAL_CONDITIONING_LIMITATION`。这说明该 checkpoint 在此历史上速度 Goal 响应很弱，输入兼容策略没有解决它。
- Carry 与旧 Manipulation history 的姿态消融只比较了模型输出；它隔离了输入姿态差异，但不是实机/仿真动力学对比。姿态统一和 tiny-std 投影的影响分别记录在 JSON。

## 日志字段与复现

Prior 可按日志间隔记录最大/P99 输入 z-score、恒定特征 OOD 比例、q6/qd6 原始量、归一化量、投影计数与投影幅度。诊断张量在 GPU 上计算，但不在每个环境步自动同步 CPU；本次结果仅来自 CPU 离线检查。

完整数据和每个扫描输出见 [审计 JSON](artifacts/m1_cvae_input_compat_audit.json)。复现：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONPATH=src \
nice -n 15 /home/yifeiy/miniconda3/envs/supermimic/bin/python \
  scripts/audit_m1_cvae_input_compat.py
```

## 仍待处理

要真正学习 q6/qd6 运动，需要新数据覆盖、重算归一化并重新训练/验证 CVAE；本次明确没有做。静止/弱 Goal 响应也仍是预训练模型目标条件能力限制。GPU 训练、MuJoCo 动态可行性、真实执行器跟踪和部署安全未验证。
