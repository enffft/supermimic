# 阶段二：SuperMimic / PPO 共用 rl_sar 部署链路

日期：2026-11-04。本文主体保留最初使用旧 SuperMimic 的链路验收记录。当天晚间已将正式从头训练 2.5 小时的 PPO 和 SuperMimic 导出并重新检查，两者当前目录均为新模型。最新结果见 [正式策略评估报告](FINAL_POLICY_EVALUATION_20261104.md)；实际部署命令见 `/home/yifeiy/rl_sar/docs/GO2_PPO_SUPERMIMIC_DEPLOY_20261104.md`。本次没有连接或控制实机。

## 导出结果

现有模型位于 `/home/yifeiy/rl_sar/policy/go2/supermimic_velocity/`：

- `policy.pt`：rl_sar 可以加载的 TorchScript，单输入 [N,225]、输出 [N,12]。
- `config.yaml`：部署观测、历史、映射、PD、动作幅度。
- `manifest.json`：源 checkpoint/导出模型 SHA256、关节与输入协议、一致性结果。
- `policy_inference.pt`：使用现有 SuperMimic 推理 schema，供已有 play/evaluate 脚本复用；**只有推理权重，不可用于恢复完整训练**。

源为 `logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt`，SHA256=`0b3d8e4bb26173d13a6ce0b99d5eecf43005927a8ab2e01947c4d0311009487a`。

两种算法共用 `supermimic.baselines.go2_export`。PPO 从上游 checkpoint 提取确定性均值，严格载入本仓库的同构 Actor，再走同一导出路径；测试已验证它与 RSL-RL 原生 JIT 输出一致。短测 PPO 导出仅在 `logs/go2_ppo/smoke_3072_20261104/export`，不用于实机。

## 输入输出协议

rl_sar 的原生 ObservationBuffer 每帧按 `[q_rel12,qd12,omega_body3,gravity_body3,previous_clean_action12,command3]` 产生 45 维。使用 `observations_history=[4,3,2,1,0]`、priority=time，得到从旧到新的 225 维。

导出模块将五帧 proprio 转为训练时的 term-major 210 维，仅使用最新命令 3 维，然后在网络内部执行一次固定缩放。YAML 的 qd、omega、commands scale 均为 1，避免二次缩放。旧命令不进入策略目标，命令切换即时生效。第一帧实测 proprio 重复填满历史，与 mjlab reset 的历史初始化对齐。

训练/模型输出顺序是 FL/FR/RL/RR；SDK 是 FR/FL/RR/RL。`joint_mapping=[3,4,5,0,1,2,9,10,11,6,7,8]` 同时用于读状态和写命令。quaternion=wxyz，omega/body 命令均为 body frame。

动作是 [-1,1] 的绝对归一化 PD 偏置：`q_target=q_default+0.25*a`，不是逐帧累积增量。Kp=40，Kd=1；q_default=`[.1,.8,-1.5, -.1,.8,-1.5, .1,1,-1.5, -.1,1,-1.5]`；力矩限制按训练设置为每腿 `[23.7,23.7,35.55]`。50 Hz 策略、200 Hz 底层。

## rl_sar 的最小修改

- `fsm_robot/fsm_go2.hpp`：通过 `RL_SAR_GO2_POLICY=supermimic_velocity|ppo_velocity` 选择；未设置仍使用 himloco。
- `rl_sdk.hpp`：增加可选的历史首帧填充，共用原来的 ObservationBuffer。
- `rl_real_go2.cpp`、`rl_sim.cpp`、`rl_sim_mujoco.cpp`：在现有 Forward 中调用该历史更新方法。未启用 warmstart 的原有策略维持插入历史的行为。

原有 Torch/ONNX 推理引擎、机器人 SDK、状态读取、命令写入、FSM 都继续复用。`rl_real_go2` 和 `rl_sim` 已在本机重新编译通过；原生 `rl_sim_mujoco` 当前未启用构建，本次用独立的离线检查程序链接同一套 rl_sar SDK/ObservationBuffer/InferenceRuntime。

## 验证结果

1. 随机且逐帧变化的历史和目标：Python 导出与原始 Actor 最大误差 0。
2. 24 帧含不同 quaternion、关节、速度和命令的序列：真实 C++ 的 `ComputeObservation → ObservationBuffer → Torch runtime → ComputeOutput` 与 Python 比较，动作和 PD 目标最大误差 **1.25e-6**。
3. 使用 `make_env` 编译出来的**同一训练模型**保存 MJB，MuJoCo 3.16 原生物理 + rl_sar C++ 推理，5 条命令各运行 10 s，无跌倒。

| 指令 (vx,vy,wz) | 实际 vx | 实际 vy | 实际 wz | 平移 RMSE |
|---|---:|---:|---:|---:|
| (0,0,0) | −0.009 | −0.001 | −0.013 | 0.015 |
| (0.3,0,0) | 0.289 | −0.003 | 0.018 | 0.037 |
| (0.5,0,0) | 0.483 | 0.000 | 0.022 | 0.060 |
| (0,0.3,0) | −0.006 | 0.300 | 0.031 | 0.038 |
| (0,0,0.4) | −0.005 | −0.002 | 0.407 | 0.015 |

完整结果 `logs/go2_comparison/deployment_validation/validation.json`。这验证了接口和名义平地闭环，不是实机性能或扰动鲁棒性的结论。该离线程序不链接机器人 SDK，不发送硬件控制。

实机仍有两项边界：起立阶段先采用 rl_sar 的 base 姿态，再切换策略默认姿态；同时现有 RLControl 下发 q/Kp/Kd，`ComputeOutput` 算出的 clipped tau 不是直接下发力矩，因此 YAML 的 torque_limits 不等于实机最终执行的严格软件限幅。真实硬件电机限幅/时延、起立衔接需要实机验证，不能声称已与仿真完全等效。

## 命令

重新导出：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python -m supermimic.baselines.go2_export \
  --checkpoint logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt \
  --policy-name supermimic_velocity
```

已有可视化，无需启动 rl_sar 或实机：

```bash
cd /home/yifeiy/ws_supermimic
PYTHONPATH=src /home/yifeiy/miniconda3/envs/supermimic/bin/python scripts/play.py \
  --ckpt /home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt \
  --num-envs 1 --vx 0.5 --vy 0 --yaw-rate 0
```

离线链路检查脚本为 `scripts/validate_go2_rl_sar.py`，编译源码在 `tools/rl_sar_go2_validation/`。GPU 忙时先不运行；此检查会短暂创建一个训练环境导出 MJB。验证程序只用于测试现有部署组件。

将来在接好设备的终端中选择策略：

```bash
cd /home/yifeiy/rl_sar
source devel/setup.bash
RL_SAR_GO2_POLICY=supermimic_velocity rosrun rl_sar rl_real_go2 <连接Go2的网卡名>
# PPO 正式训练导出后，将值改为 ppo_velocity。
```

上述实机命令本次未执行。现有操作：0/A 起立，1/RB+方向上进入策略，W/S、A/D、Q/E 调速度，Space 清零，P/LB+X 被动，9/B 趴下。正式两种算法训练结束后，启动器会自动用同一导出器更新对应策略目录。
