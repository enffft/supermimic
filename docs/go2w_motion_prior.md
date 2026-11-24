# Go2W Motion Prior 与训练阶段

本次从 `174c6c11` 的 Go2W 对角步态 Motion Bank 开始。使用 23 条 Qualified 参考（含站立），以及 78 条唯一质量标记为 `dynamic_diagonal_support` 的动态 Candidate。两类的模仿权重为 1.0 和 0.5。Motion Bank 的原始质量报告保留，训练进程记录实际读取的 manifest 和 NPZ 联合 SHA-256。数据约 3 万帧，按字段转为连续张量，训练和仿真设备各持有一份。 [Motion Bank 文档](go2w_motion_generation.md)

## SuperMimic 接入

Actor 保持 50D 本体历史×5 和 7D goal，输出 16D 混合 PD 的归一化动作。参考选择先比较命令活跃轴及符号，再在同类的最多 3 个近邻命令所对应的恒速段，用腿位置、腿速度和有效轮速度匹配相位。每条参考均预计算可用于 H8 的连续起点。一次 Actor update 只匹配一次，预测的第一步和匹配相位的下一帧比较。距离过大的命令逐步减弱到零，三轴同时非零的命令继续用任务回报。

Actor Loss 为现有负回报加固定权重乘运动状态 Smooth L1。腿位置、腿速度、轮速度尺度为 0.25 rad、3 rad/s、5 rad/s，其内部权重为 1.0、0.25、0.5。轮速按参考接地有效位掩码，存活掩码沿用 imagination 的 soft termination。用于模仿的存活权重从梯度图分离，任务回报的原有终止梯度和 Critic bootstrap 继续生效。Actor 更新期间 WM 和 Critic 冻结参数，梯度经过 WM 的状态预测到达 Actor。Critic target 仍由任务奖励生成。单次匹配、标定与回归在现有 `SuperMimicLearner` 完成。

## 物理采集和 Replay

默认活动环境并发配额约为普通 Actor 70%、RSI Actor 20%、参考 PD 10%。按 episode reset 划分控制来源，按并发人数封顶以免较长 episode 过度代表参考控制。RSI 从四轮支撑的恒速帧取样，用相对地高和世界速度重锚到该环境出生点。根四元数按 wxyz 写入 UniLab 接口，MuJoCo 自由根 qvel 所需 body-frame 角速度由接口转换。真实 reset 观测填充 Actor 历史。引导步使用参考的下一帧腿位置与轮速命令，通过现有默认中心非对称满量程 PD 逆映射得到动作。轨迹结束后在末帧停稳阶段保持目标。物理 Replay 的 action 和 next-state 由真实 `env.step()` 提供。

Recent 和 Historical Window Bank 继续共用现有 Replay。WM 目标配额为 50% 近期、30% 普通历史、20% 受保护成功窗口。成功窗口依据生存、命令方向和真实速度误差选取，台阶楼梯依据严格 `traversal_finished`。保护项不足时回退到近期或普通历史数据。窗口保持 episode、连续步和目标 ID 一致。Actor 起点继续使用 Recent Replay 的真实状态，并按平地命令类别及符号采样。Terrain 阶段平地名义起点配额为 40%，实际缺额记录在指标中。

## 阶段、奖励和课程

- `flat_prior` 使用纯平地配额和零非平地 coverage。速度按 signed 平地技能掌握程度在配置速度上限内插值，配合较低速度复习。采样保留站立、正反向直行、左右移、正反旋转及三类耦合运动，默认 80% 离散 Motion Bank 命令。参考初始化时命令与参考帧一致。
- `flat_autonomous` 继续纯平地。RSI、引导、参考权重在配置的 1800 秒内逐渐退出，采样加入定时命令切换和三轴命令。需脱离参考后通过平地和切换协议。
- `terrain` 按配置的两个八地形配比分配物理环境。切换复用已编译地形 atlas，清除重新分配环境的 mastery 和 episode 状态，使用完整批量 reset，显式闭合旧 Replay 边界。Actor、WM、Critic、优化器、normalizer 和真实历史经验继承。坡面与台阶使用配置中的正向速度，台阶与楼梯可按掌握程度加入初始 yaw 扰动。crossing direction 保持与地形同向。多级楼梯目标距出生点 2.8 m，最终区域要求轮足越过约 2 m 边界。坡面、台阶和楼梯使用 18 秒时限，楼梯采样下界还按轮足通过距离和起步余量修正，平地保持 10 秒及原有边界保护。

阶段依据两次连续固定 MuJoCo 评测的分方向通过率决定。首轮将 `max_stage` 设为 `flat_prior`，达标时记录 `ready_for_next_stage`。平地课程和真实物理评测同时检查速度误差、正确方向的运动及跌倒。较高速评测在 4.2 秒内完成以适应平地 tile 尺寸。

本 profile 将默认姿态偏差、腿和轮加速度、side drift 及轮速一致性权重设为零。共用 Reward 逻辑在这些项关闭时跳过相应计算。保留速度跟踪、站立、地形对齐、关节限位和较小的动作变化及力矩惩罚。楼梯跟踪权重为平地的 100%／30%／30%，保留越障进度和成功奖励。训练的任务奖励语义在阶段间一致。早期 profile 的 checkpoint 契约和原配置兼容。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate go2_unilab
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic-motion-prior
export PYTHONPATH=.:src:../UniLab/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python tools/train_go2w.py --mode formal --profile motion_prior --fresh \
    --envs 4096 --seconds 10800 --max-stage flat_prior --output runs/go2w_motion_prior

tail -f runs/go2w_motion_prior/train_log.csv
cat runs/go2w_motion_prior/progress.json

python -m tools.evaluate_go2w --protocol motion_prior \
    --checkpoint runs/go2w_motion_prior/best.pt --output runs/go2w_motion_prior/best_recheck.json --switches
python -m tools.evaluate_go2w --protocol motion_prior \
    --checkpoint runs/go2w_motion_prior/last.pt --output runs/go2w_motion_prior/last_recheck.json --switches

python tools/play_go2w.py --checkpoint runs/go2w_motion_prior/best.pt --vx 0 --vy 0.20
python tools/play_go2w.py --checkpoint runs/go2w_motion_prior/best.pt --vx 0 --wz -0.40

python tools/train_go2w.py --mode formal --profile motion_prior \
    --resume runs/go2w_motion_prior/last.pt --envs 4096 --seconds 10800 \
    --max-stage terrain --output runs/go2w_motion_prior_continued
```

`last.pt` 保存优化器、模型、Replay、课程、阶段和随机数状态。恢复在新建物理环境上进行，Replay 重新闭合边界。约每 30 分钟保存推理 checkpoint 并完成真实 MuJoCo 评测，根据平地通过率和跟踪误差选择推理用 `best.pt`。三小时按实际训练循环计，预采集、外部评测和 checkpoint 写盘单独统计。

## 性能和验证

实际后端为 `unisim.backend.mujoco.backend.MuJoCoBackend`，当前使用 20 个原生 MuJoCo CPU 线程，RTX 4070 SUPER 上运行 CUDA Learner。短测使用正式模型宽度、H8、Batch 2048 和十次 WM 更新。

| 条件 | 训练循环 steps/s | s/iteration | 峰值分配 MiB |
|---|---:|---:|---:|
| 2048 环境，先验 | 5522 | 0.927 | 2881 |
| 4096 环境，关闭先验 Loss | 8918 | 1.148 | 4075 |
| 4096 环境，先验 | 9030 | 1.134 | 4077 |
| 4096 环境，先验且同步 profiling | 8911 | 1.149 | 4077 |

短测时间包含真实采集和更新。启用先验后的差异落在短程整机吞吐波动范围内，同步 profiling 比同配置非同步低约 1.3%。固定梯度诊断在 4096 环境、`lambda_prior=0.25` 时分别观察到任务梯度的 14.8% 和 12.2%，与任务 Loss 梯度余弦为 0.33 和 0.98。以同批诊断线性换算后选定正式 `lambda_prior=0.60`，对应约 36% 和 29%。短测与正式运行的采集占比约 70%／20%／10%。测试包括参考窗口、动力学采集、WM 参数冻结、课程实地重分配和 checkpoint 恢复。

完整训练结果由最终 MuJoCo 评测和 H1/H8 独立真实物理留出生成。最终报告与大型文件保存在 `runs/go2w_motion_prior/`。
