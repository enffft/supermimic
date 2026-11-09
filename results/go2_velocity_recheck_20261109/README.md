# Go2 SuperMimic 速度跟踪复核与地形接入审计

日期：2026-11-09。此处只对已训练策略做推理评估，没有改动策略 Checkpoint、机器人模型或训练源代码。

## 策略来源与复核条件

- Checkpoint：`logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt`
- Checkpoint SHA256：`0b3d8e4bb26173d13a6ce0b99d5eecf43005927a8ab2e01947c4d0311009487a`
- 训练链路约 4.55 h：约 0.25 h WM 初始化、约 2.00 h 基础 Actor 训练，以及约 2.29 h 步态/滑脚续训。它由多个阶段串联，不是一次连续运行；最终检查的是 iteration 34000。
- 复核：每条命令 64 个并行环境、64 个 10 s episode，seed 2026；关闭观测、动作、Reset 噪声和 Domain Randomization。
- 所有并行环境使用相同名义初态和命令，因此 64 episodes 不是 64 个独立随机初态；结果不能用于计算鲁棒性置信区间。

## 平地速度跟踪结果

| 指令 `[vx, vy, wz]` | 实际平均 `[vx, vy, wz]` | XY 速度 RMSE | Yaw RMSE | 10 s 终点位移范数 / 跌倒率 |
|---|---:|---:|---:|---:|
| `[0.0, 0.0, 0.0]` | `[-0.009, -0.001, -0.013]` | 0.0148 m/s | 0.0288 rad/s | 0.088 m / 0% |
| `[0.3, 0.0, 0.0]` | `[0.289, -0.004, 0.018]` | 0.0368 m/s | 0.0258 rad/s | 2.885 m / 0% |
| `[0.5, 0.0, 0.0]` | `[0.483, 0.0, 0.022]` | 0.0598 m/s | 0.0316 rad/s | 4.814 m / 0% |
| `[-0.3, 0.0, 0.0]` | `[-0.29, -0.004, 0.022]` | 0.0401 m/s | 0.0290 rad/s | 2.888 m / 0% |
| `[0.0, 0.3, 0.0]` | `[-0.006, 0.3, 0.03]` | 0.0383 m/s | 0.0498 rad/s | 2.990 m / 0% |
| `[0.0, 0.0, 0.4]` | `[-0.005, -0.002, 0.407]` | 0.0148 m/s | 0.0531 rad/s | 0.004 m / 0% |
| `[0.4, 0.15, 0.3]` | `[0.385, 0.151, 0.32]` | 0.0535 m/s | 0.0433 rad/s | 2.575 m / 0% |

滑脚值使用工作树评估脚本的 FK 脚高阈值近似支撑状态，不是 MuJoCo 接触力/接触标志；只作趋势指标。

## 结论

- 平地名义条件下，前进、后退、侧移、转向和组合命令都能跟踪。448 个完整 episode 均未跌倒，但因各环境起始条件相同，这代表重复名义工况而非强鲁棒性测试。0.3 m/s 前进实测 0.289 m/s，0.5 m/s 实测 0.483 m/s，后退 -0.3 m/s 实测 -0.290 m/s。
- 停止命令仍有残余漂移：平均 vx=-0.0086 m/s，10 s 的终点偏移约 8.8 cm；平均 yaw rate=-0.0128 rad/s。
- 后退脚滑代理值 0.367 m/s，高于前进 0.167 m/s；侧移为 0.249 m/s。速度跟踪虽好，脚步/接触质量仍未通过直接接触测量确认。
- 与 2026-11-06 的 16 episode 审计数值几乎重合，说明该名义平地测评可复现；Checkpoint 尚未经过坡面或楼梯验证。

## 地形如何接入

Go2 环境在 [`multirobot_env_cfg.py`](/home/yifeiy/ws_supermimic/src/supermimic/envs/multirobot_env_cfg.py:417) 构造，目前 `SceneCfg` 将 `TerrainEntityCfg(terrain_type="plane")` 写死。G1 tracking 分支也单独写了平地，但 Go2 走 multirobot builder。安装的 mjlab 已有程序化 `TerrainGenerator`，包含：

- `HfPyramidSlopedTerrainCfg`：高度场四向金字塔坡，坡度按 rise/run 定义；不是单方向长坡。[mjlab 类型定义](</home/yifeiy/miniconda3/envs/supermimic/lib/python3.12/site-packages/mjlab/terrains/heightfield_terrains.py:216>)
- `BoxPyramidStairsTerrainCfg`：MuJoCo box 碰撞几何组成的金字塔楼梯，可设台阶高、踏面宽、顶面平台。[mjlab 类型定义](</home/yifeiy/miniconda3/envs/supermimic/lib/python3.12/site-packages/mjlab/terrains/primitive_terrains.py:55>)
- `BoxFlatTerrainCfg`：可作为混合地形中的平地对照。

推荐接入步骤：

1. 给环境参数增加可选 Go2 `terrain_profile`，默认保持 `plane`；只在 velocity Go2 场景构造 `TerrainEntityCfg(terrain_type="generator", terrain_generator=TerrainGeneratorCfg(...))`。把 terrain profile、种子、地形比例、坡度/台阶范围写进训练 config/Checkpoint。
2. 先用固定 seed、小规模物理 smoke 测 reset 高度、足底接触、碰撞数、台阶边缘和 tile 接缝。保守试验起点可用 10–15% 坡度、3–5 cm 台阶高和约 30 cm 踏面；它们是候选起点，不是已经验证的安全界限。
3. 旧策略先做零样本分项评测：平地、坡面、上楼梯、下楼梯分别记录速度 RMSE、跌倒率、base 高度/倾角和足端接触/滑动指标，不把它当作地形训练成功。
4. 训练先固定环境地形配额与难度，再分阶段加坡度/台阶高度。TerrainGenerator 支持按比例随机混合，也支持不同 terrain type 分列、按 row 增难；自动晋级 terrain level 还需把成功率/失败率接入环境课程控制。
5. 目前 Go2 actor 只看关节位置/速度、base 角速度、重力投影和上一动作，没有地形高度扫描。低台阶可以先验证本体感知；稳定爬楼梯建议加入局部高度扫描并重新训练，旧的 12 维 action Checkpoint 保持独立评估。
6. 金字塔坡和金字塔楼梯的内置生成器将机器人出生原点放在中央平台/顶部附近；若目标是“从坡脚/楼梯底向上爬”，要增加底部起点采样，不能只使用默认出生点。单向坡则需自定义 heightfield 或倾斜平面几何。

## 产物与执行边界

- [结构化逐指令数据](fixed_seed_64ep.json)
- 本次没有改源代码、创建分支、提交或推送；`scripts/evaluate_go2_velocity.py` 在任务开始前已有本地未提交改动，本次复核使用并记录了该脚本哈希。
- 本次只运行了平地复核，没有执行地形 smoke 或 terrain training。
