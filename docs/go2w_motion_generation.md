# Go2W 平地运动学参考

首批正式数据生成于 2026-11-24，起始 Go2W 提交为 `a9fe67424d78e48f32f77bef2cdd5d0992761fda`，生成器提交为 `8d823e4374701a820d60419988cb58b4371d01f0`。该模块独立于 SuperMimic 的训练入口，产物是用于后续 Motion Prior 研究的姿态、速度和接触参考。验证级别为 `kinematic_only`，物理验证标志为 `false`。数据接口采用运动状态，控制器仍使用腿部位置目标和轮速目标生成电机力矩。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate supermimic
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic-motion-prior
export OPENBLAS_NUM_THREADS=1

# 新建数据目录。已有目录会被拒绝，可换一个输出目录进行复现。
python -m tools.go2w_motion.generate --output-dir data/go2w_motion_reproduced
python -m tools.go2w_motion.generate --pilot --output-dir data/go2w_motion_pilot_reproduced
python -m tools.go2w_motion.generate --category lateral --output-dir data/go2w_lateral
python -m tools.go2w_motion.generate --gait crawl --category lateral --output-dir data/go2w_lateral_crawl
python -m tools.go2w_motion.quality_check --dataset data/go2w_motion_bank

python -m tools.go2w_motion.viewer --dataset data/go2w_motion_bank
python -m tools.go2w_motion.viewer --dataset data/go2w_motion_bank --category yaw
python -m tools.go2w_motion.viewer --dataset data/go2w_motion_bank --quality candidate --motion lateral_left_vy_035
python -m tools.go2w_motion.viewer --dataset data/go2w_motion_bank --quality candidate --motion yaw_left_wz_080
MUJOCO_GL=egl python -m tools.go2w_motion.viewer --dataset data/go2w_motion_bank \
    --motion yaw_left_wz_080 --frames 65 82 100 --render-dir data/go2w_yaw_review
python -m pytest -q tests/tools/test_go2w_motion.py
```

生成和审计依赖 NumPy、MuJoCo、PyYAML。桌面 Viewer 使用 MuJoCo 的 GLFW 窗口，PNG 导出使用 Pillow。正式数据使用 `supermimic` Conda 环境中的 Python 3.12、MuJoCo 3.16.0、NumPy 2.5.4。新增运动学测试也在训练工作区使用的 MuJoCo 3.11.0 上运行。

Viewer 中 `Space` 暂停或继续，左右箭头逐帧，`N/P` 下一条或上一条，`C` 切换类别，`Q` 切换质量筛选，`=/−` 改变倍速，`L` 开关循环，`Home` 返回起始帧，`H` 显示快捷键。数字 `1–6` 分别切换机身、CoM、规划轮心、FK 轮心、支撑区域和接触点。规划轨迹用虚线，FK 用实线，轮胎外侧短线随轮关节转动。两轮支撑绘制线段，其他支撑绘制填充轮廓。循环显式写入第 0 帧机身和全部关节状态。

## 模型和训练契约

生成、审计和回放加载 `src/unisupermimic/assets/robots/go2w/flat.xml`。正式训练的 `train.yaml` 使用 `terrain_robot.xml`，两份 MJCF 的关节轴和硬限位已逐名对照。生成器读取训练 YAML 的实体、关节和 actuator 名称，并通过 MuJoCo 的地址与传动关系写入状态。轮足顺序为 FL、FR、RL、RR，腿关节逐腿排列 hip、thigh、calf，随后是同顺序的四个 wheel。MJCF 的 qpos 每条腿后插入一个轮关节，actuator 遍历顺序为 FR、FL、RR、RL。

| 项目 | 源码或测量值 |
|---|---|
| 仿真和训练控制采样 | 0.005 s / 0.02 s |
| 腿和轮的自由度 | 12 / 4 |
| 髋座相对基座 | x = ±0.1934 m，y = ±0.0465 m |
| 髋外展偏置 / 两段腿长 | 0.0955 / 0.2130 / 0.2264 m |
| 旋转轴 | hip 局部 +X，thigh、calf 及四个 wheel 局部 +Y |
| 轮碰撞网格 | wheel.stl，最大半径 0.086 m，轴向宽约 0.0518 m |
| 质量 | 基座 6.921 kg，每条腿 hip / thigh / calf / wheel 为 0.678 / 1.152 / 0.241352 / 0.980 kg，整机 19.126408 kg |
| 腿部硬限位（rad） | hip [−1.0472, 1.0472]，前 thigh [−1.5708, 3.4907]，后 thigh [−0.5236, 4.5379]，calf [−2.7227, −0.83776] |
| 训练位置可命令范围 | `full_leg_target_range: true`，硬限位以区间中点收缩至 `0.95`，动作 −1/0/+1 映射至下限/默认/上限 |
| 训练增益 | leg 40.0 / 1.0，wheel 0.0 / 0.5 |
| 训练轮速命令 | ±15 rad/s，独立于 MJCF 的轮电机力矩限值 |
| 训练 keyframe | z = 0.42 m，前腿 `[±0.1, 0.8, −1.5]`，后腿 `[±0.1, 1.0, −1.5]`，左正右负 |
| 校准站姿 | z = 0.387112 m，前腿 `[±0.101206, 0.813263, −1.524606]`，后腿 `[±0.098810, 0.984866, −1.476018]` |

在 0.42 m 的默认 keyframe，实际轮底离地约 29–37 mm。校准保留默认轮心的水平坐标，降低基座并逐腿求地面接触。校准后的轮心高度约 80.5–80.7 mm，四轮网格离地误差小于 0.05 mm。MuJoCo 使用 MJCF 每个连杆的惯性中心及质量计算基座子树 CoM。出生姿态的 CoM 约为 `[-0.0111, 0, 0.3419]` m，校准后约为 `[-0.0111, 0, 0.309]` m。四轮正滚动方向均朝基座 +X，通过实际 FK 轴和地面法向叉积获得。

## 生成方法

1. 固定清单提供 100 条不重复命令，外加静止参考。每条 6 秒，以 0.02 秒采样，含首尾共 301 帧。种子 42 随批次记录，清单与求解均为确定性过程。机身使用 heading 速度、0.16 秒首尾静止和 0.64 秒五次速度包络，中点 twist 在 SE(2) 每个时间步作指数积分。第一批机身仅绕竖轴旋转。`synthesize` 支持直接传入逐帧命令。
2. 纯前后运动四轮滚动且保持腿姿。其余运动默认两条对角腿同步摆动，FL＋RR 与 FR＋RL 交替，中间含短暂四轮支撑期。触发依据为预测的横向工作空间偏移，落足点取摆动结束位姿下的名义轮心，再加有界的横向前馈。落点先检查解析 IK 和训练可命令范围，完整轨迹再通过逐帧碰撞与工作空间审计。支撑轮沿 heading 的纵向分量滚动，补偿支撑轮外倾变化带来的接触片横向速度。`STANCE` 表示近静止支撑，`ROLLING` 表示接触滚动，`SWING` 表示摆动。
3. 轮足水平轨迹用五次 Hermite 多项式匹配滚动端点的位置、速度和加速度。轮胎离地高度使用 `64 u³(1−u)³`，抬起和落地端点高度、速度及加速度均为零。由实际轮网格的轴向与径向包络迭代得到倾斜轮的轮心高度。三自由度解析 IK 固定在默认姿态所在的下方、负膝解支。不可达帧保留上一可达腿姿并显式标记。MuJoCo 从关节状态独立重算 FK。
4. 正式对角数据沿名义机身轨迹，CoM 到两轮支撑线段的距离作为量测。`--gait crawl` 以偏移和预测三角形裕量选择单摆腿，用有界、重叠的五次窗口修正机身平移，然后重算 IK、FK、CoM 和参考速度。两个方案共用审计与轮速求解。
5. 设轮轴世界向量为 `a`、地面法向为 `n`、轮心到接触处为 `r`。接触处速度由 `v_center + ω_carrier × r + qd (a × r)` 给出。沿归一化的 `a × n` 求解 `qd`，由梯形积分生成不包角的轮角。接触位置取实际旋转网格的最低高度及邻近接触片中心。腾空时插值轮速仅用于回放，`wheel_rate_valid=false`。

M1 的本地参考为 `/home/yifeiy/AMP-LOCO-M1` 的 `564a3c4`。V1 Pilot 与 Repair 提供了轻量生成、轴符号校验、独立 FK 和 C2 抬轮经验。V2.3 站姿报告、V2.4 速度与工作空间报告、V3.3 变速和 Goal Labeling 报告用于核对校准站姿、原始命令保留和配额。Go2W 使用命名后的 MJCF、倾斜轮网格、点 Jacobian、对角交替及双轮支撑诊断。

## NPZ 和 manifest

`T=301`。四元数采用 MuJoCo 的 `wxyz`，表示 world-from-base。Heading 取 world XY 平面上的机身 yaw，+X 前、+Y 左、+Z 上，正 yaw 向左转。参考速度以 world 向量导出，命令和速度追踪指标以 heading 计。SuperMimic 的紧凑状态使用自己的状态布局和关节映射，回放通过具名 MuJoCo 索引写入。

| 字段 | Shape | 单位、坐标系和来源 |
|---|---|---|
| `time` | [T] | s |
| `command_nominal` / `command` | [3] / [T,3] | heading 下的 m/s、m/s、rad/s，分别为原始请求和逐帧包络 |
| `base_pos_nominal` / `base_pos` | [T,3] | world m，名义积分与参考机身 |
| `base_quat` | [T,4] | world-from-base，无量纲，wxyz |
| `base_lin_vel` / `base_ang_vel` | [T,3] | world m/s 与 rad/s，由参考位姿求导 |
| `com_pos` | [T,3] | world m，MuJoCo 子树 CoM |
| `leg_q` / `leg_qd` | [T,12] | rad / rad/s，逻辑顺序，IK 与其时间导数 |
| `wheel_q` / `wheel_qd` | [T,4] | rad / rad/s，滚动估计及积分 |
| `wheel_center_target` / `wheel_center_fk` | [T,4,3] | world m，规划轮心与独立 FK 轮心 |
| `contact_point` | [T,4,3] | world m，支持或摆动轮的几何最低接触片 |
| `contact_mask` / `wheel_rate_valid` | [T,4] | bool，规划支撑与接地且 IK 有效的滚动监督掩码 |
| `motion_phase` | [T,4] | 0=STANCE，1=ROLLING，2=SWING |
| `ik_failed` / `ik_error` | [T,4] | bool / m，求解失败和轮心目标误差 |
| `rolling_residual` / `lateral_residual` | [T,4] | m/s，接触速度沿滚动方向和横向的分量，审计指标按支撑帧统计 |
| `support_margin` | [T] | m，CoM 到支撑多边形边界的有符号距离，两点支撑时为到线段距离的负值 |

Manifest 保存以上字段、采样、模型与控制配置摘要、当次参数、每条记录的原始命令、文件路径和有限尝试的结果。原始命令、逐帧目标和实际参考速度始终独立。后续 Motion Prior 可依照 manifest 选取合格的状态、接触掩码、时间窗与速度条件，为离线状态模仿、镜像复核、参考匹配及分布偏移研究提供输入。

## 独立审计

审计重读 NPZ 和快照配置，重新检查数组、采样、命令包络、四元数、命名布局、FK、关节限位和速度、加速度及 jerk。MuJoCo 点 Jacobian 从位姿导数和存储轮速重建接触速度。质量报告区分整段参考均值和恒速段均值/RMSE，并输出名义、修正后速度误差和镜像统计。报告保留每条记录的 `qualified`、`candidate` 或 `rejected`。

关键量级为 1 mm 的 IK 误差、3 mm 的接地距离与穿透碰撞检查、12 rad/s 的腿关节速度工程阈值、±15 rad/s 的控制轮速、0.035 m/s 的横向接触残差 RMS、5 mm 的多边形支撑裕量。腿速度阈值属于工程审查参数。对角阶段检查两轮组合、同步和交替顺序，记录 CoM 线段距离，审查阈值为 80 mm。进入两轮支撑的轨迹归为动态 Candidate。每个阈值可在归档配置中核查。

所有 30,401 帧均做碰撞几何检查。每帧覆盖 214 个不同、非直接相邻连杆的自碰撞候选，涵盖腿间和机身与非相邻腿段，以及 19 个非轮几何对地检查和四个真实轮网格的地面高度。直接父子连杆和同一刚体是几何排除范围，采样间的连续碰撞与动态平衡尚未验证。

## 首批数据和验收

| 类别 | 请求 / 写入 | Qualified | Candidate | Rejected | 生成时间 |
|---|---:|---:|---:|---:|---:|
| 前进、后退 | 22 / 22 | 22 | 0 | 0 | 3.81 s |
| 原地旋转 | 16 / 16 | 0 | 16 | 0 | 2.93 s |
| 侧移 | 14 / 14 | 0 | 14 | 0 | 2.56 s |
| 前进＋旋转 | 16 / 16 | 0 | 16 | 0 | 2.95 s |
| 前进＋侧移 | 16 / 16 | 0 | 16 | 0 | 2.96 s |
| 侧移＋旋转 | 16 / 16 | 0 | 16 | 0 | 2.93 s |
| 静止 | 1 / 1 | 1 | 0 | 0 | 0.17 s |

生成 18.31 s，含静止共 30,401 帧，NPZ 约 13.02 MiB。独立审计约 10 秒，manifest 状态与独立结果全部一致。正式批次包含全部原始命令，最高前进 0.75 m/s、后退 −0.35 m/s、侧移 ±0.35 m/s 和原地转速 ±0.80 rad/s。39 对镜像通过姿态与机身轨迹一致性检查。各类别的峰值、失败原因和逐条速度见 [审计摘要](validation/go2w_motion/quality_summary.json)、[逐条 CSV](validation/go2w_motion/quality_report.csv) 和 [生成报告](validation/go2w_motion/generation_report.md)。

此前的单腿方案实测为 50 Qualified、34 Candidate、16 Rejected（站立另计）。其中纯侧移的 ±0.25 至 ±0.35 m/s 受支撑期横向工作空间与腿部连续性限制而拒绝，原地旋转通过覆盖为 ±0.10 至 ±0.50 rad/s。对角方案扩大了可审查的几何覆盖，四轮接触阶段的最小支撑裕量为 194.8 mm。两轮阶段中，纯侧移、原地转向、侧移＋旋转各自的 CoM 线段距离峰值为 52.5、22.7、50.3 mm。首批的 78 条两轮支撑记录均只触发动态候选标记，真实接地、承载、力矩以及平衡仍需物理阶段验证。

验收样图包括低速和高速前进、后退、双向旋转、左右高速侧移和三种耦合运动。代表帧见 [运动总览](validation/go2w_motion/representative_motions.jpg)。全分辨率截图、`yaw_left_wz_080.mp4` 和 `lateral_left_vy_035.mp4` 位于本地 `data/go2w_motion_bank/visual_review/`。

MuJoCo 桌面窗口实际验证了模型、轮胎转动标记、类别与质量筛选、上一条/下一条、倍速、单帧、复位和覆盖层。正向最高速旋转的第 80 和 95 帧分别显示 FL＋RR 与 FR＋RL 抬轮。播放复位以及机身和轮角的状态写入也有自动测试。[第 80 帧](validation/go2w_motion/gui_yaw_pair.png) · [第 95 帧](validation/go2w_motion/gui_yaw_opposite_pair.png)

新增运动模块 15 项测试通过。现有 Go2W CPU 回归 90 项通过，另有 1 项单环境 MuJoCo 映射、控制与碰撞测试通过。现有回归使用具备 UniLab 后端依赖的 `go2_unilab` 环境。生成和回放使用共享的 `supermimic` 环境。
