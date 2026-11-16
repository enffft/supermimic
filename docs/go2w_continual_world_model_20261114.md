# Go2W：持续更新同一份 World Model 与关节范围实验

日期：2026-11-14。目标仍是纯本体感知 Actor 的全向速度跟踪与超过 20 cm 的上台阶。**本次跑通了 WM 资产续训，但运动能力尚未达标。**

## 1. WM 作为持续积累的资产

按照用户澄清，同一机器人在相同动力学、控制与高程输入协议下，奖励、命令和课程实验共享一份 WM。加载后继续按 SuperMimic 更新网络、归一化统计和 Adam 状态，Actor/Critic 仍按原算法训练。

当前统一资产：

`/home/yifeiy/ws_go2_unilab/artifacts/world_models/go2w_conditioned.pt`

实现位于 [world_model.py](../src/unisupermimic/tasks/velocity/go2w/world_model.py) 与 [训练入口](../tools/train_go2w_experiment.py)。资产包含 WM 网络及其归一化 buffer、WM 优化器、累计更新计数、动力学与架构协议、来源 checkpoint 和张量摘要。运行结束后默认写回加载路径，并将上一版保存在相邻 `history/`。当前资产标记为 candidate，不能据此声明运动能力合格。

没有保留冻结 WM 的改动；共享 learner 的训练循环未因本次资产复用修改。此前将“固定资产”理解成“冻结模型”不符合用户意图，未启动的 `go2w_fixed_wm_20261114` 快照不是当前工作流。`go2w_joint_range_95pct_candidate.pt` 也不是当前统一资产。

奖励、命令、课程和地图内容不参与 WM 动力学身份；机器人资源、状态/动作坐标、控制器、采样周期、高程传感协议参与身份校验。正确的高程输入允许同一 WM 学习不同地形，但仍需检查新地形上的预测精度；改变动力学或动作定义不能直接忽略协议差异。当前流程按单个训练进程写统一资产运行；并行实验不能同时发布到该路径。

### 实际续训证据

本次从宽范围实验 iteration 1059 继续至 1179，新增 120 轮，用时 136.10 秒。每轮仍做 10 次 WM 更新，同时更新 Actor/Critic。

|记录|加载前|续训后|
|---|---:|---:|
|WM 累计更新|10,590|11,790|
|WM Adam step（全部参数）|10,590|11,790|
|输入/输出/增量归一化 count|11,104,256|12,333,056|
|Actor 累计更新|509|629|
|Critic 累计更新|11,180|13,580|

WM 张量摘要由 `28adf8701c9f5e61…` 变为 `e1b26c0aeb47b2c9…`，说明权重/统计实际发生了更新。旧资产保存在 `artifacts/world_models/history/go2w_conditioned_28adf8701c9f5e61.pt`。

原始证据：[连续性核对](../../runs/go2w_continual_wm_20261114/continuity_verification.json)、[训练汇总](../../runs/go2w_continual_wm_20261114/training/summary.json)、[发布记录](../../runs/go2w_continual_wm_20261114/training/world_model_publish.json)。

## 2. 部署关节范围核对

用户已确认使用 `/home/yifeiy/rl_sar/policy/go2w/base.yaml` 与 `robot_lab/config.yaml`。同时核对部署 URDF、FSM、输出转换和训练模型，来源文件 SHA256 保存在 [审计结果](../../runs/go2w_joint_range_20261114/deployment_joint_range_audit.json)。以下是部署模型定义及软件命令核对，未做实机逐关节极限测量。

|关节|部署 URDF 位置范围/rad|旧训练目标范围/rad|本次 95% 目标范围/rad|
|---|---|---|---|
|Hip|[-1.0472, 1.0472]|左 [-0.15, 0.35]，右 [-0.35, 0.15]|[-0.99484, 0.99484]|
|前 Thigh|[-1.5708, 3.4907]|[0.55, 1.05]|[-1.44426, 3.36416]|
|后 Thigh|[-0.5236, 4.5379]|[0.75, 1.25]|[-0.39706, 4.41136]|
|Calf|[-2.7227, -0.83776]|[-1.75, -1.25]|[-2.67558, -0.88488]|

旧配置 `default + 0.25 × action` 的确把可请求姿态限制得很窄。本次宽范围映射使 action=-1/0/+1 对应下限/原站姿/上限，每端保留机械区间 2.5% 的余量；部署 FSM 的折叠目标 `(0, 1.36, -2.65)` 已在范围内。轮速和力矩/物理约束没有随之放大。

部署配置中的 raw action clip ±100 不能解释成机械可达范围；输出换算没有把这些极值钳到机械位置。部署与训练的 PD、站姿、关节顺序、轮速比例和部分力矩参数仍有差异，本次没有把策略声明为可部署。

固定机身 FK 网格显示，旧 ±0.25 rad 目标只能产生约 8.6–9.6 cm 的最大轮心抬升，支持检查动作范围；该结果不是动态上台阶能力证明。见 [运动学诊断](../../runs/go2w_persistent_omni_20261114/kinematic_reach.json)。

本次从偏移动作改为宽范围动作属于一次控制坐标变更，旧 WM 不能直接把同一个归一化 action 当作新物理目标。宽范围 WM 最初独立创建在用户澄清前；随后已转为对上面的同一份资产持续更新，后续奖励/课程实验无需再新建 WM。

## 3. 课程、奖励与独立评测

依据 [前期调研](go2w_reward_curriculum_research_20261113.md)，为 4096 环境中的 512 个环境保留持续 WM 地形覆盖，等级 1–7 数量分别为 `[96,96,96,80,64,48,32]`。这些环境只在 reset 时换地形，其高等级采集身份不代表 Actor mastery；WM 与 Actor 原有起点筛选机制保持区分。

Replay 审计中，旧基线各等级保留量为 `[1047150,1426,0,0,0,0,0,0]`，覆盖实验为 `[917504,24576,24576,24576,20480,16384,12288,8192]`。高等级数据不再完全被覆盖，但高楼梯样本主要仍是入口附近尝试，不能视为完成爬升的数据。

每个策略采用 55 场景 × 3 个 reset seed × 4 副本，共 660 次首回合、最长 10 秒物理评测，包含横移、旋转、时变命令及各等级地形。3 个 seed 只改变出生位置/朝向，仍是同一训练 seed 与固定地形图，不代表跨训练种子结论。提前跌倒且没有有效统计时输出 null，不把缺样本计为零误差。

|实验|训练预算/轮数|观察结果|
|---|---|---|
|旧基线|已有 checkpoint|0.3 m/s 前向命令实际约 0.262 m/s；横移近零；3 cm 上台阶 12/12 跌倒|
|持续地形覆盖|600 秒，428 轮|3 cm 上台阶 0/12 跌倒，但前进约 1.42 m，未达到 2.1 m 入口通过标准；横移仍近零|
|覆盖 + stop gate 0.02|600 秒，374 轮|前向实际约 0.152 m/s；横移没有改善|
|宽关节范围 + sigma_y 0.15 + stop gate|1202.7 秒，1059 轮|660/660 提前跌倒；没有有效的后段跟踪样本|
|同一宽范围 WM/策略继续更新|136.10 秒，新增 120 轮|仍为 660/660 提前跌倒；资产续训成功，策略能力未改善|

早期覆盖/stop 对照发生在完善资产续训代码之前，当时实验初始化重置了优化器；当前 `initialize_experiment` 已延续 WM 优化器/计数。没有单独完成 sigma_y 的隔离训练，不能把组合实验结果归因于单个奖励项。简短测试过部署策略适配作为动作数据来源，但该仿真诊断失败，没有作为训练教师，也没有继续投入模仿数据集路线。

策略证据：[旧基线及覆盖对照](../../runs/go2w_persistent_omni_20261114/evaluation/)、[宽范围评测](../../runs/go2w_joint_range_20261114/evaluation/joint_range.json)、[最新策略评测](../../runs/go2w_continual_wm_20261114/evaluation/policy.json)。

### 同一物理留出数据上的 WM 误差

使用稳定旧基线重新采集的物理窗口，未插入训练 Replay。跨动作映射比较时，将归一化 action 重映射到相同物理 PD 目标并验证一致，轮速动作不变；预测每一步重新查询对应位置高程。最新 WM 在平地上的 RMSE：

|预测步数|位置/m|线速度/(m/s)|角速度/(rad/s)|腿关节位置/rad|
|---:|---:|---:|---:|---:|
|1|0.00243|0.0984|1.185|0.0429|
|8|0.0561|0.7396|5.746|0.8792|
|16|0.1539|0.6475|2.386|1.4861|

见 [留出评估](../../runs/go2w_continual_wm_20261114/evaluation/wm_common.json)。不同 horizon 的有效窗口集合可能不同，不能据个别数值下降解释为长预测更准。宽范围策略自身因过早跌倒未产生可用长窗口，不能仅看它的单步训练 loss 推断可靠。

当前优先问题是宽动作范围下的稳定采集与 WM 多步精度。需要保留宽可达区间，同时校准初始策略/探索幅度，并用稳定运动窗口继续改善同一 WM；这些稳定性调整尚未实现。不应据本轮失败直接把区间缩回 ±0.25 rad，也不应仅追加同配置长训练期待自然解决。

## 4. 资源与验证

使用 RTX 4070 SUPER 12 GB、i5-14600KF、16 GB RAM，4096 环境、batch 2048、Actor horizon 16、WM window 8、每轮 10 次 WM 更新。Torch/OMP 单线程，原生 MuJoCo 使用 CPU 工作池。观察到 GPU 利用率约 89%，训练 GPU 分配峰值约 3.55 GiB，进程 RSS 峰值约 1.9 GiB，没有 OOM 或 nonfinite。未并发堆叠多个训练进程。

训练使用独立源代码快照，由此前训练源码叠加本次 Go2W 文件构成，排除工作区其他 Go2/共享 learner 的未提交改动；每次 run 有来源记录与 `source_manifest.json`。当前仓库的其他已有改动未清理或覆盖。

最终工作区回归：86 passed、1 skipped（CUDA 专项跳过），包括资产历史保留与过期发布拒绝测试；另显式运行单环境物理测试 1 passed。Ruff 和 `git diff --check` 通过。原续训阶段测试日志为 85 passed、1 skipped，未包含后来补充的资产发布测试。连续性原始日志及测试日志位于 `runs/go2w_continual_wm_20261114/`，最终核对摘要见 [final_verification.json](../../runs/go2w_continual_wm_20261114/final_verification.json)。所有本轮训练均已结束。

## 5. 后续复用命令

以下从已评测快照执行，避免混入工作区其他实验改动。该命令用于复现续训流程；考虑当前跌倒情况，不代表建议直接做长时间同配置训练。`--iterations` 是累计轮数上限，`--seconds` 是本次运行时间预算，输出目录必须尚不存在。

```bash
cd /home/yifeiy/ws_go2_unilab/runs/go2w_continual_wm_20261114/source_snapshot
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/src" \
/home/yifeiy/miniconda3/envs/go2_unilab/bin/python tools/train_go2w_experiment.py \
  --profile joint_range \
  --resume /home/yifeiy/ws_go2_unilab/runs/go2w_continual_wm_20261114/training/last.pt \
  --wm-from /home/yifeiy/ws_go2_unilab/artifacts/world_models/go2w_conditioned.pt \
  --wm-only 500 \
  --output /home/yifeiy/ws_go2_unilab/runs/go2w_continual_next/training \
  --envs 4096 --batch 2048 --iterations 1299 --seconds 180
```

`--wm-only 500` 在这里用于匹配已有 checkpoint 的完整训练协议；iteration 已超过 500，不会再次执行 500 轮预热。若明确需要重建 Actor/Critic，同时沿用 WM，则用 `--fresh --wm-from <同一资产> --wm-only 0` 替代 `--resume ... --wm-only 500`。只有显式 `--new-wm` 才允许 fresh Actor 时创建新 WM；常规实验应继续使用已有资产。严格全量 resume 与奖励实验的权重初始化是不同入口，不能绕过任务协议检查。
