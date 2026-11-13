# Go2W SuperMimic：轮足 RL 奖励与课程调研

日期：2026-11-13。范围：阅读公开论文全文、开源配置及本地训练源码快照、评测与日志；没有启动新训练或仿真，没有修改任务实现。

核心建议：持续保留高等级环境采集 WM 数据，同时让 Actor 主要学习当前能力附近的任务。高等级数据覆盖、模型可靠性、机器人通过能力是三项不同指标。

## 1. 本次证据和适用边界

本地训练依据为 `runs/go2w_curriculum_4096_b2048_H16_20261112_183910/source_snapshot`，提交 `13b0861e95c0673cd721ab5260b957958b1a39ee`。当前仓库有其他实验的未提交修改，本报告没有把那些代码当作已训练版本。

评测依据为该 run 的 `evaluation_20261113_121348/assessment_report.md`，54 场景 × 4 出生偏移，固定 seed=2026；这是有限场景评测，不能当作跨种子的真实通过率。

### 1.1 开源实现

审阅日期为 2026-11-13。以下链接固定到本次核实的仓库提交。原始网页、代码及元数据缓存保存在本机研究目录 `research/go2w_reward_curriculum_20261113`，不随本报告提交；本文提供外部固定版本链接及可复算公式，阅读不依赖本地缓存。

|来源|核实到的实现|值得借鉴|不能直接照搬的部分|
|---|---|---|---|
|[LocoWheeledLegged](https://github.com/zhaozijie2022/LocoWheeledLegged/tree/015b8230b1a21331ab6d2b054a2ef8f1e435db84)|Go2W、Isaac Lab，仓库提供实机演示；12 腿位置动作和 4 轮速动作，腿轮加速度分别正则|按轴命令课程；跟踪 EMA；地形按行进距离升级；腿轮分开约束|Isaac Lab/PPO 的可用接触信息、动作范围和训练协议与 SuperMimic 不同；演示不等于独立复现|
|[DreamWaQ_Go2W](https://github.com/ShengqianChen/DreamWaQ_Go2W/tree/b64f7b90856fc71595870cb283ce29dd0f4d6f0f)|社区 Go2W 改造，Isaac Gym；4096 环境，73D observation、5 帧历史、320D privileged observation；提供部署接口及演示|混合控制 PD；初始化允许多个地形等级；经典距离课程；停止与运动分开处理|不是原 DreamWaQ 论文的官方 Go2W 实证；包含估计器，不能据此要求当前 Actor 增加速度估计或特权输入|
|[robot_lab Go2W](https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py)|Go2W 配置去除 Actor 真实线速度、高程和轮角；腿轮分开正则；feet gait/slide/stumble/clearance/airtime 权重为零|简单奖励也能作为合理起点；不应把大量轮地启发式奖励当作必需项|该配置关闭了命令课程，不能把公共 curriculum 文件中的函数称为其当前启用设置；部分奖励仍依赖接触数据|

具体配置差异：

|项目|速度跟踪|腿/轮加速度|姿态及高度|课程|
|---|---|---|---|---|
|LocoWheeledLegged|x/y/yaw 各 0.75；std=√0.25=0.5|−2.5e−7 / −2.5e−9|世界水平惩罚 −0.5，目标相对高度 0.40m|命令轴倍率 0.1→1.0，EMA α=0.5，上/下阈值 0.8/0.5；地形距离课程|
|DreamWaQ_Go2W|平面 1.5，yaw 0.75；代码分母 tracking_sigma=0.25|腿 −1e−7，轮 −1e−7|orientation −0.5，高度 −10，目标 0.4m；总奖励截为非负|10 行地形、max_init_terrain_level=5；距离课程及命令扩展|
|robot_lab Go2W|平面 3.0，yaw 1.5|−2.5e−7 / −2.5e−9|flat_orientation 和 base_height 为零，另有 upward 项|此 Go2W 配置明确关闭线/角命令课程|

上述权重不能直接比较大小：存在二维联合误差/按轴误差、L1/L2、奖励裁剪、实际动作幅度及 dt 约定等差异。尤其 `tracking_sigma=0.25` 表示分母 0.25 的实现，对应标准尺度 0.5；我们当前 sigma=0.25 的分母是 0.0625。

参考源码：[Loco 奖励配置](https://github.com/zhaozijie2022/LocoWheeledLegged/blob/015b8230b1a21331ab6d2b054a2ef8f1e435db84/locowheeledlegged/config/go2w/locomotion_env_cfg.py)、[Loco 课程](https://github.com/zhaozijie2022/LocoWheeledLegged/blob/015b8230b1a21331ab6d2b054a2ef8f1e435db84/locowheeledlegged/mdp/curriculums.py)、[DreamWaQ Go2W 配置](https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/go2w/go2w_config.py)。

### 1.2 论文

**Chamorro et al., ICRA 2024, [Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots](https://arxiv.org/html/2402.06143v1)**。已读全文 III-A3、III-B、IV-A/B。

- 先学习单个台阶，再学习连续五级楼梯：12 行课程的前半段是逐渐升高的单台阶，后半段是连续楼梯。这对当前入口受阻问题很有参考价值。
- 采用位置任务，终点奖励主要在最后 2 秒给出，另有临时朝目标运动奖励和停滞惩罚，允许过台阶时暂时减速。
- 4096 机器人，从简单等级开始；接近目标 20cm 升级，距离目标超过 50cm 降级，到最高级后随机复习。
- Ascento 实机跨越 15cm；其他机器人及更高高度的结果需区分仿真。**Actor 带 stair boolean**，不是我们要求的无地形标签 Actor。因此只能借鉴单台阶课程、速度容忍和停滞诊断，不能照搬模式输入，不能作为当前 Go2W 必能过 20cm 的证明。
- 论文指出高摩擦下可能学到依靠轮与台阶边缘抓附的仿真技巧，受到真实轮胎碰撞、回弹、失接触影响。不能通过提高摩擦或修改碰撞让高台阶虚假成功。

**Lee et al., Science Robotics 2024, [Learning robust autonomous navigation and locomotion for wheeled-legged robots](https://arxiv.org/html/2405.01792v1)**。已读全文及随文补充 S3、S5。

- 有苏黎世、塞维利亚公里级实机任务，是较强的系统成熟度证据。
- 步行/轮行转换不依赖手工 CPG 或固定 gait 序列；低层奖励包括速度跟踪、姿态/机身运动、力矩和动作平滑。
- 式 14 除指数跟踪外，还包含命令方向速度内积；静止命令单独处理。但该内积会改变高速行为，不建议本次直接引入。
- S5 的地形参数过滤选择成功率介于两个阈值之间的任务，避免 Actor 主要学习过易或完全不可解的地形。
- **部署低层仍输入地形高程并使用 RNN**，不是纯盲走对照；不是 SuperMimic，不能把其模型无关 PPO 课程直接当成 WM 数据覆盖方案。

**Nahrendra et al., ICRA 2023, [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/html/2301.10602)**。已读正文 II-B4/5。

- 原论文是四足 A1，5 帧本体历史、非对称 Actor/Critic；不是 Go2W 论文，也不是我们的可微动力学 WM。
- 使用 game-inspired 地形课程，并报告 grid-adaptive 低速命令课程改善转向、防止绊脚。支持我们分别评估低速倒退、横移、左右转向，而非只看总平均奖励。
- 原论文的估计器架构和足端奖励不是本任务必须加入的模块。

**Rudin et al., CoRL 2021 / PMLR 2022, [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html)**。核实会议页面及摘要；上述社区仓库可以直接核实经典距离课程的实际代码。它支持大规模并行环境的逐级任务与复习设计，但不是轮足或可微 WM 的专项验证。

## 2. 我们当前的实际问题

### 2.1 高等级数据没有持续保留

训练快照 `go2w/diagnostics.py:begin_iteration`：前 500 轮是 wm 覆盖阶段，之后所有环境切到 skill；`TerrainCurriculum` 在 skill 中围绕各环境 mastery 采样。没有独立且持续的高等级保留配额。

最新诊断记录 iteration=7760 的当前收集量按 Level 合计为：

`[12270, 18, 0, 0, 0, 0, 0, 0]`

这是该日志迭代的采集数，不是整个训练的累计数。累计高等级数据确实采过，但不能证明末期 Replay 仍保留。最终课程 mastery 为 `[4090,6,0,0,0,0,0,0]`。

Replay 每环境仅保留 256 个时间点，控制周期 0.02s，即 **5.12s 的模拟时间历史**。因此先预热高等级、之后长期只采 Level 0，不能持续支撑高等级 WM 学习。增加总 batch 本身不能补回已被覆盖的数据。

### 2.2 Actor 起点准入存在保守闭环

`begin_iteration` 用同领域最高 mastery 限制 Actor 起点 Level；`sample_starts` 使用此 mask，`sample_windows` 的 WM 采样不使用该 mask。

这样保护了 Actor 不从高等级错误模型状态开始优化，但可能形成“没有晋级→没有下一等级起点→难以学习下一等级”的循环。不能直接全部放开，也不能把 WM 覆盖视为 mastery。

### 2.3 评测显示受阻和欠激励

- 平地 vx=0.3：实际约 0.263m/s，RMSE 0.039；前进明显改善。
- 横移命令 ±0.06、±0.15：实际 vy 接近零。
- 上楼 3cm：本次 4/4 跌倒；6–21cm：多为不跌倒，但只前进约 0.4m，停在入口，中心下方地面高度没有变化。
- 下楼 3/6cm 有进展；高等级仍大量跌倒。
- 有探索噪声评测未普遍复现训练日志高失败率，不能简单把课程停滞归因于噪声。

这支持优先诊断有效台阶交互、动作可达范围和课程，而非直接再延长训练时间。

### 2.4 奖励的解析检查

本节结果使用 Python 标准库 `math` 复算，无仿真或 Torch 初始化；完整复算脚本见附录。所有数值都是奖励率，实际单步奖励还要乘控制周期 `dt=0.02`。

当前 axis-wise baseline 在每一轴命令接近零时恢复该轴正奖励。对于命令 `(0.3,0,0)`，受阻但直立且横向/转向稳定时，约保留 `1.5 + 2.0 + 0.4 = 3.9` 的奖励率；理想前进约为 `5.808`，均未扣控制正则。

这不证明静止比成功更优，但提示安全停滞可能保留较大的回报。固定命令下的常数 baseline 对即时状态导数没有贡献；在存在软终止、存活权重、Critic 和有限 rollout 的目标中，也不能笼统说常数绝对不影响策略偏好。

横移命令 0.06m/s，sigma_y=0.25 时，理想跟踪相比静止的 y 跟踪增益仅约 **0.08396**。与此同时停止门控 `exp(-(0.06/0.05)^2)` 仍为 0.237，静止可得约 **0.11846** 的停止奖励率。停止奖励对非零小命令存在明显渗漏，是值得优先做消融的原因，尚不是已验证的唯一根因。

需要区分绝对奖励和奖励差：按同一公式，理想以 0.06m/s 横移时，停止项约为 0.10095，因此从静止变为理想横移损失的停止奖励约 0.01752，而非整个 0.11846。仅合并这两项，理想跟踪仍多约 0.06644；不能据此断言静止已经成为数学最优解。实际还受其他奖励、动作可达性和模型梯度影响。

## 3. 建议的持续 WM 覆盖方案

### 3.1 保留 4096 总环境，划出 512 个持续采集环境

这是建议初值，尚未实施或验证。保留 **3584 个技能环境 + 512 个 WM 覆盖环境（12.5%）**。它们共享当前策略、动力学和世界地形，不增加总环境数。

建议覆盖子集的固定最低配额：

|Level|楼梯高度|WM 覆盖环境数|
|---:|---:|---:|
|0|0cm|0，技能环境提供大量平地数据|
|1|3cm|96|
|2|6cm|96|
|3|9cm|96|
|4|12cm|80|
|5|15cm|64|
|6|18cm|48|
|7|21cm|32|
|合计||512|

L6–7 始终至少 80 个环境，约占全部环境 1.95%。高等级量保持小但非零，避免将大部分计算花在重复跌倒上。该表是对非平地域的合计，粗糙和斜坡仍使用各自难度映射，不能都改成相同的厘米高度。

六领域总比例维持 `25/20/10/10/17.5/17.5%`。在既有非平地域配额及其物理地形列内部选择覆盖环境，不能简单取前 512 个 env ID，否则 `env_id%5` 和 atlas 类型可能串位。具体 domain×level 整数配额应由受约束分配计算并断言容量；本报告没有声称上表已经通过实际 atlas 分配验证。

建议固定角色与 Level 配额，在 episode reset 时再改变出生位置/地形实例，避免中途重定位。所有角色、域、Level、出生位姿、课程状态进入训练元数据和 checkpoint；**不进入 Actor 观测，也不扩展 41D WM 状态**。

可采用下面的整数候选表，避免“高 Level 配额全部落到粗糙地形，楼梯仍然没有数据”。它满足上面的 Level 合计，仍需实现时核对 atlas 和固定领域映射。

|覆盖领域|L1|L2|L3|L4|L5|L6|L7|合计|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|rough|24|24|24|20|16|12|8|128|
|slope_up|12|12|12|10|8|6|4|64|
|slope_down|12|12|12|10|8|6|4|64|
|stairs_up|24|24|24|20|16|12|8|128|
|stairs_down|24|24|24|20|16|12|8|128|
|合计|96|96|96|80|64|48|32|512|

其中上楼、下楼各有 12 个 L6 和 8 个 L7 环境。该配额只是初始分配，不表示这些环境已经能通过高台阶，也没有改变技能晋级标准。

### 3.2 让高等级产生有用动力学数据

- 保留普通入口与不同合法接近距离，使样本覆盖接近、触碰、受阻、支撑变化和跌倒前后的真实状态。不能以出生穿模、悬空坠落替代越障交互。
- 增加少量单台阶场景是有论文依据的几何课程候选；先作为独立诊断/消融，不能静默改变原八级楼梯协议。
- 失败转移对 WM 有价值，不能全部丢弃；但不能把大量重复静止或出生即跌倒的数据称为充分的高等级运动覆盖。
- 保留终止转移和现有 done mask；H8 序列不跨 reset。分别记录完整 H8 窗口、部分有效窗口及其有效步数，不要求所有窗口都成功通过。
- WM batch=2048 可以继续使用；建议按 domain×level 采样，并给覆盖组约 12.5–25% 的 batch 份额作为初值。既检查真实数据数，也检查实际采样数，避免反复过采极少的高等级序列。

最低日志：`collected transitions / sampled transitions / valid H8 steps / obstacle interaction windows / episode length / terminal fraction / stalled fraction`，全部按 domain×level 分开。交互指标仅作训练诊断，不变成 rolling/slip/contact 奖励。

### 3.3 Actor 保持能力附近的学习

Actor 起点以当前技能等级和低级复习为主；对“下一等级”开放小比例起点，依据该等级有效数据量、独立 WM 误差和真实短评测逐步增加，不能一开始把 L7 全部放入 Actor 想象。

WM 数据覆盖配额不会因 Actor 没晋级而取消；Actor 的晋级也不能仅因为存在高等级数据而发生。

需要关注：H16 仅覆盖 0.32s，vx=0.3 时约 9.6cm 的前进距离。保留用户指定 H16，通过合法的近障碍起点、持久命令及可靠 Critic 衔接延迟收益，不以扩大 H 或向 Actor 加高程代替解决。

## 4. 奖励与命令课程的优先调整

以下均为候选实验，不能一次全部改后归因。

**优先 1：停止门控。** 保持 stand_still=0.5，单独测试 `stop_command_sigma: 0.05 → 0.02`。在 0.06m/s 命令下，静止停止奖励约由 0.11846 降到 0.0000617。当前实现中 `stop_command_sigma` 同时控制停止门控和 axis-wise baseline 的 near-zero restore，直接改该参数会同时改变两处语义。若只测停止门控，需要显式分离这两个尺度并将 restore 保持原值；若沿用现有参数，应把实验标为“近零命令协议调整”，而非纯停止项消融。更合理的长期接口是明确记录“真实 stop 命令”，保持真实/想象一致。

**优先 2：按轴命令能力课程。** 正反向、横移正负、yaw 正负分别统计活跃命令下的 RMSE、完成时长和跌倒率。不能让大量零命令轴的高 tracking 得分触发扩速。楼梯先测试 vx 0.20–0.35m/s 的窄范围，保留小比例方向修正；稳定后恢复原 0.15–0.60。平地保留停止、倒退、横移与组合运动，不只训练前进。

**优先 3：横移跟踪尺度。** 单独测试 sigma_y 0.25→0.15，先不改 track_vy 权重。在 0.06 命令下理想与静止的奖励差约从 0.084 增到 0.222。尺度过窄会让大误差的指数梯度消失，需结合活跃命令幅度，不应无条件越小越好。左右转向分别审查，当前较快右转仍易跌倒。

**优先 4：研究停滞时的免费轴奖励。** 可对比“每轴 baseline，但仅在整个命令接近零时恢复静止正奖励”，避免前进受阻仍保留两个零轴的完整正奖励。此调整涉及回报尺度、负奖励与提前终止风险，需要同时审查 soft termination、Critic 和 max_step_reward，并更新奖励 contract；不能当作无行为影响的重构。

**暂缓大幅削弱腿部正则。** 当前楼梯 deviation 已降至 −0.008，不能只凭系数认为它阻碍抬腿。例如 12 个关节各偏离 0.25rad，仅该项约 −0.006/秒。末期 iteration=7760 的全域采样平均：leg deviation −0.0635、leg action rate −0.0129、leg torque −0.0326、wheel torque −0.00104；这些是主要平地分布，不能代表楼梯接触阶段。先记录台阶附近各项数值、关节动作饱和比例及 Actor 梯度贡献，再决定是否减小正则。

**保留现有地形平面对齐与按域约束作为基线。** 已有坡/楼梯的 vz 和腿偏差减权，没有证据需要再强行减到零。可后续单独测试楼梯对齐容忍区间或减权，不叠加强世界水平奖励。

不建议本轮添加 rolling consistency、wheel slip、wheel engagement、强制 gait/抬脚相位或奖励世界 Z 单调增加。它们或与当前任务边界冲突，或需要当前 41D WM 不直接提供的接触信息。

## 5. 不能忽略动作可达性

当前腿部动作经过 ±1 限制、scale=0.25，因此目标位置偏移最多 ±0.25rad。参考的 Loco 和 robot_lab 虽然同样写 scale=0.25，配置动作 clip 却为 ±100，**不等价于同样的可达动作范围**；这不代表其实际输出一定很大，只说明不能仅比较 scale。

先检查台阶接触时 12 腿动作、目标位置、实际位置、限位/限矩与轮速，做几何可达性和脚轮净空诊断。如果控制动作范围不足，再提出经过模型和稳定性检查的分关节范围消融；不能指望改奖励突破硬动作限制，也不能直接放大到开源 clip=100。

## 6. 课程晋级建议

当前每个环境分别要求连续 3 个合格成功 episode、2 次失败降级。随机命令和策略波动下，同一环境的连续证据可能非常稀少；最高 mastery 又控制 Actor 起点准入。

可考虑改为 domain×level 的滚动合格 episode 证据，保留现有无跌倒、有效前进、横向漂移和跟踪约束，使用上下阈值不同的迟滞及低级复习。例如至少 64 个合格过障 episode、成功率>70%、跌倒率<15%、连续两个窗口后推进，只是待验证初值，不能引用为某篇论文的标准。

评估晋级必须包含足够有效前进与过障证据，不能用不跌倒替代；WM 覆盖角色的失败不能直接把技能等级拖到最低，也不能靠其出生在高 Level 证明晋级。若更改课程状态结构，需明确 checkpoint 版本及迁移。

## 7. 推荐实验顺序与判定

1. **P0a：持续 WM 覆盖。** 保持奖励、Actor 起点准入、架构、H16、4096 环境和 batch2048；实现持续 512 环境配额及分域分级 WM 采样/诊断。检验末期 Replay 是否仍有 L1–7 有效接触序列，独立高等级 H1/H8/H16 误差是否下降。
2. **P0b：下一等级 Actor 起点。** 在 P0a 的基础上单独开放少量经过数据量与模型误差检查的下一等级起点，区分数据覆盖收益和 Actor 采样收益。
3. **P1：仅调整停止门控。** 使用相同采集配置、种子和训练预算，观察小命令横移/低速倒退，确认停止表现没有退化。
4. **P2：命令课程与 sigma_y 分别消融。** 不能同一轮同时改变两项并声称知道原因。
5. **P3：单台阶/出生距离课程与动作范围诊断。** 先解决 3/6cm 上台阶入口受阻，再评价更高 Level。高等级 WM 覆盖始终存在，不要求先宣称全部能过。
6. 有稳定改善后扩大多 seed、方向和速度评测，再决定长程训练，不以总奖励上升或 GPU 利用率作为主要通过标准。

WM 评估至少分 domain×level 和接触阶段报告：根位置、线速度、姿态/角速度、腿状态、轮速、高程序列变化。现有训练末期 Replay 的总体 H8/H16 误差主要反映低等级分布，不能代替高台阶留出评估。预测误差低也不能自动保证 Actor 使用的动作方向梯度正确，可再用少量真实短轨迹做方向性检查。

所有数据覆盖和课程变化需显式记录 checkpoint 协议；奖励变化更新 reward contract。普通 resume 不能静默绕过校验。若设计受限 warm-start，应分别核实 WM 动力学兼容性及 Actor/Critic 回报语义，不未经实现审计就声称旧 checkpoint 可直接继续。

## 8. 本次完成与未完成

已完成：三个开源 Go2W 代码对照；三篇相关论文全文与经典并行课程论文页面核查；训练快照/最新诊断/既有物理评测审阅；纯标准库奖励数值推导；本研究报告。

尚未完成：上述建议的代码实现、domain×level 实际配额测试、新奖励消融、WM 留出验证、新策略训练和通过率验证。本次没有运行训练、仿真、GPU 测试，没有修改源代码、checkpoint 或现有训练日志。

## 附录 A：当前奖励基线与状态协议

以下描述本地已训练快照的配置，而非已经实施的优化版本。奖励由真实转移与想象共享数学函数；单步总奖励为 `dt × sum(reward_rates)`，`dt=0.02`。

定义：`v_b=Rᵀv_world`，`ω_b=Rᵀω_world`，`u=(v_bx,v_by,ω_bz)`，命令为 `g`。`q` 仅包括 12 腿位置，`qd` 包括 12 腿速度和 4 轮速。`a` 为裁剪后的实际归一化动作。

按轴跟踪项的精确基线形式为：

```text
b_i = exp(-(g_i / sigma_i)^2)
c_i = exp(-(g_i / stop_command_sigma)^2)
r_track_i = w_i × [exp(-((u_i-g_i)/sigma_i)^2) - b_i × (1-c_i)]
```

|项|奖励率数学定义|当前权重或尺度|
|---|---|---|
|track_vx / track_vy / track_yaw|上面的按轴指数跟踪与 baseline|权重 2.5 / 1.5 / 2.0；sigma 0.25 / 0.25 / 0.30|
|stand_still|`w × exp(-sum((g/0.05)^2)) × exp(-norm(v_bxy)^2/0.15² - ω_bz²/0.30²)`|w=0.5|
|terrain_alignment|`w × exp(-max(0,1-z_body·n_terrain)/sigma)`|w=0.4，sigma=0.16，目标倾角上限 35°|
|base_lin_vel_z|`w_domain × v_world_z²`|按域变化，见下表；这里使用世界 Z 速度|
|base_ang_vel_xy|`w × norm(ω_bxy)^2`|−0.08|
|base_height_low|`w × relu(0.22-h_relative)^2`|−5.0；h_relative 参考当前位置地形|
|leg_joint_deviation|`w_domain × sum((q-q_default)^2)`|按域变化，见下表|
|leg_joint_limit|`w × sum(relu(q_min-q)^2 + relu(q-q_max)^2)`|−1.0；不包括轮角|
|leg_action_rate|`w × sum((a_leg-a_prev_leg)^2)`|−0.01|
|wheel_action_rate|`w × sum((a_wheel-a_prev_wheel)^2)`|−0.003|
|leg_torque / wheel_torque|各自的 `w × sum(tau²)`|−2e−5 / −1e−5|
|leg_joint_acceleration / wheel_joint_acceleration|各自的 `w × sum(((qd_next-qd_previous)/dt)^2)`|−2.5e−7 / −2.5e−9|

|领域|竖直速度权重|腿默认位置偏差权重|
|---|---:|---:|
|flat|−0.50|−0.030|
|rough|−0.35|−0.020|
|slope_up / slope_down|−0.15|−0.015|
|stairs_up / stairs_down|−0.08|−0.008|

力矩项在真实环境使用可用的实测力矩，在想象中使用分腿轮、经过限矩的 PD proxy：

```text
q_target_leg = q_default + 0.25 × a_leg
qd_target_wheel = 15.0 × a_wheel
tau_leg = clip(40 × (q_target_leg-q_leg) - 1 × qd_leg, leg_effort_limits)
tau_wheel = clip(0.5 × (qd_target_wheel-qd_wheel), wheel_effort_limits)
```

两条链路共享奖励定义，但实测力矩与 proxy 的控制子步/采样时刻可能不同，不能称为数值上完全等价。不能把轮速动作送进腿位置 PD，也不能对轮角施加位置偏差或限位奖励。

Actor 继续使用 5 帧 50D 本体观测和单独的 3D 命令，合计 253D 输入、16D 动作。WM 状态为 `p3 + quat4 + v3 + ω3 + q_leg12 + qd_leg12 + qd_wheel4 = 41D`，局部增量为 40D。轮子绝对角度不进入 WM。Structured WM 积分和每步动态 77D 高程查询是保留项；本次调研没有重新运行端到端梯度测试。

## 附录 B：现有八级难度映射

楼梯踏面宽度为 0.40m。粗糙地面一列是生成器配置的噪声上限，坡度为 rise/run，不是角度。碰撞高度场有分辨率、量化及插值影响，因此几何配置目标和测得高度不能无条件视为逐点相等。

|Level|台阶高度 cm|粗糙噪声上限 cm|坡度 %|
|---:|---:|---:|---:|
|0|0|0|0|
|1|3|1.000|5.0|
|2|6|1.333|7.5|
|3|9|1.667|10.0|
|4|12|2.000|12.5|
|5|15|2.333|15.0|
|6|18|2.667|17.5|
|7|21|3.000|20.0|

本报告不建议因为策略尚未学会，就降低碰撞真实性或将 21cm 成功作为课程配置的既成结果。

## 附录 C：解析数值复算

以下脚本仅依赖 Python 标准库，不导入任务、Torch 或物理引擎。用于复算解析式与配额合计，不能替代仿真或学习效果验证。

```python
import math

gain_y = 1.5 * (1 - math.exp(-(.06 / .25) ** 2))
stop_at_rest = .5 * math.exp(-(.06 / .05) ** 2)
stop_at_target = stop_at_rest * math.exp(-(.06 / .15) ** 2)
print("lateral tracking gain:", gain_y)              # 0.0839588
print("stop rate at rest:", stop_at_rest)            # 0.1184639
print("stop rate at target:", stop_at_target)        # 0.1009482
print("net gain of these two terms:",
      gain_y + stop_at_target - stop_at_rest)         # 0.0664431
print("stop rate with command sigma=.02:",
      .5 * math.exp(-(.06 / .02) ** 2))               # 0.0000617
print("tracking gain with sigma_y=.15:",
      1.5 * (1 - math.exp(-(.06 / .15) ** 2)))        # 0.2217843
print("stalled forward rate:", 1.5 + 2 + .4)         # 3.9
print("ideal forward rate:",
      3.9 + 2.5 * (1 - math.exp(-(.3 / .25) ** 2)))   # 5.8076806
assert sum([96, 96, 96, 80, 64, 48, 32]) == 512
assert 3584 + 512 == 4096
assert 16 * .02 == .32
assert 256 * .02 == 5.12
```

`stalled/ideal forward rate` 的注释值忽略零附近指数的微小残差和控制正则，描述理想化直立状态；不是实测总奖励。横移数值同样是在其他状态量相同下的局部解析比较。

## 附录 D：证据定位及交付范围

本仓库固定版本实现：

- [Go2W 奖励数学与默认权重](https://github.com/enffft/supermimic/blob/13b0861e95c0673cd721ab5260b957958b1a39ee/src/unisupermimic/tasks/velocity/go2w/reward_core.py)
- [WM/skill 阶段切换和 Actor 起点准入](https://github.com/enffft/supermimic/blob/13b0861e95c0673cd721ab5260b957958b1a39ee/src/unisupermimic/tasks/velocity/go2w/diagnostics.py)
- [八级地形生成器](https://github.com/enffft/supermimic/blob/13b0861e95c0673cd721ab5260b957958b1a39ee/src/unisupermimic/tasks/velocity/go2w/terrain.py)
- [Go2W 基础配置](https://github.com/enffft/supermimic/blob/13b0861e95c0673cd721ab5260b957958b1a39ee/src/unisupermimic/conf/supermimic/task/go2w_velocity/base.yaml)
- [Replay 序列与起点采样](https://github.com/enffft/supermimic/blob/13b0861e95c0673cd721ab5260b957958b1a39ee/src/unisupermimic/rl/algos/supermimic/replay.py)

训练和评测数字来自本机已有日志与报告，未将训练目录、checkpoint、Replay、TensorBoard 或原始评测轨迹提交 GitHub。外部读者可以核查引用的实现和复算解析值；本报告没有提供可独立重放全部训练结果的数据集。

报告提交只新增此 Markdown。未实施任何建议配置或任务代码变更，未重新运行训练、仿真、GPU、性能或通过率测试。下一次实验应另行实现并验证方案，不能把本报告的配额和阈值当作已验证参数。
