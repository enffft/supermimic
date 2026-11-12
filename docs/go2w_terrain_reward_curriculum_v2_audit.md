# Go2W 地形奖励与八级课程 v2 审计

日期：2026-11-12。分支 experiment/go2-velocity-unilab，基线提交 ded7bea06d21df07d9aab0bf82cc5a8eca129035。
PASS 仅表示所列有限检查通过，不表示策略已经具备越障能力或适合直接长训。

## A. 修改概览

|文件|作用|
|---|---|
|go2w/terrain_plane.py|固定 11×7 几何的加权最小二乘法向，缓存拟合矩阵|
|go2w/reward_core.py|以地形对齐替代世界 upright；六域竖直速度、腿偏差系数|
|go2w/mdp.py|共享法向/奖励；H×B 时间优先展开 Replay 标签；恢复配置校验|
|go2w/manager_terms.py|真实奖励读取采集前域标签、法向与日志复用；修复 ActionTerm 原先引用不存在的 _device；默认八级 sampler|
|go2w/terrain.py|3cm 离散增量、八行 atlas、独立粗糙/坡度映射、覆盖统计和课程恢复|
|go2w/env.py|v2 task/reward/curriculum 协议、fingerprint、任务迭代 hook|
|go2w/checkpoint.py|仅允许同阶段及 A→B→C 的窄字段迁移|
|go2w/diagnostics.py|六域×八级的设备端聚合，周期 JSONL、资源与计时|
|go2w_velocity/base.yaml、terrain.yaml|新奖励、八级覆盖、35% 楼梯；保留原训练 batch1024/WM batches10|
|rl/algos/supermimic/learner.py|可选的任务奖励元数据、迭代和恢复校验接口|
|tools/validate_go2w.py|单一规模的有界仿真/短训、独立 checkpoint 评估、显式阶段迁移|
|test_go2w_terrain_curriculum.py、test_go2w_physics.py|CPU 解析性质与单环境物理检查|
|docs/go2w_velocity_supermimic.md|标明旧 v1 文档，并指向本报告|

复用 Uni 的碰撞 HeightField、WM、Replay、sampler、技能晋降级和训练循环。
没有对应可选方法的 Go2/G1 保持原路径；未修改原 Go2 五级默认、G1 策略、机器人碰撞/质量或跌倒标准。
已有其他 Go2 楼梯实验修改保留且不提交。基础协议回归不等于所有旧任务训练回归。

## B. 完整 Reward 表

所有表项为 reward rate，最终仅乘一次 dt=.02。
v_b=Rᵀv，ω_b=Rᵀω；a 为实际裁剪且含采集噪声的动作，Δa=a-a_prev。
q 仅有 12 腿位置，qd 有腿12+轮4。

|项|rate 公式|权重/尺度|地形变化|
|---|---|---|---|
|track_vx|w·[exp(-(v_bx-gx)²/σ²)-baseline]|2.5，σ=.25|否|
|track_vy|同上，y 轴|1.5，σ=.25|否|
|track_yaw|同上，ω_bz|2，σ=.30|否|
|stand_still|w·exp(-Σ(g/.05)²)·exp(-‖v_bxy‖²/.15²-ω_bz²/.30²)|.5|全命令接近零时开启|
|terrain_alignment|w·exp(-(1-z_body·n_terrain)/σ)|.4，σ=.16，目标倾角≤35°|局部拟合平面|
|base_lin_vel_z|w_d·v_world,z²|见下表|是|
|base_ang_vel_xy|w·‖ω_bxy‖²|-.08|否|
|base_height_low|w·relu(.22-(p_z-H(p_xy)))²|-5|局部地面高度|
|leg_joint_deviation|w_d·Σ(q-q_default)²|见下表|是|
|leg_joint_limit|w·Σ[relu(q_min-q)²+relu(q-q_max)²]|-1|否，仅腿软限位|
|leg_action_rate|w·ΣΔa_leg²|-.01|否|
|wheel_action_rate|w·ΣΔa_wheel²|-.003|否|
|leg_torque|w·Στ_leg²|-2e-5|否|
|wheel_torque|w·Στ_wheel²|-1e-5|否|
|leg_joint_acceleration|w·Σ((qd_next-qd_prev)/dt)_leg²|-2.5e-7|否|
|wheel_joint_acceleration|同上，wheel|-2.5e-9|否|

baseline=exp(-(g_i/σ_i)²)·[1-exp(-(g_i/.05)²)]。
保留轴向零速 baseline；零命令时恢复完整跟踪奖励，非零命令不会因停止项额外奖励静止。
最大正 rate 仍为6.9、step上界 .138。

|domain|编号|竖直速度系数|腿偏差系数|
|---|---|---|---|
|flat|0|-.50|-.030|
|rough|1|-.35|-.020|
|slope_up|2|-.15|-.015|
|slope_down|3|-.15|-.015|
|stairs_up|4|-.08|-.008|
|stairs_down|5|-.08|-.008|

所有项均能在想象中用 Torch 对预测状态/动作求导。真实/想象复用同一数学。
真实力矩取最后物理子步 actuator-force；想象用预测端点重算同一饱和 PD，保留时间近似，
不等同于子步平均能耗。腿 τ=clip(40(q_target-q)-qd)，轮 τ=clip(.5(15a_wheel-qd_wheel))；
hip/thigh 上限23.7、calf45.43、轮15 Nm，控制器没有改动。

Reward 使用训练特权：机身真实线速度/位姿、地形、domain 和真实力矩；这些没有进入 Actor。
没有轮角、rolling/slip/engagement、接触真值或 gait 奖励。

### 拟合、稳定性与梯度

scan=p_z-H，因此用 h=-scan 拟合斜率；先去均值，设计矩阵仍含截距。
固定 X=[x,y,1]，高斯权重 exp(-.5[(x/.42)²+(y/.28)²])，斜率 ridge=1e-6、截距不正则。
小矩阵仅冷路径求解并按 device/dtype 缓存，每步只做乘法。
保留全部扫描跨度 x=-.42…+.78m、y=-.36…+.36m，中部权重更大，以表达多个踏面的整体趋势。
normalize(-a,-b,1) 限倾角35°，按当前 yaw 转世界；z_body=R[:,2]。
无效扫描回退世界向上；保留高度场边界 clamp 和出界终止。
没有法向 stop-gradient，也没有 detach WM 地形输入；边缘仍分片可微，倾角限幅有导数拐点。
不再叠加世界水平 upright。原有安全终止保留，σ=.16 允许运动中的姿态偏差。

## C. 完整课程与命令

|Level|楼梯高度 cm|粗糙配置上界 cm|坡度 rise/run|
|---|---:|---:|---:|
|0|0|0|0|
|1|3|1.000|.050|
|2|6|1.333|.075|
|3|9|1.667|.100|
|4|12|2.000|.125|
|5|15|2.333|.150|
|6|18|2.667|.175|
|7|21|3.000|.200|

粗糙噪声 0…上界，5mm量化，离散上界向下量化；坡面为原 pyramid 面，
在中心平台/横向变化附近，窗口拟合坡度会小于配置值。纯解析坡面另有精确法向测试。
踏面 .40m，平坦 border1.2m，出生点距边缘 .60m，检查足迹 .8×.5m。
atlas8×5、每块8×8m，XY分辨率.05m、高度量化.005m，外围border2m。
下坡/下楼保持 inverted；入口高度从编译碰撞字段查询。flat 域固定 row0 平坦块中心。
轮半径仍约8.6cm；21cm 是几何目标，不是能力声明，未放宽碰撞或跌倒标准。

六域顺序 flat/rough/slope_up/slope_down/stairs_up/stairs_down：

|阶段|比例|
|---|---|
|A flat|[1,0,0,0,0,0]|
|B mixed|[.50,.20,.15,.15,0,0]|
|C terrain / curriculum|[.25,.20,.10,.10,.175,.175]|

fixed_domains 保留最大余数舍入和 env_id%5 物理列约束。
32/128/512/1024/2048/4096 已验证配额误差小于1个环境、无列容量溢出。实际配额见下方补充表。

命令模式顺序 standing/forward/backward/lateral/turning/mixed：

|领域|模式概率|vx范围|vy范围|wz范围|
|---|---|---|---|---|
|flat|[.08,.20,.18,.18,.16,.20]|[-.25,.60]|[-.15,.15]|[-.35,.35]|
|rough|[.05,.55,.15,.075,.075,.10]|[-.20,.70]|[-.15,.15]|[-.35,.35]|
|slope_up/down|[.05,.75,0,.075,.075,.05]|[.10,.70]|[-.10,.10]|[-.25,.25]|
|stairs_up/down|[.05,.80,0,.075,.075,0]|[.15,.60]|[-.06,.06]|[-.15,.15]|

保留共享 sampler 保守楼梯模式，本次不添加可选5%组合模式，避免改变原 Go2 默认。
Level≥3 楼梯 vy/wz 上界再乘.65。纯轴模式把其他分量置零，不受正 vx 下界约束。
A阶段 flat 三速度档：vx[-.15,.30]/[-.20,.45]/[-.25,.60]，vy±.06/.10/.15，
wz±.20/.25/.35；flat_speed_level 初始0，需显式扩大，不等于 terrain mastery。
平地保持2–5秒，非平地10.02秒覆盖整个10秒episode。横移仍需腿轮协同，不假设麦克纳姆滚动。

### WM 覆盖与 Actor 技能分离

WM-only 默认前500 iterations：phase=wm，reset时 Level 分布
[.20,.25,.20,.15,.10,.06,.03,.01]，固定 flat 域始终Level0。最高两级合计4%。
WM期不晋升 mastery，也不把高难跌倒当成功。
real_transitions / real_valid_transitions / real_falls 按物理类型×八级累计，含 warmup/未完成episode，
随 curriculum checkpoint 保存。valid 仅表示下一状态未触发 hard termination，不能代替完整穿越。
已有 steps/episodes/falls/timeouts 支持每级episode时长、终止率。Replay按done屏蔽跨reset序列。

达到 wm_only_iters 自动进入skill，仅影响后续reset，不强制中断已有episode或改历史Replay标签。
Actor 起点只允许各域当前环境中已掌握最高级及以下，初始仅Level0；mask为6×8，
随Replay保存并每轮按mastery刷新。WM训练窗口仍可使用所有级别真实数据。

沿用 Uni 技能判据：合格穿越命令占episode≥90%，vx≥.15、|vy|≤.1、|wz|≤.2，
累计命令前进≥1m；无跌倒，实际进度≥1.2m且≥命令70%，横漂≤.6m，
xy RMSE<.35m/s、yaw RMSE<.4rad/s，连续3次合格成功晋一级；连续2次跌倒/严重停滞降一级。
15% reset复习较低Level，上限7。只在skill且terrain_level==mastery时更新晋降级streak。
这些判据证明入口区域穿越，不等于整块8m地形全程通过。

Episode domain固定，想象跨atlas域边界时惩罚仍用Replay起点标签；几何高程/法向继续动态更新。
未来连续地形可改为局部几何权重，本次没有扩展这个系统。

### 版本与阶段迁移

task schema=go2w_velocity_supermimic_v2；generator=go2w_eight_levels_v2；
reward=terrain_plane_alignment_domain_penalties_v2_no_target_detach；
curriculum=wm_coverage_then_skill_v2_episode_fixed_domain。
状态仍为 go2w_compact_state_v1，因为维度与排序未改。
旧Go2W v1和Go2 checkpoint均不能恢复为本任务，必须新建实验。
wm_structured、terrain flags、模型、关节/轮顺序、控制、奖励、历史、网络宽度保持严格校验；
Go2W额外拒绝恢复时改变wm_only_iters、wm_window、horizon。

普通CLI resume不能换阶段。显式工具入口为 --stage-from PATH --stage mixed/terrain，
只允许同阶段或A→B、B→C，以及domain_weights/flat_velocity_levels/flat_speed_level三个审核字段。
其余contract字段精确比较；不能关闭机器人/奖励/地形检查。
新阶段写新目录，保留Replay标签并关闭旧序列边界，按新配额reset；改变域的环境mastery清零。
来源checkpoint不覆盖。工具仅支持最多20轮有界验证。
正式阶段迁移复用现有 learner.load(source, allow_command_distribution_change=True)，在新目录训练。

## D. World Model 检查

Actor保留5×50本体历史+3命令=253输入、16输出；不读取高程/domain/接触/真实线速度。
WM状态41D、增量40D、输入134D=41不变特征+16动作+77扫描；无轮绝对角度。
模拟器qpos23/qvel22保留完整轮角；逻辑腿12+轮4具名映射不变。
Structured WM半隐式根位置、12腿位置积分及16速度预测保留。
H8/H32都通过原_wm_step按当前预测pose查询扫描；奖励用每个预测端点查询法向，
没有复用起点scan/normal或读取真实未来pose。terrain_sensor_grad=True保留。
实际H32更新已反传且通过有限数检查。CPU覆盖H32位置梯度、动态高程和奖励梯度。

独立 --mode eval --checkpoint 精确恢复v2，输出H1/H8/H32的机身位置、旋转矩阵分量RMSE、
线/角速度、腿位置/速度、轮速及扫描变化RMS。只计未跨reset序列；无有效序列明确SKIPPED。
目前用Replay内样本，不是held-out测试；不能据此判断泛化或可靠性。

## E. 测试结果与环境

用户指定supermimic环境缺unilab/unisim/hydra，仅用其解释器compileall。
实际验证使用现有go2_unilab环境，没有安装/升级依赖。OMP/MKL/Torch线程数1。
主机i5-14600KF、15.4GiB RAM、RTX4070 SUPER 12GiB。
开始可用RAM约5.5GiB、swap已用10GiB、GPU桌面约844MiB。未终止未知进程。
所有本次运行产物写 /tmp/go2w-v2-*，未改原有训练目录。

|检查|结果|范围|
|---|---|---|
|git diff --check|PASS|含暂存审查|
|compileall|PASS|src/unisupermimic，supermimic解释器|
|CPU unit tests|PASS|29项：Go2W+Go2基础协议+G1 encoded/71D布局|
|暂存版本独立验证|PASS|从Git index导出源文件快照（不是worktree），排除已有无关Go2修改；30项含单环境测试、32env H32短训通过|
|1-env MuJoCo|PASS|步进、16 actuator、悬空轮符号、限幅、reset|
|单环境几何专项|PASS|10%/20%坡向、3/12/21cm上下楼、编译碰撞字段|
|32-env短训|PASS|5轮，WM-only2轮，H8及H32各独立运行|
|128-env|PASS|12步采集；H32五轮短训|
|512-env|PASS|12步采集；H32五轮，batch16和64各一次|
|1024-env|PASS|12步采集；H32五轮短训|
|2048-env|PASS|12步采集；H32五轮短训|
|4096-env压力测试|PASS|12步采集；H32小batch五轮；补测batch256/512/1024，并以原batch1024、WM每轮10次、Replay256、Critic5×4再跑五轮|
|A→B→C|PASS|A五轮，B/C各追加一轮，严格恢复|
|独立checkpoint评估|PASS|无优化器更新，计数保持WM7/Actor5/Critic5|
|高台阶策略通过能力|SKIPPED|未训练出该能力，不报告成功率|
|大规模pytest/长期benchmark|SKIPPED|未执行|
|正式长程收敛实验|SKIPPED|未执行|

短训为生产网络宽度、WM window8、horizon32（首个额外检查用8）、hist5，
WM batches1、WM/Actor batch16、warmup40、buffer64、critic_epochs/minibatches1、
compile=false、amp=none、strict_finite=true。早期小batch五轮实验各完成5WM/3Actor/3Critic更新；原尺度补测完成50WM/3Actor/60次Critic minibatch更新。
普通单环境采集40步，其他规模采集12步，各控制步含4物理子步。

初次仿真发现并修复ActionTerm设备字段错误。物理fixture曾因在reset外写mask状态被拒绝，
改为配置悬空出生后正常reset；坡度断言曾在中心平台边缘取窗，改到完整坡面。
最终专项通过，不能把这些初次失败描述为无错误首次运行。
MuJoCo拼装有默认option冲突警告，但最终模型实测dt=.005、cone=1、impratio=100，
符合机器人源设置；另有既有TorchScript弃用警告，未使用torch.compile。

## F. 性能对比

首先列出batch16的链路基线，仅用于区分采集和网络开销，不能据此评价原训练配置。下方各表均来自短窗口，不是稳定benchmark。steps/s为控制步×环境数。
RAM为进程ru_maxrss，GPU为Torch max_memory_allocated，不包括桌面/驱动全部占用。
峰值覆盖warmup、更新、保存和评估；同步profiling有开销。
短采集含经理奖励/观测，不含网络更新；完整吞吐和更新时间取最后一轮，受reset/负载影响。

|envs|短采集 steps/s|完整 steps/s|WM ms|Actor ms|Critic ms|GPU MiB|RAM MiB|错误|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|32|未独立测量|373|26.8|109.3|2.7|264.2|1717.1|无OOM/非有限数值|
|128|8497|1550|23.4|102.6|2.6|266.6|1684.7|无OOM/非有限数值|
|512|11537|5049|23.5|104.4|3.8|280.1|1660.5|无OOM/非有限数值|
|1024|12101|8580|24.4|102.7|2.6|298.2|1742.8|无OOM/非有限数值|
|2048|13742|12769|23.2|100.7|2.6|331.5|1775.4|无OOM/非有限数值|
|4096|16285|16649|23.9|106.5|2.6|426.1|1821.3|无OOM/非有限数值|

512/batch64：GPU 406.6 MiB，RAM 1729.3 MiB，WM 25.2ms，Actor 106.5ms，完整 4962 steps/s。


批量规模复核后，4096环境的batch1024已实际验证，详见下一表。不能把早期batch16的低显存解释为正式训练资源。

### 按原配置补测大 batch

用户指出batch16不能代表原配置后，重新审阅基线：WM/Actor batch均为1024，WM每轮10次，Replay256步，warmup64，Critic5 epoch×4 minibatch，H8/H32。依次扩展如下，均在排除无关工作区修改的源码快照上执行：

|4096env配置|迭代数|WM/Actor batch|每轮WM次数|WM总耗时 ms/轮|Actor ms/轮|Critic ms/轮|完整steps/s|GPU MiB|RAM MiB|结果|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|scale4096-b256|3|256/256|1|24.2|144.4|8.6|14719|1030.0|1811.6|PASS|
|scale4096-b512|3|512/512|1|29.6|157.3|6.2|13721|1752.1|1796.9|PASS|
|scale4096-b1024|3|1024/1024|1|39.0|186.2|7.1|13350|3199.7|1857.0|PASS|
|scale4096-original|5|1024/1024|10|335.9|137.6|14.1|9666|3619.9|1962.8|PASS|

前三行buffer64/warmup40/Critic1×1，最后一行恢复原buffer256/warmup64/Critic5×4；均compile=false、amp=none，WM-only仅2轮以覆盖Actor更新。
最后一行共50次WM、3次Actor、60次Critic minibatch更新，无OOM或NaN/Inf，Torch峰值约3.54GiB；
这是原训练尺度的有限压力检查，仍不是wm_only500后的收敛或长期稳定性验证。

原规模每轮WM拟合约10×1024×8=81920个预测transition，4096env、utd=.4时平均采集10240个新transition/轮，约8次预测监督/新transition；
batch16且每轮1次时仅约.0125，二者的数据复用强度不同。因此用batch16的吞吐来选择正式训练配置不成立。
Actor batch1024对应每次32768个想象步，显存压力也显著更大。

## G. 问题与下一阶段建议

主要阻塞是模型质量尚未建立：随机初始策略产生大量短episode，部分H32抽样没有未跨reset序列，
已明确SKIPPED。有效H8/H32中仍有很大腿/轮速误差；可反传不等于可靠的Actor梯度。
没有18–21cm完整通过数据或通过率，不建议直接开展最终难度长训。

|用途|建议|证据与限制|
|---|---|---|
|最小验证|32env、5轮、wm_only2、WM/Actor batch16、WM batches1、H8/H32、buffer64|已实测；不是技能训练|
|有限资源开发候选|128–512env，WM/Actor batch256–512，WM batches按数据复用调整，H8/H32、buffer256；先A后B/C|4096下batch256/512短训通过；该环境数与更新频率组合仍需开发实验|
|高环境数正式实验候选|4096env、WM/Actor batch1024、WM batches10、buffer256、warmup64、H8/H32|原尺度五轮压力检查通过；WM-only500与长期学习效果尚未验证|

保留原默认32env、WM/Actor batch1024、WM batches10；环境数仍由显式覆盖选择。
batch16只用于最小链路测试，不应作为高环境数训练建议。此前临时降低默认至64的改动已撤回。
WM-only500保留为初值，应按每级有效transition、有效序列数和held-out H8/H32误差决定是否足够。
高env每轮新数据显著增多，不能用batch16只看吞吐；应联看WM更新覆盖、UTD、Replay数据复用。4096/batch1024/WM10具备本机短时运行证据，但仍需按模型精度选择是否长用。
先在A建立可靠停止和跟踪，再B，最后C，保留低级复习。高难快速跌倒比例需单独监控。

待验证长期候选：Actor lr由4e-4乘.25、ramp200轮、critic warmup20轮；
checkpoint每200轮、编号每1000轮、日志每20–50轮。这些不是已验证收敛参数。
需held-out H8持续改善、H32不发散、梯度有限且非饱和、低Level通过可靠后再开展较长实验。
18–21cm需额外评估抬腿余量、腿关节限位和力矩饱和，不能改碰撞/跌倒掩盖失败。

日志包含所有16项reward rate、域/级别速度RMSE与命令/实际均值、法向倾角、pitch/roll、
对齐误差、轮目标/实际均值、有效transition及终止统计。
按log_every输出当前迭代聚合值，非逐环境逐步CSV；累计覆盖统计含warmup。
记录同步WM/Actor/Critic/仿真耗时、实际env steps/s、更新计数、峰值资源；采集总时间还可读既有logger。
姿态角单位rad。法向与真实奖励复用同一扫描，不为日志再查高度场。

诊断需联看轮速误差/力矩饱和、机身速度误差、目标倾角与pitch/roll、相对高度及跌倒。
悬空符号已验证，地面滑移和轮腿协同未验证。
H1好而H8/H32坏应检查积分漂移和模型失配；scan_change_rms也受base_z影响，不是纯地形形状误差。
预测XY/yaw变化却scan不变时检查世界坐标对齐，不要从真实未来pose取扫描。

复现示例（都必须使用新目录，不覆盖正在运行的日志）：

~~~bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/yifeiy/miniconda3/envs/go2_unilab/bin/python
$PY -m pytest -q tests/tasks/test_go2w_compact.py tests/tasks/test_go2w_terrain_curriculum.py tests/tasks/test_go2_velocity.py
$PY -m pytest -q -m '' tests/tasks/test_go2w_physics.py
$PY tools/validate_go2w.py --mode train --envs 32 --horizon 32 --device cuda:0 --stage flat --output /tmp/NEW-go2w-A
$PY tools/validate_go2w.py --mode train --envs 32 --horizon 32 --device cuda:0 --iterations 1 --stage mixed --stage-from /tmp/NEW-go2w-A/last.pt --output /tmp/NEW-go2w-B
$PY tools/validate_go2w.py --mode eval --envs 32 --stage mixed --checkpoint /tmp/NEW-go2w-B/last.pt --device cuda:0 --output /tmp/NEW-go2w-eval
~~~

补测原尺度的有界复现命令：

~~~bash
$PY tools/validate_go2w.py --mode train --envs 4096 --device cuda:0 --horizon 32 --batch 1024 --wm-batches 10 --buffer-steps 256 --warmup-steps 64 --critic-epochs 5 --critic-minibatches 4 --iterations 5 --output /tmp/NEW-go2w-original-scale
~~~

以上是后续复现建议，不会自动扫描规模或开展长训。

## 补充：实际固定配额与运行摘要

|envs|flat|rough|slope_up|slope_down|stairs_up|stairs_down|
|---|---:|---:|---:|---:|---:|---:|
|32|8|6|3|3|6|6|
|128|32|26|13|13|22|22|
|512|128|102|51|51|90|90|
|1024|256|205|103|102|179|179|
|2048|512|410|205|205|358|358|
|4096|1024|819|410|409|717|717|
