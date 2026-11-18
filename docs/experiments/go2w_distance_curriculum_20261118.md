# Go2W：每环境距离课程、八地形采样与额外3.5小时续训

## 状态与范围

2026-11-18，在 `experiment/go2w-unified-curriculum` 当前工作树原地修改。起始提交 `c1724f62fd06219a8ee6a05da6013e7b74a2a7e6`。没有切换分支、另建工作树或覆盖历史运行。本次用户明确要求完成后额外训练3.5小时，已满足任务文档“未经另行要求不开长训”的授权条件。

**已完成实现及短时验证；本报告初稿时正式续训尚未启动。** 后续启动记录追加在末尾；训练结束后的实际物理结果由独立的 `go2w_distance_20261118_results.md` 报告保存。不能把课程晋级、短时loss或预期改善当成最终物理能力。

保持 SuperMimic 单Actor/WM/Critic、4096环境、H8、WM window8、Replay1024步/环境、WM/Actor batch2048、历史池8192和recent/historical 50/50。没有修改21项reward、PD/action scale、网络结构、地形碰撞或严格物理成功定义。Actor仍为50维观测×5历史和7维命令；没有加入真实等级、目标位置、接触力或高度图。

## 关键问题与调用链

三个旧问题均已移出生产路径：全局skill等级控制MIX导致平地锁；`skill>=5`把rough/slope错当yaw技能；Actor按全局mastery封锁coverage高等级。

实际顺序：动作前保存命令/元数据 → 物理步 → termination flags → reward中积分当步命令和真实运动、更新宽松课程越障与严格成功 → 自动reset → command.reset在清空episode前结算课程 → 新命令/新地形 → Replay保存物理转移。UniLab的自动reset也调用公开`reset()`，必须检查其`_autoreset_reset_active`标记；手动reset才清除合法结束标记。

新增501步真实物理集成测试最初捕获了这个生命周期错误：不区分自动/手动reset会使所有课程结算失效。失败日志保留，修复后26项测试通过。没有把无episode结束的短时smoke误称为课程升级证明。

## 改动文件

|文件|核心变化|
|---|---|
|`curriculum.py`|删除窗口/hysteresis/MIX解锁；固定domain、per-env mastery与七项平地技能、逐步有符号积分、几何越障锁存、版本化状态|
|`manager_terms.py`|使用每环境平地技能缩放命令并重新计算mode；坡地/楼梯采样小vy/wz前进；reward传入动作前traversal元数据|
|`env.py`|终止标记与合法自动reset衔接，手动reset排除；更新课程协议标识|
|`replay.py`|可选domain→command分层，先过滤再定额；去重候选、记录shortage/重复率、reset/target边界padding；WM只增诊断|
|`learner.py`|新增可选actor_domain_weights并接入精确2048起点采样；其他任务默认None|
|`diagnostics.py`|记录Actor候选/实际数量及domain×level，WM recent/history实际domain×level|
|`train.yaml`|唯一生产配置增加Actor地形权重和固定物理配额，清理window/hysteresis/未消费lookahead|
|`migration.py`（新增）|显式旧课程→新课程迁移，严格保护奖励/控制/几何及训练参数，保留网络/优化器/真实Replay/历史池|
|`tools/train_go2w.py`|增加显式迁移入口、完整恢复哈希审计和额外有效秒预算；保存当前真实源码快照|
|`tools/finalize_go2w_distance.py`（新增）|等训练结束后执行847条件物理评测、21项reward、视频、独立结果报告并推送|
|`test_go2w_distance_curriculum.py`（新增）、`test_go2w_unified.py`|有符号进度、独立晋级、几何反例、quota、Replay边界、迁移拒绝、真实自动reset集成测试|

## 距离课程规则

每环境固定domain；普通环境有独立mastery（0..9，展示Level1..10），每次合法episode结束最多±1；一次合格成功即可晋级，没有窗口或连续两次要求。极短episode（少于25步）不更新。coverage固定等级且不参与晋级统计。20% review从0..当前上限抽取；低级review成功不会提升高级mastery，也不会以低级失败降高级。全局14项skill_levels只作汇总诊断。

- 平地平移：按当步命令方向积分 `Σ dot(v_body, unit(v_cmd,xy))*.02`，期望量 `Σ||v_cmd,xy||*.02`。期望≥.4m且实际≥65%时+1，低于25%才-1，中间维持。正/负vx、正/负vy分别记账。
- 原地yaw：积分带命令符号的真实wz，期望量积分abs(wz_cmd)，最低.4rad；60%晋级、20%降级；不要求平移。
- mixed及中途换命令：每步积分对应命令及运动，不使用末帧命令倒推全程。混合平移/旋转中达到最低期望量的分量均需通过，不给错误方向奖励。
- stop：累计≥5s、无跌倒、线速度RMS<.15m/s、yaw RMS<.35rad/s可晋级；超过两倍误差界才降。初始化/手动reset不能认证存活。
- rough：沿命令方向的有符号平移，60%/20%，晋级排除跌倒；没有yaw进度要求。
- slope：使用真实世界+X位移增量，要求整个episode都是有效穿越命令（vx>.08、abs(vy)<.08、abs(wz)<.15）且未出横向通道；60%/20%。非穿越episode维持，仍用于RL。实际命令主要为vx=.20..35、vy=±.02、wz=±.06。
- 平地七种技能用每环境7元素小tensor独立记账，不能由一项最高级解锁其他环境或地形。所有新状态和RNG均序列化；旧课程schema直接加载会拒绝。

## 台阶：真实立面和宽松课程越障

`terrain.py`/heightfield碰撞与`traversal.reset_targets`保持原样：相对spawn，单级最后立面在+.8m、连续楼梯最后立面在+2.0m。目标落地区地面高度由真实heightfield查询，元数据保存地面和目标基座高度（地面+.36）。

|domain|最后立面距spawn|基座越立面余量|晋级所需最小安全前进距离|
|---|---:|---:|---:|
|single_step_up|.80m|.25m|1.05m|
|single_step_down|.80m|.15m|.95m|
|multi_stairs_up|2.00m|.30m|2.30m|
|multi_stairs_down|2.00m|.20m|2.20m|

要求命令前向、始终在spawn横向±.5m通道内、基座竖直分量>.65、此前及当前未跌倒；基座高度距落地区地面+.36m不超过.16m，且高于落地区地面.22m。因此只到立面前、侧绕、错误高度、倒退命令及先跌倒都不能假晋级。达到时立即锁存`curriculum_crossed_obstacle`；先越障后跌倒不撤销已经取得的课程资格。没有四轮同时接触或十帧稳定要求。未越障且最大安全前进小于立面距离20%才降，达到一定进度则维持，Level1下限不会产生额外降级副作用。

**严格评测仍使用原四轮越最后立面、支撑/姿态/高度及10步条件。宽松课程越障与严格成功分别记录。** 四种domain各有独立环境和升降统计。Level1..10单级高度仍为3/4/5/7/9/12/15/18/20/22cm，连续楼梯每级高度仍为3/3/3/3/5/6/8/10/12/15cm。

## 实际配额：物理环境与Actor分别控制

采用largest-remainder整数配额。coverage为512环境，已包含在4096总数里，只在复杂域分配，覆盖各域全部10级；普通复杂环境从Level1开始。固定domain使课程尚未进阶时仍持续产生复杂地形数据。

|domain|物理目标|实际环境数|实际物理转移比例|Actor目标|每批实际Actor数量|
|---|---:|---:|---:|---:|---:|
|flat|40%|1638|39.9902%|40%|819|
|rough|12%|491|11.9873%|10%|205|
|slope_up|8%|328|8.0078%|8%|164|
|slope_down|8%|328|8.0078%|8%|164|
|single_step_up|12%|491|11.9873%|12%|245|
|single_step_down|8%|328|8.0078%|10%|205|
|multi_stairs_up|7%|287|7.0068%|8%|164|
|multi_stairs_down|5%|205|5.0049%|4%|82|
|合计|100%|4096|100%|100%|2048|

Actor先在recent ring取65536个随机候选ID并去重，只gather候选，不复制全量Replay；排除done/term、非法domain/level/mode、非有限状态/goal及与domain不符的4D onehot，再检查实际H8 imagination有效性。对有候选的domain归一化定额；每域内部再按原六种command模式抽样，某域模式权重均零时仅在该域真实存在类别中fallback并记录。最后打乱，避免后续截断破坏地形比例。

缺域、候选不足和重复率均记录，不能默默把全部预算转给flat。候选池是有界随机样本，极稀少地形有可能本次未命中，这也会显示missing/shortage。统一start_scope开放全部8×10，仍只允许真实Replay有效数据。历史5帧保持oldest-first；episode、reset及target_id边界以前使用padding，不跨episode拼假历史。

WM recent/historical仍每次各1024，未改变历史池插入、保留和抽样策略。4096最终smoke实际窗口起点计数如下（每行合计102400）：

|来源|flat|rough|slope up|slope down|single up|single down|multi up|multi down|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|recent|10946|21556|21965|22618|6215|8574|3264|7262|
|historical|8909|24850|27291|27656|3307|4321|1785|4281|

WM按原bucket均衡策略采样，所以其domain比例不等于物理或Actor目标。迁移初期recent ring保留旧物理轨迹，随1024步循环逐渐替换；没有把旧平地轨迹改标签。

## 验证与资源

证据：[验证目录](../validation/go2w_distance_20261118/)。129项相关回归通过、1项skip、1项slow deselect；另显式执行全部距离课程测试（包含501步MuJoCo集成）26项通过。两组存在重叠，不能相加作独立测试总数。覆盖符号/命令变化、一级升降和边界、停止、坡地非穿越、四类楼梯反例、4096配额及coverage、Actor高等级/缺域、历史padding、版本化checkpoint和奖励/控制变更拒绝。旧Go2/G1通用learner/replay默认路径回归通过。ruff及git diff whitespace检查通过。

|验证|128环境最终smoke|4096环境最终smoke|
|---|---:|---:|
|真实更新轮数|30|10|
|新增WM/Actor/Critic更新|300/30/600|100/10/200|
|Replay步数/WM与Actor batch/H|1024/2048/H8|1024/2048/H8|
|有效训练循环|约21.71秒|12.890秒|
|新增转移（含128 warmup）|17792|102400|
|转移/有效循环秒|819.5（含warmup，不能作为纯循环吞吐）|7944.2|
|PyTorch峰值GPU|约1749MiB|4061.91MiB|
|进程峰值RSS|约1.9GiB|5145.04MiB|
|Actor grad norm范围|8.791..45.816|6.278..13.867|
|Actor每批最大重复率|13.82%|3.08%|

4096还执行过30轮资源测试：38.001秒、307200真实转移、约8084转移/s、峰值4061.88MiB。该次在自动reset修复前，作为资源测量保留，不拿它代替最终逻辑验证。最终4096测试内可用RAM最低7229MiB，reserved GPU约4140MiB。GPU整卡在一次30轮测试采样时为6270MiB（含桌面/其他进程，非峰值）；PyTorch峰值不含MuJoCo/桌面全部分配。没有减少4096环境、Replay步数或batch。

最终128 smoke累计Actor各域 `[24570,6150,4920,4920,7350,6150,4920,2460]`，flat=39.9902%；WM累计recent/historical=`307200/307200`。最终4096同样达到精确批配额，十轮累计`[8190,2050,1640,1640,2450,2050,1640,820]`。所有域都有有效候选，无缺域或候选不足。课程高等级coverage已实际用于Actor（完整domain×level见JSON）。

smoke没有证明长训后能力改善；短smoke没有完整任务成绩。完整断点迁移验证不靠loss：Actor/Critic/WM（含normalizers）、三优化器、value normalization、历史bank全部数据和RNG、iteration/time/transitions/update counters逐项摘要相等。最终4096保留1024步recent ring，历史池8192；128 smoke因N变化显式清空recent后补采，历史池保留。这项smoke专用变化不能用于正式4096运行。

后台finalizer以已有847条真实评测文件做过编排dry-run，评测/渲染/git动作被mock；只证明结果处理流程可执行，**不是本次模型的新评测成绩**。

## 迁移与3.5小时预算

父断点固定为最近完整训练状态：`/home/yifeiy/ws_go2_unilab/runs/go2w_unified_h8_resume_3p5h_20261117/last.pt`，iteration21190，累计有效循环27001.998728秒。没有擅自改用19796轻量best模型。父模型已出现能力退化，旧报告记录平地约33.8%、台阶/楼梯为0；本次是否恢复必须由后续严格评测判断。

正式迁移保留三网络/全部normalizers、三优化器、价值归一化、iteration/更新/真实转移/时间、recent ring及历史bank/RNG；显式重置per-env课程、固定domain/coverage分工、命令采样器RNG、采样诊断和旧best-selection分数。物理环境重新采样，旧Replay末端关闭reset边界；保留单调命令计数避免新旧target_id碰撞。不是step-equivalent恢复，也不是从零训练。

使用`--resume <完整父last.pt> --migrate-curriculum`，仅允许该次课程/采样代码文件及actor_domain_weights变化，其他dynamics、reward、terrain fingerprint、控制和训练超参数不符即拒绝；没有删除来源哈希检查、伪造旧SHA或覆盖原run。新目录保存`migration_initial.pt`、迁移JSON、恢复审计、完整源码tar/manifest、Git信息、effective_algo、resolved_config和task_contract。

原始WM资产`artifacts/world_models/go2w_conditioned.pt`保持只读，SHA=`3c3c4d1c5e70bb12c8cc68274fda39003e465c6ec1fbe745c0729f9f7d028610`。本次继续父断点已训练后的WM，不能再用原始WM覆盖；从未使用`calibrated_candidate.pt`。旧200次校准失败证据和旧报告继续保留。方向检查继续如实标为诊断失败，按用户既有明确授权使用`--accept-direction-diagnostic`，不伪造通过。

本次新增12600秒**有效训练循环**，目标累计39601.998728秒；启动前迁移、评测、保存/渲染等时间不计入。每约1800秒有效循环做物理和WM检查；结束后自动进行同协议847条评测、21项reward、视频和结果报告推送。完整结束报告是后续产物，运行中的状态见`progress.json`，异常见`failure.json`/`finalization_failure.json`。

### 可复现命令

正式启动由后台进程执行，完整命令保存在运行launch记录。后续已经迁移的新完整断点继续训练时用普通严格`--resume`，不要重复使用只接受旧schema的`--migrate-curriculum`。

当前阶段播放正式运行的初始策略（与21190父Actor相同，任务协议已显式迁移），正式初评生成后可执行：

```bash
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic-unified
PYTHONPATH=src /home/yifeiy/miniconda3/envs/go2_unilab/bin/python tools/play_go2w.py \
  --checkpoint /home/yifeiy/ws_go2_unilab/runs/go2w_distance_resume_3p5h_20261118/evaluation/initial.pt \
  --demo
```

`--demo`包含停止、正负vx/vy/yaw及代表性单级/连续上下台阶；有桌面时打开MuJoCo窗口。单项可替换`--demo`为`--terrain single_step_down --level 5 --vx .3`。无人值守验证加`--headless --no-realtime`。训练中间策略位于`evaluation/iteration_*.pt`，结束后用`evaluation/final.pt`；不要把initial播放当成续训后能力。
