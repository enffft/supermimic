"""Build the Chinese integration report from retained, completed evidence only."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(path):
    return json.loads((RESULTS / path).read_text())


def main():
    profiles = ["flat", "rough", "slope", "stairs", "slope_down", "stairs_down", "mixed"]
    names = {
        p: (
            "blind_terrain_" + p + "_random"
            if p in ("rough", "mixed")
            else "blind_terrain_final_" + p
        )
        for p in profiles
    }
    abnames = {
        p: ("blind_wm_ab_rough_random" if p == "rough" else "blind_wm_ab_final_" + p)
        for p in profiles
        if p not in ("flat", "mixed")
    }
    audits = {
        p: ("blind_wm_action_rough_random_" if p == "rough" else "blind_wm_action_audit_" + p + "_")
        for p in abnames
    }
    text = """# Go2 Blind SuperMimic 地形集成与训练前验收

工程已接通地形条件化 WM、可微动态查询、地形相对 Reward/Termination 和课程；**当前不具备正式长训条件**。物理、接口和反向传播测试通过，WM 动作梯度、Actor 更新稳定性及真实地形运动收益尚未验收。没有运行 100–300 iteration Actor 探索训练或数小时长训。

## 1. 范围、版本与审计差异矩阵

工作区 `/home/yifeiy/ws_go2_unilab/Uni-SuperMimic`，分支 `experiment/go2-velocity-unilab`，实现基于 `1cfffb6594b852512d30253993f00f75248e695b`。没有修改原 `/home/yifeiy/ws_supermimic`、旧 Checkpoint 或 UniLab 上游。先阅读上一轮 [迁移报告](go2_unilab_migration_report.md) 及参考仓库指定的 terrain/world_model/imagine/train/driver/terrains/mdp/state/test_terrain 文件。

参考 `Zhengtianjin/go2-supermimic@d9eee9b3f248e5300e636c433255db2c1c3e086a`。UniLab `794c0bbd67771ac48407e084c0164c0e841bfd97`。完整运行版本、源文件及旧权重完整性检查见 [provenance](../results/blind_terrain_provenance.json)。既有 `migration/versions.json` 保留其历史含义，本次以新 provenance 为准。

| 审计项 | 迁移基线 / 参考项目差异 | 本次实现 |
|---|---|---|
| State | 基线37D，p减出生origin；参考70D+独立planar pose | Flat原路径保留；地形显式使用完整世界p，保留R、v/w world、q/qd |
| Actor | 基线 Direct Goal；参考包含scan且任务有Chase/Dodge | 保持213D盲Actor，不引入参考网络或任务语义 |
| 关节/Context | 当前FL/FR/RL/RR、16D；参考FR/FL/RR/RL、4D | 保持当前命名顺序和Context；参考权重不兼容 |
| WM inputs | phi37+action12+context16=65 | 可选追加77维，变成142；原MLP和训练器 |
| WM step | 局部增量→世界位置、姿态积分 | 积分不变，每次step查询预测位姿地形 |
| H8 / H32 | 无地形查询 | 同一个 `_wm_step`，逐步重采样；不会冻结初始scan |
| Replay | s/s1、executed a、clean a_prev、xi、goal、done | 保存世界状态；静态全局地图+指纹，不保存过期scan/origin索引 |
| Reset窗口 | 自动reset正确；显式reset未关闭旧窗口 | 地形路径mark_reset_boundary，防止WM跨传送点监督 |
| Reward/终止 | 根高度及足部高度默认z=0 | root XY和四足各自XY查询地面；系数不变 |
| Scanner | 既有scanner只读真实机身附近 | 新纯Torch接口支持任意虚拟p/R；旧Perceptive接口保留 |
| 地形数据 | 参考raycast后双线性，可能平滑台阶 | 读取编译后公共hfield数组，精确匹配MuJoCo三角划分 |
| 门禁 | 原来全部拒绝terrain learner | 需要显式opt-in+已绑定可信地形MDP；mjwarp拒绝仍保留 |

## 2. 最终网络与物理协议

| 模块 | 输入 | 输出 / 网络 |
|---|---|---|
| Actor | 5帧×42本体观测+3速度命令=213 | 12个归一化关节位置目标；2048/1024/512/256、SiLU、tanh |
| WM A / Flat | phi(s)37+a12+normalized xi16=65 | 36D增量；2048/2048/2048、SiLU、LayerNorm |
| WM B | 上述65+77个base_z-ground(scan_xy)=142 | 同一36D增量和积分器 |
| Critic | 原phi37+xi16+Goal3=56 | 原256/128→标量；没有额外scan |
| Replay State | p3+quat4+v3+w3+q12+qd12=37 | 真实前后状态，地形模式p为world |

phi37为世界z、去yaw姿态6D、body线/角速度各3D、q12、qd12。原动力学是无滤波的即时PD位置目标，q/qd和实际执行动作足够描述该动作接口；本次没有擅自加入Previous Action到WM。Actor仍读取Previous **Clean** Action；采集噪声后的Executed Action进入物理和WM监督。

动作仍clip[-1,1]，`q_target=DEFAULT_Q+0.25*a`，再裁剪至原soft limits。KP=40、KD=1，effort=(23.7,23.7,35.55)×4，控制周期0.02 s、仿真0.005 s，4子步。DEFAULT_Q、执行器映射、XML和旧平地默认YAML均未修改。固定速度命令和原命令重采样/DR接口保留；课程没有放入Context16。

`critic_terrain_conditioning=False`为唯一已支持值，设置True明确报错。`terrain_sensor_grad`只控制WM scan的位姿梯度；Reward本身的地形几何导数仍保留。Critic已有的特权状态保持原结构，不将任何scan加入Actor。xi里的原编码器偏差仍仅用于重建有偏本体关节测量。

## 3. 高度场来源、坐标与可微性

通过公共 `env.get_playback_model(0)` 创建独立MjData，仅选择静态world-body的命名 `go2_terrain` geom。读取 `hfield_data/nrow/ncol/adr/size`、geom世界变换，复制为不可训练Torch缓存，排除机器人几何。不使用后端私有model/data，不重复生成另一份地形，不在H8/H32里做CPU raycast。公共 `get_terrain_spawn_data().terrain_origins` 仅提供tile出生位置。

以**编译后的碰撞几何**为准，包含PNG量化、坐标转换与真实顶点间距：`dx=2*size_x/(ncol-1)`、`dy=2*size_y/(nrow-1)`，不能直接拿generator的0.05 m当作碰撞顶点间距。每格沿00→11分成两三角，按MuJoCo 3.11.0源码做分段仿射插值。允许torch autograd对XY和yaw求导；三角边、网格边及接触切换处不保证光滑，不宣称那里具有唯一梯度。

缓存位于内存/GPU，指纹覆盖实际高度、尺寸、geom变换、seed、配置和tile origins。全atlas查询天然支持跨tile边界；Replay不依赖后来可变的出生origin。越出atlas时索引clamp保证有限，`contains()`另用于真实/想象终止，不把外推高度当成有效地形。当前只支持水平或yaw旋转hfield；pitch/roll geom明确拒绝。世界模型早期预测NaN仍会由训练有限值检查中止，不用缓存掩盖。

77点为X=-0.42…0.78、Y=-0.36…0.36，间距0.12 m，11×7，随yaw旋转。定义统一为 `base_world_z-ground_world_height`，Replay只存状态，由同一地图查询；没有混用relief、世界高度和相对高度。`sensor_grad=False`仅在查询入口detach预测p/R，正常默认True。

可视化示例：

![真实楼梯碰撞高度、77点扫描和沿线高度](../results/blind_terrain_final_stairs/height_scan.png)

![修正后的真实Random Rough局部扫描](../results/blind_terrain_rough_random/height_scan.png)

## 4. WM拟合、想象与权重兼容

H8从Replay窗口初始真实状态开始，后续输入为模型预测状态；每一步重新查询预测p/R，所以这是autoregressive fitting，不是把记录的真实扫描作为整段teacher forcing。真实s→s1仍是监督目标；done mask关闭跨episode窗口。

H32每步执行：盲Actor(history,Goal3)→从当前预测世界p/R查询77点→共享WM.step→同一地形相对Reward/soft termination→只用q、qd、body角速度、projected gravity、clean action重建42D→滑动5帧历史。下一步再查新地形。Goal3在想象中保持该Replay起点的命令，真实采集中按原周期连续重采样并记录。

旧Actor独立加载，与WM初始化分开。原长训Actor hash为 `0b3d8e4bb26173d13a6ce0b99d5eecf43005927a8ab2e01947c4d0311009487a`，加载检查机器人、关节顺序、控制/scale/输出及观测协议。没有加载旧enffft WM/optimizer，也没有加载参考70D WM。

同协议UniLab平地WM可 `initialize_from_flat`：只给首层追加77个零权重列，原参数、norm统计逐项复制，新scan norm为mean0/var1，原count保留。配置 `init_wm_from` 有元数据及已审计机器人模型检查；不兼容直接拒绝。源WM太旧/太弱不意味着适合热启动，本次主smoke使用新WM；A/B以同一随机原模型初始化，B零列扩展。部分先执行的Flat/Slope/Stairs smoke在Learner构造后由validator显式调用import_actor；这些目录的initialization.json记录构造时fresh，实际采集来源以training.json.actor及上述checkpoint manifest为准。后续rough_random、mixed_random、课程和Hydra运行已走构造内的init_actor_from，直接记录路径/SHA。两种方式加载的是同一已校验Actor，没有使用随机Actor采集却标记为旧策略。

标准terrain owner `wm_only_iters=500`，8轮默认只做WM；完整更新smoke才显式缩至2轮，**该缩短不适合策略迁移**。

"""
    expansion = read("blind_wm_expansion.json")
    actor = read("blind_flat_actor_equivalence.json")
    text += f"""实际full-width检查：23个WM原始state_dict张量按映射完全一致；首层新增列全0。对已有UniLab flat smoke WM（SHA `{expansion["sha256"]}`）测试64样本，最终State最大误差 `{expansion["max_state_error"]:.3g}`；首层矩阵运算最大误差 `{max(x["max_abs_error"] for x in expansion["per_layer"]):.3g}`，来自输入宽度改变后的浮点求和顺序，不能称bitwise等价。见 [扩展证据](../results/blind_wm_expansion.json)。旧Actor所有层和最终输出误差为0，本体观测转换误差 `{actor["observation_max_abs_error"]:.3g}`，见 [Actor证据](../results/blind_flat_actor_equivalence.json)。

## 5. Reward、Termination与Reset

机身高度惩罚和hard/soft termination使用 `base_z-ground(base_xy)`，保持0.22 m奖励阈值、0.17 m跌倒阈值和原tilt阈值/系数。四足clearance各自用 `foot_z-ground(foot_xy)-0.022`，slip/diagonal gait的几何接触代理同步改为相对高度。它们是可微几何代理，**不是测量接触力**。没有新增phase、运动参考、奖励门控或调节原速度Reward系数。

Real/imag共享 `velocity_reward_terms` 和同一heightfield数学定义。对相同真实转移、动作、goal和测量torque，recompute real reward与环境记录一致。真实torque仍来自jointactuatorfrc，想象用PD proxy，因此总Reward不是严格相同；各case绝对差记录在physics.json。当前真实和想象加速度都使用控制周期FD；原源环境的瞬时子步加速度测量与FD并不等价，这个历史差异没有消失，也没有把二者混为一谈。

8 cm台阶边缘测试覆盖根在两侧/中间、相同相对跌倒高度及四足分别查高度。单点root支持定义仍有局限：机身跨悬崖或多层/垂直墙时不能代表完整支撑面；本次低台阶没有新增footprint平均门控。正式高楼梯/悬崖训练前需结合支撑区域和真实躯干碰撞判据审计。

Reset在公共Entity写入事务中选择世界出生点，按**实际缓存高度**校正root z并清除动作/历史/OU状态。采集检测ep_step=0，将当前42D重复5帧；Replay窗口同时关闭显式Reset边界。各tile/部分环境Reset和速度命令快照已自动化验证。

## 6. 地形、课程与出生

继续复用UniLab `ROUGH_TERRAINS_CFG`、`STAIRS_TERRAINS_CFG` 和 `TerrainGeneratorCfg`，未修改上游。新opt-in atlas每类型一列、5个level行，8×8 m tile、1.2 m进入边界、2 m外边界；Mixed包括rough、slope up/down、stairs up/down。原有非课程profile默认形状不改。

| Level | Random Rough高度范围 | 坡度(正/倒) | 台阶高度(正/倒) |
|---|---|---|---|
|0|0|0|0|
|1|0–1 cm|5%|2 cm|
|2|0–2 cm|10%|4 cm|
|3|0–3 cm|15%|6 cm|
|4|0–4 cm|20%|8 cm|

Rough使用0.005 m量化随机采样，不能把上下界设为同一个值。发现初版把noise_range误当幅度参数、导致平台后，已修复并增加各level内部高度必须有随机变化的测试；**之前Rough/Mixed preliminary/final目录不用于验收**。本报告只引用 `rough_random` / `mixed_random` 及重新执行的消融结果。

独立 `flat` profile继续使用原flat.xml与plane，没有更改默认物理。atlas的level0是生成hfield上的平区域，不能宣称它与原plane逐位相同；尤其Mixed统一高度量化可能有毫米级世界z偏移，Reset和相对高度查询按编译几何校正。

出生选完整平坦边界，不用平台中央fallback。正pyramid从低边进入；inverted从外围高边进入凹陷，独立下坡/下楼验证。支持0/90/180/270°进入朝向，对应旋转出生边和机身yaw；非直角进入朝向明确报错。geometry quarter_turns与进入heading是两个明确参数。

每环境有terrain_level/type、mastery、episode_progress、commanded_progress、episode_fall、tracking_error、steps。按入口朝向的有符号位移≥3.5 m、非跌倒、平均XY跟踪误差<0.35 m/s且有足够前进命令，才可晋级；这是入口到中心附近的规定区域，不是完整8 m tile穿越。跌倒或有前进需求但250步仍推进<0.4 m可降级；普通timeout保持。零指令/原地转向不因站立被判“无法推进”，也不能靠静止高Reward晋级。15%回采严格较低等级，不丢mastery；低级回采成功不能越级认证更高mastery。所有选择仅在reset发生。

固定评估关闭课程和回采。32环境Mixed level0课程smoke通过，但尚未观察到旧平地Actor自然完成多级晋升；课程长期有效性没有验收。level4下楼16环境另做短物理检查，未据此开放高等级训练。

## 7. 自动化与Flat Regression

最终自动化：**124 passed，2 skipped**（两项仅MPS适用）。包括G1 MDP数值及encoded训练器/入口回归，不代表执行了G1全资产长训。保存 [完整输出](../results/blind_terrain_tests.txt)。Ruff及git diff --check另通过。

对审计基线commit直接加载旧实现，在CPU完整网络宽度、16样本上比较：Actor输出误差0、原65D WM H32 State误差0、所有Reward term误差0、hard/soft termination完全一致。原flat默认配置不变，旧v2 checkpoint contract去除新增但未启用的字段保持兼容。见 [Flat回归](../results/blind_flat_regression.json)。还实际恢复了旧UniLab full checkpoint并做8步真实采集和一次WM更新，legacy contract完全相等、Actor张量完全一致、更新有限，原文件SHA未变，见 [Resume证据](../results/blind_legacy_resume_verified/resume.json)。

测试包含碰撞ray oracle、矩形/旋转/负高度坐标、多tile独立查询、XY/yaw有限差分、四足/根高度、台阶边缘、Driver action contract、WM零列扩展、H8反传/H32动态scan、213D Actor、真实Reward重算、显式/自动/部分Reset、课程零命令与低等级回采。

## 8. 真实MuJoCo Smoke与Profiler

每profile：16环境，level1；5种固定命令（前进、后退、侧移、原地转向、零）各64步，另从进入方向1.3/2.0 m处做两个64步接触采集探针。正常出生仍在安全边界，后两者是受控drop/contact数据探针，不是课程出生点。随后共享learner执行8次WM更新、6次H32 Actor/Critic更新，WM Window8、batch均64；生产网络宽度未缩小。全部参数及Actor梯度有限，Reward重算一致，Replay非reset前后状态误差0。

| 地形 | ray误差上界(m) | simulation中位(ms) | WM fit中位(ms) | Actor BPTT中位(ms) | Critic中位(ms) | 采样进程GPU峰值(MiB) | Torch峰值allocated(MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for profile in profiles:
        folder = names[profile]
        g = read(folder + "/geometry.json")
        t = read(folder + "/training.json")
        r = read(folder + "/resource_usage.json")
        timing = t["timings"]
        ms = [
            timing[k]["median_ms"] for k in ("simulation", "wm_fit", "actor_bptt", "critic_update")
        ]
        text += (
            f"|{profile}|{g['max_collision_error_m']:.2g}|"
            + "|".join(f"{v:.2f}" for v in ms)
            + f"|{r['peak_process_gpu_mib']:.0f}|{t['peak_torch_allocated_bytes'] / 2**20:.1f}|\n"
        )
    cur = read("blind_curriculum_mixed_verified/training.json")
    res = read("blind_curriculum_mixed_verified/resource_usage.json")
    text += f"""
32环境Mixed课程smoke仍用相同batch64和网络，进程GPU采样峰值 {res["peak_process_gpu_mib"]:.0f} MiB，Torch allocated峰值 {cur["peak_torch_allocated_bytes"] / 2**20:.1f} MiB。这里是 bounded batch smoke，不能外推至1024/4096 batch或宣称训练加速。CPU环境、CUDA learner桥接；FP32（沿用既有TF32 matmul设置）、compile=False、OMP线程4。simulation范围只含env.step，采集观测/桥接的额外成本还在CSV collect时间中。

每次实际阶段有同步range计时；另外捕获一次真实simulation、wm_update、policy_update的CPU/CUDA operator trace及算子表。首个捕获有profiler冷启动开销，表列中位数，完整计数/均值/最大值及wall时间见各training.json。算子表随代码保存，巨大的trace、Replay和权重只留本机并由.gitignore排除。nvidia-smi采样峰值是100 ms采样下界，Torch allocator峰值另记allocated/reserved，不能当成所有大batch的显存上限。不是以理论FLOPs或不同batch作性能对比；A/B耗时仅为诊断资源记录，不作加速结论。

短测并非速度跟踪或复杂地形学习成功。普通命令段多数在入口，额外探针仅增加接触数据；没有完整路径成功率、多seed或长期跌倒率评估。

## 9. Actor范围与推理独立性

使用同一256组Replay历史/Goal比较加载的旧Actor和6次更新后的Actor；推理脚本完全不创建环境、不加载地图。tanh界限、有限值和213D contract通过，但早期WM上的Actor更新导致明显漂移：

|地形|旧Actor abs(action)>0.98|更新后比例|动作变化RMSE|
|---|---:|---:|---:|
"""
    for profile in profiles:
        a = read(names[profile] + "/actor_inference.json")
        text += f"|{profile}|{a['baseline_saturation_fraction']:.1%}|{a['adapted_saturation_fraction']:.1%}|{a['action_change_rmse']:.3f}|\n"
    text += """
因此**未通过“无严重饱和/更新稳定”的策略训练验收**。旧策略原文件未变，旧策略自身部分输出也接近tanh边界；不能把整个饱和比例都归因于迁移，但较大动作漂移足以阻止提前放开BPTT。正式迁移需更充分WM warmup和梯度检查。

## 10. WM预测误差与A/B消融

A不输入terrain、B输入77点；其余结构、Actor、Reward、数据、seed、batch64、Adam、Window8一致。B以A参数零列扩展（A=8,613,924参数，B=8,771,620参数；Actor=3,195,660，Critic=47,617），两者相同采样随机序列、各100次共享wm_update；训练6144真实转移、chronological holdout2048，均显式检查s1→下一s连续性。Actor完全不更新，无BC。数据来自旧Actor/少量探索和接触探针，很多并行环境高度相关，不能把条数当独立样本数。

同一数据holdout的H32摘要如下；所有H1/H8/H16/H32七类物理单位误差及有效样本数存于对应 `comparison.json`，不是混合归一化MSE。

|地形|A/B|H32关节位置RMSE(rad)|H32线速度RMSE(m/s)|H32位置RMSE(m)|H32姿态RMSE(rad)|
|---|---|---:|---:|---:|---:|
"""
    for profile, folder in abnames.items():
        d = read(folder + "/comparison.json")
        for r in d["reports"]:
            m = r["predictions"][-1]
            text += f"|{profile}|{r['mode']}|{m['joint_position_rmse_rad']:.4f}|{m['linear_velocity_rmse_m_s']:.4f}|{m['position_rmse_m']:.4f}|{m['attitude_rmse_rad']:.4f}|\n"
    text += """
没有一致的地形输入收益，不能声称真实Actor运动改善。独立窗口各horizon的done过滤和采样分布不同，H32误差偶尔小于H16不代表预测变好。为排除此误读，另外在相同真实40步探针轨迹、同一组H32-valid起点上比较各horizon：

|地形(B)|H|有效样本|q RMSE(rad)|qd RMSE(rad/s)|p RMSE(m)|v RMSE(m/s)|姿态RMSE(rad)|foot RMSE(m)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for profile, prefix in audits.items():
        d = read(prefix + "B/matched_predictions.json")
        for r in d["results"]:
            text += f"|{profile}|{r['horizon']}|{r['valid_samples']}|{r['joint_position_rmse_rad']:.4f}|{r['joint_velocity_rmse_rad_s']:.3f}|{r['position_rmse_m']:.4f}|{r['linear_velocity_rmse_m_s']:.4f}|{r['attitude_rmse_rad']:.4f}|{r['foot_position_rmse_m']:.4f}|\n"
    text += """
A的同轨迹指标也在相邻目录。这里的matched有效起点会排除跨Reset/跌倒片段，仍有生存选择偏差；不能替代失败接触事件诊断。Flat H1/8/16/32的分量结果单列于 `blind_terrain_final_flat/predictions.json`，是8更新smoke模型，不能与100更新模型混作同预算排名。

## 11. 动作梯度诊断：未通过

从公共physics snapshot的相同qpos/qvel出发，用独立MuJoCo model/data做4子步的a±epsilon，全部12维动作，epsilon=0.01/0.03/0.1。锚动作限制在[-0.8,0.8]，所有FD严格位于动作裁剪内部；避免把clip边界的导数变化归咎于WM。另以相同动作走一次live env.step核对oracle。局部位置0/1.2/2.0 m，分别覆盖进入支撑、地形边缘与内部坡面/台阶；每点40步真实采集。接触不连续与步长敏感性不能忽略，所有分组范数、符号相反比例、归一化误差和三种epsilon均保存。

以下只列热身100更新后的B模型、epsilon=0.03的关节速度响应；符号统计排除近零物理/模型响应，方向正确不等于幅值正确：

|地形|探针进入位移(m)|方向一致|符号相反|梯度归一化误差|物理梯度范数(rad/s/action)|WM梯度范数|oracle/live最大误差(混合状态单位)|
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for profile, prefix in audits.items():
        d = read(prefix + "B/action_gradients.json")
        for r in d:
            g = r["samples"][1]["groups"]["joint_velocity"]
            text += f"|{profile}|{r['entry_offset_m']:.1f}|{g['sign_agreement']:.1%}|{g['opposite_sign_fraction']:.1%}|{g['normalized_error']:.3f}|{g['physical_gradient_norm']:.3f}|{g['wm_gradient_norm']:.3f}|{r['oracle_live_max_error']:.2g}|\n"
    text += """
A同样检查，不存在“loss下降即梯度正确”的结论。有限真实探针未覆盖足够多接触模式、所有台阶边缘相位或不同seed；oracle只有状态设置/数值小误差验证，没有证明有限迭代contact solver的导数处处连续。这些误差与动作漂移是当前停止Actor探索训练、保留WM warmup的实质原因。

## 12. 后端能力矩阵与环境

|能力|MuJoCo 3.11 / UniSim2.7.1|mjwarp / UniSim2.7.1|
|---|---|---|
|原flat模型/盲Actor|本次回归通过|前次迁移平地验证保留，本次未重跑GPU flat物理|
|UniLab生成rough/slope/stairs/inverted/mixed|本次16环境短测通过|保留NotImplementedError，未删除门禁|
|真实机身height scanner|原公开接口保留|原后端不支持，明确拒绝|
|任意预测p/R静态高度查询|本次纯Torch可微缓存，CPU/CUDA均可|Torch算子能运行不等于真实GPU地形物理获支持|
|水平/yaw旋转hfield|ray对齐和梯度测试通过|没有进行地形支持宣称|
|倾斜geom/洞穴/多层/overhang|显式不支持倾斜geom，单值高度场无法描述多层|未验证|

Ubuntu20.04、RTX4070 SUPER 12GB、driver535.230.02，torch2.11.0+cu126、MuJoCo3.11.0、mujoco-warp3.11.0、Warp1.16.0、UniLab1.5.9、UniSim2.7.1、Python3.12.15；沿用隔离conda `go2_unilab`，无安装/升级原环境，pip check通过。CUDA runtime12.6与驱动标示CUDA12.2并存的实际可运行边界按已有隔离环境记录，不推断新GPU地形后端兼容。

## 13. 修改文件

|文件|原因|
|---|---|
|`tasks/velocity/go2/heightfield.py`(新增)|公共编译高度场、三角采样、世界坐标、77点scan和梯度|
|`tasks/velocity/go2/curriculum.py`(新增)|按episode真实进展/需求/失败更新、回采、固定等级|
|`go2/terrain.py`|正/倒地形、离散等级、随机Rough、旋转进入边|
|`go2/env.py`|绑定静态地图、验证出生足迹、版本化terrain contract，保留mjwarp检查|
|`go2/manager_terms.py`|世界状态、课程出生/度量、reset地面校正、Hydra字段|
|`go2/mdp.py`, `reward_core.py`, `gait.py`|同一地形相对root/foot Reward和终止；Flat分支保留|
|`go2/checkpoint.py`|Actor独立严格加载、安全PosixPath allowlist与SHA记录|
|`go2/config.py`, `conf/.../go2_velocity/terrain.yaml`|独立opt-in owner，标准500 WM-only warmup，默认Flat不变|
|`rl/.../networks.py`|可选77D输入、同UniLab WM零列扩展，原积分不变|
|`rl/.../learner.py`|共享H8/H32动态scan、初始化/Driver接口、显式门禁和resume检查|
|`rl/.../replay.py`|显式Reset边界方法，仅由terrain opt-in路径调用|
|`rl/.../driver.py`(新增)|未来Driver真实数据采集协议，校验关节/默认姿态/scale/PD/频率/坐标|
|`tests/tasks/test_go2_blind_terrain.py`, `tests/envs/test_go2_velocity_env.py`, `tests/rl/algos/test_supermimic_replay.py`|新增核心contract、几何、梯度、真实更新/Reset/课程兼容测试|
|`tools/validate_go2_blind_terrain.py`, `compare_go2_terrain_wm.py`|实际物理/完整共享训练器smoke、相同数据A/B和梯度诊断|
|`tools/check_go2_flat_regression.py`, `check_go2_wm_expansion.py`, `check_go2_legacy_resume.py`, `diagnose_go2_actor_range.py`, `check_go2_actor.py`|Flat/权重/无环境推理证据，旧Actor工具增加独立输出路径|
|`tools/build_go2_blind_terrain_report.py`, 本报告、精选results、gitignore/attributes|从真实输出生成报告，排除权重和raw trace|

`geometry.py`、`feet.py`、`contract.py`及机器人XML无需新增协议修改；feet已有正确世界FK，本次在调用方为每只脚查询地面。

Driver是可选外部adapter入口，默认原Actor+Gaussian/OU。没有引入PPO更新器、Teacher依赖或BC loss。外部PPO/MoE部署适配及稀有事件优先Replay未实现（P2），不能把协议测试说成真实Teacher已运行。

## 14. 已实际执行的命令

以下工作目录为实验仓库。所有结果均写到独立无时间目录，validator拒绝覆盖已有目录。重跑时应将output/log_dir替换成新的未使用名字；列出的路径是本次实际已执行并保存的目录。`PY`为本机隔离解释器。

```bash
export PY=/home/yifeiy/miniconda3/envs/go2_unilab/bin/python
export OMP_NUM_THREADS=4

# Flat/G1 shared-path、terrain、Replay、WM及盲Actor自动化
$PY -m pytest tests/tasks/test_go2_blind_terrain.py tests/tasks/test_go2_velocity.py tests/tasks/test_go2_terrain.py tests/envs/test_go2_velocity_env.py tests/rl/algos tests/scripts/test_train_supermimic.py tests/tasks/test_supermimic_mdp.py -q -rs
$PY tools/check_go2_actor.py --output results/blind_flat_actor_equivalence.json
$PY tools/check_go2_flat_regression.py --output results/blind_flat_regression.json
$PY tools/check_go2_wm_expansion.py --source results/profile_serial_mujoco/last.pt --output results/blind_wm_expansion.json
$PY tools/check_go2_legacy_resume.py --source results/profile_serial_mujoco/last.pt --output results/blind_legacy_resume_verified

# 地形生成/scan可视化
$PY tools/validate_go2_blind_terrain.py --mode visualize --profile mixed --level 2 --envs 16 --output results/blind_mixed_random_visualization

# Rough / Slope / Stairs短程完整更新，均8次WM、6次Actor/Critic；不是训练成功
$PY tools/validate_go2_blind_terrain.py --profile rough --level 1 --envs 16 --iterations 8 --steps 64 --batch 64 --device cuda --output results/blind_terrain_rough_random
$PY tools/validate_go2_blind_terrain.py --profile slope --level 1 --envs 16 --iterations 8 --steps 64 --batch 64 --device cuda --output results/blind_terrain_final_slope
$PY tools/validate_go2_blind_terrain.py --profile stairs --level 1 --envs 16 --iterations 8 --steps 64 --batch 64 --device cuda --output results/blind_terrain_final_stairs

# Blind Actor离线推理（无环境、无height map）
$PY tools/diagnose_go2_actor_range.py --source results/blind_terrain_rough_random --output results/blind_terrain_rough_random/actor_inference.json

# 同数据A/B：各100次WM-only，不更新Actor
$PY tools/compare_go2_terrain_wm.py --source results/blind_terrain_rough_random --updates 100 --batch 64 --output results/blind_wm_ab_rough_random

# 热身后模型的物理FD/autograd与matched H1/8/16/32
$PY tools/validate_go2_blind_terrain.py --mode diagnose --profile rough --level 1 --envs 16 --device cuda --wm-checkpoint results/blind_wm_ab_rough_random/B/wm_only.pt --output results/blind_wm_action_rough_random_B

# 课程链路与固定等级评估；后者没有WM/Actor更新
$PY tools/validate_go2_blind_terrain.py --profile mixed --level 0 --curriculum --envs 32 --iterations 8 --wm-only 2 --steps 64 --batch 64 --device cuda --output results/blind_curriculum_mixed_verified
$PY tools/validate_go2_blind_terrain.py --mode evaluate --profile stairs_down --level 1 --envs 32 --steps 64 --device cuda --output results/blind_fixed_stairs_down_eval
$PY tools/validate_go2_blind_terrain.py --mode evaluate --profile stairs_down --level 4 --envs 16 --steps 64 --device cuda --output results/blind_stairs_down_upper_eval

# 真正Hydra训练入口：默认安全门限，8轮内冻结Actor，仅WM warmup
$PY -m unisupermimic.scripts.train_supermimic task=go2_velocity/terrain algo.num_envs=16 algo.wm_batch=64 algo.pi_batch=64 algo.iterations=8 algo.wm_only_iters=500 algo.log_backend=none env.commands.velocity.terrain_profile=mixed training.log_dir=results/blind_hydra_terrain_verified +algo.init_actor_from=/home/yifeiy/ws_supermimic/logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt
```

另外已执行Flat、倒坡、下楼、Mixed对应同规模smoke；全部参数/命令见各目录invocation.json，Hydra精确配置见run_config.json。诊断A模式使用 `--no-conditioning`，数据及其他设置相同。未提供未经验证的大batch或长程训练命令。

## 15. 最终结论、阻塞与下一步

|验收问题|结论|
|---|---|
|Blind Actor是否保持原输入|是，213D、Action12，逐层旧权重等价；无新encoder/phase|
|WM是否读到真实地形|是，编译后的实际静态碰撞hfield，不是另一份随机地图|
|H32是否动态查询|是，逐步预测世界p/R；H8同样动态查询|
|有无高度信息泄漏|没有直接输入/历史/Context/Goal泄漏；训练时WM动力学梯度使用地形是设计本意|
|Real/imag地形Reward是否一致|地形数学一致；measured torque vs PD proxy仍不同，历史瞬时加速度与FD差异单列|
|旧Flat Actor/默认链路是否正常|前向、Flat H32/Reward/终止回归通过，旧文件未改；不宣称新模型训练轨迹等价|
|MuJoCo复杂地形链路是否打通|生成/物理/Reset/Replay/WM/Actor/Critic短程工程链路通过|
|正式长训条件是否具备|否：动作梯度仍偏差、Actor过早更新明显漂移，A/B收益不一致|

下一步先保持旧Actor冻结、增加低等级实际接触数据的多样性和WM warmup预算，在 `learner._wm_rollout` / shared `wm_update` 下重做多步与物理动作响应诊断；结合Window8损失之外的真实接触覆盖审查，不能仅靠归一化loss下降放行。再定义多seed/固定路径、速度误差、跌倒率、对角动作和梯度方向/幅值的数值验收门限，通过后才开展100–300轮探索。原Actor和所有旧权重保留，当前smoke生成的Actor不可当新基线覆盖。

仍未完成：正式复杂地形运动收益与完整路径成功率验收；可靠动作梯度/足够WM warmup；自动课程真实多级晋升；更高楼梯/支撑面跨边缘碰撞终止扩展；Teacher/PPO adapter真实部署与稀有事件Replay；地形Critic；GPU terrain物理/height scanner支持、compile/大batch扩展以及跨seed统计。单值高度场不能表示洞穴、多层和垂直墙。保留这些边界，不用smoke冒充训练成功。
"""
    (ROOT / "docs/go2_blind_supermimic_terrain_integration.md").write_text(text)
    print("report generated from completed evidence")


if __name__ == "__main__":
    main()
