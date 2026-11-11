# Go2 楼梯全向续训：保持原 Reward

本轮审计起点 `26015c1768d9addaea7449d0b32af25bd0d989a3`。正式来源为 `results/omnidirectional_terrain/stageB/last.pt`，SHA256 `52ebc6fdf1e3d575733756939645e4baec1e26c3be4f010892655275ffbdf8d9`；未使用规模 benchmark 的临时权重。

本文件先记录已验证的修改与准入边界，训练终点、独立物理评测和结论将在运行完成后补齐，不将正在运行写成训练成功。

- Actor 保持盲 213→12；WM 保持原网络及 77D 地形、H8；想象保持 H32 动态地形查询；Critic、PD、动作和 dt 均不变。
- Reward、Termination 的数学实现与审计 HEAD 相同。逐文件/函数哈希见 `validation/go2_stairs_omni/reward_lock.json`；clearance 系数0、目标.05m、sigma.025m，对角步态-.15、足滑-.20。
- 独立配置 `stairs_omni_continuation.yaml`，旧默认值不动。S0/S1/S2/S3横向上限.08/.15/.22/.30m/s，yaw上限.15/.30/.45/.60rad/s；按上下楼及Terrain Level分别记录，独立于地形mastery。
- 六种楼梯模式为纯前进30%、前进修正25%、全向混合20%、纯侧移10%、纯转向10%、站立5%。Replay保留原六类模式，将修正/全向组合计入Mixed；纯前进及小修正可参加穿越，其余不得晋降地形技能。
- 穿越命令保持10.02s。机动保持2–4s，在现有楼梯入口前移1.6m受控Reset，真实地面高度上方.42m出生；物理几何不变，不把接触探针/机动Reset当穿越成功。机动中途改采前进命令也不能改变episode身份。
- 2048环境审计1500个vector step，3,072,000条真实转移，Actor参数差0；楼梯3D同时非零358,975条，stairs768,000条，非平地stairs743,976条。六域真实占比35.0098/19.9707/10.0098/10.0098/12.5/12.5%。
- 初始200轮后接触Jacobian未通过全域准入，续做至2000轮、8000次WM更新；Actor/Critic仍冻结。旧全局历史门禁未改。部分楼梯pair已全部达到80% qd符号一致率，另一些仍约70–75%。因此新增显式的想象起点范围筛选：WM使用全部真实Replay，Actor/Critic仅从通过80%逐探针及H32预测门槛的楼梯pair采样。旧平地、rough、slope训练范围保留。被拒绝pair仍采物理数据训练WM，不能通过更改门禁强行解冻。
- 更高S等级必须同时通过当前及候选范围的固定物理评测、全部七类动作探针和H32预测；不按iteration开放。全域scalar失败记录仍保留，分区准入不等于全域模型合格。
- 平地保留六模式；本轮中间范围先扩至vy±.20/wz±.50，已由旧策略独立评测验证。候选±.30/±.65需要再验收；旧策略vy=-.30仍会跌倒，不能直接承诺S3。

已经运行的测试：首次216通过，新增恢复/准入/范围过滤后219通过、2 skipped、9 deselected；Ruff及diff检查通过。真实源Checkpoint恢复保存网络/优化器/Replay/课程/value norm/RNG/累计计数，明确新物理Reset，非逐物理步等价恢复。旧Replay有原始命令标签时保留；仅缺失标签的历史数据才由Goal和tile推断。

运行目录 `results/stairs_omni_continuation_v2/`。初始诊断调用存在输出路径判断错误，已修正，从 `interrupted.pt` 恢复800次已完成WM更新及Replay。随后全域预热没有放行Actor。受控分区训练将使用该正式预热模型继续，不重置参数；其预算与已完成预热相加不超过两小时训练预算。
