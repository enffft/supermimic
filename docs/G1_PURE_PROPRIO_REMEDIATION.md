# G1 训练效果差：原因、边界与修正

## 结论

历史 G1 训练没有复现 SuperMimic 的高分，主要不是训练没有跑够，而是把不同实验口径混在了一起：旧 checkpoint 的 Actor 使用了 9D anchor-relative pose，纯本体实验和带特权状态实验被放进了同一条 lineage；资源受限实验使用 1024 个环境，而参考 e3 使用 4096 个环境；长 LAFAN1 clip 对纯本体策略存在根部位置和 yaw 漂移上限。训练日志还显示 World Model 的 imagined survival/return 已经很高，但真实 body orientation reward 接近零，说明 Actor 正在利用过于乐观的 imagined dynamics，继续增加时长不能自动修复真实控制误差。

静态对照没有发现 G1 reward、FK、四元数顺序、body 顺序或 dt 缩放的确定性错误。CPU FK 对拍和 exact-reference reward golden state 均通过。因此不应通过放宽 reward sigma、修改权重或加入隐藏状态来制造高分。

## 已实现的修正

- `EnvCfg.g1_pure_proprio=True` 是默认契约。G1 motion Actor 只接受 10 帧、每帧 93D 的本体观测（关节位置、关节速度、base 角速度、projected gravity、上一动作）和外部 reference Goal；总输入为 `930 + 680`。
- G1 pure-proprio 训练显式拒绝 `anchor_obs` 和 `anchor_orientation_obs`。旧的特权 checkpoint 仍可读，但必须显式设置 `env.g1_pure_proprio=False`，避免误把 689D anchor 模型当成可部署模型。
- `Policy` 暴露 `ActorInputContract` 并在 forward 前检查维度。`xi`、root error、torque、contact、joint acceleration 和 critic/world-model latent 没有进入 Actor 接口。
- G1 真实环境现在记录 anchor/root xy、yaw、body pose 和 joint error；想象 rollout 记录 root xy/z/yaw error。TensorBoard 可直接观察真实漂移和 imagined/real 脱钩。

## 为什么纯本体不能承诺复现带 anchor 的 0.90

参考仓库已经记录：纯本体 G1 在长 LAFAN1 整 clip 上大约只能达到 `0.56–0.59`，加入 anchor-relative 输入后才到约 `0.904`。Actor 没有世界系 xy/yaw 反馈，clip 变长后会累积 yaw 漂移；约 9 秒后 yaw 误差超过 15°，世界系 root/body velocity reward 随之下降。BONES-SEED 的短 clip 中位长度约 5.8 秒，纯本体可以接近 `0.93`，所以应把短 clip 作为纯本体的主要可比 benchmark，把 LAFAN1 整 clip 作为 drift-limited 指标单独报告。

## 新实验的验收口径

1. 从 scratch 开始，不能 resume 旧的 anchor checkpoint；`root_joint`、`anchor_obs=False`、`anchor_orientation_obs=False`、RunningNorm 关闭。
2. 保持参考的 320k cosine schedule、UTD、WM window/horizon、reward 和 termination；1024 env 是显存约束，必须记录为与 e3 的统计差异，而不是声称完全复现。
3. 在每个 checkpoint 保存点做一次短片段/全 40 clip 评估，至少同时报告 normalized return、失败数、body orientation、root yaw、anchor xy/yaw 漂移和 imagined/real reward 差距。
4. 若 body orientation 仍接近零但 World Model NSE 很低，应先做 real-vs-imagined 同帧字段对拍和 action clipping 检查；不要先延长训练时间。

训练和评估命令仍需用户明确说“开始”后再执行。
