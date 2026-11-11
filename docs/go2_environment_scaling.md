# Go2 2048/3072/4096环境实测

结论：**三种规模均能完成完整训练更新，4096环境显存和主机内存均可接受。** 本轮仅作有界资源/吞吐基准，没有启动长训或替换已评估策略。

基线代码`061bad3`，源模型`results/omnidirectional_terrain/stageB/last.pt`（SHA256见各provenance）。新增工具`tools/benchmark_go2_omni_scale.py`；不修改UniLab上游或既有训练默认值。

同一MuJoCo mixed terrain、同一全向速度分布、同一source Actor/WM/Critic和优化器。保留213D盲Actor、77D地形WM、H8、H32，WM隐藏层2048×3、Actor2048/1024/512/256；WM batch256、每轮4更新、Actor batch256、Critic5×4、Replay时间深度1024、UTD0.4、OMP4、无compile。没有以小网络或不同batch作性能比较。

改变N后明确只转移权重/优化器/累计计数，重新采样物理和Replay；课程从source统一mastery4建立，固定平地域保持Level0，其余保留15%低难度回采。不是跨N的完整Replay恢复。三次独立运行顺序为3072→2048→4096，没有同时跑相互争用资源的训练。

各运行先真实采样512个向量步（覆盖完整10秒episode及Reset），随后128个outer iteration，**实际各更新WM512次、Actor128次、Critic2560次**。排除warm fill及头16轮后，统计剩余112轮；端到端速度由第16和128轮callback的wall/真实转移差计算，排除最后checkpoint磁盘写入。env.step仿真时间由同步profile测得，包含管理器、Reward和Reset，并非单独MuJoCo kernel时间。

|环境数|env.step转移/s|端到端新转移/s|每outer秒|进程峰值GPU GiB|峰值host RSS GiB|
|---:|---:|---:|---:|---:|---:|
|2048|43,835|10,755|0.476|2.13|2.36|
|3072|47,212|15,173|0.506|2.68|2.53|
|4096|49,165|17,919|0.571|3.19|2.75|

|环境数|simulation ms/向量步|WM fit ms/更新|H32 Actor ms/批|Critic ms/批|新增真实转移|进程wall秒（含预填/保存）|
|---:|---:|---:|---:|---:|---:|---:|
|2048|46.72|39.22|158.50|20.38|1,703,936|91.0|
|3072|65.07|37.07|149.93|21.12|2,555,904|125.7|
|4096|83.31|38.58|160.76|21.28|3,407,872|167.1|

4096相对2048端到端真实采集吞吐约 **1.67倍**。各运行无OOM、无NaN/Inf，严格有限值检查保持启用。进程GPU峰值包含CUDA context，0.1秒采样峰值是下界；系统桌面显存不算入该进程。4096末可用host RAM约5.9GiB。

CPU MuJoCo逐向量步时间随N增加，WM与Actor的GPU批次固定，因此更多环境主要提高每次更新所对应的数据量；不能把采集加速解释为策略质量改善。相同outer次数下，4096每次optimizer更新对应的真实转移约为2048的2倍；没有暗改更新/采样比补偿。

当前资源支持后续选择4096。但资源验收不取消上一报告中的接触WM梯度和极限侧移门槛，正式长训及速度范围扩展仍应单独验收。benchmark产生的策略没有独立运动能力评测，不作为新的推荐部署checkpoint。

实际执行（每个输出目录均拒绝覆盖已有结果）：

```bash
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic
export OMP_NUM_THREADS=4
PY=/home/yifeiy/miniconda3/envs/go2_unilab/bin/python
$PY tools/benchmark_go2_omni_scale.py --envs 3072 --output results/omnidirectional_terrain/benchmark_3072
$PY tools/benchmark_go2_omni_scale.py --envs 2048 --output results/omnidirectional_terrain/benchmark_2048
$PY tools/benchmark_go2_omni_scale.py --envs 4096 --output results/omnidirectional_terrain/benchmark_4096
```

配置、源码/源checkpoint hash、profile、计数和资源快照见`docs/validation/go2_scaling/`；本机完整日志与benchmark checkpoint位于上述三个结果目录。旧策略文件未覆盖。
