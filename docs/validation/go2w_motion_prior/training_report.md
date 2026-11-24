# Go2W Motion Prior 训练报告

训练源码为 `08f61581c93d9e2337fdc18f3053823cb7ed577d`，由 `174c6c11` 开始开发。训练阶段为 `flat_prior`，实际训练循环 **7073.8 s**，进程在 7140.4 s 内完成 5,518 次迭代和 56,766,464 条物理 transition。
4096 个训练环境，UniLab CPU 批量 MuJoCo 与 CUDA Learner。从随机权重开始，单组正式训练。
应用户要求在迭代边界提前停止，评测基于最终落盘 checkpoint。

## 固定真实 MuJoCo 评测

| 累计训练时间 | 迭代 | Transition | 通过率 | vx RMSE | vy RMSE | wz RMSE | 跌倒率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.000 h | 0 | 0 | 0.0% | 0.429 | 0.522 | 1.259 | 100.0% |
| 0.500 h | 1599 | 16,633,856 | 0.0% | 0.476 | 0.517 | 2.552 | 100.0% |
| 1.000 h | 2969 | 30,662,656 | 23.3% | 0.171 | 0.135 | 0.349 | 19.2% |
| 1.501 h | 4280 | 44,089,344 | 50.0% | 0.107 | 0.104 | 0.270 | 8.3% |
| 1.965 h | 5518 | 56,766,464 | 40.0% | 0.158 | 0.193 | 0.616 | 19.2% |

每次固定评测使用普通站姿、Actor 自主动作和三个固定初始 yaw。线速度 RMSE 以 m/s 计，角速度以 rad/s 计。
最终命令切换通过率为 16.7%，最佳 checkpoint 重测平地通过率为 50.0%。
`ready_for_next_stage = false`。

| 运动方向 | 初始 | 最终 | 最佳 checkpoint |
|---|---:|---:|---:|
| standing | 0.0% | 100.0% | 100.0% |
| forward | 0.0% | 80.0% | 60.0% |
| backward | 0.0% | 75.0% | 100.0% |
| left | 0.0% | 0.0% | 0.0% |
| right | 0.0% | 0.0% | 0.0% |
| yaw_left | 0.0% | 20.0% | 100.0% |
| yaw_right | 0.0% | 60.0% | 80.0% |
| forward_yaw | 0.0% | 100.0% | 50.0% |
| forward_lateral | 0.0% | 0.0% | 25.0% |
| lateral_yaw | 0.0% | 0.0% | 0.0% |
| coupled | 0.0% | 33.3% | 25.0% |

## 采集和训练

本轮源占比统计使用 float32 累加，普通 Actor 计数达到 2²⁴ 后饱和。因此以计数饱和之前的日志和并发配额核验约 70%／20%／10%，最终 JSON 中的累计占比失真。
保护池、普通历史池和近期池的累计采样数见 `progress.json`。近期、普通历史、保护顺序的比例为 [0.5, 0.3003, 0.1997]。
训练全程峰值 GPU 分配为 4077 MiB。训练最后一次日志记录的归一化 WM H8 误差为 0.165793。

| 短测 | 每迭代秒数 | 真实 steps/s | 峰值 MiB |
|---|---:|---:|---:|
| go2w_prior_benchmark_2048 | 0.927 | 5522 | 2881 |
| go2w_prior_benchmark_4096_no_prior | 1.148 | 8918 | 4075 |
| go2w_prior_benchmark_4096 | 1.134 | 9030 | 4077 |
| go2w_prior_benchmark_4096_profile | 1.149 | 8911 | 4077 |

## 独立物理 World Model 预测

| 数据来源 | 类别 | Horizon | 存活 / 样本 | 线速度 RMSE | 角速度 RMSE | 腿位置 RMSE | 腿速度 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| actor | standing | 1 | 60/60 | 0.0064 | 0.1103 | 0.0042 | 0.3818 |
| actor | forward | 1 | 198/198 | 0.0077 | 0.1196 | 0.0054 | 0.4263 |
| actor | backward | 1 | 240/240 | 0.0095 | 0.1055 | 0.0058 | 0.4843 |
| actor | left | 1 | 84/84 | 0.0793 | 0.8484 | 0.0411 | 3.0261 |
| actor | right | 1 | 240/240 | 0.0323 | 0.1935 | 0.0130 | 1.0918 |
| actor | yaw_left | 1 | 300/300 | 0.0161 | 0.1629 | 0.0087 | 0.7339 |
| actor | yaw_right | 1 | 300/300 | 0.0148 | 0.1885 | 0.0083 | 0.6799 |
| actor | forward_yaw | 1 | 174/174 | 0.0074 | 0.1452 | 0.0053 | 0.4305 |
| actor | forward_lateral | 1 | 190/190 | 0.0242 | 0.2491 | 0.0138 | 0.9093 |
| actor | lateral_yaw | 1 | 147/147 | 0.0627 | 0.5527 | 0.0259 | 2.1140 |
| actor | switch | 1 | 263/263 | 0.0285 | 0.3357 | 0.0166 | 1.2695 |
| reference_pd | forward_backward | 1 | 6/6 | 0.0131 | 0.1376 | 0.0037 | 0.2649 |
| reference_pd | yaw | 1 | 2/2 | 0.0101 | 0.1117 | 0.0028 | 0.2710 |
| reference_pd | lateral | 1 | 5/5 | 0.0121 | 0.1235 | 0.0038 | 0.2542 |
| reference_pd | forward_yaw | 1 | 2/2 | 0.0096 | 0.1323 | 0.0030 | 0.2921 |
| reference_pd | forward_lateral | 1 | 4/4 | 0.0109 | 0.1586 | 0.0038 | 0.3049 |
| reference_pd | lateral_yaw | 1 | 3/3 | 0.0123 | 0.1004 | 0.0037 | 0.2348 |
| reference_pd | standing | 1 | 1/1 | 0.0116 | 0.1059 | 0.0029 | 0.2557 |
| perturbed_actor | forward_backward | 1 | 6/6 | 0.0195 | 0.2286 | 0.0126 | 0.9444 |
| perturbed_actor | yaw | 1 | 2/2 | 0.0151 | 0.1735 | 0.0083 | 0.7165 |
| perturbed_actor | lateral | 1 | 5/5 | 0.0200 | 0.5061 | 0.0194 | 2.0290 |
| perturbed_actor | forward_yaw | 1 | 2/2 | 0.0112 | 0.1294 | 0.0087 | 0.8397 |
| perturbed_actor | forward_lateral | 1 | 4/4 | 0.0233 | 0.4138 | 0.0203 | 1.6339 |
| perturbed_actor | lateral_yaw | 1 | 3/3 | 0.0139 | 0.3129 | 0.0233 | 2.0797 |
| perturbed_actor | standing | 1 | 1/1 | 0.0152 | 0.0982 | 0.0094 | 0.9190 |
| actor | standing | 8 | 60/60 | 0.0368 | 0.1759 | 0.0296 | 0.6522 |
| actor | forward | 8 | 198/198 | 0.0509 | 0.1546 | 0.0272 | 0.6485 |
| actor | backward | 8 | 240/240 | 0.0368 | 0.1892 | 0.0261 | 0.6333 |
| actor | left | 8 | 84/84 | 0.2963 | 1.3487 | 0.3133 | 5.1694 |
| actor | right | 8 | 240/240 | 0.1122 | 0.4297 | 0.0799 | 1.7673 |
| actor | yaw_left | 8 | 300/300 | 0.0468 | 0.3058 | 0.0286 | 0.9289 |
| actor | yaw_right | 8 | 300/300 | 0.0454 | 0.3857 | 0.0506 | 1.0674 |
| actor | forward_yaw | 8 | 174/174 | 0.0394 | 0.1371 | 0.0263 | 0.6279 |
| actor | forward_lateral | 8 | 190/190 | 0.0818 | 0.7332 | 0.0763 | 1.7319 |
| actor | lateral_yaw | 8 | 147/147 | 0.1663 | 0.9432 | 0.1555 | 3.3395 |
| actor | switch | 8 | 263/263 | 0.1056 | 0.5305 | 0.1005 | 1.7360 |
| reference_pd | forward_backward | 8 | 6/6 | 0.0283 | 0.2239 | 0.0165 | 0.4412 |
| reference_pd | yaw | 8 | 2/2 | 0.0586 | 0.3843 | 0.0376 | 1.2704 |
| reference_pd | lateral | 8 | 5/5 | 0.1311 | 1.4767 | 0.1150 | 4.8470 |
| reference_pd | forward_yaw | 8 | 2/2 | 0.0657 | 0.6436 | 0.0529 | 1.5760 |
| reference_pd | forward_lateral | 8 | 4/4 | 0.0612 | 0.5998 | 0.0418 | 1.7310 |
| reference_pd | lateral_yaw | 8 | 3/3 | 0.0485 | 0.8716 | 0.0777 | 2.2190 |
| reference_pd | standing | 8 | 1/1 | 0.0165 | 0.1145 | 0.0091 | 0.3722 |
| perturbed_actor | forward_backward | 8 | 6/6 | 0.1152 | 0.4943 | 0.0884 | 1.2978 |
| perturbed_actor | yaw | 8 | 2/2 | 0.1200 | 0.3017 | 0.0672 | 0.9405 |
| perturbed_actor | lateral | 8 | 5/5 | 0.1338 | 0.5024 | 0.0863 | 1.5012 |
| perturbed_actor | forward_yaw | 8 | 2/2 | 0.1059 | 0.2463 | 0.0667 | 0.8203 |
| perturbed_actor | forward_lateral | 8 | 4/4 | 0.1867 | 0.4456 | 0.0904 | 1.4505 |
| perturbed_actor | lateral_yaw | 8 | 3/3 | 0.1109 | 0.4606 | 0.0832 | 1.6036 |
| perturbed_actor | standing | 8 | 1/1 | 0.0869 | 0.2256 | 0.0621 | 1.2234 |

## 结果文件

- `best.pt` 保存推理权重。 `last.pt` 保存优化器、Replay、历史池、阶段及课程等恢复信息。
- `evaluation/final.json` 和 `best_physical.json` 保存每个速度及命令切换的真实评测。
- `evaluation/final_wm.json` 和 `best_wm.json` 保存 H1/H8 独立物理留出测试。
- `evaluation/final_videos/` 和 `best_videos/` 中的画面取自真实 Actor rollout。

一组正式实验可描述从零学习曲线。对 Motion Prior 或去掉默认姿态奖励的因果贡献，需要匹配对照实验。

## 分析与复现

- 90 分钟 checkpoint 的原地转向最稳定。正向 15/15、负向 12/15，其中 `−0.60 rad/s` 在相同三个初始 yaw 下均未通过。最终 checkpoint 的正向原地转向通过率降到 20%，负向降到 60%。同一普通 reset 协议下，最佳与最终模型的总体跌倒率分别为 8.3% 和 19.2%。
- 左右侧移的所有 24 条固定评测均未通过。最终 checkpoint 在中间初始 yaw 的 `+0.20 m/s` 左移个例中出现约 `+0.59 m/s` 的回合均速，随后跌倒。`−0.20 m/s` 右移的回合均速约 `−0.02 m/s`。前进＋转向是最终 checkpoint 中表现最好的耦合类别，12/12 通过。
- 最终 World Model 的 Actor 留出集 H1/H8 线速度 RMSE 为 0.029/0.098 m/s，角速度 RMSE 为 0.290/0.521 rad/s。参考 PD 留出窗口的 H8 角速度 RMSE 为 0.836 rad/s。模型误差与真实策略通过率分别报告，当前结果支持保留在 `flat_prior`。
- 首轮采集使用 23 条 Qualified 与 78 条动态 Candidate。停止迭代记录的匹配覆盖率为 90.6%，其中 70.7% 为 Candidate。4096 环境短测中，以 `lambda_prior=0.25` 测得 12.2% 和 14.8% 的先验／任务 Actor 梯度比例，余弦分别为 0.98 和 0.33。正式设为 0.60，WM 和 Critic 在 Actor 更新中冻结。正式运行使用 20 个 MuJoCo 原生 CPU 线程，累计真实采集净吞吐约 7,988 transitions/s。短测的同步 profiling 数据记录了总吞吐，缺少可核实的逐阶段耗时分解。

使用 [实现文档](../../go2w_motion_prior.md) 中的正式训练、评测、播放与恢复命令可复现流程。[学习曲线](learning_curves.png) 同时展示真实评测、训练 reward 和 WM H8 误差。[固定协议逐回合结果](physical/) 包含真实速度误差、跌倒、命令和通过标志。[最终 World Model 留出](final_wm.json) · [最佳 World Model 留出](best_wm.json) · [已解析配置](resolved_config.yaml)
