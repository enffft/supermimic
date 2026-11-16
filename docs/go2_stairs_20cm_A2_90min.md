# Go2 A2：4cm分级 / 20cm上限 / 2048环境90分钟续训

2026-11-11追加用户指令：取消60分钟安排，楼梯级差改4cm，最高Level5，保留A2奖励，启动90分钟。当前状态是**训练运行中，尚无本轮最终技能结论**。原14cm消融记录见 [报告](go2_stairs_14cm_clearance_ablation.md)。

## 真实启动状态

- 启动：`2026-11-11T18:48:39.188347+08:00`；supervisor PID `1384500`，训练 PID `1384501`。
- 2048 CPU MuJoCo环境；RTX4070SUPER训练WM/Actor/Critic；wall budget5400s，iteration上限200000，不因500或1500轮默认值提前结束。
- 日志 `results/stairs_clearance_20cm_A2/training.log`；实时状态 `run_status.json`；输出 `longrun_90min/`；训练后自动运行新旧同地图完整物理评测并写 `comparison.json`。
- 原best.pt不覆盖。本轮继承A2短程候选 `/home/yifeiy/ws_go2_unilab/Uni-SuperMimic/results/stairs_clearance_B4/soft_deficit/trial/last.pt`，SHA256 `aba2ebf8ee68556fa6c4cf81cefdb7e511a54d3cb813f642d005a98a564cc7c9`；新地图迁移checkpoint SHA256 `65342fed3b8729f6e7bf588087bebb60d78b169e5d193b6e2be7d2dd60d97d70`。该选择代表选用A2研究方案，不代表A2短程策略已优于原best。

## 新课程与编译地形

| Level | 名义楼梯高度 | 编译上楼首立面 | 编译下楼首落差 |
|---|---:|---:|---:|
| 1 | 4cm | 3.765cm | 4.392cm |
| 2 | 8cm | 7.529cm | 8.157cm |
| 3 | 12cm | 11.922cm | 12.549cm |
| 4 | 16cm | 15.686cm | 16.314cm |
| 5 | 20cm | 19.451cm | 20.078cm |

Level0平地；Rough及Slope Up/Down仍封顶Level4，原幅度保持。MuJoCo导入PNG高度图存在8bit量化，一个量化单位约0.628cm，所以表中区分名义20cm与实际编译高度；真实/想象均使用同一个编译表面查询。

迁移 `go2_stairs_4cm_level5_atlas_v1` 仅允许地形尺寸、级差和相关课程布局变化，其他控制/网络/A2 reward严格一致。旧Replay世界坐标不适用，显式清空并重新真实采样，累计转移和优化计数仍继承。旧楼梯mastery用 `floor(old_level/2)` 向下保守映射，例如旧8cm Level4→新8cm Level2；rough/slope不变；新地图统计及streak重建，旧统计归档。没有把旧Level5当作新20cm。

## A2与更新策略

A2: feet_clearance=.05、target=.12、sigma=.065、tau=.02、soft_deficit、仅楼梯tile。其他奖励、PD、动作尺度.25、dt=.02、213D Blind Actor、77D WM、H8/H32均保持。未因20cm上限自动提高目标摆腿高度。

先200轮WM适应、100轮Critic warmup，再低LR Actor（独立倍率.01及原渐增机制）。每200轮加入真实高楼梯接触探针；每1000轮重新执行原七探针/80%/normalized error/H32准入，未通过的高等级Actor起点持续阻断。12/16/20cm真实数据持续进入WM；新增独立的WM高等级均衡开关，35%稀有窗口在实际存在的高楼梯等级间均衡，避免旧消融L5占据约82%的高等级WM起点预算。

20分钟一次小规模物理保护，不进行每500轮全网格评测；训练结束后统一五种子完整评测。NaN/Inf、动作漂移/饱和、资源不足及连续物理退化会早停并保留诊断、回退Actor，WM及数据保留。外部watchdog防止失控超时。旧消融的longrun_allowed=false不改写；本轮依据用户明确要求开展更长探索，仍不绕过高楼梯WM准入。

## 实际命令

```bash
OMP_NUM_THREADS=4 /home/yifeiy/miniconda3/envs/go2_unilab/bin/python tools/run_go2_clearance_90min.py
# 查看状态，不干预训练
tail -n 5 results/stairs_clearance_20cm_A2/training.log
cat results/stairs_clearance_20cm_A2/run_status.json
```

程序已真实后台启动；以上入口拒绝覆盖已有实验目录，重新运行应选择新的root并准备相应迁移输入。结束后自动附加真实计数和同地图通过率，未完成前不声称已学会12/16/20cm。


## 自动收尾结果

运行状态：完成。完整训练统计见 `results/stairs_clearance_20cm_A2/run_status.json`，同条件新旧策略对照见 `comparison.json`。

```json
{
  "stop_reason": "wall_budget",
  "result": {
    "iteration": 45707,
    "wall_h": 5.386589198112488,
    "sim_steps": 252295168,
    "wm_loss": 0.0895055378973484,
    "wm_err8": 0.1491760179400444,
    "pi_loss": -3.231554613113403,
    "v_loss": 0.00016475020616780968,
    "imag_rew": 0.3116486418247223,
    "pi_gradnorm": 0.8277193808555603,
    "imag_survival": 0.9930862784385681,
    "sim_rew": 0.3582912636181665,
    "ep_len": null,
    "ep_ret_norm": null,
    "term_frac": 0.04081632653061224,
    "it_per_s": 3.1758008010736547,
    "lr": 0.00018444011051150253,
    "completed": false,
    "wall_time": 19400.898369789124
  },
  "wall_s": 5410.344008462969,
  "new_transitions": 88080384,
  "new_updates": {
    "wm": 57600,
    "actor": 14100,
    "critic": 284000
  },
  "wm_sampled_starts_by_terrain": [
    3518952,
    156500,
    1661502,
    2452435,
    5112381,
    1843830,
    0,
    0
  ],
  "actor_starts_domain_terrain_speed": [
    [
      [
        0,
        1761893,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ],
    [
      [
        0,
        30132,
        0,
        0,
        0
      ],
      [
        0,
        28995,
        0,
        0,
        0
      ],
      [
        0,
        29310,
        0,
        0,
        0
      ],
      [
        0,
        29412,
        0,
        0,
        0
      ],
      [
        0,
        665581,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ],
    [
      [
        0,
        12611,
        0,
        0,
        0
      ],
      [
        0,
        11596,
        0,
        0,
        0
      ],
      [
        0,
        12012,
        0,
        0,
        0
      ],
      [
        0,
        11471,
        0,
        0,
        0
      ],
      [
        0,
        267341,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ],
    [
      [
        0,
        10725,
        0,
        0,
        0
      ],
      [
        0,
        13149,
        0,
        0,
        0
      ],
      [
        0,
        11952,
        0,
        0,
        0
      ],
      [
        0,
        11691,
        0,
        0,
        0
      ],
      [
        0,
        265015,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ],
    [
      [
        131,
        0,
        0,
        0,
        0
      ],
      [
        288,
        0,
        0,
        0,
        0
      ],
      [
        220406,
        0,
        0,
        0,
        0
      ],
      [
        8308,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ],
    [
      [
        1517,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        206064,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ]
  ],
  "unknown_speed_index": 4,
  "checkpoint": "/home/yifeiy/ws_go2_unilab/Uni-SuperMimic/results/stairs_clearance_20cm_A2/longrun_90min/last.pt"
}
```

| 楼梯 | 新地图来源通过率 | 续训候选通过率 | 来源跌倒率 | 候选跌倒率 |
|---|---:|---:|---:|---:|
| up L1 (4cm) | 100% | 100% | 0% | 0% |
| up L2 (8cm) | 80% | 100% | 0% | 0% |
| up L3 (12cm) | 0% | 0% | 0% | 20% |
| up L4 (16cm) | 0% | 0% | 0% | 0% |
| up L5 (20cm) | 0% | 0% | 0% | 0% |
| down L1 (4cm) | 80% | 100% | 0% | 0% |
| down L2 (8cm) | 100% | 100% | 0% | 0% |
| down L3 (12cm) | 0% | 0% | 0% | 40% |
| down L4 (16cm) | 0% | 0% | 100% | 100% |
| down L5 (20cm) | 0% | 0% | 100% | 100% |
