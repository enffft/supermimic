"""Build portable PPO/SuperMimic comparison evidence from completed physical runs."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    run,baseline,out=args.run,args.baseline,args.output
    if load(run/'supervisor_status.json')['state']!='complete':
        raise ValueError('Training and final evaluations must complete before publishing results')
    out.mkdir(parents=True,exist_ok=True)
    files=['training/summary.json','training/arguments.json','training/ppo_config.json',
           'training/task_contract.json','training/initial_curriculum.json','training/progress.jsonl',
           'evaluation/bootstrap.json','evaluation/final.json','evaluation/final_extended.json',
           'provenance.json','source_manifest.json','supervisor_status.json','test_result.txt']
    files += [f'evaluation/{name}.json' for name in ['early','at_1h','late'] if (run/f'evaluation/{name}.json').exists()]
    for relative in files:
        dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/relative,dst)
    for name in ['bootstrap','final','final_extended']:
        dst=out/f'evaluation/supermimic_{name}.json';shutil.copy2(baseline/f'evaluation/{name}.json',dst)
    for relative in ['src/unisupermimic/tasks/velocity/go2w/ppo.py','tools/train_go2w_ppo.py',
                     'tools/run_go2w_ppo.py','tools/evaluate_go2w_experiment.py','tests/tasks/test_go2w_ppo.py']:
        dst=out/'protocol'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(run/'source_snapshot'/relative,dst)
    datasets={'initial':load(run/'evaluation/bootstrap.json'),
              'SuperMimic':load(baseline/'evaluation/final.json'),
              'PPO':load(run/'evaluation/final.json')}
    rows=[]
    for method,data in datasets.items():
        for row in data['rows']:
            actual=row['mean_actual_vx_vy_yaw'] or [None]*3
            rmse=row['RMSE_vx_vy_yaw'] or [None]*3
            rows.append({'method':method,'scenario':row['name'],'domain':row['domain'],'level':row['level'],
                         'trials':row['replicas'],'falls':row['falls'],'survived_10s':row['survived_10s'],
                         'actual_vx':actual[0],'actual_vy':actual[1],'actual_yaw':actual[2],
                         'RMSE_vx':rmse[0],'RMSE_vy':rmse[1],'RMSE_yaw':rmse[2],
                         'mean_progress_m':sum(row['progress_m'])/row['replicas'],
                         'mean_ground_gain_m':sum(row['max_ground_gain_m'])/row['replicas'],
                         'entry_successes':row['entry_successes'],'climb_successes':row['climb_successes']})
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    progress=[json.loads(line) for line in (run/'training/progress.jsonl').read_text().splitlines()]
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    hours=[r['elapsed_seconds']/3600 for r in progress]
    for ax,key,label in zip(axs.flat,['mean_episode_reward','mean_episode_length','action_std','learning_rate'],
                            ['Episode reward (last 100 completions)','Episode length (steps)','Mean Gaussian std','Adaptive learning rate']):
        ax.plot(hours,[r[key] for r in progress]);ax.set(xlabel='Training wall time (h)',ylabel=label);ax.grid(alpha=.2)
    fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    names=['forward_030','forward_060','backward_015','left_015','right_015','yaw_left_035','yaw_right_035']
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,selected,axis in zip(axs,[names[:3],names[3:5],names[5:]],[0,1,2]):
        x=list(range(len(selected)))
        for index,(method,data) in enumerate(datasets.items()):
            lookup={r['name']:r for r in data['rows']}
            bars=ax.bar([i+(index-1)*.24 for i in x],
                        [lookup[n]['mean_actual_vx_vy_yaw'][axis] if lookup[n]['mean_actual_vx_vy_yaw'] else float('nan') for n in selected],
                        width=.24,label=method)
            for bar,name in zip(bars,selected):
                ax.annotate(f"{lookup[name]['falls']}/12 falls",(bar.get_x()+bar.get_width()/2,bar.get_height()),
                            xytext=(0,3),textcoords='offset points',ha='center',fontsize=6,rotation=40)
        lookup={r['name']:r for r in datasets['PPO']['rows']}
        ax.scatter(x,[lookup[n]['goal'][axis] for n in selected],marker='_',s=220,c='black',label='Command')
        ax.set_xticks(x,selected,rotation=25,ha='right');ax.set_ylabel(['vx (m/s)','vy (m/s)','yaw (rad/s)'][axis]);ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=8);fig.savefig(out/'tracking.png',dpi=160);plt.close(fig)
    summary={'totals':{method:{'falls':sum(r['falls'] for r in data['rows']),
                            'trials':sum(r['replicas'] for r in data['rows']),
                            'stair_climb_successes':sum(r['climb_successes'] or 0 for r in data['rows'])}
                       for method,data in datasets.items()},
             'note':'Compare initial physical fields independently of WM diagnostics.'}
    old=load(baseline/'evaluation/bootstrap.json')['rows']
    keys=['name','falls','survived_10s','RMSE_vx_vy_yaw','mean_actual_vx_vy_yaw','progress_m','max_ground_gain_m']
    summary['initial_physical_evaluation_matches_previous']=all(all(a[k]==b[k] for k in keys) for a,b in zip(datasets['initial']['rows'],old))
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2))

    if args.report:
        training=load(run/'training/summary.json')
        ppo={r['name']:r for r in datasets['PPO']['rows']}
        sm={r['name']:r for r in datasets['SuperMimic']['rows']}
        evidence=out.relative_to(args.report.parent)
        text=["# Go2W PPO 两小时训练与 SuperMimic 对照", "",
              "本轮使用仓库已接入的上游 RSL-RL PPO，沿用上一轮 `omni_v3` Go2W 任务与同一份初始 Actor。目标是纯本体感知速度跟踪、横移与超过 20 cm 的楼梯运动。", "",
              f"完成 {training['elapsed_seconds']:.1f} 秒训练、{training['iterations']} 轮、{training['transitions']:,} 条真实仿真转移。最终评测跌倒 {summary['totals']['PPO']['falls']}/660，SuperMimic 为 {summary['totals']['SuperMimic']['falls']}/660。楼梯完整通过 {summary['totals']['PPO']['stair_climb_successes']}/84 次。具体结果见下表，单个训练 seed 不能支持算法优劣的普遍结论。", "",
              "## 配置与可比性", "",
              "4096 个 MuJoCo 环境，物理步长 0.005 s、控制周期 0.02 s、每回合 10 s。Actor 接收 5×50 维本体历史与 3 维速度命令，网络为 2048/1024/512/256、SiLU、tanh，保留按观测项组织的历史和固定缩放；不接收高程、真实线速度或地形标签。Critic 使用 41 维状态特征与命令，网络 256/128。", "",
              "奖励、速度命令、地形图、课程与终止条件严格比对上一轮任务 contract。腿关节目标覆盖部署机械区间的 95%，action=-1/0/+1 对应下限/默认姿态/上限；轮速与物理力矩约束保持原配置。平地侧向命令 ±0.25 m/s、yaw ±0.60 rad/s。", "",
              "PPO 使用 24 步 rollout、2048 minibatch、5 epochs、gamma=0.99、lambda=0.95、clip=0.2、entropy=0.01、起始学习率 2e-4 与上游自适应 KL 调度。Gaussian 初始标准差 0.02，与前轮采集噪声相同，之后可学习。环境额外噪声关闭，避免重复加噪。", "",
              "算法差异：PPO 在真实 on-policy 转移上更新，没有 WM、可微想象或 Replay 起点 lookahead；固定的高等级覆盖环境也参与 PPO 梯度。Critic 从头训练，使用原生 RSL-RL 输入归一化。PPO 不能按 SuperMimic 的 WM/Actor 更新次数直接对齐，本轮以两小时墙钟预算对照。", "",
              f"共享 WM 未被本次 PPO 调用或写入，训练前后文件 SHA256 一致：{load(run/'provenance.json')['WM_unchanged']}。这不改变此前对 WM 持续更新的要求。", "",
              f"初始 Actor 重跑 660 次物理评测，与上轮初始策略的速度、跌倒、位移及高程指标逐值相同：{summary['initial_physical_evaluation_matches_previous']}。", "",
              "## 速度跟踪", "",
              "每个场景 12 次首回合评测，最长 10 秒。速度取首回合存活期间第 1 秒后的样本均值；跌倒前的均值不能证明稳定跟踪。0.6 m/s 等高速前向场景可能提前到达平地边界并截断，未跌倒不代表完成 10 s。下表同时列出完整 10 s 存活数；实际值对应所列命令轴，其他轴误差见 JSON。", "",
              "|命令|SuperMimic 实际值|PPO 实际值|SM 跌倒/12|PPO 跌倒/12|PPO 存活10s/12|", "|---|---:|---:|---:|---:|---:|"]
        for name,label,axis in [('forward_030','前向 +0.30 m/s',0),('forward_060','前向 +0.60 m/s',0),
                ('backward_015','倒退 -0.15 m/s',0),('left_015','左移 +0.15 m/s',1),('right_015','右移 -0.15 m/s',1),
                ('yaw_left_035','左转 +0.35 rad/s',2),('yaw_right_035','右转 -0.35 rad/s',2)]:
            a,b=sm[name],ppo[name]
            def observed(row):
                return f"{row['mean_actual_vx_vy_yaw'][axis]:+.4f}" if row['mean_actual_vx_vy_yaw'] else '无有效样本'
            text.append(f"|{label}|{observed(a)}|{observed(b)}|{a['falls']}|{b['falls']}|{b['survived_10s']}|")
        text += ["",f"![跟踪对照]({evidence}/tracking.png)","", "## 上楼梯", "",
                 "完整通过要求：10 s 存活、前进 ≥2.1 m、侧向偏移 ≤0.6 m、平面 RMSE <0.35 m/s、yaw RMSE <0.4 rad/s，且实际中心下方地面增长 ≥台阶高度−5 mm。只到入口或保持站立不算上楼梯。", "",
                 "|台阶|PPO 跌倒/12|PPO 平均前进/m|PPO 平均高程增长/m|SM 完整通过/12|PPO 完整通过/12|", "|---|---:|---:|---:|---:|---:|"]
        for level in range(1,8):
            a,b=sm[f'stairs_up_L{level}'],ppo[f'stairs_up_L{level}']
            text.append(f"|{3*level} cm|{b['falls']}|{sum(b['progress_m'])/12:.3f}|{sum(b['max_ground_gain_m'])/12:.3f}|{a['climb_successes']}|{b['climb_successes']}|")
        text += ["", "## 扩展全向评测", "",
                 "额外 21 个平地场景 ×12 次，包含侧移 ±0.25/±0.35 m/s 和 yaw ±0.60 rad/s。±0.35 m/s 侧移超出训练命令范围。", "",
                 "|命令|PPO 实际值|PPO 跌倒/12|", "|---|---:|---:|"]
        extended=load(run/'evaluation/final_extended.json')
        for row in extended['rows'][-6:]:
            axis=2 if 'yaw' in row['name'] else 1
            value=f"{row['mean_actual_vx_vy_yaw'][axis]:+.4f}" if row['mean_actual_vx_vy_yaw'] else '无有效样本'
            text.append(f"|{row['name']}|{value}|{row['falls']}|")
        text += ["", "## 验证与复现", "",
                 "6 项针对性测试通过，覆盖历史重置、Actor 与原策略均值一致、特权信息隔离、超时掩码及上游 PPO 参数更新；4096 环境试跑 8 轮通过，评测支持不加载 WM。正式训练使用上一轮独立源码快照，只叠加 PPO 接口、入口、评测兼容改动和测试，保留工作区其他未提交修改。", "",
                 f"![训练过程]({evidence}/training.png)", "",
                 f"[完整证据与复现协议]({evidence}/README.md)、[55 场景对照 CSV]({evidence}/comparison.csv)、[训练汇总]({evidence}/training/summary.json)。三个 reset seed 仅改变同一固定地形图的出生扰动；不是跨训练 seed 或新地形泛化评估。", "",
                 f"本地运行目录：`{run.resolve()}`；策略权重：`training/last.pt`；优化器快照：`training/ppo_resume.pt`。权重仅保留本地，未纳入报告附件。", ""]
        text.insert(4, "**总体目标尚未达标：3 cm 台阶通过率改善，但纯横移和 6–21 cm 台阶仍是主要短板。**")
        text.insert(5, "")
        text += ["## 结果解释与后续方向", "",
                 "本轮只验证同一任务配置下的 PPO 基线，没有改变奖励来针对 PPO 调参。扩展侧移命令 ±0.25 m/s 也几乎没有实际响应，横移失败不能仅用评测命令幅度小来解释。训练 Gaussian 标准差由 0.02 增至约 0.035；自适应学习率首轮降至 1e-5，此后保持该值。这些是后续探索幅度和策略更新消融的线索，尚未验证为失败原因。", "",
                 "下一步宜单独比较较强的腿部探索或横移动作初始化，再判断是否需要改奖励或课程。本轮固定配置结果应保留为对照。", ""]
        args.report.write_text('\n'.join(text))

    (out/'README.md').write_text('# Go2W PPO two-hour experiment\n\nPortable scalar evidence, physical evaluation JSON and exact PPO bridge/runner protocol. Binary weights and full source snapshot remain in the local run recorded in provenance.\n\nInitial Actor, physics, reward and terrain are shared with the prior SuperMimic run. PPO uses native RSL-RL on-policy optimization, not WM imagination. Persistent high-level collection environments contribute PPO gradients.\n\n`comparison.csv`: 55 physical scenarios per method, 12 trials each; `final_extended.json`: 21 flat scenarios. Multiple reset seeds perturb the same atlas; this is one training seed.\n')
    shutil.copy2(Path(__file__),out/'protocol/summarize_go2w_ppo.py')
    hashes={str(path.relative_to(out)):hashlib.sha256(path.read_bytes()).hexdigest() for path in out.rglob('*') if path.is_file() and path.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
