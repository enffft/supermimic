"""Fill manuscripts with audited JSON-derived tables and write provenance registries."""
from pathlib import Path
import json,csv
from build_documents import table,OUT,DATA,load,M,G,gp,gs,mp,ms,net,go2_rewards,m1_rewards,wmtable,gradtable,goalbins

def triple(r,keys):return '('+', '.join(f'{r[k]:.3f}' for k in keys)+')'
def pc(v):return '—' if v is None else f'{v*100:.2f}%'
rows=[]
for p,s in zip(G['go2_ppo'],G['go2_supermimic']):
 rows.append([str(p['command_body_xy_yaw']),triple(p,['mean_vx','mean_vy','mean_yaw_rate']),triple(s,['mean_vx','mean_vy','mean_yaw_rate']),f"{p['velocity_rmse']:.4f}/{p['yaw_rate_rmse']:.4f}",f"{s['velocity_rmse']:.4f}/{s['yaw_rate_rmse']:.4f}",pc(p['fall_rate'])+'/'+pc(s['fall_rate']),str(p['episodes'])+'/'+str(s['episodes'])])
g_results=table(['命令vx/vy/wz','PPO实际vx/vy/wz','SM实际vx/vy/wz','PPO XY/yaw RMSE','SM XY/yaw RMSE','跌倒PPO/SM','结束episode PPO/SM'],rows)
cl=load('go2_lineage_clocks');rows=[]
for name,label,it in [('go2_new_cold_2p5h','新从零',None),('go2_old_chain_2p95h','旧it22000','22000'),('go2_old_actor_2p96h','旧it24000','24000'),('go2_old_final_chain_4p55h','旧it34000','34000')]:
 d=load(name);r=d['real_mujoco'][0];a=load('go2_supermimic_training_progress')['run_wall_seconds']/3600 if it is None else cl['base_actor']['hours']+cl['gait_saved'][it]['hours'];full=a if it is None else a+cl['wm_source']['hours']
 rows.append([label,f'{a:.5f}',f'{full:.5f}',f"{r['mean_vx']:.4f}",f"{r['stance_slip_rms_m_s']:.4f}",pc(r['diagonal_support_fraction']),'/'.join(f'{x*100:.2f}' for x in r['mean_episode_peak_foot_clearance_m'])])
g_hist=table(['模型','Actor链h','完整链h','vx m/s','足滑RMS m/s','对角支撑','FL/FR/RL/RR峰高cm'],rows)
pL=load('m1_ppo_forward')['metrics'];sL=load('m1_supermimic_forward')['metrics'];pM=load('m1_ppo_manipulation')['metrics'];sM=load('m1_supermimic_manipulation')['metrics'];rows=[]
for label,k,pm,sm in [('L实际vx m/s','L_actual_vx',pL,sL),('L XY RMSE m/s','L_velocity_xy_rmse',pL,sL),('L yaw RMSE rad/s','L_yaw_rate_rmse',pL,sL),('L跌倒率/结束episode','L_fall_rate',pL,sL),('M目标成功率','M_goal_success_rate',pM,sM),('M成功goal数','M_successful_goals',pM,sM),('M结束goal数','M_completed_goals',pM,sM),('M活跃删失goal数','M_censored_goals',pM,sM),('M每env每min成功数','M_successes_per_M_env_minute',pM,sM),('M成功goal平均耗时s','M_successful_goal_time_s',pM,sM),('M时间平均TCP距离m','M_distance_m',pM,sM),('M跌倒率','M_fall_rate',pM,sM),('M结束episode数','M_completed_episodes',pM,sM)]:
 def fmt(v):return '不适用/无样本' if v is None else pc(v) if 'rate' in k and 'yaw' not in k else str(v) if isinstance(v,int) else f'{v:.6f}'
 rows.append([label,fmt(pm[k]),fmt(sm[k])])
m_results=table(['指标','PPO','SuperMimic'],rows)
post=load('m1_policy_posture_probe')['results'];rows=[]
for alg in ['PPO','SuperMimic']:
 for task in ['L','M']:
  p=post[alg]['summary'][task];rows.append([alg+' '+task,f"{p['height_mean_m']:.4f}",f"{p['action_abs_mean']:.6f}",pc(p['action_over_0p95_fraction']),f"{p['tanh_local_slope_mean']:.6f}"])
posture=table(['策略/任务','root平均高度m','平均∣a∣','∣a∣>.95','平均1−a²'],rows)
web=json.loads((OUT/'external_sources/index.json').read_text());rows=[]
for r in web:
 readme=next(f for f in r['files'] if f['path']=='README.md');p=r['repo'].replace('/','__');code=[f for f in r['files'] if f['path'].endswith(('/train.py','/networks.py','/env.py','rigid_resets.py','auto_train.py'))]
 rows.append([f"[{r['repo']}]({readme['url']})",r['sha'],'；'.join(f"[{f['path']}]({f['url']})" for f in code[:4]),f"[本地快照](external_sources/{p}/)"])
webtab=table(['项目/公开说明','本次读取commit','相关实际源码','留存'],rows)
repl={'GO2_REWARDS':go2_rewards,'NETWORK_TABLE':net,'GO2_RESULTS':g_results,'GO2_HISTORY':g_hist,'M1_REWARDS':m1_rewards,'M1_RESULTS':m_results,'GOAL_BINS':goalbins,'POSTURE':posture,'WM_ERRORS':wmtable,'M1_GRADIENTS':gradtable,'WEB_SOURCES':webtab}
for p in (OUT/'scripts/templates').glob('*.md.in'):
 text=p.read_text()
 for k,v in repl.items():text=text.replace('<!-- '+k+' -->',v)
 (OUT/p.name.removesuffix('.in')).write_text(text)
# Figure registry: every visual maps to its exact data inputs and scope.
figsources={
 'G01':['go2_ppo_config','go2_supermimic_config','go2_ppo_training_completion','go2_ppo_training_progress','go2_supermimic_training_progress'],
 'G02':['go2_results'],'G03':['go2_results'],'G04':['go2_new_cold_2p5h','go2_old_chain_2p95h','go2_old_actor_2p96h','go2_old_final_chain_4p55h','go2_lineage_clocks'],
 'G05':['go2_supermimic_config'],'G06':['go2_results'],
 'M01':['m1_ppo_config','m1_supermimic_config','m1_ppo_completion','m1_supermimic_completion','experiment_manifest'],
 'M02':['m1_ppo_forward','m1_supermimic_forward','m1_ppo_manipulation','m1_supermimic_manipulation'],
 'M03':['m1_ppo_manipulation','m1_supermimic_manipulation'],
 'M04':['training_curves','m1_fixed_input_snapshots','m1_policy_posture_probe','m1_wm_response_probe'],
 'M05':['m1_wm_response_probe_eps0.01'],'M06':['m1_supermimic_config'],'M07':['m1_gradient_components','m1_action_history_feedback'],
 'A01':['go2_ppo_config','m1_ppo_config','go2_supermimic_config','m1_supermimic_config'],
 'A02':['m1_policy_posture_probe','m1_gradient_components','m1_action_history_feedback','m1_supermimic_config'],
 'A03':['training_curves'],'A04':['go2_supermimic_config','m1_supermimic_config','experiment_manifest']}
registry={}
for p in sorted((OUT/'figures').glob('*.png')):
 prefix=p.stem.split('_')[0];registry[p.stem]={'inputs':['data/'+x+'.json' for x in figsources[prefix]],'script':'scripts/'+('plot_architecture.py' if prefix in ['A01','A02','A04'] else 'plot_training_curves.py' if prefix in ['G03','G06','M03','M04','M07','A03'] else 'plot_tables.py'),'formats':['png','svg'],'size_px':[3840,2160]}
(DATA/'figure_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2))
# Numeric pointers for aggregate metrics, configs, model shapes, diagnostics and curves.
records=[]
def flatten(v,pointer='',source=''):
 if isinstance(v,dict):
  for k,w in v.items():
   if k not in ['goals','policy_shapes','status_at_extraction','available_tags','source_sha256'] and not (k=='episodes' and isinstance(w,list)):flatten(w,pointer+'/'+str(k).replace('~','~0').replace('/','~1'),source)
 elif isinstance(v,list):
  for i,w in enumerate(v):flatten(w,pointer+'/'+str(i),source)
 elif isinstance(v,(int,float)) and not isinstance(v,bool):records.append({'source_snapshot':'data/'+source+'.json','json_pointer':pointer,'value':v})
for name in sorted(set(sum(figsources.values(),[])) | {'m1_wm_response_probe','m1_selected_training_scalars','go2_lineage_clocks'}):
 flatten(load(name),source=name)
with (DATA/'numeric_evidence.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['source_snapshot','json_pointer','value']);w.writeheader();w.writerows(records)
# Human evidence index.
verification=load('evidence_verification')
prov='''# 实验证据索引与来源等级

2026-11-06。本次输出不代表训练代码发行版。Git main / HEAD `'''+M['head']+'''`，原有dirty实现保留；未切分支、建worktree、reset/stash/clean、修改训练实现或权重、commit/push。只新增本组会目录。

## 1. 怎样从图追到原始数字

1. [figure_registry.json](data/figure_registry.json) 给出每张图的输入JSON与绘图脚本。
2. [numeric_evidence.csv](data/numeric_evidence.csv) 给出汇总指标、配置与诊断曲线的JSON Pointer和完整精度数值；目标分桶从原始goals列表重新统计，不含censored。
3. [experiment_manifest.json](data/experiment_manifest.json) 将data快照映射回本机原始路径、SHA256，并列出checkpoint、网络参数形状、版本和审计起点状态。
4. [evidence_verification.json](data/evidence_verification.json) 保存CPU导出一致性、训练配置共同字段比较、原始TensorBoard时钟复核和前次审计源码哈希一致性。
5. [source_snapshot](data/source_snapshot/) 固定本次读取的相关本机源码；[upstream_runtime](data/upstream_runtime/) 固定本机安装版本PPO关键实现。它们不是重新构建出的历史完整源码。

原始JSON中的null保留为无样本/不适用，不擅自填0。训练曲线只保留有限标量，不把warmup的`nan(n=0)`解释成权重NaN；原始TensorBoard路径和hash仍在manifest。

## 2. 模型身份与直接checkpoint核验

'''
rows=[]
for key,v in M['checkpoints'].items():
 rows.append([key,v['path'],v['sha256'],str(v.get('metadata',{}).get('iteration',v.get('metadata',{}).get('iter','推理包')))])
prov+=table(['身份','本机checkpoint','SHA256','iteration'],rows)
prov+='''Go2两种部署包和M1 PPO推理包已在CPU逐张量核对来源Actor，完全相同；64组随机输入的动作最大误差均0。检查的是导出和身份，没有运行新的真实仿真，也不证明实机闭环安全或新viewer可用。

## 3. 数字—条件—来源映射

'''
prov+=table(['数据族','主要data文件','评估条件 / checkpoint / 口径','证据级别'],[
 ['Go2等时长配置、样本、更新','go2_*config、*training_progress、*training_completion、experiment_manifest','从零seed2026；3072环境；学习循环约2.5h；checkpoint形状/metadata','本次直接读取配置、checkpoint、日志'],
 ['Go2九命令正式结果','go2_{ppo,supermimic}_{evaluation,extra}','各指令64结束episode，10s、seed2026、deterministic/noDR/noise；指向对应部署包','本次读取原始评测JSON；不重跑'],
 ['Go2历史步态','go2_new_cold_2p5h、go2_old_chain_2p95h、go2_old_actor_2p96h、go2_old_final_chain_4p55h','4权重×4命令，各16episode、10s、seed2026；排除起步1s足端指标','本次读原始JSON与ckpt哈希'],
 ['Go2累计训练时间','go2_lineage_clocks、evidence_verification.old_clocks','time/hours原始TB；WM仅导入、Actor链不含source Actor；保存与未保存段分开','本次独立复读原始TB并匹配checkpoint来源'],
 ['M1正式结果','m1_{ppo,supermimic}_{balanced,forward,manipulation}','v2，各64env60s，seed123；共享真实reward、SM独有endpoint；M成功换点、删失单列','本次读原始JSON、config与ckpt'],
 ['M1目标高度/初始距离桶','m1_*_manipulation.goals','仅已结束goal；生成脚本排除censored；成功后随机轨迹分叉','从原始逐goal记录本次重新汇总'],
 ['M1最终姿态/指令敏感性','m1_policy_posture_probe','4env5s balanced，统计1–5s；L固定.12；冻结策略','已有真实MuJoCo probe，本次只读'],
 ['M1模型H1/8/16/32误差','m1_wm_response_probe、*_eps0.01','一个名义姿态，12分支常动作，epsilon.03/.01，目标[1.5,0,.8]；不是闭环或全分布','已有真实FD/WM probe，本次只读'],
 ['M1全快照、动作反馈','m1_fixed_input_snapshots、m1_action_history_feedback','43快照固定合成物理输入；反馈64次，无物理/优化；非历史Replay','已有CPU反事实JSON；本次重绘'],
 ['M1分项梯度','m1_gradient_components','6快照、4相同起点、H32、frozenWM/Critic，J/H，裁剪前；不是历史训练贡献百分比','已有冻结梯度诊断，本次只读'],
 ['M1训练时序/正则实际值','training_curves、m1_selected_training_scalars、m1_reward_scalar_audit','原始TB的真实/想象标签分别保留；重复step不强行合并','本次读TB并保留已有导出'],
 ['Reward公式/默认正则','source_snapshot/.../tasks/{velocity_reward,go2_gait,m1_reward}.py + 保存config','显式字段从config读取；固定系数从审计源码和历史报告交叉核对；全部dt一次','审计源码与前次审计10文件哈希相同；非完整训练时源码封存'],
 ['开源机制','external_sources/index.json、各repo/provenance.json','6仓库固定SHA，README及选定实现；未安装、运行或复现实验','本次联网核验公开说明与代码']])
prov+='''\n## 4. 本次新增复核结果\n\n'''
prov+=table(['检查','结果'],[
 ['共同环境配置字段','Go2无差异；M1仅PPO未使用WM的Replay容量2、SM256（共同键比较）'],
 ['最终课程','checkpoint：Go2 PPO stage1(B)、SM stage2(C)'],
 ['导出包','Go2 PPO/SM与M1 PPO：Actor tensor exact equality，动作误差0'],
 ['原始时钟','WM .251819h；base Actor2.004372h；gait22000/24000/34000/34600分别.695349/.956017/2.289729/2.372166h'],
 ['前次审计源hash','objective/policy_trainer/rollout/tasks/nets/m1_reward/go2_gait/loop/evaluate_go2/training.m1_reward共10项均相同'],
 ['硬件与运行环境','RTX4070 SUPER 12GiB；Conda supermimic；包版本见manifest；本任务未用venv或启动GPU训练/评测']])
prov+='''\n## 5. 仍依赖历史记录与未验证边界

- 旧训练具体人工启动/中断背景、旧端到端部署C++/sim2sim误差和硬件部署边界，引用既有报告；本次没有重建C++、重新跑sim2sim或实机。当前导出身份另做了CPU核验。
- Go2固定reward系数与历史实现用保存配置、历史报告和当前源码核对；缺少每次训练完整未提交源码归档，不能断言全部训练文件已通过本次HEAD完整复现。M1本次reward源码hash与当天前次审计一致，仍不等于训练启动时全仓库快照。
- 当前源码所存学习式term head存在，但本轮Actor实际用解析高度/倾角终止；通用论文描述与任务实现分开。
- WM反号是局部物理证据；Critic方向没有真实梯度真值；反馈闭环因果图尚无完整消融。Replay分布恶化、共享多任务冲突和低目标不可达未证明。
- 单训练seed、无DR独立名义评测不能给出真实硬件或随机地形鲁棒性结论。没有新长程训练、录像或PPT文件。
- 论文细节的页码/原实验硬件来自仓库PAPER_SPEC与既有报告，本次未重新下载核对论文全文；公开复现的分钟数也仅为作者报告，不是本机复测。

## 6. 必读历史材料索引

'''
for name in ['FINAL_POLICY_EVALUATION_20261104','GO2_POLICY_PROVENANCE_AND_GAIT_20261106','GO2_FINAL_EVALUATION_20261103','GO2_TRAINING_20261103','GO2_COMPARISON_STAGE1_PPO_20261104','GO2_COMPARISON_STAGE2_DEPLOY_20261104','GO2_COMPARISON_STAGE3_RUNBOOK_20261104','GO2_GAIT_REWARD','M1_PPO_BASELINE_20261105','M1_PPO_SUPERMIMIC_EVALUATION_20261105','M1_V2_AND_SEQUENTIAL_TRAINING_20261105','M1_V2_RESULTS_AND_FAILURE_ANALYSIS_20261106','M1_SATURATION_AND_GO2_TRAINING_HISTORY_20261106','M1_PIPER_TRAINING_DESIGN_AUDIT_20261105','M1_REWARD_ONLY_BASELINE_20261104','M1_ACTION22_P0_P1_REMEDIATION_20261104','TRAINING_AUDIT_INDEX_20261106','SUPERMIMIC_ALGORITHM_MAP','SUPERMIMIC_CODE_WALKTHROUGH','PAPER_SPEC']:
 prov+=f'- [{name}](../{name}.md)\n'
(OUT/'EXPERIMENT_PROVENANCE.md').write_text(prov)
print('Assembled 3 manuscripts, provenance, figure registry, numeric source pointers:',len(records))
