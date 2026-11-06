"""Chinese research tables from saved source JSON; no manual numeric overlays."""
from style import *
import csv
G=load('go2_results');man=load('experiment_manifest'); gp=load('go2_ppo_config');gs=load('go2_supermimic_config');mp=load('m1_ppo_config');ms=load('m1_supermimic_config')
pp=load('go2_ppo_training_progress');pp['training_wall_seconds']=load('go2_ppo_training_completion')['wall_seconds'];sp=load('go2_supermimic_training_progress');mc=load('m1_ppo_completion');sc=load('m1_supermimic_completion')
def cfg_rows(robot,p,s,pc,steps,trans,updates):
 e=s['env'];a=s['policy_trainer'];w=s['wm_trainer'];r=p['runner_config'];shape='×'.join(map(str,[s['policy']['history'],3*s['policy']['num_joints']+6]))
 return [['并行环境 / seed',f"{e['num_envs']} / {s['seed']}",f"{e['num_envs']} / {s['seed']}"],['物理步长 / 策略频率','5 ms / 50 Hz','5 ms / 50 Hz'],['实际训练墙钟',f"{pc['training_wall_seconds']/3600:.4f} h",f"{steps[0]/3600:.4f} h"],['真实环境步 / 每环境',f"{pc['real_env_steps_per_env']:,}",f"{steps[1]:,}"],['真实 transitions',f"{pc['real_transitions']:,}",f"{trans:,}"],['完整更新计数',f"{pc['completed_ppo_updates']} PPO updates",f"{updates[0]} WM事件 / {updates[1]} Actor"],['Actor 输入 / 输出',f"{shape} + Goal{s['policy']['goal_dim']} → {s['policy']['num_joints']}",f"{shape} + Goal{s['policy']['goal_dim']} → {s['policy']['num_joints']}"],['Rollout / Imagination',f"{r['num_steps_per_env']} 步真实",f"H={a['horizon']} 步预测"],['逻辑 batch / microbatch','1024 / 不适用',f"{a['batch_size']} / {a['micro_batch_size']}"],['Epoch / WM fit',f"{r['algorithm']['num_learning_epochs']} epochs × {r['algorithm']['num_mini_batches']} minibatches",f"{w['batches_per_update']} fits/事件；WM窗口 {w['window']}"],['基础 Actor / Critic LR','2e−4 / 2e−4',f"{a['lr']:.0e} / {a['critic_lr']:.0e}"],['数据与目标差异','Gaussian policy + 环境探索', '确定性 Actor + 环境探索'+('；额外终点项' if robot=='m1' else '')]]
table('Go2｜从零训练 2.5 h 的配置对照','同物理环境、Actor、真实奖励；更新含义与最终课程分布不同',['项目','PPO','SuperMimic'],cfg_rows('go2',gp,gs,pp,(sp['run_wall_seconds'],18600),sp['real_transitions'],(7424,7224)),'G01_go2_training_config','来源：data/go2_*config.json、训练完成记录、checkpoint；RTX 4070 SUPER 12 GiB；单 seed。',[.26,.35,.39],14)
rows=[]
for p,s in zip(G['go2_ppo'],G['go2_supermimic']):
 assert p['command_body_xy_yaw']==s['command_body_xy_yaw']
 rows.append([trip(p['command_body_xy_yaw']),trip([p['mean_vx'],p['mean_vy'],p['mean_yaw_rate']]),trip([s['mean_vx'],s['mean_vy'],s['mean_yaw_rate']]),f"{p['velocity_rmse']:.3f} / {p['yaw_rate_rmse']:.3f}",f"{s['velocity_rmse']:.3f} / {s['yaw_rate_rmse']:.3f}",pct(p['fall_rate'])+' / '+pct(s['fall_rate'])])
table('Go2｜九条固定命令的完整评测','从零 2.5 h；速度 vx/vy（m/s）、wz（rad/s）；两者各指令 64 个已结束 episode',['命令 vx/vy/wz','PPO 实际','SM 实际','PPO XY/yaw误差','SM XY/yaw误差','跌倒 PPO/SM'],rows,'G02_go2_velocity_results','来源：data/go2_*evaluation.json、*extra.json；seed=2026，确定性、无DR/噪声；名义初态重复，含起步瞬态。',[.17,.17,.17,.18,.18,.13],12.3)
cl=load('go2_lineage_clocks');base=cl['base_actor']['hours'];wm=cl['wm_source']['hours']
rows=[]
for name,label,it in [('go2_new_cold_2p5h','新从零',None),('go2_old_chain_2p95h','旧续训 it22000','22000'),('go2_old_actor_2p96h','旧续训 it24000','24000'),('go2_old_final_chain_4p55h','旧续训 it34000','34000')]:
 d=load(name);r=d['real_mujoco'][0];a=sp['run_wall_seconds']/3600 if it is None else base+cl['gait_saved'][it]['hours'];c=a if it is None else a+wm
 rows.append([label,f'{a:.3f} / {c:.3f}',num(r['mean_vx']),num(r['stance_slip_rms_m_s']),pct(r['diagonal_support_fraction']),'/'.join(f'{v*100:.2f}' for v in r['mean_episode_peak_foot_clearance_m'])])
table('Go2｜步态改善发生在更长的旧训练链','固定 vx=0.3 m/s；历史发展对照，不能当作等时长算法消融',['策略来源','Actor/完整链 h','实际 vx','足滑 RMS↓','对角支撑','FL/FR/RL/RR 峰高 cm'],rows,'G04_go2_gait_history','来源：data/go2_{new,old}*.json、go2_lineage_clocks.json；各16 episode，10s；足滑/支撑为FK几何代理。',[.21,.18,.11,.12,.11,.27],14,bottom=.25)
table('M1-Piper｜v2 各训练 3.5 h','固定 1024 L + 1024 M；无 Motion Prior；真实样本量 PPO / SM = '+f"{mc['real_transitions']/sc['real_transitions']:.3f} 倍",['项目','PPO','SuperMimic'],cfg_rows('m1',mp,ms,mc,(man['checkpoints']['m1_supermimic']['metadata']['training_wall_seconds'],sc['iterations']),sc['real_transitions'],(sc['wm_update_events'],sc['actor_updates'])),'M01_m1_training_config','来源：data/m1_*config.json、*completion.json、checkpoint；同真实reward，SM另有权重1.0的H32终点距离。',[.26,.35,.39],14)
pL=load('m1_ppo_forward')['metrics'];sL=load('m1_supermimic_forward')['metrics'];pM=load('m1_ppo_manipulation')['metrics'];sM=load('m1_supermimic_manipulation')['metrics']
rows=[['L：命令 vx=0.12，实际 vx (m/s)',num(pL['L_actual_vx'],5),num(sL['L_actual_vx'],6)],['L：XY RMSE↓ (m/s)',num(pL['L_velocity_xy_rmse'],5),num(sL['L_velocity_xy_rmse'],5)],['L：yaw RMSE↓ (rad/s)',num(pL['L_yaw_rate_rmse'],5),num(sL['L_yaw_rate_rmse'],5)],['L：跌倒 / 结束 episode',f"0/{pL['L_completed_episodes']}",f"0/{sL['L_completed_episodes']}"],['M：成功 / 已结束 goal',f"{pM['M_successful_goals']}/{pM['M_completed_goals']} ({pct(pM['M_goal_success_rate'])})",f"{sM['M_successful_goals']}/{sM['M_completed_goals']} ({pct(sM['M_goal_success_rate'])})"],['M：成功数 / env / min↑',num(pM['M_successes_per_M_env_minute']),num(sM['M_successes_per_M_env_minute'])],['M：成功目标平均耗时↓ (s)',num(pM['M_successful_goal_time_s']),'无成功样本'],['M：时间平均 TCP 距离↓ (m)',num(pM['M_distance_m']),num(sM['M_distance_m'])],['M：跌倒 / 结束 episode',f"34/{pM['M_completed_episodes']} ({pct(pM['M_fall_rate'])})",f"0/{sM['M_completed_episodes']}"],['M：截止时活跃 goal（不进分母）',str(pM['M_censored_goals']),str(sM['M_censored_goals'])]]
table('M1-Piper｜PPO 部分学会到点，SM 低位停滞','v2；各64环境、60s、seed=123；L固定命令，M按训练分布连续换点',['分项指标','PPO 3.5 h','SuperMimic 3.5 h'],rows,'M02_m1_final_results','来源：data/m1_{ppo,supermimic}_{forward,manipulation}.json；goal成功与episode跌倒使用不同分母。',[.49,.26,.25],14.5)
# Exact finite-difference examples; source labels are selected, values are read.
probe=load('m1_wm_response_probe_eps0.01');rows=[]
for direction,h,metric,label in [('leg_calf_common',8,'dvx','四小腿同向 / 速度vx'),('leg_calf_common',8,'ddistance','四小腿同向 / 目标距离'),('wheel_front_rear',32,'ddistance','轮子前后差分 / 目标距离')]:
 r=next(x for x in probe['responses'][direction] if x['horizon']==h)
 suffix='dvx_dalpha' if metric=='dvx' else 'ddistance_dalpha'
 rows.append([label,str(h),f"{r['real_'+suffix]:+.4f}",f"{r['wm_fd_'+suffix]:+.4f}",f"{r['wm_autograd_'+suffix]:+.4f}",'符号相反'])
table('M1-Piper｜模型梯度可算，但物理方向可能错误','名义姿态附近，归一化动作扰动 ±0.01；另有 ±0.03 复核；非全分布结论',['扰动方向 / 输出','H','MuJoCo 差分','WM 自身差分','WM autograd','判定'],rows,'M05_m1_wm_gradient_audit','来源：data/m1_wm_response_probe_eps0.01.json；固定目标[1.5,0,0.8]；12分支常动作，未更新参数。',[.31,.06,.16,.16,.17,.14],14,bottom=.31)
# Additional readable reward cards.
e=gs['env'];g=e['go2_gait']
rows=[['速度任务',f"XY {e['velocity_reward_linear_tracking']}；yaw {e['velocity_reward_yaw_tracking']}；方向 {e['velocity_reward_direction']}",'速度核 σ²=0.09；扣除静止基线'],['姿态 / 高度','upright +0.5；低高度 −5','高度罚只在 h<0.22m 生效'],['速度稳定','vz² −1；ωxy² −0.05','不直接指定步频或摆腿相位'],['动作 / 加速度','Δa平方和 −0.03；qdd²和 −2.5e−7','真实qdd测量 / 想象控制周期差分'],['关节 / 力矩','偏离 −0.02；越限 −1；τ²和 −1e−5','真实力矩 / 想象限幅PD代理'],['足端抬高',str(g['feet_clearance'])+'（关闭）','5cm参数保留，但不参与优化'],['对角协调',str(g['diagonal_gait']),'近地几何软约束；不保证标准Trot'],['支撑脚滑移',str(g['feet_slip']),'近地足端世界水平速度平方']]
table('Go2｜真实奖励相同，想象使用物理代理','所有数值为乘 dt=0.02 前的奖励率；详细公式见 Go2 分析文档',['类别','有效权重','行为含义 / 限制'],rows,'G05_go2_reward_summary','来源：保存config + velocity_reward.py、go2_gait.py；力矩和关节加速度的真实/想象实现并非逐值相同。',[.17,.43,.40],14)
r=ms['env']['m1_reward']
rows=[['L 速度',f"XY {r['linear_tracking']} / yaw {r['yaw_tracking']}",'仅L；σ=0.15m/s、0.25rad/s'],['M 距离',f"粗 {r['reach_coarse']} / 精 {r['reach_fine']}",'σ=1.50m / 0.08m'],['M 进度 / 近点低速',f"{r['reach_progress']} / {r['reach_hold']}",'进度tanh；低速核0.10m/s'],['M 时间成本',f"−{r['distance_time_weight']} d/(d+{r['distance_time_scale_m']})",'远处成本有界，距离梯度逐渐减弱'],['M 成功事件',f"+{r['success_bonus']:.0f} / goal",'8cm内稳定首次到达；布尔事件'],['SM 独有终点',f"−γ³² × {ms['policy_trainer']['m1_terminal_distance_weight']:.1f} × d32",'不乘dt；不进入PPO或Critic拟合']]
table('M1-Piper｜过程、事件与终点三种信号','v2大范围目标；共享真实reward ≠ 相同总优化目标',['类别','实际权重 / 公式','任务分支与尺度'],rows,'M06_m1_task_rewards','来源：保存 M1RewardCfg、m1_reward.py、M1RewardTask.terminal_reward、PolicyTrainer.update。',[.23,.33,.44],15,bottom=.2)
# Export CSV for every evaluated command, retaining exact precision.
with (DATA/'go2_results.csv').open('w') as f:
 fields=['algorithm','vx_cmd','vy_cmd','wz_cmd','episodes','mean_vx','mean_vy','mean_yaw_rate','velocity_rmse','yaw_rate_rmse','fall_rate','stance_slip_rms_m_s','diagonal_support_fraction']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for a in ['ppo','supermimic']:
  for z in G['go2_'+a]:w.writerow({'algorithm':a,**dict(zip(['vx_cmd','vy_cmd','wz_cmd'],z['command_body_xy_yaw'])),**{k:z[k] for k in fields[4:]}})
