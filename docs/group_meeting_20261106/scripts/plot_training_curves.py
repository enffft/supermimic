from style import *
import numpy as np
G=load('go2_results');colors=[BLUE,TEAL]
f=canvas('Go2｜覆盖全部九条指令的跟踪误差','从零 2.5 h；误差越低越好；提前跌倒的短轨迹仍需结合跌倒率解读')
axes=[f.add_axes([.07,.52,.87,.30]),f.add_axes([.07,.15,.87,.28])];x=np.arange(9);labels=[trip(r['command_body_xy_yaw']) for r in G['go2_ppo']]
for ax,key,ylabel in zip(axes,['velocity_rmse','yaw_rate_rmse'],['XY RMSE (m/s)','yaw RMSE (rad/s)']):
 for k,(alg,color) in enumerate(zip(['ppo','supermimic'],colors)):
  rows=G['go2_'+alg];xx=x+(k-.5)*.34;bars=ax.bar(xx,[r[key] for r in rows],.32,color=color,label=['PPO','SuperMimic'][k])
  for z,r in zip(bars,rows):
   if r['fall_rate']>0:ax.text(z.get_x()+z.get_width()/2,z.get_height()+.006,f"跌{100*r['fall_rate']:.0f}%",ha='center',fontsize=10,color=RED)
 ax.set_ylabel(ylabel);ax.set_xticks(x,labels,fontsize=11);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True);ax.set_ylim(0,ax.get_ylim()[1]*1.2)
axes[0].legend(loc='upper right',ncol=2,frameon=False)
finish(f,'G03_go2_velocity_comparison','来源：data/go2_results.json；各指令64个结束episode；标签为vx/vy/wz；单训练seed、名义初态，无DR或观测噪声。')
# M success and height bins, complete-goal denominators.
M={a:load('m1_'+a+'_manipulation') for a in ['ppo','supermimic']}
f=canvas('M1-Piper｜PPO 成功率 71.2%，低位目标仍困难','v2 · 3.5 h · 64环境 × 60s；已结束goal成功率，截止活跃goal单独删失')
a=f.add_axes([.075,.19,.35,.61]);names=['PPO','SuperMimic'];vals=[M[k]['metrics']['M_goal_success_rate']*100 for k in M]
a.bar(names,vals,color=colors,width=.54)
for i,k in enumerate(M):
 m=M[k]['metrics'];a.text(i,vals[i]+3,f"{vals[i]:.1f}%\n{m['M_successful_goals']}/{m['M_completed_goals']}",ha='center',fontsize=17)
a.set_ylim(0,112);a.set_ylabel('成功 / 已结束 goal (%)');a.set_title('全部操作目标',fontsize=17)
b=f.add_axes([.54,.19,.40,.61]);bins=[(.03,.30),(.30,.80),(.80,1.100001)];xx=np.arange(3)
for j,(k,c) in enumerate(zip(M,colors)):
 vals=[];cnt=[]
 for lo,hi in bins:
  goals=[g for g in M[k]['goals'] if not g.get('censored',False) and lo<=g['target_world'][2]<hi];cnt.append((sum(g['success'] for g in goals),len(goals)));vals.append(100*cnt[-1][0]/cnt[-1][1] if cnt[-1][1] else 0)
 bars=b.bar(xx+(j-.5)*.35,vals,.33,color=c,label=names[j])
 for bar,(s,n),v in zip(bars,cnt,vals):b.text(bar.get_x()+bar.get_width()/2,v+2,f'{s}/{n}',ha='center',fontsize=12)
b.set_xticks(xx,['低 0.03–0.30m','中 0.30–0.80m','高 0.80–1.10m'],fontsize=12);b.set_ylim(0,112);b.set_title('按目标高度分桶',fontsize=17);b.legend(frameon=False);b.grid(axis='y',alpha=.15)
finish(f,'M03_m1_reach_success','来源：data/m1_*_manipulation.json；PPO另有27个活跃goal；成功换点后轨迹不再逐目标配对；高度桶有结束样本选择偏差。')
# Decay diagnostics, strictly distinguish physical vs synthetic data.
f=canvas('M1-Piper｜从策略漂移到饱和：四类独立证据','训练日志、固定输入快照、真实姿态和冻结WM响应测试；这些不是同一条时序曲线')
axs=[f.add_axes([.075,.53,.37,.27]),f.add_axes([.56,.53,.37,.27]),f.add_axes([.075,.15,.37,.27]),f.add_axes([.56,.15,.37,.27])]
tb=load('training_curves')['m1_sm']['series'];row=tb['policy/action_abs_mean'];t0=min(r['wall_time'] for r in tb['time/hours']);axs[0].plot([(r['wall_time']-t0)/3600 for r in row],[r['value'] for r in row],color=TEAL);axs[0].set(xlabel='相对首个日志时间 (h)',ylabel='平均 |action|',title='训练想象动作幅度');axs[0].axhline(.95,color=RED,ls='--',lw=1)
snap=load('m1_fixed_input_snapshots')['snapshots'];axs[1].plot([r['hours'] for r in snap],[100*r['fixed_input_saturated_fraction_0p95'] for r in snap],color=TEAL);axs[1].set(xlabel='checkpoint训练时间 (h)',ylabel='|a|>0.95 的比例 (%)',title='固定物理输入、上一动作历史=0')
post=load('m1_policy_posture_probe')['results'];xx=np.arange(2)
for j,(k,c) in enumerate(zip(['PPO','SuperMimic'],colors)):
 vals=[post[k]['summary'][t]['height_mean_m'] for t in ['L','M']];axs[2].bar(xx+(j-.5)*.32,vals,.3,label=k,color=c)
 for i,v in enumerate(vals):axs[2].text(xx[i]+(j-.5)*.32,v+.012,f'{v:.3f}',ha='center',fontsize=11)
axs[2].axhline(.2,color=RED,ls='--',label='跌倒高度线');axs[2].set_xticks(xx,['L','M']);axs[2].set(ylabel='root height (m)',title='真实仿真最终姿态：4env、1–5s',ylim=(0,.6));axs[2].legend(fontsize=10,ncol=2,frameon=False)
wm=load('m1_wm_response_probe')['metrics'];sel=[r for r in wm if r['horizon'] in [1,8,16,32]]
for key,label,c in [('base_xy_error_m','底盘XY',BLUE),('tcp_error_m','TCP',TEAL)]:axs[3].plot([r['horizon'] for r in sel],[r[key] for r in sel],marker='o',color=c,label=label)
axs[3].axhline(.08,color=RED,ls='--',label='8cm成功半径');axs[3].set(xlabel='常动作预测步数',ylabel='位置误差 (m)',title='局部WM测试：12分支、±0.03');axs[3].legend(fontsize=10,frameon=False)
for a in axs:a.grid(alpha=.14)
finish(f,'M04_m1_training_failure','来源：data/training_curves、m1_fixed_input_snapshots、m1_policy_posture_probe、m1_wm_response_probe；不作全分布或因果贡献估计。')
# Gradient decomposition, mechanism versus fact.
f=canvas('M1-Piper｜Critic 梯度可主导 Actor，终点项并非始终最强','冻结6份checkpoint；相同4个名义起点；H32；各项均已除H；裁剪前Actor参数梯度范数')
a=f.add_axes([.075,.17,.47,.62]);snap=load('m1_gradient_components')['snapshots'];xx=np.arange(len(snap))
for off,key,color,label in [(-.24,'dense',BLUE,'过程奖励'),(0,'critic',TEAL,'Critic终值'),(.24,'terminal','#D69A37','终点距离')]:a.bar(xx+off,[r['components'][key]['parameter_gradient_norm'] for r in snap],.22,color=color,label=label)
a.set_yscale('log');a.set_xticks(xx,[f"{r['hours']*60:.0f}m" for r in snap]);a.set_ylabel('梯度 L2 (log scale)');a.legend(frameon=False);a.grid(axis='y',alpha=.15)
b=f.add_axes([.64,.17,.30,.62]);fb=load('m1_action_history_feedback')['results']
for r in fb:
 if any(n in r['checkpoint'] for n in ['060','090','100','120','latest']):b.plot([q['step']+1 for q in r['rows']],[q['abs_action_mean'] for q in r['rows']],marker='.',label=r['checkpoint'].replace('ckpt_wall_','').replace('ckpt_','').replace('.pt',''))
b.set(xlabel='纯动作历史反馈次数',ylabel='平均 |action|',title='固定物理状态，无仿真/优化');b.legend(fontsize=11,frameon=False);b.grid(alpha=.15)
finish(f,'M07_m1_gradient_and_feedback','来源：data/m1_gradient_components、m1_action_history_feedback；不是历史Replay更新归因；Critic方向尚无真实价值梯度真值。')
# Supplementary source curves, no incomparable return scales overlaid.
f=canvas('训练日志｜奖励曲线不能替代独立物理评测','不同任务/算法的日志定义不同；这里分别展示优化诊断，不比较绝对return高低')
axs=[f.add_axes([.07,.54,.39,.27]),f.add_axes([.56,.54,.38,.27]),f.add_axes([.07,.15,.39,.27]),f.add_axes([.56,.15,.38,.27])]
alltb=load('training_curves')
for a,k in zip(axs[:2],['go2_ppo','m1_ppo']):
 r=alltb[k]['series']['Train/mean_reward'];t=min(v['wall_time'] for vs in alltb[k]['series'].values() for v in vs);a.plot([(v['wall_time']-t)/3600 for v in r],[v['value'] for v in r],color=BLUE);a.set(title=k+'：已结束回合训练奖励',xlabel='相对首个日志 (h)',ylabel='训练回合reward')
for a,k in zip(axs[2:],['go2_sm','m1_sm']):
 ss=alltb[k]['series'];t=min(v['wall_time'] for vs in ss.values() for v in vs)
 for tag,label,c in [('wm/nse_1step','1-step NSE',BLUE),('wm/nse_8step','8-step NSE',TEAL)]:
  r=ss[tag];a.plot([(v['wall_time']-t)/3600 for v in r],[v['value'] for v in r],color=c,label=label)
 a.set(title=k+'：Replay预测误差',xlabel='相对首个日志 (h)',ylabel='平均归一化平方误差');a.legend(fontsize=11,frameon=False)
for a in axs:a.grid(alpha=.15)
finish(f,'A03_training_curves','来源：原始TensorBoard → data/training_curves.json；重复step保留事件顺序；早期静止分布的小NSE不等于动作Jacobian正确。')
# Gait fair comparison all nine commands, compact metrics only.
rows=[]
for p,s in zip(G['go2_ppo'],G['go2_supermimic']):rows.append([trip(p['command_body_xy_yaw']),f"{p['mean_base_height']:.3f}/{s['mean_base_height']:.3f}",f"{p['stance_slip_rms_m_s']:.3f}/{s['stance_slip_rms_m_s']:.3f}",f"{100*p['diagonal_support_fraction']:.1f}/{100*s['diagonal_support_fraction']:.1f}",'/'.join(f'{100*v:.1f}' for v in p['mean_episode_peak_foot_clearance_m']),'/'.join(f'{100*v:.1f}' for v in s['mean_episode_peak_foot_clearance_m'])])
table('Go2｜等时长算法比较的几何步态指标','从零2.5h；高度/滑移列为PPO/SM；跌倒情况必须同时查G02',['vx/vy/wz','机身高度 m','足滑 RMS m/s','对角支撑 %','PPO 四足峰高 cm','SM 四足峰高 cm'],rows,'G06_go2_fair_gait','来源：data/go2_results.json；足端指标排除起步1s、按近地高度判定支撑；跌倒时峰高大不代表优良步态。',[.18,.15,.16,.15,.18,.18],12.5)
