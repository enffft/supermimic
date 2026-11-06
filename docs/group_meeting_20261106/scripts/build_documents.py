"""Build evidence tables and navigation; explanatory manuscripts are kept as Markdown."""
from pathlib import Path
import json, csv, hashlib
OUT=Path(__file__).resolve().parents[1];DATA=OUT/'data';ROOT=OUT.parents[1]
def load(n):return json.loads((DATA/(n+'.json')).read_text())
def table_cell(value):
 # TeX norm delimiters must survive Markdown pipe-table parsing.
 return str(value).replace(r'\|',r'\Vert').replace('|',r'\vert ')
def table(headers,rows):return '\n'+'| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(table_cell(x) for x in r)+' |\n' for r in rows)+'\n'
def write(n,s): (OUT/n).write_text(s,encoding='utf-8')
M=load('experiment_manifest');G=load('go2_results');C=load('training_curves')
gp=load('go2_ppo_config');gs=load('go2_supermimic_config');mp=load('m1_ppo_config');ms=load('m1_supermimic_config')
# Machine-derived appendix, reused in primary manuscripts.
rows=[]
for name,p,s in [('Go2',gp,gs),('M1-Piper',mp,ms)]:
 nj=s['policy']['num_joints'];hist=s['policy']['history'];goal=s['policy']['goal_dim'];sd=13+2*nj;xi=nj+4
 rows.append([name,f'{hist}×{3*nj+6}+{goal}={hist*(3*nj+6)+goal}',f'{sd}+{goal}+{xi}+4={sd+goal+xi+4}',f'{sd}+{nj}+{xi}={sd+nj+xi}',12+2*nj,M['checkpoints']['go2_supermimic' if name=='Go2' else 'm1_supermimic']['networks']['policy']['linear_parameter_count']])
net=table(['机器人','Actor输入','Critic输入','WM输入','WM输出','Actor参数（不含std）'],rows)
# Reward details: coefficient, normalization, unit, bounded scale and per-term consequences.
GO2=[
('XY速度跟踪','track_lin_vel_xy','velocity_reward_linear_tracking',r'$w[\exp(-\|v^b_{xy}-u_{xy}\|^2/0.09)-\exp(-\|u_{xy}\|^2/0.09)]$','m/s平方按0.09归一化；率≤3','直接要求真实机体跟踪；低抬脚仍可高回报'),
('命令方向','command_direction','velocity_reward_direction',r'$w\frac{\|u_{xy}\|}{\|u_{xy}\|+0.1}\tanh[\frac{v^b_{xy}\cdot u_{xy}}{\max(\|u_{xy}\|,10^{-4})\,0.3}]$','有符号、绝对值<0.5；0.1/0.3单位m/s','鼓励沿目标方向运动；纯yaw时为0'),
('偏航跟踪','track_yaw_rate','velocity_reward_yaw_tracking',r'$w[\exp(-(\omega^b_z-u_z)^2/0.09)-\exp(-u_z^2/0.09)]$','角速度误差按0.09 (rad/s)²归一化；率≤1.2','同一0.09用于XY与yaw；不要误写成不同σ'),
('直立','upright',0.5,r'$w\exp(-\|g^b_{xy}\|^2/0.16)$','重力为单位向量；[0,0.5]','倾斜约束；并非姿态硬限制'),
('竖直速度','base_lin_vel_z',-1.,r'$w(v^w_z)^2$','(m/s)²；无固定负下界，最大0','抑制跳动；也可能使保守步态有利'),
('Roll/Pitch角速度','base_ang_vel_xy',-.05,r'$w\|\omega^b_{xy}\|^2$','(rad/s)²；最大0','抑制机体快速摆动'),
('动作变化','action_rate',-.03,r'$w\sum_{j=1}^{12}(a_j-a_{j,prev})^2$','归一化动作平方和；[-1,1]内率≥−1.44','平滑动作，但不约束恒定边界动作'),
('关节力矩','joint_torque',-1e-5,r'$w\sum_j\tau_j^2$','(N·m)²；最大0；力矩受执行器限幅','真实MuJoCo测量；想象使用限幅PD代理，不能宣称逐值相等'),
('关节加速度','joint_acceleration',-2.5e-7,r'$w\sum_j\ddot q_j^2$','(rad/s²)²；最大0','真实joint_acc；想象(下一qd−当前qd)/0.02，时间尺度不同'),
('偏离默认姿态','joint_deviation',-.02,r'$w\sum_j(q_j-q^0_j)^2$','rad²平方和；最大0','压制极端关节姿态，也有默认姿态偏置'),
('软限位','joint_limit',-1.,r'$w\sum_j([q^{min}_j-q_j]_+^2+[q_j-q^{max}_j]_+^2)$','rad²；最大0；界内为0','不限制软限位内的极限动作'),
('低机身高度','base_height_low',-5.,r'$w[0.22-h]_+^2$','m²；最大0；h=0.20时率−0.002','0.22m以上无高度引导，不是追踪默认高度'),
('足端抬高','feet_clearance','feet_clearance',r'$w\,m\,\operatorname{mean}_i[e^{-((h_i-.05)/.025)^2}\tanh((\sqrt{\|v_{i,xy}\|^2+10^{-6}}-.001)/.5)]$','当前恒为0；高度m、速度m/s','明确关闭；不能将旧较好步态归功于抬脚奖励'),
('对角协调','diagonal_gait','diagonal_gait',r'$w\,m[\frac12((s_{FL}-s_{RR})^2+(s_{FR}-s_{RL})^2)+(\frac{\sum_i s_i-2}{2})^2]$','无量纲；当前负权重；率在[−0.15,0]','软几何协调，无相位时钟，无固定步频'),
('足滑','feet_slip','feet_slip',r'$w\operatorname{mean}_i[\sigma((.015-h_i)/.005)\|v^w_{i,xy}\|^2]$','(m/s)²；最大0；平地几何近地权重','权重较小且依赖正确WM速度导数；存在不等于足滑消失'),
]
rows=[]
for cn,en,key,formula,scale,meaning in GO2:
 w=(gs['env'][key] if key in gs['env'] else gs['env']['go2_gait'][key]) if isinstance(key,str) else key
 rows.append([cn+' / `'+en+'`',f'{w:g}',formula,scale,'两算法真实奖励同核；SM想象按预测状态计算'+('，用物理代理' if en in ['joint_torque','joint_acceleration'] else ''),meaning])
go2_rewards=table(['中文 / 源码名','实际权重w','加权奖励率公式','归一化、单位与尺度','真实与想象','行为影响'],rows)
# M1 complete reward table from actual saved coefficients.
r=ms['env']['m1_reward']
M1=[
('L线速度','l_linear','linear_tracking','L',r'$w e^{-\|v^b_{xy}-u_{xy}\|^2/0.15^2}$','m/s误差平方；[0,3]/s','稠密可微；只要求速度，不要求抬腿'),
('L偏航','l_yaw','yaw_tracking','L',r'$w e^{-(\omega_z-u_z)^2/0.25^2}$','rad/s误差；[0,1]/s','稠密可微；侧移/转向能力仍需实测'),
('粗到点','m_reach_coarse','reach_coarse','M',r'$w e^{-d^2/1.5^2}$','距离m；[0,2]/s','稠密可微；远端梯度仍弱；不保证可达'),
('精到点','m_reach_fine','reach_fine','M',r'$w e^{-d^2/0.08^2}$','距离m；[0,1]/s','远离8cm时接近零'),
('接近进度','m_progress','reach_progress','M',r'$w\tanh((d_{prev}-d)/(0.02\cdot0.5))$','有限差分接近速度按0.5m/s缩放；[−0.25,0.25]/s','有符号可微；不是保证策略不绕路的潜在函数差'),
('近点低速','m_hold','reach_hold','M',r'$w e^{-d^2/.08^2}e^{-\|(p^{TCP}_{next}-p^{TCP}_{prev})/.02\|^2/.10^2}$','TCP速度m/s；[0,1]/s','稠密可微；成功马上换点使作用窗口短；不等于保持0.5s'),
('距离相关时间成本','m_distance_time','distance_time_weight','M',r'$-w d/(d+0.5)$','距离m；(−0.5,0]/s','稠密可微；不是常数时间罚，远处导数衰减'),
('首次成功','m_success','success_bonus','M',r'$(w/\Delta t)\mathbf1[d<.08\land h\ge .20\land tilt<70^\circ]$','率最高600/s；乘dt后每goal +12','布尔事件；几乎处处零路径导数；真实成功换点防重复'),
('直立','upright','upright','L/M',r'$w e^{-(1+g^b_z)/.1}$','单位重力；[0,0.5]/s','可微；倾斜后奖励及恢复导数均减小'),
('底盘高度','height','height','L/M',r'$w e^{-((h-.47)/.10)^2}$','h单位m；[0,0.5]/s','可微；h≈.219时位于核尾部，恢复导数很弱'),
('竖直速度','vertical_velocity','vertical_velocity','L/M',r'$w(v^b_z)^2$','(m/s)²；最大0，无固定负下界','可微；静止低姿态不被此项惩罚'),
('横滚俯仰角速度','roll_pitch_rate','roll_pitch_rate','L/M',r'$w(\omega_x^2+\omega_y^2)$','(rad/s)²；最大0','可微；静态倾斜不产生此项罚'),
('腿默认姿态','leg_pose','leg_pose','L/M',r'$w\sum_{j\in leg}(q_j-q^0_j)^2$','12腿关节rad²；最大0','可微；未包含轮子或臂'),
('臂收纳','l_arm_carry','arm_carry','L',r'$w\sum_{j=16}^{21}(q_j-q^0_j)^2$','6臂关节rad²；最大0','只对L，M不惩罚离开收纳姿态'),
('动作变化','action_rate','action_rate','L/M',r'$w\operatorname{mean}_{22}(a-a_{prev})^2$','无量纲；有界动作率≥−0.04','可微；恒定±1也为0'),
('位置软越限','joint_limit','joint_limit','L/M',r'$w\sum_{j\notin wheel}([q^{min}_j-q_j]_+^2+[q_j-q^{max}_j]_+^2)$','18位置关节rad²；最大0','轮角度排除；界内动作饱和不受约束'),
('关节速度','joint_velocity','joint_velocity','L/M',r'$w\operatorname{mean}_{22}\dot q^2$','(rad/s)²；最大0','含轮子；不同物理动作组没有单独归一化'),
('关节加速度','joint_acceleration','joint_acceleration','L/M',r'$w\operatorname{mean}_{22}((\dot q_{next}-\dot q_{prev})/.02)^2$','(rad/s²)²；最大0','真实与想象都用控制周期差分'),
('控制幅度代理','control_effort','control_effort','L/M',r'$w\operatorname{mean}_{22}a^2$','无量纲；有界动作率≥−0.005','非测量力矩/功率；全饱和每步仅−0.0001'),
('M底盘速度','m_base_speed','mani_base_speed','M',r'$w(\|v^b_{xy}\|^2+\omega_z^2)$','m/s和rad/s平方直接混合；最大0','M命令为0；移动会付成本，与远点快速到达有权衡'),
]
rows=[]
for cn,en,key,mask,formula,scale,local in M1:
 rows.append([cn+' / `'+en+'`',f'{r[key]:g}',mask,formula,scale,local,'共享真实/想象解析核；事件/终止处理见后文'])
m1_rewards=table(['名称 / 源码名','权重w','分支','加权奖励率','单位 / 最大量级','性质与局部最优风险','一致性'],rows)
# Derived diagnostics and formal results.
fixed=load('m1_fixed_input_snapshots')['snapshots'];grad=load('m1_gradient_components')['snapshots']
gradtable=table(['快照min','过程梯度L2','Critic梯度L2','终点梯度L2','总梯度L2'],[[f"{z['hours']*60:.0f}",*[f"{z['components'][k]['parameter_gradient_norm']:.6f}" for k in ['dense','critic','terminal','total']]] for z in grad])
wm=load('m1_wm_response_probe')['metrics'];wmtable=table(['H','时域s','底盘XY误差m','TCP误差m','vx RMSE m/s','全状态NSE'],[[z['horizon'],z['time_s'],f"{z['base_xy_error_m']:.4f}",f"{z['tcp_error_m']:.4f}",f"{z['base_vx_rmse']:.4f}",f"{z['nse_mean']:.3f}"] for z in wm if z['horizon'] in [1,8,16,32]])
goalbins=''
for field,bins,label in [('height',[(.03,.3),(.3,.8),(.8,1.100001)],'目标世界高度m'),('initial_distance_m',[(0,.5),(.5,1.5),(1.5,2.5),(2.5,100)],'初始TCP距离m')]:
 rows=[]
 for lo,hi in bins:
  row=[f'[{lo:g}, +∞)' if hi==100 else f'[{lo:g}, {hi:g})']
  for a in ['ppo','supermimic']:
   goals=load('m1_'+a+'_manipulation')['goals'];gs1=[g for g in goals if not g.get('censored',False) and lo<=(g['target_world'][2] if field=='height' else g[field])<hi];n=len(gs1);s=sum(g['success'] for g in gs1);row.append(f'{s}/{n} = {100*s/n:.2f}%' if n else '无结束样本')
  rows.append(row)
 goalbins+='\n### 按'+label+'分桶\n'+table([label,'PPO 成功/结束goal','SM 成功/结束goal'],rows)
# Store machine-generated fragments for transparent regeneration.
for n,s in [('network_dimensions',net),('go2_reward_table',go2_rewards),('m1_reward_table',m1_rewards),('m1_wm_errors',wmtable),('m1_gradients',gradtable),('m1_goal_bins',goalbins)]:(DATA/(n+'.md')).write_text(s)
