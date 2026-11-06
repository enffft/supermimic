"""Bounded nominal real-simulation policy/posture audit, no optimization."""
import sys
from pathlib import Path
import json
import torch
root=Path.cwd(); sys.path.insert(0,str(root/'scripts'))
from play import _load_checkpoint,_M1RewardDictPolicy
from supermimic.training.m1_reward import evaluation_args
from supermimic.envs import make_env
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.tasks.m1_reward import m1_actor_goal
out=Path(__file__).parent;tag='v2_lm50_2048_3p5h_seed2026'
torch.set_num_threads(1)
results={}
for label,directory,name in [('PPO','m1_ppo','policy_inference.pt'),('SuperMimic','m1_reward','ckpt_latest.pt')]:
 policy,config,_=_load_checkpoint(root/'logs'/directory/tag/name,'cuda:0')
 env=make_env(evaluation_args(config,num_envs=4,task_type='balanced',seed=123),device='cuda:0')
 samples=[];falls={'L':0,'M':0};goal_sensitivity={}
 try:
  obs,_=env.reset();pol=_M1RewardDictPolicy(policy,env,(.12,0.,0.))
  for step in range(250):
   with torch.no_grad():action=pol(obs)
   obs,_,term,tout,_=env.step(action)
   cache=env._supermimic_m1_reward_cache; state=cache['state_next'];context=cache['context']
   gravity=state.new_zeros(4,3);gravity[:,2]=-1.
   tilt=torch.acos((-quat_apply_inverse(state[:,3:7],gravity)[:,2]).clamp(-1,1))
   for task in ('L','M'):
    mask=context[:,0]<.5 if task=='L' else context[:,0]>=.5
    falls[task]+=int(term[mask].sum())
    if step>=50:
     samples.append({'task':task,'height':state[mask,2].tolist(),'tilt_deg':(tilt[mask]*180/torch.pi).tolist(),
                     'action_abs_mean':float(action[mask].abs().mean()),
                     'action_over_0p95':float((action[mask].abs()>.95).float().mean()),
                     'tanh_local_slope':float((1-action[mask].square()).mean()),
                     'wheel_rate_abs_rad_s':float(state[mask][:,[38,42,46,50]].abs().mean())})
  command=env.command_manager.get_term('m1_prior_joint');state=pol.state_io.read()
  for task in ('L','M'):
   gplus=m1_actor_goal(state,command.goal).clone();gminus=gplus.clone()
   if task=='L':
    for g,v in [(gplus,.12),(gminus,-.12)]:
     g[:,:3]=g.new_tensor([v,0.,0.]);g[:,3:6]=0.;g[:,6:]=g.new_tensor([1.,0.])
   else:
    for g,v in [(gplus,1.5),(gminus,-1.5)]:
     g[:,:3]=0.;g[:,3:6]=g.new_tensor([v,0.,.8])-state[:,:3];g[:,6:]=g.new_tensor([0.,1.])
   with torch.no_grad():plus=policy(obs['policy'],gplus);minus=policy(obs['policy'],gminus)
   goal_sensitivity[task]=float((plus-minus).abs().mean())
  summary={}
  for task in ('L','M'):
   rows=[r for r in samples if r['task']==task];heights=[h for r in rows for h in r['height']];tilts=[h for r in rows for h in r['tilt_deg']]
   summary[task]={'height_mean_m':sum(heights)/len(heights),'height_min_m':min(heights),'height_max_m':max(heights),
                  'tilt_mean_deg':sum(tilts)/len(tilts),'action_abs_mean':sum(r['action_abs_mean'] for r in rows)/len(rows),
                  'action_over_0p95_fraction':sum(r['action_over_0p95'] for r in rows)/len(rows),
                  'tanh_local_slope_mean':sum(r['tanh_local_slope'] for r in rows)/len(rows),'falls':falls[task]}
  results[label]={'summary':summary,'goal_counterfactual_mean_abs_action_difference':goal_sensitivity}
  print(label,json.dumps(results[label],indent=2),flush=True)
 finally:
  env.close()
(out/'policy_posture_probe.json').write_text(json.dumps({'method':'4 envs 5s; balanced L/M, L vx=.12; no DR/noise; posture/action statistics from t=1..5s; no updates','limitations':'Small nominal probe, not a full robustness evaluation; goal sensitivity changes goal while holding same observed state/history', 'results':results},indent=2)+'\n')
