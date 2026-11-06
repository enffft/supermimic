"""Post-training nominal-pose response audit. NO optimizer updates."""
from dataclasses import asdict
import json
import sys
from pathlib import Path
import torch
from supermimic.baselines.go2_supermimic import apply_config
from supermimic.config import TrainConfig
from supermimic.envs import make_env
from supermimic.training.m1_reward import evaluation_args
from supermimic.robots.spec import get_robot_spec
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel
from supermimic.policy.nets import Policy
from supermimic.tasks.m1_reward import m1_actor_goal
from supermimic.tasks.world_point_reward import tcp_position_world,m1_tcp_proxy_indices

root=Path.cwd(); tag='v2_lm50_2048_3p5h_seed2026'; out=Path(__file__).parent
checkpoint=root/'logs/m1_reward'/tag/'ckpt_latest.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
cfg=apply_config(TrainConfig(),saved['config'])
torch.set_num_threads(1)
args=evaluation_args(saved['config'],num_envs=12,task_type='L',seed=123)
env=make_env(args,device='cuda:0')
try:
 obs,_=env.reset(); robot=env.scene['robot']
 spec=get_robot_spec('m1_piper',m1_pose_profile=args.m1_pose_profile,m1_action_contract=args.m1_action_contract)
 fk=G1Kinematics(load_robot_spec_model('m1_piper'),device=env.device)
 heads=AnalyticHeads(fk,spec.tracked_body_names,spec.anchor_body_name,spec.root_body_name,
                     robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone(),
                     default_base_height=spec.default_base_height,reward_anchor_at_start=True)
 wm=WorldModel(cfg.world_model,heads,env.device).eval().requires_grad_(False)
 wm.load_state_dict(saved['wm'],strict=True)
 policy=Policy(cfg.policy).to(env.device).eval().requires_grad_(False)
 policy.load_state_dict(saved['policy'],strict=True)
 command=env.command_manager.get_term('m1_prior_joint')
 command.velocity_command_body[:]=torch.tensor([.12,0.,0.],device=env.device)
 recorder=env.recorder_manager.get_term('replay')
 recorder.prev_goal.copy_(command.goal)
 state0=recorder.state_io.read().clone(); xi=recorder.ctx.read().clone()
 with torch.no_grad():
  base=policy(obs['policy'],m1_actor_goal(state0,command.goal))
 names=['wheel_common','wheel_left_right','wheel_front_rear','arm_joint1','arm_joint2','leg_calf_common']
 directions=torch.zeros(6,22,device=env.device)
 directions[0,[3,7,11,15]]=1.
 directions[1,[3,7,11,15]]=torch.tensor([1.,-1.,1.,-1.],device=env.device)
 directions[2,[3,7,11,15]]=torch.tensor([1.,1.,-1.,-1.],device=env.device)
 directions[3,16]=1.; directions[4,17]=1.; directions[5,[2,6,10,14]]=1.
 eps=float(sys.argv[1]) if len(sys.argv)>1 else .03
 a=(base + directions.repeat_interleave(2,dim=0)*torch.tensor([eps,-eps]*6,device=env.device)[:,None]).clamp(-1,1)
 pred=state0.clone(); alive=torch.ones(12,dtype=torch.bool,device=env.device)
 rows=[]; snapshots={}
 tcp_ids=m1_tcp_proxy_indices(fk)
 target=torch.tensor([1.5,0.,.8],device=env.device)
 for t in range(1,33):
  with torch.no_grad():
   pred=wm.step(pred,a,xi)
   obs,_,term,tout,_=env.step(a)
   actual=env._supermimic_m1_reward_cache['state_next'].clone()
   alive &= ~(term|tout)
  if t in (1,4,8,16,32):
   with torch.no_grad():
    pred_d=(tcp_position_world(pred,fk,tcp_ids)-target).norm(dim=-1)
    real_d=(tcp_position_world(actual,fk,tcp_ids)-target).norm(dim=-1)
    snapshots[t]={'pred':pred.clone(),'actual':actual.clone(),'pred_d':pred_d,'real_d':real_d,'alive':alive.clone()}
    rows.append({'horizon':t,'time_s':t*.02,'nse_mean':float(wm.spec.normalized_sq_error(pred,actual).mean()),
                 'base_xy_error_m':float((pred[:,:2]-actual[:,:2]).norm(dim=-1).mean()),
                 'base_height_error_m':float((pred[:,2]-actual[:,2]).abs().mean()),
                 'base_vx_rmse':float((pred[:,29]-actual[:,29]).square().mean().sqrt()),
                 'tcp_error_m':float((tcp_position_world(pred,fk,tcp_ids)-tcp_position_world(actual,fk,tcp_ids)).norm(dim=-1).mean()),
                 'branches_still_alive':int(alive.sum())})
 alpha=torch.zeros(6,device=env.device,requires_grad=True)
 s=state0[::2].clone(); fixed_a=(base[::2]+alpha[:,None]*directions).clamp(-1,1)
 responses={name:[] for name in names}
 for t in range(1,33):
  s=wm.step(s,fixed_a,xi[::2])
  if t in snapshots:
   grad_vx,=torch.autograd.grad(s[:,29].sum(),alpha,retain_graph=True)
   distance=(tcp_position_world(s,fk,tcp_ids)-target).norm(dim=-1)
   grad_d,=torch.autograd.grad(distance.sum(),alpha,retain_graph=True)
   snap=snapshots[t]
   for i,name in enumerate(names):
    pos,neg=2*i,2*i+1
    responses[name].append({'horizon':t,'real_branches_survived':bool(snap['alive'][pos]&snap['alive'][neg]),
       'real_dvx_dalpha':float((snap['actual'][pos,29]-snap['actual'][neg,29])/(2*eps)),
       'wm_fd_dvx_dalpha':float((snap['pred'][pos,29]-snap['pred'][neg,29])/(2*eps)),
       'wm_autograd_dvx_dalpha':float(grad_vx[i]),
       'real_ddistance_dalpha':float((snap['real_d'][pos]-snap['real_d'][neg])/(2*eps)),
       'wm_fd_ddistance_dalpha':float((snap['pred_d'][pos]-snap['pred_d'][neg])/(2*eps)),
       'wm_autograd_ddistance_dalpha':float(grad_d[i]),
       'real_vx_plus_minus':[float(snap['actual'][pos,29]),float(snap['actual'][neg,29])],
       'wm_vx_plus_minus':[float(snap['pred'][pos,29]),float(snap['pred'][neg,29])]})
 result={'checkpoint':str(checkpoint),'num_envs':12,'eps_normalized_action':eps,
         'method':'six +/- constant-action perturbations around final SM Actor at nominal reset pose; replay true executed controls through frozen WM; no updates',
         'limitations':['One nominal initial pose and one perturbation amplitude, not a global Jacobian calibration','Contact dynamics can make central finite differences nonsmooth','L actor action baseline; distance derivative uses fixed diagnostic point [1.5,0,.8]'],
         'base_action':base[0].tolist(),'metrics':rows,'responses':responses}
 (out/('wm_response_probe.json' if eps==.03 else f'wm_response_probe_eps{eps:.2f}.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'metrics':rows,'wheel_common':responses['wheel_common'],'arm_joint2':responses['arm_joint2']},indent=2),flush=True)
finally:
 env.close()
