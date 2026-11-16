"""Reproducible reward, command, FK and PPO-scale checks; no optimizer updates."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.tasks.velocity.go2.command_sampler import TerrainAwareVelocitySampler, fixed_domains
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import MODEL, LEG_JOINTS, WHEEL_JOINTS, LEGS, DEFAULT_Q, leg_position_target
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ppo',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2026)
    cfg=load_config('omni_v3');control=cfg.actions['mixed'].control
    mdp=Go2WVelocityMdp(control=control,reward=cfg.commands['velocity'].reward)
    def state(vy):
        return (torch.tensor([[0.,0.,.42]]),torch.eye(3)[None],torch.stack((vy*0,vy,vy*0),-1),
                torch.zeros(1,3),mdp.default_q[None],torch.zeros(1,16))
    reward=[]
    for cmd in [.01,.03,.06,.10,.15,.25,.35]:
        values={}
        for name,velocity in [('stationary',0.),('matched',cmd),('wrong_direction',-cmd)]:
            vy=torch.tensor([velocity],requires_grad=True);s=state(vy)
            terms=mdp.terms(s,s,torch.tensor([[0.,cmd,0.]]),torch.zeros(1,16),torch.zeros(1,16))
            total=torch.stack(list(terms.values())).sum();total.backward()
            values[name]={'reward_rate':total.item(),'d_rate_d_vy':vy.grad.item(),
                          'terms':{key:float(value.detach()) for key,value in terms.items()}}
        reward.append({'command_vy':cmd,'gain_matched_over_stationary':values['matched']['reward_rate']-values['stationary']['reward_rate'],**values})
    # Pure lateral exposure in the actual sampler, not just YAML probabilities.
    sampler=TerrainAwareVelocitySampler(cfg.commands['velocity'].sampler,'cpu')
    domains=fixed_domains(4096,cfg.commands['velocity'].sampler.domain_weights,'cpu')
    mode_counts=torch.zeros(6,6,dtype=torch.long);lateral=[];flat_mixed=[]
    for _ in range(200):
        goals,modes,_,_=sampler.sample(domains,torch.zeros_like(domains))
        mode_counts.view(-1).scatter_add_(0,domains*6+modes,torch.ones_like(domains))
        lateral.append(goals[(domains==0)&(modes==3),1]);flat_mixed.append(goals[(domains==0)&(modes==5),1])
    lateral=torch.cat(lateral);flat_mixed=torch.cat(flat_mixed)
    command={'samples':4096*200,'domain_mode_counts':mode_counts.tolist(),
             'flat_pure_lateral_fraction_all':mode_counts[0,3].item()/(4096*200),
             'flat_pure_lateral_mean_abs':lateral.abs().mean().item(),
             'flat_pure_lateral_positive_fraction':(lateral>0).float().mean().item(),
             'flat_pure_lateral_fraction_abs_below_006':(lateral.abs()<.06).float().mean().item(),
             'flat_mixed_mean_abs_vy':flat_mixed.abs().mean().item()}
    model=mujoco.MjModel.from_xml_path(str(MODEL));data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)
    qids=np.array([model.joint(j).qposadr[0] for j in LEG_JOINTS]);data.qpos[qids]=DEFAULT_Q
    mujoco.mj_forward(model,data)
    geom=[]
    for leg,name in enumerate(LEGS):
        body=model.body(f'robot/{name}_wheel_link').id;wheel=model.joint(WHEEL_JOINTS[leg]).id
        axis=data.xaxis[wheel].copy();nominal=data.xpos[body].copy()
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(model,data,jp,jr,body)
        dofs=[model.joint(j).dofadr[0] for j in LEG_JOINTS[leg*3:leg*3+3]]
        geom.append({'leg':name,'wheel_axis_world_at_nominal':axis.tolist(),
                     'rolling_velocity_per_unit_radius_and_speed':np.cross(axis,[0,0,-1]).tolist(),
                     'wheel_center_jacobian_xyz_by_hip_thigh_calf':jp[:,dofs].tolist()})
    workspace=[]
    for half_width in [.04,.15,.30]:
        for leg,name in enumerate(LEGS):
            body=model.body(f'robot/{name}_wheel_link').id
            data.qpos[qids]=DEFAULT_Q;mujoco.mj_kinematics(model,data);nominal=data.xpos[body].copy()
            samples=[]
            for delta in np.array(np.meshgrid(*[np.linspace(-half_width,half_width,9)]*3)).reshape(3,-1).T:
                data.qpos[qids]=DEFAULT_Q
                part=slice(leg*3,leg*3+3)
                data.qpos[qids[part]]=np.clip(np.array(DEFAULT_Q)[part]+delta,mdp.limits[part,0].numpy(),mdp.limits[part,1].numpy())
                mujoco.mj_kinematics(model,data);samples.append(data.xpos[body].copy()-nominal)
            samples=np.array(samples)
            workspace.append({'leg':name,'joint_box_half_width_rad':half_width,
                              'wheel_center_y_span_m':float(np.ptp(samples[:,1])),
                              'max_wheel_center_lift_m':float(samples[:,2].max())})
    scales={}
    for std in [.02,.035,.10]:
        g=torch.Generator().manual_seed(2026)
        a=torch.randn(100000,16,generator=g)*std
        q=leg_position_target(a,control,mdp.default_q,mdp.limits)
        scales[str(std)]={'physical_target_std_at_default_rad':q.std(0).tolist(),
                          'wheel_target_std_rad_s':15*std,
                          'note':'Local default-centered mapping; trained means alter asymmetric branch occupancy'}
    saved=torch.load(args.ppo,weights_only=True,map_location='cpu',mmap=True)
    log_std=saved['actor_state_dict']['distribution.log_std_param']
    scales['final_learned_std']=log_std.exp().tolist()
    scales['optimizer_learning_rate']=saved['optimizer_state_dict']['param_groups'][0]['lr']
    result={'reward_analysis':reward,'command_sampler':command,'wheel_geometry':geom,'workspace_FK':workspace,
            'exploration_scaling':scales,'gamma_099_effective_time_s':.02/(1-.99),
            'GAE_gamma_lambda_decay_time_s':-.02/np.log(.99*.95),
            'hist_frame_span_s':(5-1)*.02,'model_timestep':model.opt.timestep,
            'limitations':'Synthetic equal-pose reward comparison and fixed-base FK are not dynamically feasible locomotion proof; native PPO GAE and critic bootstrap extend beyond a single rollout.'}
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2));print(args.output)

if __name__=='__main__':main()
