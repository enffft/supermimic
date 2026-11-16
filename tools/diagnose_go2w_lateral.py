"""Physical lateral-motion diagnostics, without training or changing saved assets.

Use PYTHONPATH pointing at the trained source snapshot. Wheel clearance is a
collision-mesh geometric proxy reconstructed with MuJoCo FK, not a force sensor.
"""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.velocity.go2.command_sampler import command_modes
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT, POLICY_SPEC, LEGS, LEG_JOINTS, leg_position_target
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env


class Go2LegTransfer(torch.nn.Module):
    """Zero-shot leg-target transfer only; wheel targets zero, no fitting."""
    def __init__(self, path, control):
        super().__init__()
        from unisupermimic.tasks.velocity.go2w.contract import DEFAULT_Q, soft_limits
        saved=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        self.pi=Policy(42,3,hist=5,num_joints=12,hidden=tuple(saved['args']['policy_hidden']),goal_mode='direct')
        self.pi.load_state_dict(saved['pi'])
        self.register_buffer('default',torch.tensor(DEFAULT_Q))
        self.register_buffer('limits',soft_limits(fraction=control['leg_limit_fraction']))

    def forward(self,history,goal):
        prev=history[...,34:46]
        spans=torch.where(prev>=0,self.limits[:,1]-self.default,self.default-self.limits[:,0])
        old_prev=prev*spans/.25
        old_obs=torch.cat((history[...,:24],history[...,28:34],old_prev),-1)
        old_action=self.pi(old_obs,goal).clamp(-1,1)
        span=torch.where(old_action>=0,self.limits[:,1]-self.default,self.default-self.limits[:,0])
        return torch.cat((old_action*.25/span,torch.zeros_like(old_action[...,:4])),-1)


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    actors = {}
    contract = None
    for name,path in [('initial',args.initial),('PPO',args.ppo),('SuperMimic',args.supermimic)]:
        saved = torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        hidden=tuple(saved.get('args',{}).get('policy_hidden',[2048,1024,512,256]))
        actor=Policy(50,3,hist=5,num_joints=16,hidden=hidden,goal_mode='direct',direct_spec=POLICY_SPEC).cuda().eval()
        actor.load_state_dict(saved['pi']);actors[name]=actor
        if contract is not None and saved['task_contract'] != contract:
            raise ValueError('Checkpoint task mismatch')
        contract=saved['task_contract']
    if args.go2_from:
        actors['Go2_zero_shot']=Go2LegTransfer(args.go2_from,contract['actions']['mixed']['control']).cuda().eval()
    if args.transfer_only:
        actors={key:value for key,value in actors.items() if key=='Go2_zero_shot'}
    cases=[]
    for policy in actors:
        for name,goal in [('stop',(0.,0.,0.)),('forward_030',(.3,0.,0.)),
                          ('left_015',(0.,.15,0.)),('right_015',(0.,-.15,0.)),
                          ('left_025',(0.,.25,0.)),('right_025',(0.,-.25,0.)),
                          ('yaw_left_060',(0.,0.,.6)),('yaw_right_060',(0.,0.,-.6))]:
            cases.append(dict(name=name,policy=policy,goal=goal,std=0.,hold=1))
    for policy in ([] if args.transfer_only else ['initial','PPO']):
        for std,hold in [(.02,1),(.035,1),(.10,1),(.20,1),(.10,5),(.10,10)]:
            cases.append(dict(name=f'left_noise_{std}_hold{hold}',policy=policy,goal=(0.,.15,0.),std=std,hold=hold))
    replicas=args.replicas
    n=len(cases)*replicas
    env=make_go2w_env(load_config('omni_v3'),n)
    try:
        assert env.supermimic_contract==contract
        env.reset();c=env.command_manager.get_term('velocity');cur=c.curriculum
        c.command_domain.zero_();cur.enabled=False;cur.phase='fixed';cur.fixed_domain=c.command_domain
        cur.terrain_type.copy_(torch.arange(n)%5);cur.terrain_level.zero_();cur.mastery.zero_();cur.steps.zero_();cur.episode_fall.zero_()
        env.reset();env.set_collection_exploration(0.)
        offsets=(torch.rand(replicas,3,generator=torch.Generator().manual_seed(2026))-.5)*torch.tensor([.04,.04,.05])
        offsets=offsets.repeat(len(cases),1)
        root=torch.zeros(n,13);root[:,:3]=c.state()[0];root[:,:2]+=offsets[:,:2]
        root[:,3]=torch.cos(offsets[:,2]/2);root[:,6]=torch.sin(offsets[:,2]/2)
        reset_mask=torch.ones(n,dtype=torch.bool)
        with env._reset_state.scoped_device_event_tensor(reset_mask):
            c.robot.write_motion_state_tensor_to_sim(root_state=root,position=torch.cat((c.mdp.default_q.expand(n,12),torch.zeros(n,4)),-1),velocity=torch.zeros(n,16),reset_mask=reset_mask)
        fixed=torch.tensor([case['goal'] for case in cases]).repeat_interleave(replicas,0)
        groups={p:torch.tensor([i for i,case in enumerate(cases) if case['policy']==p]).repeat_interleave(replicas)*replicas+torch.arange(replicas).repeat(sum(case['policy']==p for case in cases)) for p in actors}
        model=env.get_playback_model(0);data=mujoco.MjData(model)
        wheel_geoms=[model.geom(f'robot/{leg}_wheel_collision').id for leg in LEGS]
        mesh_vertices=[]
        for gid in wheel_geoms:
            mesh=model.geom_dataid[gid];a=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
            mesh_vertices.append(np.array(model.mesh_vert[a:a+count]))
        history=torch.zeros(n,5,50);live=torch.ones(n,dtype=torch.bool);fall=torch.zeros_like(live)
        start=c.state()[0].clone();position=start.clone()
        trace=[];gaps=[];noise=torch.zeros(n,16)
        for step in range(500):
            c.g.copy_(fixed);c.command_mode.copy_(command_modes(fixed));c.crossing_episode.zero_()
            info=env.current_info();state=unflatten_state(info['s'],LAYOUT)
            frame=c.mdp.obs(state,info['a_prev'],info['xi'],info['f'])
            history=torch.cat((history[:,1:],frame[:,None]),1)
            history=torch.where((info['ep_step']==0)[:,None,None],frame[:,None].expand_as(history),history)
            action=torch.zeros(n,16)
            for policy,ids in groups.items():
                action[ids]=actors[policy](history[ids].cuda(),fixed[ids].cuda()).cpu()
            for i,case in enumerate(cases):
                part=slice(i*replicas,(i+1)*replicas)
                if case['std'] and step%case['hold']==0:
                    noise[part]=torch.randn(replicas,16)*case['std']
            mean_action=action.clone();action=(action+noise).clamp(-1,1)
            before=live.clone();row=env.step(action).info[SUPERMIMIC_TRANSITION_KEY]
            assert torch.equal(row['g'],fixed)
            previous=unflatten_state(row['s'],LAYOUT);state=unflatten_state(row['s1'],LAYOUT)
            velocity=torch.cat((matvec(state[1].transpose(-1,-2),state[2])[:,:2],matvec(state[1].transpose(-1,-2),state[3])[:,2:3]),-1)
            normal=torch.zeros(n,3);normal[:,2]=1
            terms=c.mdp.terms(previous,state,fixed,row['a'],row['a_prev'],measured_torque=env.velocity_measured_torque,terrain_normal=normal,command_domain=torch.zeros(n,dtype=torch.long))
            reconstructed=.02*torch.stack(list(terms.values())).sum(0)
            if not torch.allclose(reconstructed,row['r'],atol=2e-5,rtol=1e-4):
                raise ValueError(f'Reward reconstruction mismatch {float((reconstructed-row["r"]).abs().max())}')
            target=leg_position_target(row['a'],c.mdp.control,c.mdp.default_q,c.mdp.limits)
            position=torch.where(live[:,None],state[0],position)
            trace.append({'live':before.numpy(),'velocity':velocity.numpy(),'q':state[4].numpy(),
                          'qd':state[5].numpy(),'target':target.numpy(),'action':row['a'].numpy(),
                          'mean_action':mean_action.numpy(),'height':state[0][:,2].numpy(),
                          'terms':{key:value.numpy() for key,value in terms.items()}})
            fall |= live&row['term'];live &= ~row['done']
            if step>=50 and step%5==0:
                qpos=env.backend.get_state_views(('qpos',),device='cpu')['qpos'].numpy()
                gap=np.full((n,4),np.nan)
                for j in np.flatnonzero(live.numpy()):
                    data.qpos[:]=qpos[j];mujoco.mj_kinematics(model,data)
                    for k,gid in enumerate(wheel_geoms):
                        low=np.min(mesh_vertices[k]@data.geom_xmat[gid].reshape(3,3)[2])+data.geom_xpos[gid,2]
                        # All selected origins are the same flat level 0 height.
                        low-=float(c.mdp.terrain.ground_height(state[0][j:j+1,:2])[0])
                        gap[j,k]=low
                gaps.append(gap)
            if step%100==99:print(f'Physical steps {step+1}/500',flush=True)
        live_all=np.stack([x['live'] for x in trace]);velocity=np.stack([x['velocity'] for x in trace]);q=np.stack([x['q'] for x in trace]);target=np.stack([x['target'] for x in trace]);qd=np.stack([x['qd'] for x in trace]);gaps=np.stack(gaps)
        result=[]
        for i,case in enumerate(cases):
            part=slice(i*replicas,(i+1)*replicas);valid=live_all[50:,part]
            def mean(values):
                return np.asarray(values[50:,part])[valid].mean(axis=0).tolist() if valid.any() else None
            q_rms=np.full((replicas,12),np.nan);target_rms=q_rms.copy();cycle_lateral=[]
            for j in range(replicas):
                mask=valid[:,j]
                if mask.any():
                    q_rms[j]=q[50:,i*replicas+j][mask].std(0)
                    target_rms[j]=target[50:,i*replicas+j][mask].std(0)
                    cycle_lateral.append(float(np.mean(velocity[50:,i*replicas+j,1][mask])))
            gap=gaps[:,part];gap_valid=np.isfinite(gap);per_foot_lift=np.nansum(gap>.02,axis=(0,1))/np.maximum(1,gap_valid.sum(axis=(0,1)))
            clearance_threshold_analysis={}
            for threshold in [.005,.01,.02]:
                clearance_threshold_analysis[str(threshold)]={
                    'per_foot_fraction':((gap>threshold).sum(axis=(0,1))/np.maximum(1,gap_valid.sum(axis=(0,1)))).tolist(),
                    'any_foot_fraction':float(np.any(gap>threshold,axis=-1).sum()/max(1,np.any(gap_valid,axis=-1).sum()))}
            terms={key:mean(np.stack([x['terms'][key] for x in trace])) for key in trace[0]['terms']}
            result.append(case|{'replicas':replicas,'falls':int(fall[part].sum()),'survived_10s':int(live[part].sum()),
                'valid_samples_after_1s':int(valid.sum()),'mean_actual':mean(velocity),
                'per_trial_mean_vy':cycle_lateral,'displacement_world_xyz':(position[part]-start[part]).tolist(),
                'mean_absolute_velocity':mean(np.abs(velocity)),
                'q_temporal_std_rad':np.nanmean(q_rms,axis=0).tolist(),
                'target_temporal_std_rad':np.nanmean(target_rms,axis=0).tolist(),
                'mean_abs_joint_velocity':mean(np.abs(qd)),
                'mean_reward_rates':terms,'total_mean_reward_rate':sum(terms.values()) if all(value is not None for value in terms.values()) else None,
                'wheel_gap_over_2cm_fraction_FL_FR_RL_RR':per_foot_lift.tolist(),
                'clearance_threshold_analysis':clearance_threshold_analysis,
                'any_wheel_gap_over_2cm_fraction':float(np.any(gap>.02,axis=-1).sum()/max(1,np.any(gap_valid,axis=-1).sum())),
                'max_wheel_gap_p95_m':float(np.nanquantile(gap,.95)),
                'mean_abs_mean_action':mean(np.abs(np.stack([x['mean_action'] for x in trace]))),
            })
        def finite_json(value):
            if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
            if isinstance(value,list):return [finite_json(v) for v in value]
            if isinstance(value,float) and not np.isfinite(value):return None
            return value
        result=finite_json(result)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'summary.json').write_text(json.dumps({'cases':result,'replicas':replicas,
            'protocol':'500 control steps, fixed flat L0, identical xy/yaw reset offsets, one seed; no learning',
            'wheel_clearance':'minimum compiled collision-mesh vertex z minus local ground, FK, sampled 10Hz after 1s; not force-based contacts',
            'noise':'independent Gaussian in normalized action coordinates, all 16 joints, held for specified control steps; inference-only probes',
            'source':str(Path(__import__('unisupermimic').__file__).resolve()),
            'checkpoints':{k:str(v) for k,v in [('initial',args.initial),('PPO',args.ppo),('SuperMimic',args.supermimic)]}},indent=2))
        np.savez_compressed(args.output/'trajectories.npz',velocity=velocity,q=q,target=target,qd=qd,live=live_all,wheel_gap=gaps)
        with (args.output/'time_series.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['time','policy','case','samples','vx','vy','yaw','hip_q_std','thigh_q_std','calf_q_std'])
            for step in range(0,500,5):
                for i,case in enumerate(cases):
                    part=slice(i*replicas,(i+1)*replicas);ok=live_all[step,part]
                    values=velocity[step,part][ok].mean(0) if ok.any() else [None]*3
                    s=q[step,part][ok].reshape(-1,4,3).std(axis=(0,1)) if ok.any() else [None]*3
                    writer.writerow([step*.02,case['policy'],case['name'],int(ok.sum()),*values,*s])
        print(args.output/'summary.json')
    finally:
        env.close();env.backend.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',type=Path,required=True);parser.add_argument('--ppo',type=Path,required=True)
    parser.add_argument('--supermimic',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicas',type=int,default=12)
    parser.add_argument('--go2-from',type=Path);parser.add_argument('--transfer-only',action='store_true')
    run(parser.parse_args())
