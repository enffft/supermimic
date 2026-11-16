"""Export lateral-motion diagnostic figures and portable evidence."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--research',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];r=args.research;o=args.output;o.mkdir(exist_ok=True,parents=True)
    for name in ['physics/summary.json','go2_transfer/summary.json','config_analysis.json','model_audit.json','target_spectrum.json','github_sources.json','ppo_first_update_kl.json','ppo_first_update_kl_clip.json','provenance.json']:
        dst=o/name;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(r/name,dst)
    for name in ['diagnose_go2w_lateral.py','analyze_go2w_lateral_config.py','plot_go2w_lateral_diagnosis.py','diagnose_go2w_ppo_kl.py']:
        dst=o/'protocol'/name;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(root/'tools'/name,dst)
    s=json.loads((r/'physics/summary.json').read_text());cases={(x['policy'],x['name']):x for x in s['cases']}
    a=json.loads((r/'config_analysis.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    policies=['initial','PPO','SuperMimic'];colors=['#6c8ebf','#2b8c59','#dd8737']
    x=np.arange(2)
    for i,(policy,color) in enumerate(zip(policies,colors)):
        values=[cases[policy,name]['mean_actual'][1] for name in ['left_015','right_015']]
        axs[0].bar(x+(i-1)*.23,values,.23,label=policy,color=color)
        lift=[cases[policy,name]['clearance_threshold_analysis']['0.005']['any_foot_fraction']*100 for name in ['left_015','right_015']]
        axs[1].bar(x+(i-1)*.23,lift,.23,label=policy,color=color)
    axs[0].scatter(x,[.15,-.15],marker='_',color='black',s=300,label='Command')
    axs[0].set(xticks=x,xticklabels=['Left +0.15','Right -0.15'],ylabel='Mean body vy (m/s)');axs[0].legend(fontsize=8)
    axs[1].set(xticks=x,xticklabels=['Left +0.15','Right -0.15'],ylabel='Any wheel gap >5 mm (% valid frames)')
    noise_cases=['left_015','left_noise_0.02_hold1','left_noise_0.035_hold1','left_noise_0.1_hold1','left_noise_0.2_hold1']
    values=[cases['PPO',name]['falls'] for name in noise_cases]
    bars=axs[2].bar(np.arange(5),values,color='#bd4545')
    axs[2].bar_label(bars,labels=[f'{v}/12' for v in values]);axs[2].set(xticks=np.arange(5),xticklabels=['0','.02','.035','.10','.20'],ylabel='Falls /12 trials',xlabel='Independent Gaussian std (all 16 actions)',ylim=(0,14))
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Physical probes: a grounded policy; more white noise causes falls')
    fig.savefig(o/'physical_diagnosis.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    cmd=np.array([row['command_vy'] for row in a['reward_analysis']]);gain=[row['gain_matched_over_stationary'] for row in a['reward_analysis']]
    axs[0].plot(cmd,gain,'o-');axs[0].set(xlabel='Lateral command magnitude (m/s)',ylabel='Ideal tracking gain vs rest (reward/s)');axs[0].grid(alpha=.2)
    axs[0].set_title('Current reward prefers correct lateral motion')
    cfg=json.loads((r/'physics/summary.json').read_text())
    rates=cases['PPO','left_015']['mean_reward_rates']
    selected=['track_vx','track_vy','track_yaw','terrain_alignment','tracking_error_cost','leg_joint_deviation','leg_torque','leg_action_rate']
    vals=[rates[key] for key in selected]
    axs[1].barh(range(len(vals)),vals,color=['#4b8c68' if v>=0 else '#b54e44' for v in vals]);axs[1].set(yticks=range(len(vals)),yticklabels=selected,xlabel='Measured reward rate');axs[1].invert_yaxis();axs[1].grid(axis='x',alpha=.2)
    axs[1].set_title('PPO left command: strong return without side progress')
    fig.savefig(o/'reward_diagnosis.png',dpi=170);plt.close(fig)
    z=np.load(r/'physics/trajectories.npz');fig,axs=plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
    rep=s['replicas'];time=np.arange(500)*.02
    for i,policy in enumerate(['PPO','SuperMimic']):
        case_idx=next(k for k,row in enumerate(s['cases']) if row['policy']==policy and row['name']=='left_015')
        index=case_idx*rep
        axs[0,i].plot(time,z['target'][:,index,1],label='Thigh target',linewidth=.8)
        axs[0,i].plot(time,z['q'][:,index,1],label='Measured thigh',linewidth=1)
        axs[0,i].set(title=f'{policy}: FL thigh, left +0.15 m/s',xlabel='Time (s)',ylabel='Angle (rad)');axs[0,i].legend(fontsize=8)
        gap=z['wheel_gap'][:,case_idx*rep:(case_idx+1)*rep]
        axs[1,i].plot(np.arange(50,500,5)*.02,np.nanmax(gap,axis=(1,2))*1000)
        axs[1,i].axhline(5,color='black',linestyle='--',linewidth=.8)
        axs[1,i].set(xlabel='Time (s)',ylabel='Max gap over 12 trials, 4 wheels (mm)')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(o/'joint_and_clearance.png',dpi=170);plt.close(fig)
    kl=json.loads((r/'ppo_first_update_kl_clip.json').read_text())['records']
    fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    it=[v['minibatch'] for v in kl]
    axs[0].plot(it,[max(v['KL_mean'],1e-5) for v in kl]);axs[0].axhline(.02,color='red',linestyle='--',label='LR reduction threshold .02')
    axs[0].set(yscale='log',xlabel='Minibatch in first PPO update',ylabel='Mean KL(old || new)');axs[0].legend(fontsize=8)
    axs[1].plot(it,[100*v['ratio_outside_clip_fraction'] for v in kl]);axs[1].set(xlabel='Minibatch',ylabel='Ratios outside [0.8,1.2] (%)',ylim=(0,101))
    axs[2].plot(it,[v['learning_rate_after_schedule'] for v in kl]);axs[2].set(xlabel='Minibatch',ylabel='Learning rate',yscale='log')
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(o/'ppo_kl_diagnosis.png',dpi=170);plt.close(fig)
    hashes={str(p.relative_to(o)):hashlib.sha256(p.read_bytes()).hexdigest() for p in o.rglob('*') if p.is_file() and p.name!='sha256.json'}
    (o/'sha256.json').write_text(json.dumps(hashes,indent=2));print(o)

if __name__=='__main__':main()
