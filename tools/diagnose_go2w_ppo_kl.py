"""Inspect one native PPO update in memory; never writes trained weights or WM."""
import argparse
import json
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.ppo import Go2WPpoEnv, training_config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2026)
    env=make_go2w_env(load_config('omni_v3'),4096)
    try:
        env.reset();bridge=Go2WPpoEnv(env);cfg=training_config()
        runner=OnPolicyRunner(bridge,cfg,device='cuda:0')
        saved=torch.load(args.actor,weights_only=True,map_location='cpu')
        assert saved['task_contract']==env.supermimic_contract
        runner.alg.actor.mlp.policy.load_state_dict(saved['pi'])
        records=[];native=runner.alg.actor.get_kl_divergence
        active={}
        native_generator=runner.alg.storage.mini_batch_generator
        def batches(*a,**kw):
            for batch in native_generator(*a,**kw):
                active['batch']=batch
                yield batch
        runner.alg.storage.mini_batch_generator=batches
        def get_kl(old,new):
            result=native(old,new)
            batch=active['batch']
            log_ratio=runner.alg.actor.get_output_log_prob(batch.actions)-batch.old_actions_log_prob.squeeze(-1)
            ratio=log_ratio.exp()
            records.append({'ratio_outside_clip_fraction':float(((ratio<.8)|(ratio>1.2)).float().mean()),
                            'action_mean_delta_RMS':float((new[0]-old[0]).square().mean().sqrt()),
                            'mean_log_ratio':float(log_ratio.mean()),'minibatch':len(records)+1,'KL_mean':float(result.mean().detach()),
                            'learning_rate_before_schedule':runner.alg.learning_rate})
            return result
        runner.alg.actor.get_kl_divergence=get_kl
        native_step=runner.alg.optimizer.step
        def step(*a,**kw):
            records[-1]['learning_rate_after_schedule']=runner.alg.optimizer.param_groups[0]['lr']
            return native_step(*a,**kw)
        runner.alg.optimizer.step=step
        result={}
        def log(**kw):
            result.update({key:kw[key] for key in ['collect_time','learn_time','loss_dict','learning_rate']})
        runner.logger.log=log
        runner.learn(1)
        result.update(records=records,initial_std=.02,initial_learning_rate=2e-4,
                      transitions=4096*24,native_optimizer_steps=len(records),
                      note='One diagnostic native PPO update, in-memory Actor/Critic only; no saved weights or WM mutated')
        args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2));print(args.output)
    finally:env.close();env.backend.close()

if __name__=='__main__':main()
