"""CPU-only exact policy export and historical clocks verification."""
from pathlib import Path
import json, hashlib, shutil, sys
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];D=OUT/'data';sys.path.insert(0,str(ROOT/'src'))
from supermimic.policy.nets import Policy,PolicyCfg

def load(n):return json.loads((D/(n+'.json')).read_text())
def ck(p):return torch.load(ROOT/p,map_location='cpu',weights_only=False,mmap=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1);torch.manual_seed(123)
result={'method':'CPU exact state tensors and random-input actions; no simulator, optimization or new training','exports':{},'old_clocks':{},'config_differences':{},'historical_source_hash_matches':{}}
for key,inf,raw in [
 ('go2_sm','/home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt','logs/go2_comparison/supermimic_2p5h_seed2026/ckpt_latest.pt'),
 ('go2_ppo','/home/yifeiy/rl_sar/policy/go2/ppo_velocity/policy_inference.pt','logs/go2_comparison/ppo_2p5h_seed2026/model_758.pt'),
 ('m1_ppo','logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/policy_inference.pt','logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/model_533.pt')]:
 i=ck(inf);r=ck(raw);cfg=PolicyCfg(**i['config']['policy']);p=Policy(cfg);p.load_state_dict(i['policy']);p.eval()
 if key.endswith('_ppo'):
  a=r['actor_state_dict'];split=cfg.history*(3*cfg.num_joints+6)
  sd={'decoder.scale':a['mlp.0.scale'][:split], 'decoder.goal_scale':a['mlp.0.scale'][split:]}
  sd.update({'decoder.mlp.net.'+k.removeprefix('mlp.1.'):v for k,v in a.items() if k.startswith('mlp.1.')})
 else:sd=r['policy']
 q=Policy(cfg);q.load_state_dict(sd);q.eval();obs=torch.randn(64,cfg.history*(3*cfg.num_joints+6));goal=torch.randn(64,cfg.goal_dim)
 with torch.no_grad():diff=(p(obs,goal)-q(obs,goal)).abs().max().item()
 result['exports'][key]={'inference':str(inf),'source':raw,'inference_sha256':sha(ROOT/inf),'source_sha256':sha(ROOT/raw),'policy_tensors_exact_equal':all(torch.equal(sd[k],i['policy'][k]) for k in sd),'random_input_max_action_error':diff}
 if key=='go2_sm':result['go2_sm_final_stage']=r['command_state'].get('stage')
 if key=='go2_ppo':result['go2_ppo_final_stage']=r.get('infos',{}).get('velocity_command',{}).get('stage');result['go2_ppo_infos_keys']=list(r.get('infos',{}))
 del r,i,p,q,sd
for key,folder in [('wm_source','logs/go2_velocity_phase1_3072_b1024_20261027'),('base_actor','logs/phase3_training/go2_velocity_deployable_v2_20261102'),('gait','logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103')]:
 tb=ROOT/folder/'tb';ea=EventAccumulator(str(tb),size_guidance={'scalars':0});ea.Reload();events=ea.Scalars('time/hours');iters=[1800] if key=='wm_source' else [17000] if key=='base_actor' else [22000,24000,34000,34600]
 result['old_clocks'][key]={'path':str(tb.relative_to(ROOT)),'values':[{'step':e.step,'hours':e.value,'wall_time':e.wall_time} for e in events if e.step in iters]}
for robot in ['go2','m1']:
 p=load(robot+'_ppo_config')['env_args'];s=load(robot+'_supermimic_config')['env'];result['config_differences'][robot]={k:{'PPO':p[k],'SM':s[k]} for k in sorted(p.keys()&s.keys()) if p[k]!=s[k]}
old=load('provenance')['source_sha256']
for p,h in old.items():
 path=ROOT/p
 result['historical_source_hash_matches'][p]={'exists':path.exists(),'matches':path.exists() and sha(path)==h}
# Additional source coverage referenced in analysis.
for rel in ['src/supermimic/envs/mdp/observations.py','src/supermimic/motion_prior/velocity.py','src/supermimic/tasks/action_contract.py','scripts/play.py','scripts/evaluate_go2_velocity.py','scripts/evaluate_m1_reward.py']:
 p=ROOT/rel;dest=D/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
# Installed PPO implementation is evidence; never modify packages.
import rsl_rl
base=Path(rsl_rl.__file__).parent
for rel in ['algorithms/ppo.py','modules/distribution.py','runners/on_policy_runner.py','storage/rollout_storage.py']:
 p=base/rel
 if p.exists():
  dest=D/'upstream_runtime'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
(D/'evidence_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k not in ['historical_source_hash_matches']},ensure_ascii=False,indent=2))
