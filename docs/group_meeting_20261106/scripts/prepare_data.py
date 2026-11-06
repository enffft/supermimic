"""Read-only local evidence extraction; run from any cwd with the Conda interpreter."""
from pathlib import Path
import json, hashlib, subprocess, shutil, gc, sys, importlib.metadata
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
D=OUT/'data'; D.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
def read(p): return json.loads((ROOT/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
manifest={'date':'2026-11-06','root':str(ROOT),'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'status_at_extraction':git('status','--short'),'scope':'Only output directory changes; no training, recording, commit or push. User records videos; provide play commands.','versions':{},'sources':[],'checkpoints':{}}
for pkg in ['torch','mjlab','rsl-rl-lib','mujoco','matplotlib','numpy','pandas']:
 try: manifest['versions'][pkg]=importlib.metadata.version(pkg)
 except importlib.metadata.PackageNotFoundError: pass
sources={}
def copy_source(name,p):
 p=Path(p); p=p if p.is_absolute() else ROOT/p
 dest=D/(name+'.json');shutil.copyfile(p,dest)
 manifest['sources'].append({'id':name,'path':str(p),'snapshot':str(dest.relative_to(OUT)),'sha256':sha(p)})
 return json.loads(p.read_text())
for alg in ['ppo','supermimic']:
 rows=[]
 for suffix in ['evaluation','extra']:
  obj=copy_source(f'go2_{alg}_{suffix}',f'logs/final_evaluation_20261104/{alg}_velocity_{suffix}.json'); rows+=obj['real_mujoco']
 sources[f'go2_{alg}']=rows
 for task in ['forward','manipulation','balanced','completion']:
  copy_source(f'm1_{alg}_{task}',f'docs/results/m1_v2_20261106/{alg}_{task}.json')
 for robot,base in [('go2','logs/go2_comparison'),('m1','logs/m1_ppo' if alg=='ppo' else 'logs/m1_reward')]:
  tag=f'{alg}_2p5h_seed2026' if robot=='go2' else 'v2_lm50_2048_3p5h_seed2026'
  p=f'{base}/{tag}/'+('comparison_config.json' if alg=='ppo' else 'config.json')
  copy_source(f'{robot}_{alg}_config',p)
  for kind in ['progress','completion']:
   path=ROOT/base/tag/(kind+'.json')
   if path.exists(): copy_source(f'{robot}_{alg}_training_{kind}',path)
for p in sorted((ROOT/'docs/results/training_audit_20261106').glob('*.json')):
 copy_source(p.stem,p)
for name in ['policy_posture_probe','wm_response_probe','wm_response_probe_eps0.01','reward_scalar_audit','provenance']:
 copy_source('m1_'+name,ROOT/'docs/results/m1_v2_20261106'/f'{name}.json')
save(D/'go2_results.json',sources)
# Actual checkpoint metadata and state-dict dimensions; memory-mapped, CPU only.
import torch
torch.set_num_threads(1)
ckpts={
 'go2_ppo':'logs/go2_comparison/ppo_2p5h_seed2026/model_758.pt',
 'go2_supermimic':'logs/go2_comparison/supermimic_2p5h_seed2026/ckpt_latest.pt',
 'm1_ppo':'logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/model_533.pt',
 'm1_supermimic':'logs/m1_reward/v2_lm50_2048_3p5h_seed2026/ckpt_latest.pt',
 'go2_old_22000':'logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0022000.pt',
 'go2_old_24000':'logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0024000.pt',
 'go2_old_34000':'logs/phase3_training/go2_velocity_diagonal_slip_3072_2p5h_20261103/ckpt_0034000.pt',
}
for key,path in ckpts.items():
 p=ROOT/path
 if not p.exists(): manifest['checkpoints'][key]={'path':path,'exists':False};continue
 obj=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 m={'path':path,'exists':True,'sha256':sha(p),'keys':list(obj),'metadata':{},'networks':{}}
 for k,v in obj.items():
  if isinstance(v,(str,int,float,bool)) or v is None: m['metadata'][k]=v
  if isinstance(v,dict) and any(torch.is_tensor(t) for t in v.values()):
   shapes={n:list(t.shape) for n,t in v.items() if torch.is_tensor(t)}
   params=sum(t.numel() for n,t in v.items() if torch.is_tensor(t) and (n.endswith('.weight') or n.endswith('.bias')))
   m['networks'][k]={'linear_parameter_count':params,'shapes':shapes}
 if 'config' in obj: save(D/f'{key}_checkpoint_config.json',obj['config'])
 manifest['checkpoints'][key]=m
 del obj; gc.collect()
 print('Checkpoint verified:',key,flush=True)
# Inference package provenance and exact Actor match, one algorithm at a time.
for key,path in {
 'go2_ppo_inference':'/home/yifeiy/rl_sar/policy/go2/ppo_velocity/policy_inference.pt',
 'go2_sm_inference':'/home/yifeiy/rl_sar/policy/go2/supermimic_velocity/policy_inference.pt',
 'm1_ppo_inference':str(ROOT/'logs/m1_ppo/v2_lm50_2048_3p5h_seed2026/policy_inference.pt')}.items():
 p=Path(path);obj=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 manifest['checkpoints'][key]={'path':path,'sha256':sha(p),'source_checkpoint':obj.get('source_checkpoint'),'policy_shapes':{k:list(v.shape) for k,v in obj['policy'].items()}}
 del obj;gc.collect()
for p in sorted((ROOT/'src/supermimic').rglob('*.py')):
 # Preserve exact audit-time source for relevant modules without changing them.
 if p.parent.name in ['tasks','policy','world_model','imagination','baselines'] or p.name in ['commands.py','multirobot_env_cfg.py','loop.py','spec.py','actions.py','state_io.py','terminations.py','rewards.py','m1_reward.py']:
  rel=p.relative_to(ROOT); dest=OUT/'data/source_snapshot'/rel
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  manifest['sources'].append({'id':'source','path':str(rel),'snapshot':str(dest.relative_to(OUT)),'sha256':sha(p)})
# Bounded scalar export; no simulator or GPU use.
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
curves={}
for key,base in [('go2_ppo','logs/go2_comparison/ppo_2p5h_seed2026'),('go2_sm','logs/go2_comparison/supermimic_2p5h_seed2026/tb'),('m1_ppo','logs/m1_ppo/v2_lm50_2048_3p5h_seed2026'),('m1_sm','logs/m1_reward/v2_lm50_2048_3p5h_seed2026/tb')]:
 ea=EventAccumulator(str(ROOT/base),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars']
 selected=[t for t in tags if any(s in t for s in ['nse','action_abs','grad_norm','actor_lr','critic_lr','mean_std','mean_noise_std','learning_rate','reward_real/height','real/M/ee_distance','real/L/base_vx','time/hours','actor_updates','Loss/','Train/mean_reward'])]
 curves[key]={'path':base,'available_tags':tags,'series':{t:[{'step':e.step,'wall_time':e.wall_time,'value':e.value} for e in ea.Scalars(t) if __import__('math').isfinite(e.value)] for t in selected}}
 for p in (ROOT/base).glob('events*'): manifest['sources'].append({'id':'tensorboard','path':str(p.relative_to(ROOT)),'sha256':sha(p)})
save(D/'training_curves.json',curves)
save(D/'experiment_manifest.json',manifest)
print('Saved manifest and original JSON snapshots; no GPU jobs started.')
