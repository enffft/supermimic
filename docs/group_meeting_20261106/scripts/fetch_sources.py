"""Fetch bounded public GitHub evidence, no clone or dependency installation."""
import concurrent.futures, json, subprocess, hashlib, re
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'external_sources'
REPOS=['Huy-Quang-Dao/instantTrack','fuyichen0725/blinkmimic','CuboiLeo/SuperMimic-Reproduced','zita-ch/supermimic_vibed','emoPointer/supermimic_manipulation','locomanip-duet/RoboDuet']
EXTRA={
 'emoPointer/supermimic_manipulation':['docs/supermimic.md'],
 'locomanip-duet/RoboDuet':['scripts/auto_train.py'],
}
def get(url):
 return subprocess.check_output(['curl','-fLsS','--max-time','15','--connect-timeout','5','-A','research-audit-readonly',url],stderr=subprocess.DEVNULL)
def work(repo):
 d=OUT/repo.replace('/','__');d.mkdir(parents=True,exist_ok=True)
 result={'repo':repo,'retrieved_date':'2026-11-06','files':[],'errors':[]}
 try:
  meta=json.loads(get('https://api.github.com/repos/'+repo));ref=meta['default_branch']
  commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{ref}'))['sha'];result['sha']=commit
  tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))
  (d/'tree.json').write_text(json.dumps(tree,indent=2))
  paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob']
  chosen=[p for p in paths if re.search(r'(^|/)README[^/]*$',p,re.I)][:3]
  # Prioritize concrete mechanisms and training implementations, maximum 16 source files.
  py=[p for p in paths if p.endswith('.py') and not '/assets/' in p]
  ranked=sorted(py,key=lambda p:sum(s in p.lower() for s in ['train','world_model','rollout','reset','curriculum','replay','policy','actor','dynamics']),reverse=True)
  chosen+=ranked[:16]
  chosen=list(dict.fromkeys(chosen+[p for p in EXTRA.get(repo,[]) if p in paths]))
  for p in chosen:
   try:
    raw=get(f'https://raw.githubusercontent.com/{repo}/{commit}/{p}')
    dest=d/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    result['files'].append({'path':p,'url':f'https://github.com/{repo}/blob/{commit}/{p}','sha256':hashlib.sha256(raw).hexdigest()})
   except Exception as e:result['errors'].append(p+': '+str(e))
 except Exception as e:result['errors'].append(str(e))
 (d/'provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 results=list(ex.map(work,REPOS))
(OUT/'index.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
for r in results:print(r['repo'],r.get('sha'),len(r['files']),r['errors'][:2])
