"""Persist a bounded PPO run and identical physical evaluations across UI sessions."""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=7200)
    args=p.parse_args()
    run=args.run.resolve(); source=run/'source_snapshot'
    environment=os.environ | {'PYTHONPATH':str(source/'src'),'OMP_NUM_THREADS':'1',
                              'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    record={'pid':os.getpid(),'started_at':dt.datetime.now().astimezone().isoformat(),
            'training_budget_seconds':args.seconds}
    def status(state,**values):
        record.update(state=state,**values)
        tmp=run/'supervisor_status.json.tmp';tmp.write_text(json.dumps(record,indent=2))
        tmp.replace(run/'supervisor_status.json')
    def execute(stage,command):
        with (run/f'{stage}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,*map(str,command)],cwd=source,env=environment,
                                  stdout=log,stderr=subprocess.STDOUT)
            status(stage,child_pid=proc.pid,command=list(map(str,command)))
            result=proc.wait()
        if result:
            raise RuntimeError(f'{stage}: exit {result}; see {run / (stage+".log")}')
    try:
        execute('training',[source/'tools/train_go2w_ppo.py','--output',run/'training',
                            '--actor-from',run/'bootstrap/actor.pt','--seconds',args.seconds])
        record['training_finished_at']=dt.datetime.now().astimezone().isoformat()
        for name,extra in [('final',[]),('final_extended',['--extended-omni'])]:
            execute(name+'_evaluation',[source/'tools/evaluate_go2w_experiment.py',
                '--checkpoint',run/'training/last.pt','--profile','omni_v3',
                '--output',run/f'evaluation/{name}.json',*extra])
        provenance=json.loads((run/'provenance.json').read_text())
        wm=Path(provenance['shared_WM'])
        provenance['WM_sha256_after']=hashlib.sha256(wm.read_bytes()).hexdigest()
        provenance['WM_unchanged']=provenance['WM_sha256_before']==provenance['WM_sha256_after']
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
        status('complete',finished_at=dt.datetime.now().astimezone().isoformat(),child_pid=None)
    except Exception as error:
        status('failed',error=repr(error))
        raise

if __name__=='__main__':
    main()
