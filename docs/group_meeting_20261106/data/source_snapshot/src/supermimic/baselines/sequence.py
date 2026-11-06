"""Restartable subprocess stages; algorithm implementations stay upstream."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time


@dataclass
class Stage:
    name: str
    command: list[str]
    log: Path
    # Native checkpoint to resume if a training process was interrupted.
    checkpoint: Path | None = None
    completion: Path | None = None
    resume_extra: tuple[str, ...] = ()
    # PPO's native checkpoint filename is written to progress.json.
    progress: Path | None = None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def stage_checkpoint(stage):
    if stage.checkpoint is not None and stage.checkpoint.is_file():
        return stage.checkpoint
    candidates = list(stage.log.parent.glob('model_[0-9]*.pt')) if stage.progress else []
    if candidates:
        return max(candidates, key=lambda p: int(p.stem.split('_')[-1]))
    if stage.progress is not None and stage.progress.is_file():
        checkpoint = Path(json.loads(stage.progress.read_text())['checkpoint'])
        if checkpoint.is_file():
            return checkpoint
    return None


def run_stages(stages, state_path, *, cwd, env, resume=False, runner=None):
    from supermimic.baselines.resources import run_guarded
    runner = runner or run_guarded
    plan = [{'name': s.name, 'command': s.command} for s in stages]
    state = {'plan': plan, 'stages': {}}
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if not resume:
            raise FileExistsError(f'Queue exists: {state_path}; use --resume or a new --tag')
        if saved['plan'] != plan:
            raise ValueError('Queue plan changed: retain the original budgets/envs/options when resuming')
        state = saved
    elif resume:
        raise FileNotFoundError(f"No queue manifest: {state_path}; omit --resume for a fresh tag")
    else:
        for stage in stages:
            if stage.completion and (stage.completion.exists() or stage_checkpoint(stage)):
                raise FileExistsError(f'Existing training artifacts: {stage.log.parent}; use a new --tag (no queue manifest to verify this run)')

    state["completed"] = False

    def status(stage, value, **extra):
        state['stages'][stage.name] = {'status': value, 'updated_unix': time.time(), **extra}
        state['launcher_pid'] = os.getpid()
        write_json(state_path, state)

    for stage in stages:
        if state['stages'].get(stage.name, {}).get('status') == 'completed':
            print(f'[queue] skip completed {stage.name}', flush=True)
            continue
        if resume and stage.completion and stage.completion.exists():
            status(stage, 'completed', recovered_completion=str(stage.completion))
            continue
        command = stage.command.copy()
        checkpoint = stage_checkpoint(stage) if resume else None
        if checkpoint:
            command += ['--resume', str(checkpoint), *stage.resume_extra]
        stage.log.parent.mkdir(parents=True, exist_ok=True)
        status(stage, 'running', argv=command, resumed_from=str(checkpoint) if checkpoint else None)
        try:
            with stage.log.open('a') as log:
                log.write(f'\n[queue] {time.ctime()} {stage.name}\n')
                log.flush()
                code = runner(command, cwd=cwd, env=env, stdout=log)
            if code:
                raise RuntimeError(f'{stage.name} exited {code}; see {stage.log}')
            if stage.completion and not stage.completion.exists():
                raise RuntimeError(f'{stage.name} returned without completion metadata')
            status(stage, 'completed')
            print(f'[queue] completed {stage.name}', flush=True)
        except BaseException as exc:
            status(stage, 'failed', error=f'{type(exc).__name__}: {exc}')
            raise
    state['completed'] = True
    write_json(state_path, state)


def launch_environment(root):
    return {**os.environ, 'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MAX_JOBS': '1',
            'TORCHINDUCTOR_COMPILE_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': '0'}


def detach(script, argv, out, *, root, env):
    import subprocess
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'launcher.log').open('a') as log:
        process = subprocess.Popen([str(Path.home() / 'miniconda3/envs/supermimic/bin/python'), '-u',
                                    str(script), *[a for a in argv if a != '--detach']],
                                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(out / 'launcher.json', {'pid': process.pid, 'log': str(out / 'launcher.log')})
    print(f'Background launcher PID {process.pid}; log: {out / "launcher.log"}', flush=True)
