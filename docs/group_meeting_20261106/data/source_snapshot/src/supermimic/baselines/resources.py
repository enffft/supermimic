"""Small resource watchdog for this task's subprocesses only (no unrelated kills)."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
import re

GIB = 1024 ** 3


def gpu_compute_jobs():
    """Compute users, excluding the known desktop video encoder on this host."""
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    rows = []
    for line in raw.splitlines():
        fields = line.split(",", 2)
        if len(fields) >= 2 and fields[1].strip() == "/opt/todesk/bin/ToDesk_Session":
            continue
        rows.append(line)
    return "\n".join(rows)


def active_ml_jobs():
    """Catch CPU setup / gaps between GPU evaluation children, too."""
    found = []
    pattern = re.compile(r"(?:^|/)(?:train[^/]*\.py|evaluate_[^/]+\.py|run_m1_[^/]*sweep\.py)$")
    for path in Path("/proc").glob("[0-9]*"):
        try:
            if int(path.name) in (os.getpid(), os.getppid()):
                continue
            argv = (path / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            if not argv or "python" not in Path(argv[0]).name:
                continue
            matched = [a for a in argv[1:] if pattern.search(a) or a.startswith((
                "supermimic.scripts.train", "supermimic.training.m1_reward",
                "supermimic.baselines.go2_ppo", "supermimic.baselines.go2_supermimic",
                "supermimic.baselines.m1_ppo"))]
            if matched:
                found.append(f"pid={path.name}: {' '.join(matched)}")
        except (OSError, ValueError):
            pass
    return found


def memory_status():
    values = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def process_group_rss(pgid):
    total = 0
    for path in Path("/proc").glob("[0-9]*"):
        try:
            if int((path / "stat").read_text().rsplit(") ", 1)[1].split()[2]) != pgid:
                continue
            for line in (path / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    return total


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_guarded(command, *, cwd, env, stdout=None, max_rss_gib=8., check_gpu=True):
    available, swap_before = memory_status()
    if available < 4 * GIB:
        raise RuntimeError(f"Only {available/GIB:.1f} GiB RAM available; refusing to start")
    if check_gpu:
        jobs = active_ml_jobs()
        if jobs:
            raise RuntimeError("Another training/evaluation job is active; refusing to start:\n" + "\n".join(jobs))
        active = gpu_compute_jobs()
        if active:
            raise RuntimeError("GPU compute process already running; refusing to start:\n" + active)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            available, swap = memory_status()
            rss = process_group_rss(process.pid)
            if available < 1.5 * GIB or swap - swap_before > GIB or rss > max_rss_gib * GIB:
                stop_group(process)
                raise RuntimeError(f"Stopped only this job's process group {process.pid}: "
                                   f"RAM available={available/GIB:.2f} GiB, job RSS={rss/GIB:.2f} GiB, "
                                   f"new swap={(swap-swap_before)/GIB:.2f} GiB; logs/checkpoints retained")
            time.sleep(2)
        return process.returncode
    finally:
        if process is not None:
            stop_group(process)
        signal.signal(signal.SIGTERM, previous_handler)
