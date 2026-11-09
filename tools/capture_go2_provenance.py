"""Capture reproducible dependency state and verify the read-only source inputs."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "migration/environment"
SOURCE = Path("/home/yifeiy/ws_supermimic")
CONDA = "/home/yifeiy/miniconda3/bin/conda"


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, args in {
        "conda-explicit.txt": [CONDA, "list", "--explicit", "-n", "go2_unilab"],
        "pip-freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "pip-check.txt": [sys.executable, "-m", "pip", "check"],
        "nvidia-smi.txt": ["nvidia-smi"],
        "nvidia-query.csv": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv",
        ],
        "os-release.txt": ["cat", "/etc/os-release"],
        "original-conda-explicit.txt": [CONDA, "list", "--explicit", "-n", "supermimic"],
    }.items():
        (OUT / name).write_text(command(args))
    package_names = [
        "torch",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "unilab",
        "unisim-core",
        "unisupermimic",
        "mjbatch-uni",
        "numpy",
        "hydra-core",
        "pytest",
        "ruff",
    ]
    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "import_paths": {
            name: importlib.util.find_spec(name).origin
            for name in ["torch", "mujoco", "warp", "unilab", "unisim", "unisupermimic"]
        },
    }
    import torch

    runtime["torch_cuda"] = torch.version.cuda
    runtime["cuda_available"] = torch.cuda.is_available()
    runtime["repositories"] = {}
    for name, path in [
        ("source", SOURCE),
        ("unilab", ROOT.parent / "UniLab"),
        ("experiment", ROOT),
    ]:
        runtime["repositories"][name] = {
            "path": str(path),
            "head": command(["git", "rev-parse", "HEAD"], path).strip(),
            "branch": command(["git", "branch", "--show-current"], path).strip(),
            "status": command(["git", "status", "--porcelain"], path),
        }
    reference = ROOT / "migration/source_reference"
    source_manifest = json.loads((reference / "manifest.json").read_text())
    frozen_files = {
        name: sha for name, sha in source_manifest.items() if (reference / name).is_file()
    }
    changed = [name for name, sha in frozen_files.items() if digest(SOURCE / name) != sha]
    checkpoint = json.loads((reference / "checkpoint_manifest.json").read_text())
    runtime["source_integrity"] = {
        "checked_source_files": len(frozen_files),
        "changed_files": changed,
        "git_status_matches_snapshot": runtime["repositories"]["source"]["status"]
        == (reference / "git_status.txt").read_text(),
        "checkpoint_sha256": digest(Path(checkpoint["path"])),
        "checkpoint_matches_snapshot": digest(Path(checkpoint["path"])) == checkpoint["sha256"],
    }
    assert not changed and runtime["source_integrity"]["checkpoint_matches_snapshot"]
    assert runtime["repositories"]["unilab"]["status"] == "", "UniLab dependency must remain clean"
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    files = [
        *ROOT.glob("src/unisupermimic/tasks/velocity/**/*.py"),
        *ROOT.glob("src/unisupermimic/assets/robots/go2_velocity/**/*"),
        *ROOT.glob("src/unisupermimic/conf/supermimic/task/go2_velocity/*.yaml"),
        *ROOT.glob("tools/*go2*.py"),
        *ROOT.glob("tests/tasks/test_go2*.py"),
        ROOT / "tests/envs/test_go2_velocity_env.py",
        ROOT / "docs/go2_unilab_migration_report.md",
    ]
    files += [ROOT / name for name in command(["git", "diff", "--name-only"], ROOT).splitlines()]
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in files if path.is_file()}
    (OUT / "go2-files-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(runtime["source_integrity"], indent=2))


if __name__ == "__main__":
    main()
