"""Pinned formal experiment configuration and reproducible source snapshots."""

import hashlib
import json
import subprocess
import tarfile
from dataclasses import asdict, fields
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig

WM_PATH = Path("/home/yifeiy/ws_go2_unilab/artifacts/world_models/go2w_conditioned.pt")
WM_FILE_SHA = "3c3c4d1c5e70bb12c8cc68274fda39003e465c6ec1fbe745c0729f9f7d028610"
WM_TENSOR_SHA = "ed04040096a0b0254fb58ef18ffa65ef7bf2c1f07e15fa7a2e6df36c7428c490"
ROOT = Path(__file__).resolve().parents[5]
BUDGET_SECONDS = 14400


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def configuration():
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        owner = compose("config", overrides=["task=go2w_velocity/train"])
    raw = OmegaConf.to_container(owner.algo, resolve=True)
    names = {f.name for f in fields(SuperMimicConfig)}
    metadata = {"algo", "algo_log_name", "load_run", "checkpoint", "resume", "num_envs"}
    if unknown := raw.keys() - names - metadata:
        raise ValueError(f"Unknown learner config: {unknown}")
    cfg = SuperMimicConfig(**{k: v for k, v in raw.items() if k in names})
    expected = dict(
        horizon=8,
        wm_window=8,
        buffer_steps=1024,
        wm_batch=2048,
        pi_batch=2048,
        wm_batches=10,
        wm_only_iters=0,
        goal_mode="direct",
        wm_structured=True,
        critic_terrain_conditioning=True,
        checkpoint_replay=True,
        strict_finite=True,
    )
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"Formal {key} must equal {value}")
    if raw["num_envs"] != 4096:
        raise ValueError("Formal experiment requires 4096 environments")
    if cfg.gamma != float(owner.env.commands.velocity.reward.progress_gamma):
        raise ValueError("Potential discount differs")
    if cfg.init_actor_from or cfg.init_wm_from or raw["resume"]:
        raise ValueError("Use only strict WM-only initialization")
    return owner, cfg


def source_identity():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).split(b"\0")
    manifest = {}
    for raw in sorted(set(paths)):
        if not raw:
            continue
        relative = raw.decode()
        path = ROOT / relative
        if path.is_file() and (
            relative.startswith(("src/", "tools/", "tests/"))
            or relative in ("pyproject.toml", "uv.lock")
        ):
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return digest, manifest


def snapshot(output, owner, cfg):
    identity, manifest = source_identity()
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in manifest:
            archive.add(ROOT / path, arcname=path, recursive=False)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(owner, resolve=True))
    write_json(output / "effective_algo.json", asdict(cfg))
    write_json(output / "source_manifest.json", dict(sha256=identity, files=manifest))
    write_json(
        output / "provenance.json",
        dict(
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            source_sha256=identity,
        ),
    )
    (output / "dirty.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    )
    return identity
