"""Materialize a Go2W task owner without constructing a backend."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides


def load_config(profile="terrain"):
    if profile not in ("base", "mujoco", "flat", "mixed", "terrain", "curriculum"):
        raise ValueError("Unknown Go2W config owner")
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"):
        owner = compose("config", overrides=[f"task=go2w_velocity/{profile}"])
    registry.ensure_registries()
    from .registry import TASK_NAME
    cfg = registry.materialize_env_config(TASK_NAME)
    apply_cfg_overrides(cfg, EnvConfigAdapter(owner, root_dir=root, algo_name="supermimic").build_task_env_cfg_override())
    cfg.validate()
    return cfg
