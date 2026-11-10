"""Materialize the task's Hydra owner without modifying any defaults."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides


def load_config(backend="mujoco", *, terrain=False):
    if terrain and backend != "mujoco":
        raise NotImplementedError("Terrain owner currently supports only MuJoCo")
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        owner = compose(
            "config", overrides=[f"task=go2_velocity/{'terrain' if terrain else backend}"]
        )
    # Register lazily, just as the public task plugin does.
    registry.ensure_registries()
    from . import registry as go2_registry

    cfg = registry.materialize_env_config(go2_registry.TASK_NAME)
    override = EnvConfigAdapter(
        owner, root_dir=root, algo_name="supermimic"
    ).build_task_env_cfg_override()
    apply_cfg_overrides(cfg, override)
    cfg.validate()
    return cfg


def enable_randomization(cfg):
    """Explicit opt-in; the nominal flat baseline is deterministic."""
    from unilab.managers import EventTermCfg

    from .manager_terms import ContextEvent

    cfg.commands["velocity"].domain_randomization = True
    cfg.events["go2_context_startup"] = EventTermCfg(func=ContextEvent, mode="startup")
    cfg.events["go2_context_reset"] = EventTermCfg(func=ContextEvent, mode="reset")
    return cfg
