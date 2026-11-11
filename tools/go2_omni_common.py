"""Shared configuration/materialization for the explicit distribution continuation."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.base.config_materialization import apply_cfg_overrides

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/actor_training_B/last.pt"


def configuration(task="omnidirectional_terrain"):
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        return compose("config", overrides=[f"task=go2_velocity/{task}"])


def environment(cfg, num_envs=2048):
    from unilab.base import registry

    from unisupermimic.tasks.velocity.go2 import registry as go2_registry
    from unisupermimic.tasks.velocity.go2.env import make_go2_env

    registry.ensure_registries()
    env_cfg = registry.materialize_env_config(go2_registry.TASK_NAME)
    apply_cfg_overrides(
        env_cfg,
        EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override(),
    )
    return make_go2_env(env_cfg, num_envs)
