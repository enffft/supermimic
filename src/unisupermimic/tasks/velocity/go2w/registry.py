"""Register independently; no change to Go2/G1 owners or backend defaults."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .env import make_go2w_env

TASK_NAME = "Go2WVelocitySuperMimic"
registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
registry.register_env(TASK_NAME, make_go2w_env, sim_backend="mujoco")
