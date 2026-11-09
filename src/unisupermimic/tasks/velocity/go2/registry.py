from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .env import make_go2_env

TASK_NAME = "Go2VelocitySuperMimic"
registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
for backend in ("mujoco", "mjwarp"):
    registry.register_env(TASK_NAME, make_go2_env, sim_backend=backend)
