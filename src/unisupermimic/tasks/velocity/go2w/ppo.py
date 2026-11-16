"""Go2W task bridge to upstream RSL-RL PPO; no algorithm math lives here."""

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from .contract import LAYOUT, POLICY_SPEC


def advance_history(history, observation, episode_step):
    """Newest frame last; replicate reset frame instead of leaking old episodes."""
    shifted = torch.cat((history[:, 1:], observation[:, None]), dim=1)
    return torch.where((episode_step == 0)[:, None, None],
                       observation[:, None].expand_as(shifted), shifted)


class PolicyHead(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.policy = Policy(50, 3, hist=5, num_joints=16, hidden=hidden,
                             goal_mode="direct", direct_spec=POLICY_SPEC)

    def forward(self, packed):
        return self.policy(packed[..., :250].reshape(-1, 5, 50), packed[..., 250:])


class AlignedActor(MLPModel):
    """Native Gaussian model with the exact previous deterministic Actor mean."""
    def __init__(self, *args, hidden_dims=(2048, 1024, 512, 256), **kwargs):
        super().__init__(*args, hidden_dims=hidden_dims, **kwargs)
        if self.obs_dim != 253 or self.obs_normalization:
            raise ValueError("Aligned Go2W Actor requires unnormalized 5x50+3 inputs")
        self.mlp = PolicyHead(hidden_dims)


class Go2WPpoEnv(VecEnv):
    """Expose the existing task rewards, resets and curriculum to native PPO."""
    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = 16
        self.max_episode_length = round(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.device = torch.device(device)
        self.cfg = {"task_contract": env.supermimic_contract}
        self.history = torch.zeros(self.num_envs, 5, 50)
        self.transitions = 0
        env.set_collection_exploration(0.)  # Gaussian sampling is owned by PPO.
        self._observation = self._observe()

    @property
    def episode_length_buf(self):
        return self.env.current_info()["ep_step"].to(self.device)

    def _observe(self):
        info = self.env.current_info()
        state = unflatten_state(info["s"], LAYOUT)
        mdp = self.env.supermimic_mdp
        obs = mdp.obs(state, info["a_prev"], info["xi"], info["f"])
        self.history = advance_history(self.history, obs, info["ep_step"])
        actor = torch.cat((self.history.flatten(1), info["g"]), -1)
        critic = torch.cat((mdp.critic_features(state, info["f"], info["xi"]), info["g"]), -1)
        return TensorDict({"policy": actor.to(self.device), "critic": critic.to(self.device)},
                          batch_size=[self.num_envs])

    def get_observations(self):
        # Calls from runner initialization must not advance temporal history.
        return self._observation

    def step(self, actions):
        step = self.env.step(actions.detach().cpu())
        row = step.info[SUPERMIMIC_TRANSITION_KEY]
        self.transitions += self.num_envs
        self._observation = self._observe()  # post-auto-reset observation
        extras = {"time_outs": (row["done"] & ~row["term"]).to(self.device)}
        return (self._observation, row["r"].to(self.device),
                row["done"].to(self.device), extras)


def training_config(envs=4096, rollout=24, minibatch=2048, epochs=5):
    samples = envs * rollout
    if samples % minibatch:
        raise ValueError("Rollout size must divide evenly into minibatches")
    return {
        "num_steps_per_env": rollout, "save_interval": 1000000,
        "logger": "tensorboard", "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "unisupermimic.tasks.velocity.go2w.ppo:AlignedActor",
                  "hidden_dims": [2048, 1024, 512, 256], "activation": "swish",
                  "obs_normalization": False,
                  "distribution_cfg": {"class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                                       "init_std": .02, "std_type": "log"}},
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [256, 128],
                   "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "rsl_rl.algorithms:PPO", "value_loss_coef": 1.,
                      "use_clipped_value_loss": True, "clip_param": .2,
                      "entropy_coef": .01, "num_learning_epochs": epochs,
                      "num_mini_batches": samples // minibatch, "learning_rate": 2e-4,
                      "schedule": "adaptive", "gamma": .99, "lam": .95,
                      "desired_kl": .01, "max_grad_norm": 1.},
    }
