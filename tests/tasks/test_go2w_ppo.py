"""PPO integration contracts: reset history, privileged isolation and native update."""
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from unisupermimic.rl.algos.supermimic.geometry import flatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp
from unisupermimic.tasks.velocity.go2w.ppo import AlignedActor, Go2WPpoEnv, advance_history, training_config


def test_history_reset_and_shift():
    old = torch.arange(2*5*50).reshape(2,5,50).float()
    frame = torch.full((2,50), -7.)
    updated = advance_history(old, frame, torch.tensor([0, 17]))
    torch.testing.assert_close(updated[0], frame[0].expand(5,50))
    torch.testing.assert_close(updated[1,:4], old[1,1:])
    torch.testing.assert_close(updated[1,-1], frame[1])


def test_actor_mean_and_privileged_isolation():
    obs = TensorDict({'policy': torch.randn(3,253), 'critic': torch.randn(3,44)}, batch_size=[3])
    model = AlignedActor(obs, {'actor':['policy']}, 'actor', 16, hidden_dims=(32,16),
                         distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution',
                                           'init_std':.02, 'std_type':'log'})
    expected = model.mlp.policy(obs['policy'][:,:250].reshape(3,5,50),obs['policy'][:,250:])
    torch.testing.assert_close(model(obs), expected)
    obs['critic'].fill_(1e6)
    torch.testing.assert_close(model(obs), expected)


class FakeTask:
    num_envs = 8
    cfg = SimpleNamespace(max_episode_seconds=10., ctrl_dt=.02)
    supermimic_contract = {}
    supermimic_mdp = Go2WVelocityMdp()
    def __init__(self):
        self.info = {'s':flatten_state((torch.tensor([[0.,0.,.42]]).expand(8,3),
                       torch.eye(3).expand(8,3,3), torch.zeros(8,3), torch.zeros(8,3),
                       self.supermimic_mdp.default_q.expand(8,12), torch.zeros(8,16))),
                     'g':torch.zeros(8,3), 'a_prev':torch.zeros(8,16),
                     'xi':torch.empty(8,0), 'f':torch.empty(8,0), 'ep_step':torch.zeros(8,dtype=torch.long)}
    def set_collection_exploration(self, std):
        assert std == 0.
    def current_info(self):
        return self.info
    def step(self, action):
        self.info['ep_step'] += 1
        self.info['a_prev'] = action.clamp(-1,1)
        done, term = torch.zeros(8,dtype=torch.bool), torch.zeros(8,dtype=torch.bool)
        done[:2], term[0] = True, True
        self.info['ep_step'][:2] = 0
        return SimpleNamespace(info={SUPERMIMIC_TRANSITION_KEY:{
            'r': 1.-action.square().mean(-1), 'done': done, 'term':term}})


def test_bridge_timeout_masks_and_upstream_ppo_update():
    env = Go2WPpoEnv(FakeTask(), device='cpu')
    initial = env.get_observations()['policy'].clone()
    env.get_observations()
    torch.testing.assert_close(initial, env.get_observations()['policy'])
    cfg = training_config(8,2,8,1)
    cfg['actor']['hidden_dims'] = [32,16]
    cfg['critic']['hidden_dims'] = [32,16]
    runner = OnPolicyRunner(env,cfg,device='cpu')
    before = {k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(2):
            action = runner.alg.act(obs)
            obs,reward,done,extras = env.step(action)
            assert extras['time_outs'].tolist() == [False,True,False,False,False,False,False,False]
            runner.alg.process_env_step(obs,reward,done,extras)
        runner.alg.compute_returns(obs)
    loss = runner.alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in loss.values())
    assert any(not torch.equal(value,before[key]) for key,value in runner.alg.actor.state_dict().items())
