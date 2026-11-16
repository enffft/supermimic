"""Dense tracking gradients and physical control costs for the v3 experiment."""

import torch

from unisupermimic.tasks.velocity.go2w.contract import MixedControlCfg
from unisupermimic.tasks.velocity.go2w.mdp import Go2WVelocityMdp
from unisupermimic.tasks.velocity.go2w.reward_core import VelocityRewardCfg


def state(mdp, velocity):
    return (torch.tensor([[0., 0., .42]]), torch.eye(3)[None], velocity,
            torch.zeros(1, 3), mdp.default_q[None], torch.zeros(1, 16))


def test_dense_tracking_retains_gradient_far_from_goal_and_minimum_at_target():
    mdp = Go2WVelocityMdp(reward=VelocityRewardCfg(tracking_error_cost=(.25, .6, .3)))
    velocity = torch.tensor([[0., -1.5, 0.]], requires_grad=True)
    current = state(mdp, velocity)
    goal, action = torch.tensor([[0., .15, 0.]]), torch.zeros(1, 16)
    cost = mdp.terms(current, current, goal, action, action)["tracking_error_cost"]
    cost.backward()
    assert velocity.grad[0, 1] > 2
    target = state(mdp, torch.tensor([[0., .15, 0.]]))
    assert mdp.terms(target, target, goal, action, action)["tracking_error_cost"] == 0


def test_action_rate_cost_matches_identical_physical_target_changes():
    reward = VelocityRewardCfg(physical_leg_action_rate=True)
    old = Go2WVelocityMdp(reward=reward)
    new = Go2WVelocityMdp(control=MixedControlCfg(full_leg_target_range=True, leg_limit_fraction=.95), reward=reward)
    action = torch.full((1, 16), .2)
    mapped = action.clone()
    mapped[:, :12] *= .25 / (new.limits[:, 1] - new.default_q)
    a = old.terms(state(old, torch.zeros(1, 3)), state(old, torch.zeros(1, 3)),
                  torch.zeros(1, 3), action, torch.zeros_like(action))
    b = new.terms(state(new, torch.zeros(1, 3)), state(new, torch.zeros(1, 3)),
                  torch.zeros(1, 3), mapped, torch.zeros_like(action))
    torch.testing.assert_close(a["leg_action_rate"], b["leg_action_rate"])


def test_v3_keeps_wide_control_and_changes_collection_and_reward():
    from unisupermimic.tasks.velocity.go2w.config import load_config

    previous, current = load_config("joint_range"), load_config("omni_v3")
    assert previous.actions == current.actions
    assert current.commands["velocity"].actor_level_lookahead == 1
    assert current.commands["velocity"].reward.track_vy == 3
    assert current.commands["velocity"].persistent_wm_coverage
