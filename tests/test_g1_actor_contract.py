"""G1 pure-proprio and deployment-facing Actor contracts."""

import pytest
import torch


def test_g1_pure_proprio_rejects_anchor_inputs():
    from supermimic.config import EnvCfg, validate_g1_actor_contract

    cfg = EnvCfg(anchor_obs=True)
    with pytest.raises(ValueError, match="pure-proprio"):
        validate_g1_actor_contract(cfg)

    # Legacy privileged checkpoints remain an explicit, opt-in path.
    cfg.g1_pure_proprio = False
    validate_g1_actor_contract(cfg)

    cfg.g1_pure_proprio = True
    cfg.anchor_obs = False
    cfg.motion_goal_schema = "full_body"
    with pytest.raises(ValueError, match="68D/frame"):
        validate_g1_actor_contract(cfg)


def test_g1_actor_contract_is_930_plus_680_and_excludes_critic_inputs():
    from supermimic.policy.nets import ActorInputContract, Policy, PolicyCfg

    contract = ActorInputContract(history=10, num_joints=29, goal_dim=680)
    assert contract.proprioception_dim_per_frame == 93
    assert contract.obs_dim == 930
    contract.validate(torch.zeros(4, 930), torch.zeros(4, 680))
    with pytest.raises(ValueError, match="930"):
        contract.validate(torch.zeros(4, 934), torch.zeros(4, 680))
    with pytest.raises(ValueError, match="680"):
        contract.validate(torch.zeros(4, 930), torch.zeros(4, 689))

    policy = Policy(PolicyCfg(goal_dim=680, goal_encoding="encoded"))
    assert not hasattr(policy, "xi")
    assert not hasattr(policy, "e_root")
    assert policy.input_contract.obs_dim == 930
    assert policy.input_contract.goal_dim == 680
