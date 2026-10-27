"""Go2 direct-command actor shape and legacy checkpoint compatibility."""

from __future__ import annotations

import torch

from supermimic.policy.nets import Policy, PolicyCfg


def _velocity_cfg(**kwargs) -> PolicyCfg:
    return PolicyCfg(
        num_joints=12,
        history=10,
        goal_dim=3,
        goal_interface="velocity",
        goal_term_sizes=(3,),
        goal_term_scale=(1.0,),
        decoder_hidden=(128, 64),
        encoder_hidden=(128, 64),
        **kwargs,
    )


def test_direct_velocity_actor_has_423d_input_and_goal_gradient():
    torch.manual_seed(12)
    policy = Policy(_velocity_cfg(goal_encoding="direct"))
    obs = torch.randn(8, 420)
    goal = torch.tensor([[0.3, 0.0, 0.0]]).repeat(8, 1).requires_grad_()

    action = policy(obs, goal)
    first = policy.decoder.mlp.net[0]
    assert first.in_features == 423
    assert action.shape == (8, 12)
    assert not policy.encoders
    action.square().sum().backward()
    assert goal.grad is not None
    assert torch.isfinite(goal.grad).all()
    assert goal.grad.abs().sum() > 0

    with torch.no_grad():
        commands = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0]])
        outputs = policy(obs[:1].expand(3, -1), commands)
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])


def test_new_direct_actor_tanh_keeps_action_inside_execution_support_with_gradient():
    policy = Policy(_velocity_cfg(goal_encoding="direct", action_output="tanh"))
    obs = torch.randn(8, 420)
    goal = torch.randn(8, 3, requires_grad=True)
    action = policy(obs, goal)
    assert action.shape == (8, 12)
    assert action.abs().max() <= 1.0
    action.sum().backward()
    assert goal.grad is not None and torch.isfinite(goal.grad).all()
    assert goal.grad.abs().sum() > 0


def test_encoded_policies_and_old_checkpoint_without_mode_field_stay_compatible(tmp_path):
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    # The default remains the original encoder + decoder, as required for G1/M1 and
    # for historical Go2 checkpoints whose config predates goal_encoding.
    cfg = _velocity_cfg()
    old_policy = Policy(cfg)
    assert cfg.goal_encoding == "encoded"
    assert "encoders.velocity.mlp.net.0.weight" in old_policy.state_dict()
    assert old_policy.decoder.mlp.net[0].in_features == 420 + cfg.latent_dim

    old_cfg = cfg.__dict__.copy()
    old_cfg.pop("goal_encoding")
    path = tmp_path / "legacy_go2_encoded.pt"
    torch.save({"policy": old_policy.state_dict(), "config": {"policy": old_cfg}}, path)

    loaded, saved_cfg = load_policy_from_checkpoint(path, "cpu")
    assert loaded.cfg.goal_encoding == "encoded"
    assert loaded.decoder.mlp.net[0].in_features == 452
    assert saved_cfg["policy"]["goal_interface"] == "velocity"


def test_go2_direct_checkpoint_loads_as_direct_policy(tmp_path):
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    cfg = _velocity_cfg(goal_encoding="direct")
    policy = Policy(cfg)
    path = tmp_path / "go2_direct.pt"
    torch.save({"policy": policy.state_dict(), "config": {"policy": cfg.__dict__}}, path)

    loaded, _ = load_policy_from_checkpoint(path, "cpu")
    assert loaded.cfg.goal_encoding == "direct"
    assert loaded.decoder.mlp.net[0].in_features == 423
    assert not loaded.encoders
