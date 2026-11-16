"""Exact WM-only transfers reject corrupted assets before mutating a learner."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2w.checkpoint import initialize_wm_only
from unisupermimic.tasks.velocity.go2w.contract import DEFAULT_Q, LAYOUT
from unisupermimic.tasks.velocity.go2w.world_model import tensor_digest


def fixture(tmp_path):
    cfg = replace(
        SuperMimicConfig(),
        wm_hidden=(8,),
        wm_structured=True,
        terrain_conditioning=True,
        terrain_sensor_grad=True,
    )
    contract = {
        key: "same"
        for key in (
            "schema",
            "state_layout",
            "state",
            "state_coordinates",
            "joints",
            "position_joints",
            "wheel_joints",
            "physical_qpos_joints",
            "physical_actuators",
            "control_protocol",
            "actions",
            "context",
            "model_assets_sha256",
            "terrain_sensor",
            "sim_dt",
            "ctrl_dt",
        )
    }

    def model():
        return WorldModel(
            hidden=(8,),
            num_joints=16,
            context_dim=0,
            structured=True,
            terrain_dim=77,
            state_layout=LAYOUT,
        )

    source = model()
    with torch.no_grad():
        source.net[-1].weight.normal_(std=0.001)
        for norm in (source.in_norm, source.out_norm, source.inc_norm):
            norm.mean.uniform_(-0.001, 0.001)
            norm.var.fill_(0.1)
            norm.count.fill_(12345)
    learner = SimpleNamespace(
        cfg=cfg,
        env=SimpleNamespace(supermimic_contract=contract),
        wm=model(),
        pi=torch.nn.Linear(7, 16),
        critic=torch.nn.Linear(48, 1),
        pi_avg=None,
        iteration=0,
        sim_steps=0,
        buffer=SimpleNamespace(size=0),
        extra_state={},
        update_counts=dict(wm=0, actor=0, critic=0),
        initialization=dict(wm="fresh", actor="fresh", critic="fresh"),
        log_dir=tmp_path,
    )
    learner.opt_wm = torch.optim.Adam(learner.wm.parameters())
    learner.opt_pi = torch.optim.Adam(learner.pi.parameters())
    learner.opt_v = torch.optim.Adam(learner.critic.parameters())
    asset = dict(
        format="go2w_continual_dynamics_asset",
        wm=source.state_dict(),
        dynamics_contract=contract.copy(),
        architecture=dict(
            wm_hidden=[8], wm_structured=True, terrain_conditioning=True, terrain_sensor_grad=True
        ),
        tensor_sha256=tensor_digest(source.state_dict()),
        wm_updates=77,
        # A former optimizer is intentionally present and must be ignored.
        opt_wm={"invalid_old_optimizer": True},
        pi={"do_not_load": torch.ones(1)},
        critic={"do_not_load": torch.ones(1)},
    )
    path = tmp_path / "wm.pt"
    return learner, source, asset, path


def transfer(learner, asset, path):
    torch.save(asset, path)
    return initialize_wm_only(
        learner,
        path,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_tensor_sha256=asset["tensor_sha256"],
    )


def test_wm_only_warmstart(tmp_path):
    learner, source, asset, path = fixture(tmp_path)
    actor = tensor_digest(learner.pi.state_dict())
    critic = tensor_digest(learner.critic.state_dict())
    record = transfer(learner, asset, path)
    assert tensor_digest(learner.wm.state_dict()) == tensor_digest(source.state_dict())
    assert tensor_digest(learner.pi.state_dict()) == actor
    assert tensor_digest(learner.critic.state_dict()) == critic
    assert learner.update_counts == dict(wm=0, actor=0, critic=0)
    assert all(not opt.state for opt in (learner.opt_wm, learner.opt_pi, learner.opt_v))
    assert record["normalizer_counts"] == {
        prefix + ".count": 12345.0 for prefix in ("in_norm", "out_norm", "inc_norm")
    }
    state = (
        torch.tensor([[0.0, 0.0, 0.42]]),
        torch.eye(3)[None],
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.tensor([DEFAULT_Q]),
        torch.zeros(1, 16),
    )
    a, b = state, state
    for _ in range(8):
        inputs = (torch.zeros(1, 16), torch.empty(1, 0), torch.full((1, 77), 0.42))
        a = learner.wm.step(a, *inputs)
        b = source.step(b, *inputs)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    before = learner.wm.in_norm.count.item()
    learner.wm.in_norm.update(torch.zeros(4, 134))
    assert learner.wm.in_norm.count.item() == before + 4
    learner.wm.step(state, *inputs)[2].sum().backward()
    learner.opt_wm.step()
    assert all(v["step"] == 1 for v in learner.opt_wm.state.values())


@pytest.mark.parametrize(
    "corruption",
    ["missing", "shape", "dtype", "nan", "variance", "count", "hash", "architecture", "protocol"],
)
def test_reject_invalid_asset_without_partial_transfer(tmp_path, corruption):
    learner, source, asset, path = fixture(tmp_path)
    before = tensor_digest(learner.wm.state_dict())
    if corruption == "missing":
        asset["wm"].pop("inc_norm.count")
    elif corruption == "shape":
        asset["wm"]["net.0.weight"] = torch.zeros(8, 133)
    elif corruption == "dtype":
        asset["wm"]["in_norm.mean"] = asset["wm"]["in_norm.mean"].double()
    elif corruption == "nan":
        asset["wm"]["out_norm.mean"][0] = float("nan")
    elif corruption == "variance":
        asset["wm"]["in_norm.var"][0] = -1
    elif corruption == "count":
        asset["wm"]["inc_norm.count"].fill_(-1)
    elif corruption == "hash":
        asset["tensor_sha256"] = "wrong"
    elif corruption == "architecture":
        asset["architecture"]["wm_hidden"] = [16]
    else:
        asset["dynamics_contract"]["actions"] = "different"
    with pytest.raises(ValueError):
        transfer(learner, asset, path)
    assert tensor_digest(learner.wm.state_dict()) == before
    assert not learner.opt_wm.state


def test_reject_nonfresh_and_file_hash(tmp_path):
    learner, _, asset, path = fixture(tmp_path)
    learner.iteration = 1
    with pytest.raises(ValueError, match="fresh"):
        transfer(learner, asset, path)
    learner.iteration = 0
    with pytest.raises(ValueError, match="file SHA256"):
        initialize_wm_only(
            learner,
            path,
            expected_file_sha256="wrong",
            expected_tensor_sha256=asset["tensor_sha256"],
        )
