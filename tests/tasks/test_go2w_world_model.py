"""Reusable WM identity excludes task preferences and includes real dynamics."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig
from unisupermimic.tasks.velocity.go2w.world_model import (
    dynamics_contract,
    export_world_model,
    load_world_model,
    tensor_digest,
)


def contract():
    keys = (
        "schema", "state_layout", "state", "state_coordinates", "joints", "position_joints",
        "wheel_joints", "physical_qpos_joints", "physical_actuators", "control_protocol",
        "actions", "context", "model_assets_sha256", "terrain_sensor", "sim_dt", "ctrl_dt",
    )
    return {key: "same" for key in keys} | {"reward": "one", "commands": "one", "terrain_fingerprint": "map1"}


def test_reward_commands_and_height_map_are_not_world_model_identity():
    a = contract()
    b = deepcopy(a)
    b.update(reward="two", commands="two", terrain_fingerprint="map2")
    assert dynamics_contract(a) == dynamics_contract(b)
    for key in ("actions", "state_layout", "model_assets_sha256", "terrain_sensor"):
        changed = deepcopy(b)
        changed[key] = "different"
        assert dynamics_contract(changed) != dynamics_contract(a)


def test_asset_roundtrip_continues_optimizer_and_rejects_control_change(tmp_path):
    cfg = SuperMimicConfig()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    source = tmp_path / "training.pt"
    torch.save(dict(args=asdict(cfg), wm=model.state_dict(), opt_wm=optimizer.state_dict(),
                    update_counts={"wm": 42}, task_contract=contract(), iteration=42), source)
    path = tmp_path / "wm.pt"
    export_world_model(source, path)
    learner = SimpleNamespace(cfg=cfg, env=SimpleNamespace(supermimic_contract=contract()),
                              wm=torch.nn.Linear(3, 2), extra_state={}, update_counts={"wm": 0})
    learner.opt_wm = torch.optim.Adam(learner.wm.parameters())
    record = load_world_model(learner, path)
    assert record["source_iteration"] == 42
    assert not record["frozen"]
    assert learner.update_counts["wm"] == 42
    assert tensor_digest(learner.wm.state_dict()) == tensor_digest(model.state_dict())
    assert all(parameter.requires_grad for parameter in learner.wm.parameters())
    assert all(value["step"] == 1 for value in learner.opt_wm.state.values())
    # Standard next optimization step continues Adam momentum and counter.
    learner.opt_wm.zero_grad()
    learner.wm(torch.ones(1, 3)).sum().backward()
    learner.opt_wm.step()
    assert all(value["step"] == 2 for value in learner.opt_wm.state.values())
    learner.env.supermimic_contract["reward"] = "other reward"
    load_world_model(learner, path)
    learner.env.supermimic_contract["actions"] = "different control"
    with pytest.raises(ValueError, match="control or sensor"):
        load_world_model(learner, path)


def test_asset_publish_preserves_history_and_rejects_stale_writer(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    source = tmp_path / "training.pt"
    path = tmp_path / "wm.pt"

    def save_source():
        torch.save(dict(args=asdict(SuperMimicConfig()), wm=model.state_dict(),
                        opt_wm=optimizer.state_dict(), update_counts={"wm": 0},
                        task_contract=contract(), iteration=0), source)

    save_source()
    original = export_world_model(source, path)["tensor_sha256"]
    with torch.no_grad():
        model.bias.add_(1)
    save_source()
    current = export_world_model(source, path, expected_digest=original)["tensor_sha256"]
    assert current != original
    history = tmp_path / "history" / f"wm_{original[:16]}.pt"
    assert torch.load(history, weights_only=True)["tensor_sha256"] == original
    with pytest.raises(ValueError, match="another writer"):
        export_world_model(source, path, expected_digest=original)
    assert torch.load(path, weights_only=True)["tensor_sha256"] == current
