"""Reusable Go2W conditional dynamics assets, independent of reward and policy."""

import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path

import torch


def dynamics_contract(task_contract):
    # High-resolution map CONTENT and reward/curriculum are deliberately absent.
    # Each step is conditioned on a scan of the current predicted world pose.
    # A different height map is a distribution/validation question, not a new WM.
    keys = (
        "schema", "state_layout", "state", "state_coordinates", "joints", "position_joints",
        "wheel_joints", "physical_qpos_joints", "physical_actuators", "control_protocol",
        "actions", "context", "model_assets_sha256", "terrain_sensor", "sim_dt", "ctrl_dt",
    )
    return {key: task_contract[key] for key in keys}


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def export_world_model(checkpoint, destination, expected_digest=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "opt_wm" not in saved:
        raise ValueError("A continual WM asset requires the WM optimizer state")
    architecture = {key: saved["args"][key] for key in (
        "wm_hidden", "wm_structured", "terrain_conditioning", "terrain_sensor_grad",
    )}
    asset = dict(
        format="go2w_continual_dynamics_asset", wm=saved["wm"], opt_wm=saved["opt_wm"],
        dynamics_contract=dynamics_contract(saved["task_contract"]), architecture=architecture,
        tensor_sha256=tensor_digest(saved["wm"]), source_checkpoint=str(Path(checkpoint).resolve()),
        source_iteration=saved["iteration"], source_updates=saved.get("update_counts", {}),
        wm_updates=saved.get("update_counts", {}).get("wm", 0),
        validation_status="candidate; consult independent physics and WM assessment before skill claims",
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        if expected_digest and existing["tensor_sha256"] != expected_digest:
            raise ValueError("Shared WM asset changed since loading; refusing to overwrite another writer")
        history = destination.parent / "history"
        history.mkdir(exist_ok=True)
        backup = history / f"{destination.stem}_{existing['tensor_sha256'][:16]}.pt"
        if not backup.exists():
            shutil.copy2(destination, backup)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(asset, temporary)
    temporary.replace(destination)
    return {key: value for key, value in asset.items() if key not in ("wm", "opt_wm", "dynamics_contract")}


def load_world_model(learner, path):
    asset = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if asset.get("format") != "go2w_continual_dynamics_asset" or "opt_wm" not in asset:
        raise ValueError("Expected an exported Go2W world-model asset")
    if asset["dynamics_contract"] != dynamics_contract(learner.env.supermimic_contract):
        raise ValueError("World-model state, robot, control or sensor contract differs")
    args = asdict(learner.cfg)
    for key, value in asset["architecture"].items():
        current = args[key]
        if isinstance(value, (list, tuple)):
            value, current = tuple(value), tuple(current)
        if value != current:
            raise ValueError(f"World-model architecture differs: {key}")
    if tensor_digest(asset["wm"]) != asset["tensor_sha256"]:
        raise ValueError("World-model tensor fingerprint does not match")
    learner.wm.load_state_dict(asset["wm"])
    learner.opt_wm.load_state_dict(asset["opt_wm"])
    learner.update_counts["wm"] = asset["wm_updates"]
    learner.wm.requires_grad_(True)
    learner.wm.train()
    record = {key: value for key, value in asset.items() if key not in ("wm", "opt_wm", "dynamics_contract")}
    record.update(path=str(Path(path).resolve()), frozen=False,
                  restored=["network", "all_normalizers", "WM_optimizer", "WM_update_count"],
                  update_mode="unchanged SuperMimic online world-model updates")
    learner.extra_state["continual_world_model"] = record
    return record
