"""Strict AMP checkpoint loading and provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.motion_prior.cvae import FEATURE_VERSION, PRIOR_INPUT_COMPATIBILITY, M1MotionPrior

EXPECTED_MOTION_PRIOR_SHA256 = "05a858520790d474f6c1c3dd1f791507c3da91f68899ca88f00348ea4bbff10c"
LEGACY_MOTION_PRIOR_SHA256 = "1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_hash(normalization: dict) -> str:
    payload = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in normalization.items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_m1_motion_prior(path: str | Path, *, device: str | torch.device = "cpu",
                         expected_sha256: str | None = None,
                         input_compatibility: str = "legacy_v1") -> M1MotionPrior:
    """Load AMP CVAE weights only after schema, metadata and normalization checks.

    ``expected_sha256`` can pin a known artifact. Passing ``None`` still checks
    the checkpoint's internal metadata and the strict state-dict structure.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"AMP M1 CVAE checkpoint is missing: {path}; prior was not initialized")
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"AMP M1 CVAE SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")
    if input_compatibility not in PRIOR_INPUT_COMPATIBILITY:
        raise ValueError(f"input_compatibility must be one of {PRIOR_INPUT_COMPATIBILITY}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "model_config", "normalization", "metadata", "step"}
    missing = required - set(ckpt)
    if missing:
        raise ValueError(f"AMP M1 CVAE checkpoint is missing fields: {sorted(missing)}")
    meta = ckpt["metadata"]
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"unsupported AMP CVAE feature schema: {meta.get('feature_version')!r}")
    if meta.get("condition_contract") != "History[4,53]+Goal[8]; L=Velocity, M/LM=Reach; no future condition":
        raise ValueError("AMP CVAE condition contract is absent or incompatible")
    if meta.get("target_contract") != "Future[10,31], 0.2 seconds at 0.02 seconds/frame":
        raise ValueError("AMP CVAE Future31 time/unit contract is absent or incompatible")
    model_cfg = ckpt["model_config"]
    if model_cfg != {"latent_dim": 32, "hidden_dim": 256, "layers": 3}:
        raise ValueError(f"unsupported AMP CVAE architecture: {model_cfg!r}")
    normalization = ckpt["normalization"]
    for key, shape in (("x_mean", (220,)), ("x_std", (220,)), ("y_mean", (31,)), ("y_std", (31,))):
        value = torch.as_tensor(normalization.get(key), dtype=torch.float32)
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint normalization field {key}: expected {shape}")
        if key.endswith("_std") and torch.any(value <= 0):
            raise ValueError(f"normalization {key} must be strictly positive")
    recorded_norm_hash = meta.get("normalization_hash")
    if not recorded_norm_hash or _normalization_hash(normalization) != recorded_norm_hash:
        raise ValueError("AMP CVAE normalization hash mismatch")
    if not meta.get("dataset_hash") or not meta.get("split_hash"):
        raise ValueError("AMP CVAE checkpoint is missing dataset/split provenance")
    prior = M1MotionPrior(
        normalization["x_mean"], normalization["x_std"],
        normalization["y_mean"], normalization["y_std"], **model_cfg,
        input_compatibility=input_compatibility,
    )
    prior.model.load_state_dict(ckpt["model"], strict=True)
    prior.to(device)
    prior.eval()
    prior.source = {
        "path": str(path.resolve()), "sha256": actual_sha,
        "step": int(ckpt["step"]), "feature_version": FEATURE_VERSION,
        "condition_contract": str(meta["condition_contract"]),
        "model_config": dict(model_cfg),
        "dataset_hash": str(meta["dataset_hash"]), "split_hash": str(meta["split_hash"]),
        "normalization_hash": str(recorded_norm_hash), "target_contract": meta["target_contract"],
        "assurance_mode": meta.get("assurance_mode"),
        "strict_fcl_only": meta.get("strict_fcl_only"),
        "input_compatibility": input_compatibility,
        "projected_condition_indices": ("q6/qd6 all four history frames"
            if input_compatibility == "train_constant_compat_v1" else ()),
    }
    return prior
