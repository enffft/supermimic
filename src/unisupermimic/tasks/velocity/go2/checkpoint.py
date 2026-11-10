"""Fail-closed actor-only import. WM/critic/optimizer formats are incompatible."""

import torch

from .contract import JOINTS


def map_actor(policy, checkpoint):
    if getattr(policy, "goal_mode", None) != "direct":
        raise ValueError("checkpoint import requires a direct Go2 Actor")
    cfg = checkpoint["config"]["policy"]
    required = {
        "history": 5,
        "num_joints": 12,
        "goal_dim": 3,
        "goal_encoding": "direct",
        "action_output": "tanh",
        "obs_term_scale": [1.0, 5.0, 2.0, 1.0, 1.0],
        "goal_term_scale": [1.0],
        "goal_term_sizes": [3],
    }
    for key, value in required.items():
        actual = cfg.get(key)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            raise ValueError(f"actor semantic mismatch {key}: {actual!r} != {value!r}")
    if checkpoint.get("robot_name") != "go2" or checkpoint.get("task_name") != "velocity_command":
        raise ValueError("expected Go2 velocity checkpoint")
    if tuple(checkpoint.get("robot_joint_names", ())) != tuple(
        n.removeprefix("robot/") for n in JOINTS
    ):
        raise ValueError("checkpoint joint order is missing or incompatible")
    if (
        checkpoint.get("actuator_model_version")
        != "go2_urdf_builtin_pd_kp40_kd1_effort_23p7_35p55_v2"
    ):
        raise ValueError("checkpoint actuator model is incompatible")
    if (
        checkpoint.get("robot_model_sha256")
        != "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4"
    ):
        raise ValueError("checkpoint source robot model fingerprint is incompatible")
    src = checkpoint["policy"]
    if any(k.startswith("encoders.") for k in src):
        raise ValueError("encoded actor cannot map into direct mode")
    if not torch.equal(src["decoder.scale"].cpu(), policy.obs_scale.cpu()):
        raise ValueError("fixed observation scales differ")
    if not torch.equal(src["decoder.goal_scale"].cpu(), torch.ones(3)):
        raise ValueError("fixed goal scales differ")
    mapped = {
        k.removeprefix("decoder.mlp.net."): v
        for k, v in src.items()
        if k.startswith("decoder.mlp.net.")
    }
    expected = policy.dec.state_dict()
    if set(mapped) != set(expected) or any(mapped[k].shape != expected[k].shape for k in expected):
        raise ValueError("actor layer names/widths differ")
    if not all(torch.isfinite(t).all() for t in mapped.values()):
        raise ValueError("actor checkpoint has non-finite parameters")
    policy.dec.load_state_dict(mapped, strict=True)
    return {
        "loaded": "actor_only",
        "tensor_count": len(mapped),
        "optimizer": "fresh",
        "world_model": "fresh",
        "critic": "fresh",
    }


def initialize_actor(policy, path, target_contract):
    """Import only Actor parameters; do not resume WM, optimizers or curriculum."""
    import hashlib
    from pathlib import Path, PosixPath

    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "policy" in checkpoint:
        result = map_actor(policy, checkpoint)
    else:
        source = checkpoint.get("task_contract", {})
        args = checkpoint.get("args", {})
        for key in ("actor", "joints", "context", "action", "sim_dt", "ctrl_dt"):
            if key not in source or source[key] != target_contract[key]:
                raise ValueError(f"Actor protocol mismatch: {key}")
        if args.get("goal_mode") != "direct" or args.get("hist") != 5:
            raise ValueError("Actor initialization requires proven direct History5 protocol")
        from .contract import MODEL

        known_flat = hashlib.sha256(MODEL.read_bytes()).hexdigest()
        if source.get("model_sha256") not in (target_contract["model_sha256"], known_flat):
            raise ValueError("Unrecognized source robot model")
        values = checkpoint["pi"]
        if not torch.equal(values["obs_scale"].cpu(), policy.obs_scale.cpu()):
            raise ValueError("Actor scales differ")
        if not all(torch.isfinite(t).all() for t in values.values()):
            raise ValueError("Actor has nonfinite tensors")
        policy.load_state_dict(values, strict=True)
        result = {"loaded": "unilab_actor_only", "world_model": "independent", "optimizer": "fresh"}
    result.update(
        path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    )
    return result
