"""Full-width, tensor-by-tensor UniLab flat WM -> 77D terrain expansion check."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2.contract import MODEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    values = torch.load(args.source, map_location="cpu", weights_only=True)
    contract = values["task_contract"]
    cfg = values["args"]
    assert contract["model_sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert contract["terrain"] is None and contract["context"] == "friction,bias12,com3"
    assert not cfg.get("terrain_conditioning", False) and cfg["goal_mode"] == "direct"
    source = WorldModel(
        hidden=cfg["wm_hidden"], num_joints=12, context_dim=16, structured=cfg["wm_structured"]
    )
    source.load_state_dict(values["wm"], strict=True)
    target = WorldModel(
        hidden=cfg["wm_hidden"],
        num_joints=12,
        context_dim=16,
        structured=cfg["wm_structured"],
        terrain_dim=77,
    )
    target.initialize_from_flat(source)
    tensor_count = 0
    for key, value in source.state_dict().items():
        copied = target.state_dict()[key]
        if key == "net.0.weight":
            copied = copied[:, :65]
        if key in ("in_norm.mean", "in_norm.var"):
            copied = copied[:65]
        assert torch.equal(value, copied), key
        tensor_count += 1
    assert target.net[0].weight[:, 65:].eq(0).all()
    torch.manual_seed(61)
    flat = torch.randn(64, 37) * 0.1
    flat[:, 3] = 1.0
    flat[:, 2] = 0.3
    state = unflatten_state(flat)
    action = torch.randn(64, 12) * 0.2
    context = torch.randn(64, 16)
    old_layers = []
    new_layers = []
    hooks = [
        layer.register_forward_hook(lambda m, a, o: old_layers.append(o.detach()))
        for layer in source.net
    ]
    hooks += [
        layer.register_forward_hook(lambda m, a, o: new_layers.append(o.detach()))
        for layer in target.net
    ]
    with torch.no_grad():
        old = source.step(state, action, context)
        new = target.step(state, action, context, torch.randn(64, 77) * 0.3)
    for hook in hooks:
        hook.remove()
    errors = []
    for index, (a, b) in enumerate(zip(old_layers, new_layers, strict=True)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        errors.append({"layer": index, "max_abs_error": float((a - b).abs().max())})
    maximum = max(float((a - b).abs().max()) for a, b in zip(old, new))
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    result = {
        "source": str(args.source),
        "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "mapped_tensor_count": tensor_count,
        "wm_widths": cfg["wm_hidden"],
        "input_dims": [65, 142],
        "new_columns_zero": True,
        "scan_normalizer_initialization": "mean=0,var=1; source count preserved",
        "per_layer": errors,
        "max_state_error": maximum,
        "batch": 64,
        "device": "cpu",
        "amp": "none",
        "meaning": "forward equivalence for same state/action/context; not training/optimizer equivalence",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
