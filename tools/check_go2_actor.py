"""Check every imported tensor and every actor layer against the frozen source."""

import argparse
import ast
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration/source_reference/src"))
import importlib.util

from supermimic.policy.nets import Policy as SourcePolicy
from supermimic.policy.nets import PolicyCfg

math_path = ROOT / "migration/source_reference/mjlab_math.py"
spec = importlib.util.spec_from_file_location("source_mjlab_math", math_path)
source_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_math)
quat_apply_inverse = source_math.quat_apply_inverse
rotation_path = ROOT / "migration/source_reference/src/supermimic/kinematics/rotations.py"
rotation_nodes = [
    n
    for n in ast.parse(rotation_path.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "projected_gravity"
]
exec(
    compile(ast.Module(body=rotation_nodes, type_ignores=[]), str(rotation_path), "exec"), globals()
)
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.velocity.go2.checkpoint import map_actor
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp

# Execute the two unmodified pure source functions without importing the old
# mjlab environment (which requires MuJoCo 3.16 and cannot coexist with 3.11).
head_path = ROOT / "migration/source_reference/src/supermimic/world_model/heads.py"
nodes = ast.parse(head_path.read_text()).body
nodes = [
    n
    for n in nodes
    if isinstance(n, ast.FunctionDef)
    and n.name in ("analytic_observation", "flatten_history_term_major")
]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(head_path), "exec"), globals())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/actor_equivalence.json")
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "migration/source_reference/checkpoint_manifest.json").read_text()
    )
    saved = torch.load(manifest["path"], map_location="cpu", weights_only=False, mmap=True)
    cfg = PolicyCfg(**saved["config"]["policy"])
    source = SourcePolicy(cfg)
    source.load_state_dict(saved["policy"], strict=True)
    policy = Policy(42, 3, hist=5, num_joints=12, hidden=cfg.decoder_hidden, goal_mode="direct")
    result = map_actor(policy, saved)
    torch.manual_seed(2026)
    mdp = Go2VelocityMdp()
    flat = torch.randn(64 * 5, 37) * 0.1
    flat[:, 2] = 0.3
    flat[:, 3:7] = torch.nn.functional.normalize(torch.randn(320, 4), dim=-1)
    state = unflatten_state(flat)
    xi = torch.randn(320, 16) * 0.01
    prev = torch.randn(320, 12) * 0.1
    src_state = mdp.source_state(state)
    old_obs = analytic_observation(src_state, prev, xi, mdp.default_q)
    new_obs = mdp.obs(state, prev, xi, torch.zeros(320, dtype=torch.long))
    torch.testing.assert_close(new_obs, old_obs, atol=1e-6, rtol=1e-6)
    # Identical tensor input isolates weight/layout equivalence from quaternion
    # arithmetic roundoff already checked independently above.
    history = old_obs.reshape(64, 5, 42)
    old_history = flatten_history_term_major(
        old_obs.reshape(64, 5, 42).transpose(0, 1), num_joints=12
    )
    goal = torch.randn(64, 3)
    old_layers = []
    new_layers = []
    handles = [
        m.register_forward_hook(lambda m, a, o: old_layers.append(o.detach()))
        for m in source.decoder.mlp.net
    ]
    handles += [
        m.register_forward_hook(lambda m, a, o: new_layers.append(o.detach())) for m in policy.dec
    ]
    old = source(old_history, goal)
    new = policy(history, goal)
    errors = []
    for i, (a, b) in enumerate(zip(old_layers, new_layers, strict=True)):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        errors.append(
            {"layer": i, "shape": list(a.shape), "max_abs_error": float((a - b).abs().max())}
        )
    torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-6)
    result.update(
        {
            "checkpoint_sha256": manifest["sha256"],
            "observation_max_abs_error": float((old_obs - new_obs).abs().max()),
            "output_max_abs_error": float((old - new).abs().max()),
            "layers": errors,
            "batch": 64,
            "limitations": "Actor forward only; no claim of WM/critic/optimizer or training trajectory equivalence",
        }
    )
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
