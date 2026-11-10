"""Compare current flat math to audited migration commit without changing files."""

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.tasks.velocity.go2.mdp import Go2VelocityMdp


def legacy(path, name, revision):
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="1cfffb6594b852512d30253993f00f75248e695b")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pkg = "unisupermimic.rl.algos.supermimic"
    old = legacy(
        "src/unisupermimic/rl/algos/supermimic/networks.py",
        pkg + "._baseline_networks",
        args.baseline,
    )
    pkg = "unisupermimic.tasks.velocity.go2"
    gait = legacy(
        "src/unisupermimic/tasks/velocity/go2/gait.py", pkg + "._baseline_gait", args.baseline
    )
    reward = legacy(
        "src/unisupermimic/tasks/velocity/go2/reward_core.py",
        pkg + "._baseline_reward",
        args.baseline,
    )
    reward.go2_gait_reward_terms = gait.go2_gait_reward_terms
    mdp = legacy(
        "src/unisupermimic/tasks/velocity/go2/mdp.py", pkg + "._baseline_mdp", args.baseline
    )
    mdp.velocity_reward_terms = reward.velocity_reward_terms
    torch.manual_seed(19)
    flat = torch.randn(16, 37) * 0.1
    flat[:, 2] = torch.linspace(0.12, 0.42, 16)
    flat[:, 3] = 1.0
    state = unflatten_state(flat)
    a = torch.randn(16, 12) * 0.3
    xi = torch.randn(16, 16) * 0.01
    g = torch.randn(16, 3) * 0.3
    previous = torch.zeros_like(a)
    frames = torch.zeros(16, dtype=torch.long)
    m0, m1 = mdp.Go2VelocityMdp(), Go2VelocityMdp()
    terms0 = m0.terms(state, state, g, a, previous)
    terms1 = m1.terms(state, state, g, a, previous)
    for key in terms0:
        torch.testing.assert_close(terms0[key], terms1[key], atol=0, rtol=0)
    assert torch.equal(m0.hard_termination(state), m1.hard_termination(state))
    assert torch.equal(m0.soft_termination(state, frames), m1.soft_termination(state, frames))
    p0, p1 = (
        old.Policy(42, 3, hist=5, num_joints=12, goal_mode="direct"),
        networks.Policy(42, 3, hist=5, num_joints=12, goal_mode="direct"),
    )
    p1.load_state_dict(p0.state_dict())
    obs = m0.obs(state, a, xi, frames)[:, None].expand(16, 5, 42)
    with torch.no_grad():
        torch.testing.assert_close(p0(obs, g), p1(obs, g), atol=0, rtol=0)
    w0, w1 = (
        old.WorldModel(num_joints=12, context_dim=16),
        networks.WorldModel(num_joints=12, context_dim=16),
    )
    with torch.no_grad():
        w0.net[-1].weight.normal_(std=0.001)
    w1.load_state_dict(w0.state_dict())
    s0, s1 = state, state
    with torch.no_grad():
        for _ in range(32):
            s0 = w0.step(s0, a, xi)
            s1 = w1.step(s1, a, xi)
            for t0, t1 in zip(s0, s1):
                torch.testing.assert_close(t0, t1, atol=0, rtol=0)
    result = {
        "baseline": args.baseline,
        "actor_input_dim": 213,
        "wm_input_dim": 65,
        "actor_output_max_abs_error": 0.0,
        "wm_h32_max_abs_error": 0.0,
        "reward_terms_max_abs_error": 0.0,
        "hard_soft_termination_identical": True,
        "wm_widths": [2048, 2048, 2048],
        "actor_widths": [2048, 1024, 512, 256],
        "device": "cpu",
        "amp": "none",
        "batch": 16,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
