"""Bounded physical sanity check of the existing deployment policy adapter."""

import argparse
import json
from pathlib import Path

import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec, unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.velocity.go2.command_sampler import command_modes
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT
from unisupermimic.tasks.velocity.go2w.deployment import DeploymentPolicy
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, default=Path("/home/yifeiy/rl_sar/policy/go2w"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = [("stop", (0., 0., 0.)), ("forward", (.3, 0., 0.)), ("left", (0., .15, 0.)),
             ("right", (0., -.15, 0.)), ("turn_left", (0., 0., .2)), ("turn_right", (0., 0., -.2))]
    cfg = load_config("joint_range")
    cfg.commands["velocity"].sampler.domain_weights = (1., 0., 0., 0., 0., 0.)
    cfg.commands["velocity"].persistent_wm_coverage = False
    env = make_go2w_env(cfg, len(cases) * 4)
    try:
        env.reset()
        env.set_episode_length_buf(torch.zeros(env.num_envs, dtype=torch.long))
        c = env.command_manager.get_term("velocity")
        teacher = DeploymentPolicy(args.deployment, env.num_envs, c.mdp.control, c.mdp.default_q, c.mdp.limits)
        goal = torch.tensor([case[1] for case in cases]).repeat_interleave(4, dim=0)
        live = torch.ones(env.num_envs, dtype=torch.bool)
        falls = torch.zeros_like(live)
        count = torch.zeros(env.num_envs)
        actual, sq = torch.zeros(env.num_envs, 3), torch.zeros(env.num_envs, 3)
        qmin, qmax = torch.full((12,), float("inf")), torch.full((12,), -float("inf"))
        clipped = 0
        target_samples = 0
        for step in range(500):
            info = env.current_info()
            state = unflatten_state(info["s"], LAYOUT)
            c.g.copy_(goal)
            c.command_mode.copy_(command_modes(goal))
            action = teacher.action(state, goal, info["ep_step"] == 0)
            row = env.step(action).info[SUPERMIMIC_TRANSITION_KEY]
            state = unflatten_state(row["s1"], LAYOUT)
            measured = torch.cat((matvec(state[1].transpose(-1, -2), state[2])[:, :2],
                                  matvec(state[1].transpose(-1, -2), state[3])[:, 2:3]), -1)
            valid = live & (step >= 50)
            actual += measured * valid[:, None]
            sq += (measured - goal).square() * valid[:, None]
            count += valid
            if live.any():
                qmin = torch.minimum(qmin, teacher.last_target[live].amin(0))
                qmax = torch.maximum(qmax, teacher.last_target[live].amax(0))
                clipped += int(teacher.last_clipped[live].sum())
                target_samples += int(live.sum()) * 12
            falls |= live & row["term"]
            live &= ~row["done"]
        rows = []
        for i, case in enumerate(cases):
            part = slice(i * 4, (i + 1) * 4)
            num = count[part].sum()
            rows.append(dict(name=case[0], goal=case[1], falls=int(falls[part].sum()), samples=int(num),
                             actual=(actual[part].sum(0) / num).tolist() if num else None,
                             RMSE=(sq[part].sum(0) / num).sqrt().tolist() if num else None))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(rows=rows, provenance=teacher.provenance,
                                              target_min=qmin.tolist(), target_max=qmax.tolist(),
                                              clipped_target_fraction=clipped / max(target_samples, 1),
                                              note="4 nominal deterministic replicas; transfer feasibility only, not independent success rate"), indent=2))
        print(json.dumps(rows, indent=2))
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
