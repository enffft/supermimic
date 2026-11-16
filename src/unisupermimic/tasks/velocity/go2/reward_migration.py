"""Strict, named clearance-only migration. No generic strict-load bypass."""

from copy import deepcopy

import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

from .contract import DT
from .feet import go2_feet_world
from .gait import Go2GaitCfg, clearance_region_weight, go2_gait_reward_terms

MIGRATION = "go2_stairs_clearance_reward_v1"
ALLOWED = {
    "feet_clearance",
    "clearance_target",
    "clearance_sigma",
    "clearance_mode",
    "clearance_tau",
    "clearance_region",
}


def validate_reward_migration(old, new):
    """Reject any protected difference, including non-clearance gait rewards."""
    a, b = deepcopy(old), deepcopy(new)
    changes = []
    for contract in (a, b):
        gait = contract["reward"]["gait"]
        for key in ALLOWED:
            gait.pop(key, None)
        velocity = contract["reward_manager"]["velocity"]
        params = velocity.get("params", {})
        override = params.get("gait", {})
        if set(override) - ALLOWED:
            raise ValueError("Reward migration cannot change other gait fields")
        params.pop("gait", None)
    if a != b:
        raise ValueError("Protected task semantics changed outside clearance reward")
    for key in sorted(ALLOWED):
        defaults = getattr(Go2GaitCfg(), key)
        x, y = old["reward"]["gait"].get(key, defaults), new["reward"]["gait"].get(key, defaults)
        if x != y:
            changes.append(dict(field=f"gait.{key}", old=x, new=y))
    return changes


@torch.no_grad()
def relabel_replay_clearance(replay, old_cfg, mdp, chunk=16384):
    """Only replace the clearance contribution; retain measured torque rewards.

    Replay.r is cached logging data, NOT the critic's training target. Critic
    targets are newly imagined rewards. Still relabel so coverage stats are honest.
    """
    data = replay["data"]
    t, n = data["r"].shape
    rows = (torch.arange(replay["size"]) + replay["ptr"] - replay["size"]) % t
    idx = (rows[:, None] * n + torch.arange(n)[None]).flatten()
    rates = []
    for start in range(0, len(idx), chunk):
        pick = idx[start : start + chunk]
        state = unflatten_state(data["s1"].view(t * n, -1)[pick].to(mdp.device))
        source = mdp.source_state(state)
        feet, _ = go2_feet_world(source)
        ground = mdp.terrain.ground_height(feet[..., :2]) if mdp.terrain is not None else None
        goal = data["g"].view(t * n, -1)[pick].to(mdp.device)
        contributions = []
        for cfg in (old_cfg, mdp.params.cfg.gait):
            weight = (
                clearance_region_weight(feet[..., :2], mdp.clearance_boxes)
                if cfg.clearance_region == "stairs"
                else None
            )
            contributions.append(
                go2_gait_reward_terms(
                    source, goal, cfg, foot_ground=ground, clearance_weight=weight
                )["feet_clearance"]
            )
        delta = DT * (contributions[1] - contributions[0])
        data["r"].view(-1)[pick] += delta.cpu()
        rates.append(delta.cpu())
    delta = torch.cat(rates) if rates else torch.zeros(1)
    return dict(
        transitions=len(idx),
        cached_reward_delta_mean=float(delta.mean()),
        cached_reward_delta_min=float(delta.min()),
        cached_reward_delta_max=float(delta.max()),
        method="replace clearance only; retain all measured-torque/other reward terms",
    )
