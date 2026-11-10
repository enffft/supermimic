"""Sample collision geometry at real replay states using public MuJoCo APIs.

These are reconstructed contact samples, not a recording of live contact forces.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat
from unisupermimic.tasks.velocity.go2.contract import JOINTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_go2_migration import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    values = torch.load(a.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    replay = values["replay"]["data"]
    model = mujoco.MjModel.from_binary_path(str(a.model))
    d = mujoco.MjData(model)
    qi = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]
    vi = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "go2_terrain")
    origins = values["curriculum"]["tensors"]["origins"]
    records = []
    rng = np.random.default_rng(4326)
    for kind in range(5):
        for level in range(4):
            candidates = (
                (replay["terrain_type"] == kind) & (replay["terrain_level"] == level)
            ).nonzero()
            indices = candidates[torch.tensor(rng.choice(len(candidates), 128, replace=False))]
            states = replay["s"][indices[:, 0], indices[:, 1]]
            rotations = quat_to_mat(states[:, 3:7])
            angular = (rotations.transpose(-1, -2) @ states[:, 10:13, None]).squeeze(-1)
            contacts = interior = nonfoot = 0
            for i, s in enumerate(states):
                d.qpos[:7] = s[:7].numpy()
                d.qpos[qi] = s[13:25].numpy()
                d.qvel[:3] = s[7:10].numpy()
                d.qvel[3:6] = angular[i].numpy()
                d.qvel[vi] = s[25:37].numpy()
                mujoco.mj_forward(model, d)
                ground = [c for c in d.contact if terrain in (c.geom1, c.geom2)]
                contacts += bool(ground)
                for c in ground:
                    other = c.geom2 if c.geom1 == terrain else c.geom1
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                    if "foot_collision" not in name:
                        nonfoot += 1
                # .6m beyond the entry crosses the 1.2m flat border into the tile.
                o = origins[level, kind, :2].numpy()
                interior += level > 0 and any(
                    0.6 < c.pos[0] - o[0] < 6.2 and abs(c.pos[1] - o[1]) < 3.8 for c in ground
                )
            records.append(
                dict(
                    type=kind,
                    level=level,
                    source_transitions=len(candidates),
                    sampled_states=128,
                    ground_contact_states=contacts,
                    interior_ground_contact_states=interior,
                    nonfoot_contact_pairs=nonfoot,
                )
            )
    dump(
        a.output / "collision_coverage.json",
        dict(
            rows=records,
            method="Uniform within each real replay type/level; mj_forward at stored qpos/qvel in public saved model",
            limitation="Sampled reconstructed collision events; not full live force/contact-switch telemetry",
        ),
    )


if __name__ == "__main__":
    main()
