"""Independent fixed physical holdout, H1/H8/H16 errors over domains/levels/phases."""

import argparse
import hashlib
from pathlib import Path

import torch
from audit_go2w_warmstart import prediction_rmse

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import WorldModel
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import LAYOUT
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.experiment import WM_PATH, write_json
from unisupermimic.tasks.velocity.go2w.world_model import tensor_digest


def holdout(path):
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    cases = [
        dict(domain=d, level=l, offset=o)
        for d in range(8)
        for l in range(10)
        for o in (0.0, 0.6, 2.8)
    ]
    env = make_go2w_env(load_config("train"), len(cases))
    try:
        c = env.command_manager.get_term("velocity")
        cur = c.curriculum
        cur.enabled = False
        cur.coverage_level.fill_(-1)
        c.command_domain[:] = torch.tensor([x["domain"] for x in cases])
        cur.terrain_type[:] = (c.command_domain - 1).clamp_min(0)
        cur.terrain_level[:] = torch.tensor([x["level"] for x in cases])
        env.reset()
        views = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        qpos = views["qpos"].clone()
        root = qpos[:, c.root_q].clone()
        root[:, 0] += torch.tensor([x["offset"] for x in cases])
        root[:, 2] = c.mdp.terrain.ground_height(root[:, :2]) + c.cfg.spawn_height
        qpos[:, c.root_q] = root
        env.backend.set_state_tensor(
            torch.arange(len(cases)), qpos.contiguous(), views["qvel"].clone().contiguous()
        )
        initial = flatten_state(c.state()).clone()
        g = torch.Generator().manual_seed(73129)
        actions = torch.randn(16, len(cases), 16, generator=g) * 0.02
        actions[:, :, 12:] *= 5
        truth = []
        valid = []
        alive = torch.ones(len(cases), dtype=torch.bool)
        for step in range(16):
            result = env.step(actions[step])
            truth.append(flatten_state(env.velocity_next_state).clone())
            alive &= ~(result.terminated | result.truncated)
            valid.append(alive.clone())
        data = dict(
            cases=cases,
            initial=initial,
            actions=actions,
            truth=torch.stack(truth),
            valid=torch.stack(valid),
            contract=env.supermimic_contract,
            terrain_fingerprint=c.mdp.terrain.fingerprint,
            seed=73129,
            scope="independent physical holdout; never training replay; approach/contact/platform spawn probes",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        return data
    finally:
        env.close()
        env.backend.close()


def evaluate(model_path, holdout_path, output):
    torch.set_num_threads(1)
    data = holdout(holdout_path)
    env = make_go2w_env(load_config("train"), 1)
    try:
        c = env.command_manager.get_term("velocity")
        if c.mdp.terrain.fingerprint != data["terrain_fingerprint"]:
            raise ValueError("Holdout terrain differs")
        asset = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
        wm = WorldModel(
            hidden=(2048, 2048, 2048),
            num_joints=16,
            context_dim=0,
            structured=True,
            terrain_dim=77,
            state_layout=LAYOUT,
        )
        wm.load_state_dict(asset["wm"], strict=True)
        wm.eval()
        digest = tensor_digest(wm.state_dict())
        del asset
        s = unflatten_state(data["initial"], LAYOUT)
        rows = []
        with torch.no_grad():
            for step in range(16):
                s = wm.step(
                    s,
                    data["actions"][step],
                    torch.empty(len(data["cases"]), 0),
                    c.mdp.terrain.local_scan(s[0], s[1], sensor_grad=True),
                )
                if step + 1 not in (1, 8, 16):
                    continue
                actual = unflatten_state(data["truth"][step], LAYOUT)
                for domain in range(8):
                    for phase in (0.0, 0.6, 2.8):
                        selected = (
                            torch.tensor(
                                [
                                    x["domain"] == domain and x["offset"] == phase
                                    for x in data["cases"]
                                ]
                            )
                            & data["valid"][step]
                        )
                        row = dict(
                            horizon=step + 1,
                            domain=domain,
                            phase_offset=phase,
                            valid=int(selected.sum()),
                            total=10,
                        )
                        if selected.any():
                            row["rmse"] = prediction_rmse(
                                tuple(t[selected] for t in s), tuple(t[selected] for t in actual)
                            )
                        rows.append(row)
        report = dict(
            model=str(model_path),
            wm_tensor_sha256=digest,
            holdout=str(holdout_path),
            holdout_file_sha256=hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
            independent=True,
            rows=rows,
            censored="episodes terminated before requested horizon are excluded and counted",
        )
        write_json(output, report)
        print("holdout saved", output)
        return report
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=WM_PATH)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.holdout, args.output)
