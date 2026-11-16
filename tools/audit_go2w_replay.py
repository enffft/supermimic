"""Audit retained replay coverage and reset-safe windows, without training."""

import argparse
import json
from pathlib import Path

import torch


def audit(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    replay = saved["replay"]
    size, capacity = replay["size"], replay["T"]
    order = (torch.arange(size) + replay["ptr"] - size) % capacity
    data = replay["data"]
    domain = data["command_domain"][order]
    level = data["terrain_level"][order]
    done = data["done"][order].bool()
    scope = domain * 8 + level

    def counts(indices, selected=None):
        return torch.bincount(indices.flatten() if selected is None else indices[selected], minlength=48).reshape(6, 8).tolist()

    output = dict(checkpoint=str(checkpoint), iteration=saved["iteration"],
                  retained_time_steps=size, num_envs=replay["N"], retained_transitions=counts(scope),
                  episode_end_transitions=counts(scope, done), windows={})
    for horizon in (8, 16):
        if size < horizon:
            continue
        windows = done.unfold(0, horizon, 1)
        complete = ~windows[..., :-1].any(-1)
        starts = scope[:size - horizon + 1]
        partial = torch.cumprod(torch.cat((torch.ones_like(windows[..., :1]), ~windows[..., :-1]), -1).long(), -1).sum(-1)
        valid_steps = torch.zeros(48, dtype=torch.long)
        valid_steps.scatter_add_(0, starts.flatten(), partial.flatten())
        output["windows"][str(horizon)] = dict(
            possible_starts=counts(starts), full_reset_safe=counts(starts, complete),
            valid_steps_including_partial=valid_steps.reshape(6, 8).tolist(),
            mask_rule="Start always valid; later predictions excluded after the first done. Terminal endpoint can be included, reset crossing cannot.",
        )
    origins = saved["curriculum"]["tensors"]["origins"]
    types = torch.arange(replay["N"]) % origins.shape[1]
    initial_x = origins[level, types[None], 0] + (domain == 0) * 3.4
    displacement = data["s1"][order, :, 0] - initial_x
    output["entry_progress_ge_025m"] = counts(scope, (domain > 0) & (displacement >= .25))
    output["entry_progress_ge_085m"] = counts(scope, (domain > 0) & (displacement >= .85))
    output["interpretation"] = "Retained buffer only. Progress thresholds are interaction-region proxies, not contact truth, climbing success, or independent WM validation. Uniform WM sampling is used in these source snapshots."
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit(args.checkpoint), indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
