"""Bounded CPU bank of physical, episode-contiguous WM windows, never Actor starts."""

import torch


class HistoricalWindowBank:
    def __init__(self, capacity, k, seed=2026, protocol=""):
        if capacity < 1 or k < 1:
            raise ValueError("Positive history capacity/window required")
        self.capacity, self.k, self.protocol = capacity, k, protocol
        self.rng = torch.Generator().manual_seed(seed)
        self.data = {}
        self.size = 0
        self.inserted = 0
        self.buckets = torch.zeros(capacity, dtype=torch.long)

    @staticmethod
    def bucket(window):
        domain = window["command_domain"][:, 0].long()
        level = window["terrain_level"][:, 0].long()
        mode = window["command_mode"][:, 0].long()
        goal = window["g"][:, 0, :3]
        dominant = goal.gather(1, goal.abs().argmax(-1)[:, None])[:, 0]
        sign = (dominant < 0).long()
        phase = torch.zeros_like(domain)
        if "traversal" in window:
            meta = window["traversal"][:, 0]
            along = ((window["s"][:, 0, :2] - meta[:, 3:5]) * meta[:, 5:7]).sum(-1)
            phase = (along > -0.2).long() + (along > 0.3).long()
        return (((domain * 10 + level) * 6 + mode) * 2 + sign) * 3 + phase

    def add(self, windows):
        if windows["s"].shape[1] != self.k:
            raise ValueError("History window width differs")
        valid = ~windows["done"][:, :-1].any(-1)
        valid &= (windows["ep_step"][:, 1:] - windows["ep_step"][:, :-1] == 1).all(-1)
        if "target_id" in windows:
            valid &= (windows["target_id"] == windows["target_id"][:, :1]).all(-1)
        cpu = {k: v[valid].detach().cpu() for k, v in windows.items()}
        n = len(cpu["s"])
        if not n:
            return
        if not self.data:
            self.data = {
                k: torch.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype) for k, v in cpu.items()
            }
        buckets = self.bucket(cpu)
        # First balance incoming examples; avoid naturally dominant domains erasing rare windows.
        counts = torch.bincount(buckets)
        n = min(n, 64, self.capacity)
        choice = torch.multinomial(
            1 / counts[buckets].float(), n, replacement=False, generator=self.rng
        )
        free = min(n, self.capacity - self.size)
        slots = torch.arange(self.size, self.size + free)
        if free < n:
            counts = torch.bincount(self.buckets[: self.size])
            removal = counts[self.buckets[: self.size]].float()
            slots = torch.cat(
                (slots, torch.multinomial(removal, n - free, replacement=False, generator=self.rng))
            )
        for key, value in cpu.items():
            self.data[key][slots] = value[choice]
        self.buckets[slots] = buckets[choice]
        self.size += free
        self.inserted += n

    def sample(self, n, device):
        if not self.size:
            raise ValueError("Empty historical bank")
        b = self.buckets[: self.size]
        counts = torch.bincount(b)
        ids = torch.multinomial(1 / counts[b].float(), n, replacement=True, generator=self.rng)
        return {k: v[ids].to(device, non_blocking=False) for k, v in self.data.items()}

    def state_dict(self):
        return dict(
            capacity=self.capacity,
            k=self.k,
            protocol=self.protocol,
            size=self.size,
            inserted=self.inserted,
            data=self.data,
            buckets=self.buckets,
            rng=self.rng.get_state(),
        )

    def load_state_dict(self, state):
        if (state["capacity"], state["k"], state["protocol"]) != (
            self.capacity,
            self.k,
            self.protocol,
        ):
            raise ValueError("Historical bank protocol/capacity differs")
        self.size = int(state["size"])
        self.inserted = int(state["inserted"])
        if not 0 <= self.size <= self.capacity:
            raise ValueError("Invalid historical size")
        self.data = {k: v.cpu().clone() for k, v in state["data"].items()}
        self.buckets = state["buckets"].cpu().clone()
        self.rng.set_state(state["rng"].cpu())
