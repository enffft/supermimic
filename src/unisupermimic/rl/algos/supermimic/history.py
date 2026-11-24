"""Bounded CPU bank of physical, episode-contiguous WM windows, never Actor starts."""

import torch

from .geometry import matvec, quat_to_mat


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
        self.protection_enabled = False
        self.protected = torch.zeros(capacity, dtype=torch.bool)

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

    def sampling_bucket(self, window):
        bucket = self.bucket(window)
        if self.protection_enabled:
            goal = window["g"][:, 0, :3]
            sign = torch.where(goal.abs() >= 0.025, goal.sign(), 0).long()
            signed_goal = ((sign + 1) * sign.new_tensor([1, 3, 9])).sum(-1)
            bucket = bucket * 27 + signed_goal
            if "collection_source" in window:
                from unisupermimic.tasks.velocity.go2w.motion_prior import command_speed_bins

                bucket = 3 * bucket + command_speed_bins(goal)
        return bucket

    @staticmethod
    def successful(windows):
        if "actor_skill_success" in windows:
            domain = windows["command_domain"][:, 0]
            flat = (domain == 0) & windows["actor_skill_success"].all(-1)
            crossing = (domain >= 4) & windows["traversal_success"][:, -1]
            return (flat | crossing) & ~windows["term"].any(-1)
        state = windows["s1"]
        # Go2W compact state packs world position, wxyz, world linear/angular and joints.
        rotation = quat_to_mat(state[..., 3:7]).transpose(-1, -2)
        lin = matvec(rotation, state[..., 7:10])
        ang = matvec(rotation, state[..., 10:13])
        actual = torch.cat((lin[..., :2], ang[..., 2:3]), -1)
        goal = windows["g"][..., :3]
        rms = (actual - goal).square().mean(1).sqrt()
        moving_correctly = (goal[:, 0].abs() < 0.05) | (
            actual.mean(1) * goal[:, 0] > 0.25 * goal[:, 0].square()
        )
        flat_success = (rms <= rms.new_tensor([0.10, 0.10, 0.18])).all(-1) & moving_correctly.all(
            -1
        )
        # Terminal strict crossing tag belongs to a real step, alongside standing and tracking windows.
        crossed = windows["traversal_success"][:, -1]
        return torch.where(windows["command_domain"][:, 0] >= 4, crossed, flat_success) & ~windows[
            "term"
        ].any(-1)

    def add(self, windows):
        if windows["s"].shape[1] != self.k:
            raise ValueError("History window width differs")
        valid = ~windows["done"][:, :-1].any(-1)
        valid &= (windows["ep_step"][:, 1:] - windows["ep_step"][:, :-1] == 1).all(-1)
        if "target_id" in windows:
            valid &= (windows["target_id"] == windows["target_id"][:, :1]).all(-1)
        if "collection_source" in windows:
            sources = windows["collection_source"]
            valid &= (sources == sources[:, :1]).all(-1)
        cpu = {k: v[valid].detach().cpu() for k, v in windows.items()}
        n = len(cpu["s"])
        if not n:
            return
        if not self.data:
            self.data = {
                k: torch.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype) for k, v in cpu.items()
            }
        buckets = self.sampling_bucket(cpu)
        protected = (
            self.successful(cpu) if self.protection_enabled else torch.zeros(n, dtype=torch.bool)
        )
        # First balance incoming examples; avoid naturally dominant domains erasing rare windows.
        counts = torch.bincount(buckets)
        budget = min(n, 64, self.capacity)
        weights = 1 / counts[buckets].float()
        old_protected = self.protected[: self.size].nonzero().flatten()
        if self.protection_enabled:
            capacity = max(1, self.capacity // 4)
            new_success = protected.nonzero().flatten()
            new_ordinary = (~protected).nonzero().flatten()
            ordinary_capacity = self.capacity - len(old_protected)
            ns = min(len(new_success), capacity, budget)
            no = min(len(new_ordinary), budget - ns, ordinary_capacity)
            success = self._choose(new_success, weights, ns)
            ordinary = self._choose(new_ordinary, weights, no)
            replace = max(0, len(old_protected) + ns - capacity)
            old_counts = torch.bincount(self.buckets[: self.size])
            old_weights = old_counts[self.buckets[: self.size]].float()
            spots = self._choose(old_protected, old_weights, replace)
            self._write(cpu, buckets, protected, success[:replace], spots)
            choice = torch.cat((success[replace:], ordinary))
        else:
            choice = torch.multinomial(weights, budget, replacement=False, generator=self.rng)
        n = len(choice)
        free = min(n, self.capacity - self.size)
        slots = torch.arange(self.size, self.size + free)
        if free < n:
            old_buckets = self.buckets[: self.size]
            removal = torch.bincount(old_buckets)[old_buckets].float()
            allowed = (
                ~self.protected[: self.size]
                if self.protection_enabled
                else torch.ones_like(self.protected[: self.size])
            )
            old = allowed.nonzero().flatten()
            if n - free > len(old):
                choice = choice[: free + len(old)]
                n = len(choice)
            slots = torch.cat((slots, self._choose(old, removal, n - free)))
        self._write(cpu, buckets, protected, choice, slots)
        self.size += free
        self.inserted += n

    def _choose(self, candidates, weights, n):
        if not n:
            return candidates[:0]
        selected = torch.multinomial(weights[candidates], n, replacement=False, generator=self.rng)
        return candidates[selected]

    def _write(self, cpu, buckets, protected, choice, slots):
        for key, value in cpu.items():
            self.data[key][slots] = value[choice]
        self.buckets[slots] = buckets[choice]
        self.protected[slots] = protected[choice]

    def sample(self, n, device, *, protected=False):
        if not self.size:
            raise ValueError("Empty historical bank")
        b = self.buckets[: self.size]
        counts = torch.bincount(b)
        weights = 1 / counts[b].float()
        if self.protection_enabled:
            eligible = self.protected[: self.size] == protected
            if eligible.any():
                weights *= eligible
        ids = torch.multinomial(weights, n, replacement=True, generator=self.rng)
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
            protected=self.protected,
            protection_enabled=self.protection_enabled,
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
        if state.get("protection_enabled", False) != self.protection_enabled:
            raise ValueError("Protected-history sampling configuration differs")
        self.protected = state.get("protected", self.protected).cpu().clone()
        self.rng.set_state(state["rng"].cpu())
