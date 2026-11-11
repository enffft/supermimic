"""Fixed-size device-resident replay laid out as ``(time, environment)``."""

from __future__ import annotations

import torch


def _make_fields(
    num_joints: int,
    context_dim: int,
) -> dict[str, tuple[int | tuple[()], torch.dtype]]:
    state_dim = 13 + 2 * num_joints
    return {
        "s": (state_dim, torch.float32),
        "s1": (state_dim, torch.float32),
        "a": (num_joints, torch.float32),
        "a_prev": (num_joints, torch.float32),
        "xi": (context_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
    }


FIELDS = _make_fields(29, 33)


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str | torch.device,
        num_joints: int = 29,
        context_dim: int = 33,
        goal_dim: int = 0,
        terrain_metadata: bool = False,
        command_metadata: bool = False,
    ):
        self.T = T
        self.N = N
        self.device = torch.device(device)
        self.fields = _make_fields(num_joints, context_dim)
        if goal_dim:
            self.fields["g"] = (goal_dim, torch.float32)
        if terrain_metadata:
            self.fields.update(
                terrain_type=((), torch.long),
                terrain_level=((), torch.long),
                terrain_probe=((), torch.bool),
            )
        if command_metadata:
            self.fields.update(
                command_domain=((), torch.long),
                command_mode=((), torch.long),
                crossing_episode=((), torch.bool),
                term=((), torch.bool),
            )
        self.sampling_counts = {
            k: torch.zeros(6, dtype=torch.long, device=self.device)
            for k in ("wm_modes", "actor_modes")
        }
        self.data = {}
        for key, (shape, dtype) in self.fields.items():
            field_shape = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[key] = torch.zeros(field_shape, dtype=dtype, device=self.device)
        self.ptr = 0
        self.size = 0
        # Opt-in collector metadata admission, never an observation feature.
        self.start_scope_mask = None

    def mark_reset_boundary(self, reset_mask: torch.Tensor) -> None:
        """Close prior transitions after an explicit reset outside env.step().

        Auto-resets already carry done. This opt-in also handles data-collection
        resets and tile reassignment without changing legacy flat/G1 paths.
        """
        if self.size:
            self.data["done"][(self.ptr - 1) % self.T].logical_or_(reset_mask)

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for key in self.fields:
            self.data[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def state_dict(self):
        return {
            "T": self.T,
            "N": self.N,
            "ptr": self.ptr,
            "size": self.size,
            "data": self.data,
            "sampling_counts": self.sampling_counts,
            "start_scope_mask": self.start_scope_mask,
        }

    def load_state_dict(self, values, *, infer_command_metadata=False):
        if infer_command_metadata:
            from unisupermimic.tasks.velocity.go2.command_sampler import command_modes

            old = values["data"]
            if "command_mode" not in old:
                old["command_mode"] = command_modes(old["g"])
                old["command_domain"] = torch.where(
                    old["terrain_level"] == 0, 0, old["terrain_type"] + 1
                )
                old["term"] = old["done"] & (old["ep_step"] < 499)
                old["crossing_episode"] = (old["command_mode"] == 1) & (old["command_domain"] != 0)

        if (values["T"], values["N"]) != (self.T, self.N) or values[
            "data"
        ].keys() != self.data.keys():
            raise ValueError("Replay layout mismatch; use explicit fresh-data resume")
        for name, tensor in values["data"].items():
            if tensor.shape != self.data[name].shape or tensor.dtype != self.data[name].dtype:
                raise ValueError(f"Replay field mismatch: {name}")
            self.data[name].copy_(tensor)
        self.ptr, self.size = values["ptr"], values["size"]
        scope = values.get("start_scope_mask")
        if scope is not None:
            if scope.shape != (6, 5) or scope.dtype != torch.bool:
                raise ValueError("Invalid imagination domain/terrain-level admission")
            self.start_scope_mask = scope.to(self.device).clone()
        for name, count in values.get("sampling_counts", {}).items():
            self.sampling_counts[name].copy_(count)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time indices (zero is oldest) to storage rows."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int) -> dict[str, torch.Tensor]:
        """Sample ``k``-step windows and mask transitions after episode ends."""
        if not 1 <= k <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {k}")
        start = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = start.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        envs = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in ("s", "s1", "a", "xi", "done")}
        for key in ("g", "command_mode", "command_domain"):
            if key in self.fields:
                out[key] = self.data[key][rows, envs]
        done = out["done"].float()
        mask = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - done[:, :-1]), -1),
            -1,
        )
        out["mask"] = mask
        if "command_mode" in out:
            self.sampling_counts["wm_modes"].scatter_add_(
                0, out["command_mode"].flatten(), mask.long().flatten()
            )
        return out

    def sample_starts(
        self,
        B: int,
        hist: int,
        command_bucket_weights=None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample imagination starts and oldest-first, boundary-padded histories."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if hist < 1:
            raise ValueError(f"history length must be positive, got {hist}")
        pool = (
            B
            if command_bucket_weights is None and self.start_scope_mask is None
            else max(8192, B * 32)
        )
        age = torch.randint(0, self.size, (pool,), device=self.device)
        env = torch.randint(0, self.N, (pool,), device=self.device)
        if self.start_scope_mask is not None:
            if not {"command_domain", "terrain_level"} <= self.fields.keys():
                raise ValueError("Scope admission requires command and terrain metadata")
            rows = self._row(age)
            valid = self.start_scope_mask[
                self.data["command_domain"][rows, env], self.data["terrain_level"][rows, env]
            ]
            age, env = age[valid], env[valid]
            if not len(age):
                raise ValueError("No admitted imagination starts in sampled replay pool")
            if command_bucket_weights is None:
                selected = torch.randint(len(age), (B,), device=self.device)
                age, env = age[selected], env[selected]
        if command_bucket_weights is not None:
            if "command_mode" not in self.fields:
                raise ValueError("Stratified starts require explicit command metadata")
            weights = torch.as_tensor(command_bucket_weights, device=self.device)
            if (
                weights.shape != (6,)
                or not torch.isfinite(weights).all()
                or (weights < 0).any()
                or weights.sum() <= 0
            ):
                raise ValueError("Expected six finite nonnegative start weights")
            modes = self.data["command_mode"][self._row(age), env]
            counts = torch.bincount(modes, minlength=6)
            weights = weights * (counts > 0)
            target = torch.multinomial(weights, B, replacement=True)
            offsets = counts.cumsum(0) - counts
            chosen = torch.argsort(modes)[
                offsets[target] + (torch.rand(B, device=self.device) * counts[target]).long()
            ]
            age, env = age[chosen], env[chosen]
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        history_rows = self._row(age.unsqueeze(-1) - back)
        history_envs = env.unsqueeze(-1).expand_as(history_rows)
        start = {key: self.data[key][row, env] for key in ("s", "a_prev", "xi", "f")}
        if "g" in self.fields:
            start["g"] = self.data["g"][row, env]
        for key in ("command_mode", "command_domain", "terrain_level"):
            if key in self.fields:
                start[key] = self.data[key][row, env]
        if "command_mode" in start:
            self.sampling_counts["actor_modes"].add_(
                torch.bincount(start["command_mode"], minlength=6)
            )
        history = {
            key: self.data[key][history_rows, history_envs] for key in ("s", "a_prev", "xi", "f")
        }
        return start, history


__all__ = ["FIELDS", "ReplayBuffer"]
