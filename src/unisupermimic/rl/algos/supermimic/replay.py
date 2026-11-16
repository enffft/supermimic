"""Fixed-size device-resident replay laid out as ``(time, environment)``."""

from __future__ import annotations

import torch

from unisupermimic.rl.contracts.layout import StateLayout


def _make_fields(
    num_joints: int,
    context_dim: int,
    state_layout: StateLayout | None = None,
) -> dict[str, tuple[int | tuple[()], torch.dtype]]:
    layout = state_layout or StateLayout.legacy(num_joints)
    state_dim = layout.state_dim
    return {
        "s": (state_dim, torch.float32),
        "s1": (state_dim, torch.float32),
        "a": (layout.action_dim, torch.float32),
        "a_prev": (layout.action_dim, torch.float32),
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
        state_layout: StateLayout | None = None,
        level_count: int = 8,
        domain_count: int = 6,
        extra_fields=None,
        history_capacity: int = 0,
        history_fraction: float = 0.5,
        history_window: int = 8,
    ):
        self.T = T
        self.N = N
        self.device = torch.device(device)
        self.level_count, self.domain_count = level_count, domain_count
        if not 0 <= history_fraction <= 1:
            raise ValueError("History fraction must lie in [0,1]")
        self.history_fraction = history_fraction
        self.history = None
        self.history_actual = [0, 0]
        if history_capacity:
            from .history import HistoricalWindowBank

            self.history = HistoricalWindowBank(history_capacity, history_window)
        self.fields = _make_fields(num_joints, context_dim, state_layout)
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
                command_speed_level=((), torch.long),
            )
        if extra_fields:
            self.fields.update(extra_fields)
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
        # Opt-in WM-only tail sampling; Actor admission remains independent.
        self.wm_rare_fraction = 0.0
        self.wm_rare_min_level = 5
        self.wm_rare_balance_levels = False
        self.wm_level_counts = torch.zeros(self.level_count, device=self.device, dtype=torch.long)

    def mark_reset_boundary(self, reset_mask: torch.Tensor) -> None:
        """Close prior transitions after an explicit reset outside env.step().

        Auto-resets already carry done. This opt-in also handles data-collection
        resets and tile reassignment without changing legacy flat/G1 paths.
        """
        if self.size:
            self.data["done"][(self.ptr - 1) % self.T].logical_or_(reset_mask)

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for key in self.fields:
            if key == "command_speed_level" and key not in transition:
                self.data[key][self.ptr].fill_(-1)  # Explicit unknown historical label.
            else:
                self.data[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.history is not None and self.size >= self.history.k and self.ptr % 16 == 0:
            self.history.add(self._sample_recent_windows(256, self.history.k, record=False))

    def state_dict(self):
        return {
            "history": None if self.history is None else self.history.state_dict(),
            "history_actual": self.history_actual,
            "T": self.T,
            "N": self.N,
            "ptr": self.ptr,
            "size": self.size,
            "data": self.data,
            "sampling_counts": self.sampling_counts,
            "start_scope_mask": self.start_scope_mask,
            "wm_rare_fraction": self.wm_rare_fraction,
            "wm_rare_min_level": self.wm_rare_min_level,
            "wm_rare_balance_levels": self.wm_rare_balance_levels,
            "wm_level_counts": self.wm_level_counts,
        }

    def load_state_dict(self, values, *, infer_command_metadata=False):
        if self.history is not None:
            if values.get("history") is None:
                raise ValueError("Missing historical bank")
            self.history.load_state_dict(values["history"])
            self.history_actual = list(values.get("history_actual", [0, 0]))
        values = dict(values)
        values["data"] = dict(values["data"])
        if "command_speed_level" in self.fields and "command_speed_level" not in values["data"]:
            values["data"]["command_speed_level"] = torch.full_like(values["data"]["f"], -1)
        self.wm_rare_fraction = float(values.get("wm_rare_fraction", 0.0))
        self.wm_rare_min_level = int(values.get("wm_rare_min_level", 5))
        self.wm_rare_balance_levels = bool(values.get("wm_rare_balance_levels", False))
        self.wm_level_counts.copy_(
            values.get("wm_level_counts", torch.zeros_like(self.wm_level_counts))
        )
        if not 0 <= self.wm_rare_fraction <= 1:
            raise ValueError("Invalid WM rare sampling fraction")
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
            if (
                scope.ndim != 2
                or scope.shape[0] != self.domain_count
                or not 5 <= scope.shape[1] <= self.level_count
                or scope.dtype != torch.bool
            ):
                raise ValueError("Invalid imagination domain/terrain-level admission")
            self.start_scope_mask = scope.to(self.device).clone()
        for name, count in values.get("sampling_counts", {}).items():
            self.sampling_counts[name].copy_(count)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time indices (zero is oldest) to storage rows."""
        return (self.ptr - self.size + age) % self.T

    def _sample_recent_windows(self, B: int, k: int, *, record=True) -> dict[str, torch.Tensor]:
        """Sample ``k``-step windows and mask transitions after episode ends."""
        if not 1 <= k <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {k}")
        start = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        rare_n = int(B * self.wm_rare_fraction)
        if rare_n:
            if "terrain_level" not in self.fields:
                raise ValueError("WM tail sampling requires terrain metadata")
            pool = max(4096, B * 32)
            rare_age = torch.randint(0, self.size - k + 1, (pool,), device=self.device)
            rare_env = torch.randint(0, self.N, (pool,), device=self.device)
            candidate_rows = self._row(rare_age)
            levels = self.data["terrain_level"][candidate_rows, rare_env]
            rare_mask = levels >= self.wm_rare_min_level
            if "command_domain" in self.fields:
                rare_mask &= self.data["command_domain"][candidate_rows, rare_env] >= 4
            rare_age, rare_env = rare_age[rare_mask], rare_env[rare_mask]
            if len(rare_age):
                if self.wm_rare_balance_levels:
                    scopes = levels[rare_mask]
                    counts = torch.bincount(scopes, minlength=self.level_count)
                    weights = (counts > 0).float()
                    targets = torch.multinomial(weights, rare_n, replacement=True)
                    offsets = counts.cumsum(0) - counts
                    pick = torch.argsort(scopes)[
                        offsets[targets]
                        + (torch.rand(rare_n, device=self.device) * counts[targets]).long()
                    ]
                else:
                    pick = torch.randint(len(rare_age), (rare_n,), device=self.device)
                start[:rare_n], env[:rare_n] = rare_age[pick], rare_env[pick]
        ages = start.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        envs = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in self.fields}
        for key in ("g", "command_mode", "command_domain"):
            if key in self.fields:
                out[key] = self.data[key][rows, envs]
        if record and "terrain_level" in self.fields:
            self.wm_level_counts.add_(
                torch.bincount(
                    self.data["terrain_level"][rows[:, 0], env], minlength=self.level_count
                )
            )
        done = out["done"].float()
        mask = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - done[:, :-1]), -1),
            -1,
        )
        out["mask"] = mask
        if record and "command_mode" in out:
            self.sampling_counts["wm_modes"].scatter_add_(
                0, out["command_mode"].flatten(), mask.long().flatten()
            )
        return out

    def sample_windows(self, B: int, k: int):
        if self.history is None:
            return self._sample_recent_windows(B, k)
        # Reject full windows crossing reset rather than merely masking their suffix.
        pool = self._sample_recent_windows(max(B * 4, 1024), k, record=False)
        valid = (pool["mask"] == 1).all(-1)
        valid &= (pool["ep_step"][:, 1:] - pool["ep_step"][:, :-1] == 1).all(-1)
        if "target_id" in pool:
            valid &= (pool["target_id"] == pool["target_id"][:, :1]).all(-1)
        ids = valid.nonzero().flatten()
        if not len(ids):
            raise ValueError("No continuous WM windows available")
        historical = int(B * self.history_fraction) if self.history.size else 0
        recent = B - historical
        buckets = self.history.bucket({key: v[ids] for key, v in pool.items()})
        counts = torch.bincount(buckets)
        selected = ids[torch.multinomial(1 / counts[buckets].float(), recent, replacement=True)]
        result = {key: value[selected] for key, value in pool.items()}
        if historical:
            old = self.history.sample(historical, self.device)
            result = {key: torch.cat((value, old[key]), 0) for key, value in result.items()}
        self.wm_level_counts.add_(
            torch.bincount(result["terrain_level"][:, 0], minlength=self.level_count)
        )
        self.sampling_counts["wm_modes"].add_(
            torch.bincount(result["command_mode"].flatten(), minlength=6)
        )
        self.history_actual[0] += recent
        self.history_actual[1] += historical
        self.last_history_split = recent
        return result

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
        for key in (
            "command_mode",
            "command_domain",
            "terrain_level",
            "command_speed_level",
            "traversal",
            "target_id",
            "success_peak",
        ):
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
