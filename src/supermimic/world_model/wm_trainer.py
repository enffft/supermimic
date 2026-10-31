"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import ReferenceFrame
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    # Optional GPU micro-batch. When set below batch_size, several micro-batches
    # are accumulated into one optimizer step, preserving the logical batch size
    # without allocating the full window batch at once.
    micro_batch_size: int | None = None
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary | None, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        if self.lib is None:
            # Reference-free tasks use base-height/upright termination features.
            refs = [None] * k
        else:
            # Only the root pose of references is required for the termination features.
            r = self.lib.ref(w.frame.reshape(-1))
            refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        micro = self.cfg.micro_batch_size or self.cfg.batch_size
        if micro <= 0 or self.cfg.batch_size <= 0 or self.cfg.batch_size % micro:
            raise ValueError(
                "wm batch_size must be a positive multiple of micro_batch_size; "
                f"got batch_size={self.cfg.batch_size}, micro_batch_size={micro}"
            )
        n_micro = self.cfg.batch_size // micro
        for _ in range(self.cfg.batches_per_update):
            self.opt.zero_grad(set_to_none=True)
            batch_metrics: dict[str, torch.Tensor] = {}
            for _ in range(n_micro):
                w = buffer.sample_windows(micro, self.cfg.window)
                loss, m = self.loss_on(w)
                # Each micro-batch contributes its share of the logical mean loss.
                (loss / n_micro).backward()
                batch_metrics["loss"] = batch_metrics.get("loss", 0.0) + loss.detach() / n_micro
                for k, v in m.items():
                    batch_metrics[k] = batch_metrics.get(k, 0.0) + v.detach() / n_micro
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(
                self.wm.parameters(), self.cfg.grad_clip, error_if_nonfinite=True
            )
            self.opt.step()
            batch_metrics["grad_norm"] = gn.detach()
            for k, v in batch_metrics.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
