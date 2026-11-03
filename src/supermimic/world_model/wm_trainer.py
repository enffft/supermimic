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

    def update(self, buffer: ReplayBuffer, *, task_balanced: bool = False) -> dict[str, float]:
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
                w = buffer.sample_windows(micro, self.cfg.window, task_balanced=task_balanced)
                loss, m = self.loss_on(w)
                # Each micro-batch contributes its share of the logical mean loss.
                (loss / n_micro).backward()
                batch_metrics["loss"] = batch_metrics.get("loss", 0.0) + loss.detach() / n_micro
                for k, v in m.items():
                    batch_metrics[k] = batch_metrics.get(k, 0.0) + v.detach() / n_micro
                if w.task_type is not None:
                    batch_metrics["task_l_batch_count"] = batch_metrics.get(
                        "task_l_batch_count", 0.0,
                    ) + (w.task_type == 0).float().sum()
                    batch_metrics["task_m_batch_count"] = batch_metrics.get(
                        "task_m_batch_count", 0.0,
                    ) + (w.task_type == 1).float().sum()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(
                self.wm.parameters(), self.cfg.grad_clip, error_if_nonfinite=True
            )
            self.opt.step()
            batch_metrics["grad_norm"] = gn.detach()
            for k, v in batch_metrics.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        result = {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}
        l_count, m_count = result.get("task_l_batch_count", 0.0), result.get("task_m_batch_count", 0.0)
        total_count = l_count + m_count
        if total_count:
            result["task_l_batch_fraction"] = l_count / total_count
            result["task_m_batch_fraction"] = m_count / total_count
        return result

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)

    @torch.no_grad()
    def online_error_components(self, buffer: ReplayBuffer, n: int = 256, k: int = 8) -> dict[str, float]:
        """Replay-resampled open-loop error, split into physical M1 state groups.

        This is a diagnostic sample from training replay, not a held-out
        validation estimate. Errors are measured after each predicted state is
        fed back into the next World Model step.
        """
        if self.wm.spec.nj != 22:
            raise ValueError("component diagnostics are defined for the M1 22-joint state")
        w = buffer.sample_windows(n, k)
        spec = self.wm.spec
        state = w.states[0]
        per_step: list[dict[str, torch.Tensor]] = []
        for step in range(k):
            state = self.wm.step(state, w.a_exec[step], w.xi)
            truth = w.states[step + 1]
            err = spec.error_vector(state, truth) * spec.err_scale
            q_error = err[..., 12:34]
            qd_error = err[..., 34:56]
            from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS

            leg_ids = torch.tensor(
                [M1_JOINTS.index(name) for name in M1_AMP_JOINTS[:12]],
                device=state.device,
            )
            arm_ids = torch.tensor(
                [M1_JOINTS.index(name) for name in M1_AMP_JOINTS[12:]],
                device=state.device,
            )
            wheel_ids = torch.as_tensor(spec.wheel_indices, device=state.device)
            groups = {
                "nse": err.square().div(spec.err_scale.square()).mean(-1).mean(),
                "root_position_rmse_m": err[..., 0:3].square().sum(-1).mean().sqrt(),
                "root_orientation_rmse_rad": err[..., 3:6].square().sum(-1).mean().sqrt(),
                "root_velocity_rmse_m_s": err[..., 6:9].square().sum(-1).mean().sqrt(),
                "root_angular_velocity_rmse_rad_s": err[..., 9:12].square().sum(-1).mean().sqrt(),
                "leg_position_rmse_rad": q_error.index_select(-1, leg_ids).square().mean().sqrt(),
                "arm_position_rmse_rad": q_error.index_select(-1, arm_ids).square().mean().sqrt(),
                "wheel_position_rmse_rad": q_error.index_select(-1, wheel_ids).square().mean().sqrt(),
                "leg_velocity_rmse_rad_s": qd_error.index_select(-1, leg_ids).square().mean().sqrt(),
                "arm_velocity_rmse_rad_s": qd_error.index_select(-1, arm_ids).square().mean().sqrt(),
                "wheel_velocity_rmse_rad_s": qd_error.index_select(-1, wheel_ids).square().mean().sqrt(),
            }
            per_step.append(groups)
        result: dict[str, float] = {}
        for index in sorted({0, k - 1}):
            prefix = "nse_1step" if index == 0 else f"nse_{k}step"
            for name, value in per_step[index].items():
                if name == "nse":
                    result[prefix] = float(value)
                else:
                    result[f"{prefix}/{name}"] = float(value)
        return result
