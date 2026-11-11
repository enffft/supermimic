"""Real 2048-env collector audit without optimizer updates; retain source weights/replay."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config
from unisupermimic.tasks.velocity.go2.command_sampler import DOMAINS, MODES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go2_omni_common import SOURCE, configuration, environment


class CommandCoverage:
    def __init__(self, learner):
        self.l = learner
        self.last = learner.sim_steps
        self.count = torch.zeros(6, 6, dtype=torch.long, device=learner.device)
        self.metrics = torch.zeros(6, 6, 5, device=learner.device, dtype=torch.float64)
        self.episodes = torch.zeros_like(self.count)
        self.physical_flat_counts = torch.zeros(2, dtype=torch.long, device=learner.device)
        self.hist = torch.zeros(3, 40, device=learner.device, dtype=torch.long)
        self.signs = torch.zeros(3, 3, device=learner.device, dtype=torch.long)
        self.scatter = []
        state = learner.extra_state.get("omni_coverage")
        if state:
            for k in ("count", "metrics", "episodes", "hist", "signs", "physical_flat_counts"):
                if k in state:
                    getattr(self, k).copy_(state[k])

    @torch.no_grad()
    def observe(self):
        l = self.l
        b = l.buffer
        n = (l.sim_steps - self.last) // b.N
        if n > b.T:
            raise RuntimeError("Coverage callback missed replay rows")
        rows = (torch.arange(n, device=b.device) + b.ptr - n) % b.T
        if not n:
            return
        d = {k: v[rows].flatten(0, 1) for k, v in b.data.items()}
        idx = d["command_domain"] * 6 + d["command_mode"]
        self.count.view(-1).add_(torch.bincount(idx, minlength=36))
        self.physical_flat_counts.add_(
            torch.bincount((d["terrain_level"] != 0).long(), minlength=2)
        )
        self.episodes.view(-1).scatter_add_(0, idx, (d["ep_step"] == 0).long())
        p, R, v, w, *_ = unflatten_state(d["s1"])
        lv = (R.transpose(-1, -2) @ v[..., None]).squeeze(-1)
        lw = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
        vals = torch.stack(
            (
                (lv[:, :2] - d["g"][:, :2]).square().sum(-1),
                (lw[:, 2] - d["g"][:, 2]).square(),
                d["r"],
                d["term"].float(),
                d["done"].float(),
            ),
            -1,
        )
        # Separate true hard termination from the flat-domain boundary timeout.
        self.metrics.view(-1, 5).index_add_(0, idx, vals.double())
        for axis in range(3):
            bucket = ((d["g"][:, axis] + 1) * 20).long().clamp(0, 39)
            self.hist[axis].add_(torch.bincount(bucket, minlength=40))
            self.signs[axis].add_(
                torch.bincount(torch.sign(d["g"][:, axis]).long() + 1, minlength=3)
            )
        if len(self.scatter) < 100:
            self.scatter.append(
                torch.cat((d["command_domain"][:2048, None], d["g"][:2048]), -1).cpu()
            )
        self.last = l.sim_steps
        l.extra_state["omni_coverage"] = {
            k: getattr(self, k)
            for k in ("count", "metrics", "episodes", "hist", "signs", "physical_flat_counts")
        }

    def summary(self):
        count = self.count.double().clamp_min(1)
        return dict(
            domains=DOMAINS,
            modes=MODES,
            transitions=self.count.cpu().tolist(),
            episode_starts=self.episodes.cpu().tolist(),
            domain_fractions=(self.count.sum(-1) / self.count.sum().clamp_min(1)).cpu().tolist(),
            xy_rmse_m_s=(self.metrics[..., 0] / count).sqrt().cpu().tolist(),
            yaw_rmse_rad_s=(self.metrics[..., 1] / count).sqrt().cpu().tolist(),
            mean_reward=(self.metrics[..., 2] / count).cpu().tolist(),
            hard_falls=self.metrics[..., 3].cpu().tolist(),
            done=self.metrics[..., 4].cpu().tolist(),
            histogram_edges=[-1 + i * 0.05 for i in range(41)],
            goal_histograms=self.hist.cpu().tolist(),
            negative_zero_positive=self.signs.cpu().tolist(),
            wm_sample_modes=self.l.buffer.sampling_counts["wm_modes"].cpu().tolist(),
            policy_start_modes=self.l.extra_state.get("actual_policy_start_modes", torch.zeros(6))
            .cpu()
            .tolist(),
            actual_actor_start_modes=self.l.extra_state.get(
                "actual_actor_start_modes", torch.zeros(6)
            )
            .cpu()
            .tolist(),
            physical_flat_nonflat_counts_since_instrumentation=self.physical_flat_counts.cpu().tolist(),
            distribution_note="Configured domains; terrain level0 is physically flat, measured separately below",
        )

    def save(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "coverage.json").write_text(json.dumps(self.summary(), indent=2))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        for i, ax in enumerate(axes):
            ax.bar([-0.975 + j * 0.05 for j in range(40)], self.hist[i].cpu(), width=0.05)
            ax.set_title(("vx (m/s)", "vy (m/s)", "wz (rad/s)")[i])
        fig.tight_layout()
        fig.savefig(out / "velocity_histograms.png")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(MODES, self.count.sum(0).cpu())
        axes[1].bar(DOMAINS, self.count.sum(1).cpu())
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / "mode_domain_counts.png")
        plt.close(fig)
        if self.scatter:
            pts = torch.cat(self.scatter)
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for d, ax in enumerate(axes.flat):
                x = pts[pts[:, 0] == d]
                ax.scatter(x[:, 1], x[:, 2], s=2, c=x[:, 3], vmin=-0.9, vmax=0.9)
                ax.set_title(DOMAINS[d])
                ax.set_xlabel("vx m/s")
                ax.set_ylabel("vy m/s")
            fig.tight_layout()
            fig.savefig(out / "domain_command_scatter.png")
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    a = p.parse_args()
    if a.steps < 1000:
        raise ValueError("Audit requires at least two full episode budgets")
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = configuration()
    OmegaConf.save(cfg, a.output / "config.yaml")
    started = time.monotonic()
    env = environment(cfg)
    env.init_state()
    try:
        l = SuperMimicLearner(
            env, build_learner_config(cfg), a.output, device="cuda", backend="mujoco"
        )
        l.load(a.checkpoint, allow_command_distribution_change=True)
        source_counts = l.update_counts.copy()
        before = [p.detach().clone() for p in l.pi.parameters()]
        coverage = CommandCoverage(l)
        for i in range(a.steps):
            l.collect(1)
            coverage.observe()
            if (i + 1) % 250 == 0:
                print(
                    json.dumps(dict(vector_steps=i + 1, wall_s=time.monotonic() - started)),
                    flush=True,
                )
        assert all(torch.equal(a, b) for a, b in zip(before, l.pi.parameters(), strict=True))
        assert source_counts == l.update_counts
        coverage.save(a.output)
        l.save(a.output / "last.pt")
        (a.output / "summary.json").write_text(
            json.dumps(
                dict(
                    vector_steps=a.steps,
                    new_real_transitions=a.steps * 2048,
                    wall_s=time.monotonic() - started,
                    optimizer_updates=0,
                    actor_parameters_unchanged=True,
                    checkpoint=str(a.checkpoint),
                    source_sha256=hashlib.file_digest(
                        a.checkpoint.open("rb"), "sha256"
                    ).hexdigest(),
                ),
                indent=2,
            )
        )
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
