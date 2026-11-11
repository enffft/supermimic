"""Command-only, bounded stairs continuation using the existing UniLab learner.

The historical stageC flag is untouched. S0 admits a small-LR controlled trial;
higher speed promotion needs independent physics and >=80% per-probe qd signs.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from omegaconf import OmegaConf

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.scripts.train_supermimic import build_learner_config
from unisupermimic.tasks.velocity.go2.speed_curriculum import (
    imagination_scope_admission,
    stairs_speed_admission,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_go2_command_coverage import CommandCoverage
from go2_omni_common import ROOT, configuration, environment
from train_go2_omnidirectional import physical_regressions
from validate_go2_migration import ResourceMonitor, dump


class StairsCoverage:
    def __init__(self, l):
        self.l = l
        self.last = l.sim_steps
        self.nlevels = l.env.command_manager.get_term("velocity").sampler.stairs_levels.shape[1]
        self.count = torch.zeros(2, self.nlevels, 4, device=l.device, dtype=torch.long)
        self.triples = torch.zeros_like(self.count)
        self.crossing = torch.zeros_like(self.count)
        self.falls = torch.zeros_like(self.count)
        self.near = torch.zeros(8, device=l.device, dtype=torch.long)
        self.maxima = torch.zeros(3, device=l.device)
        self.steps = 0
        self.nonflat = 0
        if "stairs_command_coverage" in l.extra_state:
            st = l.extra_state["stairs_command_coverage"]
            for k in ("count", "triples", "crossing", "falls", "near", "maxima"):
                getattr(self, k).copy_(st[k])
            self.steps, self.nonflat = st["steps"], st["nonflat"]

    @torch.no_grad()
    def observe(self):
        l, b = self.l, self.l.buffer
        n = (l.sim_steps - self.last) // b.N
        if not n:
            return
        if n > b.T:
            raise ValueError("Missed replay rows")
        rows = (torch.arange(n, device=b.device) + b.ptr - n) % b.T
        d = {k: v[rows].flatten(0, 1) for k, v in b.data.items()}
        mask = d["command_domain"] >= 4
        direction = d["command_domain"][mask] - 4
        terrain = d["terrain_level"][mask]
        levels = l.env.command_manager.get_term("velocity").sampler.stairs_levels.to(b.device)[
            direction, terrain
        ]
        idx = (direction * self.nlevels + terrain) * 4 + levels
        for target, value in (
            (self.count, torch.ones_like(idx)),
            (self.triples, (d["g"][mask] != 0).all(-1).long()),
            (self.crossing, d["crossing_episode"][mask].long()),
            (self.falls, d["term"][mask].long()),
        ):
            target.view(-1).scatter_add_(0, idx, value)
        goals = d["g"][mask]
        if len(goals):
            self.maxima = torch.maximum(self.maxima, goals.abs().amax(0))
            anchors = goals.new_tensor(
                [
                    [0.4, 0.15, 0],
                    [0.4, -0.15, 0],
                    [0.4, 0, 0.3],
                    [0.4, 0, -0.3],
                    [0.4, 0.15, 0.3],
                    [0.4, -0.15, -0.3],
                    [0.3, 0.2, 0.4],
                    [0.3, -0.2, -0.4],
                ]
            )
            self.near += (
                ((goals[:, None, :] - anchors).abs() < goals.new_tensor([0.06, 0.04, 0.07]))
                .all(-1)
                .sum(0)
            )
        self.steps += int(mask.sum())
        self.nonflat += int((mask & (d["terrain_level"] > 0)).sum())
        self.last = l.sim_steps
        l.extra_state["stairs_command_coverage"] = {
            k: getattr(self, k)
            for k in ("count", "triples", "crossing", "falls", "near", "maxima", "steps", "nonflat")
        }

    def summary(self):
        return {
            k: (
                getattr(self, k).cpu().tolist()
                if torch.is_tensor(getattr(self, k))
                else getattr(self, k)
            )
            for k in ("count", "triples", "crossing", "falls", "near", "maxima", "steps", "nonflat")
        }


def run_tool(script, args, log, timeout=360):
    with log.open("w") as f:
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / script), *map(str, args)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def continuation_regressions(evaluation, baseline, retained_max_level=None):
    """Stairs prioritize survival/crossing; flat keeps strict velocity guards.

    Stair RMSE remains logged. Stop on persistent mean motion changes or falls,
    not isolated contact-induced velocity oscillations.
    """
    flat = {k: v for k, v in evaluation.items() if not k.startswith("stairs_")}
    regressions = physical_regressions(flat, {k: baseline[k] for k in flat})
    for name, value in evaluation.items():
        if not name.startswith("stairs_"):
            continue
        if (
            retained_max_level is not None
            and int(name.split("_L", 1)[1].split("_", 1)[0]) > retained_max_level
        ):
            # New, unmastered terrain is exploratory. Log its failures separately;
            # retained skills and numerical/gradient safety still stop the run.
            continue
        old = baseline[name]
        import math

        from evaluate_go2_omnidirectional import PERTURBATIONS, STAIRS_OMNI, stairs_speed_commands

        if "_cmd" in name:
            command = PERTURBATIONS[int(name.rsplit("_cmd", 1)[1])]
        elif "_omni" in name:
            command = STAIRS_OMNI[int(name.rsplit("_omni", 1)[1])]
        else:
            suffix = name.rsplit("_S", 1)[1].split("_")
            command = stairs_speed_commands(int(suffix[0]))[int(suffix[1])]
        old_bias = math.hypot(old["actual_vx_m_s"] - command[0], old["actual_vy_m_s"] - command[1])
        new_bias = math.hypot(
            value["actual_vx_m_s"] - command[0], value["actual_vy_m_s"] - command[1]
        )
        if new_bias > old_bias + 0.15:
            regressions.append(
                dict(
                    scenario=name,
                    metric="sustained_mean_xy_bias",
                    value=new_bias,
                    baseline=old_bias,
                )
            )
        old_yaw = abs(old["actual_wz_rad_s"] - command[2])
        new_yaw = abs(value["actual_wz_rad_s"] - command[2])
        if new_yaw > old_yaw + 0.30:
            regressions.append(
                dict(
                    scenario=name, metric="sustained_mean_yaw_bias", value=new_yaw, baseline=old_yaw
                )
            )
        if value["fall"] >= old["fall"] + 0.4:
            regressions.append(
                dict(scenario=name, metric="fall", value=value["fall"], baseline=old["fall"])
            )
        if (
            name.endswith("_cmd0")
            and old["crossing_passed"] >= 0.8
            and value["crossing_passed"] < old["crossing_passed"] - 0.4
        ):
            regressions.append(
                dict(
                    scenario=name,
                    metric="crossing_passed",
                    value=value["crossing_passed"],
                    baseline=old["crossing_passed"],
                )
            )
    return regressions


def wm_gate(current, baseline, nlevels=5):
    gs = [
        g
        for g in current["gradients"]
        if g["quantity"] == "joint_velocity"
        and g["epsilon"] == 0.03
        and g["sign_agreement"] is not None
    ]
    old = {
        g["scenario"]: g
        for g in baseline["gradients"]
        if g["quantity"] == "joint_velocity"
        and g["epsilon"] == 0.03
        and g["sign_agreement"] is not None
    }
    prediction = [p for p in current["prediction"] if p["horizon"] == 32]
    minimum = min((g["sign_agreement"] for g in gs), default=0)
    mean = sum(g["sign_agreement"] for g in gs) / max(len(gs), 1)
    regressions = [
        g["scenario"]
        for g in gs
        if g["scenario"] in old
        and g["sign_agreement"] < old[g["scenario"]]["sign_agreement"] - 0.05
    ]
    errors = bool(prediction) and all(
        p["position_m"] < 0.08
        and p["velocity_m_s"] < 0.15
        and p["joint_position_rad"] < 0.12
        and p["attitude_rad"] < 0.15
        for p in prediction
    )
    # Limited S0 range, inherited contact tail, same-dataset paired baseline.
    # No admission of a new range is made by this aggregate low-speed gate.
    passed = (
        len(gs) >= 40
        and minimum >= 0.75
        and mean >= 0.85
        and len(regressions) <= len(gs) * 0.10
        and errors
    )
    mask, scopes = imagination_scope_admission({0: current}, [[0] * nlevels, [0] * nlevels])
    scoped_pass = any(v["admitted"] for v in scopes.values())
    return dict(
        passed=scoped_pass,
        global_s0_passed=passed,
        start_scope_mask=mask.tolist(),
        scope_details=scopes,
        minimum_qd_sign=minimum,
        mean_qd_sign=mean,
        regression_probes=regressions,
        prediction_pass=errors,
        probes=len(gs),
        scope="S0 controlled continuation only; higher speed uses >=.80 per probe",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", type=Path, default=ROOT / "results/omnidirectional_terrain/stageB/last.pt"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=7200)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--audit-steps", type=int, default=0)
    p.add_argument("--eval-period", type=int, default=500)
    p.add_argument("--wm-adaptation", type=int, default=200)
    p.add_argument(
        "--retained-max-level",
        type=int,
        default=None,
        help="Explicit retained-skill guard ceiling; higher levels remain diagnostic",
    )
    p.add_argument("--actor-lr-scale", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--task",
        default="stairs_omni_continuation",
        choices=("stairs_omni_continuation", "stairs_omni_level7"),
    )
    a = p.parse_args()
    if not 0 < a.seconds <= 7200 or a.iterations < 1 or a.eval_period < 100:
        raise ValueError("Invalid bounded budget")
    a.output.mkdir(parents=True, exist_ok=a.resume)
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = configuration(a.task)
    base = json.loads(a.baseline.read_text())["aggregate"]
    if a.actor_lr_scale is not None:
        if not 0 < a.actor_lr_scale <= 0.1:
            raise ValueError("Continuation actor scale must be in (0,.1]")
        cfg.algo.actor_lr_scale = a.actor_lr_scale
    schedule = source["extra_state"].get("stairs_schedule") if a.resume else None
    cfg.algo.wm_only_iters = (
        schedule["wm_only_iters"] if schedule else int(source["iteration"]) + a.wm_adaptation
    )
    cfg.algo.critic_warmup_iters = 50
    cfg.algo.iterations = int(source["iteration"]) + a.iterations
    cfg.algo.numbered_ckpt_every = 2000
    source_info = {k: source[k] for k in ("iteration", "sim_steps", "update_counts", "wall_time")}
    protected = {
        k: copy.deepcopy(source["task_contract"][k])
        for k in source["task_contract"]
        if k != "commands"
    }
    source_pi = {k: v.clone() for k, v in source["pi"].items()} if a.audit_steps else None
    del source
    OmegaConf.save(cfg, a.output / "config.yaml")
    started = time.monotonic()
    monitor = ResourceMonitor()
    monitor.thread.start()
    l = env = None
    reason = "iteration_budget"
    try:
        env = environment(cfg, 2048)
        env.init_state()
        l = SuperMimicLearner(
            env, build_learner_config(cfg), a.output, device="cuda", backend="mujoco"
        )
        # This experiment permits no protected-field differences, even the
        # generic migration helper's historical flat-boundary exception.
        actual = {k: v for k, v in env.supermimic_contract.items() if k != "commands"}
        if protected != actual:
            raise ValueError("Protected reward/physics/network contract changed")
        l.load(a.checkpoint, allow_command_distribution_change=not a.resume)
        cmd = env.command_manager.get_term("velocity")
        if not a.resume:
            for key in (
                "omni_coverage",
                "actual_actor_start_modes",
                "actual_policy_start_modes",
                "stairs_command_coverage",
                "stairs_warm_admission",
            ):
                l.extra_state.pop(key, None)
        l.extra_state["stairs_schedule"] = dict(
            wm_only_iters=cfg.algo.wm_only_iters, critic_warmup_iters=50
        )
        l.extra_state.setdefault(
            "stairs_run_origin",
            l.extra_state.get("stairs_source", dict(path=str(a.checkpoint), **source_info)),
        )
        l.extra_state["stairs_source"] = dict(path=str(a.checkpoint), **source_info)
        coverage = CommandCoverage(l)
        stairs = StairsCoverage(l)
        manifest = dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            argv=sys.argv,
            source=str(a.checkpoint.resolve()),
            source_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
            source_counters=source_info,
            num_envs=2048,
            resume_mode=l.resume_mode,
            protected_contract_equal=True,
            reward=env.supermimic_contract["reward"],
            historical_stageC_gate_unchanged=True,
            retained_max_level=a.retained_max_level,
            higher_levels="exploration, not retained-skill regression; all falls logged",
            safe_spawn_change="opt-in minimal vertical lift, default legacy false; command-only migration",
            gate="S0 min qd sign .75, mean .85, paired deterioration <=5pp in <=10% probes; promotion min .80 per probe plus physics",
        )
        dump(a.output / "invocation.json", manifest)
        (a.output / "source.patch").write_text(
            subprocess.check_output(["git", "diff", "HEAD"], text=True)
        )
        if a.audit_steps:
            for _ in range(a.audit_steps):
                l.collect(1)
                coverage.observe()
                stairs.observe()
            delta = max(
                float((v - l.pi.state_dict()[k].cpu()).abs().max()) for k, v in source_pi.items()
            )
            if delta != 0:
                raise ValueError("Audit changed actor")
            l.save()
            coverage.save(a.output)
            dump(a.output / "stairs_coverage.json", stairs.summary())
            dump(
                a.output / "summary.json",
                dict(
                    actor_max_tensor_diff=delta,
                    new_transitions=l.sim_steps - source_info["sim_steps"],
                    wall_s=time.monotonic() - started,
                    optimizer_updates=0,
                ),
            )
            return
        if l.buffer.size < l.cfg.warmup_steps:
            l.collect(l.cfg.warmup_steps - l.buffer.size)
            coverage.observe()
            stairs.observe()
        start, hist = l.buffer.sample_starts(512, 5)
        fixed = l.mdp.obs(
            unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"]
        ).detach()
        goal = start["g"].detach()
        with torch.no_grad():
            reference = l.pi(fixed, goal).detach().clone()
        stable = {
            "pi": copy.deepcopy(l.pi.state_dict()),
            "opt": copy.deepcopy(l.opt_pi.state_dict()),
            "iteration": l.iteration,
        }
        best_score = float("inf")
        strikes = 0
        last_eval = 0
        warm_gate = False
        diagnostic_cache = {}
        for speed in range(4):
            dataset = a.output / f"holdout_S{speed}"
            baseline_diag = a.output / f"wm_source_S{speed}"
            if not (dataset / "heldout.pt").exists():
                run_tool(
                    "diagnose_go2_omni_wm.py",
                    [
                        "--mode",
                        "collect",
                        "--checkpoint",
                        a.checkpoint,
                        "--output",
                        dataset,
                        "--stairs-speed",
                        speed,
                        "--max-level",
                        cfg.env.commands.velocity.terrain_max_level,
                    ],
                    a.output / f"collect_S{speed}.log",
                )
            if not (baseline_diag / "diagnostics.json").exists():
                run_tool(
                    "diagnose_go2_omni_wm.py",
                    [
                        "--mode",
                        "evaluate",
                        "--checkpoint",
                        a.checkpoint,
                        "--dataset",
                        dataset,
                        "--output",
                        baseline_diag,
                    ],
                    a.output / f"wm_source_S{speed}.log",
                )
            diagnostic_cache[speed] = json.loads((baseline_diag / "diagnostics.json").read_text())
        # Broader flat commands only open if independently demonstrated already.
        flat_names = (
            "backward_medium",
            "forward_high",
            "lateral_left",
            "lateral_right",
            "turn_left",
            "turn_right",
        )
        flat_ok = all(
            base[k]["fall"] == 0
            and base[k]["xy_rmse_m_s"] < 0.10
            and base[k]["yaw_rmse_rad_s"] < 0.15
            for k in flat_names
        )
        if flat_ok and not a.resume:
            cmd.sampler.set_speed_level(1)
        dump(
            a.output / "flat_speed_admission.json",
            dict(level=cmd.sampler.speed_level, evidence_commands=flat_names, physics_pass=flat_ok),
        )

        def evaluate_at(label):
            policy = l.save(a.output / "evaluation_policy.pt", include_optimizers=False)
            out = a.output / f"eval_{label}"
            run_tool(
                "evaluate_go2_omnidirectional.py",
                [
                    "--checkpoint",
                    policy,
                    "--output",
                    out,
                    "--stairs-omni",
                    "--max-level",
                    cfg.env.commands.velocity.terrain_max_level,
                    "--seeds",
                    2426,
                    2427,
                    2428,
                    2429,
                    2430,
                ],
                a.output / f"eval_{label}.log",
            )
            return json.loads((out / "evaluation.json").read_text())["aggregate"]

        def diagnostics_at(label, speed):
            policy = l.save(a.output / "evaluation_policy.pt", include_optimizers=False)
            out = a.output / f"wm_{label}_S{speed}"
            if not (out / "diagnostics.json").exists():
                run_tool(
                    "diagnose_go2_omni_wm.py",
                    [
                        "--mode",
                        "evaluate",
                        "--checkpoint",
                        policy,
                        "--dataset",
                        a.output / f"holdout_S{speed}",
                        "--output",
                        out,
                    ],
                    a.output / f"wm_{label}_S{speed}.log",
                )
            return json.loads((out / "diagnostics.json").read_text())

        def callback(learner):
            nonlocal reason, stable, strikes, last_eval, best_score, warm_gate
            coverage.observe()
            stairs.observe()
            elapsed = time.monotonic() - started
            delta = l.iteration - source_info["iteration"]
            if elapsed >= a.seconds - 15:
                reason = "wall_budget"
                l.stop_requested = True
                return
            if delta % 25 == 0:
                with torch.no_grad():
                    pred = l.pi(fixed, goal)
                    drift = float((pred - reference).square().mean().sqrt())
                    sat = float((pred.abs() > 0.98).float().mean())
                row = dict(
                    iteration=l.iteration,
                    new_iterations=delta,
                    new_transitions=l.sim_steps - source_info["sim_steps"],
                    updates={
                        k: l.update_counts[k] - source_info["update_counts"][k]
                        for k in l.update_counts
                    },
                    wall_s=elapsed,
                    actor_drift_rmse=drift,
                    saturation=sat,
                    actor_lr=l.opt_pi.param_groups[0]["lr"],
                    flat_speed_level=cmd.sampler.speed_level,
                    stairs_speed_levels=cmd.sampler.stairs_levels.tolist(),
                    gpu_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    rss_mib=psutil.Process().memory_info().rss / 2**20,
                )
                with (a.output / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                dump(a.output / "coverage.json", coverage.summary())
                dump(a.output / "stairs_coverage.json", stairs.summary())
                if not torch.isfinite(pred).all() or sat > 0.6 or drift > 0.8:
                    reason = "actor_output_guard"
                    l.stop_requested = True
            if l.iteration == l.cfg.wm_only_iters and not l.extra_state.get(
                "stairs_warm_admission", {}
            ).get("passed", False):
                diag = diagnostics_at(f"warm_{l.iteration}", 0)
                gate = wm_gate(diag, diagnostic_cache[0], cmd.sampler.stairs_levels.shape[1])
                dump(a.output / "warm_admission.json", gate)
                warm_gate = gate["passed"]
                l.extra_state["stairs_warm_admission"] = gate
                if gate["passed"]:
                    l.buffer.start_scope_mask = torch.tensor(
                        gate["start_scope_mask"], device=l.device, dtype=torch.bool
                    )
                with (a.output / "warm_attempts.jsonl").open("a") as f:
                    f.write(json.dumps(dict(iteration=l.iteration, **gate)) + "\n")
                if not warm_gate:
                    origin = l.extra_state["stairs_run_origin"]["iteration"]
                    if l.iteration - origin < 2000 and time.monotonic() - started < a.seconds - 600:
                        l.cfg.wm_only_iters = l.iteration + 200
                        cfg.algo.wm_only_iters = l.cfg.wm_only_iters
                        l.extra_state["stairs_schedule"]["wm_only_iters"] = l.cfg.wm_only_iters
                    else:
                        reason = "wm_new_distribution_gate"
                        l.stop_requested = True
            if a.resume:
                warm_gate = l.extra_state.get("stairs_warm_admission", {}).get("passed", False)
            if (
                delta >= last_eval + a.eval_period
                and warm_gate
                and not l.stop_requested
                and elapsed < a.seconds - 300
            ):
                evaluation = evaluate_at(str(l.iteration))
                # Compare only inherited core task grid, not out-of-range diagnostics.
                core = {
                    k: v
                    for k, v in evaluation.items()
                    if k in base
                    and ("_omni" not in k and "_S" not in k and k != "lateral_boundary")
                }
                regressions = continuation_regressions(
                    core, {k: base[k] for k in core}, a.retained_max_level
                )
                # New maneuvers have their own baseline-relative fall/tracking guard.
                for k, v in evaluation.items():
                    if "_S0_" in k:
                        old = base[k]
                        regressions.extend(
                            continuation_regressions({k: v}, {k: old}, a.retained_max_level)
                        )
                strikes = strikes + 1 if regressions else 0
                promotions = []
                diags = {}
                for speed in sorted(set(cmd.sampler.stairs_levels[:, 1:].flatten().tolist())):
                    diags[speed] = diagnostics_at(str(l.iteration), speed)
                current_gate = (
                    wm_gate(
                        diags.get(0, diagnostic_cache[0]),
                        diagnostic_cache[0],
                        cmd.sampler.stairs_levels.shape[1],
                    )
                    if 0 in diags
                    else {"passed": True}
                )
                scope_mask, scope_details = imagination_scope_admission(
                    diags, cmd.sampler.stairs_levels.tolist()
                )
                l.buffer.start_scope_mask = scope_mask.to(l.device)
                current_gate["active_scopes"] = scope_details
                if not any(v["admitted"] for v in scope_details.values()):
                    reason = "wm_action_gradient_guard"
                    l.stop_requested = True
                if not regressions and not l.stop_requested:
                    expanded_flat = all(
                        evaluation[f"flat_expanded_{i}"]["fall"] == 0
                        and evaluation[f"flat_expanded_{i}"]["xy_rmse_m_s"] < 0.10
                        and evaluation[f"flat_expanded_{i}"]["yaw_rmse_rad_s"] < 0.15
                        for i in range(4)
                    )
                    if expanded_flat:
                        cmd.sampler.set_speed_level(2)
                    for direction in range(2):
                        for level in range(1, cmd.sampler.stairs_levels.shape[1]):
                            speed = int(cmd.sampler.stairs_levels[direction, level])
                            gate = stairs_speed_admission(
                                evaluation, diags[speed], direction, level, speed
                            )
                            if (
                                gate["admitted"]
                                and speed < 3
                                and (speed == 0 or cmd.sampler.speed_level >= 2)
                            ):
                                # Candidate range must itself have reliable WM gradients;
                                # physics at candidate is measured before any promotion.
                                candidate = diags.get(speed + 1)
                                if candidate is None:
                                    candidate = diagnostics_at(str(l.iteration), speed + 1)
                                    diags[speed + 1] = candidate
                                candidate_gate = stairs_speed_admission(
                                    evaluation, candidate, direction, level, speed + 1
                                )
                                if candidate_gate["admitted"]:
                                    cmd.sampler.set_stairs_speed_level(direction, level, speed + 1)
                                    promotions.append(
                                        dict(
                                            direction=direction,
                                            terrain_level=level,
                                            from_speed=speed,
                                            to_speed=speed + 1,
                                            evidence=candidate_gate,
                                        )
                                    )
                    stable = {
                        "pi": copy.deepcopy(l.pi.state_dict()),
                        "opt": copy.deepcopy(l.opt_pi.state_dict()),
                        "iteration": l.iteration,
                    }
                    traversal = [
                        v
                        for k, v in evaluation.items()
                        if k.startswith("stairs_") and k.endswith("_cmd0")
                    ]
                    score = sum(
                        v["fall"] * 5 + (1 - v["crossing_passed"]) * 2 for v in traversal
                    ) / len(traversal)
                    score += sum(
                        v["fall"] + 0.05 * v["xy_rmse_m_s"]
                        for k, v in evaluation.items()
                        if "_S0_" in k
                    ) / max(1, sum("_S0_" in k for k in evaluation))
                    if score < best_score:
                        best_score = score
                        l.save(a.output / "best.pt")
                        dump(
                            a.output / "best.json",
                            dict(
                                iteration=l.iteration, score=score, evaluation=f"eval_{l.iteration}"
                            ),
                        )
                event = dict(
                    iteration=l.iteration,
                    wall_s=time.monotonic() - started,
                    regressions=regressions,
                    exploratory_regressions=continuation_regressions(
                        {
                            k: v
                            for k, v in evaluation.items()
                            if k in base
                            and k.startswith("stairs_")
                            and int(k.split("_L", 1)[1].split("_", 1)[0])
                            > (a.retained_max_level or 7)
                        },
                        base,
                    ),
                    strikes=strikes,
                    promotions=promotions,
                    wm_gate=current_gate,
                    stairs_speed_levels=cmd.sampler.stairs_levels.tolist(),
                )
                with (a.output / "physics_guard.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                if strikes >= 2:
                    reason = "physical_regression"
                    l.stop_requested = True
                last_eval = delta
            if (
                psutil.virtual_memory().available < 600 * 2**20
                or psutil.Process().memory_info().rss > 6.5 * 2**30
            ):
                reason = "host_memory_guard"
                l.stop_requested = True
            if shutil.disk_usage(a.output).free < 20 * 2**30:
                reason = "disk_guard"
                l.stop_requested = True
            if time.monotonic() - started >= a.seconds and not l.stop_requested:
                reason = "wall_budget"
                l.stop_requested = True
            if l.stop_requested and reason not in ("wall_budget", "iteration_budget"):
                l.save(a.output / "pre_rollback.pt")
                l.pi.load_state_dict(stable["pi"])
                l.opt_pi.load_state_dict(stable["opt"])
                dump(
                    a.output / "rollback.json",
                    dict(
                        reason=reason,
                        stable_iteration=stable["iteration"],
                        wm_replay_preserved=True,
                    ),
                )

        if (
            a.resume
            and not l.extra_state.get("stairs_warm_admission", {}).get("passed", False)
            and l.iteration >= l.cfg.wm_only_iters
        ):
            gate = wm_gate(
                diagnostics_at(f"resume_warm_{l.iteration}", 0),
                diagnostic_cache[0],
                cmd.sampler.stairs_levels.shape[1],
            )
            dump(a.output / "warm_admission.json", gate)
            l.extra_state["stairs_warm_admission"] = gate
            if gate["passed"]:
                l.buffer.start_scope_mask = torch.tensor(
                    gate["start_scope_mask"], device=l.device, dtype=torch.bool
                )
            warm_gate = gate["passed"]
            if not warm_gate:
                origin = l.extra_state["stairs_run_origin"]["iteration"]
                if l.iteration - origin < 2000:
                    l.cfg.wm_only_iters = l.iteration + 200
                    cfg.algo.wm_only_iters = l.cfg.wm_only_iters
                    l.extra_state["stairs_schedule"]["wm_only_iters"] = l.cfg.wm_only_iters
                else:
                    reason = "wm_new_distribution_gate"
                    l.stop_requested = True
        l.on_iteration = callback
        result = (
            l.train()
            if not l.stop_requested
            else dict(
                completed=False, iteration=l.iteration, sim_steps=l.sim_steps, wall_time=l.wall_time
            )
        )
        l.save()
        coverage.save(a.output)
        dump(a.output / "stairs_coverage.json", stairs.summary())
        dump(a.output / "curriculum.json", cmd.curriculum.summary())
        dump(
            a.output / "summary.json",
            dict(
                stop_reason=reason,
                result=result,
                new_transitions=l.sim_steps - source_info["sim_steps"],
                new_updates={
                    k: l.update_counts[k] - source_info["update_counts"][k] for k in l.update_counts
                },
                wall_s=time.monotonic() - started,
                stairs_speed_levels=cmd.sampler.stairs_levels.tolist(),
                flat_speed_level=cmd.sampler.speed_level,
                profile={
                    k: dict(count=len(v), mean_ms=sum(v) / len(v))
                    for k, v in l.profile_ms.items()
                    if v
                },
                checkpoint=str((a.output / "last.pt").resolve()),
            ),
        )
    except BaseException as e:
        dump(a.output / "failure.json", dict(type=type(e).__name__, message=str(e)))
        if l is not None:
            l.save(a.output / "interrupted.pt")
        raise
    finally:
        monitor.finish(a.output)
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
