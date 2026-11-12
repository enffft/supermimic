"""Device-side transition aggregation; one JSON record per training iteration."""

import json
import resource
import time

import torch

from unisupermimic.rl.algos.supermimic.geometry import matvec


@torch.no_grad()
def record_transition(env, terms, state, goal, normal):
    c = env.command_manager.get_term("velocity")
    p, rotation, velocity, angular, *_ = state
    actual = torch.cat(
        (
            matvec(rotation.transpose(-1, -2), velocity)[..., :2],
            matvec(rotation.transpose(-1, -2), angular)[..., 2:3],
        ),
        -1,
    )
    values = dict(terms)
    for i, axis in enumerate(("vx", "vy", "yaw")):
        values["error_sq_" + axis] = (actual[:, i] - goal[:, i]).square()
        values["mean_command_" + axis] = goal[:, i]
        values["mean_actual_" + axis] = actual[:, i]
    values.update(
        terrain_normal_angle=torch.acos(normal[:, 2].clamp(-1, 1)),
        body_pitch=torch.asin((-rotation[:, 2, 0]).clamp(-1, 1)),
        body_roll=torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2]),
        terrain_alignment_error=1 - (normal * rotation[:, :, 2]).sum(-1),
        transitions=torch.ones_like(p[:, 0]),
        valid_transitions=(~c.mdp.hard_termination(state)).float(),
        falls=c.mdp.hard_termination(state).float(),
        wheel_speed_mean=state[5][:, 12:].mean(-1),
        wheel_target_mean=(
            env.velocity_executed_action[:, 12:] * c.mdp.control.wheel_velocity_scale
        ).mean(-1),
    )
    if not hasattr(env, "go2w_metrics"):
        env.go2w_metrics = {k: p.new_zeros(6, 8) for k in values}
    tags = env.velocity_terrain_tags
    scope = tags["command_domain"] * 8 + tags["terrain_level"]
    for key, value in values.items():
        env.go2w_metrics[key].view(-1).scatter_add_(0, scope, value)
    # Cumulative WM data coverage includes warmup and partial episodes and is
    # checkpointed with the existing curriculum statistics (type x level).
    atlas_scope = c.curriculum.terrain_type * 8 + tags["terrain_level"]
    for name in ("transitions", "valid_transitions", "falls"):
        target = c.curriculum.statistics["real_" + name]
        target.view(-1).scatter_add_(0, atlas_scope, values[name].double())


def begin_iteration(env, learner):
    c = env.command_manager.get_term("velocity")
    curriculum = c.curriculum
    phase = "wm" if learner.iteration < learner.cfg.wm_only_iters else "skill"
    if c.cfg.automatic_wm_skill_transition and curriculum.phase != phase:
        curriculum.phase = phase
        # Existing episodes finish at their recorded levels; no manual reset or
        # promotion from WM coverage. New episodes use measured skill ceilings.
        curriculum.success_streak.zero_()
        curriculum.failure_streak.zero_()
    ceilings = torch.zeros(6, dtype=torch.long, device=env.device)
    for domain in range(1, 6):
        selected = curriculum.mastery[c.command_domain == domain]
        if selected.numel():
            ceilings[domain] = selected.max()
    learner.buffer.start_scope_mask = (
        torch.arange(8, device=env.device)[None] <= ceilings[:, None]
    ).to(learner.device)
    env.go2w_iteration_started = time.perf_counter()
    env.go2w_update_mark = {k: len(v) for k, v in learner.profile_ms.items()}
    # The warmup has no optimizer iteration; exclude it from iteration throughput.
    if hasattr(env, "go2w_metrics"):
        for value in env.go2w_metrics.values():
            value.zero_()


def finish_iteration(env, learner):
    if not hasattr(env, "go2w_metrics"):
        return
    if learner.iteration % learner.cfg.log_every and learner.iteration != learner.cfg.iterations:
        learner.profile_ms.clear()
        return
    metrics = env.go2w_metrics
    count = metrics["transitions"].clamp_min(1)
    # One small transfer per metric at iteration boundary, never per-env CSV.
    data = {
        k: (v / count).cpu().tolist()
        for k, v in metrics.items()
        if k not in ("transitions", "valid_transitions", "falls")
    }
    for axis in ("vx", "vy", "yaw"):
        data["RMSE_" + axis] = (metrics["error_sq_" + axis] / count).sqrt().cpu().tolist()
    for key in ("transitions", "valid_transitions", "falls"):
        data[key] = metrics[key].cpu().tolist()
    timings = {
        k: sum(v[env.go2w_update_mark.get(k, 0) :]) / 1000 for k, v in learner.profile_ms.items()
    }
    elapsed = time.perf_counter() - env.go2w_iteration_started
    data.update(
        iteration=learner.iteration,
        phase=env.command_manager.get_term("velocity").curriculum.phase,
        timings_seconds=timings,
        update_counts=dict(learner.update_counts),
        env_steps_per_second=float(metrics["transitions"].sum()) / max(elapsed, 1e-9),
        peak_CPU_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        peak_GPU_MiB=(
            torch.cuda.max_memory_allocated(learner.device) / 2**20
            if learner.device.type == "cuda"
            else 0
        ),
        curriculum=env.command_manager.get_term("velocity").curriculum.summary(),
    )
    with (learner.log_dir / "go2w_diagnostics.jsonl").open("a") as stream:
        stream.write(json.dumps(data) + "\n")
    for value in metrics.values():
        value.zero_()
    # Profiling arrays are consumed above, bounded for long runs.
    learner.profile_ms.clear()
