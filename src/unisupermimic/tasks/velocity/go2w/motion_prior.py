"""Command-conditioned reference windows and physical reference collection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat, matvec, quat_to_mat

from .contract import LEG_JOINTS, WHEEL_JOINTS

FIELDS = {
    "command": 3,
    "leg_q": 12,
    "leg_qd": 12,
    "wheel_q": 4,
    "wheel_qd": 4,
    "wheel_rate_valid": 4,
    "contact_mask": 4,
    "motion_phase": 4,
    "base_pos": 3,
    "base_quat": 4,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
}
NORMAL_ACTOR, RSI_ACTOR, REFERENCE_ROLLOUT, REFERENCE_LOCAL = range(4)
STARTUP, STEADY, STOPPING = range(3)
EARLY_CATEGORY_WEIGHTS = (0.10, 0.25, 0.30, 0.04, 0.25, 0.03, 0.03, 0.0)
CATEGORY_WEIGHTS = (0.05, 0.20, 0.25, 0.10, 0.20, 0.10, 0.10, 0.0)
FLAT_DOMAINS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def command_keys(goal):
    """Bit pattern of active axes and ternary code of their signs."""
    sign = torch.where(goal.abs() >= 0.025, goal.sign(), 0).long()
    category = (sign.abs() * sign.new_tensor([1, 2, 4])).sum(-1)
    signed = ((sign + 1) * sign.new_tensor([1, 3, 9])).sum(-1)
    return category, signed


def command_speed_bins(goal):
    """Three speed bands using each axis's physical command scale."""
    normalized = (goal.abs() / goal.new_tensor([0.75, 0.35, 0.80])).amax(-1)
    return (normalized >= 0.25).long() + (normalized >= 0.60).long()


def reference_action(q, wheel_rate, control, default, limits):
    delta = q - default
    if control.full_leg_target_range:
        span = torch.where(delta >= 0, limits[:, 1] - default, default - limits[:, 0])
    else:
        span = torch.full_like(q, control.leg_position_scale)
    raw = torch.cat((delta / span, wheel_rate / control.wheel_velocity_scale), -1)
    return raw.clamp(-1, 1), raw.abs() > 1


class MotionBank:
    def __init__(self, path, device, cfg):
        path = Path(path)
        manifest_file = path / "manifest.json"
        manifest_bytes = manifest_file.read_bytes()
        manifest = json.loads(manifest_bytes)
        if (
            manifest.get("validation_level") != "kinematic_only"
            or manifest.get("physics_validated") is not False
            or manifest.get("quaternion_order") != "wxyz"
            or manifest.get("dt") != 0.02
            or tuple(manifest.get("leg_joint_names", [])) != LEG_JOINTS
            or tuple(manifest.get("wheel_joint_names", [])) != WHEEL_JOINTS
        ):
            raise ValueError("Motion Bank state contract differs")
        digest = hashlib.sha256(manifest_bytes)
        self.rows, nominal, weight, arrays = [], [], [], {k: [] for k in FIELDS}
        phase_starts, reset_starts, startup_starts, stopping_starts, phases, local_starts = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.horizon = int(cfg["horizon"])
        self.config = cfg
        for row in manifest["motions"]:
            status = row["quality_status"]
            if status == "qualified" and not row["failure_reasons"]:
                quality = cfg["qualified_weight"]
            elif status == "candidate" and row["failure_reasons"] == ["dynamic_diagonal_support"]:
                quality = cfg["dynamic_candidate_weight"]
            else:
                continue
            file = (path / row["file_path"]).resolve()
            if not file.is_relative_to(path.resolve()):
                raise ValueError("Invalid NPZ path")
            digest.update(file.read_bytes())
            with np.load(file, allow_pickle=False) as clip:
                command = np.asarray(row["nominal_command"], dtype=np.float32)
                count = row["frame_count"]
                if (
                    count != 301
                    or not np.allclose(clip["command_nominal"], command)
                    or not np.allclose(clip["time"], np.arange(count) * 0.02)
                    or np.any(clip["ik_failed"])
                ):
                    raise ValueError(f"Invalid timeline or IK in {file.name}")
                for key, width in FIELDS.items():
                    value = clip[key]
                    if value.shape != (count, width) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid field {key} in {file.name}")
                    arrays[key].append(value)
                plateau = np.max(np.abs(clip["command"] - command), axis=-1) < 1e-5
                valid = np.lib.stride_tricks.sliding_window_view(plateau, self.horizon + 1).all(-1)
                rows = np.flatnonzero(valid)
                if len(rows) < 1:
                    raise ValueError(f"No plateau window in {file.name}")
                phase_starts.append(
                    rows[np.linspace(0, len(rows) - 1, cfg["phase_samples"]).round().astype(int)]
                )
                frame = np.arange(len(valid))
                phase = np.full(count, STEADY)
                if np.any(command):
                    first, last = np.flatnonzero(plateau)[[0, -1]]
                    phase[:first] = STARTUP
                    phase[last:] = STOPPING
                phases.append(phase)

                def sampled(indices):
                    if not len(indices):
                        indices = rows
                    return indices[
                        np.linspace(0, len(indices) - 1, cfg["phase_samples"]).round().astype(int)
                    ]

                startup_starts.append(sampled(frame[phase[: len(frame)] == STARTUP]))
                stopping_starts.append(sampled(frame[phase[: len(frame)] == STOPPING]))
                local = frame[clip["contact_mask"][: len(frame)].all(-1) & (frame + 32 < count)]
                # Retain the three reference phases for short real exploration.
                local_starts.append(sampled(local))
                stable = valid & clip["contact_mask"][: len(valid)].all(-1)
                stable = np.flatnonzero(stable)
                if len(stable) < 1:
                    raise ValueError(f"No four-wheel plateau start in {file.name}")
                reset_starts.append(
                    stable[
                        np.linspace(0, len(stable) - 1, cfg["phase_samples"]).round().astype(int)
                    ]
                )
            self.rows.append(row)
            nominal.append(command)
            weight.append(quality)
        if not self.rows:
            raise ValueError("Motion Bank has no usable clips")
        self.identity = digest.hexdigest()
        self.nominal = torch.as_tensor(np.stack(nominal), device=device)
        self.data = {
            k: torch.as_tensor(
                np.stack(v),
                dtype=torch.bool
                if k.endswith(("_valid", "_mask"))
                else torch.int8
                if k == "motion_phase"
                else torch.float32,
                device=device,
            )
            for k, v in arrays.items()
        }
        self.weight = torch.tensor(weight, device=device)
        self.candidate = torch.tensor(
            [r["quality_status"] == "candidate" for r in self.rows], device=device
        )
        self.categories, self.keys = command_keys(self.nominal)
        self.phase_starts = torch.as_tensor(np.stack(phase_starts), device=device)
        self.reset_starts = torch.as_tensor(np.stack(reset_starts), device=device)
        self.local_starts = torch.as_tensor(np.stack(local_starts), device=device)
        self.frame_phase = torch.as_tensor(np.stack(phases), dtype=torch.int8, device=device)
        self.windows = torch.as_tensor(
            np.stack((startup_starts, phase_starts, stopping_starts), axis=1), device=device
        )
        self.standing = self.nominal.abs().sum(-1).argmin()

        self.scales = self.nominal.new_tensor(cfg["feature_scales"])
        self.match_weights = self.nominal.new_tensor(cfg["matching_weights"])
        self.command_scale = self.nominal.new_tensor(cfg["command_scale"])
        self.steps = torch.arange(1, self.horizon + 1, device=device)
        self.positions = self.data["leg_q"]
        self.velocity = torch.cat((self.data["leg_qd"], self.data["wheel_qd"]), -1)

    @torch.no_grad()
    def match(self, state, goal, flat, clip_hint=None, phase_hint=None):
        category, key = command_keys(goal)
        phase = torch.full_like(category, STEADY)
        stopping = torch.zeros_like(flat)
        if state[1] is not None:
            rotation = state[1].transpose(-1, -2)
            velocity = torch.cat(
                (matvec(rotation, state[2])[:, :2], matvec(rotation, state[3])[:, 2:3]), -1
            )
            projected = (velocity * goal / self.command_scale.square()).sum(-1)
            target = (goal / self.command_scale).square().sum(-1)
            phase = torch.where((category != 0) & (projected < 0.35 * target), STARTUP, phase)
            stopping = (goal == 0).all(-1) & ((velocity / self.command_scale).norm(dim=-1) > 0.12)
            phase = torch.where(stopping, STOPPING, phase)
            query = torch.where(stopping[:, None], velocity, goal)
            _, key = command_keys(query)
        else:
            query = goal
        distance = ((query[:, None] - self.nominal) / self.command_scale).square().sum(-1)
        eligible_clip = (key[:, None] == self.keys) & ((goal == 0).all(-1) | (category != 0))[
            :, None
        ]
        # An actual velocity at stop time may have drift on all three axes.
        stop_candidates = (query[:, None] * self.nominal >= 0).all(-1) & (self.nominal != 0).any(-1)
        distance = distance.masked_fill(
            ~torch.where(stopping[:, None], stop_candidates, eligible_clip), torch.inf
        )
        best_distance, clip = distance.min(-1)
        if clip_hint is not None:
            known = clip_hint >= 0
            hint = clip_hint.clamp_min(0).long()
            hinted = ((goal - self.nominal[hint]).abs().amax(-1) < 1e-5) & ~stopping
            if phase_hint is not None:
                guided = known & (phase_hint >= 0)
                phase = torch.where(guided, phase_hint.long(), phase)
                hinted |= guided
            known &= hinted
            clip = torch.where(known, hint, clip)
            best_distance = torch.where(known, 0.0, best_distance)
        starts = self.windows[clip, phase]
        n = torch.arange(len(goal), device=goal.device)
        index = clip[:, None]
        q = self.positions[index, starts]
        qd = self.velocity[index, starts]
        valid = self.data["wheel_rate_valid"][index, starts]
        dq = ((q - state[4][:, None]) / self.scales[0]).square().mean(-1)
        dv = (qd - state[5][:, None]).square()
        dv_leg = dv[..., :12].mean(-1) / self.scales[1].square()
        dv_wheel = (
            (dv[..., 12:] * valid).sum(-1) / valid.sum(-1).clamp_min(1) / self.scales[2].square()
        )
        costs = (
            dq * self.match_weights[0]
            + dv_leg * self.match_weights[1]
            + dv_wheel * self.match_weights[2]
        )
        chosen = costs.argmin(-1)
        frame = starts[n, chosen]
        matched_distance = best_distance.sqrt()
        distance_limit = self.config["max_command_distance"]
        usable = (matched_distance <= distance_limit) & flat
        state_distance = costs[n, chosen].sqrt()
        attenuation = 1 / (1 + 0.25 * state_distance)
        self.match_metrics = {
            "exact_command_match_fraction": ((matched_distance < 1e-5) & usable & ~stopping)
            .float()
            .mean(),
            "reference_state_distance": (state_distance * usable).sum() / usable.sum().clamp_min(1),
            "phase_counts": torch.bincount(phase[usable], minlength=3),
            "reference_phase_position_error": (dq[n, chosen].sqrt() * usable).sum()
            / usable.sum().clamp_min(1),
            "reference_phase_velocity_error": (dv_leg[n, chosen].sqrt() * usable).sum()
            / usable.sum().clamp_min(1),
            "reference_phase_wheel_error": (dv_wheel[n, chosen].sqrt() * usable).sum()
            / usable.sum().clamp_min(1),
        }
        future = frame[:, None] + self.steps
        validity = usable * (1 - matched_distance.clamp(max=distance_limit) / distance_limit)
        return (
            self.positions[index, future],
            self.velocity[index, future],
            self.data["wheel_rate_valid"][index, future],
            self.weight[clip] * validity * attenuation,
            usable,
            matched_distance,
            self.candidate[clip],
            self.categories[clip],
        )

    def loss(self, states, matched, alive):
        q, qd, wheel_valid, weight, matched_ok, distance, candidate, categories = matched
        pred_q, pred_qd = states[4], states[5]
        dims = (q, qd[..., :12], qd[..., 12:])
        predicted = (pred_q, pred_qd[..., :12], pred_qd[..., 12:])
        live = alive.detach().T
        losses = []
        for i, (prediction, reference, scale) in enumerate(zip(predicted, dims, self.scales)):
            mask = live[..., None]
            if i == 2:
                mask = mask * wheel_valid
            error = F.smooth_l1_loss(prediction / scale, reference / scale, reduction="none")
            per_sample = (error * mask).sum((1, 2)) / mask.expand_as(error).sum((1, 2)).clamp_min(1)
            losses.append((per_sample * weight).sum() / matched_ok.sum().clamp_min(1))
        loss = sum(w * x for w, x in zip(self.config["feature_weights"], losses))
        stats = dict(
            zip(("leg_position", "leg_velocity", "wheel_velocity"), (x.detach() for x in losses))
        )
        stats.update(
            effective_prior_weight=weight.sum() / matched_ok.sum().clamp_min(1),
            coverage=matched_ok.float().mean(),
            valid_reference_window_fraction=matched_ok.float().mean(),
            distance=distance.masked_fill(~matched_ok, 0).sum() / matched_ok.sum().clamp_min(1),
            candidate=(candidate & matched_ok).sum() / matched_ok.sum().clamp_min(1),
            qualified=(~candidate & matched_ok).sum() / matched_ok.sum().clamp_min(1),
            categories=torch.zeros(8, device=categories.device).scatter_add_(
                0, categories, matched_ok.float()
            ),
        )
        return loss, stats

    def reset_state(self, clip, frame, origin, yaw):
        c, s = yaw.cos(), yaw.sin()
        z, o = torch.zeros_like(c), torch.ones_like(c)
        rotation = torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(-1, 3, 3)
        data = self.data
        position = origin.clone()
        position[:, 2] += data["base_pos"][clip, frame, 2]
        quat = mat_to_quat(rotation @ quat_to_mat(data["base_quat"][clip, frame]))
        root = torch.cat(
            (
                position,
                quat,
                matvec(rotation, data["base_lin_vel"][clip, frame]),
                matvec(rotation, data["base_ang_vel"][clip, frame]),
            ),
            -1,
        )
        joint = torch.cat((data["leg_q"][clip, frame], data["wheel_q"][clip, frame]), -1)
        return root, joint, self.velocity[clip, frame]


class ReferenceCollection:
    """CPU or GPU batched, with an episode-fixed control source."""

    def __init__(self, bank, env, cfg, stage):
        self.bank, self.cfg, self.stage = bank, cfg, stage
        self.rng = torch.Generator(device=env.device).manual_seed(env.cfg.seed + 191)
        self.source = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.clip = torch.zeros_like(self.source)
        self.frame = torch.zeros_like(self.source)
        self.source_counts = torch.zeros(4, dtype=torch.long, device=env.device)
        self.end = torch.zeros_like(self.source)
        self.clip_hint = torch.full_like(self.source, -1)
        self.command_level = torch.zeros_like(self.source)
        self.command_skill = torch.full_like(self.source, -1)
        self.review = torch.zeros_like(self.source, dtype=torch.bool)
        self.switch_commands = torch.zeros_like(self.review)
        self.noise = torch.zeros(env.num_envs, 16, device=env.device)
        self.command_steps = torch.zeros_like(self.source)
        self.command_sum = torch.zeros(env.num_envs, 3, device=env.device)
        self.command_error = torch.zeros_like(self.command_sum)
        self.command_tolerance = self.noise.new_tensor([0.05, 0.05, 0.10])
        self.source_skills = torch.zeros(4 * 27, dtype=torch.long, device=env.device)
        self.reference_finished_count = torch.zeros((), dtype=torch.long, device=env.device)
        self.reference_phase_counts = torch.zeros(4, dtype=torch.long, device=env.device)
        self.reference_moving = torch.zeros_like(self.reference_finished_count)
        self.reference_hold = torch.zeros_like(self.reference_finished_count)
        self.reference_action_enabled = (
            any(
                fractions[1] > 0
                for fractions in (
                    cfg.get("warmup_collection", (0, cfg.get("reference_guided_fraction", 0))),
                    cfg.get("early_collection", (0, 0)),
                    cfg.get("late_collection", (0, 0)),
                )
            )
            or cfg.get("local_reference_blend", 0) > 0
        )
        self.reference_length_sum = torch.zeros(4, dtype=torch.long, device=env.device)
        self.reference_episode_count = torch.zeros(4, dtype=torch.long, device=env.device)
        self.level_counts = torch.zeros(10, dtype=torch.long, device=env.device)
        self.recent_snapshot = None
        self.streak = self.source.new_zeros((len(self.source), 7))
        self.category_counts = torch.zeros(8, dtype=torch.long, device=env.device)
        self.signed_counts = torch.zeros(27, dtype=torch.long, device=env.device)
        self.speed_sum = torch.zeros(3, device=env.device)
        self.samples = 0
        self.clipped = torch.zeros((), dtype=torch.long, device=env.device)
        self.active_components = torch.zeros_like(self.clipped)
        self.tracking_sse = torch.zeros(27, 3, dtype=torch.float64, device=env.device)
        self.tracking_count = torch.zeros(27, dtype=torch.long, device=env.device)
        self.ended_count = torch.zeros_like(self.reference_finished_count)
        self.fall_count = torch.zeros_like(self.ended_count)

    def random(self, *shape):
        return torch.rand(shape, device=self.source.device, generator=self.rng)

    def flat_commands(self, term, ids, reference):
        bank, n = self.bank, len(ids)
        mix = self.stage.prior_progress(self.cfg) if self.stage.name == "flat_prior" else 1.0
        early = bank.nominal.new_tensor(EARLY_CATEGORY_WEIGHTS)
        category_prob = early.lerp(bank.nominal.new_tensor(CATEGORY_WEIGHTS), mix)
        if self.stage.name == "flat_autonomous":
            category_prob[7] = self.cfg["autonomous_three_axis_weight"]
        category = torch.multinomial(category_prob, n, replacement=True, generator=self.rng)
        signs = self.random(n, 3) >= 0.5
        active = (category[:, None] & category.new_tensor([1, 2, 4])) != 0
        skills = torch.arange(3, device=ids.device) * 2 + 1 + (~signs).long()
        levels = term.curriculum.flat_levels[ids].gather(1, skills)
        frontier = torch.where(active, levels, 9).amin(-1)
        frontier = torch.where(category == 0, term.curriculum.flat_levels[ids, 0], frontier)
        review = (self.random(n) < term.cfg.curriculum.review_fraction) & (frontier > 0)
        used = torch.where(review, (self.random(n) * frontier).long(), frontier)
        low = bank.nominal.new_tensor(self.cfg["flat_initial_bounds"])
        high = bank.nominal.new_tensor(self.cfg["terrain_bounds"][0])
        cap = torch.lerp(
            torch.where(signs, low[:, 1], -low[:, 0]),
            torch.where(signs, high[:, 1], -high[:, 0]),
            used[:, None] / 9,
        )
        magnitude = torch.minimum(
            cap, cap.new_tensor(self.cfg.get("flat_min_speed", [0.10, 0.05, 0.10]))
        )
        goal = (
            (magnitude + self.random(n, 3) * (cap - magnitude))
            * torch.where(signs, 1.0, -1.0)
            * active
        )
        _, key = command_keys(goal)
        eligible = (key[:, None] == bank.keys) & (bank.nominal.abs() <= cap[:, None] + 1e-6).all(-1)
        above_minimum = ~active[:, None] | (bank.nominal.abs() + 1e-6 >= magnitude[:, None])
        preferred = eligible & above_minimum.all(-1)
        # RSI can use a coupled clip slightly below the continuous sample floor.
        eligible = torch.where(
            preferred.any(-1, keepdim=True), preferred, eligible & reference[:, None]
        )
        selection = self.random(n, len(bank.rows)).masked_fill(~eligible, -1).argmax(-1)
        choose_bank = (
            reference | (self.random(n) < self.stage.bank_fraction(self.cfg))
        ) & eligible.any(-1)
        single = active.sum(-1) == 1
        skill = (skills * active).sum(-1)
        skill = torch.where(single | (category == 0), skill, -1)
        return (
            torch.where(choose_bank[:, None], bank.nominal[selection], goal),
            selection,
            choose_bank,
            used,
            review,
            skill,
        )

    def _assign_source(self, jobs):
        shuffled = jobs[torch.randperm(len(jobs), device=jobs.device, generator=self.rng)]
        live = torch.bincount(self.source, minlength=4)[1:]
        limits = live.new_tensor(
            [int(len(self.source) * w) for w in self.stage.collection_fractions(self.cfg)]
        )
        vacant = (limits - live).clamp_min(0)
        # Among simultaneous resets, distribute available slots without favoring
        # the source that happens to have the shortest episode.
        count = min(len(jobs), int(vacant.sum()))
        if count:
            slots = torch.arange(1, 4, device=jobs.device).repeat_interleave(vacant)
            selected = torch.randperm(len(slots), device=jobs.device, generator=self.rng)[:count]
            self.source[shuffled[:count]] = slots[selected]

    def commands(self, term, mask, *, reset):
        from unisupermimic.tasks.velocity.go2.command_sampler import command_modes

        ids = mask.nonzero().flatten()
        domain = term.command_domain[ids]
        flat_ids = ids[domain == 0]
        if reset:
            self.source[ids] = NORMAL_ACTOR
            self.noise[ids] = 0
            self.switch_commands[ids] = False
            self._assign_source(flat_ids)
        self.command_steps[ids] = 0
        self.command_sum[ids] = 0
        self.command_error[ids] = 0
        goal = term.g[:, :3].clone()
        if len(flat_ids):
            targets, selection, chosen, level, review, skill = self.flat_commands(
                term, flat_ids, self.source[flat_ids] != NORMAL_ACTOR
            )
            self.clip_hint[flat_ids] = torch.where(chosen, selection, -1)
            self.command_level[flat_ids], self.review[flat_ids], self.command_skill[flat_ids] = (
                level,
                review,
                skill,
            )
            goal[flat_ids] = targets
            if reset:
                self.source[flat_ids] *= chosen
                entering = self.source[flat_ids]
                rollout = entering == REFERENCE_ROLLOUT
                allowed = self.bank.categories[selection] == 1
                entering = torch.where(rollout & ~allowed, REFERENCE_LOCAL, entering)
                self.source[flat_ids] = entering
                self.clip[flat_ids] = selection
                idx = (self.random(len(flat_ids)) * self.bank.reset_starts.shape[1]).long()
                frame = torch.where(
                    entering == REFERENCE_LOCAL,
                    self.bank.local_starts[selection, idx],
                    self.bank.reset_starts[selection, idx],
                )
                self.frame[flat_ids] = frame
                length = torch.tensor(self.cfg.get("local_steps", [8, 16]), device=ids.device)
                picked = length[(self.random(len(flat_ids)) * len(length)).long()]
                self.end[flat_ids] = torch.where(
                    entering == REFERENCE_LOCAL, frame + picked, self.bank.positions.shape[1] - 1
                )
                goal[flat_ids] = torch.where(
                    (entering == REFERENCE_ROLLOUT)[:, None],
                    self.bank.data["command"][selection, frame],
                    targets,
                )
        terrain = ids[domain != 0]
        if len(terrain):
            self.clip_hint[terrain] = -1
            self.command_level[terrain] = -1
            self.command_skill[terrain] = -1
            self.review[terrain] = False
            limits = goal.new_tensor(self.cfg["terrain_bounds"])[term.command_domain[terrain]]
            goal[terrain] = torch.lerp(limits[..., 0], limits[..., 1], self.random(len(terrain), 3))
            stairs = term.command_domain[terrain] >= 4
            distance = torch.where(term.command_domain[terrain] >= 6, 2.45, 1.25)
            goal[terrain, 0] = torch.where(
                stairs,
                goal[terrain, 0].clamp_min(distance / (term._env.cfg.max_episode_seconds - 1.5)),
                goal[terrain, 0],
            )
        category, signed = command_keys(goal[ids])
        self.category_counts += torch.bincount(category, minlength=8)
        self.signed_counts += torch.bincount(signed, minlength=27)
        self.level_counts += torch.bincount(self.command_level[flat_ids], minlength=10)
        self.speed_sum += goal[ids].abs().sum(0)
        self.samples += len(ids)
        hold = torch.full_like(term.time_left, term._env.cfg.max_episode_seconds + 0.02)
        if self.stage.name == "flat_autonomous":
            ordinary = flat_ids[self.source[flat_ids] == NORMAL_ACTOR]
            if reset:
                fraction = 0.25 + 0.40 * (1 - self.stage.fraction(self.cfg))
                self.switch_commands[ordinary] = self.random(len(ordinary)) < fraction
            switching = ordinary[self.switch_commands[ordinary]]
            hold[switching] = 2.0 + 3.0 * self.random(len(switching))
        return goal, command_modes(goal), term.command_domain >= 4, hold

    def finished(self, term):
        return (self.source >= REFERENCE_ROLLOUT) & (self.frame + term.ep_step + 1 >= self.end)

    def record_tracking(self, goal, state, live):
        rotation = state[1].transpose(-1, -2)
        actual = torch.cat(
            (matvec(rotation, state[2])[:, :2], matvec(rotation, state[3])[:, 2:3]), -1
        )
        self.command_steps += live
        self.command_sum += actual * live[:, None]
        self.command_error += (actual - goal).square() * live[:, None]

    def skill_quality(self, goal):
        enough = self.command_steps >= self.cfg.get("flat_success_steps", 100)
        steps = self.command_steps[:, None].clamp_min(1)
        rmse = (self.command_error / steps).sqrt()
        mean = self.command_sum / steps
        limit = torch.maximum(self.command_tolerance, 0.35 * goal.abs())
        moving = ((goal == 0) | (mean * goal > 0.4 * goal.square())).all(-1)
        return enough & moving & (rmse <= limit).all(-1) & ((mean - goal).abs() <= limit).all(-1)

    def finish_curriculum(self, term, ids):
        ids = ids[term.curriculum.episode_ended[ids] & (term.command_domain[ids] == 0)]
        eligible = (
            (self.source[ids] == NORMAL_ACTOR)
            & (self.command_skill[ids] >= 0)
            & ~self.review[ids]
            & (self.command_steps[ids] >= 50)
        )
        ids = ids[eligible]
        if not len(ids):
            return
        skill = self.command_skill[ids]
        levels = term.curriculum.flat_levels
        current = levels[ids, skill]
        frontier = self.command_level[ids] == current
        success = self.skill_quality(term.g[:, :3])[ids] & ~term.curriculum.episode_fall[ids]
        old = self.streak[ids, skill]
        count = torch.where(success, old.clamp_min(0) + 1, old.clamp_max(0) - 1)
        delta = (count >= 2).long() - (count <= -3).long()
        levels[ids, skill] = (current + torch.where(frontier, delta, 0)).clamp(0, 9)
        self.streak[ids, skill] = torch.where(frontier & (delta == 0), count, 0)

    def reset_pose(self, term, ids, root, position, velocity):
        subset = ids[self.source[ids] != 0]
        if len(subset):
            yaw = torch.zeros_like(root[subset, 0])
            origin = root[subset, :3].clone()
            origin[:, 2] -= term.cfg.spawn_height
            r, q, qd = self.bank.reset_state(self.clip[subset], self.frame[subset], origin, yaw)
            root[subset], position[subset], velocity[subset] = r, q, qd
        stairs = ids[term.command_domain[ids] >= 4]
        if len(stairs):
            level = term.curriculum.mastery[stairs]
            bounds = self.cfg["stair_yaw_degrees"]
            band = (level >= 2).long() + (level >= 4).long() + (level >= 7).long()
            theta = (
                (2 * self.random(len(stairs)) - 1)
                * root.new_tensor(bounds)[band]
                * (torch.pi / 180)
            )
            root[stairs, 3], root[stairs, 6] = (theta / 2).cos(), (theta / 2).sin()

    def advance_command(self, term, live):
        frame = (self.frame + term.ep_step.long()).clamp_max(self.bank.positions.shape[1] - 1)
        term.g[:, :3].copy_(
            torch.where(
                ((self.source == REFERENCE_ROLLOUT) & live)[:, None],
                self.bank.data["command"][self.clip, frame],
                term.g[:, :3],
            )
        )

    def tags(self, term):
        frame = (self.frame + term.ep_step.long()).clamp_max(self.bank.positions.shape[1] - 1)
        phase = self.bank.frame_phase[self.clip, frame]
        return dict(
            collection_source=self.source.to(torch.int8),
            reference_phase=torch.where(self.source >= REFERENCE_ROLLOUT, phase, -1).to(torch.int8),
            reference_clip=self.clip_hint.to(torch.int16),
            command_speed_level=self.command_level.clone(),
            command_skill=self.command_skill.clone(),
            command_is_review=self.review.clone(),
        )

    def action(self, proposed, command, mixed):
        frame = self.frame + command.ep_step.long() + 1
        rollout, local = self.source == REFERENCE_ROLLOUT, self.source == REFERENCE_LOCAL
        guide = rollout | local
        self.source_counts += torch.bincount(self.source, minlength=4)
        self.reference_hold += (guide & (frame > self.end)).sum()
        frame = frame.clamp_max(self.bank.positions.shape[1] - 1)
        data = self.bank.data
        phase = self.bank.frame_phase[self.clip, frame].long()
        self.reference_phase_counts += torch.bincount(torch.where(guide, phase, 3), minlength=4)
        self.reference_moving += (guide & (data["command"][self.clip, frame] != 0).any(-1)).sum()
        action = proposed
        if self.reference_action_enabled:
            ref, clipped = reference_action(
                data["leg_q"][self.clip, frame],
                data["wheel_qd"][self.clip, frame],
                mixed.cfg.control,
                mixed.default,
                mixed.limits,
            )
            blend = torch.where(
                rollout, 1.0, self.cfg.get("local_reference_blend", 0) * local.float()
            )
            action = proposed.lerp(ref, blend[:, None])
            self.clipped += (clipped & rollout[:, None]).sum()
        correlation = self.cfg.get("local_correlation", 0.8)
        sigma = proposed.new_tensor(
            [self.cfg.get("local_leg_std", 0.01)] * 12 + [self.cfg.get("local_wheel_std", 0.03)] * 4
        )
        self.noise.mul_(correlation).add_(
            torch.randn(self.noise.shape, device=proposed.device, generator=self.rng) * sigma
        )
        action = action + self.noise * local[:, None]
        self.clipped += ((action.abs() > 1) & guide[:, None]).sum()
        self.active_components += 16 * guide.sum()
        return action.clamp(-1, 1), guide

    def observe(self, transition, state, physical_fall):
        rotation = state[1].transpose(-1, -2)
        local = torch.cat(
            (matvec(rotation, state[2])[..., :2], matvec(rotation, state[3])[..., 2:3]), -1
        )
        goal = transition["g"][..., :3]
        _, signed = command_keys(goal)
        self.tracking_sse.index_add_(0, signed, (local - goal).square().double())
        self.tracking_count.index_add_(0, signed, torch.ones_like(signed))
        source = transition["collection_source"].long()
        self.source_skills += torch.bincount(source * 27 + signed, minlength=108)
        ended = transition["done"]
        self.reference_episode_count += torch.bincount(source[ended], minlength=4)
        self.reference_length_sum.scatter_add_(
            0, source, (transition["ep_step"].long() + 1) * ended
        )
        self.reference_finished_count += transition["reference_finished"].sum()
        self.ended_count += ended.sum()
        self.fall_count += physical_fall.sum()

    def metrics(self):
        counts = (
            self.source_counts,
            self.tracking_sse,
            self.tracking_count,
            self.ended_count,
            self.fall_count,
        )
        recent = (
            tuple(t.clone() for t in counts)
            if self.recent_snapshot is None
            else tuple(a - b for a, b in zip(counts, self.recent_snapshot))
        )
        self.recent_snapshot = tuple(t.clone() for t in counts)
        result = {
            "recent_source_fractions": (recent[0] / recent[0].sum().clamp_min(1)).tolist(),
            "recent_tracking_rmse_by_sign_key": (recent[1] / recent[2][:, None].clamp_min(1))
            .clamp_min(0)
            .sqrt()
            .tolist(),
            "recent_fall_rate": float(recent[4] / recent[3].clamp_min(1)),
            "active_source_fractions": (
                torch.bincount(self.source, minlength=4) / len(self.source)
            ).tolist(),
            "source_by_signed_command": self.source_skills.view(4, 27).tolist(),
            "reference_finished_count": int(self.reference_finished_count),
            "active_reference_motion_fraction": float(
                self.reference_moving / self.reference_phase_counts[:3].sum().clamp_min(1)
            ),
            "reference_terminal_hold_fraction": float(
                self.reference_hold / self.reference_phase_counts[:3].sum().clamp_min(1)
            ),
            "source_episode_length": (
                self.reference_length_sum / self.reference_episode_count.clamp_min(1)
            ).tolist(),
            "speed_level_counts": self.level_counts.tolist(),
            "source_fractions": (
                self.source_counts / self.source_counts.sum().clamp_min(1)
            ).tolist(),
            "command_categories": self.category_counts.tolist(),
            "signed_commands": self.signed_counts.tolist(),
            "real_tracking_rmse_by_sign_key": (
                self.tracking_sse / self.tracking_count[:, None].clamp_min(1)
            )
            .sqrt()
            .tolist(),
            "real_tracking_transition_count_by_sign_key": self.tracking_count.tolist(),
            "real_fall_rate_at_episode_end": float(self.fall_count / self.ended_count.clamp_min(1)),
            "mean_abs_command": (self.speed_sum / max(1, self.samples)).tolist(),
            "reference_action_clipped_fraction": float(
                self.clipped / self.active_components.clamp_min(1)
            ),
        }
        return result


def apply_stage(learner, stage, config, *, reset=False):
    """Change allocation at an iteration boundary while retaining replay and optimizers."""
    term = learner.env.command_manager.get_term("velocity")
    learner.motion_prior_weight = (
        config["lambda_prior"] * stage.prior_fraction(config) if config["enabled"] else 0.0
    )
    learner.actor_domain_weights = (
        config["terrain_actor_weights"] if stage.name == "terrain" else FLAT_DOMAINS
    )
    learner.buffer.actor_three_axis_weight = (
        config["autonomous_three_axis_weight"] if stage.name == "flat_autonomous" else 0.0
    )
    learner.buffer.wm_reference_bias = (
        (1.0 if stage.iteration < stage.wm_only_iters else 0.5 * (1 - stage.prior_progress(config)))
        if stage.name == "flat_prior"
        else 0.0
    )
    learner.buffer.actor_ordinary_bias = 2.0 * (
        stage.prior_progress(config) if stage.name == "flat_prior" else 1.0
    )
    learner.buffer.actor_review_fraction = 0.20
    if reset:
        if stage.name == "terrain":
            weights = (
                config["terrain_entry_weights"]
                if stage.terrain_review
                else config["terrain_weights"]
            )
        else:
            weights = FLAT_DOMAINS
        previous_domain = term.command_domain.clone()
        term.curriculum.reassign_domains(weights, coverage_fraction=0.0)
        learner.buffer.mark_reset_boundary(
            torch.ones(learner.env.num_envs, device=learner.device, dtype=torch.bool)
        )
        if getattr(term, "reference_collection", None) is not None:
            term.reference_collection.streak[previous_domain != term.command_domain] = 0
        learner.env.reset()
        learner.obs_hist.zero_()
