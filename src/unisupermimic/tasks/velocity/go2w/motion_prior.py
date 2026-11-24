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
CATEGORY_WEIGHTS = (0.05, 0.20, 0.25, 0.10, 0.20, 0.10, 0.10, 0.0)
FLAT_DOMAINS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def command_keys(goal):
    """Bit pattern of active axes and ternary code of their signs."""
    sign = torch.where(goal.abs() >= 0.025, goal.sign(), 0).long()
    category = (sign.abs() * sign.new_tensor([1, 2, 4])).sum(-1)
    signed = ((sign + 1) * sign.new_tensor([1, 3, 9])).sum(-1)
    return category, signed


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
        phase_starts, reset_starts = [], []
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
        self.scales = self.nominal.new_tensor(cfg["feature_scales"])
        self.match_weights = self.nominal.new_tensor(cfg["matching_weights"])
        self.command_scale = self.nominal.new_tensor(cfg["command_scale"])
        self.steps = torch.arange(1, self.horizon + 1, device=device)
        self.positions = self.data["leg_q"]
        self.velocity = torch.cat((self.data["leg_qd"], self.data["wheel_qd"]), -1)

    @torch.no_grad()
    def match(self, state, goal, flat):
        _, key = command_keys(goal)
        distance = ((goal[:, None] - self.nominal) / self.command_scale).square().sum(-1)
        distance = distance.masked_fill(key[:, None] != self.keys, torch.inf)
        distances, clips = torch.topk(
            distance, min(self.config["command_neighbors"], len(self.rows)), largest=False, dim=1
        )
        starts = self.phase_starts[clips]
        index = clips[..., None]
        q = self.positions[index, starts]
        qd = self.velocity[index, starts]
        valid = self.data["wheel_rate_valid"][index, starts]
        dq = ((q - state[4][:, None, None]) / self.scales[0]).square().mean(-1)
        dv = (qd - state[5][:, None, None]).square()
        dv_leg = dv[..., :12].mean(-1) / self.scales[1].square()
        dv_wheel = (
            (dv[..., 12:] * valid).sum(-1) / valid.sum(-1).clamp_min(1) / self.scales[2].square()
        )
        phase_cost = (
            self.match_weights[0] * dq
            + self.match_weights[1] * dv_leg
            + self.match_weights[2] * dv_wheel
        )
        distance_limit = self.config["max_command_distance"]
        usable = distances <= distance_limit**2
        cost = (phase_cost + 3 * distances[..., None]).masked_fill(~usable[..., None], torch.inf)
        best = cost.flatten(1).argmin(-1)
        n = torch.arange(len(goal), device=goal.device)
        k, j = best // starts.shape[-1], best % starts.shape[-1]
        clip, frame = clips[n, k], starts[n, k, j]
        valid_match = usable[n, k] & flat
        matched_distance = distances[n, k].sqrt()
        validity = valid_match * (1 - matched_distance.clamp(max=distance_limit) / distance_limit)
        future = frame[:, None] + self.steps
        return (
            self.positions[clip[:, None], future],
            self.velocity[clip[:, None], future],
            self.data["wheel_rate_valid"][clip[:, None], future],
            self.weight[clip] * validity,
            valid_match,
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
        self.source_counts = torch.zeros(3, device=env.device)
        self.category_counts = torch.zeros(8, device=env.device)
        self.signed_counts = torch.zeros(27, device=env.device)
        self.speed_sum = torch.zeros(3, device=env.device)
        self.samples = 0
        self.clipped = torch.zeros((), device=env.device)
        self.active_components = torch.zeros_like(self.clipped)
        self.tracking_sse = torch.zeros(27, 3, device=env.device)
        self.tracking_count = torch.zeros(27, device=env.device)
        self.ended_count = torch.zeros((), device=env.device)
        self.fall_count = torch.zeros_like(self.ended_count)

    def random(self, *shape):
        return torch.rand(shape, device=self.source.device, generator=self.rng)

    def flat_commands(self, term, ids, reference):
        bank, n = self.bank, len(ids)
        category_prob = bank.nominal.new_tensor(CATEGORY_WEIGHTS)
        if self.stage.name == "flat_autonomous":
            category_prob[7] = self.cfg["autonomous_three_axis_weight"]
        category = torch.multinomial(category_prob, n, replacement=True, generator=self.rng)
        level = term.curriculum.flat_levels[ids]
        low = bank.nominal.new_tensor(self.cfg["flat_initial_bounds"])
        high = bank.nominal.new_tensor(self.cfg["terrain_bounds"][0])
        signs = self.random(n, 3) >= 0.5
        skill = torch.arange(3, device=ids.device) * 2 + 1
        approach = level.gather(1, skill + (~signs).long()) / 9
        review = self.random(n, 1) < term.cfg.curriculum.review_fraction
        approach = approach * torch.where(review, self.random(n, 1), 1.0)
        cap = torch.lerp(
            torch.where(signs, low[:, 1], -low[:, 0]),
            torch.where(signs, high[:, 1], -high[:, 0]),
            approach,
        )
        active = (category[:, None] & category.new_tensor([1, 2, 4])) != 0
        goal = (0.04 + self.random(n, 3) * (cap - 0.04)) * torch.where(signs, 1, -1) * active
        # Pure directions and coupled mirror groups are equiprobable within a category.
        allowed_category = category == 7
        candidates = (category[:, None] == bank.categories) | (
            allowed_category[:, None] & (bank.categories == 3)
        )
        clip_signs = bank.nominal >= 0
        clip_active = bank.nominal != 0
        sign_mismatch = ((clip_signs[None] != signs[:, None]) & clip_active).sum(-1)
        lo_by_sign = (
            level[:, None]
            .expand(-1, len(bank.rows), -1)
            .gather(2, (skill + (~clip_signs).long())[None].expand(n, -1, -1))
            / 9
        )
        lo_by_sign = lo_by_sign * torch.where(review, 0.5, 1.0)[:, None]
        clip_cap = torch.lerp(
            torch.where(clip_signs, low[:, 1], -low[:, 0]),
            torch.where(clip_signs, high[:, 1], -high[:, 0]),
            lo_by_sign,
        )
        eligible = candidates & (bank.nominal.abs() <= clip_cap + 1e-6).all(-1)
        priority = sign_mismatch.float() + self.random(n, len(bank.rows))
        priority = priority.masked_fill(~eligible, torch.inf)
        selection = priority.argmin(-1)
        choose_bank = (
            reference | (self.random(n) < self.stage.bank_fraction(self.cfg))
        ) & ~allowed_category
        choose_bank &= eligible.any(-1)
        return (
            torch.where(choose_bank[:, None], bank.nominal[selection], goal),
            selection,
            choose_bank,
        )

    def commands(self, term, mask, *, reset):
        from unisupermimic.tasks.velocity.go2.command_sampler import command_modes

        ids = mask.nonzero().flatten()
        n = len(ids)
        domain = term.command_domain[ids]
        guided, rsi = self.stage.collection_fractions(self.cfg)
        if reset:
            self.source[ids] = 0
            # Cap the number of concurrently active reference episodes. Sampling by
            # episode probability alone overrepresents sources that survive longer.
            jobs = ids[domain == 0]
            order = jobs[torch.randperm(len(jobs), device=ids.device, generator=self.rng)]
            live_count = torch.bincount(self.source, minlength=3)
            limits = (int(len(self.source) * rsi), int(len(self.source) * guided))
            take = (live_count[1:].new_tensor(limits) - live_count[1:]).clamp_min(0)
            rank = torch.arange(len(jobs), device=ids.device)
            source = (rank < take.sum()).long() + (rank < take[1]).long()
            self.source[order] = source
        flat_ids = ids[domain == 0]
        goal = term.g[:, :3].clone()
        if len(flat_ids):
            flat_goal, selection, chosen = self.flat_commands(
                term, flat_ids, self.source[flat_ids] != 0
            )
            goal[flat_ids] = flat_goal
            if reset:
                self.source[flat_ids] *= chosen
                self.clip[flat_ids] = selection
                frame_idx = (self.random(len(flat_ids)) * self.bank.reset_starts.shape[1]).long()
                self.frame[flat_ids] = self.bank.reset_starts[selection, frame_idx]
        terrain = ids[domain != 0]
        if len(terrain):
            limits = goal.new_tensor(self.cfg["terrain_bounds"])[term.command_domain[terrain]]
            goal[terrain] = torch.lerp(limits[..., 0], limits[..., 1], self.random(len(terrain), 3))
            # At the slowest stair command the final wheel needs ~16 s to clear
            # the 2 m target. Use each tile's time-feasible portion of the band.
            stairs = term.command_domain[terrain] >= 4
            distance = torch.where(term.command_domain[terrain] >= 6, 2.45, 1.25)
            goal[terrain, 0] = torch.where(
                stairs,
                goal[terrain, 0].clamp_min(distance / (term._env.cfg.max_episode_seconds - 1.5)),
                goal[terrain, 0],
            )
        category, signed = command_keys(goal[ids])
        self.category_counts.scatter_add_(
            0, category, torch.ones_like(category, dtype=torch.float32)
        )
        self.signed_counts.scatter_add_(0, signed, torch.ones_like(signed, dtype=torch.float32))
        self.speed_sum += goal[ids].abs().sum(0)
        self.samples += n
        hold = torch.full_like(term.time_left, term._env.cfg.max_episode_seconds + 0.02)
        if self.stage.name == "flat_autonomous":
            ordinary = flat_ids[self.source[flat_ids] == 0]
            hold[ordinary] = 2.0 + 3.0 * self.random(len(ordinary))
        return goal, command_modes(goal), term.command_domain >= 4, hold

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
        frame = (self.frame + term.ep_step.long()).clamp(max=self.bank.positions.shape[1] - 1)
        term.g[:, :3].copy_(
            torch.where(
                ((self.source == 2) & live)[:, None],
                self.bank.data["command"][self.clip, frame],
                term.g[:, :3],
            )
        )

    def action(self, proposed, command, mixed):
        frame = self.frame + command.ep_step.long() + 1
        guide = self.source == 2
        self.source_counts.scatter_add_(0, self.source, torch.ones_like(frame, dtype=torch.float32))
        # The final reference sample is a standing hold. There is no cyclic jump.
        frame = frame.clamp(max=self.bank.positions.shape[1] - 1)
        data = self.bank.data
        q, qd = data["leg_q"][self.clip, frame], data["wheel_qd"][self.clip, frame]
        action, clipped = reference_action(q, qd, mixed.cfg.control, mixed.default, mixed.limits)
        self.clipped += (clipped & guide[:, None]).sum()
        self.active_components += 16 * guide.sum()
        return torch.where(guide[:, None], action, proposed), guide

    def observe(self, transition, state):
        rotation = state[1].transpose(-1, -2)
        local = torch.cat(
            (matvec(rotation, state[2])[..., :2], matvec(rotation, state[3])[..., 2:3]), -1
        )
        goal = transition["g"][..., :3]
        _, signed = command_keys(goal)
        self.tracking_sse.index_add_(0, signed, (local - goal).square())
        self.tracking_count.index_add_(0, signed, torch.ones_like(signed, dtype=torch.float32))
        self.ended_count += transition["done"].sum()
        self.fall_count += transition["term"].sum()

    def metrics(self):
        result = {
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
        config["lambda_prior"] * stage.fraction(config) if config["enabled"] else 0.0
    )
    learner.actor_domain_weights = (
        config["terrain_actor_weights"] if stage.name == "terrain" else FLAT_DOMAINS
    )
    learner.buffer.actor_three_axis_weight = (
        config["autonomous_three_axis_weight"] if stage.name == "flat_autonomous" else 0.0
    )
    if reset:
        if stage.name == "terrain":
            weights = (
                config["terrain_entry_weights"]
                if stage.terrain_review
                else config["terrain_weights"]
            )
            coverage = 0.05 if stage.terrain_review else 0.125
        else:
            weights, coverage = FLAT_DOMAINS, 0.0
        term.curriculum.reassign_domains(weights, coverage)
        learner.buffer.mark_reset_boundary(
            torch.ones(learner.env.num_envs, device=learner.device, dtype=torch.bool)
        )
        learner.env.reset()
        learner.obs_hist.zero_()
