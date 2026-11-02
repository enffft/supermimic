"""Read-only CPU evidence for the 2026-11-02 M1 audit; no training or mj_step.

Requires the original AMP checkpoint, merged bank, and existing window cache.
Run with the supermimic Conda interpreter and PYTHONPATH=src; see audit report.
Only the explicitly requested JSON output is written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amp-root', type=Path, default=Path.home() / 'AMP-LOCO-M1')
    parser.add_argument('--cache', type=Path, default=Path('/tmp/m1_cvae_window_cache_v2/cvae_v1_strict_fcl_2272'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    sys.path.insert(0, str(args.amp_root))
    from supermimic.motion_prior.checkpoint import load_m1_motion_prior
    from supermimic.motion_prior.features import state_history_to_cvae53
    from supermimic.motion_prior.reference import decode_future31
    from supermimic.robots.spec import get_robot_spec, M1_AMP_JOINTS, M1_JOINTS
    from supermimic.kinematics.rotations import quat_apply
    from motion_prior.cvae_features import encode_state_history, encode_future
    from motion_prior.cvae_inference import load_cvae_checkpoint
    from tools.motion_generation.m1_kinematics import URDFModel
    from tools.motion_generation.m1_task_motion_schema import _quat_xyzw_to_matrix

    repo = Path(__file__).resolve().parents[1]
    artifact = args.amp_root / 'logs/motion_prior/cvae_v1_strict_fcl_2272'
    checkpoint = artifact / 'checkpoint_best_val.pt'
    prior = load_m1_motion_prior(checkpoint, device='cpu')
    spec = get_robot_spec('m1_piper')
    amp_ids = [M1_JOINTS.index(n) for n in M1_AMP_JOINTS]
    wheel_ids = [i for i, n in enumerate(M1_JOINTS) if n.endswith('FOOT_JOINT')]
    result = {
        'python': sys.executable, 'torch_device': 'cpu', 'simulation_steps': 0,
        'git': {str(p): subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()
                for p in (repo, args.amp_root)},
        'checkpoint_sha256': sha256(checkpoint), 'urdf_sha256': sha256(spec.model_path),
        'cache': str(args.cache), 'sampling_seed': 20261102,
        'caveat': 'Synthetic histories and cached kinematic windows, not recorded real run states. No physical Jacobian validation.',
    }
    history = torch.zeros(1, 4, 57)
    history[:, :, 2] = .334
    history[:, :, 3] = 1
    history[:, :, 7:29] = torch.tensor(spec.default_joint_pos)
    goals = torch.tensor([[.3, 0, 0, 0, 0, 0, 1, 0], [-.3, 0, 0, 0, 0, 0, 1, 0]])

    def sweep(model):
        rows = []
        for feature, index in [('q6_rad', 28), ('qd6_rad_s', 56)]:
            for value in (-.1, -1e-2, -1e-3, -1e-4, -1e-5, 0., 1e-5, 1e-4, 1e-3, 1e-2, .1):
                h = history.expand(2, -1, -1).clone()
                h[:, :, index] = value
                cond = torch.cat((state_history_to_cvae53(h).flatten(1), goals), dim=-1)
                with torch.no_grad():
                    y = model(h, goals)
                rows.append({'feature': feature, 'value': value,
                             'max_normalized_input': float(((cond-model.x_mean)/model.x_std).abs().max()),
                             'forward_dx_m': float(y[0, -1, 0]), 'backward_dx_m': float(y[1, -1, 0]),
                             'command_output_mean_abs_difference': float((y[0]-y[1]).abs().mean()),
                             'forward_mean_wheel_rates': y[0, :, 27:31].mean(0).tolist()})
        return rows

    result['normalization'] = {'q6_std': float(prior.x_std[189]), 'qd6_std': float(prior.x_std[207]),
                               'tiny_std_indices': torch.where(prior.x_std <= 1.01e-6)[0].tolist()}
    result['source_prior_sweep'] = sweep(prior)
    run = repo / 'logs/m1_prior_actor/m1_prior_L_2048_30min_ctrl_v2_20261102/ckpt_latest.pt'
    if run.exists():
        saved = torch.load(run, map_location='cpu', weights_only=False)
        updated = copy.deepcopy(prior)
        updated.load_state_dict(saved['motion_prior'])
        result['L_final_prior_sweep'] = sweep(updated)
        result['L_checkpoint'] = {k: saved[k] for k in ('iteration', 'wm_optimizer_steps', 'actor_optimizer_steps', 'prior_optimizer_steps')}
        from supermimic.policy.nets import M1PriorActorCfg, M1PriorConditionedActor
        from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
        actor = M1PriorConditionedActor(M1PriorActorCfg(**saved['actor_schema']))
        actor.load_state_dict(saved['actor'])
        obs = analytic_observation(history[:, -1], torch.zeros(1, 22), torch.zeros(1, 26),
                                   torch.tensor(spec.default_joint_pos), tuple(wheel_ids))
        obs = flatten_history_term_major(obs[None].expand(actor.cfg.history, -1, -1), num_joints=22)
        with torch.no_grad():
            ref = updated(history, goals[:1])
            actions = actor(obs.expand(10, -1), goals[:1].expand(10, -1), ref.expand(10, -1, -1), torch.arange(10))
        result['L_final_actor_synthetic_default_history'] = {
            'q6_target_by_phase_rad': (actions[:, 21]*spec.action_scales[21]).tolist(),
            'mean_wheel_target_rad_s': (actions[:, wheel_ids]*5).mean(0).tolist(),
            'caveat': 'Offline actor outputs only; these are setpoints, not measured physical states.',
        }
        del saved, updated

    x, y, meta, mask = [np.load(args.cache / (n+'.npy'), mmap_mode='r') for n in ('x', 'y', 'meta', 'wheel_mask')]
    result['cache_shapes'] = {n: list(a.shape) for n, a in [('x', x), ('y', y), ('meta', meta), ('mask', mask)]}
    rng = np.random.default_rng(20261102)
    defaults = np.asarray(spec.default_joint_pos)[amp_ids]
    scales = np.asarray(spec.action_scales)[amp_ids]
    result['action_envelope'] = {'joint_names': list(M1_AMP_JOINTS), 'lower': (defaults-scales).tolist(), 'upper': (defaults+scales).tolist()}
    result['dataset'] = {}
    for code, task in enumerate(('L', 'M', 'LM')):
        indices = np.flatnonzero(meta[:, 2] == code)
        sample = rng.choice(indices, min(2048, len(indices)), replace=False)
        targets = np.asarray(y[sample])
        valid = np.asarray(mask[sample]).astype(bool)
        q = targets[:, :, 9:27]
        wheel = np.abs(targets[:, :, 27:31])
        stats = {'windows': len(indices), 'sampled_windows': len(sample),
                 'arm_target_min': q[:, :, 12:].min((0, 1)).tolist(), 'arm_target_max': q[:, :, 12:].max((0, 1)).tolist(),
                 'outside_action_setpoint_fraction': ((q < defaults-scales) | (q > defaults+scales)).mean((0, 1)).tolist(),
                 'wheel_abs_quantiles_50_90_99': np.quantile(wheel, [.5, .9, .99]).tolist(),
                 'wheel_supervision_valid_fraction': float(valid.mean()),
                 'valid_wheel_abs_quantiles_50_90_99': np.quantile(wheel[valid], [.5, .9, .99]).tolist() if valid.any() else None}
        # All windows, in bounded chunks; the target/envelope stats above are a sample.
        max_diff = np.zeros(3)
        goal_min, goal_max = np.full(3, np.inf), np.full(3, -np.inf)
        q6_max, qd6_max = 0., 0.
        for start in range(0, len(indices), 8192):
            rows = np.asarray(x[indices[start:start+8192]])
            if task == 'L':
                max_diff = np.maximum(max_diff, np.abs(rows[:, 212:215]-rows[:, [166, 167, 171]]).max(0))
                goal_min = np.minimum(goal_min, rows[:, 212:215].min(0))
                goal_max = np.maximum(goal_max, rows[:, 212:215].max(0))
            q6_max = max(q6_max, float(np.abs(rows[:, [30, 83, 136, 189]]).max()))
            qd6_max = max(qd6_max, float(np.abs(rows[:, [48, 101, 154, 207]]).max()))
        stats.update(history_q6_abs_max=q6_max, history_qd6_abs_max=qd6_max)
        if task == 'L':
            stats.update(all_window_goal_current_velocity_max_abs_difference=max_diff.tolist(),
                         all_window_goal_min=goal_min.tolist(), all_window_goal_max=goal_max.tolist())
        result['dataset'][task] = stats

    # Real source-frame adapter/decode/FK checks, matched by original file hash.
    bank = args.amp_root / 'dataset/m1_kinematic_motion_bank_v3_2_combined'
    manifest = json.loads((bank/'manifest.json').read_text())
    audit = json.loads((artifact/'dataset_audit.json').read_text())
    merged_by_hash = {r.get('merged_motion_sha256', r.get('source_motion_sha256')): r
                      for r in manifest['motions']}
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.mjlab_assets import load_robot_spec_model
    fk = G1Kinematics(load_robot_spec_model('m1_piper'), device='cpu')
    urdf = URDFModel(spec.model_path)
    source_predictor = load_cvae_checkpoint(checkpoint, device='cpu')
    parity = []
    for task in ('L', 'M', 'LM'):
        old_rows = [r for r in audit['motions'] if r['task_type'] == task]
        for old in (old_rows[0], old_rows[len(old_rows)//2], old_rows[-1]):
            merged = merged_by_hash[old['motion_sha256']]
            motion_path = bank / merged['motion_file']
            assert sha256(motion_path) == old['motion_sha256']
            frames = np.asarray(json.loads(motion_path.read_text())['Frames'], dtype=np.float32)
            labels = np.load(artifact/'derived_labels'/old['derived_label_file'])
            t = max(3, min(len(frames)//2, len(frames)-11))
            raw = frames[t-3:t+11]
            wheels = labels['wheel_angular_velocities_rad_s'][t-3:t+11]
            states = torch.zeros(len(raw), 57)
            states[:, :3] = torch.from_numpy(raw[:, :3])
            states[:, 3:7] = torch.from_numpy(raw[:, [6, 3, 4, 5]])
            states[:, np.array(amp_ids)+7] = torch.from_numpy(raw[:, 7:25])
            states[:, 29:32] = quat_apply(states[:, 3:7], torch.from_numpy(raw[:, 40:43]))
            states[:, 32:35] = torch.from_numpy(raw[:, 43:46])
            states[:, np.array(amp_ids)+35] = torch.from_numpy(raw[:, 46:64])
            states[:, np.array(wheel_ids)+35] = torch.from_numpy(wheels)
            h = states[:4][None]
            source_features = torch.from_numpy(encode_state_history(raw[:4], wheels[:4]))[None]
            features = state_history_to_cvae53(h)
            g = torch.tensor([[.2, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)
            target = torch.from_numpy(encode_future(raw[3], raw[4:], wheels[4:]))[None]
            decoded = decode_future31(states[3:4], target)[0]
            body = fk(states[:, :29], states[:, 29:])['pos']
            fk_error = []
            for i, frame in enumerate(raw):
                joints = dict(zip(M1_AMP_JOINTS, frame[7:25]))
                rotation = _quat_xyzw_to_matrix(frame[3:7])
                for name in ('piper_x_link7', 'FAR_FOOT_LINK', 'FBL_FOOT_LINK', 'RAR_FOOT_LINK', 'RBL_FOOT_LINK'):
                    p = frame[:3] + rotation @ urdf.link_transform(name, joints)[1]
                    fk_error.append(float(np.max(np.abs(body[i, fk.body_index(name)].numpy()-p))))
            with torch.no_grad():
                direct = prior(h, g)
                source = source_predictor.predict(source_features, g)
                source_same_features = source_predictor.predict(features, g)
            feature_diff = (features-source_features).flatten(1)
            norm_diff = (feature_diff/prior.x_std[:212]).abs()
            parity.append({'task': task, 'old_id': old['motion_id'], 'merged_id': merged['motion_id'],
                           'source_file_sha256': old['motion_sha256'], 'feature_max_abs_error': float((features-source_features).abs().max()),
                           'prior_output_max_abs_error': float((direct-source).abs().max()),
                           'identical_feature_model_output_max_abs_error': float((direct-source_same_features).abs().max()),
                           'normalized_feature_max_abs_error': float(norm_diff.max()),
                           'largest_normalized_feature_error_index': int(norm_diff.argmax()),
                           'decoded_root_position_max_abs_error_m': float((decoded[:, :3]-states[4:, :3]).abs().max()),
                           'decoded_joint_position_max_abs_error_rad': float((decoded[:, np.array(amp_ids)+7]-states[4:, np.array(amp_ids)+7]).abs().max()),
                           'source_urdf_vs_runtime_fk_max_abs_error_m': max(fk_error)})
    result['source_frame_parity'] = parity
    directions = {}
    for i in wheel_ids:
        name = M1_JOINTS[i]
        link = name.replace('_JOINT', '_LINK')
        r, _ = urdf.link_transform(link, dict(zip(M1_JOINTS, spec.default_joint_pos)))
        axis = r @ urdf.joints[name].axis
        d = np.cross(axis, [0., 0., 1.])
        directions[name] = (d/np.linalg.norm(d)).tolist()
    result['wheel_positive_rolling_directions_at_default'] = directions
    result['wheel_radius_m'] = urdf.wheel_radius
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    result['last_logged_wm_components'] = {}
    for task in ('L', 'M'):
        tb = repo / f'logs/m1_prior_actor/m1_prior_{task}_2048_30min_ctrl_v2_20261102/tb'
        if tb.exists():
            events = EventAccumulator(str(tb))
            events.Reload()
            metrics = {}
            for tag in events.Tags()['scalars']:
                if 'nse_8step/' in tag:
                    event = events.Scalars(tag)[-1]
                    metrics[tag] = {'step': event.step, 'value': event.value}
            result['last_logged_wm_components'][task] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'parity_rows': len(parity), 'simulation_steps': 0}))


if __name__ == '__main__':
    main()
