"""Fixed-seed real-simulation evaluation of prior-free M1 checkpoints."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import tyro

from play import _load_checkpoint, _M1RewardDictPolicy
from supermimic.envs import make_env
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.motion_prior.velocity import velocity_command_errors
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.tasks.world_point_reward import tcp_position_world, m1_tcp_proxy_indices
from supermimic.training.m1_reward import evaluation_args
from supermimic.tasks.m1_reward import M1RewardCfg


def main(checkpoint: Path, out: Path, task_type: Literal['balanced', 'L', 'M'] = 'balanced',
         num_envs: int = 16, seconds: float = 20., seed: int = 123,
         vx: float | None = None, vy: float = 0., yaw_rate: float = 0.,
         target_world: tuple[float, float, float] | None = None, device: str = 'cuda:0'):
    if seconds <= 0 or num_envs < 1 or (task_type == 'balanced' and (num_envs < 2 or num_envs % 2)):
        raise ValueError('positive seconds/envs; balanced requires an even env count >= 2')
    policy, config, _ = _load_checkpoint(checkpoint, device)
    env = make_env(evaluation_args(config, num_envs=num_envs, task_type=task_type, seed=seed), device=device)
    fk = G1Kinematics(load_robot_spec_model('m1_piper'), device=device)
    tcp_ids = m1_tcp_proxy_indices(fk)
    pol = _M1RewardDictPolicy(policy, env, None if vx is None else (vx, vy, yaw_rate), target_world)
    continuous = env.cfg.commands["m1_prior_joint"].target_cfg.resample_on_success
    sums, counts = {}, {}
    goal_rows = []
    goal_age = torch.zeros(num_envs, device=device)
    goal_initial_distance = torch.full((num_envs,), float("nan"), device=device)
    hold = torch.zeros(num_envs, device=device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_rows = []
    def record(name, value):
        sums[name] = sums.get(name, 0.) + float(value.sum())
        counts[name] = counts.get(name, 0) + value.numel()
    try:
        obs, _ = env.reset()
        for _ in range(math.ceil(seconds / env.step_dt)):
            with torch.no_grad():
                action = pol(obs)
                command = env.command_manager.get_term("m1_prior_joint")
                start_state = pol.state_io.read()
                start_distance = (tcp_position_world(start_state, fk, tcp_ids) - command.target_position_world).norm(dim=-1)
                goal_initial_distance = torch.where(goal_age == 0, start_distance, goal_initial_distance)
            obs, _, term, tout, _ = env.step(action)
            # RewardManager runs before auto-reset. This cache keeps terminal state
            # and its original command; never score the reset pose as the endpoint.
            cache = env._supermimic_m1_reward_cache
            state, context = cache['state_next'], cache['context']
            is_l, is_m = context[:, 0] < .5, context[:, 0] >= .5
            xy, xy_sq, yaw, _ = velocity_command_errors(state, context[:, 1:4])
            distance = (tcp_position_world(state, fk, tcp_ids) - context[:, 4:7]).norm(dim=-1)
            stable = ~term
            threshold = env.cfg.commands["m1_prior_joint"].success_distance
            within = is_m & stable & (distance < threshold)
            if continuous:
                goal_age += env.step_dt
                reached = (cache["terms"]["m_success"] > 0) & stable
                ended = is_m & (reached | term | tout)
                for i in ended.nonzero().flatten().tolist():
                    goal_rows.append({"env": i, "success": bool(reached[i]), "fall": bool(term[i]),
                                      "timeout": bool(tout[i]), "duration_s": float(goal_age[i]),
                                      "initial_distance_m": float(goal_initial_distance[i]),
                                      "terminal_distance_m": float(distance[i]),
                                      "target_world": context[i, 4:7].tolist()})
                goal_age[ended | term | tout] = 0.
            hold = torch.where(within, hold + env.step_dt, 0.)
            success |= hold >= .5
            record('L_velocity_xy_mse', xy_sq[is_l]); record('L_yaw_rate_mse', yaw.square()[is_l])
            record('L_actual_vx', (xy[:, 0] + context[:, 1])[is_l])
            record('L_command_vx', context[:, 1][is_l])
            positive = is_l & (context[:, 1] > .05)
            actual_vx = xy[:, 0] + context[:, 1]
            record('L_positive_command_negative_vx_fraction', (actual_vx < 0).float()[positive])
            # Keep the signed statistic, but distinguish actual retreat from
            # sub-centimetre/second jitter of an effectively stationary robot.
            record('L_positive_command_reverse_vx_below_minus_0p02_fraction', (actual_vx < -.02).float()[positive])
            record('L_positive_command_idle_vx_within_0p02_fraction', (actual_vx.abs() < .02).float()[positive])
            record('M_distance_m', distance[is_m]); record('M_within_threshold_fraction', within.float()[is_m])
            done = term | tout
            for i in done.nonzero().flatten().tolist():
                episode_rows.append({'task': 'L' if bool(is_l[i]) else 'M', 'fall': bool(term[i]),
                                     'terminal_distance_m': float(distance[i]) if bool(is_m[i]) else None,
                                     'hold_success': bool(success[i]) if bool(is_m[i]) else None})
            hold[done] = 0.; success[done] = False
        metrics = {k: sums[k] / counts[k] if counts[k] else None for k in sums}
        for name in ('L_velocity_xy', 'L_yaw_rate'):
            metrics[name + '_rmse'] = math.sqrt(metrics[name + '_mse']) if metrics[name + '_mse'] is not None else None
        for name in ('L', 'M'):
            rows = [r for r in episode_rows if r['task'] == name]
            metrics[name + '_completed_episodes'] = len(rows)
            metrics[name + '_fall_rate'] = sum(r['fall'] for r in rows) / len(rows) if rows else None
            if name == 'M':
                metrics['M_terminal_distance_m'] = sum(r['terminal_distance_m'] for r in rows) / len(rows) if rows else None
                metrics['M_hold_success_rate'] = (sum(r['hold_success'] for r in rows) / len(rows) if rows else None) if not continuous else None
        if continuous:
            # Include goals still active at cutoff as censored, not failed.
            for i in ((command.task_type == 1) & (goal_age > 0)).nonzero().flatten().tolist():
                goal_rows.append({"env": i, "success": False, "censored": True,
                                  "duration_s": float(goal_age[i]), "initial_distance_m": float(goal_initial_distance[i])})
            completed = [r for r in goal_rows if not r.get("censored", False)]
            reached_rows = [r for r in completed if r['success']]
            metrics['M_completed_goals'] = len(completed)
            metrics['M_successful_goals'] = len(reached_rows)
            metrics['M_goal_success_rate'] = len(reached_rows) / len(completed) if completed else None
            metrics['M_successful_goal_time_s'] = sum(r['duration_s'] for r in reached_rows) / len(reached_rows) if reached_rows else None
            metrics['M_censored_goals'] = sum(r.get('censored', False) for r in goal_rows)
            metrics['M_successes_per_M_env_minute'] = len(reached_rows) / ((num_envs // 2 if task_type == 'balanced' else num_envs if task_type == 'M' else 0) * seconds / 60.) if task_type != 'L' else None
        result = {'checkpoint': str(checkpoint.resolve()), 'seed': seed, 'task_type': task_type,
                  'metrics': metrics, 'episodes': episode_rows, 'goals': goal_rows, 'continuous_goals': continuous,
                   'task_contract': {
                      'target': asdict(command.cfg.target_cfg),
                      'reward': asdict(M1RewardCfg(**config['env'].get('m1_reward', {}))),
                      'episode_length_s': config['env'].get('episode_length_s', 10.),
                      'robot_model_sha256': config.get('robot_model_sha256'),
                      'action_mapping': config.get('m1_reward_action_mapping'),
                      'manual_target_world': target_world, 'fixed_velocity': None if vx is None else (vx, vy, yaw_rate),
                  }, 'duration_s': seconds,
                  'unfinished_episodes_excluded_from_episode_metrics': True}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(metrics, indent=2, allow_nan=False))
    finally:
        env.close()


if __name__ == '__main__':
    tyro.cli(main)
