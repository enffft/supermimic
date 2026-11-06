"""Frozen-checkpoint gradient decomposition on four controlled nominal starts.

No optimizer, parameter update, or training. This is a local counterfactual,
not a reconstruction of the original Replay batches or proof of causality.
"""
import gc
import json
from pathlib import Path

import torch

from supermimic.baselines.go2_supermimic import apply_config
from supermimic.config import TrainConfig
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs import make_env
from supermimic.imagination.compiled import VelocityImagStep
from supermimic.imagination.rollout import imagine
from supermimic.imagination.tasks import M1RewardTask
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robots.mjlab_assets import load_robot_spec_model
from supermimic.robots.spec import get_robot_spec
from supermimic.tasks.world_point_reward import WorldPointRobotParams, m1_tcp_proxy_indices
from supermimic.training.m1_reward import evaluation_args
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "logs/m1_reward/v2_lm50_2048_3p5h_seed2026"
OUT = Path(__file__).parent


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    saved = torch.load(RUN / "ckpt_latest.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg = apply_config(TrainConfig(), saved["config"])
    args = evaluation_args(saved["config"], num_envs=4, task_type="balanced", seed=123)
    env = make_env(args, device="cuda:0")
    try:
        env.reset()
        recorder = env.recorder_manager.get_term("replay")
        history = []
        for _ in range(50):
            with torch.no_grad():
                env.step(torch.zeros(4, 22, device=env.device))
                history.append(recorder.state_io.read().clone())
        state = history[-1]
        context = torch.tensor([[0, .12, 0, 0, 0, 0, 0], [0, -.12, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 1.5, 0, .8], [1, 0, 0, 0, -1.5, 0, .8]], device=env.device)
        context[2:, 4:6] += state[2:, :2]
        start = StartBatch(state=state, xi=recorder.ctx.read().clone(),
                           frame=torch.zeros(4, device=env.device, dtype=torch.long),
                           prev_action=torch.zeros(4, 22, device=env.device),
                           hist_states=torch.stack(history[-cfg.policy.history:]),
                           hist_actions=torch.zeros(cfg.policy.history, 4, 22, device=env.device),
                           goal=context)
        robot = env.scene["robot"]
        spec = get_robot_spec("m1_piper", m1_pose_profile=args.m1_pose_profile,
                              m1_action_contract=args.m1_action_contract)
        fk = G1Kinematics(load_robot_spec_model("m1_piper"), device=env.device)
        q0 = robot.data.default_joint_pos[0].clone()
        limits = robot.data.soft_joint_pos_limits[0].clone()
        heads = AnalyticHeads(fk, spec.tracked_body_names, spec.anchor_body_name, spec.root_body_name,
                              limits, q0, default_base_height=spec.default_base_height,
                              reward_anchor_at_start=True)
        wm = WorldModel(cfg.world_model, heads, env.device).eval().requires_grad_(False)
        policy = Policy(cfg.policy).to(env.device).eval()
        critic = Critic(cfg.critic, feat_dim=wm.spec.feat_dim, goal_dim=8, xi_dim=26,
                        pcfg=cfg.policy).to(env.device).eval().requires_grad_(False)
        task = M1RewardTask(params=WorldPointRobotParams(q0, limits, fall_height=cfg.env.m1_fall_height),
                            fk=fk, tcp_body_indices=m1_tcp_proxy_indices(fk), dt=.02,
                            reward_cfg=cfg.env.m1_reward, base_height=spec.default_base_height,
                            terminal_distance_weight=cfg.policy_trainer.m1_terminal_distance_weight,
                            resample_on_success=True, task_balanced=True)
        step_fn = VelocityImagStep(wm, task, q0, compile_mode=None)
        result = {"method": "4 nominal starts after 1s zero-action settling; L vx+/-0.12, M goals +/-1.5m,z0.8; identical starts across frozen snapshots; H32; no optimization",
                  "limitations": "Not historical Replay; one nominal posture/context; local gradient attribution cannot establish training causality; physical response checked separately",
                  "start_state": state.tolist(), "context": context.tolist(), "snapshots": []}
        for name in ("ckpt_wall_015m.pt", "ckpt_wall_060m.pt", "ckpt_wall_090m.pt",
                     "ckpt_wall_105m.pt", "ckpt_wall_120m.pt", "ckpt_latest.pt"):
            saved = torch.load(RUN / name, map_location="cpu", weights_only=False, mmap=True)
            wm.load_state_dict(saved["wm"]); policy.load_state_dict(saved["policy"])
            critic.load_state_dict(saved["critic"])
            logits = []
            hook = policy.decoder.mlp.register_forward_hook(lambda module, inputs, output: logits.append(output))
            ro = imagine(wm, policy, task, start, 32, obs_noise=False, step_fn=step_fn)
            hook.remove()
            disc = cfg.policy_trainer.gamma ** torch.arange(32, device=env.device)
            value = critic(wm.spec.canonical(ro.states[-1]), ro.goals[-1],
                           wm.spec.normalize_xi(start.xi), ro.e_root[-1])
            components = {
                "dense": (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0).mean() / 32,
                "dense_survival_detached": (disc[:, None] * ro.survival[:-1].detach() * ro.rewards).sum(0).mean() / 32,
                "critic": cfg.policy_trainer.gamma ** 32 * (ro.survival[-1] * value).mean() / 32,
                "critic_survival_detached": cfg.policy_trainer.gamma ** 32 * (ro.survival[-1].detach() * value).mean() / 32,
                "terminal": cfg.policy_trainer.gamma ** 32 * task.terminal_reward(ro.states, context).mean() / 32,
            }
            components["total"] = components["dense"] + components["critic"] + components["terminal"]
            params = tuple(policy.parameters())
            gradients, metrics = {}, {}
            for label, objective in components.items():
                grads = torch.autograd.grad(objective, params + tuple(logits), retain_graph=True, allow_unused=True)
                flat = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                                  for p, g in zip(params, grads[:len(params)])])
                gradients[label] = flat
                logit_grad = torch.stack([g if g is not None else torch.zeros_like(z)
                                         for z, g in zip(logits, grads[len(params):])]).detach()
                z = torch.stack(logits).detach()
                active = z.abs() > .05
                metrics[label] = {"objective_div_H": float(objective.detach()),
                                  "parameter_gradient_norm": float(flat.norm()),
                                  "logit_gradient_abs_sum": float(logit_grad.abs().sum()),
                                  "outward_logit_gradient_mass_fraction": float(
                                      (logit_grad.abs() * active * (logit_grad * z > 0)).sum()
                                      / (logit_grad.abs() * active).sum().clamp_min(1e-20))}
            total = gradients["total"]
            for label in components:
                g = gradients[label]
                metrics[label]["cosine_with_total"] = float(torch.dot(g, total) / (g.norm() * total.norm()).clamp_min(1e-20))
                metrics[label]["projection_on_total_ratio"] = float(torch.dot(g, total) / total.square().sum().clamp_min(1e-20))
            survival_g = gradients["dense"] - gradients["dense_survival_detached"] + gradients["critic"] - gradients["critic_survival_detached"]
            row = {"checkpoint": name, "iteration": saved["iteration"], "actor_updates": saved["actor_updates"],
                   "hours": saved["training_wall_seconds"] / 3600,
                   "imag_action_abs_mean": float(ro.actions.detach().abs().mean()),
                   "imag_abs_action_over_0p95_fraction": float((ro.actions.detach().abs() > .95).float().mean()),
                   "imag_survival_H": ro.survival[-1].detach().tolist(),
                   "critic_H": value.detach().tolist(),
                   "terminal_reward": task.terminal_reward(ro.states.detach(), context).tolist(),
                   "components": metrics, "survival_gradient_norm": float(survival_g.norm()),
                   "survival_gradient_projection_on_total_ratio": float(torch.dot(survival_g, total) / total.square().sum().clamp_min(1e-20))}
            result["snapshots"].append(row)
            (OUT / "m1_gradient_components.json").write_text(json.dumps(result, indent=2) + "\n")
            print("GRADIENT_RESULT", json.dumps(row), flush=True)
            del ro, logits, components, grads, gradients, total, flat, logit_grad, z, saved, value, objective, survival_g
            gc.collect()
    finally:
        env.close()


if __name__ == "__main__":
    main()
