"""Contract and gradient checks for the opt-in M1 CVAE planning path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

AMP_ROOT = Path("/home/yifeiy/AMP-LOCO-M1")
PRIOR_PATH = AMP_ROOT / "logs/motion_prior/cvae_v1_strict_fcl_2272/checkpoint_best_val.pt"
PRIOR_SHA256 = "1624741585c9a5dac7301bfb500ebf920a2c4e87f101ec1b60c945f86367106d"


def _amp_imports():
    if str(AMP_ROOT) not in sys.path:
        sys.path.insert(0, str(AMP_ROOT))
    from motion_prior.cvae_features import encode_future, encode_state_history, encode_task_goal
    from motion_prior.cvae_inference import load_cvae_checkpoint
    from tools.motion_generation.m1_task_motion_schema import world_to_heading

    return encode_future, encode_state_history, encode_task_goal, world_to_heading, load_cvae_checkpoint


def _state(batch: int, *, device="cpu") -> torch.Tensor:
    from supermimic.robots.spec import M1_DEFAULT

    state = torch.zeros(batch, 57, dtype=torch.float32, device=device)
    state[:, 2] = 0.65
    state[:, 3] = 1.0
    state[:, 7:29] = torch.tensor(M1_DEFAULT, dtype=state.dtype, device=state.device)
    return state


def _amp_frame(state: torch.Tensor) -> np.ndarray:
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS

    value = state.detach().cpu().numpy()
    frame = np.zeros(79, dtype=np.float32)
    frame[:3] = value[:3]
    frame[3:7] = value[[4, 5, 6, 3]]
    for i, name in enumerate(M1_AMP_JOINTS):
        frame[7 + i] = value[7 + M1_JOINTS.index(name)]
        frame[46 + i] = value[35 + M1_JOINTS.index(name)]
    body_velocity = quat_apply_inverse(state[3:7], state[29:32]).detach().cpu().numpy()
    frame[40:43] = body_velocity
    frame[43:46] = value[32:35]
    return frame


def test_state_goal_and_future_match_amp_contract():
    from supermimic.kinematics.rotations import quat_apply, quat_from_rotvec, quat_mul
    from supermimic.motion_prior.features import state_history_to_cvae53, task_goal8
    from supermimic.motion_prior.reference import decode_future31
    from supermimic.robots.spec import M1_AMP_JOINTS, M1_JOINTS, M1_LEG_JOINTS

    encode_future, encode_state_history, encode_task_goal, world_to_heading, _ = _amp_imports()
    torch.manual_seed(17)
    history = _state(1)[:, None].expand(-1, 4, -1).clone()
    body_velocity = torch.randn(1, 4, 3)
    rotvec = torch.randn(1, 4, 3) * 0.35

    history[..., 3:7] = quat_from_rotvec(rotvec)
    history[..., 29:32] = quat_apply(history[..., 3:7], body_velocity)
    history[..., 32:35] = torch.randn(1, 4, 3)
    for name in M1_AMP_JOINTS:
        history[..., 7 + M1_JOINTS.index(name)] = torch.randn(1, 4) * 0.2
        history[..., 35 + M1_JOINTS.index(name)] = torch.randn(1, 4)
    raw_frames = np.stack([_amp_frame(row) for row in history[0]])
    wheel_rates = np.stack([
        history[0, :, 35 + M1_JOINTS.index(name)].numpy()
        for name in M1_LEG_JOINTS if name.endswith("FOOT_JOINT")
    ], axis=-1)
    expected_state = encode_state_history(raw_frames, wheel_rates)
    np.testing.assert_allclose(state_history_to_cvae53(history)[0].numpy(), expected_state, atol=5e-7, rtol=0)

    goal_velocity = np.array([0.25, -0.1, 0.2], dtype=np.float32)
    l_goal = task_goal8(
        history[:, -1], torch.tensor([0]), torch.tensor(goal_velocity)[None], torch.zeros(1, 3),
    )[0].numpy()
    np.testing.assert_allclose(l_goal, encode_task_goal("L", goal_velocity, np.zeros(3, np.float32)), atol=1e-6)
    target = np.array([0.9, -0.2, 0.8], dtype=np.float32)
    root = history[0, -1, :3].numpy()
    q = history[0, -1, 3:7].numpy()
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    ee_heading = world_to_heading(target, root, yaw)
    m_goal = task_goal8(
        history[:, -1], torch.tensor([1]), torch.zeros(1, 3), torch.tensor(target)[None],
    )[0].numpy()
    np.testing.assert_allclose(m_goal, encode_task_goal("M", np.zeros(3, np.float32), ee_heading), atol=1e-6)

    current = _state(1)[0]
    current[3:7] = quat_from_rotvec(torch.tensor([0.2, -0.1, 0.35]))
    future_states = []
    rates = []
    for frame_i in range(10):
        row = current.clone()
        row[:3] += torch.tensor([0.01 * frame_i, -0.004 * frame_i, 0.002 * frame_i])
        row[3:7] = quat_mul(current[3:7], quat_from_rotvec(torch.tensor([0.01 * frame_i, 0.0, 0.02 * frame_i])))
        for joint_i, name in enumerate(M1_AMP_JOINTS):
            row[7 + M1_JOINTS.index(name)] += 0.01 * frame_i * (joint_i + 1)
        wheel = np.arange(4, dtype=np.float32) + frame_i * 0.2
        future_states.append(row)
        rates.append(wheel)
    future_states = torch.stack(future_states)
    encoded_future = encode_future(
        _amp_frame(current), np.stack([_amp_frame(row) for row in future_states]), np.stack(rates),
    )
    decoded = decode_future31(current[None], torch.from_numpy(encoded_future)[None])[0]
    torch.testing.assert_close(decoded[:, :3], future_states[:, :3], atol=2e-6, rtol=0)
    from supermimic.kinematics.rotations import quat_error_magnitude

    torch.testing.assert_close(
        quat_error_magnitude(decoded[:, 3:7], future_states[:, 3:7]),
        torch.zeros(10), atol=2e-5, rtol=0,
    )
    nonwheel = [M1_JOINTS.index(name) for name in M1_AMP_JOINTS]
    torch.testing.assert_close(decoded[:, 7:29][:, nonwheel], future_states[:, 7:29][:, nonwheel], atol=1e-6, rtol=0)
    wheel_ids = [M1_JOINTS.index(name) for name in M1_LEG_JOINTS if name.endswith("FOOT_JOINT")]
    torch.testing.assert_close(decoded[:, 35:57][:, wheel_ids], torch.from_numpy(np.stack(rates)), atol=1e-6, rtol=0)


def test_link7_matches_gripper_link7_link8_midpoint_for_position_fk():
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.mjlab_assets import load_robot_spec_model
    from supermimic.robots.spec import get_robot_spec

    spec = get_robot_spec("m1_piper")
    fk = G1Kinematics(load_robot_spec_model("m1_piper"), device="cpu")
    qpos = torch.zeros(8, 7 + spec.num_joints)
    qpos[:, 2] = 0.65
    qpos[:, 3] = 1.0
    qpos[:, 7:] = torch.tensor(spec.default_joint_pos)
    qpos[:, 7 + 16:7 + 22] += torch.randn(8, 6) * 0.5
    bodies = fk(qpos)["pos"]
    link7 = bodies[:, fk.body_index("piper_x_link7")]
    link8 = bodies[:, fk.body_index("piper_x_link8")]
    torch.testing.assert_close(link7, link8, atol=1e-7, rtol=0)
    torch.testing.assert_close(link7, (link7 + link8) * 0.5, atol=1e-7, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity check needs the configured GPU")
def test_pretrained_prior_strict_load_and_amp_cpu_gpu_parity():
    if not PRIOR_PATH.is_file():
        pytest.fail(f"required AMP CVAE checkpoint missing: {PRIOR_PATH}")
    _, _, encode_task_goal, _, load_cvae_checkpoint = _amp_imports()
    from supermimic.motion_prior import load_m1_motion_prior
    from supermimic.motion_prior.features import state_history_to_cvae53

    cpu_reference = load_cvae_checkpoint(PRIOR_PATH, device="cpu")
    cpu_prior = load_m1_motion_prior(PRIOR_PATH, device="cpu", expected_sha256=PRIOR_SHA256)
    gpu_prior = load_m1_motion_prior(PRIOR_PATH, device="cuda:0", expected_sha256=PRIOR_SHA256)
    state = _state(2, device="cuda:0")[:, None].expand(-1, 4, -1).clone()
    goals_np = np.stack([
        encode_task_goal("L", np.array([0.2, 0.0, 0.15], np.float32), np.zeros(3, np.float32)),
        encode_task_goal("M", np.zeros(3, np.float32), np.array([0.2, 0.1, 0.15], np.float32)),
    ])
    goals = torch.as_tensor(goals_np, device="cuda:0")
    features = state_history_to_cvae53(state).cpu().numpy()
    amp = cpu_reference.predict(features, goals.cpu().numpy()).numpy()
    torch.testing.assert_close(cpu_prior(state.cpu(), goals.cpu()), torch.from_numpy(amp), atol=2e-6, rtol=1e-6)
    gpu = gpu_prior(state, goals)
    # The Simulator backend enables CUDA TF32; this still limits physical-unit
    # output error to < 3e-4 while CPU checkpoint inference matches tightly.
    torch.testing.assert_close(gpu.cpu(), torch.from_numpy(amp), atol=4e-4, rtol=2e-4)
    assert gpu.shape == (2, 10, 31) and torch.isfinite(gpu).all()


class _TinyWM(nn.Module):
    def __init__(self):
        super().__init__()
        from supermimic.robots.spec import M1_JOINTS

        wheels = tuple(M1_JOINTS.index(name) for name in (
            "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
        ))
        self.spec = SimpleNamespace(wheel_indices=wheels, nj=22)
        self.gain = nn.Parameter(torch.tensor(0.7))

    def step(self, state, action, xi):
        del xi
        root_velocity = action[:, :3] * self.gain
        angular_velocity = action[:, 3:6] * self.gain
        joint_position = state[:, 7:29] + 0.02 * action
        joint_velocity = action
        return torch.cat((
            state[:, :3] + 0.02 * root_velocity,
            state[:, 3:7], joint_position,
            root_velocity, angular_velocity, joint_velocity,
        ), dim=-1)


class _TinyFK:
    def __call__(self, qpos, qvel=None):
        from supermimic.kinematics.rotations import quat_apply

        del qvel
        arm = qpos[:, 23:29]
        offset = torch.stack((0.35 + 0.015 * arm.sum(-1), 0.01 * arm[:, 0], 0.25 + 0.01 * arm[:, 1]), dim=-1)
        tcp = qpos[:, :3] + quat_apply(qpos[:, 3:7], offset)
        return {"pos": tcp[:, None, :]}


def _tiny_joint_buffer(device: torch.device):
    from supermimic.data.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(2, 16, 57, 22, 26, device, goal_dim=7, m1_joint_context=True)
    state = _state(2, device=device)
    target = torch.tensor([[0.75, 0.05, 0.90], [0.72, -0.04, 0.88]], device=device)
    ids = torch.arange(2, device=device)
    for t in range(8):
        buffer.add(
            ids, state, torch.zeros(2, 22, device=device), torch.zeros(2, 22, device=device),
            torch.zeros(2, 26, device=device), torch.zeros(2, device=device),
            torch.zeros(2, device=device), torch.zeros(2, dtype=torch.long, device=device),
            torch.zeros(2, dtype=torch.long, device=device), torch.zeros(2, dtype=torch.bool, device=device),
            torch.full((2,), t == 0, dtype=torch.bool, device=device),
            goal=torch.cat((torch.ones(2, 1, device=device), torch.zeros(2, 6, device=device)), dim=-1),
            task_type=torch.ones(2, dtype=torch.long, device=device),
            velocity_command=torch.zeros(2, 3, device=device), fixed_world_target=target,
        )
    return buffer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gradient ownership check is run on CUDA")
def test_planner_actor_gradient_ownership_and_horizon3_finite_difference():
    from dataclasses import replace

    from supermimic.motion_prior import load_m1_motion_prior
    from supermimic.motion_prior.losses import M1JointLossCfg, planner_losses
    from supermimic.motion_prior.rollout import differentiable_m1_rollout
    from supermimic.motion_prior.trainer import M1AlternatingTrainer, frozen_parameters
    from supermimic.policy.nets import M1PriorActorCfg, M1PriorConditionedActor
    from supermimic.robots.spec import M1_DEFAULT, M1_JOINTS

    if not PRIOR_PATH.is_file():
        pytest.fail(f"required AMP CVAE checkpoint missing: {PRIOR_PATH}")
    device = torch.device("cuda:0")
    prior = load_m1_motion_prior(PRIOR_PATH, device=device, expected_sha256=PRIOR_SHA256)
    actor = M1PriorConditionedActor(M1PriorActorCfg(history=4, hidden=(64, 32), reference_embedding=16)).to(device)
    wm = _TinyWM().to(device)
    buffer = _tiny_joint_buffer(device)
    wm_trainer = SimpleNamespace(opt=torch.optim.SGD(wm.parameters(), lr=1e-3), cfg=SimpleNamespace(batches_per_update=1))
    cfg = SimpleNamespace(
        prior_horizon=10, actor_horizon=10, replanning_cadence=10,
        state_source="privileged", prior_mode="prior_mean", batch_size=2, micro_batch_size=2,
        prior_lr=1e-4, actor_lr=2e-4, grad_clip_norm=1.0,
        loss=M1JointLossCfg(), wm_updates_per_event=1, prior_updates_per_event=1,
        actor_updates_per_event=1, freeze_prior=True, actor_pretrain_updates=0,
        prior_warmup_updates=0, readiness_window_steps=4, readiness_min_samples=4,
        readiness_min_survival=0.9, readiness_max_root_tracking_error=0.2,
        readiness_max_arm_tracking_error=0.25, readiness_max_wheel_tracking_error=0.75,
        readiness_max_action_saturation=0.1,
    )
    class _Command:
        def state_dict(self):
            return {"task_type": "M"}

    joint = M1AlternatingTrainer(
        wm=wm, wm_trainer=wm_trainer, prior=prior, actor=actor, buffer=buffer,
        default_joint_pos=torch.tensor(M1_DEFAULT, device=device),
        soft_limits=torch.tensor([[-3.0, 3.0]] * 22, device=device),
        fk=_TinyFK(), tcp_index=0, cfg=cfg, robot_joint_names=tuple(M1_JOINTS),
        robot_model_sha256="test-model", command=_Command(),
    )

    # A two-sample, three-step planner loss has a finite Autograd derivative
    # through the frozen Actor and World Model input Jacobians into the CVAE.
    from supermimic.motion_prior.features import task_goal8

    start = buffer.sample_starts(2, actor.cfg.history)
    goal = task_goal8(start.state, start.task_type, start.velocity_command, start.fixed_world_target)
    history = joint._prior_state_history(start)
    parameter = prior.model.decoder[1].bias
    index = 0

    def objective():
        ref = prior(history, goal)
        with frozen_parameters(actor, wm):
            rollout = differentiable_m1_rollout(
                wm, actor, start, goal, ref, joint.default_joint_pos, horizon=3,
            )
            terms = planner_losses(
                rollout.states, rollout.actions, start.state, ref,
                joint.prior_initial(history, goal), start.task_type,
                start.velocity_command, start.fixed_world_target,
                fk=joint.fk, tcp_index=0, soft_limits=joint.soft_limits,
                cfg=joint.loss_cfg,
            )
        return terms["task"]

    loss = objective()
    grad = torch.autograd.grad(loss, parameter)[0][index]
    assert torch.isfinite(grad) and grad.abs() > 1e-8
    original = parameter[index].detach().clone()
    epsilon = 2e-2
    with torch.no_grad():
        parameter[index] = original + epsilon
        plus = objective().detach()
        parameter[index] = original - epsilon
        minus = objective().detach()
        parameter[index] = original
    finite_difference = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(grad, finite_difference, atol=2e-3, rtol=0.12)

    # The repaired L objective must also backpropagate through the frozen
    # Actor and WM into the CVAE, and its wz target must not touch root vz.
    l_command = torch.tensor([[0.2, 0.0, 0.3], [0.2, 0.0, 0.3]], device=device)
    l_task = torch.zeros(2, dtype=torch.long, device=device)
    l_target = torch.zeros(2, 3, device=device)
    l_start = replace(
        start, task_type=l_task, velocity_command=l_command,
        fixed_world_target=l_target,
    )
    l_goal = task_goal8(l_start.state, l_task, l_command, l_target)
    l_reference = prior(history, l_goal)
    with frozen_parameters(actor, wm):
        l_rollout = differentiable_m1_rollout(
            wm, actor, l_start, l_goal, l_reference,
            joint.default_joint_pos, horizon=3,
        )
        l_terms = planner_losses(
            l_rollout.states, l_rollout.actions, l_start.state,
            l_reference, joint.prior_initial(history, l_goal), l_task,
            l_command, l_target, fk=joint.fk, tcp_index=0,
            soft_limits=joint.soft_limits, cfg=joint.loss_cfg,
        )
    l_grad = torch.autograd.grad(l_terms["task_l"], parameter, retain_graph=True)[0]
    assert torch.isfinite(l_grad).all() and l_grad.norm() > 1e-8
    probe_states = l_rollout.states.detach().clone().requires_grad_(True)
    from supermimic.motion_prior.losses import _task_planner_terms

    probe_terms = _task_planner_terms(
        probe_states, l_task, l_command, l_target, joint.fk, 0, joint.loss_cfg,
    )
    state_grad = torch.autograd.grad(probe_terms["task_l"], probe_states)[0]
    assert torch.equal(state_grad[..., 31], torch.zeros_like(state_grad[..., 31]))

    prior_before = [p.detach().clone() for p in prior.parameters()]
    actor_before = [p.detach().clone() for p in actor.parameters()]
    wm_before = [p.detach().clone() for p in wm.parameters()]
    joint._update_prior()
    assert any(not torch.equal(a, b) for a, b in zip(prior_before, prior.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(wm_before, wm.parameters()))

    prior_before = [p.detach().clone() for p in prior.parameters()]
    actor_before = [p.detach().clone() for p in actor.parameters()]
    wm_before = [p.detach().clone() for p in wm.parameters()]
    joint._update_actor()
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(prior_before, prior.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(wm_before, wm.parameters()))
