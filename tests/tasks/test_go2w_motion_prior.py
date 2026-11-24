import json
from pathlib import Path

import pytest
import torch
from tools.train_go2w_motion_prior import owner_config

from unisupermimic.rl.algos.supermimic.geometry import flatten_state, matvec
from unisupermimic.rl.algos.supermimic.history import HistoricalWindowBank
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2w.config import load_config
from unisupermimic.tasks.velocity.go2w.contract import leg_position_target
from unisupermimic.tasks.velocity.go2w.env import make_go2w_env
from unisupermimic.tasks.velocity.go2w.motion_prior import (
    MotionBank,
    ReferenceCollection,
    apply_stage,
    reference_action,
)
from unisupermimic.tasks.velocity.go2w.training_stages import TrainingStage

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prior():
    from omegaconf import OmegaConf

    owner = owner_config()
    cfg = OmegaConf.to_container(owner.motion_prior)
    bank = ROOT / cfg["bank"]
    if not bank.is_dir():
        pytest.skip("local audited Motion Bank is required")
    return bank, cfg, MotionBank(bank, "cpu", cfg), OmegaConf.to_container(owner.stage)


def test_motion_bank_admission_match_and_horizon(prior, tmp_path):
    path, cfg, bank, _ = prior
    assert len(bank.rows) == 101
    assert int(bank.candidate.sum()) == 78
    assert int((~bank.candidate).sum()) == 23
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["motions"].extend(
        [
            {**manifest["motions"][0], "quality_status": "rejected", "file_path": "invalid.npz"},
            {
                **manifest["motions"][0],
                "quality_status": "candidate",
                "failure_reasons": ["joint_jump"],
                "file_path": "invalid.npz",
            },
        ]
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "motions").mkdir()
    for item in manifest["motions"][:-2]:
        source, target = path / item["file_path"], tmp_path / item["file_path"]
        target.parent.mkdir(exist_ok=True)
        import shutil

        shutil.copyfile(source, target)
    assert len(MotionBank(tmp_path, "cpu", cfg).rows) == 101

    wanted = torch.tensor([[0, 0.20, 0], [0, -0.20, 0], [0, 0, 0], [0.25, 0.2, 0.3]])
    clips = (wanted[:3, None] == bank.nominal).all(-1).int().argmax(-1)
    frame = bank.phase_starts[clips, 20]
    q = torch.cat((bank.positions[clips, frame], bank.positions[0, :1]))
    qd = torch.cat((bank.velocity[clips, frame], bank.velocity[0, :1]))
    got_q, got_qd, mask, weight, valid, *_ = bank.match(
        (None,) * 4 + (q, qd), wanted, torch.ones(4, dtype=torch.bool)
    )
    assert valid.tolist() == [True, True, True, False]
    future = frame[:, None] + torch.arange(1, 9)
    torch.testing.assert_close(got_q[:3], bank.positions[clips[:, None], future])
    torch.testing.assert_close(got_qd[:3], bank.velocity[clips[:, None], future])
    assert torch.equal(mask[:3], bank.data["wheel_rate_valid"][clips[:, None], future])
    assert torch.all(weight[:2] == 0.5) and weight[2] == 1 and weight[3] == 0
    active = torch.ones(8, 4)
    altered = got_qd.clone()
    altered[..., 12:] += 100 * ~mask
    zero, _ = bank.loss(
        (None,) * 4 + (got_q, altered),
        (
            got_q,
            got_qd,
            mask,
            weight,
            valid,
            torch.zeros(4),
            bank.candidate[torch.zeros(4, dtype=torch.long)],
            torch.zeros(4, dtype=torch.long),
        ),
        active,
    )
    assert zero == 0


def test_reference_command_and_phase_priority(prior):
    _, _, bank, _ = prior
    goal = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])
    clip = (bank.nominal == goal[0]).all(-1).nonzero().item()
    steady = bank.phase_starts[clip, 20]
    slow_clip = (bank.nominal == torch.tensor([0.0, 0.05, 0.0])).all(-1).nonzero().item()
    wrong_pose = bank.positions[slow_clip, steady].expand(3, -1)
    wrong_vel = bank.velocity[slow_clip, steady].expand(3, -1)
    velocity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    state = (
        None,
        torch.eye(3).expand(3, -1, -1),
        velocity,
        torch.zeros_like(velocity),
        wrong_pose,
        wrong_vel,
    )
    q, _, _, weight, usable, distance, _, _ = bank.match(
        state, goal, torch.ones(3, dtype=torch.bool)
    )
    assert usable.tolist() == [True, True, False]
    assert distance[0] == 0 and weight[-1] == 0
    for index, phase in enumerate((0, 2)):
        starts = bank.windows[clip, phase]
        candidates = bank.positions[clip, starts[:, None] + torch.arange(1, 9)]
        assert (candidates == q[index]).all(dim=(1, 2)).any()


def test_stage_readiness_must_survive_limit(prior):
    _, cfg, _, settings = prior
    stage = TrainingStage()
    good = {"pass_rates": settings["pass_rates"], "fall_rate": 0.0, "switch_pass_rate": 1.0}
    assert not stage.assess(good, settings | cfg)
    assert not stage.assess(good, settings | cfg)
    assert stage.ready_for_next_stage
    restored = TrainingStage(max_stage="terrain")
    restored.restore(stage.state_dict())
    assert restored.assess(good, settings | cfg) and restored.name == "flat_autonomous"
    assert restored.fraction(cfg) == 1
    restored.training_seconds = cfg["fade_seconds"]
    good["pass_rates"] = settings["autonomous"]["pass_rates"]
    good["iteration"] = 1
    assert not restored.assess(good, settings | cfg)
    assert not restored.assess(good, settings | cfg)
    good["iteration"] = 2
    assert restored.assess(good, settings | cfg) and restored.name == "terrain"


def test_physical_reset_collection_gradient_and_reassignment(prior, tmp_path):
    _, cfg, bank, _ = prior
    torch.set_num_threads(1)
    env = make_go2w_env(load_config("motion_prior"), 16)
    try:
        stage = TrainingStage(max_stage="terrain")
        term = env.command_manager.get_term("velocity")
        env.reference_collection = term.reference_collection = guide = ReferenceCollection(
            bank, env, cfg, stage
        )
        env.reset()
        assert not term.command_domain.any()
        assert (term.curriculum.coverage_level == -1).all()
        selected = guide.source > 0
        assert selected.sum() <= 9 and selected.any()
        assert not term.a_prev.any() and not term.ep_step.any()
        position, rotation, linear, angular, q, qd = term.state()
        expected = bank.velocity[guide.clip[selected], guide.frame[selected]]
        torch.testing.assert_close(qd[selected], expected)
        torch.testing.assert_close(term.g[selected, :3], bank.nominal[guide.clip[selected]])
        root, _, _ = bank.reset_state(
            guide.clip[selected],
            guide.frame[selected],
            torch.zeros_like(position[selected]),
            torch.full_like(position[selected, 0], torch.pi / 2),
        )
        torch.testing.assert_close(root[:, 7], -linear[selected, 1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(root[:, 8], linear[selected, 0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(root[:, 10:13], angular[selected])

        mixed = env.action_manager.get_term("mixed")
        target = bank.positions[guide.clip[selected], guide.frame[selected]]
        action, clipped = reference_action(
            target, expected[:, 12:], mixed.cfg.control, mixed.default, mixed.limits
        )
        assert not clipped.any()
        torch.testing.assert_close(
            leg_position_target(action, mixed.cfg.control, mixed.default, mixed.limits), target
        )
        torch.testing.assert_close(
            action[:, 12:] * mixed.cfg.control.wheel_velocity_scale, expected[:, 12:]
        )

        learner = SuperMimicLearner(
            env,
            SuperMimicConfig(
                hist=5,
                horizon=8,
                goal_frames=1,
                goal_mode="direct",
                wm_window=8,
                terrain_training=True,
                terrain_conditioning=True,
                critic_terrain_conditioning=True,
                wm_structured=True,
                checkpoint_replay=True,
                buffer_steps=24,
                wm_batch=12,
                pi_batch=12,
                wm_batches=1,
                critic_epochs=1,
                critic_minibatches=1,
                history_capacity=64,
                policy_hidden=(32, 32),
                wm_hidden=(32, 32),
                critic_hidden=(32, 32),
                compile=False,
                amp="none",
                no_sync=False,
                strict_finite=True,
                command_start_weights=(1,) * 6,
            ),
            tmp_path,
            device="cpu",
            backend="mujoco",
        )
        learner.motion_prior = bank
        learner.motion_prior_diagnostics = True
        learner.buffer.protected_success_fraction = 0.2
        learner.buffer.history.protection_enabled = True
        apply_stage(learner, stage, cfg)
        guide.source_counts[0] = 2**24
        learner.collect(16)
        assert int(guide.source_counts.sum()) == 2**24 + 16 * env.num_envs
        assert learner.buffer.size == 16
        data = learner.buffer.data
        # The latest buffer truth is the pre-autoreset post-physics state and executed action.
        torch.testing.assert_close(data["s1"][15], flatten_state(env.velocity_next_state))
        torch.testing.assert_close(data["a"][15], env.velocity_executed_action)
        assert not data["command_domain"][:16].any()
        assert (data["collection_source"][:16] == 3).any()
        finished = data["reference_finished"][:16] & ~data["term"][:16]
        assert finished.any() and data["done"][:16][finished].all()
        assert not data["actor_skill_success"][:16][data["collection_source"][:16] != 0].any()
        valid_windows = learner.buffer.sample_windows(32, 8)
        assert (valid_windows["target_id"] == valid_windows["target_id"][:, :1]).all()
        assert (
            valid_windows["collection_source"] == valid_windows["collection_source"][:, :1]
        ).all()
        assert (valid_windows["ep_step"].diff(dim=1) == 1).all()
        assert not valid_windows["done"][:, :-1].any()
        expected_source = guide.source.clone()
        actions = torch.zeros_like(term.a_prev)
        proposed, mask = guide.action(actions, term, mixed)
        local = expected_source == 3
        assert torch.equal(mask, local) and local.any()
        torch.testing.assert_close(proposed[local], guide.noise[local])
        torch.testing.assert_close(proposed[~local], actions[~local])
        learner.wm_update()
        actor_before = [p.detach().clone() for p in learner.pi.parameters()]
        wm_before = [p.detach().clone() for p in learner.wm.parameters()]
        learner.policy_update()
        assert any(not torch.equal(a, b) for a, b in zip(actor_before, learner.pi.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(wm_before, learner.wm.parameters()))
        assert learner.extra_state["motion_prior_metrics"]["prior_gradient_norm"] > 0

        learner.motion_prior = None
        learner.policy_update()
        stage.name = "flat_autonomous"
        stage.training_seconds = cfg["fade_seconds"]
        apply_stage(learner, stage, cfg, reset=True)
        assert not guide.source.any()
        previous_target = term.target_id.clone()
        term.time_left.fill_(0)
        env.step(torch.zeros(16, 16))
        assert (term.target_id != previous_target).all()
        previous_target = term.target_id.clone()
        term.time_left.fill_(0)
        env.step(torch.zeros(16, 16))
        assert (term.target_id != previous_target).all()
        stage.name = "terrain"
        apply_stage(learner, stage, cfg, reset=True)
        assert term.command_domain.any()
        assert torch.all(learner.buffer.data["done"][15])
        stair = term.command_domain >= 4
        assert stair.any()
        assert (term.g[stair, 0] > 0).all() and not term.g[stair, 1:3].any()
        term.curriculum.mastery[stair] = 5
        env.reset()
        torch.testing.assert_close(
            term.traversal[stair, 5:7], torch.tensor([1.0, 0.0]).expand(stair.sum(), 2)
        )
        assert not term.g[stair, 1:3].any()
        saved = learner.save()
        learner.load(saved)
        assert stage.name == "terrain" and learner.buffer.size == 16
        assert not term.a_prev.any() and not term.ep_step.any()
    finally:
        env.close()
        env.backend.close()


def test_historical_success_is_balanced_and_protected():
    k = 8
    history = HistoricalWindowBank(64, k)
    history.protection_enabled = True
    n = 64
    q = torch.zeros(n, k, 41)
    q[..., 3] = 1
    q[..., 8] = 0.2
    goal = torch.zeros(n, k, 7)
    goal[..., 1] = 0.2
    z = torch.zeros(n, k, dtype=torch.long)
    window = dict(
        s=q.clone(),
        s1=q.clone(),
        g=goal,
        command_domain=z,
        terrain_level=z,
        command_mode=z,
        done=z.bool(),
        term=z.bool(),
        traversal_success=z.bool(),
        ep_step=torch.arange(k).expand(n, -1),
    )
    window["collection_source"] = torch.ones_like(z, dtype=torch.int8) * 3
    window["actor_skill_success"] = z.bool()
    history.add(window)
    assert not history.protected.any()
    window["actor_skill_success"] = ~z.bool()
    window["collection_source"].zero_()
    history.add(window)
    assert history.protected.any()
    window["s1"][..., 8] = -0.3
    window["actor_skill_success"].zero_()
    history.add(window)
    trained = history.sample(16, "cpu", protected=True)
    assert (trained["s1"][..., 8] == 0.2).all()
    plain = history.sample(16, "cpu")
    assert not plain["actor_skill_success"].any()
