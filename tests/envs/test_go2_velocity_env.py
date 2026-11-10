import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
from unisupermimic.tasks.velocity.go2.config import enable_randomization, load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env
from unisupermimic.tasks.velocity.go2.terrain import apply_profile


def test_reset_replay_history_and_command_boundary(tmp_path):
    cfg = load_config()
    cfg.max_episode_seconds = 0.06
    cfg.commands["velocity"].resampling_time_range = (0.04, 0.04)
    e = make_go2_env(cfg, num_envs=2)
    e.init_state()
    try:
        lc = SuperMimicConfig(
            goal_mode="direct",
            goal_frames=1,
            hist=5,
            wm_hidden=(32, 32),
            policy_hidden=(32, 16),
            critic_hidden=(16,),
            buffer_steps=16,
            compile=False,
            amp="none",
            no_sync=False,
            wm_batch=4,
            pi_batch=4,
            pi_opt="adam",
            wm_opt="adam",
            wm_window=2,
            horizon=8,
        )
        l = SuperMimicLearner(e, lc, tmp_path)
        for i in range(9):
            before = e.current_info()
            l.collect(1)
            row = (l.buffer.ptr - 1) % l.buffer.T
            torch.testing.assert_close(l.buffer.data["s"][row], before["s"])
            torch.testing.assert_close(l.buffer.data["g"][row], before["g"])
            assert (l.buffer.data["ep_step"][row] == before["ep_step"]).all()
            if (before["ep_step"] == 0).all():
                assert torch.equal(l.obs_hist, l.obs_hist[:, :1].expand_as(l.obs_hist))
            current = e.current_info()
            if current["ep_step"].eq(0).any():
                assert current["a_prev"].eq(0).all()
                assert current["s"][:, 2].gt(0.4).all()
        starts, hist = l.buffer.sample_starts(100, 5)
        # Repeated padded rows must all belong to the same episode.
        oldest = hist["f"][:, 0]
        newest = hist["f"][:, -1]
        assert ((newest - oldest) >= 0).all()
        assert (newest <= 2).all()
        s = unflatten_state(starts["s"])
        obs = l.mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"])
        next_state, a, h, f, g = l._imag_step(
            s,
            obs,
            starts["a_prev"],
            starts["f"],
            starts["xi"],
            l.mdp.normalize_context(starts["xi"]),
            starts["g"],
        )
        torch.testing.assert_close(g, starts["g"], rtol=0, atol=0)
    finally:
        e.close()
        e.backend.close()


def test_context_randomization_partial_reset_and_clean_executed_actions():
    cfg = enable_randomization(load_config())
    cfg.actions["joint_pos"].ou_env_fraction = 1.0
    e = make_go2_env(cfg, num_envs=4)
    e.init_state()
    try:
        c = e.command_manager.get_term("velocity")
        before = c.xi.clone()
        body = e.backend.get_body_id("robot/base")
        default_com = e.backend.get_body_ipos()[body]
        np.testing.assert_allclose(
            e.backend.get_body_ipos(env_ids=np.arange(4))[:, body] - default_com,
            before[:, 13:].cpu().numpy(),
            atol=1e-6,
        )
        assert ((before[:, 0] >= 0.3) & (before[:, 0] <= 1.2)).all()
        assert before[:, 1:13].abs().max() <= 0.01
        assert before[:, 13:].abs().max() <= 0.025
        e.reset(torch.tensor([0]))
        assert torch.equal(before[1:], c.xi[1:])
        assert torch.equal(before[:, 13:], c.xi[:, 13:])
        np.testing.assert_allclose(
            e.backend.get_body_ipos(env_ids=np.arange(4))[:, body] - default_com,
            before[:, 13:].cpu().numpy(),
            atol=1e-6,
        )
        e.set_collection_exploration(0.1)
        st = e.step(torch.zeros(4, 12))
        tr = st.info["supermimic"]
        assert tr["a"].abs().max() > 0
        assert tr["a_clean"].eq(0).all()
        assert e.current_info()["a_prev"].eq(0).all()
        recomputed = c.mdp.reward_transition(
            unflatten_state(tr["s"]),
            unflatten_state(tr["s1"]),
            tr["g"],
            tr["a_clean"],
            tr["a_prev"],
            executed_action=tr["a"],
            measured_torque=e.velocity_measured_torque,
        )
        torch.testing.assert_close(recomputed, tr["r"])
        e.reset(torch.tensor([0]))
        assert e.action_manager.get_term("joint_pos").ou_state[0].eq(0).all()
    finally:
        e.close()
        e.backend.close()


def test_resume_contract_rejects_command_and_randomization_changes(tmp_path):
    cfg = load_config()
    e = make_go2_env(cfg, num_envs=2)
    e.init_state()
    try:
        lc = SuperMimicConfig(
            goal_mode="direct",
            goal_frames=1,
            hist=5,
            wm_hidden=(16,),
            policy_hidden=(16,),
            critic_hidden=(16,),
            buffer_steps=16,
            compile=False,
            amp="none",
            pi_opt="adam",
            wm_opt="adam",
        )
        learner = SuperMimicLearner(e, lc, tmp_path)
        path = learner.save()
        learner.load(path)
        cfg.commands["velocity"].standing_fraction = 0.2
        with pytest.raises(ValueError, match="semantics"):
            learner.load(path)
    finally:
        e.close()
        e.backend.close()


def test_terrain_learner_is_gated(tmp_path):
    env = make_go2_env(apply_profile(load_config(), "stairs"), num_envs=2)
    env.init_state()
    try:
        model = env.get_playback_model(0)
        assert model.opt.timestep == 0.005
        assert model.opt.iterations == 10
        assert model.opt.ls_iterations == 20
        assert not (model.geom_type == 0).any(), "generated terrain must not retain an extra plane"
        with pytest.raises(ValueError, match="terrain training is gated"):
            SuperMimicLearner(
                env, SuperMimicConfig(goal_mode="direct", hist=5, goal_frames=1), tmp_path
            )
    finally:
        env.close()
        env.backend.close()


@pytest.mark.parametrize(
    "profile", ["rough", "slope", "slope_down", "stairs", "stairs_down", "mixed"]
)
def test_blind_terrain_world_replay_relative_reward_partial_reset(tmp_path, profile):
    cfg = load_config(terrain=True)
    cfg.commands["velocity"].terrain_profile = profile
    cfg.commands["velocity"].terrain_fixed_level = 1
    cfg.max_episode_seconds = 0.10
    env = make_go2_env(cfg, num_envs=4)
    try:
        env.init_state()
        command = env.command_manager.get_term("velocity")
        lc = SuperMimicConfig(
            goal_mode="direct",
            goal_frames=1,
            hist=5,
            terrain_training=True,
            terrain_conditioning=True,
            compile=False,
            amp="none",
            no_sync=False,
            wm_hidden=(32, 32),
            policy_hidden=(32, 16),
            critic_hidden=(16,),
            buffer_steps=64,
            wm_batch=4,
            pi_batch=4,
            wm_window=8,
            horizon=32,
            wm_opt="adam",
            pi_opt="adam",
        )
        learner = SuperMimicLearner(env, lc, tmp_path / profile)
        for _ in range(12):
            before = env.current_info()
            learner.collect(1)
            row = (learner.buffer.ptr - 1) % learner.buffer.T
            assert torch.equal(learner.buffer.data["s"][row], before["s"])
            tr = env.state.info["supermimic"]
            terms = command.mdp.reward_transition(
                unflatten_state(tr["s"]),
                unflatten_state(tr["s1"]),
                tr["g"],
                tr["a_clean"],
                tr["a_prev"],
                executed_action=tr["a"],
                measured_torque=env.velocity_measured_torque,
            )
            torch.testing.assert_close(terms, tr["r"])
            if (before["ep_step"] == 0).all():
                assert torch.equal(
                    learner.obs_hist, learner.obs_hist[:, :1].expand_as(learner.obs_hist)
                )
        before = env.current_info()["s"].clone()
        env.reset(torch.tensor([0]))
        assert torch.equal(before[1:], env.current_info()["s"][1:])
        assert command.mdp.local_height(command.state())[0] == pytest.approx(0.42, abs=1e-6)
        assert learner.wm.in_dim == 142 and learner.pi.dec[0].in_features == 213
        assert learner.critic.norm.mean.numel() == 56
        loss, _ = learner.wm_update()
        assert torch.isfinite(loss)
        metrics = learner.policy_update()
        assert all(torch.isfinite(x).all() for x in metrics)
    finally:
        env.close()
        env.backend.close()


def test_actor_and_unilab_wm_initialization_are_independent(tmp_path):
    from dataclasses import replace

    cfg = load_config()
    env = make_go2_env(cfg, 2)
    env.init_state()
    common = SuperMimicConfig(
        goal_mode="direct",
        hist=5,
        goal_frames=1,
        compile=False,
        amp="none",
        wm_hidden=(16,),
        policy_hidden=(16,),
        critic_hidden=(16,),
        wm_opt="adam",
        pi_opt="adam",
    )
    try:
        source = SuperMimicLearner(env, common, tmp_path / "source")
        with torch.no_grad():
            source.wm.net[-1].weight.normal_(std=0.01)
        checkpoint = source.save()
        target_cfg = load_config(terrain=True)
        target_cfg.commands["velocity"].terrain_profile = "rough"
        target = make_go2_env(target_cfg, 2)
        target.init_state()
        try:
            adapted = SuperMimicLearner(
                target,
                replace(
                    common,
                    terrain_training=True,
                    terrain_conditioning=True,
                    init_actor_from=str(checkpoint),
                    init_wm_from=str(checkpoint),
                ),
                tmp_path / "target",
            )
            for key, value in source.pi.state_dict().items():
                assert torch.equal(value, adapted.pi.state_dict()[key])
            assert adapted.wm.net[0].weight[:, 65:].eq(0).all()
            assert torch.equal(source.wm.net[0].weight, adapted.wm.net[0].weight[:, :65])
            assert source.wm.net[-1].weight.equal(adapted.wm.net[-1].weight)
            assert not adapted.opt_wm.state  # fresh optimizer
        finally:
            target.close()
            target.backend.close()
    finally:
        env.close()
        env.backend.close()


@pytest.mark.parametrize("heading", [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_terrain_heading_controls_entry_and_root_reset(heading):
    cfg = load_config(terrain=True)
    cmdcfg = cfg.commands["velocity"]
    cmdcfg.terrain_profile = "stairs_down"
    cmdcfg.terrain_fixed_level = 2
    cmdcfg.spawn_heading = heading
    env = make_go2_env(cfg, 2)
    try:
        env.init_state()
        c = env.command_manager.get_term("velocity")
        p, R, *_ = c.state()
        torch.testing.assert_close(
            c.mdp.local_height(c.state()), torch.full((2,), 0.42), atol=1e-6, rtol=1e-6
        )
        assert float(R[0, 0, 0]) == pytest.approx(np.cos(heading), abs=1e-6)
        assert float(R[0, 1, 0]) == pytest.approx(np.sin(heading), abs=1e-6)
        direction = p.new_tensor([np.cos(heading), np.sin(heading)])
        ahead = p[:, :2] + direction * 2.0
        assert (
            c.mdp.terrain.ground_height(ahead) < c.mdp.terrain.ground_height(p[:, :2]) - 0.01
        ).all()
    finally:
        env.close()
        env.backend.close()
