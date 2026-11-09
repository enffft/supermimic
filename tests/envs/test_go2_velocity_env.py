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
