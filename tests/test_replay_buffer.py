import torch

from supermimic.data.replay_buffer import ReplayBuffer


def test_window_done_labels_align_with_predicted_successor_state():
    """A terminal transition labels its successor, never the pre-terminal state."""
    buf = ReplayBuffer(num_envs=1, capacity=32, state_dim=1, act_dim=1, xi_dim=1, device="cpu")
    env = torch.tensor([0])
    for state_id in range(10):
        done = torch.tensor([float(state_id == 9)])
        state = torch.tensor([[float(state_id)]])
        buf.add(
            env,
            s=state,
            a=torch.zeros(1, 1),
            a_exec=torch.zeros(1, 1),
            xi=torch.zeros(1, 1),
            r=torch.zeros(1),
            done=done,
            frame=torch.zeros(1, dtype=torch.long),
            clip=torch.zeros(1, dtype=torch.long),
            terminal=torch.zeros(1, dtype=torch.bool),
            ep_start=torch.tensor([state_id == 0]),
        )

    # The terminal row has no outgoing transition, but records the terminal target state.
    buf.add(
        env,
        s=torch.tensor([[10.0]]),
        a=torch.zeros(1, 1),
        a_exec=torch.zeros(1, 1),
        xi=torch.zeros(1, 1),
        r=torch.zeros(1),
        done=torch.ones(1),
        frame=torch.zeros(1, dtype=torch.long),
        clip=torch.zeros(1, dtype=torch.long),
        terminal=torch.ones(1, dtype=torch.bool),
        ep_start=torch.zeros(1, dtype=torch.bool),
    )
    buf.end_episode(env)

    batch = buf.sample_windows(n=1024, k=4)
    positive = batch.done.bool()
    assert positive.any()
    # Every positive successor is the terminal state (state_id=10), and no earlier
    # prediction in an accepted window is mislabeled terminal.
    assert torch.all(batch.states[1:][positive] == 10.0)
    assert not batch.done[:-1].any()


def test_m1_replay_actor_and_wm_batches_are_stratified_l_m():
    buf = ReplayBuffer(4, 20, 57, 22, 2, "cpu", goal_dim=8, m1_joint_context=True)
    env_ids = torch.arange(4)
    task_type = torch.tensor([0, 0, 1, 1])
    state = torch.zeros(4, 57)
    state[:, 2] = 0.47
    state[:, 3] = 1.0
    for step in range(12):
        buf.add(
            env_ids, state, torch.zeros(4, 22), torch.zeros(4, 22),
            torch.zeros(4, 2), torch.zeros(4), torch.zeros(4),
            torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.bool), torch.full((4,), step == 0),
            goal=torch.zeros(4, 8), task_type=task_type,
            velocity_command=torch.zeros(4, 3), fixed_world_target=torch.zeros(4, 3),
        )

    actor_batch = buf.sample_starts(32, hist=5, task_balanced=True)
    wm_batch = buf.sample_windows(32, k=8, task_balanced=True)
    assert torch.equal(torch.bincount(actor_batch.task_type, minlength=2), torch.tensor([16, 16]))
    assert torch.equal(torch.bincount(wm_batch.task_type, minlength=2), torch.tensor([16, 16]))


def test_task_balanced_replay_fails_closed_if_one_task_has_no_valid_samples():
    buf = ReplayBuffer(2, 12, 57, 22, 1, "cpu", goal_dim=8, m1_joint_context=True)
    env_ids = torch.arange(2)
    state = torch.zeros(2, 57)
    for step in range(6):
        buf.add(
            env_ids, state, torch.zeros(2, 22), torch.zeros(2, 22),
            torch.zeros(2, 1), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.bool), torch.full((2,), step == 0),
            goal=torch.zeros(2, 8), task_type=torch.zeros(2, dtype=torch.long),
            velocity_command=torch.zeros(2, 3), fixed_world_target=torch.zeros(2, 3),
        )
    import pytest

    with pytest.raises(RuntimeError, match="no sampleable rows"):
        buf.sample_starts(8, hist=3, task_balanced=True)
