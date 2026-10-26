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
