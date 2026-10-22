import numpy as np
import pytest
import torch
from racing.sac import SACAgent, SACLearner, SequenceReplayBuffer, validate_restore
from racing.sac_networks import ARCHITECTURES, ObservationHistory
from racing.train import parser


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_architecture_update_and_checkpoint_roundtrip(arch):
    torch.set_num_threads(2)
    args = parser().parse_args(["--algo", "sac", "--device", "cpu", "--sac-arch", arch])
    learner = SACLearner(args)
    shape = (2, 8, 4, 64, 64) if arch == "gru" else (2, 4, 64, 64)
    obs = torch.randint(0, 256, shape, dtype=torch.uint8)
    before = {k: v.clone() for k, v in learner.agent.state_dict().items()}
    m = learner.update((obs, torch.tensor([[0., .4, .1]]*2), torch.ones(2), obs, torch.tensor([False, True])))
    assert all(np.isfinite(v) for v in m.values())
    for prefix in ("actor", "q1", "q2"):
        assert any(not torch.equal(before[k], v) for k, v in learner.agent.state_dict().items() if k.startswith(prefix))
    reloaded = SACAgent(arch)
    reloaded.load_state_dict(learner.agent.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(reloaded.action(obs, True)[0], learner.agent.action(obs, True)[0])
    if arch == "gru":
        assert learner.agent.actor.net.gru.weight_ih_l0.grad.abs().sum() > 0


def test_sequence_replay_matches_online_history_and_truncation():
    replay = SequenceReplayBuffer(20, 2, 3, shape=(1,))
    obs = np.array([[1], [11]], np.uint8)
    history = ObservationHistory(obs, 3)
    expected = {}
    for step in range(5):
        final = obs + 1
        # Environment 0 truncates at step 1, env 1 terminates at step 3.
        boundary = np.array([step == 1, step == 3])
        term = np.array([False, step == 3])
        for env in range(2):
            expected[step*2+env] = history.data[env].copy()
        replay.add(obs, np.zeros((2, 3)), np.ones(2), final, term, boundary)
        next_obs = final.copy(); next_obs[boundary] += 20
        history.advance(next_obs, boundary)
        obs = next_obs
    sequences, next_sequences = replay.sequences(np.arange(10))
    for i in range(10):
        np.testing.assert_array_equal(sequences[i], expected[i])
        np.testing.assert_array_equal(next_sequences[i, :-1], sequences[i, 1:])
        np.testing.assert_array_equal(next_sequences[i, -1], replay.next_obs[i])
    assert not replay.terminated[2]  # Truncation still bootstraps the final context.
    assert np.all(sequences[4, :-1] == 0)  # New episode has no old history.
    assert replay.terminated[7]


def test_sequence_replay_wrap_and_resume_cannot_leak_other_episodes():
    replay = SequenceReplayBuffer(4, 1, 3, shape=(1,))
    for i in range(1, 8):
        replay.add(np.array([[i]], np.uint8), np.zeros((1, 3)), np.ones(1),
                   np.array([[i+1]], np.uint8), np.array([False]))
    # Oldest surviving observation is 4. Its predecessor has been overwritten.
    seq, _ = replay.sequences(np.array([3, 2]))
    np.testing.assert_array_equal(seq[:, :, 0], [[0, 0, 4], [5, 6, 7]])
    restored = SequenceReplayBuffer(4, 1, 3, shape=(1,))
    restored.load_state_dict(replay.state_dict())
    np.testing.assert_array_equal(restored.sequences(np.array([2]))[0], seq[1:])
    restored.start_episodes()
    restored.add(np.array([[50]], np.uint8), np.zeros((1, 3)), np.ones(1),
                 np.array([[51]], np.uint8), np.array([False]))
    np.testing.assert_array_equal(restored.sequences(np.array([3]))[0][0, :, 0], [0, 0, 50])


def test_restore_rejects_architecture_change():
    config = vars(parser().parse_args(["--algo", "sac"]))
    with pytest.raises(ValueError, match="sac_arch"):
        validate_restore({"config": config, "replay": {}}, dict(config, sac_arch="gru"))
