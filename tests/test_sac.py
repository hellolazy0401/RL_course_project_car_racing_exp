import math
import numpy as np
import pytest
import torch
from racing.sac import (SACActor, SACLearner, ReplayBuffer, bellman_target,
                        terminal_observations, validate_args, validate_restore)
from racing.train import parser


def arguments(*extra):
    return parser().parse_args(["--algo", "sac", "--device", "cpu", "--batch-size", "2", *extra])


def test_squashed_density_and_reparameterized_gradient():
    torch.set_num_threads(2)
    actor = SACActor()
    obs = torch.zeros(3, 4, 64, 64, dtype=torch.uint8)
    a, z, lp, _ = actor(obs)
    mean, logstd = actor.net(obs.float()/255).chunk(2, -1)
    expected = (torch.distributions.Normal(mean, logstd.clamp(-20, 2).exp()).log_prob(z)
                - torch.log(1-z.tanh().square())).sum(-1)
    torch.testing.assert_close(lp, expected, atol=1e-5, rtol=1e-5)
    assert a[:, 0].abs().max() <= 1 and a[:, 1:].min() >= 0 and a[:, 1:].max() <= 1
    a.sum().backward()
    assert actor.net[-1].bias.grad.abs().sum() > 0
    torch.testing.assert_close(actor(obs, True)[0], actor(obs, True)[0])
    with torch.no_grad():
        actor.net[-1].bias[:3].fill_(100)
    assert torch.isfinite(actor(obs, True)[2]).all()


def test_terminal_mask_bootstraps_time_limit():
    y = bellman_target(torch.tensor([1., 1.]), torch.tensor([True, False]),
                       torch.tensor([10., 10.]), torch.tensor([-2., -2.]), .2, .9)
    torch.testing.assert_close(y, torch.tensor([1., 10.36]))
    reset = np.zeros((2, 4, 64, 64), np.uint8)
    final = np.full((4, 64, 64), 7, np.uint8)
    stored = terminal_observations(reset, [{"terminal_observation": final}, {}])
    np.testing.assert_array_equal(stored[0], final)
    assert stored[1].sum() == 0


def test_replay_wraparound_and_restore():
    replay = ReplayBuffer(3, (1,))
    obs = np.arange(5, dtype=np.uint8)[:, None]
    replay.add(obs, np.zeros((5, 3)), np.arange(5), obs+1, np.zeros(5, bool))
    assert replay.size == 3 and replay.pos == 2
    assert set(replay.rewards) == {2., 3., 4.}
    restored = ReplayBuffer(3, (1,))
    restored.load_state_dict(replay.state_dict())
    np.random.seed(12)
    a = replay.sample(20, "cpu")
    np.random.seed(12)
    b = restored.sample(20, "cpu")
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    np.testing.assert_array_equal(restored.next_obs, restored.obs+1)


@pytest.mark.parametrize("automatic", [True, False])
def test_update_changes_actor_twins_and_soft_targets(automatic):
    torch.set_num_threads(2)
    torch.manual_seed(4)
    args = arguments(*([] if automatic else ["--no-sac-auto-alpha"]))
    learner = SACLearner(args)
    before = {k: v.clone() for k, v in learner.agent.state_dict().items()}
    old_targets = [p.clone() for p in learner.targets.parameters()]
    obs = torch.randint(256, (2, 4, 64, 64), dtype=torch.uint8)
    metrics = learner.update((obs, torch.zeros(2, 3), torch.ones(2), obs, torch.tensor([True, False])))
    assert all(math.isfinite(x) for x in metrics.values())
    for prefix in ("actor.", "q1.", "q2."):
        assert any(not torch.equal(v, before[k]) for k, v in learner.agent.state_dict().items() if k.startswith(prefix))
    for old, target, src in zip(old_targets, learner.targets.parameters(), learner.qparams):
        torch.testing.assert_close(target, old*(1-args.sac_tau)+src*args.sac_tau)
        assert not target.requires_grad and target.grad is None
    assert (metrics["alpha"] != pytest.approx(args.sac_alpha)) if automatic else (metrics["alpha"] == pytest.approx(args.sac_alpha))
    assert all(p.grad is None for p in learner.qparams)


@pytest.mark.parametrize("extra", [["--sac-tau", "0"], ["--sac-alpha", "nan"],
                                    ["--steps", "3"], ["--sac-buffer-size", "1"],
                                    ["--target-kl", ".01"], ["--sac-learning-starts", "-1"]])
def test_invalid_configuration_rejected(extra):
    with pytest.raises(SystemExit):
        validate_args(arguments(*extra), parser())


def test_restore_requires_replay_and_same_protocol():
    config = vars(arguments())
    with pytest.raises(ValueError, match="evaluation-only"):
        validate_restore({"config": config}, config)
    ck = {"config": config, "replay": {}}
    validate_restore(ck, config)
    with pytest.raises(ValueError, match="completion_mode"):
        validate_restore(ck, dict(config, completion_mode="lap95"))


def test_suite_smoke_really_learns(tmp_path, monkeypatch):
    from racing import suite
    calls = []
    monkeypatch.setattr(suite.sys, "argv", ["suite", "--preset", "smoke", "--algos", "sac", "--out", str(tmp_path)])
    monkeypatch.setattr(suite.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    suite.main()
    train = calls[0]
    assert train[train.index("--sac-learning-starts")+1] == "128"
    assert train[train.index("--sac-buffer-size")+1] == "1024"
    assert len(calls) == 4
