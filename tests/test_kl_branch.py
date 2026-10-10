from copy import deepcopy
import numpy as np
import pytest
import torch

from racing.model import Agent
from racing.train import parser, update, validate_restore
from racing.kl_experiment import training_command


def test_kl_stops_actor_but_preserves_critic_budget_and_disabled_behavior():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    original = Agent()
    obs = np.zeros((8, 4, 64, 64), np.uint8)
    with torch.no_grad():
        _, latent, logs, _ = original.action(torch.as_tensor(obs))
    data = obs, latent, logs, torch.arange(8.).float(), torch.ones(8)
    results = []
    for target in (None, 1e-10):
        agent = deepcopy(original)
        args = parser().parse_args(["--algo", "ppo", "--ppo-epochs", "3", "--batch-size", "4"])
        args.target_kl = target
        ao = torch.optim.Adam(agent.actor_parameters(), lr=.001)
        co = torch.optim.Adam(agent.critic.parameters(), lr=.001)
        np.random.seed(42)
        metrics = update(agent, ao, co, data, args, "cpu")
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["critic_updates"] == 6
        results.append((agent, metrics))
    assert results[0][1]["actor_updates"] == 6
    assert results[0][1]["kl_early_stop"] == 0
    assert results[1][1]["actor_updates"] == 1
    assert results[1][1]["kl_early_stop"] == 1
    assert results[1][1]["stop_kl"] > 1e-10
    for a, b in zip(results[0][0].critic.parameters(), results[1][0].critic.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_old_checkpoint_compatibility_and_branch_allows_only_kl_change():
    cfg = vars(parser().parse_args(["--algo", "ppo"]))
    old = deepcopy(cfg)
    old.pop("target_kl")
    ck = {"config": old}
    validate_restore(ck, cfg)
    cfg["target_kl"] = .02
    with pytest.raises(ValueError, match="target_kl"):
        validate_restore(ck, cfg)
    validate_restore(ck, cfg, branch=True)
    cfg["actor_lr"] = .5
    with pytest.raises(ValueError, match="actor_lr"):
        validate_restore(ck, cfg, branch=True)


def test_paired_commands_inherit_configuration():
    from types import SimpleNamespace
    cfg = vars(parser().parse_args(["--algo", "ppo", "--actor-lr", ".0001"]))
    args = SimpleNamespace(device="cuda", eval_every=100000, eval_episodes=20,
                           eval_seed_start=1000000, target_kl=.02)
    commands = [training_command("parent.pt", cfg, 1000032, "out", arm, args) for arm in ("control", "kl")]
    parsed = [vars(parser().parse_args(cmd[3:])) for cmd in commands]
    assert {k for k in parsed[0] if parsed[0][k] != parsed[1][k]} == {"name", "target_kl"}
    assert parsed[0]["actor_lr"] == .0001
    assert parsed[0]["branch_from"] == "parent.pt"
    assert parsed[0]["steps"] == 1000032
