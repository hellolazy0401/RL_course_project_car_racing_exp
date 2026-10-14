from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from racing.grpo_c import (dbscan_states, discounted_returns, kmeans_trajectories,
                           state_aware_advantages, update_grpo_c, update_reference)
from racing.model import Agent


def test_trajectory_and_state_clusters_are_deterministic():
    features = np.array([[0, 0], [.1, 0], [4, 4], [4.1, 4]], np.float32)
    first = kmeans_trajectories(features, 2)
    second = kmeans_trajectories(features, 2)
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[1] and first[2] == first[3] and first[0] != first[2]
    labels = dbscan_states(np.array([[0], [.05], [2], [2.05], [8]], np.float32), .1, 2)
    assert labels[0] == labels[1] >= 0
    assert labels[2] == labels[3] >= 0 and labels[0] != labels[2]
    assert labels[4] == -1


def test_state_aware_advantage_uses_return_to_go_and_trajectory_clusters():
    assert discounted_returns([1, 2, 3], .5).tolist() == [2.75, 3.5, 3.0]
    zero = np.zeros((4, 64, 64), np.uint8)
    trajectories = [dict(obs=[zero]*3, rewards=[1, 1, 1]),
                    dict(obs=[zero]*3, rewards=[0, 0, 0])]
    info = state_aware_advantages(trajectories, np.array([0, 0]), 1., .5, 2, .2)
    first, second = [t['advantage'] for t in trajectories]
    assert first.mean() > 0 and second.mean() < 0
    np.testing.assert_allclose(np.mean(np.concatenate([first, second])), 0, atol=1e-6)
    assert info['state_clusters'] == 1 and info['state_noise_fraction'] == 0
    assert trajectories[0]['clip'] == trajectories[1]['clip'] == .2


def test_joint_policy_update_and_reference_top_member():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    policies = [Agent(with_critic=False), Agent(with_critic=False)]
    reference = deepcopy(policies[0]).requires_grad_(False)
    obs = np.zeros((2, 4, 64, 64), np.uint8)
    trajectories = []
    for policy, advantage in zip(policies, [1., -1.]):
        with torch.no_grad():
            _, latent, logprob, _ = policy.action(torch.as_tensor(obs))
        trajectories.append(dict(obs=list(obs), latent=list(latent.numpy()),
                                 logprob=list(logprob.numpy()), advantage=np.full(2, advantage), clip=.2))
    optimizers = [torch.optim.SGD(policy.actor_parameters(), lr=.001) for policy in policies]
    args = SimpleNamespace(device='cpu', grpo_c_epochs=1, batch_size=1,
                           grpo_c_beta=.01, grpo_c_smooth=.01, grpo_c_diversity=.01,
                           grpo_c_diversity_threshold=.2, max_grad_norm=1., entropy_coef=0.)
    before = [[p.detach().clone() for p in policy.actor_parameters()] for policy in policies]
    metrics = update_grpo_c(policies, reference, optimizers, trajectories, np.array([0, 1]), args)
    assert metrics['actor_updates'] == 2 and metrics['critic_updates'] == 0
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(any(not torch.equal(a, b) for a, b in zip(previous, policy.actor_parameters()))
               for previous, policy in zip(before, policies))
    assert all(p.grad is None for p in reference.parameters())
    update_reference(reference, policies, [0., 1.], 1.)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(parameter, dict(policies[1].named_parameters())[name])
