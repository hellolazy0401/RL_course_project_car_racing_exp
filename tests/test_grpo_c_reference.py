from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from racing.grpo_c_reference import ReferenceMixture


class SmallPolicy(nn.Module):
    def __init__(self, mean=0.0, logstd=-0.5):
        super().__init__()
        self.location = nn.Parameter(torch.full((3,), float(mean)))
        self.logstd = nn.Parameter(torch.full((3,), float(logstd)))

    def distribution(self, obs):
        return Normal(self.location.expand(len(obs), -1), self.logstd.exp())


def test_identical_components_have_zero_kl_and_one_gaussian_matches_analytic_kl():
    obs = torch.zeros(4, 1)
    policy = SmallPolicy()
    identical = ReferenceMixture([policy, deepcopy(policy)])
    torch.testing.assert_close(identical.kl(policy.distribution(obs), obs), torch.zeros(4),
                               atol=1e-6, rtol=0)
    other = SmallPolicy(mean=1.25, logstd=-0.2)
    single = ReferenceMixture([other])
    expected = kl_divergence(policy.distribution(obs), other.distribution(obs)).sum(-1)
    torch.testing.assert_close(single.kl(policy.distribution(obs), obs), expected)


def test_component_permutation_preserves_density_and_quadrature_kl():
    obs = torch.zeros(5, 1)
    policies = [SmallPolicy(mean=-1.3), SmallPolicy(mean=0.8)]
    reference = ReferenceMixture(policies)
    permuted = ReferenceMixture(list(reversed(policies)))
    reference.weights.copy_(torch.tensor([.25, .75]))
    permuted.weights.copy_(reference.weights.flip(0))
    latent = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    torch.testing.assert_close(reference.log_prob(obs, latent), permuted.log_prob(obs, latent))
    current = SmallPolicy(mean=.3).distribution(obs)
    torch.testing.assert_close(reference.kl(current, obs), permuted.kl(current, obs))


def test_ema_preserves_member_identity_and_updates_selection_weights():
    policies = [SmallPolicy(mean=-2), SmallPolicy(mean=2)]
    reference = ReferenceMixture(policies)
    with torch.no_grad():
        policies[0].location.add_(1)
        policies[1].location.add_(3)
    reference.update(policies, scores=[-10., 20.], tau=.25)
    torch.testing.assert_close(reference.components[0].location, torch.full((3,), -1.75))
    torch.testing.assert_close(reference.components[1].location, torch.full((3,), 2.75))
    torch.testing.assert_close(reference.weights, torch.tensor([.375, .625]))
    reference.update(policies, scores=[-10., 20.], tau=1.)
    torch.testing.assert_close(reference.weights, torch.tensor([0., 1.]))
    obs = torch.zeros(2, 1)
    latent = torch.zeros(2, 3)
    torch.testing.assert_close(reference.log_prob(obs, latent),
                               policies[1].distribution(obs).log_prob(latent).sum(-1))


def test_kl_has_finite_policy_gradient_and_frozen_reference_and_serializes():
    policies = [SmallPolicy(mean=-.4), SmallPolicy(mean=.6)]
    reference = ReferenceMixture(policies)
    current = SmallPolicy(mean=1.1, logstd=-.1)
    obs = torch.zeros(3, 1)
    value = reference.kl(current.distribution(obs), obs)
    assert torch.isfinite(value).all()
    value.mean().backward()
    assert all(parameter.grad is None and not parameter.requires_grad
               for parameter in reference.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in current.parameters())
    assert current.location.grad.abs().sum() > 0
    restored = ReferenceMixture(policies)
    restored.load_state_dict(reference.state_dict())
    torch.testing.assert_close(restored.kl(current.distribution(obs), obs), value)
