"""Distribution-space reference mixtures for the continuous-control ensemble.

Independent neural networks do not share a hidden-unit coordinate system. Their
parameters must therefore never be averaged across ensemble members. Each frozen
component follows only its corresponding policy; the reference combines their
action densities. KL to that mixture uses deterministic Gaussian quadrature.
"""
from copy import deepcopy
from itertools import product
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ReferenceMixture(nn.Module):
    """Frozen diagonal-Gaussian components with identity-aligned EMA updates.

    ``kl(distribution, obs)`` returns one approximate KL per observation, with
    gradients through the current distribution. Five quadrature nodes per action
    dimension require 125 points for CarRacing's three-dimensional latent action.
    This is a numerical approximation, not the closed-form Gaussian-to-Gaussian
    KL; finite quadrature can yield a slightly negative estimate. The estimate is
    left unclamped to avoid silently discarding its gradient.
    """

    def __init__(self, policies, action_dim=3, quadrature_points=5):
        super().__init__()
        if not policies:
            raise ValueError("ReferenceMixture requires at least one policy")
        if action_dim < 1 or quadrature_points < 1:
            raise ValueError("Action dimensions and quadrature points must be positive")
        self.components = nn.ModuleList([deepcopy(policy).requires_grad_(False).eval()
                                         for policy in policies])
        device = next(self.components[0].parameters()).device
        self.register_buffer("weights", torch.full((len(policies),),
                                                   1.0 / len(policies), device=device))
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
        combinations = np.asarray(list(product(range(quadrature_points), repeat=action_dim)))
        points = math.sqrt(2.0) * nodes[combinations]
        probabilities = np.prod(weights[combinations] / math.sqrt(math.pi), axis=1)
        self.register_buffer("quadrature_nodes", torch.tensor(points, dtype=torch.float32,
                                                               device=device))
        self.register_buffer("quadrature_weights", torch.tensor(probabilities, dtype=torch.float32,
                                                                 device=device))

    @torch.no_grad()
    def _parameters_at(self, obs):
        distributions = [component.distribution(obs) for component in self.components]
        if any(not isinstance(distribution, Normal) for distribution in distributions):
            raise TypeError("Reference components must provide diagonal Normal distributions")
        return (torch.stack([distribution.loc for distribution in distributions]),
                torch.stack([distribution.scale for distribution in distributions]))

    def _mixture_log_prob(self, latent, locations, scales):
        # latent (..., batch, action), components (member, batch, action).
        component_logprob = Normal(locations, scales).log_prob(latent.unsqueeze(-3)).sum(-1)
        return torch.logsumexp(component_logprob + self.weights.log().unsqueeze(-1), dim=-2)

    def log_prob(self, obs, latent):
        """Return mixture log density for latent actions, summed over action axes."""
        locations, scales = self._parameters_at(obs)
        return self._mixture_log_prob(latent, locations, scales)

    def kl(self, distribution, obs):
        """Approximate KL(current || reference) with differentiable quadrature."""
        if not isinstance(distribution, Normal) or distribution.loc.ndim != 2:
            raise ValueError("Expected a diagonal Normal with (batch, action) parameters")
        if distribution.loc.shape[-1] != self.quadrature_nodes.shape[-1]:
            raise ValueError("Policy action dimensions do not match reference quadrature")
        locations, scales = self._parameters_at(obs)
        nodes = self.quadrature_nodes.to(distribution.loc)
        latent = distribution.loc.unsqueeze(0) + distribution.scale.unsqueeze(0) * nodes.unsqueeze(1)
        log_current = distribution.log_prob(latent).sum(-1)
        log_reference = self._mixture_log_prob(latent, locations, scales)
        return ((log_current - log_reference) * self.quadrature_weights.to(log_current).unsqueeze(1)).sum(0)

    @torch.no_grad()
    def update(self, policies, scores, tau):
        """Follow each policy independently and shift weights toward the best half."""
        if len(policies) != len(self.components) or len(scores) != len(policies):
            raise ValueError("Policies, scores, and reference components must have equal lengths")
        if not math.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError("Reference update tau must be finite and in [0, 1]")
        score_tensor = torch.as_tensor(scores, device=self.weights.device, dtype=self.weights.dtype)
        if not torch.isfinite(score_tensor).all():
            raise ValueError("Reference update scores must all be finite")
        # Stable ordering makes ties reproducible without changing component IDs.
        top = torch.argsort(score_tensor, stable=True)[-max(1, len(policies) // 2):]
        target_weights = torch.zeros_like(self.weights)
        target_weights[top] = 1.0 / len(top)
        self.weights.lerp_(target_weights, tau)
        self.weights.div_(self.weights.sum())
        for component, policy in zip(self.components, policies):
            source = policy.state_dict()
            for name, destination in component.state_dict().items():
                value = source[name].to(destination)
                if destination.is_floating_point() or destination.is_complex():
                    destination.lerp_(value, tau)
                else:
                    destination.copy_(value)
            component.requires_grad_(False).eval()
