import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def tower(outputs, output_gain):
    layers = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, outputs))
    for layer in layers:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, output_gain)
    return layers


class Agent(nn.Module):
    def __init__(self, with_critic=True):
        super().__init__()
        # Separate CNNs: extra critic steps must never silently change the actor.
        self.actor = tower(3, 0.01)
        self.critic = tower(1, 1.) if with_critic else None
        self.logstd = nn.Parameter(torch.full((3,), -0.5))

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.logstd]

    def value(self, obs):
        return self.critic(obs.float() / 255.).squeeze(-1)

    def distribution(self, obs):
        return Normal(self.actor(obs.float() / 255.), self.logstd.clamp(-5, 2).exp())

    def action(self, obs, latent=None, deterministic=False):
        dist = self.distribution(obs)
        if latent is None:
            latent = dist.mean if deterministic else dist.sample()
        # Store latent actions, not clipped actions. Tanh Jacobian cancels in
        # likelihood ratios; score-function gradient is identical for fixed z.
        logprob = dist.log_prob(latent).sum(-1)
        u = latent.tanh()
        action = torch.stack((u[..., 0], (u[..., 1] + 1) / 2,
                              (u[..., 2] + 1) / 2), dim=-1)
        return action, latent, logprob, dist.entropy().sum(-1)


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """True terminals stop bootstrapping; time limits bootstrap final obs.

    Both boundaries stop the trace, so the reset episode cannot leak backwards.
    """
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        adv[t] = carry
    return adv, adv + values
