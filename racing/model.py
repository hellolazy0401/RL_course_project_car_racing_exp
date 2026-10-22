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
    def __init__(self, with_critic=True, road_state=False, road_scale=.25):
        super().__init__()
        # Separate CNNs: extra critic steps must never silently change the actor.
        self.actor = tower(3, 0.01)
        self.critic = tower(1, 1.) if with_critic else None
        self.road_state = road_state
        self.road_scale = road_scale
        self.road_head = nn.Linear(16, 3) if road_state else None
        if self.road_head is not None:
            nn.init.zeros_(self.road_head.weight)
            nn.init.zeros_(self.road_head.bias)
        self.logstd = nn.Parameter(torch.full((3,), -0.5))

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.logstd] + (list(self.road_head.parameters()) if self.road_head is not None else [])

    def value(self, obs):
        return self.critic(obs[:, :4].float() / 255.).squeeze(-1)

    def distribution(self, obs):
        expected_channels = 5 if self.road_state else 4
        if obs.ndim != 4 or obs.shape[1] != expected_channels:
            raise ValueError(f"Policy expects {expected_channels} observation channels")
        mean = self.actor(obs[:, :4].float() / 255.)
        if self.road_state:
            state = (obs[:, 4].flatten(1)[:, :16].float()-128)/127
            mean = mean + self.road_scale * self.road_head(state).tanh()
        return Normal(mean, self.logstd.clamp(-5, 2).exp())

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
