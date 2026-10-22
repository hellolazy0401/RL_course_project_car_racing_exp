"""One-factor SAC architecture ablations; baseline state-dict stays unchanged."""
import math
import numpy as np
import torch
from torch import nn
from .model import tower


ARCHITECTURES = ("baseline", "layernorm", "split", "residual", "gru")


def normalized_tower(outputs, gain):
    layers = list(tower(outputs, gain).children())
    # Normalize the 256-dimensional visual representation before ReLU.
    layers.insert(8, nn.LayerNorm(256))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class ResidualEncoder(nn.Module):
    def __init__(self, outputs=256, gain=1.):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 32, 5, stride=2, padding=2), nn.ReLU(),
                                 nn.MaxPool2d(2), ResidualBlock(32),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                 ResidualBlock(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                                 nn.Flatten(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, outputs))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain)

    def forward(self, x):
        return self.net(x)


class SplitActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = tower(256, 1.)
        self.steer = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
        self.pedals = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))
        for head in (self.steer, self.pedals):
            nn.init.orthogonal_(head[0].weight, math.sqrt(2)); nn.init.zeros_(head[0].bias)
            nn.init.orthogonal_(head[-1].weight, .01); nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        features = self.encoder(x)
        steer, pedals = self.steer(features), self.pedals(features)
        return torch.cat((steer[:, :1], pedals[:, :2], steer[:, 1:], pedals[:, 2:]), -1)


class RecurrentEncoder(nn.Module):
    """Finite-window GRU recomputed from zero for collection AND replay.

    No stale hidden states or cross-episode carry. Loss is on the final
    transition of a contiguous observation sequence, with gradients through
    the whole window; padded prefix is exactly zero at all call sites.
    """
    def __init__(self, outputs=256, gain=1.):
        super().__init__()
        self.encoder = tower(256, 1.)
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.out = nn.Linear(256, outputs)
        nn.init.orthogonal_(self.out.weight, gain)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("GRU expects [batch,time,channels,height,width]")
        b, t = x.shape[:2]
        features = self.encoder(x.flatten(0, 1)).reshape(b, t, -1)
        self.gru.flatten_parameters()  # Also refresh cuDNN storage in copied target Qs.
        sequence, _ = self.gru(features)
        return self.out(sequence[:, -1])


def actor_net(architecture):
    return {"baseline": lambda: tower(6, .01), "layernorm": lambda: normalized_tower(6, .01),
            "split": SplitActorNet, "residual": lambda: ResidualEncoder(6, .01),
            "gru": lambda: RecurrentEncoder(6, .01)}[architecture]()


def q_encoder(architecture):
    return {"baseline": lambda: tower(256, 1.), "split": lambda: tower(256, 1.),
            "layernorm": lambda: normalized_tower(256, 1.), "residual": ResidualEncoder,
            "gru": RecurrentEncoder}[architecture]()


class ObservationHistory:
    def __init__(self, obs, length):
        self.data = np.zeros((len(obs), length, *obs.shape[1:]), dtype=np.uint8)
        self.data[:, -1] = obs

    def advance(self, next_obs, boundaries):
        self.data[:, :-1] = self.data[:, 1:].copy()
        self.data[np.asarray(boundaries, dtype=bool)] = 0
        self.data[:, -1] = next_obs
