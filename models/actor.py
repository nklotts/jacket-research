"""
Стохастическая политика SimbaV2 (Gaussian + tanh).
Перенесено из paper.py.

ВАЖНО про action space:
    Политика выдаёт действие в [-1, 1] через tanh.
    LED-яркость живёт в [0, max_brightness/255].
    Преобразование выполняется в train.py:
        led = (tanh + 1) / 2 * max_brightness/255
    В буфер кладём действие в tanh-пространстве.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import Encoder


class NormalTanhPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing.
    Paper Section 3.1, Eq.1
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, d_model=128, num_blocks=1):
        super().__init__()
        self.enc = Encoder(obs_dim, d_model, num_blocks)
        self.mu_head = nn.Linear(d_model, act_dim)
        self.log_std_head = nn.Linear(d_model, act_dim)

    def forward(self, obs):
        h = self.enc(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        y = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return y, log_prob

    def project(self):
        self.enc.project()


# Алиас для обратной совместимости с прежними импортами
Actor = NormalTanhPolicy
