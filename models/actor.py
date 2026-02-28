"""
SAC Actor (стохастическая сеть политики).
Генерирует LED-паттерн в диапазоне [0, max_action] через sigmoid + clamp.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Стохастический актор для Soft Actor-Critic.

    Аргументы:
        state_dim:   размерность входного состояния
        action_dim:  размерность действия (n_superpixels * 3)
        hidden_dims: список размеров скрытых слоёв
        max_action:  верхняя граница действия (например, 196/255)
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list, max_action: float):
        super().__init__()
        self.max_action = max_action

        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h

        self.encoder        = nn.Sequential(*layers)
        self.mean_linear    = nn.Linear(prev, action_dim)
        self.log_std_linear = nn.Linear(prev, action_dim)

        nn.init.constant_(self.mean_linear.bias, 0)

    def forward(self, state: torch.Tensor):
        x       = self.encoder(state)
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """Сэмплирование действия с reparameterization trick (используется при обучении)."""
        mean, log_std = self.forward(state)
        std    = log_std.exp()
        normal = Normal(mean, std)
        x_t    = normal.rsample()
        action = torch.sigmoid(x_t).clamp(0.0, self.max_action)

        # Логарифм вероятности с поправкой на sigmoid
        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(action * (1.0 - action) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Получить действие для инференса (без вычисления градиентов)."""
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.sigmoid(mean).clamp(0.0, self.max_action)
        std    = log_std.exp()
        normal = Normal(mean, std)
        x_t    = normal.sample()
        return torch.sigmoid(x_t).clamp(0.0, self.max_action)
