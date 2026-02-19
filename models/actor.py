"""
SAC Actor (stochastic policy network).
Produces LED pattern actions in the range [0, max_action] via sigmoid + clamp.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Stochastic actor for Soft Actor-Critic.

    Args:
        state_dim:   input state dimensionality
        action_dim:  output action dimensionality (N_LEDS * 3)
        hidden_dims: list of hidden layer sizes
        max_action:  upper bound of the action range (e.g. 196/255)
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list,
        max_action: float,
    ):
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

    def forward(self, state: torch.Tensor):
        x       = self.encoder(state)
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """Sample action with reparameterization trick (used during training)."""
        mean, log_std = self.forward(state)
        std    = log_std.exp()
        normal = Normal(mean, std)
        x_t    = normal.rsample()
        action = torch.sigmoid(x_t).clamp(0.0, self.max_action)

        # Log-probability with sigmoid correction
        log_prob  = normal.log_prob(x_t)
        log_prob -= torch.log(action * (1.0 - action) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference (no gradient computation required)."""
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.sigmoid(mean).clamp(0.0, self.max_action)
        std    = log_std.exp()
        normal = Normal(mean, std)
        x_t    = normal.sample()
        return torch.sigmoid(x_t).clamp(0.0, self.max_action)
