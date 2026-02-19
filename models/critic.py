"""
SAC Critic — twin Q-networks for clipped double Q-learning.
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Two independent Q-networks for Soft Actor-Critic.

    Args:
        state_dim:   state dimensionality
        action_dim:  action dimensionality
        hidden_dims: list of hidden layer sizes
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        self.q1 = self._build_q(state_dim + action_dim, hidden_dims)
        self.q2 = self._build_q(state_dim + action_dim, hidden_dims)

    @staticmethod
    def _build_q(in_dim: int, hidden_dims: list) -> nn.Sequential:
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
