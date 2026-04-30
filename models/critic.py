"""
Distributional categorical critic (C51) с SimbaV2 энкодером.
Перенесено из paper.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Encoder, NormalizedDense


class CategoricalQNetwork(nn.Module):
    """
    Distributional critic (C51).
    Paper Section 4.3, Eq.13-16
    """

    def __init__(self, obs_dim, act_dim, d_model, num_blocks,
                 natoms=101, vmin=-5.0, vmax=5.0):
        super().__init__()
        self.natoms = natoms
        self.vmin = vmin
        self.vmax = vmax
        self.delta = (vmax - vmin) / (natoms - 1)
        self.register_buffer("atoms", torch.linspace(vmin, vmax, natoms))

        # Encoder takes concatenated observation and action
        self.enc = Encoder(obs_dim + act_dim, d_model, num_blocks)

        # Output layers per paper (Section 4.3, Eq.14)
        self.fc1 = NormalizedDense(
            d_model, d_model,
            sinit=math.sqrt(2.0 / d_model),
            sscale=math.sqrt(2.0 / d_model)
        )
        self.fc2 = NormalizedDense(
            d_model, natoms,
            sinit=math.sqrt(2.0 / d_model),
            sscale=math.sqrt(2.0 / d_model)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        h = self.enc(x)
        h = self.fc1(h)
        logits = self.fc2(h)
        return F.softmax(logits, dim=-1)

    def qval(self, obs, act):
        """Expected Q-value (mean of distribution) (Eq.16)"""
        probs = self.forward(obs, act)
        return (probs * self.atoms).sum(-1, keepdim=True)

    def project(self):
        self.enc.project()
        self.fc1.project()
        self.fc2.project()

    def bellman_target(self, reward, done, next_probs, gamma, ent_bonus=None):
        """Compute target distribution using distributional Bellman equation"""
        atoms = self.atoms.unsqueeze(0)
        if ent_bonus is not None:
            atoms = atoms + ent_bonus

        Tz = (reward + gamma * (1 - done) * atoms).clamp(self.vmin, self.vmax)
        b = (Tz - self.vmin) / self.delta
        l = b.floor().long().clamp(0, self.natoms - 1)
        u = b.ceil().long().clamp(0, self.natoms - 1)

        target = torch.zeros_like(next_probs)
        target.scatter_add_(1, l, next_probs * (u.float() - b))
        target.scatter_add_(1, u, next_probs * (b - l.float()))
        return target.detach()


# Алиас для обратной совместимости
Critic = CategoricalQNetwork
