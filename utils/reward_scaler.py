"""
Reward scaling для SimbaV2 SAC.
Перенесено из paper.py (Section 4.3, Eq.17-19).

Ведёт онлайн-статистику дисконтированной отдачи G_t и масштабирует награду
на знаменатель max(sqrt(var(G)), Gmax/vmax). До 100 наблюдений используется
только Gmax/vmax — иначе variance шумит на старте.
"""

import math


class RewardScaler:
    """
    Reward scaling based on running statistics of discounted return G_t.
    Paper Section 4.3, Eq.17-19
    """

    def __init__(self, gamma=0.99, vmax=5.0, eps=1e-6):
        self.gamma = gamma
        self.vmax  = vmax
        self.eps   = eps
        self.G     = 0.0
        self.mean  = 0.0
        self.M2    = 0.0
        self.count = 0
        self.Gmax  = eps

    def reset(self):
        """Call at the start of a new episode."""
        self.G = 0.0

    def scale(self, r: float) -> float:
        # Update discounted return (Eq.17)
        self.G = self.gamma * self.G + r

        # Welford update for mean and variance of G
        self.count += 1
        delta = self.G - self.mean
        self.mean += delta / self.count
        delta2 = self.G - self.mean
        self.M2 += delta * delta2

        # Update running maximum (Eq.18)
        self.Gmax = max(self.Gmax, abs(self.G))

        # FIX: Only use variance after sufficient samples (Eq.19)
        if self.count < 100:
            denom = max(1.0, self.Gmax / self.vmax)
        else:
            var = self.M2 / self.count
            denom = max(math.sqrt(var + self.eps), self.Gmax / self.vmax)

        return r / denom
