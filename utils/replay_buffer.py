"""
Буфер воспроизведения для SimbaV2 SAC.
Перенесено из paper.py: numpy-backed, sample возвращает torch.Tensor сразу
на нужном устройстве (один to(device) вместо посэмпльного копирования).
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Кольцевой буфер фиксированной ёмкости.

    Хранение в numpy для быстрой случайной выборки. При sample данные
    копируются на устройство одним батчем.
    """

    def __init__(self, obs_dim, act_dim, max_size=1_000_000, device=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device if device is not None else torch.device("cpu")

        self.obs  = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act  = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew  = np.zeros((max_size, 1),       dtype=np.float32)
        self.nobs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1),       dtype=np.float32)

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i]  = o
        self.act[i]  = a
        self.rew[i]  = r
        self.nobs[i] = no
        self.done[i] = d
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        to_torch = lambda arr: torch.from_numpy(arr[idx]).to(self.device)
        return (to_torch(self.obs),
                to_torch(self.act),
                to_torch(self.rew),
                to_torch(self.nobs),
                to_torch(self.done))

    def __len__(self) -> int:
        return self.size
