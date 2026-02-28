"""
Буфер воспроизведения опыта для обучения SAC.
"""

import random
from collections import deque


class ReplayBuffer:
    """
    Кольцевой буфер фиксированной ёмкости, хранящий переходы (s, a, r, s', done).

    Аргументы:
        capacity: максимальное количество хранимых переходов
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)
