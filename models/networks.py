"""
Общие строительные блоки SimbaV2.
Перенесено из paper.py практически без изменений.

Содержит:
    l2n              — L2-нормализация по оси
    soft_update      — мягкое обновление целевых сетей
    RunningNorm      — RSNorm (Welford) онлайн mean/var
    NormalizedDense  — линейный слой с весами на гиперсфере + scaler
    ResBlock         — residual блок с LERP и обучаемой alpha
    Encoder          — энкодер наблюдения с SimbaV2 блоками
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Утилиты
# ============================================================================
def l2n(x, dim=-1, eps=1e-6):
    """L2 normalization along specified dimension (Section 4.1)"""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def soft_update(src, tgt, tau):
    """Soft update for target networks (Section 3.1)"""
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# ============================================================================
# RunningNorm — Welford online mean/var (RSNorm), обновляется батчами
# ============================================================================
class RunningNorm(nn.Module):
    """
    Welford's algorithm for online mean and variance.
    Paper Section 3.2, Eq.3-4
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def update(self, x: torch.Tensor):
        """Update statistics with batch of data (not single sample)"""
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            bm = x.mean(0)
            bv = x.var(0, unbiased=False)
            n = x.shape[0]

            if self.count == 0:
                self.mean.copy_(bm)
                self.var.copy_(bv)
            else:
                d = bm - self.mean
                tot = self.count + n
                self.mean.copy_(self.mean + d * n / tot)
                self.var.copy_(
                    (self.var * self.count + bv * n + d ** 2 * self.count * n / tot) / tot
                )
            self.count += n

    def forward(self, x):
        return (x - self.mean) / (self.var.clamp(min=0).sqrt() + self.eps)


# ============================================================================
# NormalizedDense — линейный слой с весами на гиперсфере + scaler
# ============================================================================
class NormalizedDense(nn.Module):
    """
    Linear layer with weights constrained to unit hypersphere,
    plus learnable scaling vector (element-wise).
    Paper Section 4.1, 4.4, Eq.10, 20-21
    """

    def __init__(self, in_features, out_features, sinit=None, sscale=None):
        super().__init__()
        # Weight matrix - explicitly named 'W' for easy collection
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.W)

        # Scaler initialization per paper (Section 4.4, Appendix A.2)
        if sinit is None:
            sinit = math.sqrt(2.0 / in_features)
        if sscale is None:
            sscale = sinit

        # Decoupled scaler (sinit and sscale per paper Eq.21)
        self.sinit = sinit
        self.sscale = nn.Parameter(torch.full((out_features,), sscale))
        self.register_buffer("forward_scale", torch.tensor(sinit / sscale))

    def forward(self, x):
        # Normalize weights along input dimension (Eq.20)
        W_norm = l2n(self.W, dim=1)
        # Apply scaler with decoupled initialization (Eq.21)
        return F.linear(x, W_norm) * (self.sscale * self.forward_scale)

    def project(self):
        """Project weights back to unit sphere after gradient update (Eq.20)."""
        with torch.no_grad():
            self.W.data = l2n(self.W.data, dim=1)


# ============================================================================
# ResBlock — residual блок с LERP и обучаемой alpha
# ============================================================================
class ResBlock(nn.Module):
    """
    Residual block with LERP and learnable alpha.
    Paper Section 4.2, 4.4, Eq.11-12
    """

    def __init__(self, d_model, num_blocks):
        super().__init__()
        d_ff = d_model * 4

        # First linear layer: d_model -> 4d_model
        # Paper: s_init = s_scale = sqrt(2/4d_h) for MLP layers (Section 4.4)
        self.fc1 = NormalizedDense(
            d_model, d_ff,
            sinit=math.sqrt(2.0 / (4 * d_model)),
            sscale=math.sqrt(2.0 / (4 * d_model))
        )

        # Second linear layer: 4d_model -> d_model
        self.fc2 = NormalizedDense(
            d_ff, d_model,
            sinit=math.sqrt(2.0 / d_model),
            sscale=math.sqrt(2.0 / d_model)
        )

        # LERP alpha initialization per paper (Section 4.4)
        # alpha_init = 1/(L+1), alpha_scale = 1/sqrt(d_h)
        alpha_init = 1.0 / (num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(d_model)

        # Store as logit for sigmoid
        raw = math.log(alpha_init / (1 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.full((d_model,), raw))
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

    def forward(self, x):
        # MLP part (Eq.11)
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        h = l2n(h)  # L2-norm after MLP

        # LERP with learnable alpha (Eq.12)
        alpha = torch.sigmoid(self.alpha_logit) * self.alpha_scale
        alpha = alpha.clamp(0, 1)  # Ensure valid interpolation
        out = (1 - alpha) * x + alpha * h
        out = l2n(out)  # Project back to hypersphere
        return out

    def project(self):
        self.fc1.project()
        self.fc2.project()


# ============================================================================
# Encoder — обёртка с RSNorm + shift + блоки
# ============================================================================
class Encoder(nn.Module):
    """
    Observation encoder with SimbaV2 blocks.
    Paper Section 4.1, Fig.3
    """

    def __init__(self, in_dim, d_model, num_blocks, cshift=3.0):
        super().__init__()
        self.cshift = cshift
        self.rms = RunningNorm(in_dim)

        # Input embedding: s_init = s_scale = sqrt(2/d_h) (Section 4.4)
        self.embed = NormalizedDense(
            in_dim + 1, d_model,
            sinit=math.sqrt(2.0 / d_model),
            sscale=math.sqrt(2.0 / d_model)
        )

        self.blocks = nn.ModuleList([
            ResBlock(d_model, num_blocks) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # RSNorm (Section 3.2, Eq.4)
        x = self.rms(x)

        # Shift + L2-norm (Section 4.1, Eq.9)
        shift = x.new_full((*x.shape[:-1], 1), self.cshift)
        x = torch.cat([x, shift], dim=-1)
        x = l2n(x)

        # Linear + Scaler + L2-norm (Section 4.1, Eq.10)
        x = self.embed(x)
        x = l2n(x)

        # Residual blocks (Section 4.2, Eq.11-12)
        for block in self.blocks:
            x = block(x)
        return x

    def project(self):
        self.embed.project()
        for block in self.blocks:
            block.project()

    def update_stats(self, x):
        """Update running statistics with batch"""
        self.rms.update(x)
