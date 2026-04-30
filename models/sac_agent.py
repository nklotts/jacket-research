"""
SimbaV2 SAC агент.

Перенесено из paper.py с минимальными изменениями:
    - device настраивается через параметр (вместо модульного DEVICE)
    - добавлены методы save/load для интеграции с train.py
    - import RewardScaler вынесен наверх

Алгоритм не изменён ни в одной строке: distributional critic (C51),
clipped double Q, soft update target, learnable alpha с target_entropy=-|A|/2,
UTD-петля, проекция весов на гиперсферу после каждого обновления.
"""

import math

import torch
from torch.optim import Adam, lr_scheduler

from utils.reward_scaler import RewardScaler

from .actor import NormalTanhPolicy
from .critic import CategoricalQNetwork
from .networks import NormalizedDense, l2n, soft_update


class SacAgent:
    """
    SAC agent with SimbaV2 architecture.
    Paper Section 3.1, 4, Appendix C
    """

    def __init__(self, obs_dim, act_dim, act_limit, cfg: dict, total_steps: int,
                 device=None):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.act_limit = act_limit
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.utd = cfg.get("utd", 2)
        self.batch_size = cfg.get("batch_size", 256)

        natoms = cfg.get("natoms", 101)
        vmin = cfg.get("vmin", -5.0)
        vmax = cfg.get("vmax", 5.0)

        policy_d = cfg.get("actor_d", 128)
        policy_nb = cfg.get("actor_nb", 1)
        critic_d = cfg.get("critic_d", 512)
        critic_nb = cfg.get("critic_nb", 2)

        lr = cfg.get("lr", 1e-4)
        lr_final = cfg.get("lr_final", 3e-5)  # Paper Table 3: 3e-5

        # ---- Сети ----
        self.policy = NormalTanhPolicy(
            obs_dim, act_dim, policy_d, policy_nb
        ).to(self.device)
        self.critic1 = CategoricalQNetwork(
            obs_dim, act_dim, critic_d, critic_nb, natoms, vmin, vmax
        ).to(self.device)
        self.critic2 = CategoricalQNetwork(
            obs_dim, act_dim, critic_d, critic_nb, natoms, vmin, vmax
        ).to(self.device)
        self.target_critic1 = CategoricalQNetwork(
            obs_dim, act_dim, critic_d, critic_nb, natoms, vmin, vmax
        ).to(self.device)
        self.target_critic2 = CategoricalQNetwork(
            obs_dim, act_dim, critic_d, critic_nb, natoms, vmin, vmax
        ).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        for p in list(self.target_critic1.parameters()) + list(self.target_critic2.parameters()):
            p.requires_grad_(False)

        # Share running statistics between online and target critics
        self.target_critic1.enc.rms = self.critic1.enc.rms
        self.target_critic2.enc.rms = self.critic2.enc.rms

        # FIX: Target entropy = -|A|/2 per paper (Appendix C, Table 3)
        self.target_entropy = -float(act_dim) * 0.5
        self.log_alpha = torch.tensor(
            math.log(0.01), requires_grad=True, device=self.device
        )

        # ---- Оптимизаторы ----
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)
        self.critic_opt = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        self.alpha_opt = Adam([self.log_alpha], lr=lr)

        # Learning rate decay (Section 5.1, Table 3)
        upd_steps = max(1, total_steps * self.utd)
        lambda_lr = lambda s: 1.0 - min(s / upd_steps, 1.0) * (1.0 - lr_final / lr)
        self.policy_sch = lr_scheduler.LambdaLR(self.policy_opt, lambda_lr)
        self.critic_sch = lr_scheduler.LambdaLR(self.critic_opt, lambda_lr)

        # Reward scaler (Section 4.3, Eq.19)
        self.reward_scaler = RewardScaler(gamma=self.gamma, vmax=vmax)

        # FIX: Collect ALL NormalizedDense.W parameters explicitly
        # SacAgent is not nn.Module, so can't use .modules()
        self._weight_params = []

        def collect_weights(module):
            weights = []
            for m in module.modules():
                if isinstance(m, NormalizedDense):
                    weights.append(m.W)
            return weights

        self._weight_params.extend(collect_weights(self.policy))
        self._weight_params.extend(collect_weights(self.critic1))
        self._weight_params.extend(collect_weights(self.critic2))

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """
        Получить действие политики.

        Возвращает:
            np.ndarray в диапазоне [-act_limit, act_limit] (tanh-пространство).
            При act_limit=1.0 — в [-1, 1].
        """
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if deterministic:
            mu, _ = self.policy(obs_t)
            action = torch.tanh(mu)
        else:
            action, _ = self.policy.sample(obs_t)
        return (action * self.act_limit).squeeze(0).cpu().numpy()

    def _project_weights(self):
        """Project all weight matrices back to unit sphere (Section 4.4, Eq.20)"""
        with torch.no_grad():
            for p in self._weight_params:
                p.data = l2n(p.data, dim=1)

    def update(self, replay_buffer, obs=None, act=None):
        """
        Один env-step запускает UTD обновлений критика и одно обновление политики.
        Paper Section 3.1, 4

        FIX: Returns losses dict for training loop.
        FIX: Update running stats with full batch, not single sample.
        """
        losses_dict = {
            'critic_loss': 0.0,
            'policy_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': self.alpha
        }

        for _ in range(self.utd):
            # Sample batch FIRST
            o, a, r, no, d = replay_buffer.sample(self.batch_size)

            # FIX: Update running statistics with FULL BATCH (not single sample)
            # Paper Section 3.2 - running stats should be updated with batch data
            self.policy.enc.update_stats(o)
            oa = torch.cat([o, a], dim=-1)
            self.critic1.enc.update_stats(oa)
            self.critic2.enc.update_stats(oa)

            with torch.no_grad():
                # Target actions and log probs
                na, nlogp = self.policy.sample(no)
                na_scaled = na * self.act_limit

                # Target distributions from both critics
                np1 = self.target_critic1(no, na_scaled)
                np2 = self.target_critic2(no, na_scaled)

                # Clipped double Q (Section 3.1)
                ev1 = (np1 * self.target_critic1.atoms).sum(-1, keepdim=True)
                ev2 = (np2 * self.target_critic2.atoms).sum(-1, keepdim=True)
                nprobs = torch.where((ev1 <= ev2).expand_as(np1), np1, np2)

                # Entropy bonus (Section 3.1, Eq.1)
                ent_bonus = -self.alpha * nlogp

                # Target distributions
                tgt1 = self.critic1.bellman_target(r, d, nprobs, self.gamma, ent_bonus)
                tgt2 = self.critic2.bellman_target(r, d, nprobs, self.gamma, ent_bonus)

            # Critic loss (KL divergence) (Section 4.3)
            p1 = self.critic1(o, a)
            p2 = self.critic2(o, a)
            loss_c = (
                -(tgt1 * (p1 + 1e-8).log()).sum(-1).mean() +
                -(tgt2 * (p2 + 1e-8).log()).sum(-1).mean()
            )

            self.critic_opt.zero_grad()
            loss_c.backward()
            self.critic_opt.step()
            self.critic_sch.step()

            losses_dict['critic_loss'] = loss_c.item()

        # Soft update target networks (Section 3.1)
        soft_update(self.critic1, self.target_critic1, self.tau)
        soft_update(self.critic2, self.target_critic2, self.tau)

        # Policy update (Section 3.1, Eq.1)
        o, a, r, no, d = replay_buffer.sample(self.batch_size)
        pi, logp = self.policy.sample(o)
        pi_scaled = pi * self.act_limit
        q1 = self.critic1.qval(o, pi_scaled)
        q2 = self.critic2.qval(o, pi_scaled)
        q_min = torch.min(q1, q2)
        loss_pi = (self.alpha * logp - q_min).mean()

        self.policy_opt.zero_grad()
        loss_pi.backward()
        self.policy_opt.step()
        self.policy_sch.step()

        losses_dict['policy_loss'] = loss_pi.item()

        # Alpha update (Appendix C)
        with torch.no_grad():
            _, logp2 = self.policy.sample(o)
        loss_alpha = (self.log_alpha * (-logp2 - self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        loss_alpha.backward()
        self.alpha_opt.step()

        self._project_weights()

        losses_dict['alpha_loss'] = loss_alpha.item()
        losses_dict['alpha'] = self.alpha

        # FIX: Return losses dict
        return losses_dict

    # ---------------------------------------------------------------------
    # Сохранение / загрузка чекпоинтов
    # ---------------------------------------------------------------------
    def save(self, path: str, episode: int,
             episode_rewards: list, episode_detections: list):
        torch.save({
            'эпизод':                       episode,
            'policy_state_dict':            self.policy.state_dict(),
            'critic1_state_dict':           self.critic1.state_dict(),
            'critic2_state_dict':           self.critic2.state_dict(),
            'target_critic1_state_dict':    self.target_critic1.state_dict(),
            'target_critic2_state_dict':    self.target_critic2.state_dict(),
            'policy_opt':                   self.policy_opt.state_dict(),
            'critic_opt':                   self.critic_opt.state_dict(),
            'alpha_opt':                    self.alpha_opt.state_dict(),
            'policy_sch':                   self.policy_sch.state_dict(),
            'critic_sch':                   self.critic_sch.state_dict(),
            'log_alpha':                    self.log_alpha.detach().cpu(),
            'награды_по_эпизодам':          episode_rewards,
            'детекции_по_эпизодам':         episode_detections,
        }, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.critic1.load_state_dict(ckpt['critic1_state_dict'])
        self.critic2.load_state_dict(ckpt['critic2_state_dict'])
        self.target_critic1.load_state_dict(ckpt['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(ckpt['target_critic2_state_dict'])
        self.policy_opt.load_state_dict(ckpt['policy_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
        if 'policy_sch' in ckpt:
            self.policy_sch.load_state_dict(ckpt['policy_sch'])
            self.critic_sch.load_state_dict(ckpt['critic_sch'])

        # log_alpha — переписываем in-place чтобы не сломать привязанный оптимизатор
        with torch.no_grad():
            self.log_alpha.copy_(ckpt['log_alpha'].to(self.device))
        return ckpt


# Алиас для обратной совместимости со старыми импортами SACAgent
SACAgent = SacAgent
