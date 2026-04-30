"""
Основной цикл обучения SimbaV2 SAC для оптимизации adversarial LED-паттернов.

Алгоритм перенесён из paper.py:
    - SimbaV2 архитектура (RSNorm + L2-нормализация весов на гиперсфере + LERP residual)
    - Distributional critic (C51)
    - Reward scaling по running статистикам дисконтированной отдачи
    - Tanh-bounded Gaussian policy + clipped double Q + learnable alpha

Преобразование пространства действий:
    Политика выдаёт действие в tanh-пространстве [-1, 1] (стандартный SAC).
    LED-яркость требует [0, max_brightness/255].
    Преобразование:  led = (tanh + 1) / 2 * max_brightness/255
    В replay buffer кладётся действие в tanh-пространстве —
    чтобы Q-сеть и политика работали в одном координатном пространстве.

Стратегия исследования:
    1. Warmup: первые WARMUP_STEPS шагов делаются равномерно случайные действия в tanh-пространстве.
    2. Энтропийная регуляризация SAC (alpha обучается автоматически).
    3. Опциональный шум на состояние/действие (STATE_NOISE_STD, ACTION_NOISE_STD).
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from config import TrainingConfig as cfg
from env import AdversarialJacketEnv
from models import SacAgent
from utils.replay_buffer import ReplayBuffer


# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================
def _print_config():
    """Вывести все параметры конфига в консоль."""
    print("\n" + "=" * 70)
    print("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА")
    print("=" * 70)
    for key, val in sorted(vars(cfg).items()):
        if not key.startswith('_'):
            print(f"  {key} = {val}")
    print("=" * 70 + "\n")


def _write_config_header(log_file: str):
    """Записать все параметры конфига в начало лог-файла."""
    with open(log_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА (SimbaV2 SAC)\n")
        f.write("=" * 70 + "\n")
        for key, val in sorted(vars(cfg).items()):
            if not key.startswith('_'):
                f.write(f"{key} = {val}\n")
        f.write("=" * 70 + "\n")
        f.write("эпизод,награда,детекция,avg100_награда,avg100_детекция,"
                "critic_loss,policy_loss,alpha_loss,alpha\n")


# =============================================================================
# ПРЕОБРАЗОВАНИЕ ПРОСТРАНСТВА ДЕЙСТВИЙ
# =============================================================================
def tanh_to_led(action_tanh: np.ndarray, max_action: float) -> np.ndarray:
    """
    Преобразовать действие из tanh-пространства [-1, 1]
    в LED-пространство [0, max_action] линейно.
    """
    return (action_tanh + 1.0) * 0.5 * max_action


# =============================================================================
# ОБУЧЕНИЕ
# =============================================================================
def train():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR,   exist_ok=True)
    if cfg.SAVE_DATASET:
        os.makedirs(cfg.DATASET_DIR, exist_ok=True)

    _print_config()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Устройство: {device}")

    # -- Окружение --
    env = AdversarialJacketEnv(cfg)
    if not env.connect_to_rpi():
        print("[ОШИБКА] Не удалось подключиться к RPi. Завершение.")
        return

    obs_dim    = env.state_dim
    act_dim    = env.action_dim
    max_action = cfg.MAX_BRIGHTNESS / 255.0

    # -- Конфиг для SimbaV2 SAC --
    sac_cfg = {
        "gamma":      cfg.GAMMA,
        "tau":        cfg.TAU,
        "batch_size": cfg.BATCH_SIZE,
        "natoms":     cfg.NATOMS,
        "vmin":       cfg.VMIN,
        "vmax":       cfg.VMAX,
        "actor_d":    cfg.ACTOR_D,
        "actor_nb":   cfg.ACTOR_NB,
        "critic_d":   cfg.CRITIC_D,
        "critic_nb":  cfg.CRITIC_NB,
        "lr":         cfg.LR,
        "lr_final":   cfg.LR_FINAL,
        "utd":        cfg.UTD,
        "cshift":     cfg.CSHIFT,
    }

    total_steps = cfg.NUM_EPISODES * cfg.MAX_STEPS_PER_EPISODE

    # -- Агент --
    print("[INFO] Инициализация SimbaV2 SAC агента...")
    # act_limit=1.0: политика выдаёт действие в [-1, 1], преобразование в LED делаем в train.py
    agent = SacAgent(
        obs_dim, act_dim,
        act_limit=1.0,
        cfg=sac_cfg,
        total_steps=total_steps,
        device=device,
    )
    print("[INFO] SAC агент готов.")

    # -- Replay buffer --
    replay = ReplayBuffer(obs_dim, act_dim,
                          max_size=cfg.BUFFER_SIZE, device=device)

    episode_rewards    = []
    episode_detections = []
    last_losses = {'critic_loss': None, 'policy_loss': None,
                   'alpha_loss': None, 'alpha': None}

    log_file = os.path.join(
        cfg.LOGS_DIR,
        f'simbav2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    _write_config_header(log_file)

    print("=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ (SimbaV2 SAC)")
    print("=" * 70 + "\n")

    state, raw_images = env.reset()
    if state is None:
        print("[ОШИБКА] Не удалось получить начальное состояние. Завершение.")
        return

    global_step = 0

    for episode in range(1, cfg.NUM_EPISODES + 1):
        print(f"\n[Эпизод {episode:05d}/{cfg.NUM_EPISODES}]" + "-" * 40)

        episode_reward    = 0.0
        episode_detection = 0.0

        # Reward scaler сбрасывается в начале каждого эпизода (Section 4.3)
        agent.reward_scaler.reset()

        for step in range(cfg.MAX_STEPS_PER_EPISODE):
            global_step += 1

            # -- Шум на состояние (опционально) --
            if cfg.STATE_NOISE_STD > 0:
                noisy_state = state + np.random.normal(
                    0, cfg.STATE_NOISE_STD, size=state.shape
                )
            else:
                noisy_state = state

            # -- Выбор действия в tanh-пространстве [-1, 1] --
            if global_step < cfg.WARMUP_STEPS:
                # Warmup: равномерные случайные действия
                action_tanh = np.random.uniform(
                    -1.0, 1.0, size=act_dim
                ).astype(np.float32)
            else:
                action_tanh = agent.act(noisy_state.astype(np.float32))

            # -- Опциональный шум на действие в tanh-пространстве --
            if cfg.ACTION_NOISE_STD > 0:
                action_tanh = action_tanh + np.random.normal(
                    0, cfg.ACTION_NOISE_STD, size=action_tanh.shape
                )
                action_tanh = np.clip(action_tanh, -1.0, 1.0).astype(np.float32)

            # -- Преобразование в LED-пространство [0, max_action] для среды --
            action_led = tanh_to_led(action_tanh, max_action).astype(np.float32)

            reward, done, info = env.step(action_led, episode=episode, step_num=step)

            if done and not info:
                print("[ОШИБКА] Шаг завершился неудачей.")
                break

            episode_detection  = info['detection_confidence']
            episode_reward    += reward

            next_state, raw_images = env.reset()
            if next_state is None:
                print("[ОШИБКА] Не удалось получить следующее состояние.")
                done = True
                break

            # -- Reward scaling (Section 4.3, Eq.17-19) --
            scaled_reward = agent.reward_scaler.scale(float(reward))

            # В буфер кладём ТАNH-действие (соответствует распределению политики)
            replay.add(state, action_tanh, scaled_reward, next_state, float(done))

            if done:
                break

            # -- Обновление сетей --
            if global_step >= cfg.WARMUP_STEPS and replay.size >= cfg.BATCH_SIZE:
                losses = agent.update(replay)
                last_losses = losses
                if cfg.VERBOSE:
                    print(f"  [ОБУЧЕНИЕ] critic={losses['critic_loss']:.4f}  "
                          f"policy={losses['policy_loss']:.4f}  "
                          f"alpha_loss={losses['alpha_loss']:.4f}  "
                          f"alpha={losses['alpha']:.4f}")

            if cfg.SAVE_DATASET and raw_images is not None:
                _save_dataset_step(episode, step, raw_images, info, action_led,
                                   episode_detection, reward, env)

            state = next_state
            print(f"  [Шаг {step + 1}] награда={reward:.4f}  "
                  f"scaled={scaled_reward:.4f}  детекция={episode_detection:.4f}")

        episode_rewards.append(episode_reward)
        episode_detections.append(episode_detection)
        avg_r = np.mean(episode_rewards[-100:])
        avg_d = np.mean(episode_detections[-100:])

        print(f"  награда={episode_reward:.4f}  детекция={episode_detection:.4f}  "
              f"avg100_награда={avg_r:.4f}  avg100_детекция={avg_d:.4f}  "
              f"буфер={replay.size}")

        if episode % cfg.SAVE_MODEL_EVERY == 0:
            path = os.path.join(cfg.MODELS_DIR,
                                f'simbav2_sac_эпизод_{episode:05d}.pth')
            agent.save(path, episode, episode_rewards, episode_detections)
            print(f"[INFO] Чекпоинт сохранён: {path}")

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "н/д"

        with open(log_file, 'a') as f:
            f.write(f"{episode},{episode_reward:.4f},{episode_detection:.4f},"
                    f"{avg_r:.4f},{avg_d:.4f},"
                    f"{_fmt(last_losses['critic_loss'])},"
                    f"{_fmt(last_losses['policy_loss'])},"
                    f"{_fmt(last_losses['alpha_loss'])},"
                    f"{_fmt(last_losses['alpha'])}\n")

    env.close()
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


def _save_dataset_step(episode, step, raw_images, info, action,
                       detection, reward, env):
    """Сохранить изображения и метаданные для одного шага обучения."""
    ep_dir = os.path.join(cfg.DATASET_DIR, f'эпизод_{episode:05d}')
    os.makedirs(ep_dir, exist_ok=True)

    cv2.imwrite(os.path.join(ep_dir, 'env_img1.jpg'),  raw_images[0])
    cv2.imwrite(os.path.join(ep_dir, 'env_img2.jpg'),  raw_images[1])
    cv2.imwrite(os.path.join(ep_dir, 'pc_camera.jpg'), info['frame'])

    if cfg.SAVE_VISUALIZATION and env.last_visualization is not None:
        cv2.imwrite(os.path.join(ep_dir, 'yolo_detection.jpg'),
                    env.last_visualization)

    metadata = {
        'эпизод':              episode,
        'шаг':                 step,
        'уверенность_детекции': float(detection),
        'награда':             float(reward),
        'среднее_действия':    float(action.mean()),
        'std_действия':        float(action.std()),
        'размер_суперпикселя': cfg.SUPERPIXEL_SIZE,
    }
    with open(os.path.join(ep_dir, 'метаданные.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    train()
