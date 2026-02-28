"""
Основной цикл обучения SAC для оптимизации adversarial LED-паттернов.

Стратегия исследования:
  1. Шум на состояние: гауссовский шум добавляется к состоянию ДО подачи в актор.
                       Заставляет политику исследовать разнообразные паттерны.
  2. Шум на действие:  малый гауссовский шум добавляется к выходу актора.
                       Обеспечивает тонкую пертурбацию вокруг среднего политики.

Оба параметра задаются через STATE_NOISE_STD и ACTION_NOISE_STD в конфиге.
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from config import TrainingConfig as cfg
from env import AdversarialJacketEnv
from models import SACAgent


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
        f.write("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА\n")
        f.write("=" * 70 + "\n")
        for key, val in sorted(vars(cfg).items()):
            if not key.startswith('_'):
                f.write(f"{key} = {val}\n")
        f.write("=" * 70 + "\n")
        f.write("эпизод,награда,детекция,avg100_награда,avg100_детекция,"
                "critic_loss,actor_loss,alpha_loss\n")


# =============================================================================
# ОБУЧЕНИЕ
# =============================================================================
def train():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR,   exist_ok=True)
    if cfg.SAVE_DATASET:
        os.makedirs(cfg.DATASET_DIR, exist_ok=True)

    # Вывод конфига в консоль
    _print_config()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Устройство: {device}")

    # -- Окружение --
    env = AdversarialJacketEnv(cfg)
    if not env.connect_to_rpi():
        print("[ОШИБКА] Не удалось подключиться к RPi. Завершение.")
        return

    state_dim  = env.state_dim
    action_dim = env.action_dim

    # -- Агент --
    print("[INFO] Инициализация SAC агента...")
    agent = SACAgent(state_dim, action_dim, device, cfg)
    print("[INFO] SAC агент готов.")

    episode_rewards    = []
    episode_detections = []
    last_critic_loss   = None
    last_actor_loss    = None
    last_alpha_loss    = None

    log_file = os.path.join(
        cfg.LOGS_DIR,
        f'обучение_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    _write_config_header(log_file)

    print("=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 70 + "\n")

    state, raw_images = env.reset()
    if state is None:
        print("[ОШИБКА] Не удалось получить начальное состояние. Завершение.")
        return

    for episode in range(1, cfg.NUM_EPISODES + 1):
        print(f"\n[Эпизод {episode:05d}/{cfg.NUM_EPISODES}]" + "-" * 40)

        episode_reward    = 0.0
        episode_detection = 0.0

        for step in range(cfg.MAX_STEPS_PER_EPISODE):

            # -- Шум на состояние: добавляем ДО подачи в актор --
            noisy_state = state + np.random.normal(0, cfg.STATE_NOISE_STD, size=state.shape)

            # -- Выбор действия из зашумлённого состояния --
            action = agent.select_action(noisy_state)

            # -- Шум на действие: малая пертурбация ПОСЛЕ выхода актора --
            if cfg.ACTION_NOISE_STD > 0:
                action = action + np.random.normal(0, cfg.ACTION_NOISE_STD, size=action.shape)
                action = np.clip(action, 0.0, cfg.MAX_BRIGHTNESS / 255.0)

            reward, done, info = env.step(action, episode=episode, step_num=step)

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

            # В буфер кладём чистое состояние (без шума)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if done:
                break

            if len(agent.replay_buffer) >= cfg.LEARNING_STARTS:
                c_loss, a_loss, al_loss = agent.update(cfg.BATCH_SIZE)
                if c_loss is not None:
                    last_critic_loss = c_loss
                    last_actor_loss  = a_loss
                    last_alpha_loss  = al_loss
                    if cfg.VERBOSE:
                        print(f"  [ОБУЧЕНИЕ] critic={c_loss:.4f}  "
                              f"actor={a_loss:.4f}  alpha={al_loss:.4f}")

            if cfg.SAVE_DATASET and raw_images is not None:
                _save_dataset_step(episode, step, raw_images, info, action,
                                   episode_detection, reward, env)

            state = next_state
            print(f"  [Шаг {step + 1}] награда={reward:.4f}  детекция={episode_detection:.4f}")

        episode_rewards.append(episode_reward)
        episode_detections.append(episode_detection)
        avg_r = np.mean(episode_rewards[-100:])
        avg_d = np.mean(episode_detections[-100:])

        print(f"  награда={episode_reward:.4f}  детекция={episode_detection:.4f}  "
              f"avg100_награда={avg_r:.4f}  avg100_детекция={avg_d:.4f}  "
              f"буфер={len(agent.replay_buffer)}")

        if episode % cfg.SAVE_MODEL_EVERY == 0:
            path = os.path.join(cfg.MODELS_DIR, f'sac_эпизод_{episode:05d}.pth')
            agent.save(path, episode, episode_rewards, episode_detections)
            print(f"[INFO] Чекпоинт сохранён: {path}")

        cl  = f"{last_critic_loss:.4f}" if last_critic_loss is not None else "н/д"
        al  = f"{last_actor_loss:.4f}"  if last_actor_loss  is not None else "н/д"
        all_ = f"{last_alpha_loss:.4f}" if last_alpha_loss  is not None else "н/д"
        with open(log_file, 'a') as f:
            f.write(f"{episode},{episode_reward:.4f},{episode_detection:.4f},"
                    f"{avg_r:.4f},{avg_d:.4f},{cl},{al},{all_}\n")

    env.close()
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


def _save_dataset_step(episode, step, raw_images, info, action, detection, reward, env):
    """Сохранить изображения и метаданные для одного шага обучения."""
    ep_dir = os.path.join(cfg.DATASET_DIR, f'эпизод_{episode:05d}')
    os.makedirs(ep_dir, exist_ok=True)

    cv2.imwrite(os.path.join(ep_dir, 'env_img1.jpg'), raw_images[0])
    cv2.imwrite(os.path.join(ep_dir, 'env_img2.jpg'), raw_images[1])
    cv2.imwrite(os.path.join(ep_dir, 'pc_camera.jpg'), info['frame'])

    if cfg.SAVE_VISUALIZATION and env.last_visualization is not None:
        cv2.imwrite(os.path.join(ep_dir, 'yolo_detection.jpg'), env.last_visualization)

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
