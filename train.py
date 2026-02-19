"""
Training loop for SAC-based adversarial LED pattern optimization.
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from config import TrainingConfig as cfg
from env import AdversarialJacketEnv
from models import ImageEncoder, SACAgent


def train():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR,   exist_ok=True)
    if cfg.SAVE_DATASET:
        os.makedirs(cfg.DATASET_DIR, exist_ok=True)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # -- Encoder --
    print("[INFO] Loading MobileNetV2 encoder...")
    encoder = ImageEncoder(output_dim=cfg.ENCODER_DIM, freeze=cfg.ENCODER_FREEZE).to(device)
    encoder.eval()
    print(f"[INFO] Encoder ready. state_dim = {cfg.ENCODER_DIM} x 2 = {cfg.ENCODER_DIM * 2}")

    # -- Environment --
    env = AdversarialJacketEnv(cfg, encoder=encoder)
    if not env.connect_to_rpi():
        print("[ERROR] Failed to connect to RPi. Aborting.")
        return

    state_dim  = cfg.ENCODER_DIM * 2
    action_dim = cfg.N_LEDS * 3
    print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}")

    # -- Agent --
    print("[INFO] Initializing SAC agent...")
    agent = SACAgent(state_dim, action_dim, device, cfg)
    print("[INFO] SAC agent ready.")

    episode_rewards    = []
    episode_detections = []
    log_file = os.path.join(
        cfg.LOGS_DIR,
        f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70 + "\n")

    state, raw_images = env.reset()
    if state is None:
        print("[ERROR] Failed to obtain initial state. Aborting.")
        return

    for episode in range(1, cfg.NUM_EPISODES + 1):
        print(f"\n[Episode {episode:05d}/{cfg.NUM_EPISODES}]" + "-" * 40)

        episode_reward    = 0.0
        episode_detection = 0.0

        for step in range(cfg.MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)

            reward, done, info = env.step(action, episode=episode, step_num=step)
            if info is None:
                print("[ERROR] Step failed.")
                break

            episode_detection  = info['detection_confidence']
            episode_reward    += reward

            next_state, raw_images = env.reset()
            if next_state is None:
                print("[ERROR] Failed to obtain next state.")
                done = True
                break

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if done:
                break

            if len(agent.replay_buffer) >= cfg.LEARNING_STARTS:
                critic_loss, actor_loss, alpha_loss = agent.update(cfg.BATCH_SIZE)
                if critic_loss is not None and cfg.VERBOSE:
                    print(f"  [TRAIN] critic={critic_loss:.4f}  "
                          f"actor={actor_loss:.4f}  alpha={alpha_loss:.4f}")

            if cfg.SAVE_DATASET and raw_images is not None:
                _save_dataset_step(
                    episode, step, raw_images, info, action,
                    episode_detection, reward, env
                )

            state = next_state
            print(f"  [Step {step+1}] reward={reward:.4f}  detection={episode_detection:.4f}")

        episode_rewards.append(episode_reward)
        episode_detections.append(episode_detection)
        avg_r = np.mean(episode_rewards[-100:])
        avg_d = np.mean(episode_detections[-100:])

        print(f"  reward={episode_reward:.4f}  detection={episode_detection:.4f}  "
              f"avg100_reward={avg_r:.4f}  avg100_detection={avg_d:.4f}  "
              f"buffer={len(agent.replay_buffer)}")

        if episode % cfg.SAVE_MODEL_EVERY == 0:
            path = os.path.join(cfg.MODELS_DIR, f'sac_episode_{episode:05d}.pth')
            agent.save(path, episode, episode_rewards, episode_detections,
                       encoder_proj_state=encoder.proj.state_dict())
            print(f"[INFO] Checkpoint saved: {path}")

        with open(log_file, 'a') as f:
            f.write(f"{episode},{episode_reward:.4f},{episode_detection:.4f},"
                    f"{avg_r:.4f},{avg_d:.4f}\n")

    env.close()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


def _save_dataset_step(episode, step, raw_images, info, action, detection, reward, env):
    """Save images and metadata for a single training step."""
    ep_dir = os.path.join(cfg.DATASET_DIR, f'episode_{episode:05d}')
    os.makedirs(ep_dir, exist_ok=True)

    cv2.imwrite(os.path.join(ep_dir, 'env_img1.jpg'), raw_images[0])
    cv2.imwrite(os.path.join(ep_dir, 'env_img2.jpg'), raw_images[1])
    cv2.imwrite(os.path.join(ep_dir, 'pc_camera.jpg'), info['frame'])

    if cfg.SAVE_VISUALIZATION and env.last_visualization is not None:
        cv2.imwrite(os.path.join(ep_dir, 'yolo_detection.jpg'), env.last_visualization)

    metadata = {
        'episode':              episode,
        'step':                 step,
        'detection_confidence': float(detection),
        'reward':               float(reward),
        'action_mean':          float(action.mean()),
        'action_std':           float(action.std()),
    }
    with open(os.path.join(ep_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    train()
