"""
Training loop for SAC-based adversarial LED pattern optimization.

Key change vs. original:
  The SAC agent no longer predicts N_LEDS*3 = 29 184 values directly.
  Instead it works in the latent space of a pretrained SD VAE:

      state → SAC Actor → latent (624,) → VAE Decoder → LED pattern (29 184,)

  The VAE is fully frozen. Only the SAC actor/critic are trained.
  Compression ratio: ~47×  (29 184 → 624)
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from config import TrainingConfig as cfg
from env import AdversarialJacketEnv
from led_latent import LEDPatternCodec, LATENT_DIM        # ← NEW
from models import ImageEncoder, SACAgent


def train():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR,   exist_ok=True)
    if cfg.SAVE_DATASET:
        os.makedirs(cfg.DATASET_DIR, exist_ok=True)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # -- Pretrained VAE codec (frozen) -----------------------------------------
    print("[INFO] Loading pretrained LED pattern VAE codec...")
    codec = LEDPatternCodec(device)
    # action_dim for SAC is now the VAE latent dimension, not N_LEDS*3
    action_dim = LATENT_DIM                                   # 624  (was 29 184)
    print(f"[INFO] SAC action_dim = {action_dim}  "
          f"(was {cfg.N_LEDS * 3}, {cfg.N_LEDS * 3 / action_dim:.0f}× compression)")

    # -- Image encoder ----------------------------------------------------------
    print("[INFO] Loading MobileNetV2 encoder...")
    encoder = ImageEncoder(output_dim=cfg.ENCODER_DIM, freeze=cfg.ENCODER_FREEZE).to(device)
    encoder.eval()
    print(f"[INFO] Encoder ready. state_dim = {cfg.ENCODER_DIM} x 2 = {cfg.ENCODER_DIM * 2}")

    # -- Environment ------------------------------------------------------------
    env = AdversarialJacketEnv(cfg, encoder=encoder)
    if not env.connect_to_rpi():
        print("[ERROR] Failed to connect to RPi. Aborting.")
        return

    state_dim = cfg.ENCODER_DIM * 2
    print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}")

    # -- SAC agent (operates in latent space) -----------------------------------
    print("[INFO] Initializing SAC agent in latent space...")
    agent = SACAgent(state_dim, action_dim, device, cfg)
    print("[INFO] SAC agent ready.")

    episode_rewards    = []
    episode_detections = []
    log_file = os.path.join(
        cfg.LOGS_DIR,
        f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    print("\n" + "=" * 70)
    print("TRAINING START  (latent-space SAC + pretrained VAE decoder)")
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

            # 1. SAC selects a latent code  (shape: (LATENT_DIM,), values in [-1,1])
            latent_action = agent.select_action(state)

            # 2. Decode latent → full LED pattern  (shape: (N_LEDS*3,), values in [0,1])
            led_pattern = codec.decode(latent_action)

            # 3. Send real pattern to RPi, get reward
            reward, done, info = env.step(led_pattern, episode=episode, step_num=step)
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

            # 4. Store *latent* action in replay buffer (not the raw LED pattern)
            agent.replay_buffer.push(state, latent_action, reward, next_state, done)

            if done:
                break

            # 5. Update SAC
            if len(agent.replay_buffer) >= cfg.LEARNING_STARTS:
                critic_loss, actor_loss, alpha_loss = agent.update(cfg.BATCH_SIZE)
                if critic_loss is not None and cfg.VERBOSE:
                    print(f"  [TRAIN] critic={critic_loss:.4f}  "
                          f"actor={actor_loss:.4f}  alpha={alpha_loss:.4f}")

            # 6. Optionally save dataset
            if cfg.SAVE_DATASET and raw_images is not None:
                _save_dataset_step(
                    episode, step, raw_images, info, latent_action,
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


def _save_dataset_step(episode, step, raw_images, info, latent_action,
                       detection, reward, env):
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
        # Store latent stats instead of full pattern
        'latent_mean':          float(latent_action.mean()),
        'latent_std':           float(latent_action.std()),
        'latent_min':           float(latent_action.min()),
        'latent_max':           float(latent_action.max()),
    }
    with open(os.path.join(ep_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    train()
