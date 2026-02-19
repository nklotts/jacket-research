"""
AdversarialJacketEnv — RL environment for adversarial LED pattern training.

Handles:
  - TCP connection to Raspberry Pi
  - State acquisition (MobileNetV2 embeddings of two RPi camera images)
  - LED pattern transmission to RPi
  - PC camera capture and YOLO person detection
  - Reward computation
"""

import socket
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from models.encoder import preprocess_image
from utils.socket_utils import send_data, recv_data
from utils.visualization import Visualizer


class AdversarialJacketEnv:

    def __init__(self, cfg, encoder: torch.nn.Module):
        """
        Args:
            cfg:     TrainingConfig instance
            encoder: Initialized ImageEncoder placed on the correct device
        """
        self.cfg     = cfg
        self.encoder = encoder
        self.device  = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

        self.socket    = None
        self.pc_camera = None
        self.last_visualization = None

        self.max_action = cfg.MAX_BRIGHTNESS / 255.0

        self.vis = Visualizer(
            window_positions=cfg.WINDOW_POSITIONS,
            delay=cfg.VISUALIZATION_DELAY,
        )

        print("[INFO] Loading YOLO model...")
        self.yolo = YOLO(cfg.YOLO_MODEL)
        if torch.cuda.is_available() and cfg.DEVICE == 'cuda':
            self.yolo.to(self.device)
        print("[INFO] YOLO model loaded.")

        print("[INFO] Initializing PC camera...")
        self.pc_camera = cv2.VideoCapture(cfg.PC_CAMERA_INDEX)
        if not self.pc_camera.isOpened():
            raise RuntimeError("[ERROR] Failed to open PC camera.")
        print(f"[INFO] Warming up camera ({cfg.CAMERA_WARMUP_TIME}s)...")
        time.sleep(cfg.CAMERA_WARMUP_TIME)
        for _ in range(5):
            self.pc_camera.read()
        print("[INFO] PC camera ready.")

    # =========================================================================
    # Connection
    # =========================================================================
    def connect_to_rpi(self) -> bool:
        print(f"[INFO] Connecting to {self.cfg.SERVER_HOST}:{self.cfg.SERVER_PORT}...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        try:
            self.socket.connect((self.cfg.SERVER_HOST, self.cfg.SERVER_PORT))
            print("[INFO] Connected to RPi.")
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    def ensure_connection(self) -> bool:
        if self.socket is None:
            return self.connect_to_rpi()
        try:
            self.socket.settimeout(0.1)
            self.socket.send(b'')
            self.socket.settimeout(None)
            return True
        except (socket.error, socket.timeout):
            print("[WARN] Connection lost. Reconnecting...")
            self.socket.close()
            self.socket = None
            return self.connect_to_rpi()

    # =========================================================================
    # State
    # =========================================================================
    def get_state(self):
        """
        Receives two images from RPi and encodes them via MobileNetV2.

        Returns:
            (state_features np.ndarray [ENCODER_DIM*2], (img1, img2)) or None on error
        """
        images = recv_data(self.socket, timeout=self.cfg.SOCKET_TIMEOUT)
        if images is None:
            print("[ERROR] Failed to receive images from RPi.")
            self.socket.close()
            self.socket = None
            return None

        img1, img2 = images

        if self.cfg.SHOW_ENV_IMAGES:
            self.vis.show_rpi_images(img1, img2)

        with torch.no_grad():
            t1   = preprocess_image(img1, self.device, self.cfg.IMAGENET_MEAN, self.cfg.IMAGENET_STD)
            t2   = preprocess_image(img2, self.device, self.cfg.IMAGENET_MEAN, self.cfg.IMAGENET_STD)
            emb1 = self.encoder(t1).cpu().numpy()[0]
            emb2 = self.encoder(t2).cpu().numpy()[0]

        state = np.concatenate([emb1, emb2])
        return state, (img1, img2)

    # =========================================================================
    # Pattern transmission
    # =========================================================================
    def send_pattern_and_wait(self, pattern: np.ndarray) -> bool:
        """
        Scales pattern from [0, max_action] to [0, 255] and sends to RPi.
        Blocks until render confirmation is received.
        """
        pattern_255 = (pattern * 255).astype(np.uint8)
        if not send_data(self.socket, pattern_255):
            return False
        response = recv_data(self.socket, timeout=self.cfg.PATTERN_RENDER_TIMEOUT)
        return response is not None and response.get('rendered', False)

    # =========================================================================
    # Detection
    # =========================================================================
    def capture_and_detect(self):
        """
        Captures a frame from the PC camera and runs YOLO person detection.

        Returns:
            (max_confidence: float, frame: np.ndarray) or (None, None) on error
        """
        ret, frame = self.pc_camera.read()
        if not ret:
            print("[ERROR] Failed to capture frame from PC camera.")
            return None, None

        results = self.yolo(frame, verbose=False)

        person_confidences = []
        person_boxes       = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                mask  = boxes.cls == self.cfg.TARGET_CLASS
                if mask.sum() > 0:
                    person_confidences.extend(boxes.conf[mask].cpu().numpy().tolist())
                    person_boxes.extend(boxes.xyxy[mask].cpu().numpy().tolist())

        max_confidence = max(person_confidences) if person_confidences else 0.0

        if self.cfg.SHOW_YOLO_DETECTIONS:
            vis_frame = self.vis.show_yolo(
                frame, person_boxes, person_confidences, max_confidence
            )
            if self.cfg.SAVE_VISUALIZATION:
                self.last_visualization = vis_frame

        return max_confidence, frame

    # =========================================================================
    # RL interface
    # =========================================================================
    def step(self, action: np.ndarray, episode: int = 0, step_num: int = 0):
        """
        Execute one environment step:
          1. Optionally visualize the LED pattern
          2. Transmit pattern to RPi and wait for render confirmation
          3. Capture PC camera frame and run YOLO detection
          4. Compute and return reward

        Returns:
            (reward, done, info) or (None, -10.0, True, {}) on failure
        """
        if self.cfg.SHOW_LED_PATTERN:
            self.vis.show_led_pattern(action, self.cfg.N_LEDS, self.cfg.MAX_BRIGHTNESS)

        if not self.send_pattern_and_wait(action):
            return None, -10.0, True, {}

        detection_conf, frame = self.capture_and_detect()
        if detection_conf is None:
            return None, -10.0, True, {}

        reward = -detection_conf
        info   = {'detection_confidence': detection_conf, 'frame': frame}
        return reward, False, info

    def reset(self):
        """
        Reconnects if needed, then retrieves the initial state from RPi.

        Returns:
            (state, raw_images) or (None, None) on failure
        """
        if not self.ensure_connection():
            print("[ERROR] Could not establish connection to RPi.")
            return None, None
        result = self.get_state()
        if result is None:
            return None, None
        return result

    def close(self):
        if self.socket:
            self.socket.close()
        if self.pc_camera:
            self.pc_camera.release()
        self.vis.destroy_all()
