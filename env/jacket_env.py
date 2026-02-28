"""
AdversarialJacketEnv — RL окружение для обучения adversarial LED-паттернов.

Состояние:  среднее и стандартное отклонение по каналам двух RPi камер (12 значений)
Действие:   суперпиксельный LED-паттерн; каждое значение управляет SUPERPIXEL_SIZE² диодами.

Расположение диодов:
    37 плат, каждая 16×16, змейкой:
    чётные строки:   слева направо  (0, 1, 2, ... 15)
    нечётные строки: справа налево  (31, 30, ... 16)

Маппинг суперпикселей:
    Сетка 16×16 делится на (16/S)×(16/S) блоков размером S×S диодов.
    Одно RGB-значение назначается всем диодам блока.
    Разворачивание учитывает змейку, поэтому блоки остаются пространственно связными.
"""

import math
import socket
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.socket_utils import send_data, recv_data
from utils.visualization import Visualizer


class AdversarialJacketEnv:

    def __init__(self, cfg):
        self.cfg         = cfg
        self.socket      = None
        self.pc_camera   = None
        self.last_visualization = None
        self.prev_action = None

        self.board_h       = cfg.BOARD_H
        self.board_w       = cfg.BOARD_W
        self.n_boards      = cfg.N_BOARDS
        self.n_leds        = cfg.BOARD_H * cfg.BOARD_W * cfg.N_BOARDS
        self.superpixel_size = cfg.SUPERPIXEL_SIZE
        self.max_action    = cfg.MAX_BRIGHTNESS / 255.0

        assert cfg.BOARD_H % cfg.SUPERPIXEL_SIZE == 0, \
            f"SUPERPIXEL_SIZE ({cfg.SUPERPIXEL_SIZE}) должен нацело делить BOARD_H ({cfg.BOARD_H})"
        assert cfg.BOARD_W % cfg.SUPERPIXEL_SIZE == 0, \
            f"SUPERPIXEL_SIZE ({cfg.SUPERPIXEL_SIZE}) должен нацело делить BOARD_W ({cfg.BOARD_W})"

        S    = cfg.SUPERPIXEL_SIZE
        sp_h = cfg.BOARD_H // S
        sp_w = cfg.BOARD_W // S
        self.n_superpixels = cfg.N_BOARDS * sp_h * sp_w
        self.action_dim    = self.n_superpixels * 3
        self.state_dim     = 12   # mean+std по каналам для 2 камер

        # Предвычисляем маппинг суперпиксель → индексы диодов
        self.led_map = self._build_led_map()

        self.vis = Visualizer(
            window_positions=cfg.WINDOW_POSITIONS,
            delay=cfg.VISUALIZATION_DELAY,
        )

        print("[INFO] Загрузка модели YOLO...")
        self.yolo = YOLO(cfg.YOLO_MODEL)
        print("[INFO] Модель YOLO загружена.")

        print("[INFO] Инициализация камеры ПК...")
        self.pc_camera = cv2.VideoCapture(cfg.PC_CAMERA_INDEX)
        if not self.pc_camera.isOpened():
            raise RuntimeError("[ОШИБКА] Не удалось открыть камеру ПК.")
        print(f"[INFO] Прогрев камеры ({cfg.CAMERA_WARMUP_TIME}с)...")
        time.sleep(cfg.CAMERA_WARMUP_TIME)
        for _ in range(5):
            self.pc_camera.read()
        print("[INFO] Камера ПК готова.")

        print(f"[INFO] Схема диодов: {self.n_boards} плат × {self.board_h}×{self.board_w} "
              f"= {self.n_leds} диодов всего")
        print(f"[INFO] Суперпиксель: {S}×{S}  ->  {self.n_superpixels} суперпикселей  "
              f"action_dim={self.action_dim}  "
              f"(сжатие: {self.n_leds * 3 / self.action_dim:.1f}x)")

    # =========================================================================
    # Маппинг диодов
    # =========================================================================
    def _build_led_map(self) -> list:
        """
        Предвычислить маппинг: индекс_суперпикселя -> список абсолютных индексов диодов.

        Змейка на плате:
            чётная строка:   столбец 0..W-1  (слева направо)
            нечётная строка: столбец W-1..0  (справа налево)
        """
        S = self.superpixel_size
        H = self.board_h
        W = self.board_w
        sp_h = H // S
        sp_w = W // S

        def snake_led(board: int, row: int, col: int) -> int:
            local_col = col if row % 2 == 0 else (W - 1 - col)
            return board * H * W + row * W + local_col

        led_map = []
        for board in range(self.n_boards):
            for sp_row in range(sp_h):
                for sp_col in range(sp_w):
                    leds = []
                    for r in range(sp_row * S, (sp_row + 1) * S):
                        for c in range(sp_col * S, (sp_col + 1) * S):
                            leds.append(snake_led(board, r, c))
                    led_map.append(leds)

        return led_map

    def _expand_pattern(self, action: np.ndarray) -> np.ndarray:
        """
        Развернуть суперпиксельное действие в полный паттерн с учётом змейки.

        Аргументы:
            action: np.ndarray [n_superpixels * 3], значения в [0, max_action]

        Возвращает:
            np.ndarray [n_leds * 3] uint8, значения в [0, 255]
        """
        full   = np.zeros(self.n_leds * 3, dtype=np.uint8)
        colors = (action.reshape(-1, 3) * 255).astype(np.uint8)

        for sp_idx, led_indices in enumerate(self.led_map):
            color = colors[sp_idx]
            for led_idx in led_indices:
                full[led_idx * 3: led_idx * 3 + 3] = color

        return full

    # =========================================================================
    # Соединение
    # =========================================================================
    def connect_to_rpi(self) -> bool:
        print(f"[INFO] Подключение к {self.cfg.SERVER_HOST}:{self.cfg.SERVER_PORT}...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        try:
            self.socket.connect((self.cfg.SERVER_HOST, self.cfg.SERVER_PORT))
            print("[INFO] Подключено к RPi.")
            return True
        except Exception as e:
            print(f"[ОШИБКА] Не удалось подключиться: {e}")
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
            print("[ПРЕДУПРЕЖДЕНИЕ] Соединение потеряно. Переподключение...")
            self.socket.close()
            self.socket = None
            return self.connect_to_rpi()

    # =========================================================================
    # Состояние
    # =========================================================================
    def get_state(self):
        """
        Получить два изображения с RPi и вычислить среднее/std по каналам как состояние.

        Возвращает:
            (state np.ndarray [12], (img1, img2)) или None при ошибке
        """
        images = recv_data(self.socket, timeout=self.cfg.SOCKET_TIMEOUT)
        if images is None:
            print("[ОШИБКА] Не удалось получить изображения от RPi.")
            self.socket.close()
            self.socket = None
            return None

        img1, img2 = images

        if self.cfg.SHOW_ENV_IMAGES:
            self.vis.show_rpi_images(img1, img2)

        def image_stats(img):
            resized = cv2.resize(img, (self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE))
            norm    = resized.astype(np.float32) / 255.0
            return np.concatenate([
                norm.mean(axis=(0, 1)),
                norm.std(axis=(0, 1)),
            ])

        state = np.concatenate([image_stats(img1), image_stats(img2)])
        return state, (img1, img2)

    # =========================================================================
    # Передача паттерна
    # =========================================================================
    def send_pattern_and_wait(self, action: np.ndarray) -> bool:
        """Развернуть суперпиксели, отправить полный паттерн на RPi, ждать подтверждения."""
        pattern_full = self._expand_pattern(action)
        if not send_data(self.socket, pattern_full):
            return False
        response = recv_data(self.socket, timeout=self.cfg.PATTERN_RENDER_TIMEOUT)
        return response is not None and response.get('rendered', False)

    # =========================================================================
    # Детекция
    # =========================================================================
    def capture_and_detect(self):
        """
        Захватить кадр с камеры ПК и запустить детекцию человека через YOLO.

        Возвращает:
            (max_confidence: float, frame: np.ndarray) или (None, None) при ошибке
        """
        ret, frame = self.pc_camera.read()
        if not ret:
            print("[ОШИБКА] Не удалось захватить кадр с камеры ПК.")
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
    # RL интерфейс
    # =========================================================================
    def step(self, action: np.ndarray, episode: int = 0, step_num: int = 0):
        """
        Выполнить один шаг окружения.

        Возвращает:
            (reward, done, info) — всегда 3 значения.
            При ошибке: (-10.0, True, {})
        """
        if self.cfg.SHOW_LED_PATTERN:
            self.vis.show_led_pattern(action, self.n_superpixels,
                                      self.cfg.MAX_BRIGHTNESS, self.superpixel_size)

        if not self.send_pattern_and_wait(action):
            print("[ОШИБКА] Не удалось отправить паттерн или получить подтверждение.")
            return -10.0, True, {}

        # Усреднение уверенности детекции по N кадрам для снижения шума в награде
        conf_values = []
        last_frame  = None
        for _ in range(self.cfg.DETECTION_AVG_FRAMES):
            conf, frame = self.capture_and_detect()
            if conf is None:
                print("[ОШИБКА] Детекция завершилась неудачей.")
                return -10.0, True, {}
            conf_values.append(conf)
            last_frame = frame

        detection_conf = float(np.mean(conf_values))

        # Составляющие награды
        reward_det  = self.cfg.DETECTION_THRESHOLD - detection_conf
        reward_var  = float(np.var(action))
        reward_diff = float(np.mean((action - self.prev_action) ** 2)) \
                      if self.prev_action is not None else 0.0

        reward = (self.cfg.REWARD_A * reward_det
                  + self.cfg.REWARD_B * reward_var
                  + self.cfg.REWARD_C * reward_diff)

        if self.cfg.VERBOSE:
            print(f"  [Награда] детекция={self.cfg.REWARD_A * reward_det:.3f}  "
                  f"дисперсия={self.cfg.REWARD_B * reward_var:.3f}  "
                  f"отличие={self.cfg.REWARD_C * reward_diff:.3f}  "
                  f"итого={reward:.3f}  conf={detection_conf:.3f}")

        self.prev_action = action.copy()

        info = {'detection_confidence': detection_conf, 'frame': last_frame}
        return reward, False, info

    def reset(self):
        """
        Переподключиться при необходимости, сбросить prev_action,
        получить начальное состояние с RPi.

        Возвращает:
            (state, raw_images) или (None, None) при ошибке
        """
        self.prev_action = None
        if not self.ensure_connection():
            print("[ОШИБКА] Не удалось установить соединение с RPi.")
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
