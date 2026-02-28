"""
TCP сервер для Raspberry Pi / Orange Pi.
Обязанности:
  1. Отправить два изображения окружения клиенту (ПК)
  2. Принять LED-паттерн от клиента
  3. Отрисовать паттерн на диодах (GPIO или симуляция)
  4. Отправить подтверждение клиенту
"""

import os
import pickle
import socket
import struct
import time

import cv2
import numpy as np


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================
class ServerConfig:
    HOST = '0.0.0.0'
    PORT = 5000

    IMAGE1_PATH = 'img/image1.jpg'
    IMAGE2_PATH = 'img/image2.jpg'

    SIMULATE_LED_RENDER_TIME = 0.5   # секунд


# =============================================================================
# УТИЛИТЫ СОКЕТА
# =============================================================================
def send_data(conn: socket.socket, data) -> bool:
    try:
        serialized = pickle.dumps(data)
        conn.sendall(struct.pack('>I', len(serialized)) + serialized)
        return True
    except Exception as e:
        print(f"[ОШИБКА] send_data: {e}")
        return False


def recv_data(conn: socket.socket, timeout: float = None):
    if timeout:
        conn.settimeout(timeout)
    try:
        raw_size = b''
        while len(raw_size) < 4:
            chunk = conn.recv(4 - len(raw_size))
            if not chunk:
                return None
            raw_size += chunk

        data_size = struct.unpack('>I', raw_size)[0]
        if data_size > 500 * 1024 * 1024:
            print(f"[ОШИБКА] Слишком большой пакет: {data_size} байт")
            return None

        data = b''
        while len(data) < data_size:
            chunk = conn.recv(min(65536, data_size - len(data)))
            if not chunk:
                return None
            data += chunk

        return pickle.loads(data)
    except socket.timeout:
        print("[ОШИБКА] recv_data: таймаут")
        return None
    except Exception as e:
        print(f"[ОШИБКА] recv_data: {e}")
        return None
    finally:
        if timeout:
            conn.settimeout(None)


# =============================================================================
# КОНТРОЛЛЕР ДИОДОВ
# =============================================================================
class LEDController:
    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        режим = "симуляция" if simulate else "GPIO"
        print(f"[INFO] LEDController инициализирован в режиме: {режим}")

    def render_pattern(self, pattern: np.ndarray, render_time: float = 0.5) -> bool:
        if self.simulate:
            print(f"[INFO] Отрисовка паттерна: "
                  f"размер={len(pattern)}  мин={pattern.min()}  "
                  f"макс={pattern.max()}  среднее={pattern.mean():.1f}")
            time.sleep(render_time)
            print("[INFO] Паттерн отрисован.")
        else:
            # TODO: реализовать управление GPIO
            pass
        return True


# =============================================================================
# КАМЕРА ОКРУЖЕНИЯ
# =============================================================================
class EnvironmentCamera:
    def __init__(self, simulate: bool = True,
                 img1_path: str = ServerConfig.IMAGE1_PATH,
                 img2_path: str = ServerConfig.IMAGE2_PATH):
        self.simulate  = simulate
        self.img1_path = img1_path
        self.img2_path = img2_path

        if simulate:
            for path in (img1_path, img2_path):
                if not os.path.exists(path):
                    print(f"[ПРЕДУПРЕЖДЕНИЕ] Создаём тестовое изображение: {path}")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    cv2.imwrite(path, np.random.randint(
                        0, 255, (720, 720, 3), dtype=np.uint8))
            print("[INFO] EnvironmentCamera инициализирована в режиме симуляции")
        else:
            print("[INFO] EnvironmentCamera инициализирована с реальными камерами")

    def capture_images(self):
        if self.simulate:
            img1 = cv2.imread(self.img1_path)
            img2 = cv2.imread(self.img2_path)
            if img1 is None or img2 is None:
                print("[ОШИБКА] Не удалось загрузить изображения окружения")
                return None, None
            return img1, img2
        # TODO: реализовать захват с реальной камеры
        return None, None


# =============================================================================
# СЕРВЕР
# =============================================================================
def run_server():
    print("=" * 70)
    print("СЕРВЕР RASPBERRY PI / ORANGE PI")
    print("=" * 70)

    led_controller = LEDController(simulate=True)
    env_camera     = EnvironmentCamera(simulate=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        s.bind((ServerConfig.HOST, ServerConfig.PORT))
        s.listen(1)

        print(f"[INFO] Слушаем {ServerConfig.HOST}:{ServerConfig.PORT}")
        print("[INFO] Ожидание подключения клиента...")

        conn, addr = s.accept()
        with conn:
            print(f"[INFO] Клиент подключён: {addr}")
            episode = 0

            try:
                while True:
                    episode += 1
                    print(f"\n[Эпизод {episode}]" + "-" * 50)

                    # 1. Захват изображений
                    img1, img2 = env_camera.capture_images()
                    if img1 is None:
                        print("[ОШИБКА] Захват изображений не удался. Остановка.")
                        break
                    print(f"[INFO] Изображения захвачены: {img1.shape}, {img2.shape}")

                    # 2. Отправка изображений клиенту
                    if not send_data(conn, (img1, img2)):
                        print("[ОШИБКА] Не удалось отправить изображения.")
                        break
                    print("[INFO] Изображения отправлены.")

                    # 3. Получение LED-паттерна
                    pattern = recv_data(conn, timeout=60)
                    if pattern is None:
                        print("[ОШИБКА] Паттерн от клиента не получен.")
                        break
                    pattern_arr = np.array(pattern)
                    print(f"[INFO] Паттерн получен: форма={pattern_arr.shape}  "
                          f"мин={pattern_arr.min()}  макс={pattern_arr.max()}")

                    # 4. Отрисовка паттерна
                    success = led_controller.render_pattern(
                        pattern_arr,
                        render_time=ServerConfig.SIMULATE_LED_RENDER_TIME
                    )

                    # 5. Отправка подтверждения
                    response = {
                        'rendered':  success,
                        'episode':   episode,
                        'timestamp': time.time(),
                    }
                    if not send_data(conn, response):
                        print("[ОШИБКА] Не удалось отправить подтверждение.")
                        break
                    print(f"[INFO] Эпизод {episode} завершён.")

            except KeyboardInterrupt:
                print("\n[INFO] Сервер остановлен пользователем.")
            except Exception as e:
                print(f"[ОШИБКА] Исключение на сервере: {e}")
            finally:
                print("[INFO] Закрытие соединения.")


if __name__ == '__main__':
    run_server()
