"""
TCP server for Raspberry Pi.
Responsibilities:
  1. Send two environment images to the PC client
  2. Receive LED pattern from the PC client
  3. Render the pattern on LEDs (GPIO or simulation)
  4. Send confirmation back to the client
"""

import os
import pickle
import socket
import struct
import time

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================
class ServerConfig:
    HOST = '0.0.0.0'
    PORT = 5000

    IMAGE1_PATH = 'img/image1.jpg'
    IMAGE2_PATH = 'img/image2.jpg'

    SIMULATE_LED_RENDER_TIME = 0.5   # seconds


# =============================================================================
# SOCKET UTILITIES
# =============================================================================
def send_data(conn: socket.socket, data) -> bool:
    try:
        serialized = pickle.dumps(data)
        conn.sendall(struct.pack('>I', len(serialized)) + serialized)
        return True
    except Exception as e:
        print(f"[ERROR] send_data: {e}")
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
            print(f"[ERROR] Packet too large: {data_size} bytes")
            return None

        data = b''
        while len(data) < data_size:
            chunk = conn.recv(min(65536, data_size - len(data)))
            if not chunk:
                return None
            data += chunk

        return pickle.loads(data)
    except socket.timeout:
        print("[ERROR] recv_data: timeout")
        return None
    except Exception as e:
        print(f"[ERROR] recv_data: {e}")
        return None
    finally:
        if timeout:
            conn.settimeout(None)


# =============================================================================
# LED CONTROLLER
# =============================================================================
class LEDController:
    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        mode = "simulation" if simulate else "GPIO"
        print(f"[INFO] LEDController initialized in {mode} mode")

    def render_pattern(self, pattern: np.ndarray, render_time: float = 0.5) -> bool:
        if self.simulate:
            print(f"[INFO] Rendering pattern: "
                  f"size={len(pattern)}  min={pattern.min()}  "
                  f"max={pattern.max()}  mean={pattern.mean():.1f}")
            time.sleep(render_time)
            print("[INFO] Pattern rendered.")
        else:
            # TODO: implement GPIO control
            # TODO: implement GPIO control
            # TODO: implement GPIO control
            # e.g. set_led_colors(pattern.reshape(-1, 3))
            pass
        return True


# =============================================================================
# ENVIRONMENT CAMERA
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
                    print(f"[WARN] Creating placeholder image: {path}")
                    cv2.imwrite(path, np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8))
            print("[INFO] EnvironmentCamera initialized in simulation mode")
        else:
            print("[INFO] EnvironmentCamera initialized with real cameras")

    def capture_images(self):
        if self.simulate:
            img1 = cv2.imread(self.img1_path)
            img2 = cv2.imread(self.img2_path)
            if img1 is None or img2 is None:
                print("[ERROR] Failed to load environment images")
                return None, None
            return img1, img2
        # TODO
        # TODO: implement real PiCamera capture
        # TODO
        return None, None


# =============================================================================
# SERVER
# =============================================================================
def run_server():
    print("=" * 70)
    print("RASPBERRY PI SERVER")
    print("=" * 70)

    led_controller = LEDController(simulate=False)
    env_camera     = EnvironmentCamera(simulate=False)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        s.bind((ServerConfig.HOST, ServerConfig.PORT))
        s.listen(1)

        print(f"[INFO] Listening on {ServerConfig.HOST}:{ServerConfig.PORT}")
        print("[INFO] Waiting for client connection...")

        conn, addr = s.accept()
        with conn:
            print(f"[INFO] Client connected: {addr}")
            episode = 0

            try:
                while True:
                    episode += 1
                    print(f"\n[Episode {episode}]" + "-" * 50)

                    # 1. Capture images
                    img1, img2 = env_camera.capture_images()
                    if img1 is None:
                        print("[ERROR] Image capture failed. Stopping.")
                        break
                    print(f"[INFO] Images captured: {img1.shape}, {img2.shape}")

                    # 2. Send images to client
                    if not send_data(conn, (img1, img2)):
                        print("[ERROR] Failed to send images.")
                        break
                    print("[INFO] Images sent.")

                    # 3. Receive LED pattern
                    pattern = recv_data(conn, timeout=60)
                    if pattern is None:
                        print("[ERROR] Did not receive LED pattern from client.")
                        break
                    pattern_arr = np.array(pattern)
                    print(f"[INFO] Pattern received: shape={pattern_arr.shape}  "
                          f"min={pattern_arr.min()}  max={pattern_arr.max()}")

                    # 4. Render pattern
                    success = led_controller.render_pattern(
                        pattern_arr,
                        render_time=ServerConfig.SIMULATE_LED_RENDER_TIME
                    )

                    # 5. Send confirmation
                    response = {
                        'rendered':  success,
                        'episode':   episode,
                        'timestamp': time.time(),
                    }
                    if not send_data(conn, response):
                        print("[ERROR] Failed to send confirmation.")
                        break
                    print(f"[INFO] Episode {episode} complete.")

            except KeyboardInterrupt:
                print("\n[INFO] Server stopped by user.")
            except Exception as e:
                print(f"[ERROR] Server exception: {e}")
            finally:
                print("[INFO] Closing connection.")


if __name__ == '__main__':
    run_server()
