"""
Утилиты для надёжной передачи данных по TCP.
Протокол: 4 байта big-endian (длина полезной нагрузки) + pickle-сериализованные данные.
"""

import pickle
import socket
import struct


def send_data(sock: socket.socket, data) -> bool:
    """Сериализовать и отправить данные через TCP сокет. Возвращает True при успехе."""
    try:
        serialized = pickle.dumps(data)
        size       = struct.pack('>I', len(serialized))
        sock.sendall(size + serialized)
        return True
    except Exception as e:
        print(f"[ОШИБКА] send_data: {e}")
        return False


def recv_data(sock: socket.socket, timeout: float = None):
    """
    Принять и десериализовать данные из TCP сокета.

    Аргументы:
        sock:    подключённый сокет
        timeout: таймаут чтения в секундах (None = блокирующий)

    Возвращает:
        Десериализованный объект или None при ошибке / таймауте
    """
    if timeout is not None:
        sock.settimeout(timeout)

    try:
        raw_size = b''
        while len(raw_size) < 4:
            chunk = sock.recv(4 - len(raw_size))
            if not chunk:
                return None
            raw_size += chunk

        data_size = struct.unpack('>I', raw_size)[0]

        if data_size > 500 * 1024 * 1024:
            print(f"[ОШИБКА] recv_data: слишком большой пакет ({data_size} байт)")
            return None

        data = b''
        while len(data) < data_size:
            chunk = sock.recv(min(65536, data_size - len(data)))
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
        if timeout is not None:
            sock.settimeout(None)
