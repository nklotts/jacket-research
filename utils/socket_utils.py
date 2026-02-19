"""
Reliable TCP socket utilities.
Protocol: 4-byte big-endian payload length followed by pickle-serialized data.
"""

import pickle
import socket
import struct


def send_data(sock: socket.socket, data) -> bool:
    """Serialize and send data over a TCP socket. Returns True on success."""
    try:
        serialized = pickle.dumps(data)
        size       = struct.pack('>I', len(serialized))
        sock.sendall(size + serialized)
        return True
    except Exception as e:
        print(f"[ERROR] send_data: {e}")
        return False


def recv_data(sock: socket.socket, timeout: float = None):
    """
    Receive and deserialize data from a TCP socket.

    Args:
        sock:    connected socket
        timeout: read timeout in seconds (None = blocking)

    Returns:
        Deserialized object, or None on error / timeout
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
            print(f"[ERROR] recv_data: packet too large ({data_size} bytes)")
            return None

        data = b''
        while len(data) < data_size:
            chunk = sock.recv(min(65536, data_size - len(data)))
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
        if timeout is not None:
            sock.settimeout(None)
