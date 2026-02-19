"""
Visualization module — manages all OpenCV windows.
Window positions are applied once on first display.
"""

import cv2
import numpy as np


class Visualizer:
    """
    Centralized manager for all OpenCV display windows.

    Args:
        window_positions: dict mapping window name to (x, y) screen position
        delay:            cv2.waitKey delay in milliseconds
    """

    def __init__(self, window_positions: dict, delay: int):
        self.window_positions = window_positions
        self.delay            = delay
        self._positioned: set = set()

    # -------------------------------------------------------------------------
    def _show(self, name: str, img: np.ndarray):
        cv2.imshow(name, img)
        self._move_once(name)
        cv2.waitKey(self.delay)

    def _move_once(self, name: str):
        if name not in self._positioned and name in self.window_positions:
            x, y = self.window_positions[name]
            try:
                cv2.moveWindow(name, x, y)
            except Exception:
                pass
            self._positioned.add(name)

    def destroy_all(self):
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # LED pattern
    # -------------------------------------------------------------------------
    def show_led_pattern(self, pattern: np.ndarray, n_leds: int, max_brightness: int):
        """
        Renders the LED pattern as a grid image.

        Args:
            pattern:        flat numpy array [N_LEDS * 3], values in [0, max_action]
            n_leds:         number of LEDs
            max_brightness: brightness ceiling in 0-255 range (for info display)
        """
        try:
            arr       = np.array(pattern).reshape(-1, 3)
            grid_size = int(np.ceil(np.sqrt(n_leds)))

            grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
            for i in range(min(n_leds, grid_size * grid_size)):
                grid[i // grid_size, i % grid_size] = arr[i]

            display = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)
            display = (display * 255).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

            real    = arr * 255
            info    = np.zeros((80, 400, 3), dtype=np.uint8)
            cv2.putText(info, f'LED Pattern  ({n_leds} LEDs)',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info,
                        f'Mean: {real.mean():.1f}   Max: {real.max():.1f}   '
                        f'(limit={max_brightness})',
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            self._show('LED Pattern', np.vstack([info, display]))
        except Exception as e:
            print(f"[WARN] LED pattern visualization failed: {e}")

    # -------------------------------------------------------------------------
    # RPi camera images
    # -------------------------------------------------------------------------
    def show_rpi_images(self, img1: np.ndarray, img2: np.ndarray):
        """Display images from both RPi cameras in separate windows."""
        cv2.imshow('RPi Camera 1', self._prepare_rpi(img1, 'RPi Camera 1'))
        self._move_once('RPi Camera 1')
        cv2.imshow('RPi Camera 2', self._prepare_rpi(img2, 'RPi Camera 2'))
        self._move_once('RPi Camera 2')
        cv2.waitKey(1)

    @staticmethod
    def _prepare_rpi(img: np.ndarray, label: str,
                     w: int = 640, h: int = 480) -> np.ndarray:
        display = cv2.resize(img, (w, h))
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        cv2.putText(display, label, (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display,
                    f'mean={img.mean():.1f}  std={img.std():.1f}',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return display

    # -------------------------------------------------------------------------
    # YOLO detection
    # -------------------------------------------------------------------------
    def show_yolo(
        self,
        frame: np.ndarray,
        person_boxes: list,
        person_confidences: list,
        max_confidence: float,
    ) -> np.ndarray:
        """
        Draws person bounding boxes and detection status on the frame.

        Returns:
            Annotated frame as np.ndarray
        """
        vis = frame.copy()

        for box, conf in zip(person_boxes, person_confidences):
            x1, y1, x2, y2 = map(int, box)
            color = (int(255 * (1 - conf)), int(255 * (1 - conf)), int(255 * conf))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            label = f'Person: {conf:.2%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Header bar
        header = (f'Max Detection: {max_confidence:.2%} | '
                  f'Persons: {len(person_confidences)}')
        cv2.rectangle(vis, (10, 10), (660, 62), (0, 0, 0), -1)
        cv2.putText(vis, header, (20, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Status label
        if max_confidence < 0.3:
            status_color, status_text = (0, 200, 0),   'LOW DETECTION'
        elif max_confidence < 0.6:
            status_color, status_text = (0, 165, 255), 'MED DETECTION'
        else:
            status_color, status_text = (0, 0, 220),   'HIGH DETECTION'

        cv2.putText(vis, status_text, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        self._show('YOLO Detection - Adversarial Jacket', vis)
        return vis
