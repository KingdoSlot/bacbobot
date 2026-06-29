"""Screen capture utilities."""

import cv2
import numpy as np
from mss import mss

# Default monitor configuration (full HD)
DEFAULT_MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

_sct = None


def _get_sct():
    """Lazily initialize the mss screen capture instance."""
    global _sct
    if _sct is None:
        _sct = mss()
    return _sct


def capturar_tela(monitor=None):
    """Capture the screen and return a BGR frame.

    Args:
        monitor: Monitor region dict with top, left, width, height.
                 Defaults to full 1920x1080 screen.

    Returns:
        numpy.ndarray: BGR image frame.
    """
    if monitor is None:
        monitor = DEFAULT_MONITOR
    sct_img = _get_sct().grab(monitor)
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
