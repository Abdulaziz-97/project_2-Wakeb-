"""
Image utility functions for the car detection pipeline.

Provides letterbox resizing that preserves aspect ratio and pads with gray
(value 114) to reach the exact target square size.
"""

import numpy as np
import cv2


def letterbox(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Resize *img* with letterboxing to ``(target_size, target_size, 3)``.

    The image is scaled so its longest side fits within *target_size*, then
    padded symmetrically with gray (114) on the shorter axis.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image of shape ``(H, W, 3)``, dtype ``uint8``.
    target_size : int
        Desired square output dimension.

    Returns
    -------
    np.ndarray
        Padded image of shape ``(target_size, target_size, 3)``, dtype ``uint8``.
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas filled with gray (114)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # Center the resized image on the canvas
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas
