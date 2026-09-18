"""
Panorama Cropper Module for Intelligent Panorama Builder.

Crops empty black outer canvas borders from stitched panoramas while preserving valid content.
"""

import cv2
import numpy as np


class PanoramaCropper:
    """Manages automatic cropping of outer empty black canvas borders."""

    @staticmethod
    def crop_black_borders(img: np.ndarray, margin: int = 2) -> np.ndarray:
        """
        Crops empty zero-value (black) borders around the image.

        Args:
            img: Input BGR image array.
            margin: Safety pixel padding around non-zero ROI.

        Returns:
            Cropped BGR image array.
        """
        if img is None or img.size == 0:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img

        # Find largest bounding rectangle enclosing non-zero content
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        h_img, w_img = img.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)

        if x2 > x1 and y2 > y1:
            return np.ascontiguousarray(img[y1:y2, x1:x2].copy())

        return img
