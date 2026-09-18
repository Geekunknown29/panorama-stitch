"""
Preprocessing Module for Intelligent Panorama Builder.

Handles safe image loading, aspect-ratio preserving resizing for working copies,
and grayscale conversion for feature detection.
"""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np
from src.input_handler import InputValidationError


@dataclass
class ProcessedImage:
    """Dataclass storing original and working copies of an image along with metadata."""
    filepath: str
    filename: str
    color_img: np.ndarray      # BGR Working image
    gray_img: np.ndarray       # Grayscale image for feature detection
    original_shape: Tuple[int, int, int]  # (height, width, channels)
    working_shape: Tuple[int, int, int]   # (height, width, channels)
    scale_factor: float        # Scale applied to original image


class ImagePreprocessor:
    """Handles image preprocessing for feature detection and alignment."""

    def __init__(self, max_dimension: int = 1600):
        """
        Args:
            max_dimension: Maximum allowed dimension (width or height) for working copy.
        """
        self.max_dimension = max_dimension

    def preprocess(self, filepath: str) -> ProcessedImage:
        """
        Reads an image file safely, resizes it if it exceeds max_dimension while preserving
        aspect ratio, and converts it to grayscale for feature extraction.

        Args:
            filepath: Path to the image file.

        Returns:
            ProcessedImage instance containing metadata and image arrays.
        """
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise InputValidationError(f"Unable to read image at path: '{filepath}'")

        orig_h, orig_w = img_bgr.shape[:2]
        orig_channels = img_bgr.shape[2] if len(img_bgr.shape) > 2 else 1

        scale_factor = 1.0
        max_dim = max(orig_h, orig_w)

        if max_dim > self.max_dimension:
            scale_factor = self.max_dimension / float(max_dim)
            new_w = max(1, int(orig_w * scale_factor))
            new_h = max(1, int(orig_h * scale_factor))
            working_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            working_bgr = img_bgr.copy()

        # Convert working image to grayscale for feature extraction
        if len(working_bgr.shape) == 3 and working_bgr.shape[2] == 3:
            gray_img = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
        elif len(working_bgr.shape) == 2:
            gray_img = working_bgr.copy()
            working_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            gray_img = cv2.cvtColor(working_bgr, cv2.COLOR_BGRA2GRAY)
            working_bgr = cv2.cvtColor(working_bgr, cv2.COLOR_BGRA2BGR)

        # Ensure arrays are C-contiguous
        working_bgr = np.ascontiguousarray(working_bgr)
        gray_img = np.ascontiguousarray(gray_img)

        import os
        filename = os.path.basename(filepath)

        return ProcessedImage(
            filepath=filepath,
            filename=filename,
            color_img=working_bgr,
            gray_img=gray_img,
            original_shape=(orig_h, orig_w, orig_channels),
            working_shape=working_bgr.shape,
            scale_factor=scale_factor
        )
