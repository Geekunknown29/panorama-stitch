"""
Feature Detector Module for Intelligent Panorama Builder.

Detects keypoints and extracts local visual feature descriptors using SIFT or ORB.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from src.input_handler import PanoramaError


class FeatureExtractionError(PanoramaError):
    """Exception raised when feature detection or description fails."""
    pass


class FeatureDetector:
    """Manages feature detection and descriptor extraction."""

    def __init__(self, algorithm: str = "sift", nfeatures: int = 4000):
        """
        Args:
            algorithm: Algorithm choice ('sift' or 'orb').
            nfeatures: Target maximum number of features to detect.
        """
        self.algorithm_name = algorithm.lower().strip()
        self.nfeatures = nfeatures

        if self.algorithm_name == "sift":
            if hasattr(cv2, "SIFT_create"):
                self.detector = cv2.SIFT_create(nfeatures=self.nfeatures)
            else:
                raise FeatureExtractionError("SIFT algorithm is not available in current OpenCV build.")
        elif self.algorithm_name == "orb":
            self.detector = cv2.ORB_create(nfeatures=self.nfeatures)
        else:
            raise FeatureExtractionError(
                f"Unsupported feature detector algorithm: '{algorithm}'. Choose 'sift' or 'orb'."
            )

    def detect_and_compute(
        self,
        gray_image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detects keypoints and computes descriptors for a grayscale image.

        Args:
            gray_image: Single-channel grayscale image array.

        Returns:
            Tuple of (list of cv2.KeyPoint, descriptor numpy array).

        Raises:
            FeatureExtractionError: If zero keypoints or descriptors are detected.
        """
        if gray_image is None or gray_image.size == 0:
            raise FeatureExtractionError("Empty or invalid image provided for feature extraction.")

        keypoints, descriptors = self.detector.detectAndCompute(gray_image, None)

        if keypoints is None or len(keypoints) == 0 or descriptors is None or len(descriptors) == 0:
            raise FeatureExtractionError(
                f"Feature extraction failed using {self.algorithm_name.upper()}. "
                "No valid keypoints or descriptors detected in image."
            )

        return keypoints, descriptors

    def draw_keypoints(
        self,
        color_image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        Renders keypoint visualization on the color image for debug output.

        Args:
            color_image: Input color image (BGR).
            keypoints: Detected keypoints list.

        Returns:
            BGR visualization image showing keypoint locations and orientations.
        """
        vis_img = cv2.drawKeypoints(
            color_image,
            keypoints,
            outImage=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return vis_img
