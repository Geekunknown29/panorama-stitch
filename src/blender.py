"""
Image Blender Module for Intelligent Panorama Builder.

Combines aligned, warped image layers into a single seamless image using distance-transform
feathered alpha blending to eliminate seam boundaries without unnecessary memory allocation.
"""

from typing import List, Tuple
import cv2
import numpy as np
from src.input_handler import PanoramaError


class BlendingError(PanoramaError):
    """Exception raised when image blending fails."""
    pass


class ImageBlender:
    """Manages multi-image blending, distance-transform smoothing, and memory-efficient accumulation."""

    @staticmethod
    def create_distance_mask(img: np.ndarray) -> np.ndarray:
        """
        Creates a normalized distance-transform weight map for non-zero image pixels.
        Pixels near the border have lower weights (0.0), while central pixels have higher weights (1.0).

        Args:
            img: BGR image array.

        Returns:
            Single-channel float32 weight map with values in [0.0, 1.0].
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        binary_mask = (gray > 0).astype(np.uint8)

        if not np.any(binary_mask):
            return np.zeros_like(gray, dtype=np.float32)

        # Distance transform computes distance to nearest zero boundary pixel
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        max_val = np.max(dist)

        if max_val > 0:
            dist_normalized = dist / max_val
        else:
            dist_normalized = binary_mask.astype(np.float32)

        return dist_normalized

    def blend(
        self,
        warped_img1: np.ndarray,
        warped_img2: np.ndarray
    ) -> np.ndarray:
        """
        Blends two aligned canvas images using distance-transform weighted alpha blending.

        Args:
            warped_img1: First BGR image layer on composite canvas.
            warped_img2: Second BGR image layer on composite canvas.

        Returns:
            Blended BGR panorama image array.
        """
        if warped_img1.shape != warped_img2.shape:
            raise BlendingError(
                f"Image shape mismatch in blender: {warped_img1.shape} vs {warped_img2.shape}"
            )

        mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0)
        mask2 = (cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0)

        dist1 = self.create_distance_mask(warped_img1)
        dist2 = self.create_distance_mask(warped_img2)

        overlap = mask1 & mask2

        w1 = np.zeros_like(dist1, dtype=np.float32)
        w2 = np.zeros_like(dist2, dtype=np.float32)

        w1[mask1 & ~overlap] = 1.0
        w2[mask2 & ~overlap] = 1.0

        sum_dist = dist1 + dist2
        valid_sum = (overlap & (sum_dist > 0))

        w1[valid_sum] = dist1[valid_sum] / sum_dist[valid_sum]
        w2[valid_sum] = dist2[valid_sum] / sum_dist[valid_sum]

        w1_3d = np.repeat(w1[:, :, np.newaxis], 3, axis=2)
        w2_3d = np.repeat(w2[:, :, np.newaxis], 3, axis=2)

        blended_float = warped_img1.astype(np.float32) * w1_3d + warped_img2.astype(np.float32) * w2_3d
        blended_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)

        del mask1, mask2, dist1, dist2, overlap, w1, w2, w1_3d, w2_3d, blended_float
        return np.ascontiguousarray(blended_uint8)

    def blend_multi(
        self,
        images: List[np.ndarray],
        H_composites: List[np.ndarray],
        canvas_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Performs memory-efficient single-pass distance-weighted accumulation for N images.

        Args:
            images: List of original BGR images.
            H_composites: List of 3x3 matrices mapping each image onto global canvas.
            canvas_shape: (canvas_height, canvas_width).

        Returns:
            Blended uint8 BGR panorama image array.
        """
        canvas_h, canvas_w = canvas_shape[:2]

        accum_color = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for img, H in zip(images, H_composites):
            # Warp color image directly
            warped_img = cv2.warpPerspective(
                img, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR
            )

            # Compute pre-warp distance map to avoid large full-canvas distance transform
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary_mask = (gray > 0).astype(np.uint8)
            dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            max_d = float(dist.max())
            dist_norm = (dist / max_d) if max_d > 0 else dist.astype(np.float32)

            warped_dist = cv2.warpPerspective(
                dist_norm, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR
            )

            valid = (warped_dist > 0.001)

            for c in range(3):
                accum_color[valid, c] += warped_img[valid, c] * warped_dist[valid]
            accum_weight[valid] += warped_dist[valid]

            # Explicit memory release for intermediate warped copies
            del warped_img, warped_dist, gray, binary_mask, dist, dist_norm, valid

        valid_pixels = accum_weight > 0
        panorama_uint8 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for c in range(3):
            panorama_uint8[valid_pixels, c] = np.clip(
                accum_color[valid_pixels, c] / accum_weight[valid_pixels], 0, 255
            ).astype(np.uint8)

        del accum_color, accum_weight, valid_pixels
        return np.ascontiguousarray(panorama_uint8)
