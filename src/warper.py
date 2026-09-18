"""
Image Warper Module for Intelligent Panorama Builder.

Computes global composite panorama canvas size, accounts for negative translation offsets,
estimates memory before allocation, enforces safety limits, and applies perspective/affine transformations.
"""

from dataclasses import dataclass
from typing import Tuple, List
import cv2
import numpy as np
from src.input_handler import PanoramaError


class WarpingError(PanoramaError):
    """Exception raised when perspective warping fails or canvas exceeds memory limits."""
    pass


@dataclass
class WarpResult:
    """Dataclass storing warped image layers and translation matrix metadata."""
    warped_img1: np.ndarray       # Base reference image warped onto composite canvas
    warped_img2: np.ndarray       # Target image warped onto composite canvas
    translation_matrix: np.ndarray# 3x3 translation offset matrix
    canvas_shape: Tuple[int, int] # (canvas_height, canvas_width)
    offset_x: int
    offset_y: int


@dataclass
class MultiCanvasBounds:
    """Dataclass storing global multi-image canvas dimensions and translation offset matrix."""
    canvas_width: int
    canvas_height: int
    x_min: int
    y_min: int
    translation_matrix: np.ndarray
    estimated_memory_mb: float


class ImageWarper:
    """Manages perspective transformation, global canvas bounds, and memory safety validation."""

    def __init__(
        self,
        max_canvas_width: int = 25000,
        max_canvas_height: int = 15000,
        max_canvas_pixels: int = 100_000_000,
        max_memory_mb: float = 800.0
    ):
        self.max_canvas_width = max_canvas_width
        self.max_canvas_height = max_canvas_height
        self.max_canvas_pixels = max_canvas_pixels
        self.max_memory_mb = max_memory_mb

    @staticmethod
    def estimate_canvas_memory_mb(width: int, height: int, bytes_per_pixel: int = 3, num_buffers: int = 2) -> float:
        """Estimates required memory in Megabytes for a given canvas dimension."""
        total_pixels = float(width * height)
        total_bytes = total_pixels * bytes_per_pixel * num_buffers
        return total_bytes / (1024.0 * 1024.0)

    def validate_canvas_dimensions(self, width: int, height: int):
        """
        Enforces canvas safety dimensions and memory limits before allocation.
        """
        if width <= 0 or height <= 0:
            raise WarpingError(f"Calculated non-positive panorama canvas dimensions: ({width} x {height}).")

        if width > self.max_canvas_width or height > self.max_canvas_height:
            raise WarpingError(
                f"Panorama canvas dimensions ({width} x {height}) exceed safety limits "
                f"(Max: {self.max_canvas_width} x {self.max_canvas_height}). "
                "Transformation may have severe keystone/perspective runaway."
            )

        total_pixels = width * height
        if total_pixels > self.max_canvas_pixels:
            raise WarpingError(
                f"Total canvas pixel count ({total_pixels:,}) exceeds maximum limit ({self.max_canvas_pixels:,})."
            )

        est_ram = self.estimate_canvas_memory_mb(width, height)
        if est_ram > self.max_memory_mb:
            raise WarpingError(
                f"Estimated memory for panorama canvas ({est_ram:.1f} MB) exceeds safety threshold ({self.max_memory_mb:.1f} MB)."
            )

    @staticmethod
    def calculate_canvas_bounds(
        img1_shape: Tuple[int, int],
        img2_shape: Tuple[int, int],
        H: np.ndarray
    ) -> Tuple[int, int, int, int, np.ndarray]:
        """
        Calculates composite canvas bounding box for a 2-image pair.

        Args:
            img1_shape: (height, width) of base reference image.
            img2_shape: (height, width) of image being warped.
            H: 3x3 homography matrix mapping img2 coords to img1 coords.

        Returns:
            Tuple of (x_min, y_min, canvas_width, canvas_height, translation_matrix).
        """
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]

        pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts_img2_transformed = cv2.perspectiveTransform(pts_img2, H)

        all_pts = np.concatenate((pts_img1, pts_img2_transformed), axis=0)
        [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

        offset_x = -x_min if x_min < 0 else 0
        offset_y = -y_min if y_min < 0 else 0

        translation_matrix = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ], dtype=np.float64)

        canvas_width = int(x_max - x_min)
        canvas_height = int(y_max - y_min)

        return int(x_min), int(y_min), canvas_width, canvas_height, translation_matrix

    def calculate_multi_canvas_bounds(
        self,
        image_shapes: List[Tuple[int, int]],
        H_to_ref_list: List[np.ndarray]
    ) -> MultiCanvasBounds:
        """
        Calculates unified global composite canvas bounds across N images mapped to a central reference frame.

        Args:
            image_shapes: List of (height, width) for each image.
            H_to_ref_list: List of 3x3 matrices mapping each image to global reference coordinate frame.

        Returns:
            MultiCanvasBounds dataclass.
        """
        all_corners = []
        for shape, H in zip(image_shapes, H_to_ref_list):
            h, w = shape[:2]
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            proj_corners = cv2.perspectiveTransform(corners, H)
            all_corners.append(proj_corners)

        all_pts = np.concatenate(all_corners, axis=0)
        x_min = int(np.floor(all_pts[:, 0, 0].min()))
        y_min = int(np.floor(all_pts[:, 0, 1].min()))
        x_max = int(np.ceil(all_pts[:, 0, 0].max()))
        y_max = int(np.ceil(all_pts[:, 0, 1].max()))

        canvas_w = int(x_max - x_min)
        canvas_h = int(y_max - y_min)

        # Validate limits and memory
        self.validate_canvas_dimensions(canvas_w, canvas_h)
        est_ram = self.estimate_canvas_memory_mb(canvas_w, canvas_h)

        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)

        return MultiCanvasBounds(
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            x_min=x_min,
            y_min=y_min,
            translation_matrix=translation_matrix,
            estimated_memory_mb=est_ram
        )

    def warp_pair(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        H: np.ndarray
    ) -> WarpResult:
        """
        Warps 2-image pair (backward compatibility).
        """
        if img1 is None or img2 is None or H is None:
            raise WarpingError("Invalid image or homography matrix provided for warping.")

        x_min, y_min, canvas_w, canvas_h, H_trans = self.calculate_canvas_bounds(
            img1.shape, img2.shape, H
        )
        self.validate_canvas_dimensions(canvas_w, canvas_h)

        H_composite = H_trans @ H

        warped_img2 = cv2.warpPerspective(
            img2, H_composite, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR
        )
        warped_img1 = cv2.warpPerspective(
            img1, H_trans, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR
        )

        warped_img1 = np.ascontiguousarray(warped_img1)
        warped_img2 = np.ascontiguousarray(warped_img2)

        return WarpResult(
            warped_img1=warped_img1,
            warped_img2=warped_img2,
            translation_matrix=H_trans,
            canvas_shape=(canvas_h, canvas_w),
            offset_x=-x_min if x_min < 0 else 0,
            offset_y=-y_min if y_min < 0 else 0
        )
