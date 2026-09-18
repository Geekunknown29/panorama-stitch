"""
Utility and Logger Module for Intelligent Panorama Builder.

Handles debug artifact generation (keypoint visualizers, match visualizers, matrix exports)
and terminal summary formatting.
"""

import json
import os
import cv2
import numpy as np
from typing import Any, Dict, List, Optional
from src.evaluator import PerformanceMetrics


class DebugWriter:
    """Manages saving intermediate computer vision artifacts when --debug is enabled."""

    def __init__(self, debug_dir: str = "outputs/debug"):
        self.debug_dir = debug_dir
        self.pair_metrics: List[Dict[str, Any]] = []

    def ensure_dir(self) -> None:
        """Creates debug directory if it does not exist."""
        os.makedirs(self.debug_dir, exist_ok=True)

    def save_image(self, filename: str, img: np.ndarray) -> str:
        """Saves image array to debug folder, creating subdirectories as needed."""
        self.ensure_dir()
        filepath = os.path.join(self.debug_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, img)
        return filepath

    def save_homography(self, filename: str, H: np.ndarray, header_info: str = "") -> str:
        """Saves 3x3 homography matrix to text file."""
        self.ensure_dir()
        filepath = os.path.join(self.debug_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            if header_info:
                f.write(f"# {header_info}\n")
            f.write("# 3x3 Homography Matrix:\n")
            np.savetxt(f, H, fmt="%18.10f")
        return filepath

    def save_projected_corners(
        self,
        filename: str,
        corners: np.ndarray,
        header_info: str = ""
    ) -> str:
        """Saves projected corner coordinates to text file."""
        self.ensure_dir()
        filepath = os.path.join(self.debug_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        labels = ["TL", "BL", "BR", "TR"]
        with open(filepath, "w", encoding="utf-8") as f:
            if header_info:
                f.write(f"# {header_info}\n")
            for label, (x, y) in zip(labels, corners):
                f.write(f"{label}: ({x:.2f}, {y:.2f})\n")
        return filepath

    def save_pair_metrics(self, report_dict: Dict[str, Any]) -> None:
        """Accumulates pair quality metrics for debug metrics.json."""
        self.pair_metrics.append(report_dict)

    def save_metrics_json(self, extra: Optional[Dict[str, Any]] = None) -> str:
        """Saves accumulated debug pair metrics."""
        self.ensure_dir()
        filepath = os.path.join(self.debug_dir, "metrics.json")
        payload = {"pair_evaluations": self.pair_metrics}
        if extra:
            payload.update(extra)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return filepath


def print_terminal_summary(metrics: PerformanceMetrics, output_filepath: str) -> None:
    """Prints a clean, professional summary to the terminal."""
    d = metrics.to_dict()
    successful_pairs = len(metrics.inliers_per_pair)
    total_pairs = max(1, metrics.images_count - 1)

    print("========================================")
    print("INTELLIGENT PANORAMA BUILDER")
    print("========================================")
    print(f"Images processed: {metrics.images_count}")
    if metrics.excluded_images:
        print(f"Images excluded: {len(metrics.excluded_images)}")
        for name in metrics.excluded_images:
            print(f"  - {name}")
    print(f"Feature detector: {metrics.feature_detector.upper()}")
    print("\nTotal keypoints:")
    for name, kp_count in metrics.keypoints_per_image.items():
        print(f"  {name}: {kp_count}")

    print(f"\nSuccessful image pairs: {successful_pairs}/{total_pairs}")
    print(f"Average inlier ratio: {d['average_inlier_ratio_percent']}%")
    print(f"Processing time: {metrics.processing_time_seconds:.2f} seconds")

    w = d['panorama_dimensions']['width']
    h = d['panorama_dimensions']['height']
    print(f"\nPanorama:\n  Width: {w}\n  Height: {h}")
    print(f"\nStatus: {metrics.status}")
    print(f"\nOutput:\n{output_filepath}")
    print("========================================")
