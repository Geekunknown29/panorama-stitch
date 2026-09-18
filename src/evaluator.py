"""
Quantitative Quality Evaluator Module for Intelligent Panorama Builder.

Collects algorithmic performance metrics (keypoints, match counts, RANSAC inlier ratios,
reprojection errors, dimensions, and execution timing) and exports outputs/metrics.json.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
import os
import time


@dataclass
class PerformanceMetrics:
    """Dataclass storing pipeline evaluation metrics."""
    images_count: int = 0
    feature_detector: str = ""
    keypoints_per_image: Dict[str, int] = field(default_factory=dict)
    raw_matches_per_pair: Dict[str, int] = field(default_factory=dict)
    good_matches_per_pair: Dict[str, int] = field(default_factory=dict)
    inliers_per_pair: Dict[str, int] = field(default_factory=dict)
    inlier_ratios_per_pair: Dict[str, float] = field(default_factory=dict)
    reprojection_errors_per_pair: Dict[str, float] = field(default_factory=dict)
    input_image_dimensions: Dict[str, List[int]] = field(default_factory=dict)
    panorama_dimensions: List[int] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    status: str = "PENDING"
    excluded_images: List[str] = field(default_factory=list)
    pair_quality_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics to structured JSON-serializable dictionary."""
        avg_inlier_ratio = (
            float(sum(self.inlier_ratios_per_pair.values()) / len(self.inlier_ratios_per_pair))
            if self.inlier_ratios_per_pair else 0.0
        )
        return {
            "status": self.status,
            "images_processed": self.images_count,
            "feature_detector": self.feature_detector,
            "keypoints": self.keypoints_per_image,
            "matches": {
                "raw": self.raw_matches_per_pair,
                "good": self.good_matches_per_pair,
            },
            "inliers": self.inliers_per_pair,
            "inlier_ratios_percent": self.inlier_ratios_per_pair,
            "average_inlier_ratio_percent": round(avg_inlier_ratio, 2),
            "reprojection_errors": self.reprojection_errors_per_pair,
            "input_dimensions": self.input_image_dimensions,
            "panorama_dimensions": {
                "width": self.panorama_dimensions[0] if len(self.panorama_dimensions) > 0 else 0,
                "height": self.panorama_dimensions[1] if len(self.panorama_dimensions) > 1 else 0,
            },
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "excluded_images": self.excluded_images,
            "pair_quality": self.pair_quality_reports,
        }


class Evaluator:
    """Manages quantitative metric collection and report writing."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = 0.0

    def start_timer(self) -> None:
        """Starts timing pipeline execution."""
        self.start_time = time.time()

    def stop_timer(self) -> float:
        """Stops timer and calculates elapsed processing time."""
        elapsed = time.time() - self.start_time
        self.metrics.processing_time_seconds = elapsed
        return elapsed

    def save_json(self, output_filepath: str) -> None:
        """
        Saves collected metrics to JSON file.

        Args:
            output_filepath: Target output file path (e.g. 'outputs/metrics.json').
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=4)
