"""
Feature Matcher Module for Intelligent Panorama Builder.

Matches descriptors between image pairs using BFMatcher or FLANN, filtering
unreliable matches via Lowe's ratio test.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from src.input_handler import PanoramaError


class FeatureMatchingError(PanoramaError):
    """Exception raised when feature matching fails or yields insufficient matches."""
    pass


@dataclass
class MatchResult:
    """Dataclass storing feature matching results between two images."""
    raw_matches_count: int
    good_matches_count: int
    match_ratio: float
    good_matches: List[cv2.DMatch]
    src_pts: np.ndarray  # (N, 1, 2) float32 points from source image
    dst_pts: np.ndarray  # (N, 1, 2) float32 points from target image


class FeatureMatcher:
    """Manages descriptor matching and match filtering."""

    def __init__(self, algorithm: str = "sift", ratio_threshold: float = 0.75):
        """
        Args:
            algorithm: Detector algorithm name ('sift' or 'orb') to select distance norm.
            ratio_threshold: Lowe's ratio test threshold (typically 0.7 to 0.8).
        """
        self.algorithm = algorithm.lower().strip()
        self.ratio_threshold = ratio_threshold

        if self.algorithm == "orb":
            # Hamming distance for binary descriptors
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # L2 norm for floating point descriptors (SIFT)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match(
        self,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        descriptors2: np.ndarray,
        img1_name: str = "img1",
        img2_name: str = "img2",
        min_good_matches: int = 10
    ) -> MatchResult:
        """
        Executes KNN descriptor matching (k=2) and applies Lowe's ratio test filtering.

        Args:
            keypoints1: Source image keypoints.
            descriptors1: Source image descriptors.
            keypoints2: Target image keypoints.
            descriptors2: Target image descriptors.
            img1_name: Name of source image for reporting.
            img2_name: Name of target image for reporting.
            min_good_matches: Minimum required good matches for homography.

        Returns:
            MatchResult dataclass containing matched points and metrics.

        Raises:
            FeatureMatchingError: If match count is below min_good_matches.
        """
        if descriptors1 is None or descriptors2 is None:
            raise FeatureMatchingError("One or both feature descriptor matrices are None.")

        if len(descriptors1) < 2 or len(descriptors2) < 2:
            raise FeatureMatchingError(
                f"Insufficient descriptors for KNN matching between '{img1_name}' and '{img2_name}'."
            )

        # KNN matching with k=2
        raw_knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        raw_matches_count = len(raw_knn_matches)

        # Apply Lowe's Ratio Test filtering
        good_matches: List[cv2.DMatch] = []
        for match_pair in raw_knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        good_count = len(good_matches)
        match_ratio = (good_count / float(raw_matches_count)) * 100.0 if raw_matches_count > 0 else 0.0

        if good_count < min_good_matches:
            raise FeatureMatchingError(
                f"ERROR:\nInsufficient reliable feature matches between '{img1_name}' and '{img2_name}'.\n"
                f"Found: {good_count} good match(es)\n"
                f"Required: {min_good_matches} minimum match(es)\n"
                "Try providing images with greater visual overlap, higher resolution, or more distinct visual features."
            )

        # Extract corresponding point coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return MatchResult(
            raw_matches_count=raw_matches_count,
            good_matches_count=good_count,
            match_ratio=match_ratio,
            good_matches=good_matches,
            src_pts=src_pts,
            dst_pts=dst_pts
        )

    def draw_matches(
        self,
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch],
        max_draw: int = 150
    ) -> np.ndarray:
        """
        Draws match correspondence lines between two images for debug visual output.

        Args:
            img1: Source color image.
            kp1: Source keypoints.
            img2: Target color image.
            kp2: Target keypoints.
            good_matches: List of filtered DMatch objects.
            max_draw: Maximum matches to draw to avoid visual clutter.

        Returns:
            Combined BGR image visualizing matched keypoint pairs.
        """
        # Sort matches by distance for visual clarity
        sorted_matches = sorted(good_matches, key=lambda x: x.distance)[:max_draw]
        match_img = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            sorted_matches,
            outImg=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return match_img
