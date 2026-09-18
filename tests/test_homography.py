"""
Unit tests for homography estimation, RANSAC, corner projection validation, and geometric sanity checks.
"""

import os
import pytest
import numpy as np
import cv2
from types import SimpleNamespace
from unittest.mock import Mock
from main import PanoramaStitcher
from src.warper import MultiCanvasBounds
from src.homography import (
    HomographyEstimator,
    HomographyEstimationError,
    ValidationConfig,
)


def test_synthetic_homography_estimation():
    """Verify RANSAC homography recovery using synthetic ground truth homography matrix."""
    np.random.seed(42)

    H_gt = np.array([
        [1.02, 0.01, 15.0],
        [-0.01, 0.98, -10.0],
        [0.0001, 0.00005, 1.0]
    ], dtype=np.float64)

    src_pts = np.random.uniform(50, 800, (100, 2)).astype(np.float32)
    src_homo = np.hstack([src_pts, np.ones((100, 1))])
    dst_homo = (H_gt @ src_homo.T).T
    dst_pts = (dst_homo[:, :2] / dst_homo[:, 2:3]).astype(np.float32)

    src_pts[:10] += np.random.uniform(50, 100, (10, 2)).astype(np.float32)

    estimator = HomographyEstimator(ransac_reproj_threshold=4.0)
    res = estimator.estimate(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        min_inliers=10,
        src_shape=(1000, 800),
        dst_shape=(1000, 800),
        good_matches=100,
        validate=False
    )

    assert res.matrix is not None
    assert res.matrix.shape == (3, 3)
    assert res.inliers_count >= 80
    assert res.inlier_ratio >= 80.0
    assert res.reprojection_error < 5.0


def test_insufficient_points_homography_failure():
    """Verify error handling when providing fewer than 4 point correspondences."""
    src_pts = np.float32([[10, 10], [20, 20], [30, 30]]).reshape(-1, 1, 2)
    dst_pts = np.float32([[12, 12], [22, 22], [32, 32]]).reshape(-1, 1, 2)

    estimator = HomographyEstimator()
    with pytest.raises(HomographyEstimationError, match="Requires at least 4 point correspondences"):
        estimator.estimate(src_pts, dst_pts)


def test_projected_corners_validation_convex():
    """Verify corner projection validation detects valid convex quad."""
    orig_shape = (1000, 800)
    valid_corners = np.array([
        [10.0, 10.0],
        [15.0, 990.0],
        [790.0, 995.0],
        [795.0, 5.0]
    ], dtype=np.float32)

    is_valid, reason = HomographyEstimator.validate_projected_corners(valid_corners, orig_shape)
    assert is_valid is True
    assert reason == "Valid"


def test_projected_corners_validation_non_convex():
    """Verify corner projection validation rejects non-convex (pinched/flipped) quad."""
    orig_shape = (1000, 800)
    invalid_corners = np.array([
        [10.0, 10.0],
        [790.0, 990.0],
        [15.0, 995.0],
        [795.0, 5.0]
    ], dtype=np.float32)

    is_valid, reason = HomographyEstimator.validate_projected_corners(invalid_corners, orig_shape)
    assert is_valid is False
    assert "convex" in reason.lower()


def test_projected_corners_validation_extreme_area():
    """Verify corner projection validation rejects unreasonable area ratios."""
    orig_shape = (1000, 800)
    huge_corners = np.array([
        [0.0, 0.0],
        [0.0, 10000.0],
        [8000.0, 10000.0],
        [8000.0, 0.0]
    ], dtype=np.float32)

    is_valid, reason = HomographyEstimator.validate_projected_corners(huge_corners, orig_shape)
    assert is_valid is False
    assert "area ratio" in reason.lower()


def test_extreme_projected_corners_rejected():
    """Homography projecting corners far outside destination frame must be rejected."""
    H = np.array([
        [1.0, 0.0, -5607.0],
        [0.0, 1.0, -7090.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    src_shape = (720, 1280)
    dst_shape = (720, 1280)

    estimator = HomographyEstimator()
    is_valid, reason = estimator.validate_homography(H, src_shape, dst_shape)
    assert is_valid is False


def test_concentrated_inliers_rejected():
    """Inliers clustered in a tiny region should fail spatial coverage validation."""
    np.random.seed(7)
    src_pts = np.random.uniform(10, 30, (80, 2)).astype(np.float32)
    dst_pts = src_pts + np.float32([5.0, 3.0])
    src_pts_3d = src_pts.reshape(-1, 1, 2)
    dst_pts_3d = dst_pts.reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts_3d, dst_pts_3d, cv2.RANSAC, 4.0)
    estimator = HomographyEstimator()
    report = estimator.evaluate_pair_quality(
        H, src_pts_3d, dst_pts_3d, mask, 80,
        "clustered_test", (720, 1280), (720, 1280)
    )
    assert report.decision == "REJECT"
    assert any("coverage" in r.lower() or "quadrant" in r.lower() for r in report.rejection_reasons)


def test_edge_overlap_inliers_accept_with_one_quadrant():
    """A tall edge overlap can be valid even when all inliers occupy one quadrant."""
    points = np.float32([
        [100, 49], [280, 49], [100, 499], [280, 499],
        [150, 200], [220, 300], [180, 420], [260, 460],
        [120, 100], [240, 350],
    ]).reshape(-1, 1, 2)
    mask = np.ones((len(points), 1), dtype=np.uint8)
    H = np.eye(3, dtype=np.float64)

    estimator = HomographyEstimator()
    report = estimator.evaluate_pair_quality(
        H, points, points, mask, 10,
        "edge_overlap_test", (1000, 1000), (1000, 1000)
    )

    assert report.decision == "ACCEPT"
    assert report.spatial.quadrants_with_inliers == 1
    assert report.spatial.x_coverage_pct >= 18.0
    assert report.spatial.y_coverage_pct >= 45.0


def test_spatial_coverage_metrics():
    """Verify spatial coverage calculation spans full image when inliers are distributed."""
    pts = np.float32([
        [100, 100], [600, 100], [600, 500], [100, 500],
        [350, 300], [200, 400], [500, 200]
    ]).reshape(-1, 1, 2)
    mask = np.ones((7, 1), dtype=np.uint8)
    spatial = HomographyEstimator.compute_spatial_coverage(pts, mask, (600, 700))
    assert spatial.x_coverage_pct > 50
    assert spatial.y_coverage_pct > 50
    assert spatial.quadrants_with_inliers >= 2


def test_pair_quality_score_good_pair():
    """A geometrically sound pair should receive ACCEPT decision."""
    np.random.seed(99)
    H_gt = np.array([
        [1.01, 0.005, 50.0],
        [-0.005, 0.99, 20.0],
        [0.00005, 0.00002, 1.0]
    ], dtype=np.float64)
    src_pts = np.random.uniform(100, 900, (120, 2)).astype(np.float32)
    src_homo = np.hstack([src_pts, np.ones((120, 1))])
    dst_pts = (H_gt @ src_homo.T).T
    dst_pts = (dst_pts[:, :2] / dst_pts[:, 2:3]).astype(np.float32)

    src_3d = src_pts.reshape(-1, 1, 2)
    dst_3d = dst_pts.reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_3d, dst_3d, cv2.RANSAC, 4.0)

    estimator = HomographyEstimator()
    report = estimator.evaluate_pair_quality(
        H, src_3d, dst_3d, mask, 120,
        "good_pair", (1000, 1100), (1000, 1100)
    )
    assert report.decision == "ACCEPT"
    assert report.quality_score >= 0.45


def test_cumulative_homography_validation():
    """Chained homography with runaway scale must fail cumulative validation."""
    H_runaway = np.array([
        [3.0, 0.0, 5000.0],
        [0.0, 3.0, 5000.0],
        [0.0001, 0.0001, 1.0]
    ], dtype=np.float64)
    estimator = HomographyEstimator()
    is_valid, reason = estimator.validate_homography(H_runaway, (720, 1280), (720, 1280))
    assert is_valid is False


def test_three_image_cumulative_homography_chaining():
    """Verify forward and inverse chaining into the middle-image reference frame."""
    images = [
        SimpleNamespace(filename=f"image_{index}.jpg", color_img=np.zeros((100, 100, 3), dtype=np.uint8))
        for index in range(3)
    ]
    H_01 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]])
    H_12 = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]])
    homographies = [H_01, H_12]

    match_result = SimpleNamespace(
        raw_matches_count=4,
        good_matches_count=4,
        src_pts=np.zeros((4, 1, 2), dtype=np.float32),
        dst_pts=np.zeros((4, 1, 2), dtype=np.float32),
        good_matches=[],
    )
    homography_results = [
        SimpleNamespace(
            matrix=matrix,
            inliers_count=4,
            inlier_ratio=100.0,
            reprojection_error=0.0,
            quality_report=None,
        )
        for matrix in homographies
    ]
    matcher = Mock()
    matcher.match.return_value = match_result
    estimator = Mock()
    estimator.estimate.side_effect = homography_results
    estimator.project_corners.return_value = np.zeros((4, 2))
    estimator.validate_homography.return_value = (True, "Valid")
    warper = Mock()
    warper.calculate_multi_canvas_bounds.return_value = MultiCanvasBounds(
        canvas_width=100,
        canvas_height=100,
        x_min=0,
        y_min=0,
        translation_matrix=np.eye(3),
        estimated_memory_mb=1.0,
    )
    blender = Mock()
    blender.blend_multi.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    metrics = SimpleNamespace(
        raw_matches_per_pair={},
        good_matches_per_pair={},
        inliers_per_pair={},
        inlier_ratios_per_pair={},
        reprojection_errors_per_pair={},
        pair_quality_reports={},
    )

    stitcher = PanoramaStitcher.__new__(PanoramaStitcher)
    stitcher.matcher = matcher
    stitcher.homography_estimator = estimator
    stitcher.warper = warper
    stitcher.blender = blender
    stitcher.debug_mode = False

    stitcher._stitch_multi_images(images, [([], None)] * 3, metrics)
    chained = blender.blend_multi.call_args.kwargs["H_composites"]

    point_image_0 = np.array([2.0, 3.0, 1.0])
    point_image_2 = np.array([40.0, 7.0, 1.0])
    expected_image_0_in_ref = H_01 @ point_image_0
    expected_image_2_in_ref = np.linalg.inv(H_12) @ point_image_2

    np.testing.assert_allclose(chained[0] @ point_image_0, expected_image_0_in_ref)
    np.testing.assert_allclose(chained[1], np.eye(3))
    np.testing.assert_allclose(chained[2] @ point_image_2, expected_image_2_in_ref)
