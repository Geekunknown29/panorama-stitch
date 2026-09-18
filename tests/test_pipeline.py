"""
End-to-end integration tests for the full Intelligent Panorama Builder pipeline.
"""

import os
import json
import pytest
import cv2
import numpy as np
from main import PanoramaStitcher
from src.input_handler import InputValidationError
from src.warper import ImageWarper, WarpingError


def test_e2e_two_image_panorama(tmp_path):
    """TEST 1: Valid two-image panorama stitching."""
    p1 = os.path.abspath("data/sample/scene1_01.jpg")
    p2 = os.path.abspath("data/sample/scene1_02.jpg")
    output_path = str(tmp_path / "panorama_2img.jpg")

    stitcher = PanoramaStitcher(feature_algorithm="sift", debug_mode=True)
    res_path = stitcher.stitch(input_files=[p1, p2], output_filepath=output_path)

    assert os.path.exists(res_path)
    img = cv2.imread(res_path)
    assert img is not None
    assert img.shape[1] > 1100

    metrics_path = str(tmp_path / "metrics.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["status"] == "SUCCESS"
        assert data["images_processed"] == 2


def test_e2e_three_image_panorama(tmp_path):
    """TEST 2: Valid three-image panorama stitching."""
    p1 = os.path.abspath("data/sample/scene1_01.jpg")
    p2 = os.path.abspath("data/sample/scene1_02.jpg")
    p3 = os.path.abspath("data/sample/scene1_03.jpg")
    output_path = str(tmp_path / "panorama_3img.jpg")

    stitcher = PanoramaStitcher(feature_algorithm="sift", auto_order=True, debug_mode=False)
    res_path = stitcher.stitch(input_files=[p1, p2, p3], output_filepath=output_path)

    assert os.path.exists(res_path)
    img = cv2.imread(res_path)
    assert img is not None
    assert img.shape[1] > 1500


def test_canvas_size_safety_limits():
    """TEST 3: Verify WarpingError is raised if canvas dimensions exceed safety threshold."""
    warper = ImageWarper(max_canvas_width=1000, max_canvas_height=1000)
    with pytest.raises(WarpingError, match="exceed safety limits"):
        warper.validate_canvas_dimensions(2000, 2000)


def test_e2e_unrelated_failure_case(tmp_path):
    """TEST 4: Images with insufficient overlap / feature matches handled gracefully."""
    p1 = os.path.abspath("data/failure_cases/unrelated_01.jpg")
    p2 = os.path.abspath("data/failure_cases/unrelated_02.jpg")
    output_path = str(tmp_path / "failed_panorama.jpg")

    stitcher = PanoramaStitcher(feature_algorithm="sift")
    with pytest.raises(Exception):
        stitcher.stitch(input_files=[p1, p2], output_filepath=output_path)


def test_e2e_four_image_panorama(tmp_path):
    """TEST: Four-image panorama stitching with auto-order."""
    paths = [os.path.abspath(f"data/sample/scene1_{i:02d}.jpg") for i in range(1, 5)]
    output_path = str(tmp_path / "panorama_4img.jpg")

    stitcher = PanoramaStitcher(feature_algorithm="sift", auto_order=True)
    res_path = stitcher.stitch(input_files=paths, output_filepath=output_path)

    assert os.path.exists(res_path)
    img = cv2.imread(res_path)
    assert img is not None
    assert img.shape[1] > 1500


def test_e2e_six_image_panorama(tmp_path):
    """TEST: Six-image panorama without extreme canvas growth."""
    paths = [os.path.abspath(f"data/sample/scene1_{i:02d}.jpg") for i in range(1, 7)]
    output_path = str(tmp_path / "panorama_6img.jpg")

    stitcher = PanoramaStitcher(feature_algorithm="sift", auto_order=True, max_dimension=1200)
    res_path = stitcher.stitch(input_files=paths, output_filepath=output_path)

    assert os.path.exists(res_path)
    img = cv2.imread(res_path)
    assert img is not None
    assert img.shape[0] < 5000
    assert img.shape[1] < 8000


def test_e2e_different_resolutions(tmp_path):
    """TEST 5: Different image resolutions handled correctly by working copy scaling."""
    p1 = os.path.abspath("data/sample/scene1_01.jpg")
    p2 = os.path.abspath("data/sample/scene1_02.jpg")

    img2 = cv2.imread(p2)
    h, w = img2.shape[:2]
    img2_resized = cv2.resize(img2, (w // 2, h // 2))

    res_p2 = str(tmp_path / "res_scene_02.jpg")
    cv2.imwrite(res_p2, img2_resized)

    output_path = str(tmp_path / "diff_res_panorama.jpg")
    stitcher = PanoramaStitcher(feature_algorithm="sift")
    res_path = stitcher.stitch(input_files=[p1, res_p2], output_filepath=output_path)

    assert os.path.exists(res_path)
    img = cv2.imread(res_path)
    assert img is not None
