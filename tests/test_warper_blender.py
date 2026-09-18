"""
Unit tests for perspective warper and distance-transform blender modules.
"""

import numpy as np
import pytest
import cv2
from src.warper import ImageWarper, WarpResult
from src.blender import ImageBlender


def test_warper_canvas_computation():
    """Verify perspective warping and canvas coordinate bounding box calculation."""
    img1 = np.full((400, 500, 3), 100, dtype=np.uint8)
    img2 = np.full((400, 500, 3), 200, dtype=np.uint8)

    # Pure translation homography: shift right by 200px
    H_trans = np.array([
        [1.0, 0.0, 200.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    warper = ImageWarper()
    res = warper.warp_pair(img1, img2, H_trans)

    assert isinstance(res, WarpResult)
    # Total width should be 500 + 200 = 700px
    assert res.canvas_shape[1] == 700
    assert res.canvas_shape[0] == 400
    assert np.any(res.warped_img1 > 0)
    assert np.any(res.warped_img2 > 0)


def test_blender_feathered_blend():
    """Verify distance-transform feathered blending across overlapping regions."""
    h, w = 400, 700
    warped_img1 = np.zeros((h, w, 3), dtype=np.uint8)
    warped_img2 = np.zeros((h, w, 3), dtype=np.uint8)

    # img1 covers left [0..500]
    warped_img1[:, 0:500] = (250, 100, 100)
    # img2 covers right [200..700]
    warped_img2[:, 200:700] = (100, 100, 250)

    blender = ImageBlender()
    blended = blender.blend(warped_img1, warped_img2)

    assert blended.shape == (h, w, 3)
    # Leftmost region should match img1
    assert np.allclose(blended[200, 50], (250, 100, 100), atol=5)
    # Rightmost region should match img2
    assert np.allclose(blended[200, 650], (100, 100, 250), atol=5)
    # Overlap region [200..500] should be smoothly blended without zero pixels
    assert np.all(blended[200, 350] > 0)
