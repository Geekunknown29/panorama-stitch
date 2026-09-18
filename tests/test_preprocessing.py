"""
Unit tests for image preprocessing module.
"""

import os
import pytest
import numpy as np
import cv2
from src.preprocessing import ImagePreprocessor, ProcessedImage


def test_preprocessing_aspect_ratio_resize(tmp_path):
    """Verify aspect-ratio preserving image resize for large images."""
    large_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    large_img_path = str(tmp_path / "large.jpg")
    cv2.imwrite(large_img_path, large_img)

    preprocessor = ImagePreprocessor(max_dimension=1600)
    proc_img = preprocessor.preprocess(large_img_path)

    assert isinstance(proc_img, ProcessedImage)
    assert proc_img.original_shape == (3000, 4000, 3)
    assert max(proc_img.working_shape[:2]) == 1600
    assert abs(proc_img.scale_factor - (1600 / 4000.0)) < 1e-5
    # Verify aspect ratio preserved: 4000/3000 = 1.3333...
    aspect_orig = 4000 / 3000.0
    aspect_work = proc_img.working_shape[1] / float(proc_img.working_shape[0])
    assert abs(aspect_orig - aspect_work) < 1e-2


def test_preprocessing_grayscale_conversion():
    """Verify color working copy and single-channel grayscale output."""
    sample_path = os.path.abspath("data/sample/scene1_01.jpg")
    if os.path.exists(sample_path):
        preprocessor = ImagePreprocessor(max_dimension=1600)
        proc_img = preprocessor.preprocess(sample_path)
        assert len(proc_img.color_img.shape) == 3
        assert len(proc_img.gray_img.shape) == 2
        assert proc_img.color_img.shape[:2] == proc_img.gray_img.shape[:2]
