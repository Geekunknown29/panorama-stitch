"""
Unit tests for feature detection and description module.
"""

import os
import pytest
import numpy as np
from src.feature_detector import FeatureDetector, FeatureExtractionError
from src.preprocessing import ImagePreprocessor


def test_sift_feature_detection():
    """Verify SIFT feature detection and descriptor extraction on sample image."""
    sample_path = os.path.abspath("data/sample/scene1_01.jpg")
    if os.path.exists(sample_path):
        preprocessor = ImagePreprocessor()
        proc_img = preprocessor.preprocess(sample_path)

        detector = FeatureDetector(algorithm="sift")
        kp, des = detector.detect_and_compute(proc_img.gray_img)

        assert len(kp) > 100
        assert des is not None
        assert des.shape[0] == len(kp)
        assert des.shape[1] == 128  # SIFT descriptor dimension is 128


def test_orb_feature_detection():
    """Verify ORB feature detection and binary descriptor extraction."""
    sample_path = os.path.abspath("data/sample/scene1_01.jpg")
    if os.path.exists(sample_path):
        preprocessor = ImagePreprocessor()
        proc_img = preprocessor.preprocess(sample_path)

        detector = FeatureDetector(algorithm="orb")
        kp, des = detector.detect_and_compute(proc_img.gray_img)

        assert len(kp) > 50
        assert des is not None
        assert des.shape[0] == len(kp)
        assert des.shape[1] == 32  # ORB descriptor dimension is 32 bytes


def test_feature_detection_blank_image():
    """Verify error handling when attempting feature detection on blank image."""
    blank_img = np.zeros((200, 200), dtype=np.uint8)
    detector = FeatureDetector(algorithm="sift")

    with pytest.raises(FeatureExtractionError, match="No valid keypoints or descriptors detected"):
        detector.detect_and_compute(blank_img)
