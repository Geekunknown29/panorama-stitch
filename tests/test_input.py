"""
Unit tests for input handling and validation module.
"""

import os
import pytest
from src.input_handler import InputHandler, InputValidationError


def test_input_validation_missing_file():
    """Verify that a non-existent file path raises InputValidationError."""
    with pytest.raises(InputValidationError, match="Input image file does not exist"):
        InputHandler.validate_and_collect_paths(input_files=["non_existent_img_123.jpg", "non_existent_img_456.jpg"])


def test_input_validation_unsupported_format(tmp_path):
    """Verify that an unsupported file format raises InputValidationError."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("not an image")
    valid_file = tmp_path / "valid.jpg"
    valid_file.write_bytes(b"\x00" * 100)

    with pytest.raises(InputValidationError, match="Unsupported image file extension"):
        InputHandler.validate_and_collect_paths(input_files=[str(invalid_file), str(valid_file)])


def test_input_validation_insufficient_images():
    """Verify that providing fewer than 2 images raises InputValidationError."""
    sample_path = os.path.abspath("data/sample/scene1_01.jpg")
    if os.path.exists(sample_path):
        with pytest.raises(InputValidationError, match="Insufficient images for stitching"):
            InputHandler.validate_and_collect_paths(input_files=[sample_path])


def test_input_validation_valid_sample_paths():
    """Verify that valid sample images are correctly collected and verified."""
    p1 = os.path.abspath("data/sample/scene1_01.jpg")
    p2 = os.path.abspath("data/sample/scene1_02.jpg")
    if os.path.exists(p1) and os.path.exists(p2):
        collected = InputHandler.validate_and_collect_paths(input_files=[p1, p2])
        assert len(collected) == 2
        assert collected[0] == p1
        assert collected[1] == p2
