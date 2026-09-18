"""
Input Handler Module for Intelligent Panorama Builder.

Handles validation of file paths, directory scanning, format verification,
and readability checks for input images.
"""

import os
from typing import List
import cv2


class PanoramaError(Exception):
    """Base exception for panorama processing errors."""
    pass


class InputValidationError(PanoramaError):
    """Exception raised when input validation fails."""
    pass


class InputHandler:
    """Manages input image discovery and validation."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

    @classmethod
    def validate_and_collect_paths(
        cls,
        input_files: List[str] = None,
        input_dir: str = None
    ) -> List[str]:
        """
        Collects and validates input image file paths from explicitly provided file lists
        or directory paths.

        Args:
            input_files: List of file paths provided via CLI.
            input_dir: Path to directory containing images.

        Returns:
            List of validated, readable absolute image file paths.

        Raises:
            InputValidationError: If paths are invalid, unreadable, or fewer than 2.
        """
        paths = []

        if input_dir:
            if not os.path.exists(input_dir):
                raise InputValidationError(f"Input directory does not exist: '{input_dir}'")
            if not os.path.isdir(input_dir):
                raise InputValidationError(f"Provided path is not a directory: '{input_dir}'")

            for entry in sorted(os.listdir(input_dir)):
                ext = os.path.splitext(entry)[1].lower()
                if ext in cls.SUPPORTED_EXTENSIONS:
                    paths.append(os.path.abspath(os.path.join(input_dir, entry)))

            if not paths:
                raise InputValidationError(
                    f"No supported image files found in directory: '{input_dir}'. "
                    f"Supported formats: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
                )

        elif input_files:
            for p in input_files:
                if not os.path.exists(p):
                    raise InputValidationError(f"Input image file does not exist: '{p}'")
                if not os.path.isfile(p):
                    raise InputValidationError(f"Input path is not a valid file: '{p}'")

                ext = os.path.splitext(p)[1].lower()
                if ext not in cls.SUPPORTED_EXTENSIONS:
                    raise InputValidationError(
                        f"Unsupported image file extension '{ext}' for file '{p}'. "
                        f"Supported formats: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
                    )
                paths.append(os.path.abspath(p))
        else:
            raise InputValidationError("No input images or input directory specified.")

        if len(paths) < 2:
            raise InputValidationError(
                f"Insufficient images for stitching. Found {len(paths)} image(s), "
                "but at least 2 overlapping images are required."
            )

        # Validate image readability for all collected paths
        for path in paths:
            cls.verify_image_readable(path)

        return paths

    @staticmethod
    def verify_image_readable(filepath: str) -> None:
        """
        Verifies that an image file can be decoded and loaded properly by OpenCV.

        Args:
            filepath: Path to the image file.

        Raises:
            InputValidationError: If image cannot be read or is empty/corrupt.
        """
        # Read header only / minimal check
        img = cv2.imread(filepath)
        if img is None or img.size == 0:
            raise InputValidationError(
                f"Failed to read image file '{filepath}'. File may be corrupted, blank, or invalid."
            )
        if len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0:
            raise InputValidationError(f"Image '{filepath}' has invalid zero dimensions: {img.shape}")
