"""
Tests for interactive terminal input mode in main.py.
"""

import os
import sys
import unittest
from unittest.mock import patch
from main import prompt_interactive_input, build_cli_parser


class TestInteractiveInput(unittest.TestCase):
    """Test suite for interactive input collection."""

    def test_cli_parser_defaults_to_interactive(self):
        """Verify CLI parser allows missing input arguments."""
        parser = build_cli_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.input)
        self.assertIsNone(args.input_dir)
        self.assertFalse(args.interactive)

    def test_cli_parser_interactive_flag(self):
        """Verify CLI parser handles --interactive flag."""
        parser = build_cli_parser()
        args = parser.parse_args(["--interactive"])
        self.assertTrue(args.interactive)

    @patch("builtins.input")
    def test_prompt_interactive_input_valid_paths(self, mock_input):
        """Test collecting valid image paths interactively."""
        valid_img1 = os.path.abspath("data/sample/scene1_01.jpg")
        valid_img2 = os.path.abspath("data/sample/scene1_02.jpg")

        mock_input.side_effect = [
            "2",                # Select manual terminal path input
            f'"{valid_img1}"',  # Drag and drop path with double quotes
            f"'{valid_img2}'",  # Single quotes path
            "done"
        ]

        paths = prompt_interactive_input()
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], valid_img1)
        self.assertEqual(paths[1], valid_img2)

    @patch("builtins.input")
    def test_prompt_interactive_input_requires_at_least_two_images(self, mock_input):
        """Test interactive prompt enforces minimum 2 images requirement."""
        valid_img1 = os.path.abspath("data/sample/scene1_01.jpg")
        valid_img2 = os.path.abspath("data/sample/scene1_02.jpg")

        mock_input.side_effect = [
            "2",        # Select manual terminal path input
            valid_img1,
            "done",     # Insufficient images attempt
            valid_img2,
            "done"      # Valid completion
        ]

        paths = prompt_interactive_input()
        self.assertEqual(len(paths), 2)



if __name__ == "__main__":
    unittest.main()
