"""
Tests for OS File Picker GUI window integration in main.py.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from main import open_file_picker, build_cli_parser


class TestFilePicker(unittest.TestCase):
    """Test suite for graphical file picker integration."""

    def test_cli_parser_gui_flag(self):
        """Verify CLI parser handles --gui and -g flags."""
        parser = build_cli_parser()
        args = parser.parse_args(["--gui"])
        self.assertTrue(args.gui)

        args_short = parser.parse_args(["-g"])
        self.assertTrue(args_short.gui)

    @patch("tkinter.filedialog.askopenfilenames")
    @patch("tkinter.Tk")
    def test_open_file_picker_returns_selected_files(self, mock_tk, mock_dialog):
        """Test open_file_picker returns list of files selected in GUI."""
        file1 = os.path.abspath("data/sample/scene1_01.jpg")
        file2 = os.path.abspath("data/sample/scene1_02.jpg")

        mock_dialog.return_value = (file1, file2)

        files = open_file_picker()
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], file1)
        self.assertEqual(files[1], file2)

    @patch("tkinter.filedialog.askopenfilenames")
    @patch("tkinter.Tk")
    def test_open_file_picker_handles_cancellation(self, mock_tk, mock_dialog):
        """Test open_file_picker handles user cancelling dialog."""
        mock_dialog.return_value = ()

        files = open_file_picker()
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
