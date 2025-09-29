"""
Unit tests for scripts directory guards.
"""

import glob
import unittest


class TestScriptsGuard(unittest.TestCase):
    """Test that ztb/scripts/ contains no Python files."""

    def test_no_py_files_in_scripts(self):
        """Assert that no .py files exist in ztb/scripts/."""
        py_files = glob.glob("ztb/scripts/*.py")
        self.assertEqual(
            len(py_files),
            0,
            f"Found Python files in ztb/scripts/: {py_files}. "
            "Only shell scripts (.sh, .ps1) are allowed.",
        )
