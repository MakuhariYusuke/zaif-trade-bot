"""
Unit tests for scripts directory guards.
"""

import unittest
from pathlib import Path


class TestScriptsGuard(unittest.TestCase):
    """Test that ztb/scripts/ contains no Python files."""

    def test_no_py_files_in_scripts(self):
        """Assert that no .py files exist in ztb/scripts/."""
        scripts_dir = Path("ztb/scripts")
        py_files = list(scripts_dir.glob("*.py"))
        # Allow deprecated files that are scheduled for removal
        allowed_deprecated = {scripts_dir / "trading_service.py"}
        py_files = [f for f in py_files if f not in allowed_deprecated]

        self.assertEqual(
            len(py_files),
            0,
            f"Found Python files in ztb/scripts/: {py_files}. "
            "Only shell scripts (.sh, .ps1) are allowed.",
        )
