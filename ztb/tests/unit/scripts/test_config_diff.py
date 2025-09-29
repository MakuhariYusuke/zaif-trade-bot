import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ztb.ops.config.config_diff import deep_diff, is_major_key, load_json


class TestConfigDiff(unittest.TestCase):
    def test_load_json_success(self):
        """Test loading valid JSON."""
        data = {"test": "value"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_json(f.name)

        Path(f.name).unlink()
        self.assertEqual(result, data)

    def test_deep_diff_no_differences(self):
        """Test no differences."""
        a = {"key": "value"}
        b = {"key": "value"}
        diffs = deep_diff(a, b)
        self.assertEqual(diffs, [])

    def test_deep_diff_added(self):
        """Test added key."""
        a = {"key1": "value1"}
        b = {"key1": "value1", "key2": "value2"}
        diffs = deep_diff(a, b)
        self.assertIn("+ key2: value2", diffs)

    def test_deep_diff_removed(self):
        """Test removed key."""
        a = {"key1": "value1", "key2": "value2"}
        b = {"key1": "value1"}
        diffs = deep_diff(a, b)
        self.assertIn("- key2: value2", diffs)

    def test_deep_diff_changed(self):
        """Test changed value."""
        a = {"key": "value1"}
        b = {"key": "value2"}
        diffs = deep_diff(a, b)
        self.assertIn("~ key: value1 -> value2", diffs)

    def test_deep_diff_nested(self):
        """Test nested differences."""
        a = {"nested": {"key": "value1"}}
        b = {"nested": {"key": "value2"}}
        diffs = deep_diff(a, b)
        self.assertIn("~ nested.key: value1 -> value2", diffs)

    def test_deep_diff_list_length(self):
        """Test list length difference."""
        a = {"list": [1, 2]}
        b = {"list": [1, 2, 3]}
        diffs = deep_diff(a, b)
        self.assertIn("~ list: list length 2 -> 3", diffs)

    def test_is_major_key(self):
        """Test major key detection."""
        self.assertTrue(is_major_key("model.name"))
        self.assertTrue(is_major_key("policy"))
        self.assertTrue(is_major_key("learning_rate"))
        self.assertFalse(is_major_key("some.other.key"))

    @patch("sys.exit")
    def test_main_no_diff(self, mock_exit):
        """Test main with no differences."""
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fa,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fb,
        ):
            data = {"key": "value"}
            json.dump(data, fa)
            fa.flush()
            json.dump(data, fb)
            fb.flush()

            with patch("sys.argv", ["config_diff.py", "--a", fa.name, "--b", fb.name]):
                from ztb.ops.config.config_diff import main

                main()

        Path(fa.name).unlink()
        Path(fb.name).unlink()
        # Should have called exit(0) first
        self.assertEqual(mock_exit.call_args_list[0], ((0,), {}))

    @patch("sys.exit")
    @patch("builtins.print")
    def test_main_with_diff(self, mock_print, mock_exit):
        """Test main with differences."""
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fa,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fb,
        ):
            json.dump({"key": "value1"}, fa)
            fa.flush()
            json.dump({"key": "value2"}, fb)
            fb.flush()

            with patch("sys.argv", ["config_diff.py", "--a", fa.name, "--b", fb.name]):
                from ztb.ops.config.config_diff import main

                main()

        Path(fa.name).unlink()
        Path(fb.name).unlink()
        mock_exit.assert_called_with(2)

    @patch("sys.exit")
    @patch("builtins.print")
    def test_main_major_diff(self, mock_print, mock_exit):
        """Test main with major differences."""
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fa,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fb,
        ):
            json.dump({"model": "old"}, fa)
            fa.flush()
            json.dump({"model": "new"}, fb)
            fb.flush()

            with patch("sys.argv", ["config_diff.py", "--a", fa.name, "--b", fb.name]):
                from ztb.ops.config.config_diff import main

                main()

        Path(fa.name).unlink()
        Path(fb.name).unlink()
        mock_exit.assert_called_with(3)


if __name__ == "__main__":
    unittest.main()
