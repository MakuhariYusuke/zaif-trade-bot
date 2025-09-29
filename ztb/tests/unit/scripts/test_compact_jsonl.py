import gzip
import tempfile
import unittest
from pathlib import Path

from ztb.ops.artifacts.compact_jsonl import compact_jsonl, parse_timestamp


class TestCompactJsonl(unittest.TestCase):
    def test_parse_timestamp_valid(self):
        """Test parsing valid timestamp."""
        line = '{"timestamp": "2025-09-29T10:00:00Z", "message": "test"}'
        date = parse_timestamp(line)
        self.assertEqual(date, "2025-09-29")

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid JSON."""
        line = "invalid json"
        date = parse_timestamp(line)
        self.assertIsNone(date)

    def test_parse_timestamp_no_timestamp(self):
        """Test line without timestamp."""
        line = '{"message": "test"}'
        date = parse_timestamp(line)
        self.assertIsNone(date)

    def test_compact_jsonl_dry_run(self):
        """Test dry-run compaction."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"timestamp": "2025-09-29T10:00:00Z", "msg": "a"}\n')
            f.write('{"timestamp": "2025-09-30T11:00:00Z", "msg": "b"}\n')
            f.write('{"timestamp": "2025-09-29T12:00:00Z", "msg": "c"}\n')
            f.flush()

            import io
            from unittest.mock import patch

            captured_output = io.StringIO()
            with patch("sys.stdout", captured_output):
                compact_jsonl(f.name, apply=False)

            output = captured_output.getvalue()
            self.assertIn("Found 2 date groups", output)
            self.assertIn("2025-09-29: 2 lines", output)
            self.assertIn("2025-09-30: 1 lines", output)
            self.assertIn("Use --apply", output)

        Path(f.name).unlink()

    def test_compact_jsonl_apply(self):
        """Test apply compaction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "test.jsonl"
            with open(jsonl_path, "w") as f:
                f.write('{"timestamp": "2025-09-29T10:00:00Z", "msg": "a"}\n')
                f.write('{"timestamp": "2025-09-30T11:00:00Z", "msg": "b"}\n')

            compact_jsonl(str(jsonl_path), apply=True)

            # Check backup
            backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".backup")
            self.assertTrue(backup_path.exists())

            # Check compressed files
            gz1 = Path(temp_dir) / "2025-09-29.jsonl.gz"
            gz2 = Path(temp_dir) / "2025-09-30.jsonl.gz"
            self.assertTrue(gz1.exists())
            self.assertTrue(gz2.exists())

            # Check content
            with gzip.open(gz1, "rt") as f:
                content = f.read()
                self.assertIn('"msg": "a"', content)

            with gzip.open(gz2, "rt") as f:
                content = f.read()
                self.assertIn('"msg": "b"', content)


if __name__ == "__main__":
    unittest.main()
