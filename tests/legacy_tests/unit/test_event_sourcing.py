"""
Tests for Event Sourcing functionality in CoverageValidator.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ztb.evaluation.status import CoverageValidator


class TestEventSourcing:
    """Test Event Sourcing functionality"""

    def test_record_event_basic(self):
        """Test basic event recording"""
        coverage_data = {"events": [], "current_state": {}, "metadata": {}}

        CoverageValidator.record_event(
            coverage_data, "feature_promoted", "rsi_14", "pending", "staging"
        )

        assert len(coverage_data["events"]) == 1
        event = coverage_data["events"][0]

        assert event["type"] == "feature_promoted"
        assert event["feature"] == "rsi_14"
        assert event["from_status"] == "pending"
        assert event["to_status"] == "staging"
        assert "timestamp" in event
        assert coverage_data["metadata"]["last_updated"] == event["timestamp"]

    def test_record_event_with_details(self):
        """Test event recording with additional details"""
        coverage_data = {"events": [], "current_state": {}, "metadata": {}}

        details = {
            "sharpe_ratio": 0.45,
            "win_rate": 0.62,
            "criteria_met": ["sharpe_ratio", "win_rate"],
        }

        CoverageValidator.record_event(
            coverage_data, "feature_promoted", "ema_20", "staging", "verified", details
        )

        assert len(coverage_data["events"]) == 1
        event = coverage_data["events"][0]

        assert event["details"] == details
        assert event["from_status"] == "staging"
        assert event["to_status"] == "verified"

    def test_record_event_multiple_events(self):
        """Test recording multiple events"""
        coverage_data = {"events": [], "current_state": {}, "metadata": {}}

        # Record first event
        CoverageValidator.record_event(coverage_data, "feature_added", "rsi_14")

        # Record second event
        CoverageValidator.record_event(
            coverage_data, "feature_promoted", "rsi_14", "pending", "staging"
        )

        assert len(coverage_data["events"]) == 2

        # Check timestamps are different and in order
        ts1 = coverage_data["events"][0]["timestamp"]
        ts2 = coverage_data["events"][1]["timestamp"]
        assert ts1 <= ts2

        # Last updated should be the latest timestamp
        assert coverage_data["metadata"]["last_updated"] == ts2

    def test_archive_coverage_data(self):
        """Test archiving coverage data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_dir = Path(temp_dir) / "archive"
            archive_dir.mkdir()

            coverage_data = {
                "events": [
                    {
                        "timestamp": "2024-01-15T10:00:00",
                        "type": "feature_promoted",
                        "feature": "rsi_14",
                    },
                    {
                        "timestamp": "2024-01-16T10:00:00",
                        "type": "feature_promoted",
                        "feature": "ema_20",
                    },
                ],
                "current_state": {},
                "metadata": {},
            }

            CoverageValidator.archive_coverage_data(coverage_data, str(archive_dir))

            # Check archive file was created
            archive_file = archive_dir / "coverage_2024.json"
            assert archive_file.exists()

            # Check archive contents
            with open(archive_file, "r") as f:
                archive_content = json.load(f)

            assert "events" in archive_content
            assert len(archive_content["events"]) == 2
            assert archive_content["metadata"]["year"] == 2024

    def test_archive_coverage_data_empty_events(self):
        """Test archiving with no events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_dir = Path(temp_dir) / "archive"
            archive_dir.mkdir()

            coverage_data = {"events": [], "current_state": {}, "metadata": {}}

            # Should not create archive file
            CoverageValidator.archive_coverage_data(coverage_data, str(archive_dir))

            # No files should be created
            assert len(list(archive_dir.glob("*.json"))) == 0

    def test_load_coverage_files_with_events(self):
        """Test loading coverage files with event sourcing structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            coverage_dir = Path(temp_dir) / "coverage"
            coverage_dir.mkdir()

            # Create main coverage.json with events
            main_coverage = {
                "events": [
                    {
                        "timestamp": "2024-01-15T10:00:00",
                        "type": "feature_promoted",
                        "feature": "rsi_14",
                        "from_status": "pending",
                        "to_status": "staging",
                    }
                ],
                "current_state": {
                    "staging": ["rsi_14"],
                    "pending": ["ema_20"],
                    "verified": [],
                    "failed": [],
                    "unverified": [],
                },
                "metadata": {"last_updated": "2024-01-15T10:00:00"},
            }

            with open(coverage_dir / "coverage.json", "w") as f:
                json.dump(main_coverage, f)

            # Load coverage
            loaded = CoverageValidator.load_coverage_files(str(coverage_dir))

            # Check events are preserved
            assert "events" in loaded
            assert len(loaded["events"]) == 1
            assert loaded["events"][0]["feature"] == "rsi_14"

            # Check current_state is accessible
            assert "current_state" in loaded
            assert "staging" in loaded["current_state"]
            assert "rsi_14" in loaded["current_state"]["staging"]

    def test_merge_coverage_with_events(self):
        """Test merging coverage data that includes events"""
        target = {
            "events": [
                {
                    "timestamp": "2024-01-14T10:00:00",
                    "type": "feature_added",
                    "feature": "rsi_14",
                }
            ],
            "current_state": {
                "verified": [],
                "staging": [],
                "pending": ["rsi_14"],
                "failed": [],
                "unverified": [],
            },
            "metadata": {"last_updated": "2024-01-14T10:00:00"},
        }

        source = {
            "events": [
                {
                    "timestamp": "2024-01-15T10:00:00",
                    "type": "feature_promoted",
                    "feature": "rsi_14",
                    "from_status": "pending",
                    "to_status": "staging",
                }
            ],
            "current_state": {
                "verified": [],
                "staging": ["rsi_14"],
                "pending": [],
                "failed": [],
                "unverified": [],
            },
            "metadata": {"last_updated": "2024-01-15T10:00:00"},
        }

        CoverageValidator._merge_coverage_data(target, source, "test.json")

        # Check events are merged
        assert len(target["events"]) == 2

        # Check current_state is merged
        assert "rsi_14" in target["current_state"]["staging"]
        assert "rsi_14" not in target["current_state"]["pending"]

        # Check metadata is updated
        assert target["metadata"]["last_updated"] == "2024-01-15T10:00:00"

    def test_backward_compatibility_old_format(self):
        """Test loading old format coverage files without events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            coverage_dir = Path(temp_dir) / "coverage"
            coverage_dir.mkdir()

            # Create old format coverage.json
            old_coverage = {
                "verified": ["rsi_14"],
                "pending": ["ema_20"],
                "failed": [],
                "unverified": [],
                "metadata": {"last_updated": "2024-01-15T10:00:00"},
            }

            with open(coverage_dir / "coverage.json", "w") as f:
                json.dump(old_coverage, f)

            # Load coverage
            loaded = CoverageValidator.load_coverage_files(str(coverage_dir))

            # Check events array is created
            assert "events" in loaded
            assert isinstance(loaded["events"], list)

            # Check current_state is created from old format
            assert "current_state" in loaded
            assert loaded["current_state"]["verified"] == ["rsi_14"]
            assert loaded["current_state"]["pending"] == ["ema_20"]

    def test_event_sourcing_preserves_order(self):
        """Test that events are preserved in chronological order"""
        coverage_data = {"events": [], "current_state": {}, "metadata": {}}

        # Record events with simulated timestamps
        with patch("ztb.evaluation.status.datetime") as mock_datetime:
            # Event 1
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-15T08:00:00"
            )
            CoverageValidator.record_event(coverage_data, "feature_added", "rsi_14")

            # Event 2
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-15T09:00:00"
            )
            CoverageValidator.record_event(
                coverage_data, "feature_promoted", "rsi_14", "pending", "staging"
            )

            # Event 3
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-15T10:00:00"
            )
            CoverageValidator.record_event(
                coverage_data, "feature_promoted", "rsi_14", "staging", "verified"
            )

        assert len(coverage_data["events"]) == 3

        # Check chronological order
        timestamps = [event["timestamp"] for event in coverage_data["events"]]
        assert timestamps == sorted(timestamps)

        # Check last_updated is the latest
        assert coverage_data["metadata"]["last_updated"] == "2024-01-15T10:00:00"
