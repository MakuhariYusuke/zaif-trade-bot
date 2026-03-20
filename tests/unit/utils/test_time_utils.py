from __future__ import annotations

from datetime import datetime

from ztb.utils.time_utils import current_compact_timestamp, current_iso_timestamp


class TestTimeUtils:
    def test_current_iso_timestamp_is_parseable(self) -> None:
        timestamp = current_iso_timestamp()
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is None

    def test_current_iso_timestamp_utc_keeps_timezone(self) -> None:
        timestamp = current_iso_timestamp(utc=True)
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None

    def test_current_compact_timestamp_utc_is_compact(self) -> None:
        timestamp = current_compact_timestamp(utc=True, fmt="%Y%m%d_%H%M")
        assert len(timestamp) == 13
        assert timestamp[8] == "_"
        assert timestamp.replace("_", "").isdigit()
