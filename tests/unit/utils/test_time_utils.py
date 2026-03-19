from __future__ import annotations

from datetime import datetime

from ztb.utils.time_utils import current_iso_timestamp


class TestTimeUtils:
    def test_current_iso_timestamp_is_parseable(self) -> None:
        timestamp = current_iso_timestamp()
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is None

    def test_current_iso_timestamp_utc_keeps_timezone(self) -> None:
        timestamp = current_iso_timestamp(utc=True)
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None
