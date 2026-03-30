"""661# skip_days_of_week ユニットテスト."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.time_filter import TimeFilter


def _make_config(**overrides: object) -> FillTestConfig:
    defaults = {
        "enable_time_filter": True,
        "skip_utc_hours": [],
        "skip_utc_hours_buy": [],
        "skip_utc_hours_sell": [],
        "skip_days_of_week": [],
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)  # type: ignore[arg-type]


class TestSkipDaysOfWeek:
    """661# 曜日フィルターのテスト."""

    def test_empty_skip_days_no_filter(self) -> None:
        """skip_days_of_week が空なら曜日フィルタなし."""
        tf = TimeFilter(_make_config(skip_days_of_week=[]))
        assert tf.is_filtered() is False

    def test_skip_saturday(self) -> None:
        """土曜日(5)がスキップされる."""
        tf = TimeFilter(_make_config(skip_days_of_week=[5, 6]))
        # Saturday = weekday() == 5
        sat = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)  # Saturday
        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = sat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert tf.is_filtered() is True

    def test_skip_sunday(self) -> None:
        """日曜日(6)がスキップされる."""
        tf = TimeFilter(_make_config(skip_days_of_week=[5, 6]))
        sun = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)  # Sunday
        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = sun
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert tf.is_filtered() is True

    def test_monday_not_skipped(self) -> None:
        """月曜日(0)はスキップされない."""
        tf = TimeFilter(_make_config(skip_days_of_week=[5, 6]))
        mon = datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc)  # Monday
        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = mon
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert tf.is_filtered() is False

    def test_disabled_time_filter_ignores_days(self) -> None:
        """enable_time_filter=Falseなら曜日もスキップしない."""
        tf = TimeFilter(_make_config(enable_time_filter=False, skip_days_of_week=[5, 6]))
        assert tf.is_filtered() is False

    def test_side_specific_also_blocked_on_skip_day(self) -> None:
        """skip_dayでは side 指定でもスキップ."""
        tf = TimeFilter(_make_config(skip_days_of_week=[5, 6]))
        sat = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = sat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert tf.is_filtered(side="buy") is True
            assert tf.is_filtered(side="sell") is True

    def test_config_field_default_empty(self) -> None:
        """デフォルト値は空リスト."""
        cfg = FillTestConfig()
        assert cfg.skip_days_of_week == []
