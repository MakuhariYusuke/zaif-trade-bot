"""212# §7.1: LiveTraderConfig magic number config 化テスト."""

import re
from functools import lru_cache
from pathlib import Path

import pytest

from ztb.trading.live_trader.config import LiveTraderConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LIVE_TRADER_PATH = _PROJECT_ROOT / "ztb" / "trading" / "live_trader" / "live_trader.py"


@lru_cache(maxsize=None)
def _load_live_trader_source() -> str:
    return _LIVE_TRADER_PATH.read_text(encoding="utf-8-sig")


class TestLiveTraderConfigDefaults:
    """LiveTraderConfig デフォルト値の検証."""

    def test_ticker_timeout_default(self) -> None:
        cfg = LiveTraderConfig()
        assert cfg.ticker_timeout == 5.0

    def test_api_timeout_default(self) -> None:
        cfg = LiveTraderConfig()
        assert cfg.api_timeout == 10.0

    def test_cb_recovery_timeout_default(self) -> None:
        cfg = LiveTraderConfig()
        assert cfg.cb_recovery_timeout == 60.0

    def test_retry_interval_default(self) -> None:
        cfg = LiveTraderConfig()
        assert cfg.retry_interval == 60.0

    def test_order_poll_interval_default(self) -> None:
        cfg = LiveTraderConfig()
        assert cfg.order_poll_interval == 2.0

    def test_custom_values(self) -> None:
        cfg = LiveTraderConfig(
            ticker_timeout=3.0,
            api_timeout=15.0,
            cb_recovery_timeout=30.0,
            retry_interval=120.0,
            order_poll_interval=5.0,
        )
        assert cfg.ticker_timeout == 3.0
        assert cfg.api_timeout == 15.0
        assert cfg.cb_recovery_timeout == 30.0
        assert cfg.retry_interval == 120.0
        assert cfg.order_poll_interval == 5.0


class TestNoMagicNumbersInLiveTrader:
    """live_trader.py にマジックナンバーが残っていないことを検証."""

    @pytest.fixture(scope="class", autouse=True)
    def _load_source(self, request: pytest.FixtureRequest) -> None:
        module_source = _load_live_trader_source()
        request.cls.source = module_source
        request.cls.module_source = module_source

    def test_no_raw_timeout_5(self) -> None:
        """timeout=5 が self.trader_config.ticker_timeout に置換済み."""
        # timeout=5 が残っていないことを確認 (コメントは除外)
        lines = [
            line for line in self.source.splitlines()
            if "timeout=5" in line and not line.strip().startswith("#")
        ]
        assert not lines, f"Raw timeout=5 found: {lines}"

    def test_no_raw_timeout_10(self) -> None:
        """timeout=10 が self.trader_config.api_timeout に置換済み."""
        lines = [
            line for line in self.source.splitlines()
            if re.search(r"timeout=10\b", line)
            and not line.strip().startswith("#")
            and "trader_config" not in line
        ]
        assert not lines, f"Raw timeout=10 found: {lines}"

    def test_no_raw_sleep_60(self) -> None:
        """time.sleep(60) が self.trader_config.retry_interval に置換済み."""
        lines = [
            line for line in self.source.splitlines()
            if "sleep(60)" in line and not line.strip().startswith("#")
        ]
        assert not lines, f"Raw sleep(60) found: {lines}"

    def test_no_raw_sleep_2(self) -> None:
        """time.sleep(2) が self.trader_config.order_poll_interval に置換済み."""
        lines = [
            line for line in self.source.splitlines()
            if "sleep(2)" in line and not line.strip().startswith("#")
        ]
        assert not lines, f"Raw sleep(2) found: {lines}"

    def test_no_raw_recovery_timeout_60(self) -> None:
        """recovery_timeout=60.0 が self.trader_config.cb_recovery_timeout に置換済み."""
        lines = [
            line for line in self.source.splitlines()
            if "recovery_timeout=60" in line and not line.strip().startswith("#")
        ]
        assert not lines, f"Raw recovery_timeout=60 found: {lines}"

    def test_trader_config_referenced(self) -> None:
        """self.trader_config が使用されていることを確認."""
        assert "self.trader_config" in self.source

    def test_import_exists(self) -> None:
        """LiveTraderConfig が import されていることを確認."""
        assert "LiveTraderConfig" in self.module_source
