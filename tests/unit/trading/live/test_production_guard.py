from pathlib import Path

import pytest

from ztb.trading.live_trader.config import LiveTradingOptions
from ztb.trading.live_trader.live_trader import LiveTrader
from ztb.utils.exceptions.custom_exceptions import TradingError


def test_production_guard_requires_explicit_enable(tmp_path, monkeypatch):
    """LiveTrader should refuse to initialize in production when API keys are present unless explicitly allowed."""
    # Set API keys to simulate production environment
    monkeypatch.setenv("COINCHECK_API_KEY", "dummy_api_key")
    monkeypatch.setenv("COINCHECK_API_SECRET", "dummy_api_secret")
    monkeypatch.delenv("ZTB_ALLOW_PRODUCTION", raising=False)

    options = LiveTradingOptions(
        model_path=Path("package.json"),
        algorithm="sac",
        venue="coincheck",
        duration_hours=0.01,
        disable_risk_limits=False,
        dry_run=False,
        log_level="INFO",
        enable_metrics=False,
        metrics_port=8000,
        enable_health_check=False,
        health_port=8080,
        allow_production=False,
    )

    # Should raise TradingError because production is not allowed
    with pytest.raises(TradingError):
        LiveTrader(options)


def test_production_guard_allows_with_flag(tmp_path, monkeypatch):
    """LiveTrader initializes when allow_production is True."""
    monkeypatch.setenv("COINCHECK_API_KEY", "dummy_api_key")
    monkeypatch.setenv("COINCHECK_API_SECRET", "dummy_api_secret")
    monkeypatch.setenv("ZTB_ALLOW_PRODUCTION", "1")

    options = LiveTradingOptions(
        model_path=Path("package.json"),
        algorithm="sac",
        venue="coincheck",
        duration_hours=0.01,
        disable_risk_limits=False,
        dry_run=False,
        log_level="INFO",
        enable_metrics=False,
        metrics_port=8000,
        enable_health_check=False,
        health_port=8080,
        allow_production=True,
    )

    trader = LiveTrader(options)
    assert trader is not None
