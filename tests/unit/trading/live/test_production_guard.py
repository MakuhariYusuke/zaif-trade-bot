from pathlib import Path
from unittest.mock import Mock, patch

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
        model_path=Path("pytest.ini"),
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
        model_path=Path("pytest.ini"),
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

    with patch(
        "ztb.trading.live_trader.live_trader.get_broker_registry"
    ) as mock_registry, patch(
        "ztb.trading.live_trader.live_trader.ModelLoading"
    ) as mock_model_loading, patch(
        "ztb.trading.live_trader.live_trader.ModelManager"
    ) as mock_model_manager, patch(
        "ztb.trading.live_trader.live_trader.prometheus_available", False
    ):
        mock_registry.return_value.get_broker.return_value = Mock()
        mock_model_loading.return_value.load_model.return_value = Mock()
        mock_model_manager.return_value.initialize_model.return_value = None
        trader = LiveTrader(options)
    assert trader is not None
