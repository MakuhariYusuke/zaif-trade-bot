"""475# テスト: ranging_sell_low_vol — buy 側と対称のフィルタ."""

import pytest
from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator
from scripts.v460.lib.fill_config import FillTestConfig


def _make_config(**overrides: object) -> FillTestConfig:
    defaults: dict[str, object] = dict(
        skip_gate_enabled=False,
        skip_gate_ev_weighted_enabled=False,
        skip_ranging_buy_low_vol=False,
        skip_ranging_sell_low_vol=False,
        ranging_sell_low_vol_as_offset=False,
        low_vol_threshold=0.75,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _evaluate(config: FillTestConfig, side: str, regime: str, vol_ratio: float) -> object:
    gate = CycleGateAggregator(config)
    return gate.evaluate(
        side=side,
        regime=regime,
        vol_ratio=vol_ratio,
        inv_net_imbalance=0.0,
        is_buy_killed=False,
        is_sell_killed=False,
    )


class TestRangingSellLowVolHardSkip:
    """475# ranging_sell_low_vol ハードスキップ."""

    def test_sell_blocked_when_ranging_low_vol(self) -> None:
        config = _make_config(skip_ranging_sell_low_vol=True)
        result = _evaluate(config, "sell", "ranging", 0.5)
        assert result.blocked is True
        assert result.blocking_reason == "ranging_sell_low_vol_skip"

    def test_sell_not_blocked_when_vol_above_threshold(self) -> None:
        config = _make_config(skip_ranging_sell_low_vol=True)
        result = _evaluate(config, "sell", "ranging", 0.8)
        assert result.blocked is False

    def test_sell_not_blocked_when_trending(self) -> None:
        config = _make_config(skip_ranging_sell_low_vol=True)
        result = _evaluate(config, "sell", "trending", 0.5)
        assert result.blocked is False

    def test_buy_not_affected(self) -> None:
        config = _make_config(skip_ranging_sell_low_vol=True)
        result = _evaluate(config, "buy", "ranging", 0.5)
        assert result.blocked is False

    def test_disabled_by_default(self) -> None:
        config = _make_config()
        result = _evaluate(config, "sell", "ranging", 0.5)
        assert result.blocked is False


class TestRangingSellLowVolSoftMode:
    """475# ranging_sell_low_vol ソフト化 (offset 委譲)."""

    def test_soft_mode_does_not_block(self) -> None:
        config = _make_config(
            skip_ranging_sell_low_vol=True,
            ranging_sell_low_vol_as_offset=True,
        )
        result = _evaluate(config, "sell", "ranging", 0.5)
        assert result.blocked is False

    def test_soft_mode_gate_check_present(self) -> None:
        config = _make_config(
            skip_ranging_sell_low_vol=True,
            ranging_sell_low_vol_as_offset=True,
        )
        result = _evaluate(config, "sell", "ranging", 0.5)
        gate_names = [c.gate_name for c in result.checks]
        assert "ranging_sell_low_vol" in gate_names


class TestConfigDefaults:
    """475# config デフォルト値."""

    def test_skip_ranging_sell_low_vol_default_false(self) -> None:
        config = FillTestConfig()
        assert config.skip_ranging_sell_low_vol is False

    def test_ranging_sell_low_vol_as_offset_default_false(self) -> None:
        config = FillTestConfig()
        assert config.ranging_sell_low_vol_as_offset is False
