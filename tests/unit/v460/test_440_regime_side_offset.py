"""440# Regime-Side Offset 非対称化テスト.

§4.2: ranging offset の buy/sell 非対称化
  - side 別 discount 解決 (buy/sell/fallback)
  - OBI 非対称は side 別 discount と共存
  - buy 側 boost (>1.0) が _scale_offset_ratio で正しく動作
§4.3: unknown sell offset boost
  - sell+unknown で boost 適用
  - buy+unknown は既存挙動維持
  - 両側とも 1.0 の場合は noop
YAML mapping:
  - ranging_offset_discount_buy/sell のパース
  - unknown_sell_offset_boost のパース
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.regime_detector import FillTestRegime
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping


# ======================================================================
# Fixture helpers (test_143 パターン準用)
# ======================================================================


class _StaticFFD:
    def maybe_expire_boost(self, _side: str) -> None:
        return None

    def _get_dynamic_boost(self, _: str) -> float | None:
        return None

    def get_boost_multiplier(self, _side: str) -> float:
        return 1.0


@dataclass(slots=True)
class _OrderBook:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class _Adapter:
    def __init__(self, orderbook: _OrderBook) -> None:
        self._orderbook = orderbook

    async def get_orderbook(
        self, _symbol: str, *, depth: int | None = None,
    ) -> _OrderBook:
        del depth
        return self._orderbook


def _mock_adapter(best_bid: float = 15_000_000, best_ask: float = 15_001_000):
    return _Adapter(_OrderBook(bids=[(best_bid, 0.1)], asks=[(best_ask, 0.1)]))


def _make_calculator(
    regime_value: str | None,
    *,
    ranging_discount: float = 1.0,
    ranging_discount_buy: float | None = None,
    ranging_discount_sell: float | None = None,
    unknown_buy_boost: float = 1.0,
    unknown_sell_boost: float = 1.0,
    base_offset: float = 0.05,
) -> tuple[MakerPriceCalculator, FillTestConfig]:
    cfg = FillTestConfig(
        regime_ranging_offset_discount=ranging_discount,
        regime_ranging_offset_discount_buy=ranging_discount_buy,
        regime_ranging_offset_discount_sell=ranging_discount_sell,
        unknown_buy_offset_boost=unknown_buy_boost,
        unknown_sell_offset_boost=unknown_sell_boost,
        spread_offset_ratio=base_offset,
        max_offset_ratio=0.30,
        min_offset_ratio=0.01,
        spread_adaptive_enabled=False,
        imbalance_enabled=False,
        volatility_guard_enabled=False,
        fast_fill_defense_enabled=False,
        sell_offset_floor=0.0,
        sell_max_spread_jpy=0.0,
    )

    regime_det = None
    if regime_value is not None:
        regime_det = MagicMock()
        regime_det.current_regime = FillTestRegime(regime_value)
        regime_det.current_confidence = 0.95
        regime_det.last_volatility_ratio = 1.0
        regime_det.regime_duration_sec = 300.0
        regime_det.get_boost_multiplier = MagicMock(return_value=1.0)

    calc = MakerPriceCalculator(
        config=cfg,
        fast_fill_defense=_StaticFFD(),
        regime_detector=regime_det,
        base_offset_ratio=base_offset,
    )
    return calc, cfg


# ======================================================================
# §4.2 Ranging Side-Specific Offset
# ======================================================================


class TestRangingSideOffset:
    """440# §4.2: ranging offset buy/sell 非対称化."""

    def test_buy_boost_widens_offset(self) -> None:
        """buy 側 discount > 1.0 で offset が拡大する."""
        calc_buy_boost, _ = _make_calculator(
            "ranging", ranging_discount_buy=1.15,
        )
        calc_base, _ = _make_calculator("ranging", ranging_discount=1.0)

        adapter = _mock_adapter()
        r_boost = asyncio.run(calc_buy_boost.compute("buy", adapter, "btc_jpy"))
        r_base = asyncio.run(calc_base.compute("buy", adapter, "btc_jpy"))

        assert r_boost.effective_offset_ratio > r_base.effective_offset_ratio

    def test_sell_discount_narrows_offset(self) -> None:
        """sell 側 discount < 1.0 で offset が縮小する."""
        calc_sell_disc, _ = _make_calculator(
            "ranging", ranging_discount_sell=0.85,
        )
        calc_base, _ = _make_calculator("ranging", ranging_discount=1.0)

        adapter = _mock_adapter()
        r_disc = asyncio.run(calc_sell_disc.compute("sell", adapter, "btc_jpy"))
        r_base = asyncio.run(calc_base.compute("sell", adapter, "btc_jpy"))

        assert r_disc.effective_offset_ratio < r_base.effective_offset_ratio

    def test_side_specific_overrides_common(self) -> None:
        """side 別設定が共通値 (0.90) を上書きする."""
        calc, _ = _make_calculator(
            "ranging",
            ranging_discount=0.90,
            ranging_discount_buy=1.15,
            ranging_discount_sell=0.85,
        )
        adapter = _mock_adapter()

        r_buy = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        r_sell = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))

        # buy は拡大方向 (1.15 > 0.90)、sell は縮小方向 (0.85 < 0.90)
        # buy の offset が sell より大きい
        assert r_buy.effective_offset_ratio > r_sell.effective_offset_ratio

    def test_none_falls_back_to_common(self) -> None:
        """side 別が None の場合は共通値にフォールバック."""
        calc_common, _ = _make_calculator(
            "ranging", ranging_discount=0.85,
        )
        calc_explicit, _ = _make_calculator(
            "ranging",
            ranging_discount=0.85,
            ranging_discount_buy=None,
            ranging_discount_sell=None,
        )
        adapter = _mock_adapter()

        r_common = asyncio.run(calc_common.compute("buy", adapter, "btc_jpy"))
        r_explicit = asyncio.run(calc_explicit.compute("buy", adapter, "btc_jpy"))

        assert abs(r_common.effective_offset_ratio - r_explicit.effective_offset_ratio) < 1e-6

    def test_noop_when_discount_is_one(self) -> None:
        """discount=1.0 の場合は offset 変更なし."""
        calc_ranging, _ = _make_calculator(
            "ranging", ranging_discount=1.0,
        )
        calc_unknown, _ = _make_calculator(
            "unknown", ranging_discount=1.0,
        )
        adapter = _mock_adapter()

        r_ranging = asyncio.run(calc_ranging.compute("buy", adapter, "btc_jpy"))
        r_unknown = asyncio.run(calc_unknown.compute("buy", adapter, "btc_jpy"))

        # ranging/unknown 両方ともデフォルト boost なし → 同等
        assert abs(r_ranging.effective_offset_ratio - r_unknown.effective_offset_ratio) < 1e-6

    def test_buy_boost_clamped_to_max(self) -> None:
        """buy 側 boost が max_offset_ratio を超えない."""
        calc, cfg = _make_calculator(
            "ranging",
            ranging_discount_buy=10.0,
            base_offset=0.25,
        )
        adapter = _mock_adapter()
        result = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert result.effective_offset_ratio <= cfg.max_offset_ratio

    def test_sell_discount_clamped_to_min(self) -> None:
        """sell 側 discount が min_offset_ratio を下回らない."""
        calc, cfg = _make_calculator(
            "ranging",
            ranging_discount_sell=0.01,
            base_offset=0.02,
        )
        adapter = _mock_adapter()
        result = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert result.effective_offset_ratio >= cfg.min_offset_ratio

    def test_not_applied_on_non_ranging_regime(self) -> None:
        """ranging 以外のレジームでは side 別 discount は無効."""
        calc_trending, _ = _make_calculator(
            "trending",
            ranging_discount_buy=1.5,
            ranging_discount_sell=0.5,
        )
        calc_base, _ = _make_calculator(
            "trending",
        )
        adapter = _mock_adapter()

        r1 = asyncio.run(calc_trending.compute("buy", adapter, "btc_jpy"))
        r2 = asyncio.run(calc_base.compute("buy", adapter, "btc_jpy"))

        assert abs(r1.effective_offset_ratio - r2.effective_offset_ratio) < 1e-6


# ======================================================================
# §4.3 Unknown Sell Offset Boost
# ======================================================================


class TestUnknownSellOffset:
    """440# §4.3: unknown sell offset boost."""

    def test_sell_boost_applied(self) -> None:
        """unknown+sell で boost が適用される."""
        calc_boost, _ = _make_calculator(
            "unknown", unknown_sell_boost=1.3,
        )
        calc_base, _ = _make_calculator(
            "unknown", unknown_sell_boost=1.0,
        )
        adapter = _mock_adapter()

        r_boost = asyncio.run(calc_boost.compute("sell", adapter, "btc_jpy"))
        r_base = asyncio.run(calc_base.compute("sell", adapter, "btc_jpy"))

        assert r_boost.effective_offset_ratio > r_base.effective_offset_ratio

    def test_buy_boost_unchanged(self) -> None:
        """unknown_sell_offset_boost は buy には影響しない."""
        calc, _ = _make_calculator(
            "unknown",
            unknown_buy_boost=1.0,
            unknown_sell_boost=1.3,
        )
        calc_base, _ = _make_calculator(
            "unknown",
            unknown_buy_boost=1.0,
            unknown_sell_boost=1.0,
        )
        adapter = _mock_adapter()

        r = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        r_base = asyncio.run(calc_base.compute("buy", adapter, "btc_jpy"))

        assert abs(r.effective_offset_ratio - r_base.effective_offset_ratio) < 1e-6

    def test_buy_existing_boost_still_works(self) -> None:
        """既存の unknown_buy_offset_boost は引き続き動作する."""
        calc_boost, _ = _make_calculator(
            "unknown", unknown_buy_boost=2.0,
        )
        calc_base, _ = _make_calculator(
            "unknown", unknown_buy_boost=1.0,
        )
        adapter = _mock_adapter()

        r_boost = asyncio.run(calc_boost.compute("buy", adapter, "btc_jpy"))
        r_base = asyncio.run(calc_base.compute("buy", adapter, "btc_jpy"))

        assert r_boost.effective_offset_ratio > r_base.effective_offset_ratio

    def test_noop_when_both_1(self) -> None:
        """buy/sell とも 1.0 なら unknown regime でも offset 変更なし."""
        calc, _ = _make_calculator(
            "unknown", unknown_buy_boost=1.0, unknown_sell_boost=1.0,
        )
        calc_ranging, _ = _make_calculator("ranging")
        adapter = _mock_adapter()

        r_unk = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        r_rng = asyncio.run(calc_ranging.compute("buy", adapter, "btc_jpy"))

        assert abs(r_unk.effective_offset_ratio - r_rng.effective_offset_ratio) < 1e-6


# ======================================================================
# YAML Mapping
# ======================================================================


class TestYamlMapping440:
    """440# YAML → FillTestConfig マッピング."""

    def test_ranging_discount_buy_sell(self) -> None:
        yaml_data = {
            "regime": {
                "ranging_offset_discount": 0.90,
                "ranging_offset_discount_buy": 1.15,
                "ranging_offset_discount_sell": 0.85,
            },
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.regime_ranging_offset_discount == 0.90
        assert cfg.regime_ranging_offset_discount_buy == 1.15
        assert cfg.regime_ranging_offset_discount_sell == 0.85

    def test_ranging_discount_buy_sell_absent(self) -> None:
        yaml_data = {
            "regime": {
                "ranging_offset_discount": 0.90,
            },
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.regime_ranging_offset_discount == 0.90
        assert cfg.regime_ranging_offset_discount_buy is None
        assert cfg.regime_ranging_offset_discount_sell is None

    def test_unknown_sell_offset_boost(self) -> None:
        yaml_data = {
            "skip_gate": {
                "unknown_sell_offset_boost": 1.3,
            },
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.unknown_sell_offset_boost == 1.3

    def test_unknown_sell_offset_boost_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.unknown_sell_offset_boost == 1.0


# ======================================================================
# FillTestConfig field defaults
# ======================================================================


class TestConfigDefaults440:
    """440# 新規フィールドのデフォルト値."""

    def test_ranging_discount_buy_none(self) -> None:
        cfg = FillTestConfig()
        assert cfg.regime_ranging_offset_discount_buy is None

    def test_ranging_discount_sell_none(self) -> None:
        cfg = FillTestConfig()
        assert cfg.regime_ranging_offset_discount_sell is None

    def test_unknown_sell_boost_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.unknown_sell_offset_boost == 1.0
