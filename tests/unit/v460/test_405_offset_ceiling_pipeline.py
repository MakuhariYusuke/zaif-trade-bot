"""405# Offset Ceiling Pipeline Fix テスト.

sell_floor (0.30) == max_offset_ratio (0.30) のデッドロックが解消され、
sell 側の中間ブーストが有効に機能することを検証する。
cf. 403# §3, 404# Action 1
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig as FillConfig
from ztb.trading.risk.fast_fill_defense import FastFillDefense
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.trading.signal.regime.regime_detector import FillTestRegime, FillTestRegimeDetector


# ---- helpers ----


def _mock_regime(
    regime: FillTestRegime = FillTestRegime.HIGH_VOL,
    vol_ratio: float = 1.5,
    confidence: float = 0.75,
) -> MagicMock:
    det = MagicMock(spec=FillTestRegimeDetector)
    det.current_regime = regime
    det.last_volatility_ratio = vol_ratio
    det.current_confidence = confidence
    return det


def _make_adapter(bid: float = 13_000_000, ask: float = 13_003_000) -> AsyncMock:
    adapter = AsyncMock()

    class _OB:
        def __init__(self):
            self.bids = [(bid, 1.0)]
            self.asks = [(ask, 1.0)]

    adapter.get_orderbook = AsyncMock(return_value=_OB())
    return adapter


# ================================================================
# 1. _effective_max_ratio 単体テスト
# ================================================================


class TestEffectiveMaxRatio:
    """_effective_max_ratio が side に応じた正しい intermediate cap を返す."""

    def _make_calc(
        self,
        max_offset_ratio: float = 0.30,
        offset_ceiling_ratio_sell: float | None = 0.50,
        offset_ceiling_ratio_buy: float | None = 0.20,
    ) -> MakerPriceCalculator:
        cfg = FillConfig(
            max_offset_ratio=max_offset_ratio,
            offset_ceiling_ratio_sell=offset_ceiling_ratio_sell,
            offset_ceiling_ratio_buy=offset_ceiling_ratio_buy,
            spread_offset_ratio=0.10,
            min_spread_jpy=100,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
        )
        det = _mock_regime()
        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        return MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
        )

    def test_sell_uses_ceiling(self) -> None:
        """sell: max(0.30, 0.50) = 0.50."""
        calc = self._make_calc()
        assert calc._effective_max_ratio("sell") == 0.50

    def test_buy_uses_max_offset(self) -> None:
        """buy: max(0.30, 0.20) = 0.30 — 既存動作維持."""
        calc = self._make_calc()
        assert calc._effective_max_ratio("buy") == 0.30

    def test_sell_none_ceiling_falls_back(self) -> None:
        """sell ceiling が None なら max_offset_ratio にフォールバック."""
        calc = self._make_calc(offset_ceiling_ratio_sell=None)
        assert calc._effective_max_ratio("sell") == 0.30

    def test_buy_none_ceiling_falls_back(self) -> None:
        """buy ceiling が None なら max_offset_ratio にフォールバック."""
        calc = self._make_calc(offset_ceiling_ratio_buy=None)
        assert calc._effective_max_ratio("buy") == 0.30

    def test_sell_ceiling_lower_than_max(self) -> None:
        """sell ceiling < max_offset の場合、max_offset が使われる."""
        calc = self._make_calc(max_offset_ratio=0.40, offset_ceiling_ratio_sell=0.35)
        assert calc._effective_max_ratio("sell") == 0.40


# ================================================================
# 2. sell 側デッドロック解消テスト
# ================================================================


class TestSellDeadlockResolution:
    """sell_floor (0.30) + 旧 max_ratio (0.30) のデッドロックが解消."""

    def _make_calc(
        self,
        sell_floor: float = 0.30,
        max_offset_ratio: float = 0.30,
        offset_ceiling_sell: float = 0.50,
        regime: FillTestRegime = FillTestRegime.HIGH_VOL,
        vol_ratio: float = 1.5,
        confidence: float = 0.75,
        high_vol_boost: float = 1.5,
        mid_conf_boost: float = 1.2,
    ) -> tuple[MakerPriceCalculator, AsyncMock]:
        cfg = FillConfig(
            spread_offset_ratio=0.10,
            spread_offset_ratio_sell=0.10,
            min_spread_jpy=100,
            sell_offset_floor=sell_floor,
            max_offset_ratio=max_offset_ratio,
            offset_ceiling_ratio_sell=offset_ceiling_sell,
            offset_ceiling_ratio_buy=0.20,
            offset_ceiling_ratio=0.15,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
            regime_high_vol_offset_boost=high_vol_boost,
            low_vol_offset_boost_enabled=False,
            regime_mid_confidence_offset_boost=mid_conf_boost,
            regime_mid_confidence_lo=0.70,
            regime_mid_confidence_hi=0.90,
        )
        det = _mock_regime(regime=regime, vol_ratio=vol_ratio, confidence=confidence)
        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        calc = MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
        )
        adapter = _make_adapter()
        return calc, adapter

    @pytest.mark.asyncio
    async def test_sell_high_vol_boost_above_floor(self) -> None:
        """sell high_vol boost がフロア 0.30 を超えて適用される."""
        calc, adapter = self._make_calc(
            sell_floor=0.30,
            high_vol_boost=1.5,
            regime=FillTestRegime.HIGH_VOL,
        )
        result = await calc.compute("sell", adapter, "btc_jpy")
        # base=0.10 → sell_floor=0.30 → high_vol 1.5x → 0.45
        # final ceiling 0.50 → 0.45 通過
        assert result.effective_offset_ratio > 0.30, (
            f"sell boost should exceed floor 0.30, got {result.effective_offset_ratio}"
        )
        assert result.effective_offset_ratio <= 0.50

    @pytest.mark.asyncio
    async def test_sell_mid_confidence_boost_above_floor(self) -> None:
        """sell mid_confidence boost がフロア 0.30 を超えて適用される."""
        calc, adapter = self._make_calc(
            sell_floor=0.30,
            high_vol_boost=1.0,  # high_vol ブースト無効化
            mid_conf_boost=1.5,
            confidence=0.80,
            regime=FillTestRegime.RANGING,  # high_vol でない
        )
        result = await calc.compute("sell", adapter, "btc_jpy")
        # base=0.10 → sell_floor=0.30 → mid_conf 1.5x → 0.45
        assert result.effective_offset_ratio > 0.30, (
            f"mid_conf boost should exceed floor 0.30, got {result.effective_offset_ratio}"
        )

    @pytest.mark.asyncio
    async def test_sell_final_ceiling_still_enforced(self) -> None:
        """最終 ceiling (0.50) は引き続き適用される."""
        calc, adapter = self._make_calc(
            sell_floor=0.30,
            high_vol_boost=3.0,  # 極端なブースト
        )
        result = await calc.compute("sell", adapter, "btc_jpy")
        assert result.effective_offset_ratio <= 0.50, (
            f"final ceiling should clamp at 0.50, got {result.effective_offset_ratio}"
        )


# ================================================================
# 3. buy 側の動作不変テスト
# ================================================================


class TestBuySideBehaviorPreserved:
    """buy 側の intermediate cap = 0.30 は変更なし."""

    def _make_calc(
        self,
        regime: FillTestRegime = FillTestRegime.HIGH_VOL,
        high_vol_boost: float = 1.8,
    ) -> tuple[MakerPriceCalculator, AsyncMock]:
        cfg = FillConfig(
            spread_offset_ratio=0.20,
            min_spread_jpy=100,
            max_offset_ratio=0.30,
            offset_ceiling_ratio_sell=0.50,
            offset_ceiling_ratio_buy=0.20,
            offset_ceiling_ratio=0.15,
            sell_offset_floor=0.0,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
            regime_high_vol_offset_boost=high_vol_boost,
            low_vol_offset_boost_enabled=False,
        )
        det = _mock_regime(regime=regime, vol_ratio=1.5)
        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        calc = MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
        )
        return calc, _make_adapter()

    @pytest.mark.asyncio
    async def test_buy_capped_by_buy_ceiling(self) -> None:
        """buy は intermediate cap (max_offset_ratio=0.30) でクランプされる.

        523# で maker_price 中間 ceiling を撤廃し offset_pipeline の
        execution_final_clamp に一本化したため、compute() 出力は
        intermediate cap (max_offset_ratio) まで到達可能。
        最終 ceiling (0.20) は offset_pipeline 側で適用される。
        """
        calc, adapter = self._make_calc(high_vol_boost=1.8)
        result = await calc.compute("buy", adapter, "btc_jpy")
        # base=0.20 → high_vol 1.8x → 0.36 → intermediate cap 0.30
        # 523#: 最終 ceiling は offset_pipeline の execution_final_clamp で適用
        assert result.effective_offset_ratio <= 0.30


# ================================================================
# 4. _scale_offset_ratio 単体テスト (既存の確認)
# ================================================================


class TestScaleOffsetRatio:
    """_scale_offset_ratio の基本動作確認."""

    def test_basic_multiply(self) -> None:
        updated, applied = MakerPriceCalculator._scale_offset_ratio(0.10, 1.5)
        assert updated == pytest.approx(0.15)
        assert applied == pytest.approx(1.5)

    def test_capped_by_max(self) -> None:
        updated, applied = MakerPriceCalculator._scale_offset_ratio(
            0.25, 2.0, max_ratio=0.30,
        )
        assert updated == pytest.approx(0.30)
        assert applied == pytest.approx(1.2)

    def test_floor_by_min(self) -> None:
        updated, applied = MakerPriceCalculator._scale_offset_ratio(
            0.10, 0.3, min_ratio=0.05,
        )
        assert updated == pytest.approx(0.05)
        assert applied == pytest.approx(0.5)

    def test_zero_ratio_noop(self) -> None:
        updated, applied = MakerPriceCalculator._scale_offset_ratio(0.0, 1.5)
        assert updated == 0.0
        assert applied == 1.0

    def test_zero_multiplier_noop(self) -> None:
        updated, applied = MakerPriceCalculator._scale_offset_ratio(0.10, 0.0)
        assert updated == 0.10
        assert applied == 1.0
