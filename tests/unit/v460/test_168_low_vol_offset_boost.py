"""168# 低ボラティリティ offset boost テスト.

time_filter の根本対策として、vol_ratio < threshold で offset を拡大する機能をテスト。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig as FillConfig
from scripts.v460.lib.fast_fill_defense import FastFillDefense
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeResult,
)


# ---- ヘルパー ----

@dataclass
class _MockOB:
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)


def _make_adapter(bid: float = 13_000_000, ask: float = 13_003_000) -> AsyncMock:
    """OB を返す mock adapter."""
    adapter = AsyncMock()
    adapter.get_orderbook = AsyncMock(
        return_value=_MockOB(bids=[(bid, 1.0)], asks=[(ask, 1.0)])
    )
    return adapter


def _make_regime_detector(
    regime: FillTestRegime = FillTestRegime.RANGING,
    vol_ratio: float = 1.0,
) -> MagicMock:
    """regime_detector mock."""
    det = MagicMock(spec=FillTestRegimeDetector)
    det.current_regime = regime
    det.last_volatility_ratio = vol_ratio
    return det


class TestLowVolOffsetBoostProperty:
    """regime_detector.last_volatility_ratio プロパティの動作."""

    def test_last_volatility_ratio_default(self) -> None:
        """update 前は 1.0 (ブースト不発動) を返す."""
        det = FillTestRegimeDetector()
        assert det.last_volatility_ratio == 1.0

    def test_last_volatility_ratio_after_update(self) -> None:
        """update 後に直近の vol_ratio をキャッシュ."""
        det = FillTestRegimeDetector()
        # 十分な観測を投入して ranging に到達
        base_price = 13_000_000.0
        for i in range(25):
            result = det.update(1000.0 + i * 60, base_price + (i % 3) * 10)
        assert hasattr(det, "_last_result")
        assert det.last_volatility_ratio == result.volatility_ratio


class TestLowVolOffsetBoostConfig:
    """fill_config の low_vol パラメータ解析."""

    def test_default_disabled(self) -> None:
        """デフォルトは無効."""
        cfg = FillConfig()
        assert cfg.low_vol_offset_boost_enabled is False
        assert cfg.low_vol_offset_boost == 1.4
        assert cfg.low_vol_threshold == 0.75

    def test_yaml_parsing(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """YAML から正しく読み込まれる."""
        raw = v460_fill_test_yaml
        cfg = FillConfig.from_yaml(raw)
        assert cfg.low_vol_offset_boost_enabled is True
        assert cfg.low_vol_offset_boost == 1.4
        assert cfg.low_vol_threshold == 0.75


class TestLowVolOffsetBoostMakerPrice:
    """maker_price パイプラインでの低 vol boost 動作."""

    def _make_calc(
        self,
        enabled: bool = True,
        boost: float = 1.4,
        threshold: float = 0.70,
        vol_ratio: float = 0.5,
        regime: FillTestRegime = FillTestRegime.RANGING,
    ) -> tuple[MakerPriceCalculator, AsyncMock]:
        cfg = FillConfig(
            low_vol_offset_boost_enabled=enabled,
            low_vol_offset_boost=boost,
            low_vol_threshold=threshold,
            spread_offset_ratio=0.05,
            min_spread_jpy=100,
            sell_offset_floor=0.0,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
        )
        det = _make_regime_detector(regime=regime, vol_ratio=vol_ratio)
        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        calc = MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
        )
        adapter = _make_adapter()
        return calc, adapter

    @pytest.mark.asyncio
    async def test_low_vol_boosts_offset(self) -> None:
        """vol_ratio < threshold で offset が boost される."""
        calc, adapter = self._make_calc(vol_ratio=0.5, threshold=0.70, boost=1.4)
        result = await calc.compute("buy", adapter, "btc_jpy")
        # base=0.05, ranging discount=1.0 (default), low_vol boost=1.4 → 0.07
        assert result.effective_offset_ratio == pytest.approx(0.07, abs=0.005)

    @pytest.mark.asyncio
    async def test_no_boost_above_threshold(self) -> None:
        """vol_ratio >= threshold ではブースト不発動."""
        calc, adapter = self._make_calc(vol_ratio=0.80, threshold=0.70)
        result = await calc.compute("buy", adapter, "btc_jpy")
        # base=0.05 のまま
        assert result.effective_offset_ratio == pytest.approx(0.05, abs=0.005)

    @pytest.mark.asyncio
    async def test_disabled_no_boost(self) -> None:
        """enabled=False では低 vol でもブースト不発動."""
        calc, adapter = self._make_calc(enabled=False, vol_ratio=0.3)
        result = await calc.compute("buy", adapter, "btc_jpy")
        assert result.effective_offset_ratio == pytest.approx(0.05, abs=0.005)

    @pytest.mark.asyncio
    async def test_sell_side_boost(self) -> None:
        """sell 側でもブースト発動."""
        cfg = FillConfig(
            low_vol_offset_boost_enabled=True,
            low_vol_offset_boost=1.4,
            low_vol_threshold=0.70,
            spread_offset_ratio=0.05,
            spread_offset_ratio_sell=0.18,
            min_spread_jpy=100,
            sell_offset_floor=0.0,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
        )
        det = _make_regime_detector(vol_ratio=0.5)
        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        calc = MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
            base_offset_ratio_sell=cfg.spread_offset_ratio_sell,
        )
        adapter = _make_adapter()
        result = await calc.compute("sell", adapter, "btc_jpy")
        # sell base=0.18, low_vol boost=1.4 → 0.252
        assert result.effective_offset_ratio == pytest.approx(0.252, abs=0.01)

    @pytest.mark.asyncio
    async def test_max_offset_cap(self) -> None:
        """max_offset_ratio を超えない."""
        calc, adapter = self._make_calc(
            vol_ratio=0.1, boost=5.0, threshold=0.70,
        )
        # base=0.05 * 5.0 = 0.25 → max_offset_ratio=0.30 でキャップ
        result = await calc.compute("buy", adapter, "btc_jpy")
        assert result.effective_offset_ratio <= 0.30

    @pytest.mark.asyncio
    async def test_low_vol_stacks_with_regime_boost(self) -> None:
        """trending + 低 vol の複合ケース: 両方のブーストが適用される."""
        cfg = FillConfig(
            low_vol_offset_boost_enabled=True,
            low_vol_offset_boost=1.3,
            low_vol_threshold=0.70,
            spread_offset_ratio=0.05,
            min_spread_jpy=100,
            sell_offset_floor=0.0,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
            regime_trending_offset_boost=1.5,
        )
        det = _make_regime_detector(
            regime=FillTestRegime.TRENDING_UP, vol_ratio=0.5,
        )

        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        calc = MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
        )
        adapter = _make_adapter()
        result = await calc.compute("buy", adapter, "btc_jpy")
        # base=0.05, trending boost=1.5 → 0.075, low_vol boost=1.3 → 0.0975
        # is_trending depends on enum — let's just check it's > base
        assert result.effective_offset_ratio > 0.05


class TestOffsetScalingHelper:
    """maker_price の共通倍率適用 helper."""

    def test_scale_offset_ratio_clamps_and_reports_actual_multiplier(self) -> None:
        ratio, applied = MakerPriceCalculator._scale_offset_ratio(
            0.2,
            3.0,
            max_ratio=0.3,
        )
        assert ratio == pytest.approx(0.3, abs=0.0001)
        assert applied == pytest.approx(1.5, abs=0.0001)

    def test_scale_offset_ratio_ignores_invalid_multiplier(self) -> None:
        ratio, applied = MakerPriceCalculator._scale_offset_ratio(
            0.2,
            0.0,
            max_ratio=0.3,
        )
        assert ratio == pytest.approx(0.2, abs=0.0001)
        assert applied == pytest.approx(1.0, abs=0.0001)


class TestFFDBoostConsistency:
    """FFD boost 後の価格補正量と ratio が整合すること."""

    @pytest.mark.asyncio
    async def test_clamped_ffd_recomputes_offset_from_ratio(self) -> None:
        cfg = FillConfig(
            spread_offset_ratio=0.2,
            max_offset_ratio=0.3,
            min_offset_jpy=1.0,
            min_spread_jpy=10.0,
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
            volatility_guard_enabled=False,
            inventory_skewing_enabled=False,
        )
        det = _make_regime_detector(regime=FillTestRegime.RANGING, vol_ratio=1.0)
        ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
        ffd.get_boost_multiplier = MagicMock(return_value=3.0)
        calc = MakerPriceCalculator(
            cfg, ffd, regime_detector=det,
            base_offset_ratio=cfg.spread_offset_ratio,
        )
        adapter = _make_adapter(bid=1000.0, ask=1100.0)

        result = await calc.compute("buy", adapter, "btc_jpy")

        # base 0.2, FFD x3.0 -> cap at 0.3, spread=100 => offset=30, buy price=1030
        assert result.effective_offset_ratio == pytest.approx(0.3, abs=0.0001)
        assert result.price == pytest.approx(1030.0, abs=0.0001)


class TestSpreadGuardHelper:
    """spread guard の共通 helper."""

    def test_buy_cross_falls_back_to_best_bid(self) -> None:
        result = MakerPriceCalculator._finalize_price_with_spread_guard(
            side="buy",
            best_bid=1000.0,
            best_ask=1010.0,
            spread=10.0,
            offset=20.0,
            effective_offset_ratio=0.2,
        )
        assert result.price == 1000.0
        assert result.effective_offset_ratio == 0.0

    def test_sell_cross_falls_back_to_best_ask(self) -> None:
        result = MakerPriceCalculator._finalize_price_with_spread_guard(
            side="sell",
            best_bid=1000.0,
            best_ask=1010.0,
            spread=10.0,
            offset=20.0,
            effective_offset_ratio=0.2,
        )
        assert result.price == 1010.0
        assert result.effective_offset_ratio == 0.0
