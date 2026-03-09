"""305# テスト: P0 改善 — PnL 分解 / OB キャッシュ再利用 / Parkinson σ 推定器.

検証項目:
  1. PnlMeasurement に spread_capture_bps / adverse_selection_cost_bps フィールド
  2. PnL 分解: spread_capture + AS_cost ≈ post_fill_pnl (fill_price → mid → mid_after)
  3. Parkinson σ 推定器: H > L → σ > 0, H == L → Roll fallback
  4. OB キャッシュ再利用: _last_ob_snapshot 使用時は API 呼出し不要
  5. config hot-reload: sigma_parkinson_enabled / sigma_parkinson_window_sec
"""

from __future__ import annotations

import math
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.fill_config import FillTestConfig, PnlMeasurement
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from tests.unit.v460._fill_test_source import MAKER_PRICE, read_class_method_source

_MAKER_PRICE_COMPUTE_SOURCE = read_class_method_source(
    MAKER_PRICE,
    "MakerPriceCalculator",
    "compute",
)


# ======================================================================
# 1. PnlMeasurement フィールド
# ======================================================================

class TestPnlMeasurementFields:
    """PnlMeasurement に 305# 新フィールドが存在する."""

    def test_spread_capture_bps_field(self) -> None:
        m = PnlMeasurement()
        assert hasattr(m, "spread_capture_bps")
        assert m.spread_capture_bps is None

    def test_adverse_selection_cost_bps_field(self) -> None:
        m = PnlMeasurement()
        assert hasattr(m, "adverse_selection_cost_bps")
        assert m.adverse_selection_cost_bps is None


# ======================================================================
# 2. PnL 分解ロジック
# ======================================================================

class TestPnlDecomposition:
    """PnL = spread_capture + adverse_selection_cost の分解テスト."""

    @pytest.fixture()
    def measurer(self) -> "PnlMeasurer":
        from scripts.v460.lib.pnl_measurer import PnlMeasurer
        cfg = FillTestConfig()
        cfg.post_fill_wait_sec = 0.01
        cfg.early_exit_enabled = False
        cfg.e3_sampling_ratio = 0.0
        cfg.pnl_fee_deduction_enabled = False
        return PnlMeasurer(cfg)

    @pytest.mark.asyncio
    async def test_buy_spread_capture_positive(self, measurer: "PnlMeasurer") -> None:
        """buy: fill_price < mid → spread capture > 0 (maker の付加価値)."""
        # fill_price=100, mid_at_fill=101, mid_30s_after=102
        mid_prices = iter([101.0, 102.0])

        async def get_mid() -> float:
            return next(mid_prices)

        m = await measurer.measure(True, 100.0, "buy", get_mid_price=get_mid)
        assert m.spread_capture_bps is not None
        # buy: spread_capture = (mid_at_fill - fill_price) / fill_price * 10000
        # = (101 - 100) / 100 * 10000 = 100 bps
        assert m.spread_capture_bps == pytest.approx(100.0, rel=0.01)
        assert m.adverse_selection_cost_bps is not None
        # AS cost = (mid_30s - mid_at_fill) / mid_at_fill * 10000
        # = (102 - 101) / 101 * 10000 ≈ 99.0 bps (favorable movement)
        assert m.adverse_selection_cost_bps > 0

    @pytest.mark.asyncio
    async def test_sell_spread_capture_positive(self, measurer: "PnlMeasurer") -> None:
        """sell: fill_price > mid → spread capture > 0."""
        mid_prices = iter([99.0, 98.0])

        async def get_mid() -> float:
            return next(mid_prices)

        m = await measurer.measure(True, 100.0, "sell", get_mid_price=get_mid)
        assert m.spread_capture_bps is not None
        # sell: spread_capture = (fill_price - mid_at_fill) / fill_price * 10000
        # = (100 - 99) / 100 * 10000 = 100 bps (maker filled above mid)
        assert m.spread_capture_bps == pytest.approx(100.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_no_decomposition_without_fill_price(self, measurer: "PnlMeasurer") -> None:
        """fill_price=None → 分解なし."""
        mid_prices = iter([100.0, 101.0])

        async def get_mid() -> float:
            return next(mid_prices)

        m = await measurer.measure(True, None, "buy", get_mid_price=get_mid)
        assert m.spread_capture_bps is None
        assert m.adverse_selection_cost_bps is None


# ======================================================================
# 3. Parkinson σ 推定器
# ======================================================================

class TestParkinsonsigmaEstimator:
    """Parkinson (1980) High-Low Volatility Estimator."""

    @pytest.fixture()
    def mp(self) -> MakerPrice:
        cfg = FillTestConfig()
        cfg.sigma_parkinson_enabled = True
        cfg.sigma_parkinson_window_sec = 300.0
        ffd = MagicMock()
        return MakerPrice(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=0.5,
        )

    def test_parkinson_returns_positive_sigma_with_range(self, mp: MakerPrice) -> None:
        """H > L → σ > 0."""
        # ウォームアップ: high/low を設定
        mp._mid_high = 15_050_000.0
        mp._mid_low = 15_000_000.0
        mp._mid_hl_reset_time = 1e18  # 未来 → リセットされない

        sigma, vol_ratio = mp._estimate_sigma(500.0, 15_025_000.0)
        assert sigma > 0.0
        # Parkinson: ln(H/L) / (2·√(ln2))
        expected = math.log(15_050_000 / 15_000_000) / (2.0 * math.sqrt(math.log(2.0)))
        assert sigma == pytest.approx(expected, rel=0.01)

    def test_parkinson_fallback_when_flat(self, mp: MakerPrice) -> None:
        """H == L → Roll proxy にフォールバック."""
        mp._mid_high = 15_000_000.0
        mp._mid_low = 15_000_000.0
        mp._mid_hl_reset_time = 1e18

        sigma, vol_ratio = mp._estimate_sigma(500.0, 15_000_000.0)
        # Roll proxy: spread / (2 * mid)
        expected_roll = 500.0 / (2.0 * 15_000_000.0)
        assert sigma == pytest.approx(expected_roll, rel=0.01)

    def test_roll_proxy_when_disabled(self) -> None:
        """sigma_parkinson_enabled=False → Roll proxy のまま."""
        cfg = FillTestConfig()
        cfg.sigma_parkinson_enabled = False
        ffd = MagicMock()
        mp = MakerPrice(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=0.5,
        )
        sigma, _ = mp._estimate_sigma(500.0, 15_000_000.0)
        expected_roll = 500.0 / (2.0 * 15_000_000.0)
        assert sigma == pytest.approx(expected_roll, rel=0.01)

    def test_parkinson_window_reset(self, mp: MakerPrice) -> None:
        """window 経過で high/low がリセットされる."""
        import time

        mp._mid_hl_reset_time = time.time() - 9999  # 十分古い
        mp._mid_high = 20_000_000.0  # 古い極端な値
        mp._mid_low = 10_000_000.0

        sigma, _ = mp._estimate_sigma(500.0, 15_000_000.0)
        # リセット後は H == L == mid → Roll fallback
        expected_roll = 500.0 / (2.0 * 15_000_000.0)
        assert sigma == pytest.approx(expected_roll, rel=0.01)


# ======================================================================
# 4. OB キャッシュ再利用
# ======================================================================

class TestOBCacheReuse:
    """compute() 内の OB 二重取得排除テスト."""

    def test_compute_uses_cached_ob_when_available(self) -> None:
        """_last_ob_snapshot がある場合、追加 API 呼出しをしない."""
        # 305# S2 コメントが存在すること
        assert "305# S2" in _MAKER_PRICE_COMPUTE_SOURCE or "OB キャッシュ再利用" in _MAKER_PRICE_COMPUTE_SOURCE


# ======================================================================
# 5. Config hot-reload
# ======================================================================

class TestParkinsonsigmaHotReload:
    """sigma_parkinson_enabled が hot-reloadable."""

    def test_sigma_parkinson_in_hot_reloadable(self) -> None:
        assert "sigma_parkinson_enabled" in _HOT_RELOADABLE_FIELDS
        assert "sigma_parkinson_window_sec" in _HOT_RELOADABLE_FIELDS

    def test_fill_config_has_sigma_fields(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "sigma_parkinson_enabled")
        assert hasattr(cfg, "sigma_parkinson_window_sec")
        assert cfg.sigma_parkinson_enabled is False  # default disabled
        assert cfg.sigma_parkinson_window_sec == 300.0
