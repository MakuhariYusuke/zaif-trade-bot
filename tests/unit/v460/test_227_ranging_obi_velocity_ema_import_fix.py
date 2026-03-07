"""227# テスト: Ranging×OBI, velocity EMA, lazy import除去, getattr除去, config validation.

C1: Ranging × OBI 方向別非対称 offset
C3: velocity EMA smoothing (bid-ask bounce noise filter)
H1+H5: lazy import → file top-level
H2: getattr → direct attribute access
M1: Config validation (loss_boost_decay_tau_sec, ranging_obi_*, velocity_ema_alpha)
"""

from __future__ import annotations

import math
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from scripts.v460.lib.regime_detector import FillTestRegime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> FillTestConfig:
    """テスト用 FillTestConfig を生成。max > min offset 保証。"""
    defaults = dict(
        max_offset_ratio=0.02,
        min_offset_ratio=0.001,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_mp(config: FillTestConfig | None = None, regime=None) -> MakerPrice:
    """テスト用 MakerPriceCalculator。"""
    if config is None:
        config = _make_config()
    ffd = FastFillDefense(FastFillDefenseConfig(
        enabled=False, threshold_sec=60.0, offset_boost=1.2,
    ), base_offset_ratio=0.005)
    return MakerPrice(
        config=config,
        fast_fill_defense=ffd,
        regime_detector=regime,
        base_offset_ratio=0.005,
    )


def _mock_regime(regime_val: FillTestRegime) -> MagicMock:
    """レジームdetectorのモック"""
    det = MagicMock()
    det.current_regime = regime_val
    det.last_volatility_ratio = 1.0
    return det


async def _compute_with_ob(mp: MakerPrice, side: str, mid: float = 14_000_000) -> tuple:
    """OB付きcompute。"""
    half_spread = 500
    ob = SimpleNamespace(
        bids=[(mid - half_spread, 0.1)],
        asks=[(mid + half_spread, 0.1)],
    )

    class _Adapter:
        async def get_orderbook(self, _symbol: str, depth: int = 1) -> SimpleNamespace:
            del depth
            return ob

    adapter = _Adapter()
    return await mp.compute(side, adapter, "btc_jpy")


# ===========================================================================
# C1: Ranging × OBI 方向別非対称 offset テスト
# ===========================================================================

class TestRangingObiAsymmetry:
    """C1: ranging regime + OBI で buy/sell offset が非対称になること。"""

    @pytest.mark.asyncio
    async def test_ranging_obi_buy_bid_heavy(self):
        """bid厚 (imbalance>0) → buy有利 → buy offset さらに縮小."""
        cfg = _make_config(
            regime_ranging_offset_discount=0.8,
            ranging_obi_asymmetry_factor=0.3,
            ranging_obi_threshold=0.05,
        )
        det = _mock_regime(FillTestRegime.RANGING)
        mp = _make_mp(cfg, det)
        # imbalance > 0 (bid heavy)
        mp._last_imbalance = 0.4

        # warmup: set prev_mid
        mp._prev_mid_price = 14_000_000
        mp._prev_mid_time = time.time() - 60

        result = await _compute_with_ob(mp, "buy")
        # discount + OBI asymmetry → offset should be smaller than base
        assert result.effective_offset_ratio < 0.005

    @pytest.mark.asyncio
    async def test_ranging_obi_sell_bid_heavy(self):
        """bid厚 (imbalance>0) → sell不利 → sell discount 緩和."""
        cfg = _make_config(
            regime_ranging_offset_discount=0.8,
            ranging_obi_asymmetry_factor=0.3,
            ranging_obi_threshold=0.05,
        )
        det = _mock_regime(FillTestRegime.RANGING)
        mp = _make_mp(cfg, det)
        mp._last_imbalance = 0.4

        mp._prev_mid_price = 14_000_000
        mp._prev_mid_time = time.time() - 60

        result_sell = await _compute_with_ob(mp, "sell")

        # Compare with buy at same conditions
        mp2 = _make_mp(cfg, det)
        mp2._last_imbalance = 0.4
        mp2._prev_mid_price = 14_000_000
        mp2._prev_mid_time = time.time() - 60
        result_buy = await _compute_with_ob(mp2, "buy")

        # sell offset should be larger than buy offset (sell is disadvantaged)
        assert result_sell.effective_offset_ratio > result_buy.effective_offset_ratio

    @pytest.mark.asyncio
    async def test_ranging_obi_below_threshold_no_asymmetry(self):
        """OBI が threshold 以下では非対称化されない."""
        cfg = _make_config(
            regime_ranging_offset_discount=0.8,
            ranging_obi_asymmetry_factor=0.3,
            ranging_obi_threshold=0.5,  # high threshold
        )
        det = _mock_regime(FillTestRegime.RANGING)

        mp_buy = _make_mp(cfg, det)
        mp_buy._last_imbalance = 0.3  # below threshold
        mp_buy._prev_mid_price = 14_000_000
        mp_buy._prev_mid_time = time.time() - 60

        mp_sell = _make_mp(cfg, det)
        mp_sell._last_imbalance = 0.3
        mp_sell._prev_mid_price = 14_000_000
        mp_sell._prev_mid_time = time.time() - 60

        r_buy = await _compute_with_ob(mp_buy, "buy")
        r_sell = await _compute_with_ob(mp_sell, "sell")

        # Both should have the same discount (symmetric)
        assert abs(r_buy.effective_offset_ratio - r_sell.effective_offset_ratio) < 0.0005

    @pytest.mark.asyncio
    async def test_ranging_obi_disabled_when_factor_zero(self):
        """factor=0.0 ではOBI非対称化が無効."""
        cfg = _make_config(
            regime_ranging_offset_discount=0.8,
            ranging_obi_asymmetry_factor=0.0,  # disabled
            ranging_obi_threshold=0.05,
        )
        det = _mock_regime(FillTestRegime.RANGING)

        mp_buy = _make_mp(cfg, det)
        mp_buy._last_imbalance = 0.8  # very strong imbalance
        mp_buy._prev_mid_price = 14_000_000
        mp_buy._prev_mid_time = time.time() - 60

        mp_sell = _make_mp(cfg, det)
        mp_sell._last_imbalance = 0.8
        mp_sell._prev_mid_price = 14_000_000
        mp_sell._prev_mid_time = time.time() - 60

        r_buy = await _compute_with_ob(mp_buy, "buy")
        r_sell = await _compute_with_ob(mp_sell, "sell")

        # Symmetric (no OBI effect)
        assert abs(r_buy.effective_offset_ratio - r_sell.effective_offset_ratio) < 0.0005


# ===========================================================================
# C3: velocity EMA filter テスト
# ===========================================================================

class TestVelocityEma:
    """C3: velocity EMA smoothing."""

    @pytest.mark.asyncio
    async def test_ema_smooths_velocity(self):
        """EMA が velocity を平滑化する (2回目は前回値で dampened)."""
        cfg = _make_config(velocity_ema_alpha=0.3)
        mp = _make_mp(cfg)

        # 1st call: set prev values
        mp._prev_mid_price = 14_000_000
        mp._prev_mid_time = 1_000_000.0

        with patch("scripts.v460.lib.maker_price.time.time", side_effect=[1_000_060.0, 1_000_061.0]):
            await _compute_with_ob(mp, "buy", mid=14_010_000)  # ~+7.14 bps
            v1 = mp._smoothed_velocity_bps

            # 2nd call: EMA kicks in
            await _compute_with_ob(mp, "buy", mid=14_000_000)  # ~-7.14 bps raw
            v2 = mp._smoothed_velocity_bps

        # smoothed should be closer to 0 than raw -7.14 bps
        assert v1 is not None
        assert v2 is not None
        assert abs(v2) < 7.14  # dampened

    @pytest.mark.asyncio
    async def test_ema_alpha_1_no_smoothing(self):
        """alpha=1.0 のときは raw velocity そのまま (後方互換)."""
        cfg = _make_config(velocity_ema_alpha=1.0)
        mp = _make_mp(cfg)

        mp._prev_mid_price = 14_000_000
        mp._prev_mid_time = 1_000_000.0

        with patch("scripts.v460.lib.maker_price.time.time", return_value=1_000_060.0):
            await _compute_with_ob(mp, "buy", mid=14_010_000)
        # smoothed_velocity should be None when alpha=1.0
        assert mp._smoothed_velocity_bps is None

    @pytest.mark.asyncio
    async def test_ema_first_sample_passthrough(self):
        """最初のサンプルはそのまま通過 (前回値がない)。"""
        cfg = _make_config(velocity_ema_alpha=0.3)
        mp = _make_mp(cfg)

        mp._prev_mid_price = 14_000_000
        mp._prev_mid_time = 1_000_000.0

        with patch("scripts.v460.lib.maker_price.time.time", return_value=1_000_060.0):
            await _compute_with_ob(mp, "buy", mid=14_010_000)
        v1 = mp._smoothed_velocity_bps

        # First sample: no prior smoothed value → raw value is stored
        assert v1 is not None
        assert abs(v1) > 0


# ===========================================================================
# M1: Config validation テスト
# ===========================================================================

class TestConfigValidation:
    """M1: 新規パラメータの境界バリデーション."""

    def test_loss_boost_decay_tau_nonpositive_raises(self):
        with pytest.raises(ValueError, match="loss_boost_decay_tau_sec"):
            _make_config(loss_boost_decay_tau_sec=0.0)

    def test_loss_boost_decay_tau_negative_raises(self):
        with pytest.raises(ValueError, match="loss_boost_decay_tau_sec"):
            _make_config(loss_boost_decay_tau_sec=-10.0)

    def test_ranging_obi_factor_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ranging_obi_asymmetry_factor"):
            _make_config(ranging_obi_asymmetry_factor=1.5)

    def test_ranging_obi_factor_negative_raises(self):
        with pytest.raises(ValueError, match="ranging_obi_asymmetry_factor"):
            _make_config(ranging_obi_asymmetry_factor=-0.1)

    def test_ranging_obi_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="ranging_obi_threshold"):
            _make_config(ranging_obi_threshold=-0.05)

    def test_velocity_ema_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="velocity_ema_alpha"):
            _make_config(velocity_ema_alpha=0.0)

    def test_velocity_ema_alpha_over_1_raises(self):
        with pytest.raises(ValueError, match="velocity_ema_alpha"):
            _make_config(velocity_ema_alpha=1.5)

    def test_valid_new_params_no_raise(self):
        """正常値では例外なし。"""
        cfg = _make_config(
            loss_boost_decay_tau_sec=100.0,
            ranging_obi_asymmetry_factor=0.3,
            ranging_obi_threshold=0.1,
            velocity_ema_alpha=0.5,
        )
        assert cfg.loss_boost_decay_tau_sec == 100.0
        assert cfg.ranging_obi_asymmetry_factor == 0.3
        assert cfg.velocity_ema_alpha == 0.5


# ===========================================================================
# H1+H5: import 最適化テスト (smoke test)
# ===========================================================================

class TestImportOptimization:
    """H1+H5: lazy import がファイルトップに移動されていることを確認。"""

    def test_maker_price_has_math_at_module_level(self):
        """maker_price.py に math が file-level import されている。"""
        import scripts.v460.lib.maker_price as mp_mod
        assert hasattr(mp_mod, 'math')

    def test_orchestrator_has_datetime_at_module_level(self):
        """fill_loop_orchestrator に datetime が file-level import されている。"""
        import scripts.v460.lib.fill_loop_orchestrator as orch_mod
        assert hasattr(orch_mod, 'datetime')
        assert hasattr(orch_mod, 'timezone')

    def test_orchestrator_has_mcblevel_at_module_level(self):
        """orchestrator_pre_cycle に MCBLevel が file-level import されている.

        331# review: MCB/SAD ロジックは orchestrator_pre_cycle に移管済み。
        """
        import scripts.v460.lib.orchestrator_pre_cycle as pre_mod
        assert hasattr(pre_mod, 'MCBLevel')

    def test_orchestrator_has_sadlevel_at_module_level(self):
        """orchestrator_pre_cycle に SADLevel が file-level import されている.

        331# review: MCB/SAD ロジックは orchestrator_pre_cycle に移管済み。
        """
        import scripts.v460.lib.orchestrator_pre_cycle as pre_mod
        assert hasattr(pre_mod, 'SADLevel')

    def test_orchestrator_has_load_alert_mode_at_module_level(self):
        """fill_loop_orchestrator に load_alert_mode が file-level import されている。"""
        import scripts.v460.lib.fill_loop_orchestrator as orch_mod
        assert hasattr(orch_mod, 'load_alert_mode')


# ===========================================================================
# H2: getattr 除去検証 (orchestrator class-level 属性)
# ===========================================================================

class TestOrchestratorClassLevelAttrs:
    """H2: class-level 宣言済み属性が getattr なしでアクセス可能。"""

    def test_class_attrs_have_defaults(self):
        """Mixin のクラスレベル属性にデフォルト値がある。"""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert FillLoopOrchestratorMixin._soft_drawdown_interval_multiplier == 1.0
        assert FillLoopOrchestratorMixin._halt_start_cycle is None
        assert FillLoopOrchestratorMixin._in_hard_skip_hour is False
        assert FillLoopOrchestratorMixin._halt_iter_count == 0
        assert FillLoopOrchestratorMixin._alert_interval_mult == 1.0
        assert FillLoopOrchestratorMixin._last_balance_forced_time == 0.0
        assert FillLoopOrchestratorMixin._balance_forced_freq_count == 0
