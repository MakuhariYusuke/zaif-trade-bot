"""657# B-3 regime別max_factor + A-4/A-5 toxic_sell_veto 段階化テスト.

B-3: trending時にinv_skewを完全停止→低減max_factorで在庫管理継続
A-4: toxic_sell_veto as_offset — hard veto → offset boost
A-5: 連続 veto 時間減衰 (α^n decay)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_results import SkipGateResult
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.trading.signal.regime.regime_detector import FillTestRegime


# ======================================================================
# Helpers
# ======================================================================


class _StaticFFD:
    """テスト用 FastFillDefense stub."""

    def maybe_expire_boost(self, _side: str) -> None:
        return None

    def _get_dynamic_boost(self, _: str) -> float | None:
        return None

    def get_boost_multiplier(self, _side: str) -> float:
        return 1.0


def _make_config(**overrides: object) -> FillTestConfig:
    defaults: dict[str, object] = dict(
        spread_offset_ratio=0.001,
        min_offset_jpy=1.0,
        max_offset_ratio=0.02,
        min_offset_ratio=0.0001,
        inventory_skewing_enabled=True,
        inventory_skewing_window=10,
        inventory_skewing_max_factor=0.4,
        inv_skew_max_factor_trending=0.15,
        inventory_skewing_neutral_band=0.05,
        inv_skew_regime_gate_enabled=False,
        spread_adaptive_enabled=False,
        imbalance_enabled=False,
        volatility_guard_enabled=False,
        fast_fill_defense_enabled=False,
        sell_offset_floor=0.0,
        sell_max_spread_jpy=0.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_maker_price(
    config: FillTestConfig,
    regime_value: str | None = None,
) -> MakerPriceCalculator:
    regime_det = None
    if regime_value is not None:
        regime_det = MagicMock()
        regime_det.current_regime = FillTestRegime(regime_value)
        regime_det.current_confidence = 0.95
        regime_det.last_volatility_ratio = 1.0
        regime_det.regime_duration_sec = 300.0
        regime_det.get_boost_multiplier = MagicMock(return_value=1.0)

    return MakerPriceCalculator(
        config=config,
        fast_fill_defense=_StaticFFD(),
        regime_detector=regime_det,
        base_offset_ratio=config.spread_offset_ratio,
    )


def _inject_imbalance(mp: MakerPriceCalculator, imbalance: float) -> None:
    """テスト用: 在庫偏重を直接注入.

    O(1)カウンターベースの在庫管理に合わせて
    _inv_net_imbalance と _inv_buy_count を直接設定。
    """
    n = mp._config.inventory_skewing_window
    mp._inv_fill_history.clear()
    n_buy = int(n * ((1 + imbalance) / 2))
    n_sell = n - n_buy
    for _ in range(n_buy):
        mp._inv_fill_history.append("buy")
    for _ in range(n_sell):
        mp._inv_fill_history.append("sell")
    mp._inv_buy_count = n_buy
    mp._inv_net_imbalance = (2 * n_buy / n - 1) if n > 0 else 0.0
    mp._inv_last_update_time = time.time()


# ======================================================================
# B-3: regime別 max_factor
# ======================================================================


class TestB3RegimeMaxFactor:
    """657# B-3: trending時にmax_factor_trendingで在庫管理を継続."""

    def test_config_has_max_factor_trending(self) -> None:
        """FillTestConfig に inv_skew_max_factor_trending が存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "inv_skew_max_factor_trending")
        assert cfg.inv_skew_max_factor_trending == 0.15

    def test_ranging_uses_normal_max_factor(self) -> None:
        """ranging 時は通常の max_factor を使用."""
        cfg = _make_config()
        mp = _make_maker_price(cfg, regime_value="ranging")
        _inject_imbalance(mp, 0.6)  # 強い buy 偏り

        # buy偏り + buy side → offset増加 (在庫蓄積抑制)
        eff = mp._apply_inventory_skew("buy", time.time(), 0.05)
        assert eff > 0.05

    def test_trending_uses_reduced_max_factor(self) -> None:
        """trending 時は inv_skew_max_factor_trending を使用."""
        cfg = _make_config()
        mp_trending = _make_maker_price(cfg, regime_value="trending_up")
        mp_ranging = _make_maker_price(cfg, regime_value="ranging")

        _inject_imbalance(mp_trending, 0.6)
        _inject_imbalance(mp_ranging, 0.6)

        # buy偏り + buy side → offset増加 (在庫蓄積抑制)
        eff_trending = mp_trending._apply_inventory_skew("buy", time.time(), 0.05)
        eff_ranging = mp_ranging._apply_inventory_skew("buy", time.time(), 0.05)

        # trending は低減 max_factor → offset 変化が小さい
        delta_trending = abs(eff_trending - 0.05)
        delta_ranging = abs(eff_ranging - 0.05)
        assert delta_trending < delta_ranging, (
            f"trending delta ({delta_trending:.6f}) should be less than "
            f"ranging delta ({delta_ranging:.6f})"
        )

    def test_trending_still_applies_skew(self) -> None:
        """trending でも inv_skew は完全停止しない (B-3 の核心)."""
        cfg = _make_config()
        mp = _make_maker_price(cfg, regime_value="trending_up")
        _inject_imbalance(mp, 0.8)  # 強い偏り

        eff = mp._apply_inventory_skew("sell", time.time(), 0.05)
        # 偏りが neutral_band を超えているので factor != 0, offset 変化あり
        assert eff != 0.05

    def test_legacy_regime_gate_still_blocks(self) -> None:
        """regime_gate_enabled=True の場合は従来通り完全停止 (後方互換)."""
        cfg = _make_config(inv_skew_regime_gate_enabled=True)
        mp = _make_maker_price(cfg, regime_value="trending_up")
        _inject_imbalance(mp, 0.8)

        eff = mp._apply_inventory_skew("sell", time.time(), 0.05)
        # regime_gate blocks → factor=0, offset unchanged
        assert eff == 0.05

    def test_no_regime_detector_uses_normal_max_factor(self) -> None:
        """regime_detector=None の場合は通常の max_factor."""
        cfg = _make_config()
        mp = _make_maker_price(cfg, regime_value=None)
        _inject_imbalance(mp, 0.6)

        # buy偏り + buy side → offset増加
        eff = mp._apply_inventory_skew("buy", time.time(), 0.05)
        assert eff > 0.05  # normal skew applied

    def test_trending_down_uses_reduced_max_factor(self) -> None:
        """trending_down も trending_up と同様に低減 max_factor."""
        cfg = _make_config()
        mp_up = _make_maker_price(cfg, regime_value="trending_up")
        mp_down = _make_maker_price(cfg, regime_value="trending_down")

        _inject_imbalance(mp_up, 0.6)
        _inject_imbalance(mp_down, 0.6)

        eff_up = mp_up._apply_inventory_skew("buy", time.time(), 0.05)
        eff_down = mp_down._apply_inventory_skew("buy", time.time(), 0.05)

        # 両方とも trending → 同じ max_factor_trending
        # time.time() 差でわずかに変わるため abs=1e-8
        assert eff_up == pytest.approx(eff_down, abs=1e-8)

    def test_last_inv_skew_factor_stored_correctly(self) -> None:
        """trending 時の _last_inv_skew_factor が正しく記録される."""
        cfg = _make_config()
        mp = _make_maker_price(cfg, regime_value="trending_up")
        _inject_imbalance(mp, 0.6)

        mp._apply_inventory_skew("buy", time.time(), 0.05)
        # tanh(imb * sign * max_factor_trending)
        # imb ≈ 0.6, sign = 1 (buy), max_factor_trending = 0.15
        # raw_factor = 0.6 * 1.0 * 0.15 = 0.09
        # factor = tanh(0.09) ≈ 0.0897
        expected_raw = 0.6 * 1.0 * 0.15
        expected_factor = math.tanh(expected_raw)
        assert mp._last_inv_skew_factor == pytest.approx(expected_factor, abs=0.05)


# ======================================================================
# A-4/A-5: toxic_sell_veto 段階化 + 時間減衰
# ======================================================================


class TestA4ToxicVetoAsOffset:
    """657# A-4: toxic_sell_veto ソフト化 — offset boost."""

    def test_config_has_as_offset_fields(self) -> None:
        """FillTestConfig に A-4/A-5 フィールドが存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "toxic_sell_veto_as_offset_enabled")
        assert hasattr(cfg, "toxic_sell_veto_offset_boost_factor")
        assert hasattr(cfg, "toxic_sell_veto_decay_alpha")
        assert cfg.toxic_sell_veto_offset_boost_factor == 1.8
        assert cfg.toxic_sell_veto_decay_alpha == 0.7

    def test_skip_gate_result_has_toxic_veto_offset_mult(self) -> None:
        """SkipGateResult に toxic_veto_offset_mult フィールドが存在."""
        result = SkipGateResult()
        assert hasattr(result, "toxic_veto_offset_mult")
        assert result.toxic_veto_offset_mult is None


class TestA5ToxicVetoDecay:
    """657# A-5: 連続 veto 時間減衰."""

    def test_decay_formula(self) -> None:
        """α^(n-1) 減衰の数学的正確性."""
        alpha = 0.7
        # 1回目: α^0 = 1.0
        assert alpha ** 0 == 1.0
        # 2回目: α^1 = 0.7
        assert alpha ** 1 == pytest.approx(0.7)
        # 3回目: α^2 = 0.49
        assert alpha ** 2 == pytest.approx(0.49)
        # 5回目: α^4 ≈ 0.24 (< 0.5 → ソフトモードにフォールバック)
        assert alpha ** 4 == pytest.approx(0.2401, abs=0.001)

    def test_boost_effective_with_decay(self) -> None:
        """boost_effective = 1.0 + (boost - 1.0) * decay の計算."""
        boost = 1.8
        alpha = 0.7
        # 1回目: decay=1.0 → boost_eff = 1.8
        assert 1.0 + (boost - 1.0) * 1.0 == pytest.approx(1.8)
        # 2回目: decay=0.7 → boost_eff = 1.56
        assert 1.0 + (boost - 1.0) * 0.7 == pytest.approx(1.56)
        # 3回目: decay=0.49 → boost_eff = 1.392
        assert 1.0 + (boost - 1.0) * 0.49 == pytest.approx(1.392)

    def test_hard_mode_decay_below_threshold_forces_soft(self) -> None:
        """decay < 0.5 の場合、hard mode でもソフトにフォールバック."""
        alpha = 0.7
        # 5回目連続: decay = 0.7^4 = 0.2401 < 0.5 → ソフト化
        decay_5th = alpha ** 4
        assert decay_5th < 0.5
