"""151# P3-03: AS 確率連動ロットサイジング (confidence_lot) テスト.

§5 テスト計画 T1-T9 + §10 レビュー対応検証:
  - T1: 無効時は factor=1.0
  - T2: AS 確率 0.0 → factor=1.0
  - T3: AS 確率 0.5 → factor=0.5
  - T4: AS 確率 1.0 → floor で制限
  - T5: NaN/inf → 1.0
  - T6: None → 1.0
  - T7: regime × confidence 合成
  - T8: min_order_btc 保証
  - T9: max_lot 上限
  - §10 #2: __post_init__ バリデーション, factor クランプ [0, 1]
  - §10 #3: mode=pnl 凍結 → factor=1.0 + warning
  - §10 #5: dust_sweep_active → factor=1.0
  - §10 #7: FillRecord 新フィールド存在確認
  - FillTestConfig.from_yaml: confidence_lot セクション読込
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.run_fill_test import FillTestRunner
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# Helper: FillTestRunner の _confidence_lot_factor / _effective_order_lot
# をテスト可能にするための軽量ラッパー
# ======================================================================


def _make_runner(config: FillTestConfig) -> MagicMock:
    """FillTestRunner のモックを生成し、confidence_lot メソッドを実体化."""
    runner = MagicMock(spec=FillTestRunner)
    runner.config = config

    # _confidence_lot_factor を実メソッドとしてバインド
    runner._confidence_lot_factor = FillTestRunner._confidence_lot_factor.__get__(
        runner, FillTestRunner
    )
    runner._effective_order_lot = FillTestRunner._effective_order_lot.__get__(
        runner, FillTestRunner
    )
    return runner


# ======================================================================
# T1-T6: _confidence_lot_factor 単体テスト
# ======================================================================


class TestConfidenceLotFactor:
    """_confidence_lot_factor のユニットテスト."""

    def _factor(
        self,
        as_prob: float | None,
        *,
        enabled: bool = True,
        scale: float = 1.0,
        floor: float = 0.3,
        mode: str = "as",
        dust_sweep_active: bool = False,
    ) -> float:
        cfg = FillTestConfig(
            enable_confidence_lot=enabled,
            confidence_lot_scale=scale,
            confidence_lot_floor=floor,
            confidence_lot_mode=mode,
        )
        runner = _make_runner(cfg)
        return runner._confidence_lot_factor(as_prob, dust_sweep_active=dust_sweep_active)

    def test_t1_disabled_returns_one(self) -> None:
        """T1: 無効時は factor=1.0."""
        assert self._factor(0.5, enabled=False) == 1.0

    def test_t2_as_prob_zero(self) -> None:
        """T2: AS 確率 0.0 → factor=1.0."""
        assert self._factor(0.0) == 1.0

    def test_t3_as_prob_half(self) -> None:
        """T3: AS 確率 0.5, scale=1.0 → factor=0.5."""
        assert self._factor(0.5) == pytest.approx(0.5)

    def test_t4_as_prob_one_clamped_by_floor(self) -> None:
        """T4: AS 確率 1.0 → floor=0.3 で制限."""
        assert self._factor(1.0, floor=0.3) == pytest.approx(0.3)

    def test_t5_nan_returns_one(self) -> None:
        """T5: NaN → 1.0."""
        assert self._factor(float("nan")) == 1.0

    def test_t5_inf_returns_one(self) -> None:
        """T5: inf → 1.0."""
        assert self._factor(float("inf")) == 1.0

    def test_t5_neg_inf_returns_one(self) -> None:
        """T5: -inf → 1.0."""
        assert self._factor(float("-inf")) == 1.0

    def test_t6_none_returns_one(self) -> None:
        """T6: None → 1.0."""
        assert self._factor(None) == 1.0

    def test_various_probabilities(self) -> None:
        """各種 AS 確率値での factor 検証."""
        assert self._factor(0.1) == pytest.approx(0.9)
        assert self._factor(0.4) == pytest.approx(0.6)
        assert self._factor(0.7, floor=0.3) == pytest.approx(0.3)

    def test_scale_two(self) -> None:
        """scale=2.0 の場合のスケーリング."""
        # 1.0 - 2.0 * 0.3 = 0.4
        assert self._factor(0.3, scale=2.0) == pytest.approx(0.4)

    def test_floor_zero(self) -> None:
        """floor=0.0 で完全縮小可能."""
        assert self._factor(1.0, floor=0.0) == pytest.approx(0.0)

    def test_factor_never_exceeds_one(self) -> None:
        """§10 #2: factor は 1.0 を超えない (拡大不可)."""
        # negative as_prob は異常値だが、clamp で 1.0 以下を保証
        assert self._factor(-0.5) <= 1.0

    def test_factor_never_below_zero(self) -> None:
        """§10 #2: factor は 0.0 を下回らない."""
        # scale=10, as_prob=0.5 → 1-5=-4 → max(floor, min(1.0, max(0.0, -4))) → floor
        result = self._factor(0.5, scale=10.0, floor=0.0)
        assert result >= 0.0

    # §10 #3 + §13 #1: mode=pnl 凍結
    def test_mode_pnl_enabled_raises_valueerror(self) -> None:
        """§13 #1: enable=True + mode=pnl は __post_init__ で ValueError."""
        with pytest.raises(ValueError, match="confidence_lot_mode must be 'as' when enabled"):
            FillTestConfig(
                enable_confidence_lot=True,
                confidence_lot_mode="pnl",
            )

    def test_mode_pnl_disabled_runtime_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        """§10 #3 防御的ガード: enabled=False (→バリデーション通過) でも runtime で 1.0."""
        # enabled=False なら mode=pnl でもバリデーション通過
        cfg = FillTestConfig(
            enable_confidence_lot=False,
            confidence_lot_mode="pnl",
        )
        runner = _make_runner(cfg)
        # enabled=False → そもそも 1.0 (mode チェック前に return)
        assert runner._confidence_lot_factor(0.5) == 1.0

    # §10 #5: dust_sweep_active → factor=1.0
    def test_dust_sweep_active_returns_one(self) -> None:
        """§10 #5: dust_sweep 中は factor=1.0."""
        assert self._factor(0.5, dust_sweep_active=True) == 1.0


# ======================================================================
# T7-T9: _effective_order_lot 統合テスト
# ======================================================================


class TestEffectiveOrderLot:
    """_effective_order_lot のテスト."""

    def _lot(
        self,
        regime_lot: float,
        as_prob: float | None = None,
        *,
        enabled: bool = True,
        scale: float = 1.0,
        floor: float = 0.3,
        min_order_btc: float = 0.001,
        max_lot: float = 0.005,
        dust_sweep_active: bool = False,
    ) -> tuple[float, float]:
        cfg = FillTestConfig(
            enable_confidence_lot=enabled,
            confidence_lot_scale=scale,
            confidence_lot_floor=floor,
            confidence_lot_mode="as",
            min_order_btc=min_order_btc,
            max_lot=max_lot,
        )
        runner = _make_runner(cfg)
        return runner._effective_order_lot(
            regime_lot, as_prob=as_prob, dust_sweep_active=dust_sweep_active,
        )

    def test_t7_regime_x_confidence(self) -> None:
        """T7: regime=0.003 × confidence=0.5 → lot=0.0015."""
        lot, factor = self._lot(0.003, as_prob=0.5)
        assert factor == pytest.approx(0.5)
        assert lot == pytest.approx(0.0015)

    def test_t8_min_order_btc_guarantee(self) -> None:
        """T8: 縮小結果が min_order_btc 以下 → min_order_btc に引き上げ."""
        lot, factor = self._lot(0.001, as_prob=0.9, floor=0.1)
        assert lot >= 0.001  # min_order_btc

    def test_t9_max_lot_cap(self) -> None:
        """T9: max_lot 上限を超えない."""
        lot, factor = self._lot(0.010, as_prob=0.0, max_lot=0.005)
        assert lot <= 0.005

    def test_disabled_passthrough(self) -> None:
        """無効時は regime_lot がそのまま通る (clamp 除く)."""
        lot, factor = self._lot(0.003, as_prob=0.9, enabled=False)
        assert factor == 1.0
        assert lot == pytest.approx(0.003)

    def test_dust_sweep_passthrough(self) -> None:
        """dust_sweep 中は confidence 適用なし."""
        lot, factor = self._lot(0.003, as_prob=0.9, dust_sweep_active=True)
        assert factor == 1.0

    def test_returns_tuple(self) -> None:
        """戻り値が (lot, factor) タプル."""
        result = self._lot(0.003, as_prob=0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ======================================================================
# §10 #2: __post_init__ バリデーション
# ======================================================================


class TestConfigValidation:
    """FillTestConfig のバリデーションテスト."""

    def test_floor_out_of_range_raises(self) -> None:
        """floor > 1.0 → ValueError."""
        with pytest.raises(ValueError, match="confidence_lot_floor"):
            FillTestConfig(confidence_lot_floor=1.5)

    def test_floor_negative_raises(self) -> None:
        """floor < 0.0 → ValueError."""
        with pytest.raises(ValueError, match="confidence_lot_floor"):
            FillTestConfig(confidence_lot_floor=-0.1)

    def test_scale_negative_raises(self) -> None:
        """scale < 0 → ValueError."""
        with pytest.raises(ValueError, match="confidence_lot_scale"):
            FillTestConfig(confidence_lot_scale=-1.0)

    def test_mode_invalid_raises(self) -> None:
        """mode が 'as'/'pnl' 以外 → ValueError."""
        with pytest.raises(ValueError, match="confidence_lot_mode"):
            FillTestConfig(confidence_lot_mode="invalid")

    def test_valid_config_ok(self) -> None:
        """正常値はバリデーション通過."""
        cfg = FillTestConfig(
            enable_confidence_lot=True,
            confidence_lot_scale=1.0,
            confidence_lot_floor=0.3,
            confidence_lot_mode="as",
        )
        assert cfg.enable_confidence_lot is True
        assert cfg.confidence_lot_floor == 0.3


# ======================================================================
# from_yaml: confidence_lot セクション読込テスト
# ======================================================================


class TestFromYaml:
    """FillTestConfig.from_yaml の confidence_lot 読込テスト."""

    def test_confidence_lot_from_yaml(self) -> None:
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "confidence_lot": {
                        "enabled": True,
                        "scale": 0.8,
                        "floor": 0.4,
                        "mode": "as",
                    }
                }
            )
        )
        assert cfg.enable_confidence_lot is True
        assert cfg.confidence_lot_scale == 0.8
        assert cfg.confidence_lot_floor == 0.4
        assert cfg.confidence_lot_mode == "as"

    def test_confidence_lot_absent_uses_defaults(self) -> None:
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({}))
        assert cfg.enable_confidence_lot is False
        assert cfg.confidence_lot_scale == 1.0


# ======================================================================
# §10 #7: FillRecord 新フィールド
# ======================================================================


class TestFillRecordFields:
    """FillRecord に 151# 新フィールドが存在すること."""

    def test_confidence_lot_fields_exist(self) -> None:
        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=1e7,
            order_quantity=0.001,
            confidence_lot_factor=0.5,
            order_lot_regime=0.003,
            order_lot_effective=0.0015,
            confidence_lot_mode="as",
        )
        assert r.confidence_lot_factor == 0.5
        assert r.order_lot_regime == 0.003
        assert r.order_lot_effective == 0.0015
        assert r.confidence_lot_mode == "as"

    def test_confidence_lot_fields_default_none(self) -> None:
        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=1e7,
            order_quantity=0.001,
        )
        assert r.confidence_lot_factor is None
        assert r.order_lot_regime is None
        assert r.order_lot_effective is None
        assert r.confidence_lot_mode is None

    def test_to_dict_includes_new_fields(self) -> None:
        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=1e7,
            order_quantity=0.001,
            confidence_lot_factor=0.7,
            order_lot_regime=0.003,
        )
        d = r.to_dict()
        assert "confidence_lot_factor" in d
        assert "order_lot_regime" in d
        assert "order_lot_effective" in d
        assert "confidence_lot_mode" in d
