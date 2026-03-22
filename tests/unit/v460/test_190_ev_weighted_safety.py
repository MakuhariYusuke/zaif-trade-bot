"""190# テスト: ev_weighted デッドロック対策 + min_spread_jpy 緩和.

- A: ev_weighted 連続 skip 安全弁 (max_ev_consecutive_skip)
- B: 片側 balance 時の ev_weighted threshold 緩和
- C: min_spread_jpy YAML 更新
- D: pnl_threshold YAML 更新
- config_hot_reload キー追加
- fill_config YAML パース整合性
"""

from __future__ import annotations

import copy
import inspect
import math
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

_RUN_SINGLE_CYCLE_SIG = inspect.signature(FillCycleExecutorMixin.run_single_cycle)
_EVALUATE_SKIP_GATE_SIG = inspect.signature(FillCycleExecutorMixin._evaluate_skip_gate)
_SKIP_GATE_EVALUATE_SIG = inspect.signature(SkipGateEvaluator.evaluate)

# ======================================================================
# 1. ev_weighted 連続 skip 安全弁 (190# A)
# ======================================================================


class TestEvWeightedConsecutiveSkipSafety:
    """190# A: ev_weighted の連続 skip カウンタと安全弁."""

    def _make_evaluator(
        self,
        *,
        ev_enabled: bool = True,
        max_consecutive: int = 5,
        one_sided_shift: float = -1.0,
    ) -> "SkipGateEvaluator":
        """テスト用の SkipGateEvaluator (最小構成) を生成."""
        config = FillTestConfig(
            skip_gate_enabled=False,  # モデルロード不要
            skip_gate_ev_weighted_enabled=ev_enabled,
            skip_gate_ev_w30=0.4,
            skip_gate_ev_w120=0.6,
            skip_gate_ev_max_consecutive_skip=max_consecutive,
            skip_gate_ev_one_sided_threshold_shift=one_sided_shift,
        )
        evaluator = SkipGateEvaluator(config, Path("."))
        return evaluator

    def _make_decision(
        self, pnl: float, threshold: float = 0.1
    ) -> MagicMock:
        """primary_decision のモック."""
        d = MagicMock()
        d.predicted_pnl_bps = pnl
        d.threshold_used = threshold
        d.threshold_bps = 0.0
        d.features_used = 13
        d.as_probability = None
        d.reason = "skip" if pnl < threshold else "pass"
        d.model_used = "primary"
        d.should_skip = pnl < threshold
        return d

    def _make_alt_gate(self, pred_pnl: float = -2.0) -> MagicMock:
        """alt model ゲートのモック."""
        gate = MagicMock()
        gate.config.mode = "pnl"
        mock_decision = MagicMock()
        mock_decision.predicted_pnl_bps = pred_pnl
        mock_decision.threshold_bps = 0.0
        mock_decision.features_used = 19
        mock_decision.as_probability = None
        mock_decision.threshold_used = 0.1
        mock_decision.should_skip = True
        mock_decision.reason = "skip"
        mock_decision.model_used = "alt"
        gate.evaluate.return_value = mock_decision
        return gate

    def test_counter_increments_on_skip(self) -> None:
        """ev_weighted skip 時にカウンタが増加すること."""
        evaluator = self._make_evaluator(max_consecutive=10)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-5.0)

        primary = self._make_decision(pnl=-3.0, threshold=0.1)
        features: dict[str, object] = {"side": "buy"}

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is True
        assert evaluator._ev_consecutive_skip_count == 1

    def test_counter_resets_on_pass(self) -> None:
        """ev_weighted PASS 時にカウンタがリセットされること."""
        evaluator = self._make_evaluator(max_consecutive=10)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=5.0)

        primary = self._make_decision(pnl=3.0, threshold=0.1)
        features: dict[str, object] = {"side": "buy"}

        # まず skip を蓄積
        evaluator._ev_consecutive_skip_count = 3

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is False
        assert evaluator._ev_consecutive_skip_count == 0

    def test_safety_valve_triggers_at_limit(self) -> None:
        """max_consecutive_skip 到達時に強制 PASS すること."""
        evaluator = self._make_evaluator(max_consecutive=3)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-5.0)
        evaluator._ev_consecutive_skip_count = 2  # あと1回で発動

        primary = self._make_decision(pnl=-3.0, threshold=0.1)
        features: dict[str, object] = {"side": "buy"}

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
        )
        # 3回目の skip → 安全弁発動 → 強制 PASS
        assert result is not None
        assert result.should_skip is False
        assert evaluator._ev_consecutive_skip_count == 0
        assert "safety" in result.reason or "pass" in result.reason

    def test_safety_valve_not_triggered_below_limit(self) -> None:
        """limit 未満の場合は通常 skip."""
        evaluator = self._make_evaluator(max_consecutive=5)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-5.0)
        evaluator._ev_consecutive_skip_count = 3  # 5 未満

        primary = self._make_decision(pnl=-3.0, threshold=0.1)
        features: dict[str, object] = {"side": "buy"}

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is True
        assert evaluator._ev_consecutive_skip_count == 4

    def test_safety_valve_disabled_when_zero(self) -> None:
        """max_consecutive=0 → 安全弁無効 (永続 skip, カウンタ不変)."""
        evaluator = self._make_evaluator(max_consecutive=0)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-5.0)
        evaluator._ev_consecutive_skip_count = 100

        primary = self._make_decision(pnl=-3.0, threshold=0.1)
        features: dict[str, object] = {"side": "buy"}

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is True
        # max_consecutive=0 → カウンタはインクリメントも発動もしない
        assert evaluator._ev_consecutive_skip_count == 100

    def test_sell_side_uses_same_counter(self) -> None:
        """sell 側も同一の連続 skip カウンタを共有すること."""
        evaluator = self._make_evaluator(max_consecutive=3)
        evaluator._gate_alt_sell = self._make_alt_gate(pred_pnl=-5.0)
        evaluator._ev_consecutive_skip_count = 2

        primary = self._make_decision(pnl=-3.0, threshold=0.1)
        features: dict[str, object] = {"side": "sell"}

        result = evaluator._try_ev_weighted_decision(
            "sell", features, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is False  # 安全弁発動
        assert evaluator._ev_consecutive_skip_count == 0


# ======================================================================
# 2. 片側 balance 時の threshold 緩和 (190# B)
# ======================================================================


class TestOneSidedBalanceThresholdRelaxation:
    """190# B: one_sided_balance 時の ev_weighted threshold 緩和."""

    def _make_evaluator(
        self,
        *,
        shift: float = -1.0,
        max_consecutive: int = 0,
    ) -> "SkipGateEvaluator":
        config = FillTestConfig(
            skip_gate_enabled=False,
            skip_gate_ev_weighted_enabled=True,
            skip_gate_ev_w30=0.4,
            skip_gate_ev_w120=0.6,
            skip_gate_ev_max_consecutive_skip=max_consecutive,
            skip_gate_ev_one_sided_threshold_shift=shift,
        )
        return SkipGateEvaluator(config, Path("."))

    def _make_primary(self, pnl: float = -0.5, threshold: float = 0.1) -> MagicMock:
        d = MagicMock()
        d.predicted_pnl_bps = pnl
        d.threshold_used = threshold
        d.threshold_bps = 0.0
        d.features_used = 13
        d.as_probability = None
        d.reason = "skip"
        d.model_used = "primary"
        d.should_skip = pnl < threshold
        return d

    def _make_alt_gate(self, pred_pnl: float) -> MagicMock:
        gate = MagicMock()
        gate.config.mode = "pnl"
        mock_decision = MagicMock()
        mock_decision.predicted_pnl_bps = pred_pnl
        mock_decision.threshold_bps = 0.0
        mock_decision.features_used = 19
        mock_decision.as_probability = None
        mock_decision.threshold_used = 0.1
        mock_decision.should_skip = True
        mock_decision.reason = "skip"
        mock_decision.model_used = "alt"
        gate.evaluate.return_value = mock_decision
        return gate

    def test_one_sided_relaxes_threshold(self) -> None:
        """one_sided_balance=True → threshold が shift 分下がる."""
        evaluator = self._make_evaluator(shift=-1.0)
        # ev_score = w30*(-0.5) + w120*(-0.3) = 0.4*(-0.5) + 0.6*(-0.3) = -0.38
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-0.3)

        primary = self._make_primary(pnl=-0.5, threshold=0.1)
        features: dict[str, object] = {}

        # one_sided=False → ev_score=-0.38 < threshold=0.1 → skip
        result_normal = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
            one_sided_balance=False,
        )
        assert result_normal is not None
        assert result_normal.should_skip is True

        # リセット
        evaluator._ev_consecutive_skip_count = 0

        # one_sided=True → threshold=0.1+(-1.0)=-0.9 → ev_score=-0.38 > -0.9 → PASS
        result_relaxed = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
            one_sided_balance=True,
        )
        assert result_relaxed is not None
        assert result_relaxed.should_skip is False

    def test_no_relaxation_when_shift_zero(self) -> None:
        """shift=0.0 → one_sided_balance=True でも緩和なし."""
        evaluator = self._make_evaluator(shift=0.0)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-0.3)

        primary = self._make_primary(pnl=-0.5, threshold=0.1)
        features: dict[str, object] = {}

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
            one_sided_balance=True,
        )
        assert result is not None
        assert result.should_skip is True  # 緩和なし → skip

    def test_one_sided_false_no_relaxation(self) -> None:
        """one_sided_balance=False → shift 値に関わらず通常 threshold."""
        evaluator = self._make_evaluator(shift=-3.0)
        evaluator._gate_alt_buy = self._make_alt_gate(pred_pnl=-0.3)

        primary = self._make_primary(pnl=-0.5, threshold=0.1)
        features: dict[str, object] = {}

        result = evaluator._try_ev_weighted_decision(
            "buy", features, "ranging", 0.0, primary,
            one_sided_balance=False,
        )
        assert result is not None
        assert result.should_skip is True  # 緩和なし → skip


# ======================================================================
# 3. FillTestConfig 新フィールド
# ======================================================================


class TestFillConfigNewFields:
    """190# FillTestConfig の新規フィールド確認."""

    def test_default_max_consecutive_skip(self) -> None:
        cfg = FillTestConfig()
        assert cfg.skip_gate_ev_max_consecutive_skip == 0

    def test_default_one_sided_threshold_shift(self) -> None:
        cfg = FillTestConfig()
        assert cfg.skip_gate_ev_one_sided_threshold_shift == 0.0

    def test_yaml_parse_new_fields(self) -> None:
        """YAML パーサーが新フィールドを読み込めること."""
        yaml_dict = {
            "skip_gate": {
                "ev_max_consecutive_skip": 5,
                "ev_one_sided_threshold_shift": -1.0,
            },
        }
        config = FillTestConfig.from_yaml(yaml_dict)
        assert config.skip_gate_ev_max_consecutive_skip == 5
        assert config.skip_gate_ev_one_sided_threshold_shift == -1.0


# ======================================================================
# 4. config_hot_reload キー検証
# ======================================================================


class TestHotReload190Keys:
    """190# 新規フィールドの hot-reload 設定テスト."""

    def test_ev_max_consecutive_skip_hot_reloadable(self) -> None:
        assert "skip_gate_ev_max_consecutive_skip" in _HOT_RELOADABLE_FIELDS

    def test_ev_one_sided_threshold_shift_hot_reloadable(self) -> None:
        assert "skip_gate_ev_one_sided_threshold_shift" in _HOT_RELOADABLE_FIELDS

    def test_pnl_threshold_hot_reloadable(self) -> None:
        """190#: pnl_threshold も hot-reload 可能であること."""
        assert "skip_gate_pnl_threshold" in _HOT_RELOADABLE_FIELDS


# ======================================================================
# 5. YAML 整合性テスト (190# 更新分)
# ======================================================================


class TestYAMLIntegrity190:
    """190# YAML 更新の整合性検証."""

    @pytest.fixture(scope="class")
    def yaml_config(self, v460_fill_test_yaml_base: dict[str, object]) -> dict:
        return copy.deepcopy(v460_fill_test_yaml_base)

    def test_min_spread_jpy_reduced(self, yaml_config: dict) -> None:
        """190# C: min_spread_jpy が現行 YAML の緩和値に追随している."""
        assert yaml_config.get("min_spread_jpy") == 500  # 535# 700→500

    def test_pnl_threshold_relaxed(self, yaml_config: dict) -> None:
        """190# D: pnl_threshold が 0.0→-0.5 に更新."""
        sg = yaml_config.get("skip_gate", {})
        assert sg.get("pnl_threshold") == -0.5

    def test_ev_max_consecutive_skip_set(self, yaml_config: dict) -> None:
        """190# A: ev_max_consecutive_skip が 5 に設定."""
        sg = yaml_config.get("skip_gate", {})
        assert sg.get("ev_max_consecutive_skip") == 5

    def test_ev_one_sided_threshold_shift_set(self, yaml_config: dict) -> None:
        """190# B: ev_one_sided_threshold_shift が -1.0 に設定."""
        sg = yaml_config.get("skip_gate", {})
        assert sg.get("ev_one_sided_threshold_shift") == -1.0

    def test_ev_weights_unchanged(self, yaml_config: dict) -> None:
        """190# で ev_w30/w120 は変更されていないこと."""
        sg = yaml_config.get("skip_gate", {})
        assert sg.get("ev_w30") == 0.4
        assert sg.get("ev_w120") == 0.6


# ======================================================================
# 6. run_single_cycle / evaluate パラメータ整合性
# ======================================================================


class TestOneSidedBalanceParam:
    """190# B: one_sided_balance パラメータの伝搬テスト."""

    def test_run_single_cycle_accepts_one_sided_balance(self) -> None:
        """run_single_cycle が one_sided_balance キーワード引数を受け付けること."""
        params = _RUN_SINGLE_CYCLE_SIG.parameters
        assert "one_sided_balance" in params
        assert params["one_sided_balance"].default is False

    def test_evaluate_skip_gate_accepts_one_sided_balance(self) -> None:
        """_evaluate_skip_gate が one_sided_balance キーワード引数を受け付けること."""
        params = _EVALUATE_SKIP_GATE_SIG.parameters
        assert "one_sided_balance" in params
        assert params["one_sided_balance"].default is False

    def test_evaluator_evaluate_accepts_one_sided_balance(self) -> None:
        """SkipGateEvaluator.evaluate が one_sided_balance を受け付けること."""
        params = _SKIP_GATE_EVALUATE_SIG.parameters
        assert "one_sided_balance" in params
        assert params["one_sided_balance"].default is False


# ======================================================================
# 7. ev_weighted reason 文字列テスト
# ======================================================================


class TestEvWeightedReasonStrings:
    """190# ev_weighted のスキップ理由文字列が正しく設定されること."""

    def _make_evaluator(self, max_consec: int = 5) -> "SkipGateEvaluator":
        config = FillTestConfig(
            skip_gate_enabled=False,
            skip_gate_ev_weighted_enabled=True,
            skip_gate_ev_max_consecutive_skip=max_consec,
        )
        return SkipGateEvaluator(config, Path("."))

    def _make_alt_gate(self, pred: float) -> MagicMock:
        gate = MagicMock()
        gate.config.mode = "pnl"
        d = MagicMock()
        d.predicted_pnl_bps = pred
        d.threshold_bps = 0.0
        d.features_used = 19
        d.as_probability = None
        d.threshold_used = 0.1
        d.should_skip = True
        d.reason = "skip"
        d.model_used = "alt"
        gate.evaluate.return_value = d
        return gate

    def _make_primary(self, pnl: float) -> MagicMock:
        d = MagicMock()
        d.predicted_pnl_bps = pnl
        d.threshold_used = 0.1
        d.threshold_bps = 0.0
        d.features_used = 13
        d.as_probability = None
        d.reason = "skip"
        d.model_used = "primary"
        d.should_skip = True
        return d

    def test_reason_ev_weighted_skip(self) -> None:
        """通常の ev_weighted skip → reason=ev_weighted_skip."""
        evaluator = self._make_evaluator()
        evaluator._gate_alt_buy = self._make_alt_gate(-5.0)

        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, self._make_primary(-3.0),
        )
        assert result is not None
        assert result.reason == "ev_weighted_skip"

    def test_reason_ev_weighted_pass(self) -> None:
        """正常 PASS → reason=ev_weighted_pass."""
        evaluator = self._make_evaluator()
        evaluator._gate_alt_buy = self._make_alt_gate(5.0)

        primary = self._make_primary(3.0)
        primary.threshold_used = 0.1
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.reason == "ev_weighted_pass"

    def test_reason_safety_valve_pass(self) -> None:
        """安全弁発動 PASS → reason に safety が含まれること."""
        evaluator = self._make_evaluator(max_consec=2)
        evaluator._gate_alt_buy = self._make_alt_gate(-5.0)
        evaluator._ev_consecutive_skip_count = 1  # 次で発動

        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, self._make_primary(-3.0),
        )
        assert result is not None
        assert result.should_skip is False
        assert "safety" in result.reason or "pass" in result.reason


# ======================================================================
# 8. 統合テスト: 連続 skip → 安全弁 → 取引再開フロー
# ======================================================================


class TestEvWeightedDeadlockScenario:
    """190# 実際のデッドロックシナリオの検証."""

    def test_deadlock_scenario_resolves(self) -> None:
        """BTC=0→buy のみ→ev_weighted 5連続 skip→安全弁で取引再開."""
        evaluator = SkipGateEvaluator(
            FillTestConfig(
                skip_gate_enabled=False,
                skip_gate_ev_weighted_enabled=True,
                skip_gate_ev_max_consecutive_skip=5,
                skip_gate_ev_one_sided_threshold_shift=-1.0,
            ),
            Path("."),
        )

        # alt gate mock (常に悲観的な予測)
        gate = MagicMock()
        gate.config.mode = "pnl"
        d = MagicMock()
        d.predicted_pnl_bps = -4.0
        d.threshold_bps = 0.0
        d.features_used = 19
        d.as_probability = None
        d.threshold_used = 0.1
        d.should_skip = True
        d.reason = "skip"
        d.model_used = "alt"
        gate.evaluate.return_value = d
        evaluator._gate_alt_buy = gate

        primary = MagicMock()
        primary.predicted_pnl_bps = -3.0
        primary.threshold_used = 0.1
        primary.threshold_bps = 0.0
        primary.features_used = 13
        primary.as_probability = None
        primary.reason = "skip"
        primary.model_used = "primary"
        primary.should_skip = True

        # 4 回連続 skip (one_sided=True で threshold 緩和しても -3.4 < -0.9)
        results = []
        for i in range(5):
            result = evaluator._try_ev_weighted_decision(
                "buy", {}, "ranging", 0.0, primary,
                one_sided_balance=True,
            )
            results.append(result)

        # 最初 4 回は skip
        for r in results[:4]:
            assert r is not None
            assert r.should_skip is True

        # 5 回目で安全弁発動 → PASS
        assert results[4] is not None
        assert results[4].should_skip is False
        assert evaluator._ev_consecutive_skip_count == 0

    def test_marginal_score_passes_with_one_sided(self) -> None:
        """marginally negative score が one_sided 緩和で PASS になること."""
        evaluator = SkipGateEvaluator(
            FillTestConfig(
                skip_gate_enabled=False,
                skip_gate_ev_weighted_enabled=True,
                skip_gate_ev_max_consecutive_skip=0,  # 安全弁無効
                skip_gate_ev_one_sided_threshold_shift=-1.0,
            ),
            Path("."),
        )

        gate = MagicMock()
        gate.config.mode = "pnl"
        d = MagicMock()
        d.predicted_pnl_bps = 0.0  # alt は中立
        d.threshold_bps = 0.0
        d.features_used = 19
        d.as_probability = None
        d.threshold_used = 0.1
        d.should_skip = True
        d.reason = "skip"
        d.model_used = "alt"
        gate.evaluate.return_value = d
        evaluator._gate_alt_buy = gate

        primary = MagicMock()
        primary.predicted_pnl_bps = -0.5  # marginally negative
        primary.threshold_used = 0.1
        primary.threshold_bps = 0.0
        primary.features_used = 13
        primary.as_probability = None
        primary.reason = "skip"
        primary.model_used = "primary"
        primary.should_skip = True

        # ev_score = 0.4*(-0.5) + 0.6*(0.0) = -0.2
        # threshold = 0.1 + (-1.0) = -0.9
        # -0.2 > -0.9 → PASS
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
            one_sided_balance=True,
        )
        assert result is not None
        assert result.should_skip is False
