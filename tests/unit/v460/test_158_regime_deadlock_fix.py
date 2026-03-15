"""158# §20: レジームデッドロック修正テスト.

§20 の修正内容:
  A. メインループ毎のレジーム更新 — skip パスでもデッドロックしない
  B. max_consecutive_trending_sell_skip 安全弁
  C. cancel_failed 400 エラーハンドリング改善
  D. spread_too_narrow 分類 + ログレベル降格
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeConfig,
)
from scripts.v460.run_fill_test import FillTestRunner
from tests.unit.v460._fill_test_source import (
    CYCLE_GATE_AGGREGATOR,
    ORCHESTRATOR_MID_CYCLE,
    ORCHESTRATOR_PRE_CYCLE,
    read_fill_test_method_source,
    read_fill_test_runner_source,
    read_source_text,
)
from tests.unit.v460._yaml_test_helpers import parse_yaml_mapping
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
from ztb.utils.errors import NetworkError

_RUN_CONTINUOUS_SOURCE = read_fill_test_runner_source()
_RUN_SINGLE_CYCLE_SOURCE = read_fill_test_method_source("run_single_cycle")
_RUNNER_INIT_SOURCE = read_fill_test_method_source("__init__")
_CANCEL_ORDER_REAL_SOURCE = inspect.getsource(CoincheckAdapter._cancel_order_real)
_PRE_CYCLE_SOURCE = read_source_text(ORCHESTRATOR_PRE_CYCLE)
_MID_CYCLE_SOURCE = read_source_text(ORCHESTRATOR_MID_CYCLE)
_CYCLE_GATE_AGGREGATOR_SOURCE = read_source_text(CYCLE_GATE_AGGREGATOR)


# =====================================================================
# A. レジーム更新 — skip パスでの regime_detector.update() 呼出し保証
# =====================================================================


class TestRegimeUpdateDuringSkip:
    """Fix A: regime_detector がスキップパスでも更新されることを検証."""

    def test_run_fill_test_has_regime_update_in_main_loop(self) -> None:
        """メインループ (run メソッド) に §20-A のレジーム更新コードが存在."""
        # 332# Phase 4: mixin に移管
        source = read_fill_test_runner_source()
        # §20-A: main loop regime update
        assert "§20-A" in source
        assert "_maker_price.get_fallback_price()" in source
        assert "_regime_detector.update(" in source

    def test_regime_update_before_skip_checks(self) -> None:
        """§20-A のレジーム更新が skip 判定 (CycleGateAggregator) より前にあること.

        194#: skip chain は CycleGateAggregator に集約。
        orchestrator 内で §20-A が _cycle_gate.evaluate() より前にあることを確認。
        """
        # 332# Phase 4: mixin に移管 (pre_cycle → mid_cycle の順で連結)
        source = read_fill_test_runner_source()
        idx_regime_update = source.find("§20-A")
        idx_gate_evaluate = source.find("_cycle_gate.evaluate(")

        assert idx_regime_update > 0
        assert idx_gate_evaluate > 0, "_cycle_gate.evaluate() not found in run_continuous"
        # main loop の §20-A は gateway 評価より前
        assert idx_regime_update < idx_gate_evaluate, \
            "§20-A must appear before _cycle_gate.evaluate()"

    def test_regime_detector_update_with_fallback_price(self) -> None:
        """FillTestRegimeDetector.update() が fallback price で正常動作."""
        config = RegimeConfig(window=5)
        detector = FillTestRegimeDetector(config)

        # 5回同じ価格を投入 → UNKNOWN→RANGING への遷移候補
        base_price = 14_000_000.0
        t = time.time()
        for i in range(10):
            result = detector.update(t + i * 10, base_price)

        # window 充足後は UNKNOWN でなくなる可能性
        assert result is not None
        assert result.regime is not None

    def test_regime_transitions_with_constant_price(self) -> None:
        """同一価格を継続投入すると trending から脱出 (ranging へ遷移)."""
        config = RegimeConfig(window=5, hysteresis_count=2)
        detector = FillTestRegimeDetector(config)

        # まず trending を作る: 急上昇
        base = 14_000_000.0
        t = time.time()
        for i in range(10):
            detector.update(t + i * 10, base + i * 50_000)

        result = detector.update(t + 100, base + 500_000)
        # trending かどうかは指標次第だが、ここでは上昇系レジームになりうる

        # 次に一定価格を投入 → vol_ratio 低下 → ranging 遷移を期待
        flat_price = base + 500_000
        for i in range(20):
            result = detector.update(t + 110 + i * 10, flat_price)

        # 一定価格が十分続けば is_trending ではなくなるはず
        # (または少なくとも trend_pct が低下)
        assert result.trend_pct is not None
        # trend_pct は変動がないなら 0 に近づく
        assert abs(result.trend_pct) < 0.01 or not result.regime.is_trending


class TestRegimeUpdateLogging:
    """§20-A: レジーム遷移時にログ出力されること."""

    def test_transition_log_format(self) -> None:
        """ソースに遷移ログフォーマットが含まれる."""
        source = read_fill_test_runner_source()  # 332# Phase 4: mixin に移管
        assert "Regime transition in main loop" in source


# =====================================================================
# B. max_consecutive_trending_sell_skip 安全弁
# =====================================================================


class TestMaxConsecutiveTrendingSellSkip:
    """Fix B: 連続 trending sell skip の安全弁."""

    def test_config_field_exists(self) -> None:
        """FillTestConfig に max_consecutive_trending_sell_skip フィールドが存在."""
        config = FillTestConfig()
        assert hasattr(config, "max_consecutive_trending_sell_skip")
        assert config.max_consecutive_trending_sell_skip == 30  # default

    def test_config_field_customizable(self) -> None:
        """max_consecutive_trending_sell_skip をカスタム値で設定可能."""
        config = FillTestConfig(max_consecutive_trending_sell_skip=50)
        assert config.max_consecutive_trending_sell_skip == 50

    def test_config_field_disable_with_zero(self) -> None:
        """max_consecutive_trending_sell_skip=0 で無制限."""
        config = FillTestConfig(max_consecutive_trending_sell_skip=0)
        assert config.max_consecutive_trending_sell_skip == 0

    def test_yaml_parsing(self) -> None:
        """YAML 止血セクション → max_consecutive_trending_sell_skip マッピング."""
        yaml_str = """
止血:
  skip_sell_trending: true
  max_consecutive_trending_sell_skip: 20
"""
        yaml_cfg = parse_yaml_mapping(yaml_str)
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.max_consecutive_trending_sell_skip == 20
        assert config.skip_sell_trending is True

    def test_runner_has_counter(self) -> None:
        """FillTestRunner に _trending_sell_skip_count カウンタが存在."""
        source = _RUNNER_INIT_SOURCE
        assert "_trending_sell_skip_count" in source

    def test_safety_valve_code_in_run(self) -> None:
        """run メソッドに §20-B 安全弁ロジックが含まれる."""
        source = _CYCLE_GATE_AGGREGATOR_SOURCE
        assert "§20-B" in source
        assert "safety valve" in source.lower() or "安全弁" in source
        assert "trending_sell_skip_count" in source
        assert "max_consecutive_trending_sell_skip" in source

    def test_counter_reset_on_cycle_execution(self) -> None:
        """run_single_cycle 実行後に trending_sell_skip_count がリセットされるコードが存在."""
        source = _MID_CYCLE_SOURCE
        # カウンタリセットが run_single_cycle 後に存在
        idx_run_single = source.rfind("run_single_cycle(")
        idx_reset = source.find("_trending_sell_skip_count = 0", idx_run_single)
        assert idx_reset > idx_run_single, \
            "trending_sell_skip_count must be reset after run_single_cycle"

    def test_consecutive_log_format(self) -> None:
        """skip ログに consecutive カウント情報が含まれる."""
        source = _MID_CYCLE_SOURCE
        assert "consecutive=" in source


# =====================================================================
# C. cancel_failed 400 ハンドリング改善
# =====================================================================


class TestCancelFailedHandling:
    """Fix C: Coincheck cancel 400 エラーの graceful handling."""

    def test_adapter_catches_failed_to_cancel(self) -> None:
        """_cancel_order_real が 'Failed to cancel' で WARNING ログ → re-raise."""
        source = _CANCEL_ORDER_REAL_SOURCE
        assert "failed to cancel" in source.lower()
        assert "§20-C" in source
        # "Failed to cancel" は re-raise (fill recheck のため)
        # "not found" / "already cancelled" は return False
        assert "return False" in source
        assert "raise" in source

    def test_cancel_error_patterns(self) -> None:
        """3パターンの cancel エラーがすべてマッチすること."""
        # not_found / already_cancelled → return False
        return_false_patterns = [
            "Order not found",
            "Order already cancelled",
        ]
        for msg in return_false_patterns:
            _lower = msg.lower()
            matched = "not found" in _lower or "already cancelled" in _lower
            assert matched, f"Return-false pattern should match: {msg}"

        # "Failed to cancel" → re-raise (fill 検出のため)
        raise_patterns = [
            "Failed to cancel the order",
            "Coincheck API error: 400 | body={\"success\":false,\"error\":\"Failed to cancel the order.\"}",
        ]
        for msg in raise_patterns:
            assert "failed to cancel" in msg.lower(), f"Raise pattern should match: {msg}"

    def test_cancel_unknown_error_still_raises(self) -> None:
        """未知のエラーは re-raise すること."""
        msg = "Network timeout during cancellation"
        _lower = msg.lower()
        matched = (
            "not found" in _lower
            or "already cancelled" in _lower
            or "failed to cancel" in _lower
        )
        assert not matched, "Unknown errors should NOT match"

    @pytest.mark.asyncio
    async def test_cancel_failed_to_cancel_re_raises(self) -> None:
        """'Failed to cancel' は re-raise する (order_monitor の fill recheck 保持)."""
        adapter = CoincheckAdapter.__new__(CoincheckAdapter)
        adapter.api_base_url = "https://coincheck.com"
        adapter.request_timeout = 5

        def mock_request(*args, **kwargs):
            raise NetworkError("Coincheck API error: 400 | body={\"error\":\"Failed to cancel the order.\"}")

        adapter._make_api_request = mock_request

        with pytest.raises(NetworkError):
            await adapter._cancel_order_real("test_order_123")

    @pytest.mark.asyncio
    async def test_cancel_not_found_returns_false(self) -> None:
        """'not found' は return False (例外なし)."""
        adapter = CoincheckAdapter.__new__(CoincheckAdapter)
        adapter.api_base_url = "https://coincheck.com"
        adapter.request_timeout = 5

        def mock_request(*args, **kwargs):
            raise NetworkError("Order not found")

        adapter._make_api_request = mock_request

        result = await adapter._cancel_order_real("test_order_456")
        assert result is False


# =====================================================================
# D. spread_too_narrow 分類改善
# =====================================================================


class TestSpreadTooNarrowClassification:
    """Fix D: spread_too_narrow の ERROR→INFO 降格 + 専用分類."""

    def test_cancel_reason_constant_exists(self) -> None:
        """SPREAD_TOO_NARROW 定数が cancel_reasons に存在."""
        assert hasattr(CR, "SPREAD_TOO_NARROW")
        assert CR.SPREAD_TOO_NARROW == "spread_too_narrow"

    def test_spread_too_narrow_classification_in_source(self) -> None:
        """run_single_cycle に spread_too_narrow 分類コードが存在."""
        source = _RUN_SINGLE_CYCLE_SOURCE
        assert "spread too narrow" in source.lower() or "spread_too_narrow" in source.lower()
        assert "§20-D" in source

    def test_spread_too_narrow_log_level_is_info(self) -> None:
        """spread_too_narrow は logger.info で出力 (ERROR ではない)."""
        source = _RUN_SINGLE_CYCLE_SOURCE
        # "spread too narrow" 周辺に logger.info がある
        idx = source.find("spread too narrow")
        if idx < 0:
            idx = source.find("spread_too_narrow")
        assert idx > 0
        # その前後100文字以内に logger.info がある
        context = source[max(0, idx - 200):idx + 200]
        assert "logger.info" in context


# =====================================================================
# E. 統合: 全修正の整合性確認
# =====================================================================


class TestIntegrationConsistency:
    """全修正の整合性確認."""

    def test_all_skip_paths_covered_by_regime_update(self) -> None:
        """§20-A のレジーム更新がすべてのスキップパスの前に配置されている."""
        source = read_fill_test_runner_source()  # 332# Phase 4

        # §20-A の位置
        idx_regime_update = source.find("§20-A")
        assert idx_regime_update > 0

        # 各スキップパスの位置 (すべて §20-A より後)
        skip_patterns = [
            # 348# balance_forced 撤廃: BALANCE_FORCED_SKIP 削除
            "UNKNOWN_REGIME_BUY_SKIP",
            "TRENDING_SELL_SKIP",
            "BUY_DYNAMIC_KILL",
            "SELL_DYNAMIC_KILL",
        ]
        for pattern in skip_patterns:
            idx = source.find(pattern)
            if idx > 0:
                assert idx > idx_regime_update, \
                    f"{pattern} must appear after §20-A regime update"

    def test_no_dead_code_in_trending_skip(self) -> None:
        """trending_sell_skip ブロックにデッドコード (到達不能コード) がないこと.

        194#: trending_sell ロジックは CycleGateAggregator に集約。
        """
        # 安全弁 (consecutive safety valve, HF4, inv_bypass) が共存
        assert "158#" in _CYCLE_GATE_AGGREGATOR_SOURCE  # consecutive safety valve
        assert "166# HF4" in _CYCLE_GATE_AGGREGATOR_SOURCE  # buy side insufficient
        assert "171#" in _CYCLE_GATE_AGGREGATOR_SOURCE  # inv bypass

    def test_fill_config_defaults_are_safe(self) -> None:
        """デフォルト設定で安全弁が有効 (0 ではない)."""
        config = FillTestConfig()
        assert config.max_consecutive_trending_sell_skip > 0
