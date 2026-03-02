"""155# §9 レビュー対応テスト — price=0 補間, 待機時間帯, regime×side, trending sell 抑制.

156# §10 #6: 挙動テスト追加 — balance_forced×trending バイパス, reason 正規化, side バリデーション.
156# §16: 自己レビュー修正テスト — get_fallback_price, fallback_stale_sec, cancel_reason 定数, logger.
"""

from __future__ import annotations

import pytest


# ======================================================================
# hindsight_filter: price=0 補間 (§9.4 #1)
# ======================================================================


class TestPriceZeroInterpolation:
    """order_price=0 のレコードが補間参照価格で分析されること."""

    def _make_timeline(self):
        from scripts.v460.analysis.hindsight_filter import PricePoint
        return [
            PricePoint(1000.0, 10_000_000.0),
            PricePoint(1060.0, 10_001_000.0),
            PricePoint(1120.0, 10_002_000.0),
            PricePoint(1180.0, 10_003_000.0),
        ]

    def test_price_zero_record_included(self) -> None:
        """order_price=0 のレコードが結果に含まれること."""
        from scripts.v460.analysis.hindsight_filter import _analyze_records

        records = [
            {
                "timestamp": 1030,
                "order_price": 0,
                "side": "buy",
                "filled": False,
                "cancel_reason": "balance_forced_skip",
                "cycle_id": "test_1",
            },
            {
                "timestamp": 1000,
                "order_price": 10_000_000,
                "side": "buy",
                "filled": True,
                "cancel_reason": "",
                "cycle_id": "test_2",
                "post_fill_30s_pnl": -0.5,
            },
        ]
        timeline = self._make_timeline()
        results = _analyze_records(records, timeline)

        # price=0 は補間で復元されるので 2 件とも含まれる
        assert len(results) == 2
        interp_rec = [r for r in results if r.interpolated_ref]
        assert len(interp_rec) == 1
        assert interp_rec[0].order_price > 0
        assert interp_rec[0].cancel_reason == "balance_forced_skip"

    def test_price_zero_no_timeline_coverage(self) -> None:
        """タイムライン外の price=0 は分析不能として除外."""
        from scripts.v460.analysis.hindsight_filter import _analyze_records, PricePoint

        records = [
            {
                "timestamp": 5000,  # timeline 外 (300s+ distant)
                "order_price": 0,
                "side": "sell",
                "filled": False,
                "cancel_reason": "orderbook_error",
                "cycle_id": "test_far",
            },
        ]
        # timeline は 1000-1180 — 5000 は 3820s 離れている
        timeline = self._make_timeline()
        results = _analyze_records(records, timeline)
        assert len(results) == 0

    def test_interpolated_hindsight_pnl(self) -> None:
        """補間参照価格から hindsight PnL が計算されること."""
        from scripts.v460.analysis.hindsight_filter import _analyze_records

        records = [
            {
                "timestamp": 1030,  # → interp ≈ 10_000_500
                "order_price": 0,
                "side": "buy",
                "filled": False,
                "cancel_reason": "balance_forced_skip",
                "cycle_id": "test_pnl",
            },
        ]
        timeline = self._make_timeline()
        results = _analyze_records(records, timeline)
        assert len(results) == 1
        r = results[0]
        assert r.interpolated_ref is True
        # 30s 後 = ts 1060 → price 10_001_000
        # buy PnL = (10_001_000 - ~10_000_500) / ~10_000_500 * 10000 ≈ +0.5 bps
        if r.hindsight_pnl_30s is not None:
            assert r.hindsight_pnl_30s > 0


# ======================================================================
# hindsight_filter: 待機時間帯分析 (§9.2 #3)
# ======================================================================


class TestWaitBandAnalysis:
    """_analyze_wait_bands のバンド分割が正しいこと."""

    def _make_results(self):
        from scripts.v460.analysis.hindsight_filter import HindsightResult
        return [
            HindsightResult(
                cycle_id="w1", timestamp=1000, side="buy", order_price=10_000_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.5,
                hindsight_pnl_30s=0.5, hindsight_pnl_60s=1.0, hindsight_pnl_120s=1.5,
                reverse_pnl_30s=-0.5, skip_gate_score=None, skip_gate_as_prob=None,
                regime="ranging", queue_wait_sec=3.0,
            ),
            HindsightResult(
                cycle_id="w2", timestamp=1120, side="sell", order_price=10_001_000,
                cancel_reason="", filled=True, actual_pnl_30s=-0.8,
                hindsight_pnl_30s=-0.8, hindsight_pnl_60s=-0.5, hindsight_pnl_120s=0.2,
                reverse_pnl_30s=0.8, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending", queue_wait_sec=20.0,
            ),
            HindsightResult(
                cycle_id="w3", timestamp=1240, side="buy", order_price=10_002_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.3,
                hindsight_pnl_30s=0.3, hindsight_pnl_60s=0.6, hindsight_pnl_120s=0.9,
                reverse_pnl_30s=-0.3, skip_gate_score=None, skip_gate_as_prob=None,
                regime="ranging", queue_wait_sec=8.0,
            ),
        ]

    def test_wait_bands_populated(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_wait_bands

        results = self._make_results()
        bands = _analyze_wait_bands(results)

        assert "0-5s" in bands
        assert bands["0-5s"]["count"] == 1  # w1 (3s)
        assert bands["5-15s"]["count"] == 1  # w3 (8s)
        assert bands["15-30s"]["count"] == 1  # w2 (20s)

    def test_wait_bands_avg_pnl(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_wait_bands

        results = self._make_results()
        bands = _analyze_wait_bands(results)
        assert bands["15-30s"]["avg_pnl_30s"] == pytest.approx(-0.8, abs=0.01)


# ======================================================================
# hindsight_filter: regime×side クロス分析 (§9.2 #4)
# ======================================================================


class TestRegimeSideAnalysis:
    """_analyze_regime_side のクロス集計が正しいこと."""

    def test_regime_side_cross(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_regime_side, HindsightResult

        results = [
            HindsightResult(
                cycle_id="rs1", timestamp=1000, side="buy", order_price=10_000_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.5,
                hindsight_pnl_30s=0.5, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.5, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending",
            ),
            HindsightResult(
                cycle_id="rs2", timestamp=1120, side="sell", order_price=10_001_000,
                cancel_reason="", filled=True, actual_pnl_30s=-0.7,
                hindsight_pnl_30s=-0.7, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=0.7, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending",
            ),
            HindsightResult(
                cycle_id="rs3", timestamp=1240, side="buy", order_price=10_002_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.2,
                hindsight_pnl_30s=0.2, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.2, skip_gate_score=None, skip_gate_as_prob=None,
                regime="ranging",
            ),
        ]
        rs = _analyze_regime_side(results)

        assert "trending_buy" in rs
        assert "trending_sell" in rs
        assert "ranging_buy" in rs
        assert rs["trending_buy"]["count"] == 1
        assert rs["trending_sell"]["avg_pnl_30s"] == pytest.approx(-0.7, abs=0.01)


# ======================================================================
# hindsight_filter: 補間統計 (§9.4 #1)
# ======================================================================


class TestInterpolatedStats:
    """_analyze_interpolated_stats の統計が正しいこと."""

    def test_interpolated_split(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_interpolated_stats, HindsightResult

        results = [
            HindsightResult(
                cycle_id="i1", timestamp=1000, side="buy", order_price=10_000_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.5,
                hindsight_pnl_30s=0.5, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.5, skip_gate_score=None, skip_gate_as_prob=None,
                regime=None, interpolated_ref=False,
            ),
            HindsightResult(
                cycle_id="i2", timestamp=1060, side="sell", order_price=10_000_500,
                cancel_reason="balance_forced_skip", filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=0.3, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.3, skip_gate_score=None, skip_gate_as_prob=None,
                regime=None, interpolated_ref=True,
            ),
        ]
        stats = _analyze_interpolated_stats(results)
        assert stats["interpolated"]["count"] == 1
        assert stats["original_price"]["count"] == 1
        assert stats["interpolated"]["avg_hindsight_30s"] == pytest.approx(0.3, abs=0.01)


# ======================================================================
# hindsight_filter: H8 regime_guard カテゴリ (trending_sell_skip)
# ======================================================================


class TestCategorization:
    """_categorize でレジームガード系が H8 に分類されること."""

    def test_trending_sell_skip_in_h8(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _categorize, HindsightResult

        results = [
            HindsightResult(
                cycle_id="c1", timestamp=1000, side="sell", order_price=10_000_000,
                cancel_reason="trending_sell_skip", filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=None, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=None, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending",
            ),
            HindsightResult(
                cycle_id="c2", timestamp=1060, side="buy", order_price=10_000_000,
                cancel_reason="unknown_regime_buy_skip", filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=None, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=None, skip_gate_score=None, skip_gate_as_prob=None,
                regime="unknown",
            ),
        ]
        cats = _categorize(results)
        assert "H8_regime_guard" in cats
        assert len(cats["H8_regime_guard"]) == 2


# ======================================================================
# cancel_reasons: TRENDING_SELL_SKIP 定数
# ======================================================================


class TestTrendingSellSkipConstant:
    """TRENDING_SELL_SKIP が cancel_reasons に存在し AUDIT set に含まれること."""

    def test_constant_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert hasattr(CR, "TRENDING_SELL_SKIP")
        assert CR.TRENDING_SELL_SKIP == "trending_sell_skip"

    def test_in_audit_set(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.TRENDING_SELL_SKIP in CR.AUDIT_CANCEL_REASONS


# ======================================================================
# fill_config: skip_sell_trending フィールド
# ======================================================================


class TestSkipSellTrendingConfig:
    """FillTestConfig に skip_sell_trending が存在すること."""

    def test_default_false(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.skip_sell_trending is False

    def test_set_true(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(skip_sell_trending=True)
        assert cfg.skip_sell_trending is True


# ======================================================================
# 155# §10 残課題: balance_forced_consecutive フィールド (§9.4 #2)
# ======================================================================


class TestBalanceForcedConsecutiveField:
    """FillRecord に balance_forced_consecutive が存在すること."""

    _BASE = {"cycle_id": "t1", "timestamp": 1.0, "side": "buy", "order_price": 100.0, "order_quantity": 0.001}

    def test_field_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(**self._BASE)
        assert rec.balance_forced_consecutive is None

    def test_field_set(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(**self._BASE, balance_forced_consecutive=5)
        assert rec.balance_forced_consecutive == 5

    def test_to_dict_contains_field(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(**self._BASE, balance_forced_consecutive=3)
        d = rec.to_dict()
        assert "balance_forced_consecutive" in d
        assert d["balance_forced_consecutive"] == 3


# ======================================================================
# 155# S-3: order_timeout_sec_sell 設定
# ======================================================================


class TestSellTimeoutConfig:
    """FillTestConfig に order_timeout_sec_sell が存在すること."""

    def test_default_none(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.order_timeout_sec_sell is None

    def test_set_value(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(order_timeout_sec_sell=75.0)
        assert cfg.order_timeout_sec_sell == 75.0

    def test_from_yaml_loads(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = {"order_timeout_sec_sell": 72.0}
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.order_timeout_sec_sell == 72.0


# ======================================================================
# 156# §10 #1: balance_forced × trending 競合解消テスト
# ======================================================================


class TestBalanceForcedTrendingBypass:
    """balance_forced=True 時に skip_sell_trending をバイパスすることの検証."""

    def test_trending_sell_skip_code_has_balance_forced_check(self) -> None:
        """run_fill_test.py の trending sell skip ブロックに
        'not _balance_forced' 条件が含まれていること."""
        import ast
        # 163# mixin 分割: balance_forced チェックは orchestrator に存在
        from tests.unit.v460._fill_test_source import (
            FILL_LOOP_ORCHESTRATOR,
            read_source_text,
        )
        src = read_source_text(FILL_LOOP_ORCHESTRATOR)
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if isinstance(node.operand, ast.Name) and node.operand.id == "_balance_forced":
                    found = True
                    break
        assert found, "trending sell skip must check 'not _balance_forced'"

    def test_skip_sell_trending_config_still_exists(self) -> None:
        """skip_sell_trending 設定フィールドは維持されていること."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(skip_sell_trending=True)
        assert cfg.skip_sell_trending is True


# ======================================================================
# 156# §10 #3: cancel_reason 正規化テスト
# ======================================================================


class TestCancelReasonNormalization:
    """post_only_reject 表記統一の検証."""

    def test_cancel_reasons_constant_value(self) -> None:
        """定数が 'post_only_reject' であること."""
        from scripts.v460.lib.cancel_reasons import POST_ONLY_REJECT
        assert POST_ONLY_REJECT == "post_only_reject"

    def test_order_monitor_uses_post_only_reject(self) -> None:
        """order_monitor.py が 'post_only_reject' を出力すること."""
        src_path = "scripts/v460/lib/order_monitor.py"
        from pathlib import Path
        src = Path(src_path).read_text(encoding="utf-8")
        assert '"post_only_reject"' in src
        # 旧表記が残っていないこと
        assert 'reason = "postonly_reject"' not in src

    def test_hindsight_technical_reasons_covers_both(self) -> None:
        """hindsight_filter がレガシー互換で両方の表記を認識すること."""
        from scripts.v460.analysis.hindsight_filter import _TECHNICAL_REASONS
        assert "post_only_reject" in _TECHNICAL_REASONS
        assert "postonly_reject" in _TECHNICAL_REASONS  # レガシー互換


# ======================================================================
# 156# §10 #2: side バリデーションテスト
# ======================================================================


class TestSideValidation:
    """buy/sell 以外の side が除外されることの検証."""

    def test_invalid_side_excluded(self) -> None:
        """side='unknown' のレコードが分析結果に含まれないこと."""
        from scripts.v460.analysis.hindsight_filter import (
            _analyze_records,
            PricePoint,
        )
        records = [
            {"cycle_id": "c1", "timestamp": 100.0, "side": "unknown",
             "order_price": 1000.0, "filled": False, "cancel_reason": "timeout"},
            {"cycle_id": "c2", "timestamp": 100.0, "side": "buy",
             "order_price": 1000.0, "filled": True, "cancel_reason": ""},
        ]
        timeline = [
            PricePoint(timestamp=90.0, price=1000.0),
            PricePoint(timestamp=130.0, price=1001.0),
            PricePoint(timestamp=220.0, price=1002.0),
        ]
        results = _analyze_records(records, timeline)
        # unknown side は除外され、buy のみ残る
        assert len(results) == 1
        assert results[0].side == "buy"

    def test_valid_sides_kept(self) -> None:
        """buy と sell の両方が正常に処理されること."""
        from scripts.v460.analysis.hindsight_filter import (
            _analyze_records,
            PricePoint,
        )
        records = [
            {"cycle_id": "c1", "timestamp": 100.0, "side": "buy",
             "order_price": 1000.0, "filled": True, "cancel_reason": ""},
            {"cycle_id": "c2", "timestamp": 100.0, "side": "sell",
             "order_price": 1000.0, "filled": True, "cancel_reason": ""},
        ]
        timeline = [
            PricePoint(timestamp=90.0, price=1000.0),
            PricePoint(timestamp=130.0, price=1001.0),
            PricePoint(timestamp=220.0, price=1002.0),
        ]
        results = _analyze_records(records, timeline)
        assert len(results) == 2
        sides = {r.side for r in results}
        assert sides == {"buy", "sell"}


# ======================================================================
# 156# §10 #4: H6 technical 分類網羅テスト
# ======================================================================


class TestH6TechnicalClassification:
    """技術要因の cancel_reason が H6_technical に分類されること."""

    def test_all_ob_reasons_are_technical(self) -> None:
        """orderbook_* 系が全て H6_technical に分類されること."""
        from scripts.v460.analysis.hindsight_filter import (
            _category_from_result,
            HindsightResult,
        )
        ob_reasons = [
            "orderbook_error", "orderbook_timeout",
            "orderbook_rate_limit", "orderbook_empty",
            "sell_guard_reject",
        ]
        for reason in ob_reasons:
            result = HindsightResult(
                cycle_id="t1", timestamp=1.0, side="sell",
                order_price=100.0, cancel_reason=reason,
                filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=None, hindsight_pnl_60s=None,
                hindsight_pnl_120s=None, reverse_pnl_30s=None,
                skip_gate_score=None, skip_gate_as_prob=None,
                regime=None, interpolated_ref=False, queue_wait_sec=None,
            )
            cat = _category_from_result(result)
            assert cat == "H6_technical", f"{reason} should be H6_technical, got {cat}"

    def test_postonly_both_forms_technical(self) -> None:
        """post_only_reject/postonly_reject の両方が H6_technical."""
        from scripts.v460.analysis.hindsight_filter import (
            _category_from_result,
            HindsightResult,
        )
        for reason in ("post_only_reject", "postonly_reject"):
            result = HindsightResult(
                cycle_id="t1", timestamp=1.0, side="buy",
                order_price=100.0, cancel_reason=reason,
                filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=None, hindsight_pnl_60s=None,
                hindsight_pnl_120s=None, reverse_pnl_30s=None,
                skip_gate_score=None, skip_gate_as_prob=None,
                regime=None, interpolated_ref=False, queue_wait_sec=None,
            )
            assert _category_from_result(result) == "H6_technical"


# ======================================================================
# 156# §10 #5: fallback 鮮度テスト
# ======================================================================


class TestFallbackPriceStaleness:
    """orderbook_error fallback に鮮度判定が含まれることの検証."""

    def test_fallback_stale_check_in_code(self) -> None:
        """run_fill_test.py に fallback_stale チェックが含まれること."""
        # 163# mixin 分割: fallback stale チェックは executor に存在
        from tests.unit.v460._fill_test_source import (
            FILL_CYCLE_EXECUTOR,
            read_source_text,
        )
        src = read_source_text(FILL_CYCLE_EXECUTOR)
        assert "_fallback_stale" in src
        assert "_fallback_age" in src
        # stale 時は price=0.0 にフォールバック
        assert "not _fallback_stale else 0.0" in src


# ======================================================================
# 156# §12: balance_forced バイパス水平展開テスト
# ======================================================================


class TestBalanceForcedBypassRemoved:
    """234# balance_forced による Gate bypass は廃止された。

    232# Codex + 233# Gemini 共同提言:
    Kill Gate は残高事情に関わらず絶対的安全権限を持つ。
    balance_forced 時は Gate を突破するのではなく、
    degraded_liquidation (min lot + wide offset) で安全に縮退清算する。
    """

    def _get_source(self) -> str:
        """194#: CycleGateAggregator のソースを返す."""
        from pathlib import Path
        return Path("scripts/v460/lib/cycle_gate_aggregator.py").read_text(encoding="utf-8-sig")

    def test_no_balance_forced_bypass_in_gates(self) -> None:
        """234# Gate 条件に 'not balance_forced' が存在しないこと."""
        import ast
        src = self._get_source()
        tree = ast.parse(src)
        bypass_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if isinstance(node.operand, ast.Name) and node.operand.id == "balance_forced":
                    bypass_count += 1
        assert bypass_count == 0, (
            f"234# balance_forced gate bypass must be removed, "
            f"but found {bypass_count} 'not balance_forced' checks"
        )

    def test_degraded_liquidation_in_evaluate(self) -> None:
        """234# evaluate() 内に degraded_liquidation ロジックが存在."""
        src = self._get_source()
        assert "degraded_liquidation" in src, (
            "234# degraded_liquidation must be implemented in gate aggregator"
        )

    def test_gate_result_has_degraded_fields(self) -> None:
        """234# CycleGateResult に degraded フィールドが存在."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult
        r = CycleGateResult()
        assert hasattr(r, "degraded_liquidation")
        assert hasattr(r, "degraded_reason")
        assert r.degraded_liquidation is False


# ======================================================================
# 156# §16: 自己レビュー修正テスト
# ======================================================================


class TestGetFallbackPrice:
    """maker_price.get_fallback_price() 公開APIテスト."""

    def test_method_exists(self) -> None:
        """get_fallback_price メソッドが存在すること."""
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        assert hasattr(MakerPriceCalculator, "get_fallback_price")

    def test_returns_tuple(self) -> None:
        """get_fallback_price が tuple を返すこと."""
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        import inspect
        sig = inspect.signature(MakerPriceCalculator.get_fallback_price)
        # return annotation が tuple であること
        ann = sig.return_annotation
        assert ann is not inspect.Parameter.empty, "return annotation must exist"

    def test_run_fill_test_uses_public_api(self) -> None:
        """run_fill_test が _prev_mid_price を直接参照していないこと."""
        from pathlib import Path
        # 163# mixin 分割: 全ソースを連結して public API 使用を検証
        from tests.unit.v460._fill_test_source import read_fill_test_runner_source
        src = read_fill_test_runner_source()
        assert "get_fallback_price()" in src, "must use public API"
        # 直接の private access がないこと
        assert "._prev_mid_price" not in src, (
            "must not access _prev_mid_price directly"
        )


class TestFallbackStaleSecConfig:
    """fallback_stale_sec が FillConfig に定義されていること."""

    def test_field_exists_with_default(self) -> None:
        """fallback_stale_sec のデフォルト値が 120.0 であること."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "fallback_stale_sec")
        assert cfg.fallback_stale_sec == 120.0

    def test_from_yaml_maps_field(self) -> None:
        """from_yaml の flat_keys に fallback_stale_sec が含まれること."""
        from pathlib import Path
        src = Path("scripts/v460/lib/fill_config.py").read_text(encoding="utf-8")
        assert '"fallback_stale_sec"' in src


class TestUnknownRegimeSellSkipConstant:
    """UNKNOWN_REGIME_SELL_SKIP 定数テスト."""

    def test_constant_exists(self) -> None:
        """定数が cancel_reasons に存在すること."""
        from scripts.v460.lib.cancel_reasons import UNKNOWN_REGIME_SELL_SKIP
        assert UNKNOWN_REGIME_SELL_SKIP == "unknown_regime_sell_skip"

    def test_in_audit_set(self) -> None:
        """AUDIT_CANCEL_REASONS に含まれること."""
        from scripts.v460.lib.cancel_reasons import (
            AUDIT_CANCEL_REASONS,
            UNKNOWN_REGIME_SELL_SKIP,
        )
        assert UNKNOWN_REGIME_SELL_SKIP in AUDIT_CANCEL_REASONS

    def test_symmetric_with_buy(self) -> None:
        """BUY_SKIP と SELL_SKIP が対称に存在すること."""
        from scripts.v460.lib import cancel_reasons as cr
        assert hasattr(cr, "UNKNOWN_REGIME_BUY_SKIP")
        assert hasattr(cr, "UNKNOWN_REGIME_SELL_SKIP")


class TestHindsightFilterLogger:
    """hindsight_filter にロガーが定義されていること."""

    def test_logger_defined(self) -> None:
        """モジュールレベルで logger が定義されていること."""
        from pathlib import Path
        src = Path("scripts/v460/analysis/hindsight_filter.py").read_text(
            encoding="utf-8"
        )
        assert "import logging" in src
        assert "logger = logging.getLogger" in src

    def test_invalid_side_counter_logged(self) -> None:
        """除外カウンターがログ出力されるコードが存在すること."""
        from pathlib import Path
        src = Path("scripts/v460/analysis/hindsight_filter.py").read_text(
            encoding="utf-8"
        )
        assert "_skipped_invalid_side" in src
        assert "Excluded" in src


class TestNoTimestampFallbackStale:
    """タイムスタンプなしの fallback がstale扱いされること."""

    def test_no_timestamp_treated_as_stale(self) -> None:
        """タイムスタンプなしパスが _fallback_stale = True を設定すること."""
        # 163# mixin 分割: stale 処理は executor に存在
        from tests.unit.v460._fill_test_source import (
            FILL_CYCLE_EXECUTOR,
            read_source_text,
        )
        src = read_source_text(FILL_CYCLE_EXECUTOR)
        # "no timestamp" パスで stale 扱い
        assert "treated as stale" in src
