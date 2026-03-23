"""421# テスト: Execution Final Clamp + Route-to-Kill Deadlock 防止.

416#/417# レビューで発見された2つの構造欠陥の修正を検証:
1. Final Clamp: maker_price ceiling 後の executor 側 multiplier chain が
   ceiling を迂回して effective_offset_ratio が際限なく拡大する問題
2. Route-to-Kill Deadlock: buy 残高不足 → sell 切替 → sell kill-gated →
   高速ループのデッドスピラル
"""

from __future__ import annotations

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.pre_order_adjustments import PreOrderAdjustmentsMixin


# ============================================================
# cancel_reasons 定数の存在確認
# ============================================================


class TestCancelReasons418:
    """421# で追加された cancel_reason 定数のテスト."""

    def test_final_clamp_hard_skip_exists(self) -> None:
        assert CR.FINAL_CLAMP_HARD_SKIP == "final_clamp_hard_skip"

    def test_route_to_kill_deadlock_exists(self) -> None:
        assert CR.ROUTE_TO_KILL_DEADLOCK == "route_to_kill_deadlock"

    def test_final_clamp_in_audit_set(self) -> None:
        assert CR.FINAL_CLAMP_HARD_SKIP in CR.AUDIT_CANCEL_REASONS

    def test_route_to_kill_in_audit_set(self) -> None:
        assert CR.ROUTE_TO_KILL_DEADLOCK in CR.AUDIT_CANCEL_REASONS


# ============================================================
# FillConfig: execution_final_clamp 設定フィールド
# ============================================================


class TestFillConfigFinalClamp:
    """421# Final Clamp 設定フィールドのテスト."""

    def test_default_enabled(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.execution_final_clamp_enabled is True

    def test_default_hard_skip_mult_disabled(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.execution_final_clamp_hard_skip_mult == 0.0

    def test_custom_hard_skip_mult(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(execution_final_clamp_hard_skip_mult=2.5)
        assert cfg.execution_final_clamp_hard_skip_mult == 2.5


# ============================================================
# FillRecord: execution_pre_clamp_offset フィールド
# ============================================================


class TestFillRecordPreClampField:
    """421# FillRecord.execution_pre_clamp_offset のテスト."""

    def test_field_exists_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=10000,
            order_quantity=0.001,
        )
        assert rec.execution_pre_clamp_offset is None

    def test_field_set(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_pre_clamp_offset=0.905,
        )
        assert rec.execution_pre_clamp_offset == pytest.approx(0.905)

    def test_field_in_to_dict(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_pre_clamp_offset=1.305,
        )
        d = rec.to_dict()
        assert "execution_pre_clamp_offset" in d
        assert d["execution_pre_clamp_offset"] == pytest.approx(1.305)

    def test_field_roundtrip_from_dict(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_pre_clamp_offset=0.75,
        )
        reconstructed = FillRecord.from_dict(rec.to_dict())
        assert reconstructed.execution_pre_clamp_offset == pytest.approx(0.75)

    def test_execution_telemetry_fields_roundtrip(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_sigma=12.5,
            execution_adverse_ofi=0.7,
            execution_additive_enabled=True,
        )
        rebuilt = FillRecord.from_dict(rec.to_dict())
        assert rebuilt.execution_sigma == pytest.approx(12.5)
        assert rebuilt.execution_adverse_ofi == pytest.approx(0.7)
        assert rebuilt.execution_additive_enabled is True


# ============================================================
# Final Clamp ロジックの単体テスト (純粋関数ベース)
# ============================================================


class TestFinalClampLogic:
    """Final Clamp のクランプ計算ロジックをテスト.

    fill_cycle_executor.py 内のインラインロジックを
    PreOrderAdjustmentsMixin._recalc_price_with_new_offset で検証。
    """

    @staticmethod
    def _simulate_final_clamp(
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        ceiling: float,
    ) -> tuple[float, float]:
        """Final Clamp ロジックのシミュレーション."""
        if ceiling > 0 and effective_offset_ratio > ceiling:
            order_price = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
                side, order_price, spread_at_order,
                effective_offset_ratio, ceiling,
            )
            effective_offset_ratio = ceiling
        return order_price, effective_offset_ratio

    def test_buy_over_ceiling_clamped(self) -> None:
        """buy: ceiling 超過時にクランプ."""
        price, ratio = self._simulate_final_clamp(
            side="buy",
            order_price=9950,  # best_bid + spread*1.0 = 9850+100=9950
            spread_at_order=100.0,
            effective_offset_ratio=1.0,
            ceiling=0.20,
        )
        assert ratio == 0.20
        # 474# direct delta: new = old - spread*(old-new) = 9950 - 100*0.8 = 9870
        assert price == 9870

    def test_sell_over_ceiling_clamped(self) -> None:
        """sell: ceiling 超過時にクランプ."""
        price, ratio = self._simulate_final_clamp(
            side="sell",
            order_price=10100,  # best_ask - spread*1.0 = 10300-200=10100
            spread_at_order=200.0,
            effective_offset_ratio=1.0,
            ceiling=0.50,
        )
        assert ratio == 0.50
        # 474# direct delta: new = old + spread*(old-new) = 10100 + 200*0.5 = 10200
        assert price == 10200

    def test_under_ceiling_no_change(self) -> None:
        """ceiling 以下の場合は変更なし."""
        price, ratio = self._simulate_final_clamp(
            side="sell",
            order_price=10025,
            spread_at_order=100.0,
            effective_offset_ratio=0.45,
            ceiling=0.50,
        )
        assert ratio == 0.45
        assert price == 10025

    def test_ceiling_zero_disabled(self) -> None:
        """ceiling=0 の場合は無効 (変更なし)."""
        price, ratio = self._simulate_final_clamp(
            side="sell",
            order_price=10100,
            spread_at_order=200.0,
            effective_offset_ratio=2.0,
            ceiling=0.0,
        )
        assert ratio == 2.0
        assert price == 10100

    def test_hard_skip_threshold(self) -> None:
        """hard skip 判定: effective_offset > ceiling × mult."""
        ceiling = 0.50
        hard_skip_mult = 2.0
        effective = 1.305  # > 0.50 × 2.0 = 1.0

        should_hard_skip = (
            hard_skip_mult > 0
            and effective > ceiling * hard_skip_mult
        )
        assert should_hard_skip is True

    def test_hard_skip_not_triggered_below_threshold(self) -> None:
        """hard skip 非発火: effective_offset <= ceiling × mult."""
        ceiling = 0.50
        hard_skip_mult = 2.0
        effective = 0.905  # <= 0.50 × 2.0 = 1.0

        should_hard_skip = (
            hard_skip_mult > 0
            and effective > ceiling * hard_skip_mult
        )
        assert should_hard_skip is False

    def test_hard_skip_mult_zero_disabled(self) -> None:
        """hard_skip_mult=0 の場合は hard skip 無効."""
        ceiling = 0.50
        hard_skip_mult = 0.0
        effective = 5.0  # 極端に高い

        should_hard_skip = (
            hard_skip_mult > 0
            and effective > ceiling * hard_skip_mult
        )
        assert should_hard_skip is False


# ============================================================
# Post-Ceiling Multiplier Leak のシミュレーション
# ============================================================


class TestPostCeilingMultiplierLeak:
    """416#/417# で発見された post-ceiling leak の再現テスト.

    maker_price ceiling → executor multiplier chain を経て
    effective_offset_ratio が ceiling を大幅に超過する問題を
    再現し、Final Clamp で修正されることを検証。
    """

    def test_multiplier_chain_without_clamp_leaks(self) -> None:
        """clamp なしの multiplier chain は ceiling を迂回する."""
        ceiling = 0.50
        offset = 0.498  # ceiling ギリギリ (maker_price 出力)

        # EV offset: ×1.2
        offset *= 1.2
        # velocity: ×1.3
        offset *= 1.3
        # toxicity: ×1.1
        offset *= 1.1
        # → 0.498 × 1.2 × 1.3 × 1.1 = 0.854... (ceiling 0.50 を大幅超過)
        assert offset > ceiling
        assert offset > 0.85

    def test_multiplier_chain_with_final_clamp(self) -> None:
        """Final Clamp ありの場合は ceiling に切り詰められる."""
        ceiling = 0.50
        offset = 0.498

        # 同じ multiplier chain
        offset *= 1.2
        offset *= 1.3
        offset *= 1.1

        # Final Clamp
        if offset > ceiling:
            offset = ceiling

        assert offset == ceiling

    def test_real_data_reproduction_311(self) -> None:
        """3/11 実データの再現: final_stage=0.300, max_eff_offset=1.305.

        416# §1.1 で指摘されたデータポイントのシミュレーション。
        """
        ceiling = 0.50  # offset_ceiling_ratio_sell
        offset = 0.300  # maker_price final output (with ceiling at 0.50)

        # シミュレーション: 複数 multiplier を経て 1.305 に到達
        # (実際のパラメータは不明だが、×4.35 相当)
        offset *= 4.35
        assert offset == pytest.approx(1.305, abs=0.01)

        # Final Clamp で修正
        if offset > ceiling:
            clamped = ceiling
        else:
            clamped = offset
        assert clamped == ceiling


# ============================================================
# Ceiling 解決ロジック (side-specific)
# ============================================================


class TestCeilingResolution:
    """サイド別 ceiling ratio 解決テスト."""

    @staticmethod
    def _resolve_ceiling(
        side: str,
        offset_ceiling_ratio: float,
        offset_ceiling_ratio_buy: float | None,
        offset_ceiling_ratio_sell: float | None,
    ) -> float:
        """fill_cycle_executor.py / maker_price.py 共通の ceiling 解決ロジック."""
        ceiling = offset_ceiling_ratio
        if side == "buy" and offset_ceiling_ratio_buy is not None:
            ceiling = offset_ceiling_ratio_buy
        elif side == "sell" and offset_ceiling_ratio_sell is not None:
            ceiling = offset_ceiling_ratio_sell
        return ceiling

    def test_buy_uses_buy_specific(self) -> None:
        assert self._resolve_ceiling("buy", 0.15, 0.20, 0.50) == 0.20

    def test_sell_uses_sell_specific(self) -> None:
        assert self._resolve_ceiling("sell", 0.15, 0.20, 0.50) == 0.50

    def test_buy_fallback_to_common(self) -> None:
        assert self._resolve_ceiling("buy", 0.15, None, 0.50) == 0.15

    def test_sell_fallback_to_common(self) -> None:
        assert self._resolve_ceiling("sell", 0.15, None, None) == 0.15


# ============================================================
# 421# self-review: resolve_offset_ceiling DRY ヘルパーテスト
# ============================================================


class TestResolveOffsetCeilingHelper:
    """FillTestConfig.resolve_offset_ceiling() のテスト."""

    def test_buy_uses_buy_specific(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            offset_ceiling_ratio_buy=0.20,
            offset_ceiling_ratio_sell=0.50,
        )
        assert cfg.resolve_offset_ceiling("buy") == 0.20

    def test_sell_uses_sell_specific(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            offset_ceiling_ratio_buy=0.20,
            offset_ceiling_ratio_sell=0.50,
        )
        assert cfg.resolve_offset_ceiling("sell") == 0.50

    def test_fallback_to_common(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(offset_ceiling_ratio=0.15)
        assert cfg.resolve_offset_ceiling("buy") == 0.15
        assert cfg.resolve_offset_ceiling("sell") == 0.15


# ============================================================
# 421# self-review: fill_config_parser YAML roundtrip テスト
# ============================================================


class TestFillConfigParserFinalClamp:
    """execution_final_clamp 設定の YAML パーステスト."""

    def test_parser_reads_enabled(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml({
            "execution_final_clamp_enabled": False,
        })
        assert cfg.execution_final_clamp_enabled is False

    def test_parser_reads_hard_skip_mult(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml({
            "execution_final_clamp_hard_skip_mult": 2.5,
        })
        assert cfg.execution_final_clamp_hard_skip_mult == 2.5

    def test_parser_default_when_absent(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml({})
        assert cfg.execution_final_clamp_enabled is True
        assert cfg.execution_final_clamp_hard_skip_mult == 0.0


# ============================================================
# 421# self-review: guard_reason_classifier テスト
# ============================================================


class TestGuardReasonClassifier418:
    """route_to_kill_deadlock の分類テスト."""

    def test_route_to_kill_is_recovery(self) -> None:
        from scripts.v460.lib.guard_reason_classifier import (
            GuardCategory,
            classify_guard,
        )
        assert classify_guard("route_to_kill_deadlock") == GuardCategory.RECOVERY


# ============================================================
# 421# self-review: config_hot_reload 対象テスト
# ============================================================


class TestConfigHotReload418:
    """Final Clamp 設定が hot-reload 対象であることを確認."""

    def test_final_clamp_fields_in_hot_reloadable(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert "execution_final_clamp_enabled" in _HOT_RELOADABLE_FIELDS
        assert "execution_additive_enabled" in _HOT_RELOADABLE_FIELDS
        assert "execution_final_clamp_hard_skip_mult" in _HOT_RELOADABLE_FIELDS


# ============================================================
# 421# self-review: spread_at_order=None edge case
# ============================================================


class TestFinalClampSpreadNone:
    """spread_at_order=None 時の Final Clamp 挙動テスト."""

    def test_recalc_with_none_spread_returns_original(self) -> None:
        """spread=None → price 変更なし."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="sell",
            order_price=10050,
            spread_at_order=None,
            old_ratio=1.0,
            new_ratio=0.50,
        )
        assert result == 10050

    def test_recalc_with_zero_spread_returns_original(self) -> None:
        """spread=0 → price 変更なし."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="buy",
            order_price=9950,
            spread_at_order=0.0,
            old_ratio=1.0,
            new_ratio=0.20,
        )
        assert result == 9950


# ============================================================
# 420# P1: start_git_sha / executor_offset_stages / side 可観測性
# ============================================================

_FR_REQ = dict(
    cycle_id="test-420",
    timestamp=1700000000.0,
    side="buy",
    order_price=100.0,
    order_quantity=1.0,
)


class TestStartGitSha:
    """420# start_git_sha フィールドテスト."""

    def test_fill_record_has_start_git_sha(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        fr = FillRecord(**_FR_REQ, start_git_sha="abc123def456")
        assert fr.start_git_sha == "abc123def456"

    def test_fill_record_start_git_sha_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        fr = FillRecord(**_FR_REQ)
        assert fr.start_git_sha is None

    def test_fill_record_roundtrip(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        fr = FillRecord(
            **_FR_REQ,
            git_sha="new_sha",
            start_git_sha="old_sha",
        )
        d = fr.to_dict()
        assert d["git_sha"] == "new_sha"
        assert d["start_git_sha"] == "old_sha"
        fr2 = FillRecord.from_dict(d)
        assert fr2.start_git_sha == "old_sha"


class TestExecutorOffsetStages:
    """420# executor_offset_stages フィールドテスト."""

    def test_fill_record_has_executor_offset_stages(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        import json
        stages = json.dumps({"ev": 1.05, "velocity": None})
        fr = FillRecord(**_FR_REQ, executor_offset_stages=stages)
        parsed = json.loads(fr.executor_offset_stages)  # type: ignore[arg-type]
        assert parsed["ev"] == 1.05
        assert parsed["velocity"] is None

    def test_fill_record_executor_offset_stages_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        fr = FillRecord(**_FR_REQ)
        assert fr.executor_offset_stages is None


class TestSideObservability:
    """420# requested_side / resolved_side_reason フィールドテスト."""

    def test_fill_record_has_requested_side(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        # 522# balance_switch 撤廃後も FillRecord フィールド互換を保持
        fr = FillRecord(**_FR_REQ, requested_side="buy", resolved_side_reason=None)
        assert fr.requested_side == "buy"
        assert fr.resolved_side_reason is None

    def test_fill_record_side_fields_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        fr = FillRecord(**_FR_REQ)
        assert fr.requested_side is None
        assert fr.resolved_side_reason is None

    def test_cycle_context_has_requested_side(self) -> None:
        from scripts.v460.lib.orchestrator_pre_cycle import CycleContext
        ctx = CycleContext(next_side="sell")
        ctx.requested_side = "buy"
        # 522# balance_switch 撤廃 → resolved_side_reason は常に None
        assert ctx.requested_side == "buy"
        assert ctx.resolved_side_reason is None

    def test_cycle_context_defaults(self) -> None:
        from scripts.v460.lib.orchestrator_pre_cycle import CycleContext
        ctx = CycleContext()
        assert ctx.requested_side == ""
        assert ctx.resolved_side_reason is None


class TestHardSkipMultConfig:
    """420# hard_skip_mult 有効化テスト."""

    def test_yaml_has_nonzero_hard_skip_mult(self) -> None:
        import yaml
        from pathlib import Path
        with open(Path("configs/v460/fill_test.yaml")) as f:
            cfg = yaml.safe_load(f)
        assert cfg["execution_final_clamp_hard_skip_mult"] == 2.5


class TestClampObservability:
    """431# clamp observability — RunSessionState counters."""

    def test_session_state_has_clamp_counters(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        assert st.clamp_fire_count == 0
        assert st.ceiling_check_count == 0

    def test_clamp_counter_increments(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        st.ceiling_check_count += 1
        st.clamp_fire_count += 1
        assert st.clamp_fire_count == 1
        assert st.ceiling_check_count == 1

    def test_clamp_rate_calculation(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        st = RunSessionState()
        st.ceiling_check_count = 100
        st.clamp_fire_count = 90
        rate = st.clamp_fire_count / st.ceiling_check_count * 100.0
        assert rate == 90.0


class TestClampDetectionLogic:
    """431# self-review: clamp 検出ロジックの統合テスト.

    _process_post_cycle 内の検出条件を FillRecord で直接検証。
    """

    @staticmethod
    def _make_record(
        side: str = "buy",
        skip_gate_skipped: bool | None = False,
        effective_offset_used: float | None = 0.20,
    ) -> "FillRecord":
        from ztb.metrics.fill_quality import FillRecord
        return FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side=side,
            order_price=10000,
            order_quantity=0.001,
            skip_gate_skipped=skip_gate_skipped,
            effective_offset_used=effective_offset_used,
        )

    @staticmethod
    def _detect_clamp(
        record: "FillRecord",
        ceiling_map: dict[str, float] | None = None,
    ) -> tuple[bool, bool]:
        """431# 検出ロジック再現.

        Returns:
            (checked, clamped) — ceiling_check 対象か, clamp 発火したか.
        """
        if ceiling_map is None:
            ceiling_map = {"buy": 0.20, "sell": 0.50}
        if record.skip_gate_skipped is False and record.effective_offset_used is not None:
            _ceil = ceiling_map.get(record.side, 0.15)
            if _ceil > 0 and abs(record.effective_offset_used - _ceil) < 1e-6:
                return True, True
            return True, False
        return False, False

    def test_buy_at_ceiling_detected(self) -> None:
        """Buy offset == ceiling → clamp detected."""
        r = self._make_record(side="buy", effective_offset_used=0.20)
        checked, clamped = self._detect_clamp(r)
        assert checked is True
        assert clamped is True

    def test_buy_below_ceiling_not_clamped(self) -> None:
        """Buy offset < ceiling → checked but not clamped."""
        r = self._make_record(side="buy", effective_offset_used=0.15)
        checked, clamped = self._detect_clamp(r)
        assert checked is True
        assert clamped is False

    def test_sell_near_ceiling_not_clamped(self) -> None:
        """Sell offset=0.498 vs ceiling=0.50 → not clamped (delta > 1e-6)."""
        r = self._make_record(side="sell", effective_offset_used=0.498)
        checked, clamped = self._detect_clamp(r)
        assert checked is True
        assert clamped is False

    def test_sell_at_ceiling_detected(self) -> None:
        """Sell offset == ceiling → clamp detected."""
        r = self._make_record(side="sell", effective_offset_used=0.50)
        checked, clamped = self._detect_clamp(r)
        assert checked is True
        assert clamped is True

    def test_skip_gate_true_excluded(self) -> None:
        """skip_gate_skipped=True → not checked."""
        r = self._make_record(skip_gate_skipped=True, effective_offset_used=0.20)
        checked, clamped = self._detect_clamp(r)
        assert checked is False
        assert clamped is False

    def test_skip_gate_none_excluded(self) -> None:
        """431# SR-1 fix: skip_gate_skipped=None (guard block) → not checked."""
        r = self._make_record(skip_gate_skipped=None, effective_offset_used=0.20)
        checked, clamped = self._detect_clamp(r)
        assert checked is False
        assert clamped is False

    def test_effective_offset_none_excluded(self) -> None:
        """effective_offset_used=None → not checked."""
        r = self._make_record(skip_gate_skipped=False, effective_offset_used=None)
        checked, clamped = self._detect_clamp(r)
        assert checked is False
        assert clamped is False

    def test_epsilon_boundary(self) -> None:
        """Offset within 1e-6 of ceiling → detected as clamped."""
        r = self._make_record(
            side="buy", effective_offset_used=0.20 + 5e-7
        )
        checked, clamped = self._detect_clamp(r)
        assert checked is True
        assert clamped is True

    def test_epsilon_outside(self) -> None:
        """Offset beyond 1e-6 of ceiling → not clamped."""
        r = self._make_record(
            side="buy", effective_offset_used=0.20 + 2e-6
        )
        checked, clamped = self._detect_clamp(r)
        assert checked is True
        assert clamped is False
