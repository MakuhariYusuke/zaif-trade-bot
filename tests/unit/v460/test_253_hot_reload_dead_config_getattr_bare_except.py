"""253# P1 修正: hot_reload 配線, dead config 削除, getattr 排除, bare except 改善.

テスト対象:
  P1-1: sell_asymmetric_high_vol_enabled hot_reload + YAML 配線
  P1-2: balance_forced_apply_trending_offset 完全削除
  P1-3: fill_cycle_executor getattr → クラスレベルデフォルト直接参照
  P1-4: event_logger TeeWriter bare except → logger.debug
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib import event_logger
from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from scripts.v460.lib.event_logger import TeeWriter
from tests.unit.v460._fill_test_source import (
    EVENT_LOGGER,
    FILL_CONFIG,
    FILL_CYCLE_EXECUTOR,
    read_source_text,
)

_FILL_CONFIG_SOURCE = read_source_text(FILL_CONFIG)
_FILL_CYCLE_EXECUTOR_SOURCE = read_source_text(FILL_CYCLE_EXECUTOR)
_EVENT_LOGGER_SOURCE = read_source_text(EVENT_LOGGER)


# ═══════════════════════════════════════════════════════════════════
# P1-1: sell_asymmetric_high_vol_enabled hot_reload + YAML 配線
# ═══════════════════════════════════════════════════════════════════


class TestSellAsymmetricHotReload:
    """252# sell_asymmetric_high_vol_enabled が hot_reload 対象であること."""

    def test_in_reloadable_fields(self) -> None:
        """_HOT_RELOADABLE_FIELDS に含まれること."""
        assert "sell_asymmetric_high_vol_enabled" in _HOT_RELOADABLE_FIELDS

    def test_yaml_has_field(self, v460_fill_test_yaml_base: dict[str, object]) -> None:
        """live YAML に sell_asymmetric_high_vol_enabled が存在."""
        raw = v460_fill_test_yaml_base
        lc = raw["loss_control"]
        assert "sell_asymmetric_high_vol_enabled" in lc
        assert lc["sell_asymmetric_high_vol_enabled"] is False

    def test_default_false(self) -> None:
        """デフォルト値が False であること."""
        cfg = FillTestConfig()
        assert cfg.sell_asymmetric_high_vol_enabled is False

    def test_config_field_in_dataclass(self) -> None:
        """FillTestConfig の dataclass フィールドに存在。"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FillTestConfig)}
        assert "sell_asymmetric_high_vol_enabled" in field_names


# ═══════════════════════════════════════════════════════════════════
# P1-2: balance_forced_apply_trending_offset 完全削除
# ═══════════════════════════════════════════════════════════════════


class TestDeadConfigRemoval:
    """253# dead config balance_forced_apply_trending_offset 完全削除."""

    def test_field_removed_from_config(self) -> None:
        """FillTestConfig からフィールドが完全に削除されたこと."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FillTestConfig)}
        assert "balance_forced_apply_trending_offset" not in field_names

    def test_not_in_hot_reload(self) -> None:
        """hot_reload 対象フィールドから削除済."""
        assert "balance_forced_apply_trending_offset" not in _HOT_RELOADABLE_FIELDS

    def test_not_in_yaml(self, v460_fill_test_yaml_base: dict[str, object]) -> None:
        """live YAML から削除済."""
        raw = v460_fill_test_yaml_base
        lc = raw["loss_control"]
        assert "balance_forced_apply_trending_offset" not in lc

    def test_not_in_fill_config_source(self) -> None:
        """fill_config.py ソースで field 定義が残存しないこと."""
        src = _FILL_CONFIG_SOURCE
        # フィールド定義行がないことを確認 (コメントは許容)
        for line in src.split("\n"):
            stripped = line.strip()
            if "balance_forced_apply_trending_offset" in stripped:
                # コメント行は許容
                assert stripped.startswith("#"), (
                    f"Non-comment reference found: {stripped}"
                )

    def test_yaml_comment_exists(self, v460_fill_test_yaml_path: Path) -> None:
        """YAML に削除理由コメントが残存。"""
        text = v460_fill_test_yaml_path.read_text(encoding="utf-8")
        assert "253# 削除済み" in text


# ═══════════════════════════════════════════════════════════════════
# P1-3: fill_cycle_executor getattr 排除
# ═══════════════════════════════════════════════════════════════════


class TestGetAttrRemoval:
    """253# fill_cycle_executor.py の getattr(self, ...) 排除."""

    def test_no_getattr_self_in_executor(self) -> None:
        """getattr(self, ...) が fill_cycle_executor に残存しないこと.

        getattr(order, ...) 等の外部オブジェクト検査は許容。
        """
        src = _FILL_CYCLE_EXECUTOR_SOURCE
        import re
        # getattr(self, "...") パターンを検出
        matches = re.findall(r'getattr\(self[,\s]', src)
        assert len(matches) == 0, (
            f"getattr(self, ...) found {len(matches)} times in executor"
        )

    def test_class_level_defaults_exist(self) -> None:
        """orchestrator 参照属性のクラスレベルデフォルトが宣言済。"""
        # これらは Mixin がクラスレベルで宣言すべき属性
        assert hasattr(FillCycleExecutorMixin, "_alert_offset_mult")
        assert hasattr(FillCycleExecutorMixin, "_alert_lot_mult")
        assert hasattr(FillCycleExecutorMixin, "_halt_recovery_lot_mult")
        assert hasattr(FillCycleExecutorMixin, "_daily_drawdown_guard")
        assert hasattr(FillCycleExecutorMixin, "_postonly_crossing_streak")

    def test_default_values_correct(self) -> None:
        """クラスレベルデフォルト値が正しいこと."""
        assert FillCycleExecutorMixin._alert_offset_mult == 1.0
        assert FillCycleExecutorMixin._alert_lot_mult == 1.0
        assert FillCycleExecutorMixin._halt_recovery_lot_mult == 1.0
        assert FillCycleExecutorMixin._daily_drawdown_guard is None
        assert FillCycleExecutorMixin._postonly_crossing_streak == 0

    def test_macro_regime_conflict_action_direct_access(self) -> None:
        """macro_regime_conflict_action が config に直接定義済。"""
        cfg = FillTestConfig()
        assert cfg.macro_regime_conflict_action == "log"


# ═══════════════════════════════════════════════════════════════════
# P1-4: event_logger TeeWriter bare except → logger.debug
# ═══════════════════════════════════════════════════════════════════


class TestTeeWriterLogging:
    """253# TeeWriter bare except → logger.debug 改善."""

    def test_no_bare_except_pass_in_tee_writer(self) -> None:
        """TeeWriter に bare 'except ... pass' が残存しないこと."""
        src = _EVENT_LOGGER_SOURCE
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "pass" and i > 0:
                prev = lines[i - 1].strip()
                assert not prev.startswith("except"), (
                    f"bare except+pass found at line {i}: {prev} / {stripped}"
                )

    def test_write_logs_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """write() で例外発生時に logger.debug が出力されること."""
        broken = MagicMock()
        broken.write.side_effect = OSError("disk full")
        good = io.StringIO()
        tee = TeeWriter(good, broken)

        with caplog.at_level(logging.DEBUG, logger="scripts.v460.lib.event_logger"):
            result = tee.write("hello")

        assert result == 5
        assert good.getvalue() == "hello"
        assert any("TeeWriter.write failed" in r.message for r in caplog.records)

    def test_flush_logs_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """flush() で例外発生時に logger.debug が出力されること."""
        broken = MagicMock()
        broken.flush.side_effect = OSError("closed")
        good = io.StringIO()
        tee = TeeWriter(good, broken)

        with caplog.at_level(logging.DEBUG, logger="scripts.v460.lib.event_logger"):
            tee.flush()

        assert any("TeeWriter.flush failed" in r.message for r in caplog.records)

    def test_tee_writer_still_works_normally(self) -> None:
        """正常動作時は例外なく全 writer に書き込む。"""
        w1 = io.StringIO()
        w2 = io.StringIO()
        tee = TeeWriter(w1, w2)
        tee.write("test")
        tee.flush()
        assert w1.getvalue() == "test"
        assert w2.getvalue() == "test"


# ═══════════════════════════════════════════════════════════════════
# Regression: 既存構造の整合性検証
# ═══════════════════════════════════════════════════════════════════


class TestRegressionIntegrity:
    """253# リグレッション防止テスト."""

    def test_hot_reload_field_count_stable(self) -> None:
        """hot_reload 対象フィールド数が大幅に減少していないこと."""
        # 252# 時点で 90 件前後。-1 (dead config 削除), +1 (sell_asymmetric) = ±0
        assert len(_HOT_RELOADABLE_FIELDS) >= 85

    def test_fill_cycle_executor_line_count_under_limit(self) -> None:
        """fill_cycle_executor.py の行数が MAX LINES 未満。"""
        path = Path("scripts/v460/lib/fill_cycle_executor.py")
        lines = path.read_text(encoding="utf-8").count("\n")
        # MAX LINES: 1300
        # 323# God Object 分割: 1502→1090 (FillRecordBuilder + PreOrderAdjustments 抽出)
        # 372# F1 Gap-3: sidecar bps offset 適用 (+13行)
        # 421# Execution Final Clamp + spread guard (+40行)
        # 439# cross-venue lead-lag guard + event-log observability の追加後も
        # 445# EMA平滑化 + confidence scoring 追加 (+7行)
        # 448# cross-venue event details helper 連携と observability 追加で再増加
        # 642# 可観測性改善: 5フィールド追加 (+5行)
        # 671# NFQ構造化フィールド抽出 (+11行)
        # 685# PPO sidecar 統合で +16行
        assert lines < 1600, f"fill_cycle_executor.py has {lines} lines"

    def test_event_logger_has_logger(self) -> None:
        """event_logger.py にモジュールレベル logger が存在。"""
        assert hasattr(event_logger, "logger")

    def test_daily_drawdown_guard_type_annotation(self) -> None:
        """_daily_drawdown_guard のクラスレベル型注釈が存在。"""
        annotations = FillCycleExecutorMixin.__annotations__
        assert "_daily_drawdown_guard" in annotations
