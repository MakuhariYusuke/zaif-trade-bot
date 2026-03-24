"""
094# stale order 検出 & cancel-replace テスト.

- Config フィールド存在
- YAML パース
- FillRecord.reprice_count フィールド
- FillRecord.reprice_drift_bps フィールド (158# P1-3)
- VG 詳細ログフィールド (158# P2-6)
- 時間帯別 skip_gate 閾値調整 (158# P1-6)
- ロジック構造テスト (コードベース)
- 発動条件テスト
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.fill_config import FillMonitorResult, SkipGateResult
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.ml.skip_gate import SkipGate
from scripts.v460.run_fill_test import FillTestConfig
from tests.unit.v460._fill_test_source import ORDER_MONITOR, read_class_method_source
from tests.unit.v460._yaml_test_helpers import (
    clone_fill_test_config,
    load_fill_test_config_from_mapping,
    parse_yaml_mapping,
)
from ztb.metrics.fill_quality import FillRecord

_FILL_CONFIG_FIELDS = FillTestConfig.__dataclass_fields__
_FILL_MONITOR_RESULT_FIELDS = FillMonitorResult.__dataclass_fields__
_SKIP_GATE_EVALUATE_SIGNATURE = inspect.signature(SkipGate.evaluate)
_ORDER_MONITOR_SOURCE = read_class_method_source(ORDER_MONITOR, "OrderMonitor", "monitor")


# =====================================================================
# A. Config フィールド — stale order 検出
# =====================================================================

class TestStaleOrderConfig:
    """094# stale order Config フィールドの検証."""

    def test_stale_order_enabled_default_false(self) -> None:
        assert _FILL_CONFIG_FIELDS["stale_order_enabled"].default is False

    def test_stale_check_after_sec_default(self) -> None:
        assert _FILL_CONFIG_FIELDS["stale_check_after_sec"].default == pytest.approx(30.0)

    def test_stale_drift_bps_default(self) -> None:
        assert _FILL_CONFIG_FIELDS["stale_drift_bps"].default == pytest.approx(5.0)

    def test_stale_max_reprice_default(self) -> None:
        assert _FILL_CONFIG_FIELDS["stale_max_reprice"].default == 2

    def test_stale_cooldown_sec_default(self) -> None:
        assert _FILL_CONFIG_FIELDS["stale_cooldown_sec"].default == pytest.approx(10.0)

    def test_stale_order_explicit(self) -> None:
        cfg = FillTestConfig(
            stale_order_enabled=True,
            stale_check_after_sec=60.0,
            stale_drift_bps=8.0,
            stale_max_reprice=3,
            stale_cooldown_sec=15.0,
        )
        assert cfg.stale_order_enabled is True
        assert cfg.stale_check_after_sec == pytest.approx(60.0)
        assert cfg.stale_drift_bps == pytest.approx(8.0)
        assert cfg.stale_max_reprice == 3
        assert cfg.stale_cooldown_sec == pytest.approx(15.0)


# =====================================================================
# B. YAML パース — stale_order セクション
# =====================================================================

class TestStaleOrderYAML:
    """094# YAML stale_order セクションのパース検証."""

    def test_from_yaml_with_stale_order(self) -> None:
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "stale_order": {
                        "enabled": True,
                        "check_after_sec": 45.0,
                        "drift_bps": 7.0,
                        "max_reprice": 3,
                        "cooldown_sec": 12.0,
                    }
                }
            )
        )
        assert cfg.stale_order_enabled is True
        assert cfg.stale_check_after_sec == pytest.approx(45.0)
        assert cfg.stale_drift_bps == pytest.approx(7.0)
        assert cfg.stale_max_reprice == 3
        assert cfg.stale_cooldown_sec == pytest.approx(12.0)

    def test_from_yaml_without_stale_order(self) -> None:
        """stale_order セクション省略時はデフォルト値."""
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({}))
        assert cfg.stale_order_enabled is False
        assert cfg.stale_check_after_sec == pytest.approx(30.0)

    def test_from_yaml_partial_stale_order(self) -> None:
        """一部のみ指定 → 指定分のみ上書き."""
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "stale_order": {
                        "enabled": True,
                        "drift_bps": 10.0,
                    }
                }
            )
        )
        assert cfg.stale_order_enabled is True
        assert cfg.stale_drift_bps == pytest.approx(10.0)
        # 未指定はデフォルト
        assert cfg.stale_check_after_sec == pytest.approx(30.0)
        assert cfg.stale_max_reprice == 2

    def test_production_yaml_has_stale_order(
        self,
        v460_fill_test_yaml_base: dict[str, object],
    ) -> None:
        """本番 YAML に 094# stale_order セクションが存在."""
        y = v460_fill_test_yaml_base
        so = y.get("stale_order", {})
        assert so.get("enabled") is True
        assert so.get("drift_bps") == pytest.approx(5.0)
        assert so.get("max_reprice") == 2
        assert so.get("check_after_sec") == pytest.approx(30.0)
        assert so.get("cooldown_sec") == pytest.approx(10.0)


# =====================================================================
# C. FillRecord — reprice_count フィールド
# =====================================================================

class TestFillRecordRepriceCount:
    """094# FillRecord.reprice_count の検証."""

    def test_reprice_count_default_zero(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.reprice_count == 0

    def test_reprice_count_explicit(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            reprice_count=2,
        )
        assert r.reprice_count == 2

    def test_reprice_count_in_dict(self) -> None:
        """to_dict / from_dict でラウンドトリップ."""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            reprice_count=1,
        )
        d = r.to_dict()
        assert d["reprice_count"] == 1
        r2 = FillRecord.from_dict(d)
        assert r2.reprice_count == 1

    def test_reprice_count_absent_in_old_data(self) -> None:
        """古いデータに reprice_count がなくても from_dict が動く."""
        d = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 100.0,
            "order_quantity": 0.001,
        }
        r = FillRecord.from_dict(d)
        assert r.reprice_count == 0


# =====================================================================
# C-2. FillRecord — reprice_drift_bps フィールド (158# P1-3)
# =====================================================================

class TestFillRecordRepriceDriftBps:
    """158# P1-3: FillRecord.reprice_drift_bps の検証."""

    def test_reprice_drift_bps_default_none(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.reprice_drift_bps is None

    def test_reprice_drift_bps_explicit(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            reprice_drift_bps=12.5,
        )
        assert r.reprice_drift_bps == pytest.approx(12.5)

    def test_reprice_drift_bps_roundtrip(self) -> None:
        """to_dict / from_dict でラウンドトリップ."""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            reprice_count=2, reprice_drift_bps=8.3,
        )
        d = r.to_dict()
        assert d["reprice_drift_bps"] == pytest.approx(8.3)
        r2 = FillRecord.from_dict(d)
        assert r2.reprice_drift_bps == pytest.approx(8.3)

    def test_reprice_drift_bps_absent_in_old_data(self) -> None:
        """古いデータに reprice_drift_bps がなくても from_dict が動く."""
        d = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "sell",
            "order_price": 100.0,
            "order_quantity": 0.001,
            "reprice_count": 1,
        }
        r = FillRecord.from_dict(d)
        assert r.reprice_drift_bps is None

    def test_reprice_drift_bps_none_when_no_reprice(self) -> None:
        """reprice_count=0 の場合は drift_bps=None が期待される."""
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            reprice_count=0, reprice_drift_bps=None,
        )
        assert r.reprice_drift_bps is None


# =====================================================================
# D. ロジック構造テスト — stale order 検出がコードに存在
# =====================================================================

class TestStaleOrderLogic:
    """094# stale order ロジックがコードに存在."""

    @staticmethod
    def _monitor_source() -> str:
        return _ORDER_MONITOR_SOURCE

    def test_run_single_cycle_has_stale_order_logic(self) -> None:
        """run_single_cycle (or delegated _monitor_fill_polling) に stale_order 関連ロジックがある."""
        # 120#: stale order logic extracted to OrderMonitor.monitor
        source = self._monitor_source()
        assert "stale_order" in source
        assert "stale_drift_bps" in source
        assert "reprice_count" in source

    def test_stale_order_checks_direction(self) -> None:
        """200# stale 判定で adverse/favorable drift 方向を検証している."""
        source = self._monitor_source()
        assert "is_adverse_drift" in source
        assert "is_favorable_drift" in source
        assert "stale_adverse_drift" in source

    def test_stale_order_respects_max_reprice(self) -> None:
        """max_reprice のチェックがある."""
        source = self._monitor_source()
        assert "stale_max_reprice" in source

    def test_stale_order_has_cooldown(self) -> None:
        """cooldown の制御がある."""
        source = self._monitor_source()
        assert "stale_cooldown_sec" in source
        assert "last_reprice_time" in source

    def test_stale_order_cancel_before_replace(self) -> None:
        """cancel → place の順序になっている."""
        source = self._monitor_source()
        # stale セクション内の cancel は後半にある
        stale_section = source[source.find("[stale_order]"):]
        pos_cancel_stale = stale_section.find("cancel_order")
        pos_place_stale = stale_section.find("place_order")
        assert pos_cancel_stale < pos_place_stale, (
            "cancel_order は place_order より先に呼ばれるべき"
        )

    def test_stale_order_updates_mid_at_order(self) -> None:
        """reprice 時に mid_at_order を更新している."""
        source = self._monitor_source()
        # mid_at_order = current_mid
        assert "mid_at_order = current_mid" in source

    def test_stale_order_tracks_cumulative_drift(self) -> None:
        """158# P1-3: cumulative_drift_bps を追跡している."""
        source = self._monitor_source()
        assert "cumulative_drift_bps" in source
        assert "cumulative_drift_bps += drift_bps" in source

    def test_fill_monitor_result_has_reprice_drift_bps(self) -> None:
        """158# P1-3: FillMonitorResult に reprice_drift_bps がある."""
        assert "reprice_drift_bps" in _FILL_MONITOR_RESULT_FIELDS
        assert _FILL_MONITOR_RESULT_FIELDS["reprice_drift_bps"].default == 0.0


# =====================================================================
# E. 発動条件テスト — パラメータの整合性検証
# =====================================================================

class TestStaleOrderConditions:
    """094# stale order の発動条件が妥当か."""

    def test_check_after_exceeds_poll_interval(self) -> None:
        """check_after_sec (30s) > poll_interval_sec (5s) で安全."""
        cfg = FillTestConfig(stale_order_enabled=True)
        assert cfg.stale_check_after_sec > cfg.poll_interval_sec

    def test_check_after_less_than_timeout(self) -> None:
        """check_after_sec (30s) < order_timeout_sec (300s) で時間内に発動可能."""
        cfg = FillTestConfig(stale_order_enabled=True)
        assert cfg.stale_check_after_sec < cfg.order_timeout_sec

    def test_max_reprice_bounded(self) -> None:
        """max_reprice (2) は少なすぎず多すぎず."""
        cfg = FillTestConfig(stale_order_enabled=True)
        assert 1 <= cfg.stale_max_reprice <= 5

    def test_drift_bps_reasonable(self) -> None:
        """drift_bps (5.0) はノイズ (as_deadzone=2.5) より十分大きい."""
        cfg = FillTestConfig(stale_order_enabled=True)
        assert cfg.stale_drift_bps > cfg.as_deadzone_bps

    def test_cooldown_exceeds_poll_interval(self) -> None:
        """cooldown (10s) > poll_interval (5s) で最低 2 poll 確保."""
        cfg = FillTestConfig(stale_order_enabled=True)
        assert cfg.stale_cooldown_sec >= cfg.poll_interval_sec * 2

    def test_total_reprice_time_within_timeout(self) -> None:
        """最大 reprice してもタイムアウト内に収まる."""
        cfg = FillTestConfig(stale_order_enabled=True)
        # 最悪ケース: check_after + max_reprice × cooldown
        worst_case = cfg.stale_check_after_sec + cfg.stale_max_reprice * cfg.stale_cooldown_sec
        assert worst_case < cfg.order_timeout_sec


# =====================================================================
# F. buy/sell 方向性テスト
# =====================================================================

class TestStaleOrderDirection:
    """094# stale order の方向性判定."""

    def test_buy_upward_drift_is_stale(self) -> None:
        """buy 注文: mid が上昇 → 注文が取り残される → stale."""
        mid_at_order = 10_000_000.0
        current_mid = 10_005_000.0   # +5000 JPY ≈ 5 bps
        drift_bps = abs(current_mid - mid_at_order) / mid_at_order * 10000
        is_away = current_mid > mid_at_order  # buy: 上昇は離れる方向
        assert drift_bps >= 5.0
        assert is_away

    def test_buy_downward_drift_is_not_stale(self) -> None:
        """buy 注文: mid が下降 → 注文に近づく → stale ではない."""
        mid_at_order = 10_000_000.0
        current_mid = 9_995_000.0
        is_away = current_mid > mid_at_order
        assert not is_away

    def test_sell_downward_drift_is_stale(self) -> None:
        """sell 注文: mid が下降 → 注文が取り残される → stale."""
        mid_at_order = 10_000_000.0
        current_mid = 9_995_000.0
        drift_bps = abs(current_mid - mid_at_order) / mid_at_order * 10000
        is_away = current_mid < mid_at_order  # sell: 下降は離れる方向
        assert drift_bps >= 5.0
        assert is_away

    def test_sell_upward_drift_is_not_stale(self) -> None:
        """sell 注文: mid が上昇 → 注文に近づく → stale ではない."""
        mid_at_order = 10_000_000.0
        current_mid = 10_005_000.0
        is_away = current_mid < mid_at_order
        assert not is_away


# =====================================================================
# G. VG 詳細ログフィールド (158# P2-6)
# =====================================================================

class TestFillRecordVGDetailFields:
    """158# P2-6: VG 詳細ログフィールドの検証."""

    def test_vg_detail_fields_default_none(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.vg_velocity_bps is None
        assert r.vg_vpin is None
        assert r.vg_boost_factor is None

    def test_vg_detail_fields_explicit(self) -> None:
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="sell",
            order_price=100.0, order_quantity=0.001,
            vg_triggered=True,
            vg_velocity_bps=15.3,
            vg_vpin=0.72,
            vg_boost_factor=1.5,
        )
        assert r.vg_velocity_bps == pytest.approx(15.3)
        assert r.vg_vpin == pytest.approx(0.72)
        assert r.vg_boost_factor == pytest.approx(1.5)

    def test_vg_detail_fields_roundtrip(self) -> None:
        """to_dict / from_dict ラウンドトリップ."""
        r = FillRecord(
            cycle_id="vg", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            vg_triggered=True,
            vg_velocity_bps=8.5,
            vg_vpin=0.55,
            vg_boost_factor=1.4,
        )
        d = r.to_dict()
        assert d["vg_velocity_bps"] == pytest.approx(8.5)
        r2 = FillRecord.from_dict(d)
        assert r2.vg_velocity_bps == pytest.approx(8.5)
        assert r2.vg_vpin == pytest.approx(0.55)
        assert r2.vg_boost_factor == pytest.approx(1.4)

    def test_vg_detail_absent_in_old_data(self) -> None:
        """古いデータに VG 詳細がなくても from_dict が動く."""
        d = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 100.0,
            "order_quantity": 0.001,
            "vg_triggered": True,
        }
        r = FillRecord.from_dict(d)
        assert r.vg_velocity_bps is None
        assert r.vg_vpin is None
        assert r.vg_boost_factor is None

    def test_maker_price_vg_properties_exist(self) -> None:
        """MakerPriceCalculator に VG 詳細プロパティが存在する."""
        assert hasattr(MakerPriceCalculator, "last_vg_velocity_bps")
        assert hasattr(MakerPriceCalculator, "last_vg_vpin")
        assert hasattr(MakerPriceCalculator, "last_vg_boost_factor")


# =====================================================================
# F. 158# P1-6: 時間帯別 skip_gate 閾値調整
# =====================================================================

class TestSkipGateHourOffsets:
    """158# P1-6: skip_gate_hour_offsets の Config + FillRecord テスト."""

    def test_hour_offsets_default_empty(self) -> None:
        default_factory = _FILL_CONFIG_FIELDS["skip_gate_hour_offsets"].default_factory
        assert callable(default_factory)
        assert default_factory() == {}

    def test_hour_offsets_explicit(self) -> None:
        cfg = FillTestConfig(skip_gate_hour_offsets={0: 0.05, 14: -0.02})
        assert cfg.skip_gate_hour_offsets[0] == pytest.approx(0.05)
        assert cfg.skip_gate_hour_offsets[14] == pytest.approx(-0.02)

    def test_hour_offsets_yaml_parsing(self) -> None:
        """YAML の skip_gate.hour_offsets が正しくパースされる."""
        yaml_str = """
skip_gate:
  enabled: true
  mode: pnl
  hour_offsets:
    0: 0.05
    1: 0.03
    14: -0.02
    15: -0.03
"""
        data = parse_yaml_mapping(yaml_str)
        cfg = FillTestConfig.from_yaml(data)
        assert cfg.skip_gate_hour_offsets[0] == pytest.approx(0.05)
        assert cfg.skip_gate_hour_offsets[1] == pytest.approx(0.03)
        assert cfg.skip_gate_hour_offsets[14] == pytest.approx(-0.02)
        assert cfg.skip_gate_hour_offsets[15] == pytest.approx(-0.03)

    def test_fill_record_hour_offset_default_none(self) -> None:
        r = FillRecord(
            cycle_id="t",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
        )
        assert r.skip_gate_hour_offset is None

    def test_fill_record_hour_offset_explicit(self) -> None:
        r = FillRecord(
            cycle_id="t",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            skip_gate_hour_offset=0.05,
        )
        assert r.skip_gate_hour_offset == pytest.approx(0.05)

    def test_fill_record_hour_offset_roundtrip(self) -> None:
        """from_dict で hour_offset がラウンドトリップする."""
        r = FillRecord(
            cycle_id="t",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            skip_gate_hour_offset=-0.03,
        )
        d = r.to_dict()
        r2 = FillRecord.from_dict(d)
        assert r2.skip_gate_hour_offset == pytest.approx(-0.03)

    def test_fill_record_hour_offset_old_data_compat(self) -> None:
        """古いデータに hour_offset がなくても from_dict が動く."""
        d = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 100.0,
            "order_quantity": 0.001,
        }
        r = FillRecord.from_dict(d)
        assert r.skip_gate_hour_offset is None


class TestSkipGateThresholdOffset:
    """158# P1-6: SkipGate.evaluate の threshold_offset の動作テスト."""

    def test_skip_gate_evaluate_accepts_threshold_offset(self) -> None:
        """SkipGate.evaluate が threshold_offset パラメータを受け付ける."""
        assert "threshold_offset" in _SKIP_GATE_EVALUATE_SIGNATURE.parameters
        p = _SKIP_GATE_EVALUATE_SIGNATURE.parameters["threshold_offset"]
        assert p.default == 0.0

    def test_skip_gate_result_hour_offset_field(self) -> None:
        """SkipGateResult に hour_offset フィールドが存在."""
        r = SkipGateResult()
        assert r.hour_offset == 0.0

    def test_skip_gate_evaluator_config_hour_offsets(self) -> None:
        """SkipGateEvaluator が config.skip_gate_hour_offsets を参照可能."""
        cfg = FillTestConfig(skip_gate_hour_offsets={0: 0.1, 12: -0.05})
        assert cfg.skip_gate_hour_offsets[0] == pytest.approx(0.1)
        assert cfg.skip_gate_hour_offsets[12] == pytest.approx(-0.05)
        # 未定義時間帯は 0.0 (get default)
        assert cfg.skip_gate_hour_offsets.get(6, 0.0) == 0.0
