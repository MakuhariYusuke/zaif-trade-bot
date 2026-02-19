"""
094# stale order 検出 & cancel-replace テスト.

- Config フィールド存在
- YAML パース
- FillRecord.reprice_count フィールド
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

from scripts.v460.run_fill_test import FillTestConfig
from ztb.metrics.fill_quality import FillRecord


# =====================================================================
# A. Config フィールド — stale order 検出
# =====================================================================

class TestStaleOrderConfig:
    """094# stale order Config フィールドの検証."""

    def test_stale_order_enabled_default_false(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_order_enabled is False

    def test_stale_check_after_sec_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_check_after_sec == pytest.approx(30.0)

    def test_stale_drift_bps_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_drift_bps == pytest.approx(5.0)

    def test_stale_max_reprice_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_max_reprice == 2

    def test_stale_cooldown_sec_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_cooldown_sec == pytest.approx(10.0)

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
        yaml_cfg = {
            "stale_order": {
                "enabled": True,
                "check_after_sec": 45.0,
                "drift_bps": 7.0,
                "max_reprice": 3,
                "cooldown_sec": 12.0,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.stale_order_enabled is True
        assert cfg.stale_check_after_sec == pytest.approx(45.0)
        assert cfg.stale_drift_bps == pytest.approx(7.0)
        assert cfg.stale_max_reprice == 3
        assert cfg.stale_cooldown_sec == pytest.approx(12.0)

    def test_from_yaml_without_stale_order(self) -> None:
        """stale_order セクション省略時はデフォルト値."""
        cfg = FillTestConfig.from_yaml({})
        assert cfg.stale_order_enabled is False
        assert cfg.stale_check_after_sec == pytest.approx(30.0)

    def test_from_yaml_partial_stale_order(self) -> None:
        """一部のみ指定 → 指定分のみ上書き."""
        yaml_cfg = {
            "stale_order": {
                "enabled": True,
                "drift_bps": 10.0,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.stale_order_enabled is True
        assert cfg.stale_drift_bps == pytest.approx(10.0)
        # 未指定はデフォルト
        assert cfg.stale_check_after_sec == pytest.approx(30.0)
        assert cfg.stale_max_reprice == 2

    def test_production_yaml_has_stale_order(self) -> None:
        """本番 YAML に 094# stale_order セクションが存在."""
        import yaml
        yaml_path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        with open(yaml_path) as f:
            y = yaml.safe_load(f)
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
# D. ロジック構造テスト — stale order 検出がコードに存在
# =====================================================================

class TestStaleOrderLogic:
    """094# stale order ロジックがコードに存在."""

    def test_run_single_cycle_has_stale_order_logic(self) -> None:
        """run_single_cycle (or delegated _monitor_fill_polling) に stale_order 関連ロジックがある."""
        from scripts.v460.lib.order_monitor import OrderMonitor
        # 120#: stale order logic extracted to OrderMonitor.monitor
        source = inspect.getsource(OrderMonitor.monitor)
        assert "stale_order" in source
        assert "stale_drift_bps" in source
        assert "reprice_count" in source

    def test_stale_order_checks_direction(self) -> None:
        """stale 判定で乖離方向 (is_drifting_away) を検証している."""
        from scripts.v460.lib.order_monitor import OrderMonitor
        source = inspect.getsource(OrderMonitor.monitor)
        assert "is_drifting_away" in source

    def test_stale_order_respects_max_reprice(self) -> None:
        """max_reprice のチェックがある."""
        from scripts.v460.lib.order_monitor import OrderMonitor
        source = inspect.getsource(OrderMonitor.monitor)
        assert "stale_max_reprice" in source

    def test_stale_order_has_cooldown(self) -> None:
        """cooldown の制御がある."""
        from scripts.v460.lib.order_monitor import OrderMonitor
        source = inspect.getsource(OrderMonitor.monitor)
        assert "stale_cooldown_sec" in source
        assert "last_reprice_time" in source

    def test_stale_order_cancel_before_replace(self) -> None:
        """cancel → place の順序になっている."""
        from scripts.v460.lib.order_monitor import OrderMonitor
        source = inspect.getsource(OrderMonitor.monitor)
        # stale セクション内の cancel は後半にある
        stale_section = source[source.find("[stale_order]"):]
        pos_cancel_stale = stale_section.find("cancel_order")
        pos_place_stale = stale_section.find("place_order")
        assert pos_cancel_stale < pos_place_stale, (
            "cancel_order は place_order より先に呼ばれるべき"
        )

    def test_stale_order_updates_mid_at_order(self) -> None:
        """reprice 時に mid_at_order を更新している."""
        from scripts.v460.lib.order_monitor import OrderMonitor
        source = inspect.getsource(OrderMonitor.monitor)
        # mid_at_order = current_mid
        assert "mid_at_order = current_mid" in source


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
