"""285# テスト: 283#/284# P0 対応 — legacy config 相互制約 + FillRecord pid フィールド.

283# P0-1: FillRecord に pid フィールドを追加し、Split-Brain 検知を可能に。
283# P0-2: per_side_dd_halt_cycles=0 + inventory_escape_enabled=False の組合せ禁止。
522# inventory_escape 完全撤廃: IE 必須バリデーション削除。
598#: inventory_escape は runtime では使われず、legacy read-only field として残置。
"""

from __future__ import annotations

import os

import pytest


class TestConfigMutualConstraint:
    """283# P0-2 / 522# / 598#: legacy field の存在だけを確認."""

    def test_halt_cycles_zero_ie_disabled_no_longer_raises(self):
        """522# 撤廃: halt_cycles=0 + IE無効 でも ValueError は発生しない."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(
            per_side_dd_enabled=True,
            per_side_dd_halt_cycles=0,
            inventory_escape_enabled=False,
        )
        assert cfg.per_side_dd_halt_cycles == 0
        assert cfg.inventory_escape_enabled is False

    def test_halt_cycles_zero_ie_enabled_ok(self):
        """halt_cycles=0 + IE有効 → 許可 (IE がデッドロック脱出口)."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(
            per_side_dd_enabled=True,
            per_side_dd_halt_cycles=0,
            inventory_escape_enabled=True,
        )
        assert cfg.per_side_dd_halt_cycles == 0
        assert cfg.inventory_escape_enabled is True

    def test_halt_cycles_positive_ie_disabled_ok(self):
        """halt_cycles > 0 + IE無効 → 許可 (halt が自然解除)."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(
            per_side_dd_enabled=True,
            per_side_dd_halt_cycles=15,
            inventory_escape_enabled=False,
        )
        assert cfg.per_side_dd_halt_cycles == 15
        assert cfg.inventory_escape_enabled is False

    def test_per_side_dd_disabled_no_constraint(self):
        """per_side_dd_enabled=False ならどの組合せでも OK."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(
            per_side_dd_enabled=False,
            per_side_dd_halt_cycles=0,
            inventory_escape_enabled=False,
        )
        assert cfg.per_side_dd_enabled is False


class TestFillRecordPidField:
    """283# P0-1: FillRecord に pid フィールドが存在すること."""

    def test_pid_field_exists(self):
        """FillRecord が pid フィールドを持つ."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=1000.0,
            order_quantity=0.001,
        )
        assert hasattr(r, "pid")
        assert r.pid is None  # デフォルトは None

    def test_pid_field_set(self):
        """pid を明示的に設定可能."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=1000.0,
            order_quantity=0.001,
            pid=12345,
        )
        assert r.pid == 12345

    def test_pid_in_to_dict(self):
        """to_dict() に pid が含まれる."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=1000.0,
            order_quantity=0.001,
            pid=os.getpid(),
        )
        d = r.to_dict()
        assert "pid" in d
        assert d["pid"] == os.getpid()

    def test_pid_from_dict(self):
        """from_dict() で pid が復元される."""
        from ztb.metrics.fill_quality import FillRecord

        d = {
            "cycle_id": "test",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 1000.0,
            "order_quantity": 0.001,
            "pid": 99999,
        }
        r = FillRecord.from_dict(d)
        assert r.pid == 99999

    def test_pid_none_from_old_record(self):
        """pid なしの古いレコードでも from_dict 可能 (後方互換)."""
        from ztb.metrics.fill_quality import FillRecord

        d = {
            "cycle_id": "test",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 1000.0,
            "order_quantity": 0.001,
        }
        r = FillRecord.from_dict(d)
        assert r.pid is None
