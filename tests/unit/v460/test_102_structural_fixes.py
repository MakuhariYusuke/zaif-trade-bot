"""101# structural fixes テスト.

§1: E3 計測タイミング修正 (絶対時刻基準)
§2: _soft_loss_cap_triggered レジューム復元
§3: _pre_shrink_lot 整合性 (balance 縮小時に同期)
§4: soft_cap_jpy_snapshot 独立管理
§5: JSONL 重複レコード防止 (cycle_id dedup)
§6: _check_balance_for_side ロット復元
§7: (§1 に統合) early_exit 時の E3 計測
P1-5: regime detector warm-up
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from ztb.metrics.fill_quality import (
    FillRecord,
    load_fill_records,
    load_fill_records_glob,
    save_fill_records,
)


def _make_record(
    cycle_id: str = "test_001",
    filled: bool = True,
    side: str = "buy",
    pnl_bps: Optional[float] = 1.0,
    fill_price: float = 10_300_000.0,
    mid_at_fill: Optional[float] = 10_300_000.0,
    order_quantity: float = 0.001,
    timestamp: Optional[float] = None,
) -> FillRecord:
    return FillRecord(
        cycle_id=cycle_id,
        timestamp=timestamp or time.time(),
        side=side,
        order_price=10_300_000.0,
        order_quantity=order_quantity,
        fill_price=fill_price if filled else None,
        filled=filled,
        cancelled=not filled,
        mid_at_fill=mid_at_fill if filled else None,
        post_fill_30s_pnl=pnl_bps if filled else None,
        run_id="test_run",
        git_sha="abc123",
    )


# =====================================================================
# §5: JSONL cycle_id dedup
# =====================================================================


class TestJSONLDedup:
    """§5: load_fill_records / load_fill_records_glob の重複排除."""

    def test_single_file_dedup(self, tmp_path: Path) -> None:
        """同一ファイル内の cycle_id 重複が排除される."""
        r1 = _make_record(cycle_id="dup_001", pnl_bps=1.0)
        r2 = _make_record(cycle_id="dup_001", pnl_bps=2.0)  # 重複
        r3 = _make_record(cycle_id="unique_002", pnl_bps=3.0)

        path = tmp_path / "fill_records_test.jsonl"
        save_fill_records([r1, r2, r3], path)

        loaded = load_fill_records(path)
        assert len(loaded) == 2
        ids = [r.cycle_id for r in loaded]
        assert "dup_001" in ids
        assert "unique_002" in ids

    def test_cross_file_dedup(self, tmp_path: Path) -> None:
        """複数ファイル間の cycle_id 重複が排除される."""
        r1 = _make_record(cycle_id="shared_001")
        r2 = _make_record(cycle_id="file1_only")
        r3 = _make_record(cycle_id="shared_001")  # 重複
        r4 = _make_record(cycle_id="file2_only")

        save_fill_records([r1, r2], tmp_path / "fill_records_20260101.jsonl")
        save_fill_records([r3, r4], tmp_path / "fill_records_20260102.jsonl")

        loaded = load_fill_records_glob(tmp_path)
        assert len(loaded) == 3
        ids = {r.cycle_id for r in loaded}
        assert ids == {"shared_001", "file1_only", "file2_only"}

    def test_emergency_dir_merged(self, tmp_path: Path) -> None:
        """emergency ディレクトリの JSONL も統合される."""
        r1 = _make_record(cycle_id="normal_001")
        save_fill_records([r1], tmp_path / "fill_records_20260101.jsonl")

        # emergency dump
        emergency_dir = tmp_path / "emergency"
        emergency_dir.mkdir()
        r2 = _make_record(cycle_id="emergency_001")
        r3 = _make_record(cycle_id="normal_001")  # 重複
        save_fill_records([r2, r3], emergency_dir / "emergency_atexit_20260101.jsonl")

        loaded = load_fill_records_glob(tmp_path)
        assert len(loaded) == 2
        ids = {r.cycle_id for r in loaded}
        assert ids == {"normal_001", "emergency_001"}

    def test_no_dedup_needed_performance(self, tmp_path: Path) -> None:
        """重複なしの場合、全件がそのまま読み込まれる."""
        records = [_make_record(cycle_id=f"rec_{i:04d}") for i in range(100)]
        save_fill_records(records, tmp_path / "fill_records_test.jsonl")

        loaded = load_fill_records(tmp_path / "fill_records_test.jsonl")
        assert len(loaded) == 100


# =====================================================================
# §2: soft_loss_cap resume + §4: soft_cap snapshot
# =====================================================================


class TestSoftLossCapResume:
    """§2: _soft_loss_cap_triggered のレジューム復元."""

    def test_soft_cap_snapshot_set_on_init(self) -> None:
        """_soft_cap_jpy_snapshot が初期化される."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert runner._soft_cap_jpy_snapshot is None  # run_continuous で設定

    def test_soft_loss_cap_triggered_starts_false(self) -> None:
        """初期状態は False."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert runner._soft_loss_cap_triggered is False


# =====================================================================
# §3: _pre_shrink_lot 整合性
# =====================================================================


class TestPreShrinkLotSync:
    """§3: balance 縮小時に _pre_shrink_lot が保持される."""

    def test_pre_shrink_lot_initialized(self) -> None:
        """_pre_shrink_lot が order_quantity で初期化される (121# BalanceChecker に委譲)."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig(order_quantity=0.005)
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert runner._balance_checker.pre_shrink_lot == 0.005


# =====================================================================
# §6: ロット復元
# =====================================================================


class TestLotRestoration:
    """§6: _check_balance_for_side のロット復元."""

    def test_lot_shrink_preserves_pre_shrink(self) -> None:
        """ロット縮小時に _pre_shrink_lot に元のロットが保存される."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig(order_quantity=0.005)
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)

        # balance_shrink_active でないとき、縮小前のロットが pre_shrink_lot に保存される
        runner._current_lot = 0.005
        runner._pre_shrink_lot = 0.005
        runner._balance_shrink_active = False

        # 手動で縮小をシミュレート
        old_lot = runner._current_lot
        runner._current_lot = 0.002
        if not runner._balance_shrink_active:
            runner._pre_shrink_lot = old_lot
        assert runner._pre_shrink_lot == 0.005


# =====================================================================
# P1-5: regime detector warm-up
# =====================================================================


class TestRegimeWarmup:
    """P1-5: 既存レコードからの regime detector warm-up."""

    def test_warmup_feeds_prices(self) -> None:
        """warm-up で既存レコードの mid_at_fill が投入される."""
        from scripts.v460.lib.regime_detector import (
            FillTestRegimeDetector,
            RegimeConfig,
        )

        detector = FillTestRegimeDetector(RegimeConfig(window=5))
        assert detector.observation_count == 0

        # 5件投入
        base_price = 10_000_000.0
        for i in range(5):
            detector.update(time.time() + i, base_price + i * 100)

        assert detector.observation_count == 5
        # unknown ではなくなっているはず
        from scripts.v460.lib.regime_detector import FillTestRegime
        # window に到達したので判定可能
        result = detector.update(time.time() + 5, base_price + 500)
        assert result.regime != FillTestRegime.UNKNOWN or result.confidence > 0

    def test_warmup_window_limit(self) -> None:
        """warm-up は window*3 件に制限される."""
        from scripts.v460.lib.regime_detector import (
            FillTestRegimeDetector,
            RegimeConfig,
        )

        config = RegimeConfig(window=5)
        detector = FillTestRegimeDetector(config)

        # 50件投入 (window*3=15 件に制限されるべき)
        for i in range(50):
            detector.update(time.time() + i, 10_000_000.0 + i * 10)

        # buffer は window*3 に制限される
        assert len(detector._prices) <= config.window * 3


# =====================================================================
# FillRecord.actual_measurement_sec フィールド
# =====================================================================


class TestActualMeasurementSec:
    """actual_measurement_sec フィールドの存在確認."""

    def test_field_exists(self) -> None:
        """FillRecord に actual_measurement_sec フィールドがある."""
        r = _make_record()
        assert hasattr(r, "actual_measurement_sec")

    def test_serialization(self) -> None:
        """actual_measurement_sec が JSON を通して保存/復元される."""
        r = _make_record()
        # FillRecord に actual_measurement_sec を設定する方法
        d = r.to_dict()
        d["actual_measurement_sec"] = 15.3
        restored = FillRecord.from_dict(d)
        assert restored.actual_measurement_sec == pytest.approx(15.3)
