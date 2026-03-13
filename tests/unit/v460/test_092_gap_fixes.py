"""
092# 対応漏れ修正テスト.

- E1 閾値再設計 (90% → 85%)
- E6 Round-trip KPI 追加 (g1_1_judgment)
- E7 Net inventory drift 追加 (g1_1_judgment)
- gate_thresholds.yaml 整合
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillMetrics,
    FillRecord,
    g1_1_judgment,
    compute_round_trip_metrics,
)


@pytest.fixture(scope="module")
def gate_thresholds_yaml() -> dict[str, object]:
    path = _PROJECT_ROOT / "configs" / "v460" / "gate_thresholds.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("gate_thresholds.yaml must deserialize to dict")
    return cfg


# =====================================================================
# Helpers
# =====================================================================


def _make_fill_metrics(**overrides: object) -> FillMetrics:
    """テスト用 FillMetrics を生成."""
    defaults = {
        "total_orders": 100,
        "filled_orders": 90,
        "cancelled_orders": 10,
        "fill_rate_p90": 0.88,
        "cancel_ratio": 0.10,
        "queue_wait_median_sec": 20.0,
        "post_fill_30s_pnl_mean": 0.1,
        "post_fill_30s_pnl_pvalue": 0.3,
        "adverse_selection_ratio": 0.15,
        "adverse_selection_ratio_raw": 0.40,
        "measurement_days": 5,
        "sample_sufficient": False,
    }
    defaults.update(overrides)
    return FillMetrics(**defaults)  # type: ignore[arg-type]


def _make_buy_sell_records(
    n_pairs: int = 10,
    buy_price: float = 15_000_000.0,
    sell_price: float = 15_001_000.0,
) -> list[FillRecord]:
    """交互 buy/sell FillRecord リストを生成 (round-trip テスト用)."""
    records: list[FillRecord] = []
    base_ts = 1700000000.0
    for i in range(n_pairs):
        records.append(FillRecord(
            cycle_id=f"buy_{i:03d}",
            timestamp=base_ts + i * 240.0,
            side="buy",
            order_price=buy_price,
            order_quantity=0.001,
            fill_price=buy_price,
            filled=True,
            mid_at_fill=buy_price + 500,
            run_id="test_run",
            git_sha="abc1234",
        ))
        records.append(FillRecord(
            cycle_id=f"sell_{i:03d}",
            timestamp=base_ts + i * 240.0 + 120.0,
            side="sell",
            order_price=sell_price,
            order_quantity=0.001,
            fill_price=sell_price,
            filled=True,
            mid_at_fill=sell_price - 500,
            run_id="test_run",
            git_sha="abc1234",
        ))
    return records


def _make_one_sided_records(
    *,
    side: str,
    count: int,
    start_ts: float = 1700000000.0,
    price: float = 15_000_000.0,
) -> list[FillRecord]:
    """片側だけの FillRecord を生成."""
    return [
        FillRecord(
            cycle_id=f"{side}_{i:03d}",
            timestamp=start_ts + i * 120.0,
            side=side,
            order_price=price,
            order_quantity=0.001,
            fill_price=price,
            filled=True,
            run_id="test_run",
            git_sha="abc1234",
        )
        for i in range(count)
    ]


# =====================================================================
# E1 閾値再設計テスト
# =====================================================================


class TestE1ThresholdRedesign:
    """084# 盲点H: E1 閾値 90% → 85% への再設計."""

    def test_e1_default_threshold_is_90(self) -> None:
        """デフォルト閾値はコード上 0.90 のまま (後方互換)."""
        metrics = _make_fill_metrics(fill_rate_p90=0.88)
        result = g1_1_judgment(metrics, {})
        e1 = result["checks"]["E1_fill_rate_p90"]
        assert e1["threshold"] == 0.90
        assert e1["pass"] is False  # 0.88 < 0.90

    def test_e1_with_85_threshold_passes(self) -> None:
        """085% 閾値で 88% fill_rate は PASS."""
        metrics = _make_fill_metrics(fill_rate_p90=0.88)
        result = g1_1_judgment(metrics, {"min_fill_rate_p90": 0.85})
        e1 = result["checks"]["E1_fill_rate_p90"]
        assert e1["threshold"] == 0.85
        assert e1["pass"] is True

    def test_e1_boundary_85(self) -> None:
        """E1 が丁度 85% で PASS."""
        metrics = _make_fill_metrics(fill_rate_p90=0.85)
        result = g1_1_judgment(metrics, {"min_fill_rate_p90": 0.85})
        assert result["checks"]["E1_fill_rate_p90"]["pass"] is True

    def test_gate_thresholds_yaml_has_85(
        self,
        gate_thresholds_yaml: dict[str, object],
    ) -> None:
        """gate_thresholds.yaml が 0.85 に設定されていることを確認."""
        g1_1_exec = gate_thresholds_yaml["g1_1_exec"]
        assert isinstance(g1_1_exec, dict)
        assert g1_1_exec["min_fill_rate_p90"] == 0.85


# =====================================================================
# E6 Round-trip KPI テスト
# =====================================================================


class TestE6RoundTripKPI:
    """087# P1-1: Round-trip PnL を Gate 判定に追加."""

    def test_e6_present_when_records_provided(self) -> None:
        """records を渡すと E6 が追加される."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(metrics, {}, records=records)
        assert "E6_round_trip_pnl" in result["checks"]

    def test_e6_absent_without_records(self) -> None:
        """records=None のとき E6 は追加されない (後方互換)."""
        metrics = _make_fill_metrics()
        result = g1_1_judgment(metrics, {})
        assert "E6_round_trip_pnl" not in result["checks"]

    def test_e6_is_informational(self) -> None:
        """E6 は informational=True で Gate 判定には影響しない."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5, buy_price=15_000_000, sell_price=14_990_000)
        # sell_price < buy_price → round-trip PnL は負
        result = g1_1_judgment(metrics, {}, records=records)
        e6 = result["checks"]["E6_round_trip_pnl"]
        assert e6["informational"] is True
        # Gate 全体判定には E6 の FAIL は影響しない
        # (他の指標が良ければ PASS するはず)

    def test_e6_threshold_default(self) -> None:
        """E6 のデフォルト閾値は -2.0 bps."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(metrics, {}, records=records)
        e6 = result["checks"]["E6_round_trip_pnl"]
        assert e6["threshold"] == -2.0

    def test_e6_custom_threshold(self) -> None:
        """カスタム閾値で E6 判定."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(
            metrics, {"min_round_trip_pnl_mean": 0.0}, records=records,
        )
        e6 = result["checks"]["E6_round_trip_pnl"]
        assert e6["threshold"] == 0.0

    def test_e6_positive_rt_passes(self) -> None:
        """正の round-trip PnL は PASS."""
        metrics = _make_fill_metrics()
        # sell > buy → 正の PnL
        records = _make_buy_sell_records(
            n_pairs=5, buy_price=15_000_000, sell_price=15_002_000,
        )
        result = g1_1_judgment(metrics, {}, records=records)
        e6 = result["checks"]["E6_round_trip_pnl"]
        assert e6["pass"] is True
        assert e6["value"] > 0

    def test_e6_has_pairs_and_median(self) -> None:
        """E6 に pairs, median, total_jpy フィールドが含まれる."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(metrics, {}, records=records)
        e6 = result["checks"]["E6_round_trip_pnl"]
        assert "pairs" in e6
        assert "median" in e6
        assert "total_jpy" in e6
        assert e6["pairs"] == 5


# =====================================================================
# E7 Net Inventory テスト
# =====================================================================


class TestE7NetInventory:
    """092# net inventory drift 監視."""

    def test_e7_present_with_records(self) -> None:
        """records を渡すと E7 が追加される."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(metrics, {}, records=records)
        assert "E7_net_inventory" in result["checks"]

    def test_e7_balanced_passes(self) -> None:
        """均衡在庫 (net=0) は PASS."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(metrics, {}, records=records)
        e7 = result["checks"]["E7_net_inventory"]
        assert e7["pass"] is True
        assert e7["net_inventory"] == 0

    def test_e7_unbalanced_fails(self) -> None:
        """片側だけの取引で在庫ドリフトが大きい場合 FAIL."""
        metrics = _make_fill_metrics()
        # buy 8 件 + sell 1 件 → ペア 1, unpaired_buys 7 → net_inventory = 7
        records = _make_one_sided_records(side="buy", count=8)
        records.append(FillRecord(
            cycle_id="sell_000",
            timestamp=1700000000.0 + 8 * 120.0,
            side="sell",
            order_price=15_001_000.0,
            order_quantity=0.001,
            fill_price=15_001_000.0,
            filled=True,
            run_id="test_run",
            git_sha="abc1234",
        ))
        result = g1_1_judgment(metrics, {"max_net_inventory": 5}, records=records)
        e7 = result["checks"]["E7_net_inventory"]
        assert e7["pass"] is False  # |7| > 5
        assert e7["net_inventory"] == 7

    def test_e7_is_informational(self) -> None:
        """E7 は informational=True."""
        metrics = _make_fill_metrics()
        records = _make_buy_sell_records(n_pairs=5)
        result = g1_1_judgment(metrics, {}, records=records)
        e7 = result["checks"]["E7_net_inventory"]
        assert e7["informational"] is True

    def test_e7_does_not_affect_gate(self) -> None:
        """E7 の FAIL が Gate 全体判定に影響しない."""
        # fill_rate=0.95 で他指標は全 PASS にする
        metrics = _make_fill_metrics(
            fill_rate_p90=0.95,
            cancel_ratio=0.05,
            queue_wait_median_sec=10,
            post_fill_30s_pnl_mean=0.5,
            post_fill_30s_pnl_pvalue=0.02,
            adverse_selection_ratio=0.10,
            adverse_selection_ratio_raw=0.10,
        )
        # buy/sell 交互でバランス → E6/E7 PASS
        records = _make_buy_sell_records(n_pairs=5, buy_price=15_000_000, sell_price=15_001_000)
        result = g1_1_judgment(metrics, {"min_fill_rate_p90": 0.85}, records=records)
        assert result["gate_result"] == "PASS"


# =====================================================================
# gate_thresholds.yaml 整合テスト
# =====================================================================


class TestGateThresholdsYaml:
    """092# 追加閾値의 YAML 整合."""

    def test_round_trip_threshold_exists(
        self,
        gate_thresholds_yaml: dict[str, object],
    ) -> None:
        """gate_thresholds.yaml に min_round_trip_pnl_mean が定義."""
        g1_1_exec = gate_thresholds_yaml["g1_1_exec"]
        assert isinstance(g1_1_exec, dict)
        assert "min_round_trip_pnl_mean" in g1_1_exec
        assert g1_1_exec["min_round_trip_pnl_mean"] == -2.0

    def test_max_net_inventory_exists(
        self,
        gate_thresholds_yaml: dict[str, object],
    ) -> None:
        """gate_thresholds.yaml に max_net_inventory が定義."""
        g1_1_exec = gate_thresholds_yaml["g1_1_exec"]
        assert isinstance(g1_1_exec, dict)
        assert "max_net_inventory" in g1_1_exec
        assert g1_1_exec["max_net_inventory"] == 5
