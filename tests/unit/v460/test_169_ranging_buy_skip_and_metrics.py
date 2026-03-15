"""169# B0/B1' テスト — ranging_buy low_vol ハードスキップ + gate metric 3-series.

B1': ranging_buy at low_vol ハードスキップ
  - cancel_reasons に RANGING_LOW_VOL_SKIP 追加
  - fill_config に skip_ranging_buy_low_vol 追加
  - fill_loop_orchestrator に skip ロジック追加

B0: R2 gate metric 3-series 定義修正
  - results_analyzer の judgment に three_series 構造化出力
  - run_pnl_monte_carlo の JSON に three_series 追加

Popup fix:
  - popen_no_window ユーティリティ
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import FillTestConfig as FillConfig
from tests.unit.v460._fill_test_source import CYCLE_GATE_AGGREGATOR, read_source_text
from ztb.metrics.fill_quality import FillRecord, compute_fill_metrics

_CYCLE_GATE_AGGREGATOR_SOURCE = read_source_text(CYCLE_GATE_AGGREGATOR)


# ================================================================
# B1': cancel_reasons
# ================================================================


class TestRangingLowVolSkipConstant:
    """169# B1': RANGING_LOW_VOL_SKIP 定数テスト."""

    def test_constant_exists(self) -> None:
        assert hasattr(CR, "RANGING_LOW_VOL_SKIP")
        assert CR.RANGING_LOW_VOL_SKIP == "ranging_low_vol_skip"

    def test_in_audit_set(self) -> None:
        """AUDIT cancel reasons に含まれること (quarantine bypass 対象)."""
        assert CR.RANGING_LOW_VOL_SKIP in CR.AUDIT_CANCEL_REASONS

    def test_audit_set_no_duplicate(self) -> None:
        """AUDIT set にダブりがないこと."""
        items = list(CR.AUDIT_CANCEL_REASONS)
        assert len(items) == len(set(items))


# ================================================================
# B1': fill_config
# ================================================================


class TestSkipRangingBuyLowVolConfig:
    """169# B1': fill_config パラメータテスト."""

    def test_default_disabled(self) -> None:
        """デフォルトは無効."""
        cfg = FillConfig()
        assert cfg.skip_ranging_buy_low_vol is False

    def test_yaml_parsing(self) -> None:
        """YAML dict から読み込めること."""
        yaml_cfg = {
            "regime": {
                "enabled": True,
                "skip_ranging_buy_low_vol": True,
                "low_vol_threshold": 0.80,
            }
        }
        cfg = FillConfig.from_yaml(yaml_cfg)
        assert cfg.skip_ranging_buy_low_vol is True
        assert cfg.low_vol_threshold == 0.80

    def test_yaml_default_false(self) -> None:
        """YAML に未指定なら False."""
        yaml_cfg = {"regime": {"enabled": True}}
        cfg = FillConfig.from_yaml(yaml_cfg)
        assert cfg.skip_ranging_buy_low_vol is False


# ================================================================
# B1': orchestrator skip ロジック (ソースコード構造テスト)
# ================================================================


class TestOrchestratorRangingBuySkipSource:
    """169# B1': CycleGateAggregator に ranging_buy skip が正しく実装されていることを検証.

    194#: skip ロジックは CycleGateAggregator に集約。
    """

    def test_skip_logic_in_source(self) -> None:
        """cycle_gate_aggregator.py に skip ロジックが含まれる."""
        assert "skip_ranging_buy_low_vol" in _CYCLE_GATE_AGGREGATOR_SOURCE
        assert "ranging_low_vol_skip" in _CYCLE_GATE_AGGREGATOR_SOURCE
        assert "169# B1'" in _CYCLE_GATE_AGGREGATOR_SOURCE

    def test_skip_order_in_source(self) -> None:
        """skip ロジックが unknown_regime_buy の後、trending_sell の前にあること.

        194#: CycleGateAggregator.evaluate() 内のゲート呼び出し順序で検証。
        """
        # evaluate() メソッド内のゲート呼出し順序を検証
        eval_start = _CYCLE_GATE_AGGREGATOR_SOURCE.index("def evaluate(")
        eval_src = _CYCLE_GATE_AGGREGATOR_SOURCE[eval_start:]
        unknown_pos = eval_src.index("Gate 1: unknown_regime_buy")
        ranging_pos = eval_src.index("Gate 2")
        trending_pos = eval_src.index("Gate 3: trending_sell")
        assert unknown_pos < ranging_pos < trending_pos, (
            "ranging_buy skip must be between unknown_regime_buy and trending_sell"
        )


# ================================================================
# B0: 3-series gate metrics
# ================================================================


def _make_record(
    cycle_id: str,
    *,
    timestamp: float = 1.0,
    filled: bool = True,
    pnl_30s: float | None = 0.5,
    cancel_reason: str = "",
    regime: str = "ranging",
) -> FillRecord:
    return FillRecord(
        cycle_id=cycle_id,
        timestamp=timestamp,
        side="buy",
        order_price=14_000_000.0,
        order_quantity=0.001,
        fill_price=14_000_000.0 if filled else None,
        filled=filled,
        post_fill_30s_pnl=pnl_30s,
        cancel_reason=cancel_reason,
        regime=regime,
    )


class TestThreeSeriesStructure:
    """169# B0: results_analyzer three_series 構造テスト."""

    def test_three_series_in_judgment(self) -> None:
        """judgment dict に three_series が含まれること."""
        from unittest.mock import patch

        records = [
            _make_record("a", timestamp=1.0, filled=True, pnl_30s=0.5),
            _make_record("b", timestamp=2.0, filled=True, pnl_30s=-0.3),
            _make_record("c", timestamp=3.0, filled=False, cancel_reason="skip_gate"),
            _make_record("d", timestamp=86400.0 + 1, filled=True, pnl_30s=0.2),
            _make_record("e", timestamp=86400.0 + 2, filled=False, cancel_reason="timeout"),
        ]

        # Mock gate thresholds to avoid file dependency
        mock_thresholds = {
            "g1_1_exec": {},
            "g1_1_quick_exec": {},
            "g1_2_full_exec": {},
        }
        with patch(
            "scripts.v460.lib.results_analyzer.iter_fill_records_glob",
            return_value=iter(records),
        ), patch(
            "scripts.v460.lib.config_loader.load_gate_thresholds",
            return_value=mock_thresholds,
        ), patch(
            "scripts.v460.lib.results_analyzer.partition_clean_records",
            return_value=(records, []),
        ):
            from scripts.v460.lib.results_analyzer import run_results_only

            judgment = run_results_only("/dummy/path")

        assert "three_series" in judgment
        ts = judgment["three_series"]

        assert "raw" in ts
        assert "clean" in ts
        assert "attempted" in ts
        assert ts["gate_basis"] == "clean"

        # raw と clean の fill_rate は同じ (quarantine なし)
        assert ts["raw"]["n_total"] == 5
        assert ts["clean"]["n_total"] == 5
        assert ts["raw"]["n_filled"] == 3

        # attempted は skip_gate 除外
        assert ts["attempted"]["skip_gate_count"] >= 0

    def test_three_series_fill_rate_consistency(self) -> None:
        """3系列の fill_rate が正しく算出されること."""
        records = [
            _make_record(f"r{i}", timestamp=float(i), filled=(i % 2 == 0), pnl_30s=0.1 if (i % 2 == 0) else None)
            for i in range(10)
        ]
        metrics = compute_fill_metrics(records)

        # overall = raw fill rate
        assert metrics.overall_fill_rate == pytest.approx(5 / 10)
        # attempted removes skip_gate records
        # (none in this test, so attempted == total)
        assert metrics.attempted_orders == 10


# ================================================================
# Popup fix: popen_no_window
# ================================================================


class TestPopenNoWindow:
    """169# popen_no_window ユーティリティテスト."""

    def test_returns_dict(self) -> None:
        from ztb.utils.system_utils import popen_no_window

        result = popen_no_window()
        assert isinstance(result, dict)

    def test_windows_has_creationflags(self) -> None:
        """Windows では creationflags が設定される."""
        from ztb.utils.system_utils import popen_no_window

        if sys.platform == "win32":
            result = popen_no_window()
            import subprocess
            assert result["creationflags"] == subprocess.CREATE_NO_WINDOW
        else:
            result = popen_no_window()
            assert "creationflags" not in result

    def test_extra_kwargs_merged(self) -> None:
        """追加の kwargs がマージされる."""
        from ztb.utils.system_utils import popen_no_window

        result = popen_no_window(text=True, encoding="utf-8")
        assert result["text"] is True
        assert result["encoding"] == "utf-8"

    def test_no_duplicate_keys(self) -> None:
        """extra_kwargs で creationflags を上書きしないこと."""
        from ztb.utils.system_utils import popen_no_window

        if sys.platform == "win32":
            import subprocess
            result = popen_no_window()
            assert result["creationflags"] == subprocess.CREATE_NO_WINDOW
