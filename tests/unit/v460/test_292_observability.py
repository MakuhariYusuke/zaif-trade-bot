"""
292# P0/P1: ev_weighted 可観測性強化 & reprice deadband & forced_buy_delay 強化テスト.

- FillRecord 新フィールド (ev_score_pretrade, ev_offset_mult_applied, decision_path)
- Config: stale_reprice_min_delta_jpy
- Config: forced_buy_delay_velocity_threshold_ranging_bps
- YAML パース
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.run_fill_test import FillTestConfig
from ztb.metrics.fill_quality import FillRecord, build_fill_record


# =====================================================================
# A. FillRecord 新フィールド — 292# P0
# =====================================================================

class TestFillRecordObservabilityFields:
    """292# P0: ev_weighted 可観測性フィールドの検証."""

    def test_ev_score_pretrade_default_none(self) -> None:
        r = FillRecord(
            cycle_id="t", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.01,
        )
        assert r.ev_score_pretrade is None

    def test_ev_offset_mult_applied_default_none(self) -> None:
        r = FillRecord(
            cycle_id="t", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.01,
        )
        assert r.ev_offset_mult_applied is None

    def test_decision_path_default_none(self) -> None:
        r = FillRecord(
            cycle_id="t", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.01,
        )
        assert r.decision_path is None

    def test_round_trip_with_new_fields(self) -> None:
        r = FillRecord(
            cycle_id="obs_1", timestamp=1.0, side="sell",
            order_price=15000000.0, order_quantity=0.001,
            ev_score_pretrade=2.5,
            ev_offset_mult_applied=0.95,
            decision_path="ev_offset",
        )
        d = r.to_dict()
        assert d["ev_score_pretrade"] == pytest.approx(2.5)
        assert d["ev_offset_mult_applied"] == pytest.approx(0.95)
        assert d["decision_path"] == "ev_offset"

        r2 = FillRecord.from_dict(d)
        assert r2.ev_score_pretrade == pytest.approx(2.5)
        assert r2.ev_offset_mult_applied == pytest.approx(0.95)
        assert r2.decision_path == "ev_offset"

    def test_build_fill_record_accepts_new_fields(self) -> None:
        r = build_fill_record(
            cycle_id="br_1", timestamp=2.0, side="buy",
            order_price=100.0, order_quantity=0.01,
            ev_score_pretrade=-1.2,
            ev_offset_mult_applied=1.05,
            decision_path="ev_emergency_skip",
        )
        assert r.ev_score_pretrade == pytest.approx(-1.2)
        assert r.ev_offset_mult_applied == pytest.approx(1.05)
        assert r.decision_path == "ev_emergency_skip"

    def test_decision_path_values(self) -> None:
        """有効な decision_path 値が保持される."""
        for path in (
            "primary_only", "ev_offset", "ev_emergency_skip",
            "ev_no_change", "ev_normal_skip",
        ):
            r = FillRecord(
                cycle_id="dp", timestamp=0.0, side="buy",
                order_price=100.0, order_quantity=0.01,
                decision_path=path,
            )
            assert r.decision_path == path


# =====================================================================
# B. Config — 292# P1: Reprice deadband
# =====================================================================

class TestRepriceDeadbandConfig:
    """292# P1: stale_reprice_min_delta_jpy 設定フィールド."""

    def test_default_zero(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_reprice_min_delta_jpy == pytest.approx(0.0)

    def test_explicit_value(self) -> None:
        cfg = FillTestConfig(stale_reprice_min_delta_jpy=500.0)
        assert cfg.stale_reprice_min_delta_jpy == pytest.approx(500.0)


class TestRepriceDeadbandYAML:
    """292# P1: YAML パースで reprice_min_delta_jpy を読み込む."""

    def test_from_yaml_reprice_deadband(self) -> None:
        yaml_data = {
            "stale_order": {
                "enabled": True,
                "reprice_min_delta_jpy": 300.0,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.stale_reprice_min_delta_jpy == pytest.approx(300.0)

    def test_from_yaml_reprice_deadband_absent(self) -> None:
        """reprice_min_delta_jpy 省略時はデフォルト 0.0."""
        yaml_data = {
            "stale_order": {
                "enabled": True,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.stale_reprice_min_delta_jpy == pytest.approx(0.0)

    def test_production_yaml_has_reprice_deadband(self) -> None:
        """本番 YAML に reprice_min_delta_jpy が設定されている."""
        yaml_path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        import yaml as _yaml  # type: ignore[import-untyped]

        with open(yaml_path) as f:
            y = _yaml.safe_load(f)
        so = y.get("stale_order", {})
        assert "reprice_min_delta_jpy" in so
        assert so["reprice_min_delta_jpy"] > 0


# =====================================================================
# C. Config — 292# P1: forced_buy_delay regime-aware
# =====================================================================

class TestForcedBuyDelayRegimeConfig:
    """292# P1: forced_buy_delay_velocity_threshold_ranging_bps 設定."""

    def test_default_none(self) -> None:
        cfg = FillTestConfig()
        assert cfg.forced_buy_delay_velocity_threshold_ranging_bps is None

    def test_explicit_value(self) -> None:
        cfg = FillTestConfig(
            forced_buy_delay_velocity_threshold_ranging_bps=-3.0,
        )
        assert cfg.forced_buy_delay_velocity_threshold_ranging_bps == pytest.approx(-3.0)


class TestForcedBuyDelayRegimeYAML:
    """292# P1: YAML パースで velocity_threshold_ranging_bps を読み込む."""

    def test_from_yaml_ranging_threshold(self) -> None:
        yaml_data = {
            "止血": {
                "forced_buy_delay": {
                    "enabled": True,
                    "velocity_threshold_bps": -5.0,
                    "velocity_threshold_ranging_bps": -3.0,
                    "cycles": 3,
                }
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.forced_buy_delay_velocity_threshold_ranging_bps == pytest.approx(-3.0)

    def test_from_yaml_ranging_threshold_absent(self) -> None:
        yaml_data = {
            "止血": {
                "forced_buy_delay": {
                    "enabled": True,
                }
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.forced_buy_delay_velocity_threshold_ranging_bps is None

    def test_production_yaml_has_ranging_threshold(self) -> None:
        """本番 YAML に velocity_threshold_ranging_bps が設定されている."""
        yaml_path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        import yaml as _yaml  # type: ignore[import-untyped]

        with open(yaml_path) as f:
            y = _yaml.safe_load(f)
        fbd = y.get("loss_control", {}).get("forced_buy_delay", {})
        assert "velocity_threshold_ranging_bps" in fbd
        assert fbd["velocity_threshold_ranging_bps"] == pytest.approx(-3.0)


# =====================================================================
# D. Hot-Reload — 292# v3: 新フィールドがリロード対象
# =====================================================================

class TestHotReloadFieldCoverage:
    """292# v3: 新設 config フィールドが Hot-Reload 対象に含まれる."""

    @pytest.fixture(autouse=True)
    def _load_hot_reload_fields(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        self.fields = _HOT_RELOADABLE_FIELDS

    def test_stale_reprice_min_delta_jpy_in_hot_reload(self) -> None:
        assert "stale_reprice_min_delta_jpy" in self.fields

    def test_forced_buy_delay_fields_in_hot_reload(self) -> None:
        expected = {
            "forced_buy_delay_enabled",
            "forced_buy_delay_velocity_threshold_bps",
            "forced_buy_delay_cycles",
            "forced_buy_delay_velocity_threshold_ranging_bps",
            "forced_buy_delay_max_consecutive",
        }
        assert expected.issubset(self.fields)


# =====================================================================
# E. Config — 294# P0: forced_buy_delay_max_consecutive
# =====================================================================

class TestForcedBuyDelayMaxConsecutive:
    """294# P0: forced_buy_delay_max_consecutive によるデッドロック防止."""

    def test_default_value(self) -> None:
        cfg = FillTestConfig()
        assert cfg.forced_buy_delay_max_consecutive == 10

    def test_explicit_value(self) -> None:
        cfg = FillTestConfig(forced_buy_delay_max_consecutive=5)
        assert cfg.forced_buy_delay_max_consecutive == 5

    def test_from_yaml(self) -> None:
        yaml_data = {
            "止血": {
                "forced_buy_delay": {
                    "enabled": True,
                    "max_consecutive": 8,
                }
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.forced_buy_delay_max_consecutive == 8

    def test_production_yaml_has_max_consecutive(self) -> None:
        yaml_path = _PROJECT_ROOT / "configs" / "v460" / "fill_test.yaml"
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        import yaml as _yaml  # type: ignore[import-untyped]

        with open(yaml_path) as f:
            y = _yaml.safe_load(f)
        fbd = y.get("loss_control", {}).get("forced_buy_delay", {})
        assert "max_consecutive" in fbd
        assert fbd["max_consecutive"] == 10


# =====================================================================
# F. Hot-Reload — 295# 包括的カバレッジ検証
# =====================================================================

class TestHotReloadComprehensiveCoverage:
    """295# hot-reload 包括カバレッジ: 運用パラメータが漏れなく登録されている."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        self.fields = _HOT_RELOADABLE_FIELDS

    # --- 295# 追加カテゴリのサンプル検証 ---

    def test_as_fields(self) -> None:
        """AS/Avellaneda-Stoikov 関連."""
        expected = {
            "as_reservation_enabled", "as_reservation_gamma",
            "as_delta_star_enabled", "as_delta_star_fill_rate_k",
        }
        assert expected.issubset(self.fields)

    def test_amihud_kyle_fields(self) -> None:
        """Amihud / Kyle 市場インパクト."""
        expected = {
            "amihud_illiq_enabled", "amihud_illiq_baseline",
            "kyle_lambda_enabled", "kyle_lambda_impact_mult",
        }
        assert expected.issubset(self.fields)

    def test_inventory_fields(self) -> None:
        """Inventory skewing / balance."""
        expected = {
            "inventory_skewing_enabled", "inventory_skewing_window",
            "balance_forced_cooldown_sec", "balance_margin_ratio",
        }
        assert expected.issubset(self.fields)

    def test_degraded_and_escape_fields(self) -> None:
        """縮退清算 / Inventory Escape."""
        expected = {
            "degraded_liquidation_enabled", "degraded_liquidation_lot_mult",
            "inventory_escape_enabled", "inventory_escape_duty_cycle",
        }
        assert expected.issubset(self.fields)

    def test_dd_cooldown_recovery_fields(self) -> None:
        """DD cooldown / recovery."""
        expected = {
            "dd_cooldown_release_sec", "dd_cooldown_release_lot_scale",
            "per_side_dd_recovery_cycles", "recovery_trending_penalty",
        }
        assert expected.issubset(self.fields)

    def test_mcb_sad_fields(self) -> None:
        """MCB / SAD."""
        expected = {
            "mcb_enabled", "mcb_caution_sigma", "mcb_halt_cooldown_sec",
            "sad_enabled", "sad_wide_ratio", "sad_baseline_window_sec",
        }
        assert expected.issubset(self.fields)

    def test_skip_gate_advanced_fields(self) -> None:
        """SkipGate EV warning / adaptive."""
        expected = {
            "skip_gate_ev_warning_threshold",
            "skip_gate_adaptive_ceiling", "skip_gate_adaptive_floor",
            "skip_gate_regime_thresholds",
        }
        assert expected.issubset(self.fields)

    def test_dynamic_kill_advanced_fields(self) -> None:
        """Dynamic kill 上級パラメータ."""
        expected = {
            "sell_dynamic_kill_max_duration_sec",
            "buy_dynamic_kill_inv_relaxation_enabled",
        }
        assert expected.issubset(self.fields)

    def test_no_duplicate_in_frozenset(self) -> None:
        """frozenset なので重複は論理的に不可能だが、ソース上の重複を検知."""
        import re
        src = (_PROJECT_ROOT / "scripts" / "v460" / "lib" / "config_hot_reload.py").read_text()
        start = src.index("_HOT_RELOADABLE_FIELDS: frozenset[str] = frozenset({")
        end = src.index("})", start) + 2
        raw_fields = re.findall(r'"(\\w+)"', src[start:end])
        from collections import Counter
        dupes = {k: v for k, v in Counter(raw_fields).items() if v > 1}
        assert not dupes, f"Duplicate entries in source: {dupes}"

    def test_minimum_coverage_count(self) -> None:
        """最低 310 フィールドが登録されている (295# で 312 想定)."""
        assert len(self.fields) >= 310, (
            f"Expected >=310 hot-reloadable fields, got {len(self.fields)}"
        )
