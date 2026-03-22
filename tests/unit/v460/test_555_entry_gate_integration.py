"""555# CalibrationMap entry gate ランタイム統合テスト."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ztb.trading.signal.calibration_map import CalibrationMap


# ════════════════════════════════════════════════════════════
# FillTestConfig entry_gate フィールド
# ════════════════════════════════════════════════════════════

class TestEntryGateConfig:
    """entry_gate_* フィールドが FillTestConfig に存在する."""

    def test_default_values(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.entry_gate_enabled is False
        assert cfg.entry_gate_calibration_map_path == ""
        assert cfg.entry_gate_probability_mode == "lcb"
        assert cfg.entry_gate_ewma_tau == 100.0
        assert cfg.entry_gate_n_min == 30.0
        assert cfg.entry_gate_fee_rate == 0.0
        assert cfg.entry_gate_online_update is True

    def test_yaml_parsing(self) -> None:
        from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml

        yaml_cfg = {
            "entry_gate": {
                "enabled": True,
                "calibration_map_path": "models/v460/entry_gate_calibration.json",
                "probability_mode": "mean",
                "ewma_tau": 50.0,
                "n_min": 20.0,
                "fee_rate": 0.001,
                "c_spread": 0.4,
                "c_vol": 0.1,
                "c_imp": 0.6,
                "online_update": False,
            },
        }
        cfg = parse_fill_config_yaml(yaml_cfg)
        assert cfg.entry_gate_enabled is True
        assert cfg.entry_gate_calibration_map_path == "models/v460/entry_gate_calibration.json"
        assert cfg.entry_gate_probability_mode == "mean"
        assert cfg.entry_gate_ewma_tau == 50.0
        assert cfg.entry_gate_n_min == 20.0
        assert cfg.entry_gate_fee_rate == 0.001
        assert cfg.entry_gate_c_spread == 0.4
        assert cfg.entry_gate_c_vol == 0.1
        assert cfg.entry_gate_c_imp == 0.6
        assert cfg.entry_gate_online_update is False

    def test_yaml_parsing_empty_section(self) -> None:
        """entry_gate セクションが空でもデフォルト値で動作."""
        from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml

        cfg = parse_fill_config_yaml({})
        assert cfg.entry_gate_enabled is False
        assert cfg.entry_gate_calibration_map_path == ""


# ════════════════════════════════════════════════════════════
# CalibrationMap EV 計算ロジック
# ════════════════════════════════════════════════════════════

class TestCalibrationMapEV:
    """EV 計算の正確性テスト."""

    def _make_map_with_data(self) -> CalibrationMap:
        """p_win > 0.5, avg_win > avg_loss の CalibrationMap を作成."""
        cal = CalibrationMap({"ewma_tau": 100.0, "n_min": 5.0})
        # 60 wins (gain 100 JPY each), 40 losses (loss 50 JPY each)
        for i in range(60):
            cal.update("ranging", 0.3, 100.0, i)  # win
        for i in range(40):
            cal.update("ranging", 0.3, -50.0, 60 + i)  # loss
        return cal

    def _make_map_negative_ev(self) -> CalibrationMap:
        """p_win low, avg_win < avg_loss の CalibrationMap (EV < 0)."""
        cal = CalibrationMap({"ewma_tau": 100.0, "n_min": 5.0})
        # 20 wins (gain 30 JPY each), 80 losses (loss 100 JPY each)
        for i in range(20):
            cal.update("trending", -0.3, 30.0, i)  # win
        for i in range(80):
            cal.update("trending", -0.3, -100.0, 20 + i)  # loss
        return cal

    def test_positive_ev(self) -> None:
        cal = self._make_map_with_data()
        stats = cal.get_stats("ranging", 0.3)
        fb = stats["fallback"]
        p_win = fb["p_win_lcb"]
        ev = p_win * fb["avg_win"] - (1 - p_win) * fb["avg_loss"]
        # With 60 wins / 40 losses and good payoff, EV should be positive
        assert ev > 0, f"Expected positive EV, got {ev}"

    def test_negative_ev(self) -> None:
        cal = self._make_map_negative_ev()
        stats = cal.get_stats("trending", -0.3)
        fb = stats["fallback"]
        p_win = fb["p_win_lcb"]
        ev = p_win * fb["avg_win"] - (1 - p_win) * fb["avg_loss"]
        # With 20 wins / 80 losses and bad payoff, EV should be negative
        assert ev < 0, f"Expected negative EV, got {ev}"

    def test_side_to_action_mapping(self) -> None:
        """buy→0.3 (Buy bin), sell→-0.3 (Sell bin) のマッピング確認."""
        cal = CalibrationMap({"ewma_tau": 100.0, "n_min": 5.0})
        assert cal._get_bin(0.3) == "Buy"
        assert cal._get_bin(-0.3) == "Sell"

    def test_probability_mode_lcb(self) -> None:
        cal = self._make_map_with_data()
        stats = cal.get_stats("ranging", 0.3)
        fb = stats["fallback"]
        assert fb["p_win_lcb"] < fb["p_win_mean"] < fb["p_win_ucb"]

    def test_online_update(self) -> None:
        """online update で統計が更新される."""
        cal = CalibrationMap({"ewma_tau": 100.0, "n_min": 5.0})
        # 初期状態
        stats_before = cal.get_stats("ranging", 0.3)
        fb_before = stats_before["fallback"]
        n_eff_before = fb_before.get("n_eff", 0.0)

        # update
        for i in range(10):
            cal.update("ranging", 0.3, 50.0, i)
        stats_after = cal.get_stats("ranging", 0.3)
        fb_after = stats_after["fallback"]
        assert fb_after["n_eff"] > n_eff_before


# ════════════════════════════════════════════════════════════
# load_calibration_state テスト
# ════════════════════════════════════════════════════════════

class TestLoadCalibrationState:
    """load_calibration_state の結合テスト."""

    def test_load_from_json(self, tmp_path: Path) -> None:
        from scripts.v460.ml.calibration_batch import load_calibration_state

        # CalibrationMap を作成して JSON に保存
        cal = CalibrationMap({"ewma_tau": 100.0, "n_min": 30.0})
        for i in range(50):
            cal.update("ranging", 0.3, 100.0 if i < 30 else -50.0, i)
        state = cal.get_state()

        json_path = tmp_path / "test_calibration.json"
        json_path.write_text(json.dumps(state))

        # load_calibration_state で復元
        loaded = load_calibration_state(str(json_path), {"ewma_tau": 100.0, "n_min": 30.0})
        assert loaded is not None
        loaded_stats = loaded.get_stats("ranging", 0.3)
        assert loaded_stats["fallback"]["n_eff"] > 0

    def test_load_missing_file(self) -> None:
        from scripts.v460.ml.calibration_batch import load_calibration_state

        result = load_calibration_state("/nonexistent/path.json", {"ewma_tau": 100.0})
        assert result is None


# ════════════════════════════════════════════════════════════
# RunSessionState entry_gate フィールド
# ════════════════════════════════════════════════════════════

class TestRunSessionStateEntryGate:
    """RunSessionState に entry_gate 追跡フィールドが存在する."""

    def test_fields_exist(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        st = RunSessionState()
        assert st.entry_gate_eval_count == 0
        assert st.entry_gate_block_count == 0
        assert st.entry_gate_ev_sum == 0.0


# ════════════════════════════════════════════════════════════
# Gate cancel reason マッピング
# ════════════════════════════════════════════════════════════

class TestGateCancelReason:
    """entry_gate_ev_negative が cancel_reason マッピングに登録済み."""

    def test_mapping_exists(self) -> None:
        from scripts.v460.lib.cycle_gate_aggregator import _GATE_TO_CANCEL_REASON

        assert "entry_gate_ev_negative" in _GATE_TO_CANCEL_REASON
