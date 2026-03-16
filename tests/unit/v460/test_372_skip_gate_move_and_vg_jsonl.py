"""372# E11/E12: skip_gate 移動 + VG JSONL 構造化ログのテスト.

§1 skip_gate 移動検証
  - ztb.ml.skip_gate からの正規 import
  - scripts.v460.ml.skip_gate シムからの互換 import
  - skip_gate_features 同様

§2 VG JSONL イベントログ
  - _emit_vg_event の書き込み検証
  - _load_vg_activations_jsonl の読み込み検証
  - JSONL 優先 + regex フォールバック検証
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.v460.analysis.vg_and_trend import _load_vg_activations_jsonl
from scripts.v460.lib.maker_risk_guards import _emit_vg_event
from scripts.v460.ml.skip_gate import (
    SkipDecision as ShimSkipDecision,
    SkipGate as ShimSkipGate,
    _BASE_FEATURE_COLS,
)
from scripts.v460.ml.skip_gate_features import (
    FEATURE_NAME_MIGRATION as SHIM_FEATURE_NAME_MIGRATION,
)
from ztb.ml.skip_gate import (
    GATE_FEATURE_COLS,
    SkipDecision,
    SkipGate,
    SkipGateConfig,
    build_features_from_market_state,
    get_gate_feature_cols,
)
from ztb.ml.skip_gate_features import FEATURE_NAME_MIGRATION


# ═══════════════════════════════════════════════════════════════
# §1 skip_gate 移動検証
# ═══════════════════════════════════════════════════════════════


class TestSkipGateCanonicalImport:
    """ztb.ml.skip_gate からの正規 import が動作するか."""

    def test_import_skip_gate_class(self) -> None:
        assert hasattr(SkipGate, "evaluate")

    def test_import_skip_gate_config(self) -> None:
        cfg = SkipGateConfig()
        assert hasattr(cfg, "threshold_bps")

    def test_import_skip_decision(self) -> None:
        field_names = {f.name for f in dataclasses.fields(SkipDecision)}
        assert "should_skip" in field_names

    def test_import_feature_cols(self) -> None:
        assert len(GATE_FEATURE_COLS) > 0
        assert GATE_FEATURE_COLS == get_gate_feature_cols(use_ob=False)

    def test_import_build_features(self) -> None:
        assert callable(build_features_from_market_state)


class TestSkipGateShimImport:
    """scripts.v460.ml.skip_gate シムからの互換 import が動作するか."""

    def test_shim_skip_gate(self) -> None:
        assert ShimSkipGate is SkipGate

    def test_shim_skip_decision(self) -> None:
        assert ShimSkipDecision is SkipDecision

    def test_shim_features(self) -> None:
        assert SHIM_FEATURE_NAME_MIGRATION is FEATURE_NAME_MIGRATION

    def test_shim_private_base_feature_cols(self) -> None:
        """テストコードが _BASE_FEATURE_COLS をインポートできること."""
        assert isinstance(_BASE_FEATURE_COLS, list)
        assert len(_BASE_FEATURE_COLS) > 0


# ═══════════════════════════════════════════════════════════════
# §2 VG JSONL イベントログ
# ═══════════════════════════════════════════════════════════════


class TestVgEmitEvent:
    """maker_risk_guards._emit_vg_event の動作検証."""

    def test_emit_creates_jsonl(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "vg_events.jsonl"
        with patch(
            "scripts.v460.lib.maker_risk_guards._VG_EVENT_LOG_PATH",
            str(jsonl_path),
        ):
            _emit_vg_event(
                side="sell",
                pre_offset=0.28,
                post_offset=0.30,
                reason="vpin=0.98",
                velocity_bps=18.0,
                vpin=0.98,
                boost_factor=2.0,
            )

        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event"] == "vg_activation"
        assert data["side"] == "sell"
        assert data["pre_offset"] == 0.28
        assert data["post_offset"] == 0.30
        assert data["reason"] == "vpin=0.98"
        assert data["velocity_bps"] == 18.0
        assert data["vpin"] == 0.98
        assert data["boost_factor"] == 2.0
        assert "timestamp_iso" in data
        assert "timestamp_epoch" in data

    def test_emit_multiple_appends(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "vg_events.jsonl"
        with patch(
            "scripts.v460.lib.maker_risk_guards._VG_EVENT_LOG_PATH",
            str(jsonl_path),
        ):
            for i in range(3):
                _emit_vg_event(
                    side="buy" if i % 2 == 0 else "sell",
                    pre_offset=0.1 * i,
                    post_offset=0.1 * (i + 1),
                    reason=f"test_{i}",
                    velocity_bps=float(i),
                    vpin=0.5,
                    boost_factor=1.5,
                )

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_emit_failure_silent(self, tmp_path: Path) -> None:
        """JSONL 書き込み失敗時に例外を投げないこと."""
        # 存在しないディレクトリの存在しないパスに書くがエラーにならない
        # (append_jsonl が ensure_parent_dir するので実際は成功するが、
        # ここでは append_jsonl 自体を例外にして silent 確認)
        with patch(
            "scripts.v460.lib.maker_risk_guards._VG_EVENT_LOG_PATH",
            str(tmp_path / "vg_events.jsonl"),
        ), patch(
            "ztb.io.jsonl.open",
            side_effect=PermissionError("test"),
        ):
            # should not raise
            _emit_vg_event(
                side="buy",
                pre_offset=0.1,
                post_offset=0.2,
                reason="test",
                velocity_bps=None,
                vpin=None,
                boost_factor=1.0,
            )


class TestVgLoadJsonl:
    """vg_and_trend._load_vg_activations_jsonl の読み込み検証."""

    def test_load_empty_file(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "vg_events.jsonl"
        jsonl_path.write_text("")
        result = _load_vg_activations_jsonl(jsonl_path)
        assert result == []

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        result = _load_vg_activations_jsonl(tmp_path / "nope.jsonl")
        assert result == []

    def test_load_valid_events(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "vg_events.jsonl"
        events = [
            {
                "event": "vg_activation",
                "timestamp_iso": "2026-03-10T12:00:00+00:00",
                "timestamp_epoch": 1773208800.0,
                "side": "sell",
                "pre_offset": 0.28,
                "post_offset": 0.30,
                "reason": "vpin=0.98",
            },
            {
                "event": "vg_activation",
                "timestamp_iso": "2026-03-10T12:05:00+00:00",
                "timestamp_epoch": 1773209100.0,
                "side": "buy",
                "pre_offset": 0.15,
                "post_offset": 0.20,
                "reason": "velocity=22.0bps",
            },
        ]
        jsonl_path.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        result = _load_vg_activations_jsonl(jsonl_path)
        assert len(result) == 2
        assert result[0]["side"] == "sell"
        assert result[0]["timestamp"] == 1773208800.0
        assert result[1]["side"] == "buy"

    def test_load_ignores_non_vg_events(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "vg_events.jsonl"
        events = [
            {"event": "other_event", "timestamp_epoch": 100.0},
            {
                "event": "vg_activation",
                "timestamp_epoch": 200.0,
                "side": "buy",
                "pre_offset": 0.1,
                "post_offset": 0.2,
                "reason": "test",
            },
        ]
        jsonl_path.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        result = _load_vg_activations_jsonl(jsonl_path)
        assert len(result) == 1
        assert result[0]["timestamp"] == 200.0

    def test_load_handles_malformed_lines(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "vg_events.jsonl"
        content = (
            '{"event": "vg_activation", "timestamp_epoch": 100.0, '
            '"side": "buy", "pre_offset": 0.1, "post_offset": 0.2, "reason": "ok"}\n'
            "this is not json\n"
            '{"event": "vg_activation", "timestamp_epoch": 200.0, '
            '"side": "sell", "pre_offset": 0.3, "post_offset": 0.4, "reason": "ok2"}\n'
        )
        jsonl_path.write_text(content)
        result = _load_vg_activations_jsonl(jsonl_path)
        assert len(result) == 2
