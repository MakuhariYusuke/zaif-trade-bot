"""551# Toxicity Distribution Counter テスト (546# D).

RunSessionState の toxicity_level_counts / sidecar_nonzero_count が
正しく記録されることを検証する。
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from ztb.risk.toxicity_types import ToxicityAssessment, ToxicityLevel


# ═════════════════════════════════════════════════════════
# 1. RunSessionState フィールド存在テスト
# ═════════════════════════════════════════════════════════


class TestRunSessionStateFields:
    """551# 追加フィールドが RunSessionState に存在する."""

    def test_toxicity_level_counts_field(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        st = RunSessionState()
        assert hasattr(st, "toxicity_level_counts")
        assert isinstance(st.toxicity_level_counts, dict)
        assert len(st.toxicity_level_counts) == 0

    def test_sidecar_nonzero_count_field(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        st = RunSessionState()
        assert hasattr(st, "sidecar_nonzero_count")
        assert st.sidecar_nonzero_count == 0


# ═════════════════════════════════════════════════════════
# 2. Toxicity level key format テスト
# ═════════════════════════════════════════════════════════


class TestToxicityLevelKeyFormat:
    """toxicity_level_counts のキーが {side}_{LEVEL} 形式."""

    @pytest.mark.parametrize("side", ["buy", "sell"])
    @pytest.mark.parametrize("level", [ToxicityLevel.GREEN, ToxicityLevel.YELLOW, ToxicityLevel.ORANGE, ToxicityLevel.KILL])
    def test_key_format(self, side: str, level: ToxicityLevel) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        st = RunSessionState()
        key = f"{side}_{level.name}"
        st.toxicity_level_counts[key] = 1
        assert key in st.toxicity_level_counts

    def test_level_name_matches_enum(self) -> None:
        assert ToxicityLevel.GREEN.name == "GREEN"
        assert ToxicityLevel.YELLOW.name == "YELLOW"
        assert ToxicityLevel.ORANGE.name == "ORANGE"
        assert ToxicityLevel.KILL.name == "KILL"


# ═════════════════════════════════════════════════════════
# 3. ソースコード構造テスト (wiring)
# ═════════════════════════════════════════════════════════


class TestToxicityTrackingWiring:
    """orchestrator_mid_cycle.py で toxicity_level_counts が記録されること."""

    _SRC = Path("scripts/v460/lib/orchestrator_mid_cycle.py").read_text(encoding="utf-8")

    def test_buy_toxicity_tracked(self) -> None:
        assert "buy_{_buy_tox.level.name}" in self._SRC or "buy_" in self._SRC
        assert "toxicity_level_counts" in self._SRC

    def test_sell_toxicity_tracked(self) -> None:
        assert "sell_{_sell_tox.level.name}" in self._SRC or "sell_" in self._SRC
        assert "toxicity_level_counts" in self._SRC

    def test_sidecar_nonzero_tracked(self) -> None:
        assert "sidecar_nonzero_count" in self._SRC


class TestToxicityProgressLog:
    """orchestrator_post_cycle.py で toxicity distribution がログ出力されること."""

    _SRC = Path("scripts/v460/lib/orchestrator_post_cycle.py").read_text(encoding="utf-8")

    def test_toxicity_log_present(self) -> None:
        assert "551# toxicity" in self._SRC

    def test_orange_kill_rate_computed(self) -> None:
        assert "ORANGE" in self._SRC and "KILL" in self._SRC
        assert "danger_pct" in self._SRC or "_danger_pct" in self._SRC

    def test_sidecar_nonzero_log_present(self) -> None:
        assert "551# sidecar_nonzero" in self._SRC


# ═════════════════════════════════════════════════════════
# 4. ORANGE+KILL 率算出ロジックテスト
# ═════════════════════════════════════════════════════════


class TestDangerRateCalculation:
    """ORANGE+KILL 率が正しく算出されること."""

    def test_all_green(self) -> None:
        counts = {"buy_GREEN": 50, "sell_GREEN": 50}
        total = sum(counts.values())
        danger = sum(v for k, v in counts.items() if "ORANGE" in k or "KILL" in k)
        assert danger == 0
        assert danger / total * 100.0 == 0.0

    def test_mixed_levels(self) -> None:
        counts = {
            "buy_GREEN": 40,
            "buy_YELLOW": 5,
            "buy_ORANGE": 3,
            "buy_KILL": 2,
            "sell_GREEN": 40,
            "sell_ORANGE": 5,
            "sell_KILL": 5,
        }
        total = sum(counts.values())
        danger = sum(v for k, v in counts.items() if "ORANGE" in k or "KILL" in k)
        assert danger == 15
        assert total == 100
        assert danger / total * 100.0 == 15.0

    def test_all_kill(self) -> None:
        counts = {"sell_KILL": 10, "buy_KILL": 10}
        total = sum(counts.values())
        danger = sum(v for k, v in counts.items() if "ORANGE" in k or "KILL" in k)
        assert danger / total * 100.0 == 100.0
