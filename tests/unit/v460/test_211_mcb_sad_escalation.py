"""211# P1-D: MCB×SAD AND Escalation テスト.

両方が同時に WARNING 以上 → 即 HALT (AND 条件での escalation)。
MCB.WARNING かつ SAD.WIDE/DRY → MCB_SAD_ESCALATION で cycle skip。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from scripts.v460.lib.cancel_reasons import (
    AUDIT_CANCEL_REASONS,
    MCB_SAD_ESCALATION,
)
from scripts.v460.lib.micro_circuit_breaker import MCBLevel
from scripts.v460.lib.spread_anomaly_detector import SADLevel


# =====================================================================
# cancel_reasons 登録テスト
# =====================================================================


class TestCancelReasonRegistration:
    """MCB_SAD_ESCALATION が正しく定義・登録されているか."""

    def test_constant_value(self) -> None:
        assert MCB_SAD_ESCALATION == "mcb_sad_escalation"

    def test_in_audit_set(self) -> None:
        assert MCB_SAD_ESCALATION in AUDIT_CANCEL_REASONS


# =====================================================================
# orchestrator ソースコード検証
# =====================================================================


class TestOrchestratorEscalationCode:
    """fill_loop_orchestrator に AND escalation ロジックが正しく組み込まれているか."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        from scripts.v460.lib.orchestrator_pre_cycle import OrchestratorPreCycleMixin

        self.src = inspect.getsource(OrchestratorPreCycleMixin._check_circuit_breakers)

    def test_mcb_warning_flag_initialized(self) -> None:
        """_mcb_warning フラグが False で初期化されること."""
        assert "_mcb_warning = False" in self.src

    def test_sad_warning_flag_initialized(self) -> None:
        """_sad_warning フラグが False で初期化されること."""
        assert "_sad_warning = False" in self.src

    def test_mcb_warning_flag_set_on_warning(self) -> None:
        """MCB WARNING 時に _mcb_warning = True が設定されること."""
        lines = self.src.split("\n")
        for i, line in enumerate(lines):
            if "MCBLevel.WARNING" in line:
                # WARNING 分岐内で _mcb_warning = True が設定されるか
                block = "\n".join(lines[i : i + 5])
                assert "_mcb_warning = True" in block
                return
        pytest.fail("MCBLevel.WARNING check not found")

    def test_sad_warning_flag_set_on_dry(self) -> None:
        """SAD DRY 時に _sad_warning = True が設定されること."""
        lines = self.src.split("\n")
        for i, line in enumerate(lines):
            if "SADLevel.DRY" in line:
                block = "\n".join(lines[i : i + 5])
                assert "_sad_warning = True" in block
                return
        pytest.fail("SADLevel.DRY check not found")

    def test_sad_warning_flag_set_on_wide(self) -> None:
        """SAD WIDE 時に _sad_warning = True が設定されること."""
        lines = self.src.split("\n")
        for i, line in enumerate(lines):
            if "SADLevel.WIDE" in line:
                block = "\n".join(lines[i : i + 5])
                assert "_sad_warning = True" in block
                return
        pytest.fail("SADLevel.WIDE check not found")

    def test_and_escalation_condition(self) -> None:
        """AND 条件 (_mcb_warning and _sad_warning) が存在すること."""
        assert "_mcb_warning and _sad_warning" in self.src

    def test_escalation_uses_correct_cancel_reason(self) -> None:
        """escalation 時に CR.MCB_SAD_ESCALATION が使用されること."""
        assert "CR.MCB_SAD_ESCALATION" in self.src

    def test_escalation_fires_guard(self) -> None:
        """escalation 時に guard fire が記録されること."""
        assert 'self._inc_guard_fire("mcb_sad_escalation")' in self.src

    def test_escalation_triggers_return_true(self) -> None:
        """AND escalation ブロック内で return True (cycle skip) されること.

        330#: run_continuous から _check_circuit_breakers に抽出。
        continue → return True に変更。
        """
        lines = self.src.split("\n")
        in_escalation = False
        for line in lines:
            if "_mcb_warning and _sad_warning" in line:
                in_escalation = True
            if in_escalation and "return True" in line.strip():
                return  # OK
            # escalation ブロック外に出たら失敗
            if in_escalation and "return False" in line:
                break
        if not in_escalation:
            pytest.fail("AND escalation block not found")
        pytest.fail("return True not found in AND escalation block")

    def test_escalation_sleep_multiplier(self) -> None:
        """AND escalation で halt_sleep_multiplier が適用されること.

        276# DRY: _execute_skip(multiplier=_halt_mult) 経由に移行。
        276# config化: 5.0 → config.halt_sleep_multiplier (_halt_mult)。
        """
        lines = self.src.split("\n")
        in_escalation = False
        for line in lines:
            if "_mcb_warning and _sad_warning" in line:
                in_escalation = True
            if in_escalation and ("multiplier=5.0" in line or "multiplier=_halt_mult" in line):
                return  # OK
            if in_escalation and "return False" in line:
                break
        pytest.fail("halt multiplier not found in escalation block")

    def test_escalation_after_both_checks(self) -> None:
        """AND escalation が MCB と SAD の両チェックの後に配置されていること."""
        mcb_pos = self.src.find("MCBLevel.HALT")
        sad_pos = self.src.find("SADLevel.FROZEN")
        esc_pos = self.src.find("_mcb_warning and _sad_warning")
        assert mcb_pos > 0
        assert sad_pos > 0
        assert esc_pos > 0
        assert esc_pos > mcb_pos, "escalation must be after MCB check"
        assert esc_pos > sad_pos, "escalation must be after SAD check"


# =====================================================================
# フラグロジック単体テスト (orchestrator 非依存)
# =====================================================================


class TestEscalationFlagLogic:
    """AND escalation のフラグ判定ロジックを単独で検証."""

    @pytest.mark.parametrize(
        "mcb_level,sad_level,expect_escalation",
        [
            # 両方 NORMAL → 非発火
            (MCBLevel.NORMAL, SADLevel.NORMAL, False),
            # MCB WARNING のみ → 非発火
            (MCBLevel.WARNING, SADLevel.NORMAL, False),
            # SAD WIDE のみ → 非発火
            (MCBLevel.NORMAL, SADLevel.WIDE, False),
            # SAD DRY のみ → 非発火
            (MCBLevel.NORMAL, SADLevel.DRY, False),
            # MCB WARNING + SAD WIDE → 発火
            (MCBLevel.WARNING, SADLevel.WIDE, True),
            # MCB WARNING + SAD DRY → 発火
            (MCBLevel.WARNING, SADLevel.DRY, True),
            # MCB CAUTION + SAD WIDE → 非発火 (CAUTION は WARNING 未満)
            (MCBLevel.CAUTION, SADLevel.WIDE, False),
            # MCB CAUTION + SAD DRY → 非発火
            (MCBLevel.CAUTION, SADLevel.DRY, False),
        ],
        ids=[
            "both_normal",
            "mcb_warn_only",
            "sad_wide_only",
            "sad_dry_only",
            "warn_wide",
            "warn_dry",
            "caution_wide",
            "caution_dry",
        ],
    )
    def test_flag_combinations(
        self,
        mcb_level: MCBLevel,
        sad_level: SADLevel,
        expect_escalation: bool,
    ) -> None:
        """MCB/SAD レベル組み合わせでの escalation 判定."""
        # orchestrator と同じフラグロジックを再現
        _mcb_warning = mcb_level == MCBLevel.WARNING
        _sad_warning = sad_level in (SADLevel.WIDE, SADLevel.DRY)
        escalated = _mcb_warning and _sad_warning
        assert escalated == expect_escalation
