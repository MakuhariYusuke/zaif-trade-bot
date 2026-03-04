"""276# BlockingPolicy DRY テスト.

_execute_skip ヘルパー抽出 + halt_sleep_multiplier config 化の検証。

検証項目:
  A. _execute_skip ヘルパーが存在し、正しいシグネチャを持つ
  B. 14箇所の skip ceremony が _execute_skip 経由に移行済み
  C. orchestrator 内で multiplier=5.0 ハードコードが排除済み
  D. halt_sleep_multiplier が FillTestConfig に存在し YAML から読込可能
  E. _execute_skip の動作テスト (record 生成 / batch / flush / sleep)
  F. gate_block パスの update_last_side=True 確認 (166# 回帰)
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =====================================================================
# A. _execute_skip シグネチャ検証
# =====================================================================

class TestExecuteSkipSignature:
    """_execute_skip ヘルパーの存在とシグネチャを検証."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        self.mixin = FillLoopOrchestratorMixin

    def test_method_exists(self) -> None:
        assert hasattr(self.mixin, "_execute_skip")

    def test_method_is_coroutine(self) -> None:
        assert asyncio.iscoroutinefunction(self.mixin._execute_skip)

    def test_required_parameters(self) -> None:
        sig = inspect.signature(self.mixin._execute_skip)
        params = set(sig.parameters.keys())
        expected = {
            "self", "st", "side", "cancel_reason", "flush_context",
            "order_quantity", "heartbeat", "state_save", "state_save_context",
            "update_last_side", "sleep", "multiplier", "max_override",
        }
        assert expected <= params, f"Missing params: {expected - params}"

    def test_defaults(self) -> None:
        sig = inspect.signature(self.mixin._execute_skip)
        p = sig.parameters
        assert p["order_quantity"].default == 0.0
        assert p["heartbeat"].default is False
        assert p["state_save"].default is False
        assert p["update_last_side"].default is False
        assert p["sleep"].default is True
        assert p["multiplier"].default == 1.0
        assert p["max_override"].default == 0.0


# =====================================================================
# B. 14箇所の _execute_skip 移行確認
# =====================================================================

class TestSkipCeremonyMigration:
    """skip ceremony が _execute_skip 経由に移行済みかソースコードで検証."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.src = Path(
            "scripts/v460/lib/fill_loop_orchestrator.py"
        ).read_text(encoding="utf-8")

    def test_execute_skip_call_count(self) -> None:
        """_execute_skip 呼出が 14 箇所以上存在する."""
        count = self.src.count("await self._execute_skip(")
        assert count >= 14, f"Expected >=14 _execute_skip calls, got {count}"

    def test_operator_halt_uses_execute_skip(self) -> None:
        """operator_halt パスが _execute_skip を使用."""
        idx = self.src.find("OPERATOR_HALT")
        nearby = self.src[max(0, idx - 200):idx + 300]
        assert "_execute_skip" in nearby

    def test_mcb_halt_uses_execute_skip(self) -> None:
        """MCB HALT パスが _execute_skip を使用."""
        idx = self.src.find("MCBLevel.HALT")
        nearby = self.src[idx:idx + 500]
        assert "_execute_skip" in nearby

    def test_sad_frozen_uses_execute_skip(self) -> None:
        """SAD FROZEN パスが _execute_skip を使用."""
        idx = self.src.find("SADLevel.FROZEN")
        nearby = self.src[idx:idx + 500]
        assert "_execute_skip" in nearby

    def test_mcb_sad_escalation_uses_execute_skip(self) -> None:
        """MCB×SAD escalation パスが _execute_skip を使用."""
        idx = self.src.find("_mcb_warning and _sad_warning")
        nearby = self.src[idx:idx + 500]
        assert "_execute_skip" in nearby

    def test_toxic_veto_uses_execute_skip(self) -> None:
        """toxic_veto both-blocked パスが _execute_skip を使用."""
        idx = self.src.find("TOXIC_FILL_SIDE_VETO")
        nearby = self.src[max(0, idx - 200):idx + 300]
        assert "_execute_skip" in nearby

    def test_phantom_veto_uses_execute_skip(self) -> None:
        """phantom_veto パスが _execute_skip を使用."""
        idx = self.src.find("PHANTOM_SIDE_VETO")
        nearby = self.src[max(0, idx - 200):idx + 300]
        assert "_execute_skip" in nearby

    def test_one_sided_freeze_uses_execute_skip(self) -> None:
        """one_sided_freeze パスが _execute_skip を使用."""
        idx = self.src.find("one_sided_freeze_skip")
        assert idx >= 0
        nearby = self.src[idx:idx + 500]
        assert "_execute_skip" in nearby

    def test_balance_forced_skip_uses_execute_skip(self) -> None:
        """balance_forced_skip パスが _execute_skip を使用."""
        idx = self.src.find("BALANCE_FORCED_SKIP")
        nearby = self.src[max(0, idx - 200):idx + 300]
        assert "_execute_skip" in nearby

    def test_toxicity_participation_uses_execute_skip(self) -> None:
        """toxicity_participation_skip パスが _execute_skip を使用."""
        idx = self.src.find("toxicity_participation_skip")
        # 複数マッチあるので skip 記録用のものを探す
        while idx >= 0:
            nearby = self.src[idx:idx + 500]
            if "_execute_skip" in nearby:
                return
            idx = self.src.find("toxicity_participation_skip", idx + 1)
        pytest.fail("toxicity_participation_skip path not using _execute_skip")

    def test_degraded_liquidation_uses_execute_skip(self) -> None:
        """degraded_liquidation_duty_skip パスが _execute_skip を使用."""
        idx = self.src.find("degraded_liquidation_duty_skip")
        while idx >= 0:
            nearby = self.src[idx:idx + 500]
            if "_execute_skip" in nearby:
                return
            idx = self.src.find("degraded_liquidation_duty_skip", idx + 1)
        pytest.fail("degraded_liquidation_duty_skip not using _execute_skip")


# =====================================================================
# C. multiplier=5.0 ハードコード排除
# =====================================================================

class TestHaltMultiplierConfigified:
    """multiplier=5.0 マジックナンバーが排除され config 経由になっている."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.src = Path(
            "scripts/v460/lib/fill_loop_orchestrator.py"
        ).read_text(encoding="utf-8")

    def test_no_hardcoded_multiplier_5(self) -> None:
        """run_continuous 内で multiplier=5.0 が残っていないこと.

        docstring/コメントは許容するが、実コード行に 5.0 ハードコードは NG。
        """
        lines = self.src.split("\n")
        violations = []
        in_docstring = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # トグル式 docstring 判定
            if '"""' in stripped:
                count = stripped.count('"""')
                if count == 1:
                    in_docstring = not in_docstring
                # count==2 は 1行完結 docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("#"):
                continue
            if "multiplier=5.0" in line:
                violations.append((i, stripped))
        assert not violations, f"multiplier=5.0 hardcoded at: {violations}"

    def test_halt_mult_variable_used(self) -> None:
        """_halt_mult ローカル変数が定義され使用されている."""
        assert "_halt_mult = self.config.halt_sleep_multiplier" in self.src
        assert "multiplier=_halt_mult" in self.src


# =====================================================================
# D. halt_sleep_multiplier 設定検証
# =====================================================================

class TestHaltSleepMultiplierConfig:
    """halt_sleep_multiplier が FillTestConfig に正しく定義されている."""

    def test_config_field_exists(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "halt_sleep_multiplier")
        assert cfg.halt_sleep_multiplier == 5.0

    def test_config_customizable(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(halt_sleep_multiplier=3.0)
        assert cfg.halt_sleep_multiplier == 3.0

    def test_from_yaml_reads_halt_sleep_multiplier(self) -> None:
        """from_yaml が halt_sleep_multiplier を flat_keys で読み込むこと."""
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = {
            "halt_sleep_multiplier": 7.0,
            "max_cycle_sleep_sec": 840.0,  # 277# >= cycle_interval × halt_mult
            "results_dir": "results/test",
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.halt_sleep_multiplier == 7.0

    def test_yaml_file_has_halt_sleep_multiplier(self) -> None:
        """fill_test.yaml に halt_sleep_multiplier が定義されている."""
        import yaml
        with open("configs/v460/fill_test.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "halt_sleep_multiplier" in data
        assert data["halt_sleep_multiplier"] == 5.0


# =====================================================================
# E. _execute_skip 動作テスト
# =====================================================================

class TestExecuteSkipBehavior:
    """_execute_skip の動作を mock ベースで検証."""

    @pytest.fixture
    def orchestrator_mock(self) -> MagicMock:
        """オーケストレータの最小 mock."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        orch = MagicMock(spec=FillLoopOrchestratorMixin)
        orch._execute_skip = FillLoopOrchestratorMixin._execute_skip.__get__(orch)
        orch._make_loop_skip_record = MagicMock(return_value=MagicMock())
        orch._batch_persistence = MagicMock()
        orch._batch_persistence.maybe_flush = MagicMock(return_value=[])
        orch._update_lock_heartbeat = MagicMock()
        orch._maybe_skip_state_save = MagicMock()
        orch._effective_sleep = AsyncMock()
        orch._last_side = "buy"
        return orch

    @pytest.fixture
    def session_state(self) -> "RunSessionState":
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState
        return RunSessionState()

    @pytest.mark.asyncio
    async def test_basic_skip(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """基本的な skip: record 生成+append+count+flush+sleep."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test",
        )
        orchestrator_mock._make_loop_skip_record.assert_called_once()
        assert session_state.total_count == 1
        orchestrator_mock._batch_persistence.maybe_flush.assert_called_once()
        orchestrator_mock._effective_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heartbeat_called(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """heartbeat=True で _update_lock_heartbeat が呼ばれる."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test", heartbeat=True,
        )
        orchestrator_mock._update_lock_heartbeat.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_not_called_by_default(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """デフォルトでは _update_lock_heartbeat は呼ばれない."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test",
        )
        orchestrator_mock._update_lock_heartbeat.assert_not_called()

    @pytest.mark.asyncio
    async def test_state_save_called(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """state_save=True で _maybe_skip_state_save が呼ばれる."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test",
            state_save=True, state_save_context="ctx",
        )
        orchestrator_mock._maybe_skip_state_save.assert_called_once_with(
            session_state, "ctx",
        )

    @pytest.mark.asyncio
    async def test_update_last_side(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """update_last_side=True で _last_side が更新される."""
        await orchestrator_mock._execute_skip(
            session_state, side="sell", cancel_reason="test",
            update_last_side=True,
        )
        assert orchestrator_mock._last_side == "sell"

    @pytest.mark.asyncio
    async def test_no_sleep_when_disabled(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """sleep=False で _effective_sleep が呼ばれない."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test", sleep=False,
        )
        orchestrator_mock._effective_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiplier_passed(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """multiplier が _effective_sleep に渡される."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test", multiplier=5.0,
        )
        orchestrator_mock._effective_sleep.assert_awaited_once_with(
            multiplier=5.0, max_override=0.0,
        )

    @pytest.mark.asyncio
    async def test_flush_context_falls_back_to_cancel_reason(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """flush_context 省略時は cancel_reason が使われる."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="my_reason",
        )
        orchestrator_mock._batch_persistence.maybe_flush.assert_called_once()
        call_args = orchestrator_mock._batch_persistence.maybe_flush.call_args
        assert call_args[0][1] == "my_reason"

    @pytest.mark.asyncio
    async def test_record_kwargs_forwarded(self, orchestrator_mock: MagicMock, session_state: "RunSessionState") -> None:
        """追加の record_kwargs が _make_loop_skip_record に渡される."""
        await orchestrator_mock._execute_skip(
            session_state, side="sell", cancel_reason="test",
            order_quantity=0.001, balance_forced_switch=True,
        )
        call_kwargs = orchestrator_mock._make_loop_skip_record.call_args
        assert call_kwargs[1]["balance_forced_switch"] is True


# =====================================================================
# F. gate_block パスの update_last_side 確認 (166# 回帰)
# =====================================================================

class TestGateBlockLastSideUpdate:
    """276# gate_block の _execute_skip(update_last_side=True) 確認.

    166# deadlock fix の回帰テスト。
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.src = Path(
            "scripts/v460/lib/fill_loop_orchestrator.py"
        ).read_text(encoding="utf-8")

    def test_gate_block_has_update_last_side(self) -> None:
        """gate_result.blocked パスに update_last_side=True がある."""
        idx = self.src.find("_gate_result.blocked")
        assert idx >= 0
        nearby = self.src[idx:idx + 1500]
        has_direct = "self._last_side = next_side" in nearby
        has_via_helper = "update_last_side=True" in nearby
        assert has_direct or has_via_helper, (
            "gate_result.blocked path missing _last_side update"
        )

    def test_gate_block_sleep_false(self) -> None:
        """gate_block の _execute_skip は sleep=False（別途 quiescence 処理）."""
        idx = self.src.find("_gate_result.cancel_reason")
        assert idx >= 0
        nearby = self.src[max(0, idx - 300):idx + 500]
        assert "_execute_skip" in nearby, "gate_block path should use _execute_skip"
        assert "sleep=False" in nearby, "gate_block _execute_skip should have sleep=False"
