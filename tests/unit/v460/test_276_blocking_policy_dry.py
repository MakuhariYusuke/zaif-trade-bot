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

import pytest
from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin, RunSessionState
from tests.unit.v460._fill_test_source import read_fill_test_runner_source
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping

_EXECUTE_SKIP_SIG = inspect.signature(FillLoopOrchestratorMixin._execute_skip)


# =====================================================================
# A. _execute_skip シグネチャ検証
# =====================================================================

class TestExecuteSkipSignature:
    """_execute_skip ヘルパーの存在とシグネチャを検証."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.mixin = FillLoopOrchestratorMixin

    def test_method_exists(self) -> None:
        assert hasattr(self.mixin, "_execute_skip")

    def test_method_is_coroutine(self) -> None:
        assert asyncio.iscoroutinefunction(self.mixin._execute_skip)

    def test_required_parameters(self) -> None:
        params = set(_EXECUTE_SKIP_SIG.parameters.keys())
        expected = {
            "self", "st", "side", "cancel_reason", "flush_context",
            "order_quantity", "heartbeat", "state_save", "state_save_context",
            "update_last_side", "sleep", "multiplier", "max_override",
        }
        assert expected <= params, f"Missing params: {expected - params}"

    def test_defaults(self) -> None:
        p = _EXECUTE_SKIP_SIG.parameters
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
        self.src = read_fill_test_runner_source()

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
        nearby = self.src[idx:idx + 800]
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
        # 332# Phase 4: mixin 分割により全ソースを連結して検証
        from tests.unit.v460._fill_test_source import read_fill_test_runner_source  # 332# Phase 4
        self.src = read_fill_test_runner_source()  # 332# Phase 4

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
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_cfg))
        assert cfg.halt_sleep_multiplier == 7.0

    def test_yaml_file_has_halt_sleep_multiplier(
        self,
        v460_fill_test_yaml: dict[str, object],
    ) -> None:
        """fill_test.yaml に halt_sleep_multiplier が定義されている."""
        data = v460_fill_test_yaml
        assert "halt_sleep_multiplier" in data
        assert data["halt_sleep_multiplier"] == 5.0


# =====================================================================
# E. _execute_skip 動作テスト
# =====================================================================

class TestExecuteSkipBehavior:
    """_execute_skip の動作を mock ベースで検証."""

    class _SleepStub:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        async def __call__(self, *, multiplier: float = 1.0, max_override: float = 0.0) -> None:
            self.calls.append((multiplier, max_override))

        def assert_awaited_once(self) -> None:
            assert len(self.calls) == 1

        def assert_not_awaited(self) -> None:
            assert self.calls == []

        def assert_awaited_once_with(self, *, multiplier: float, max_override: float) -> None:
            assert self.calls == [(multiplier, max_override)]

    class _BatchPersistenceStub:
        def __init__(self) -> None:
            self.calls: list[tuple[list[object], str]] = []

        def maybe_flush(self, batch: list[object], context: str) -> list[object]:
            self.calls.append((list(batch), context))
            return []

    class _OrchestratorStub:
        def __init__(self, execute_skip: object) -> None:
            self._execute_skip = execute_skip
            self.record_calls: list[dict[str, object]] = []
            self._batch_persistence = TestExecuteSkipBehavior._BatchPersistenceStub()
            self._heartbeat_calls = 0
            self._state_save_calls: list[tuple[object, str]] = []
            self._effective_sleep = TestExecuteSkipBehavior._SleepStub()
            self._last_side = "buy"

        def _make_loop_skip_record(self, **kwargs: object) -> object:
            self.record_calls.append(dict(kwargs))
            return SimpleNamespace(**kwargs)

        def _update_lock_heartbeat(self) -> None:
            self._heartbeat_calls += 1

        def _maybe_skip_state_save(self, state: object, context: str) -> None:
            self._state_save_calls.append((state, context))

    @pytest.fixture
    def orchestrator_mock(self) -> "_OrchestratorStub":
        """オーケストレータの最小 stub."""
        orch = self._OrchestratorStub(FillLoopOrchestratorMixin._execute_skip)
        orch._execute_skip = FillLoopOrchestratorMixin._execute_skip.__get__(orch)
        return orch

    @pytest.fixture
    def session_state(self) -> "RunSessionState":
        return RunSessionState()

    @pytest.mark.asyncio
    async def test_basic_skip(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """基本的な skip: record 生成+append+count+flush+sleep."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test",
        )
        assert len(orchestrator_mock.record_calls) == 1
        assert session_state.total_count == 1
        assert len(orchestrator_mock._batch_persistence.calls) == 1
        orchestrator_mock._effective_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heartbeat_called(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """heartbeat=True で _update_lock_heartbeat が呼ばれる."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test", heartbeat=True,
        )
        assert orchestrator_mock._heartbeat_calls == 1

    @pytest.mark.asyncio
    async def test_heartbeat_not_called_by_default(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """デフォルトでは _update_lock_heartbeat は呼ばれない."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test",
        )
        assert orchestrator_mock._heartbeat_calls == 0

    @pytest.mark.asyncio
    async def test_state_save_called(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """state_save=True で _maybe_skip_state_save が呼ばれる."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test",
            state_save=True, state_save_context="ctx",
        )
        assert orchestrator_mock._state_save_calls == [(session_state, "ctx")]

    @pytest.mark.asyncio
    async def test_update_last_side(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """update_last_side=True で _last_side が更新される."""
        await orchestrator_mock._execute_skip(
            session_state, side="sell", cancel_reason="test",
            update_last_side=True,
        )
        assert orchestrator_mock._last_side == "sell"

    @pytest.mark.asyncio
    async def test_no_sleep_when_disabled(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """sleep=False で _effective_sleep が呼ばれない."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test", sleep=False,
        )
        orchestrator_mock._effective_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiplier_passed(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """multiplier が _effective_sleep に渡される."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="test", multiplier=5.0,
        )
        orchestrator_mock._effective_sleep.assert_awaited_once_with(
            multiplier=5.0, max_override=0.0,
        )

    @pytest.mark.asyncio
    async def test_flush_context_falls_back_to_cancel_reason(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """flush_context 省略時は cancel_reason が使われる."""
        await orchestrator_mock._execute_skip(
            session_state, side="none", cancel_reason="my_reason",
        )
        assert len(orchestrator_mock._batch_persistence.calls) == 1
        assert orchestrator_mock._batch_persistence.calls[0][1] == "my_reason"

    @pytest.mark.asyncio
    async def test_record_kwargs_forwarded(self, orchestrator_mock: "_OrchestratorStub", session_state: "RunSessionState") -> None:
        """追加の record_kwargs が _make_loop_skip_record に渡される."""
        await orchestrator_mock._execute_skip(
            session_state, side="sell", cancel_reason="test",
            order_quantity=0.001, balance_forced_switch=True,
        )
        assert orchestrator_mock.record_calls[0]["balance_forced_switch"] is True


# =====================================================================
# F. gate_block パスの update_last_side 確認 (166# 回帰)
# =====================================================================

class TestGateBlockLastSideUpdate:
    """276# gate_block の _execute_skip(update_last_side=True) 確認.

    166# deadlock fix の回帰テスト。
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.src = read_fill_test_runner_source()

    def test_gate_block_has_update_last_side(self) -> None:
        """gate_result.blocked パスに update_last_side=True がある."""
        idx = self.src.find("_gate_result.blocked")
        assert idx >= 0
        nearby = self.src[idx:idx + 500]
        # blocked 直後に _handle_gate_block が呼ばれていること
        has_handler_call = "_handle_gate_block(st, ctx, _gate_result)" in nearby
        # _handle_gate_block メソッド内で update_last_side=True が使われていること
        handler_idx = self.src.find("def _handle_gate_block(")
        handler_body = self.src[handler_idx:handler_idx + 1500]
        has_update_last_side = "update_last_side=True" in handler_body
        assert has_handler_call and has_update_last_side, (
            "gate_result.blocked path missing _last_side update"
        )

    def test_gate_block_sleep_false(self) -> None:
        """gate_block の _execute_skip は sleep=False（別途 quiescence 処理）."""
        idx = self.src.find("gate_result.cancel_reason")
        assert idx >= 0
        nearby = self.src[max(0, idx - 600):idx + 800]
        assert "_execute_skip" in nearby, "gate_block path should use _execute_skip"
        assert "sleep=False" in nearby, "gate_block _execute_skip should have sleep=False"
