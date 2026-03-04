"""275# テスト: 責務分離 DRY + 市場理論補強.

- _is_side_killed / _track_side_pnl 統一テスト
- _tick_toxic_veto DRY 重複解消テスト
- 市場理論 docstring 存在検証 (8 モジュール追加分)
"""

from __future__ import annotations

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. side パラメータ化テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSideParameterization:
    """275# _is_side_killed / _track_side_pnl の統一テスト."""

    def test_is_side_killed_method_exists(self) -> None:
        """_is_side_killed メソッドが存在すること."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_is_side_killed")

    def test_old_methods_removed(self) -> None:
        """旧 _is_sell_killed / _is_buy_killed が除去されていること."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert not hasattr(FillLoopOrchestratorMixin, "_is_sell_killed"), \
            "_is_sell_killed should be replaced by _is_side_killed"
        assert not hasattr(FillLoopOrchestratorMixin, "_is_buy_killed"), \
            "_is_buy_killed should be replaced by _is_side_killed"

    def test_track_side_pnl_method_exists(self) -> None:
        """_track_side_pnl メソッドが存在すること."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_track_side_pnl")

    def test_old_track_methods_removed(self) -> None:
        """旧 _track_sell_pnl / _track_buy_pnl が除去されていること."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert not hasattr(FillLoopOrchestratorMixin, "_track_sell_pnl"), \
            "_track_sell_pnl should be replaced by _track_side_pnl"
        assert not hasattr(FillLoopOrchestratorMixin, "_track_buy_pnl"), \
            "_track_buy_pnl should be replaced by _track_side_pnl"

    def test_is_side_killed_docstring_has_theory(self) -> None:
        """_is_side_killed docstring に Glosten-Milgrom 参照がある."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        doc = FillLoopOrchestratorMixin._is_side_killed.__doc__ or ""
        assert "Glosten-Milgrom" in doc or "Glosten" in doc

    def test_track_side_pnl_docstring_has_theory(self) -> None:
        """_track_side_pnl docstring に Ho & Stoll 参照がある."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        doc = FillLoopOrchestratorMixin._track_side_pnl.__doc__ or ""
        assert "Ho" in doc or "Stoll" in doc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Toxic veto DRY テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToxicVetoDRY:
    """275# サイクル末尾の toxic veto 重複コードが _tick_toxic_veto に置換されたことを検証."""

    def test_no_inline_veto_decrement_in_source(self) -> None:
        """orchestrator のサイクル末尾に inline veto デクリメントが残っていないこと."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        source = inspect.getsource(FillLoopOrchestratorMixin)
        # 旧コード: "self._toxic_veto[_veto_side] -= 1" がサイクル末尾に
        # 残っていないことを確認 (_tick_toxic_veto 内は許容)
        # _tick_toxic_veto の定義外で直接デクリメントする行が無いか検証
        lines = source.split("\n")
        inline_decrements = []
        in_tick_method = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "def _tick_toxic_veto" in stripped:
                in_tick_method = True
            elif in_tick_method and stripped.startswith("def "):
                in_tick_method = False
            if (
                not in_tick_method
                and "_toxic_veto[" in stripped
                and "-= 1" in stripped
            ):
                inline_decrements.append((i, stripped))
        assert len(inline_decrements) == 0, (
            f"Found inline veto decrements outside _tick_toxic_veto: {inline_decrements}"
        )

    def test_tick_toxic_veto_called_in_cycle_end(self) -> None:
        """サイクル末尾で _tick_toxic_veto("cycle_end") が呼ばれていること."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        source = inspect.getsource(FillLoopOrchestratorMixin)
        assert '_tick_toxic_veto("cycle_end")' in source


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 市場理論 docstring 検証 (275# 新規 8 モジュール)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMarketTheoryDocstrings275:
    """275# 新規追加分の市場理論 docstring 存在検証."""

    def test_regime_detector_theory(self) -> None:
        """regime_detector に Hamilton (1989) + AMH (Lo 2004) 参照がある."""
        import scripts.v460.lib.regime_detector as mod
        doc = mod.__doc__ or ""
        assert "Hamilton" in doc
        assert "Adaptive Market" in doc or "Lo (2004)" in doc or "AMH" in doc

    def test_side_selector_theory(self) -> None:
        """side_selector に Garman / Ho & Stoll 在庫管理理論がある."""
        import scripts.v460.lib.side_selector as mod
        doc = mod.__doc__ or ""
        assert "Garman" in doc or "Ho & Stoll" in doc
        assert "Inventory" in doc or "在庫" in doc

    def test_param_adapter_theory(self) -> None:
        """param_adapter に Avellaneda-Stoikov + Glosten-Milgrom がある."""
        import scripts.v460.lib.param_adapter as mod
        doc = mod.__doc__ or ""
        assert "Avellaneda" in doc
        assert "Glosten" in doc

    def test_micro_circuit_breaker_theory(self) -> None:
        """micro_circuit_breaker に Circuit Breaker + Liquidity Spiral がある."""
        import scripts.v460.lib.micro_circuit_breaker as mod
        doc = mod.__doc__ or ""
        assert "Circuit Breaker" in doc
        assert "Brunnermeier" in doc or "Liquidity Spiral" in doc

    def test_spread_anomaly_detector_theory(self) -> None:
        """spread_anomaly_detector に Roll + Copeland & Galai がある."""
        import scripts.v460.lib.spread_anomaly_detector as mod
        doc = mod.__doc__ or ""
        assert "Roll" in doc
        assert "Copeland" in doc or "Information-Based" in doc

    def test_velocity_math_theory(self) -> None:
        """velocity_math に Kyle λ + Hasbrouck がある."""
        import scripts.v460.lib.velocity_math as mod
        doc = mod.__doc__ or ""
        assert "Kyle" in doc
        assert "Hasbrouck" in doc

    def test_macro_regime_theory(self) -> None:
        """macro_regime に Hamilton + Regime-Switching がある."""
        import scripts.v460.lib.macro_regime as mod
        doc = mod.__doc__ or ""
        assert "Hamilton" in doc
        assert "Regime-Switching" in doc or "Regime" in doc

    def test_adaptation_engine_theory(self) -> None:
        """adaptation_engine に AMH + Kelly がある."""
        import scripts.v460.lib.adaptation_engine as mod
        doc = mod.__doc__ or ""
        assert "Adaptive Market" in doc or "AMH" in doc or "Lo (2004)" in doc
        assert "Kelly" in doc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. opposite_side 再利用性検証
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOppositeSideReuse:
    """272# で抽出された _opposite_side の活用状況を検証."""

    def test_opposite_side_exists_as_static(self) -> None:
        """_opposite_side が staticmethod として存在."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_opposite_side")
        assert FillLoopOrchestratorMixin._opposite_side("buy") == "sell"
        assert FillLoopOrchestratorMixin._opposite_side("sell") == "buy"

    def test_opposite_side_used_in_source(self) -> None:
        """_opposite_side が複数箇所で呼ばれていること (DRY 活用確認)."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        source = inspect.getsource(FillLoopOrchestratorMixin)
        # _opposite_side の呼び出し回数をカウント (定義行を除く)
        call_count = source.count("self._opposite_side(") + source.count(
            "cls._opposite_side("
        )
        assert call_count >= 3, (
            f"_opposite_side is only called {call_count} times, expected ≥ 3 for DRY benefit"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 市場理論カバレッジサマリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMarketTheoryCoverage:
    """v460 全体の市場理論カバレッジを検証."""

    @pytest.mark.parametrize("module_path,expected_theory", [
        ("scripts.v460.lib.daily_drawdown_guard", "Optimal Stopping"),
        ("scripts.v460.lib.fill_loop_orchestrator", "Avellaneda"),
        ("scripts.v460.lib.cycle_gate_aggregator", "Glosten-Milgrom"),
        ("scripts.v460.lib.regime_detector", "Hamilton"),
        ("scripts.v460.lib.side_selector", "Garman"),
        ("scripts.v460.lib.micro_circuit_breaker", "Brunnermeier"),
        ("scripts.v460.lib.spread_anomaly_detector", "Roll"),
        ("scripts.v460.lib.velocity_math", "Kyle"),
        ("scripts.v460.lib.macro_regime", "Hamilton"),
        ("scripts.v460.lib.adaptation_engine", "Kelly"),
        ("scripts.v460.lib.param_adapter", "Avellaneda"),
    ])
    def test_theory_reference_exists(
        self, module_path: str, expected_theory: str,
    ) -> None:
        """各モジュールに期待される理論参照が存在する."""
        import importlib
        mod = importlib.import_module(module_path)
        doc = mod.__doc__ or ""
        assert expected_theory in doc, (
            f"{module_path} missing theory reference: {expected_theory}"
        )
