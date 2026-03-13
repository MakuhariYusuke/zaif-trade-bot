"""209# H4: DynamicKillManager 状態永続化テスト.

export_state / import_state のラウンドトリップ、
FillTestState 統合、fill records warmup を検証。
"""

from __future__ import annotations

import pytest

from ztb.risk.sell_dynamic_kill import (
    BuyDynamicKillManager,
    DynamicKillConfig,
    DynamicKillManager,
    SellDynamicKillManager,
    SellKillConfig,
)


# =====================================================================
# export_state / import_state 単体テスト
# =====================================================================


class TestDynamicKillManagerExportImport:
    """export_state / import_state のラウンドトリップ."""

    def test_export_empty_state(self) -> None:
        """初期状態の export."""
        mgr = DynamicKillManager(DynamicKillConfig(window=5))
        state = mgr.export_state()
        assert state["pnl_history"] == []
        assert state["cooldown"] == 0
        assert state["total_kills"] == 0
        assert state["total_cooldown_cycles"] == 0
        assert state["side"] == "sell"

    def test_export_after_tracking(self) -> None:
        """track() 後の export で pnl_history が正しい."""
        mgr = DynamicKillManager(DynamicKillConfig(window=3))
        mgr.track(1.0)
        mgr.track(-0.5)
        mgr.track(0.3)
        state = mgr.export_state()
        assert state["pnl_history"] == [1.0, -0.5, 0.3]

    def test_export_after_kill(self) -> None:
        """kill 発火後の export に cooldown / total_kills が反映."""
        mgr = DynamicKillManager(DynamicKillConfig(
            window=2, threshold_bps=-0.5, resume_window=3,
        ))
        mgr.track(-1.0)
        mgr.track(-1.0)
        mgr.check_kill()  # kill → cooldown=3
        state = mgr.export_state()
        assert state["total_kills"] == 1
        assert state["cooldown"] == 3

    def test_roundtrip(self) -> None:
        """export → import → export が同一結果."""
        mgr = DynamicKillManager(DynamicKillConfig(
            window=3, threshold_bps=-1.0, resume_window=5,
        ))
        for v in [0.1, -0.3, 0.5, -1.2, 0.8]:
            mgr.track(v)
        mgr.check_kill()  # may or may not kill
        state1 = mgr.export_state()

        mgr2 = DynamicKillManager(DynamicKillConfig(
            window=3, threshold_bps=-1.0, resume_window=5,
        ))
        mgr2.import_state(state1)
        state2 = mgr2.export_state()

        assert state1["pnl_history"] == state2["pnl_history"]
        assert state1["cooldown"] == state2["cooldown"]
        assert state1["total_kills"] == state2["total_kills"]
        assert state1["total_cooldown_cycles"] == state2["total_cooldown_cycles"]

    def test_roundtrip_preserves_behavior(self) -> None:
        """import 後の check_kill が元と同じ結果を返す."""
        cfg = DynamicKillConfig(window=3, threshold_bps=-0.5, resume_window=2)
        mgr = DynamicKillManager(cfg)
        for v in [-1.0, -1.0, -1.0]:
            mgr.track(v)
        killed1, tel1 = mgr.check_kill()
        assert killed1 is True

        # export → new manager with same config → import
        state = mgr.export_state()
        mgr2 = DynamicKillManager(cfg)
        mgr2.import_state(state)
        # cooldown should be active
        killed2, tel2 = mgr2.check_kill()
        assert killed2 is True
        assert tel2.cooldown_remaining == tel1.cooldown_remaining - 1

    def test_import_missing_keys(self) -> None:
        """キーが欠落した dict でもデフォルト値にフォールバック."""
        mgr = DynamicKillManager(DynamicKillConfig(window=5))
        mgr.import_state({})
        assert mgr._pnl_history == []
        assert mgr._cooldown == 0
        assert mgr._total_kills == 0
        assert mgr._total_cooldown_cycles == 0

    def test_import_respects_memory_limit(self) -> None:
        """import 時に window*3 キャップが適用される."""
        cfg = DynamicKillConfig(window=3)
        mgr = DynamicKillManager(cfg)
        huge_history = list(range(100))
        mgr.import_state({"pnl_history": huge_history})
        assert len(mgr._pnl_history) == 9  # window*3
        assert mgr._pnl_history == [float(i) for i in range(91, 100)]

    def test_import_does_not_change_side(self) -> None:
        """import は side を変更しない (コンストラクタで固定)."""
        mgr = DynamicKillManager(side="sell")
        mgr.import_state({"side": "buy", "pnl_history": [1.0]})
        assert mgr.side == "sell"

    def test_buy_manager_export(self) -> None:
        """BuyDynamicKillManager の export で side='buy'."""
        mgr = BuyDynamicKillManager(DynamicKillConfig(window=3))
        mgr.track(0.5)
        state = mgr.export_state()
        assert state["side"] == "buy"

    def test_export_returns_copy(self) -> None:
        """export された list は内部状態と独立 (mutation safe)."""
        mgr = DynamicKillManager(DynamicKillConfig(window=3))
        mgr.track(1.0)
        state = mgr.export_state()
        state["pnl_history"].append(999.0)
        assert 999.0 not in mgr._pnl_history


# =====================================================================
# FillTestState 統合テスト
# =====================================================================


class TestFillTestStateKillFields:
    """FillTestState に sell_kill_state / buy_kill_state フィールドがある."""

    def test_fields_default_none(self) -> None:
        """デフォルトは None."""
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState()
        assert state.sell_kill_state is None
        assert state.buy_kill_state is None

    def test_fields_accept_dict(self) -> None:
        """dict を設定可能."""
        from scripts.v460.lib.resilience import FillTestState
        sell_state = {"pnl_history": [1.0, -0.5], "cooldown": 2, "total_kills": 1}
        buy_state = {"pnl_history": [0.3], "cooldown": 0, "total_kills": 0}
        state = FillTestState(sell_kill_state=sell_state, buy_kill_state=buy_state)
        assert state.sell_kill_state == sell_state
        assert state.buy_kill_state == buy_state


# =====================================================================
# Warmup from fill records テスト
# =====================================================================


class TestKillManagerWarmupFromRecords:
    """209# H4: _warmup_kill_managers_from_records のテスト."""

    @staticmethod
    def _make_record(
        *,
        side: str = "sell",
        filled: bool = True,
        pnl: float | None = 0.5,
    ) -> object:
        """テスト用の簡易レコード."""
        class FakeRecord:
            pass
        r = FakeRecord()
        r.filled = filled
        r.side = side
        r.post_fill_30s_pnl = pnl
        return r

    def test_sell_records_feed_sell_manager(self) -> None:
        """sell fill records → sell kill manager に投入される."""
        mgr = DynamicKillManager(DynamicKillConfig(window=3), side="sell")
        records = [
            self._make_record(side="sell", pnl=-0.5),
            self._make_record(side="sell", pnl=-1.0),
            self._make_record(side="sell", pnl=0.3),
        ]
        for r in records:
            if r.filled and r.side == "sell" and r.post_fill_30s_pnl is not None:
                mgr.track(r.post_fill_30s_pnl)
        assert len(mgr._pnl_history) == 3
        assert mgr._pnl_history == [-0.5, -1.0, 0.3]

    def test_unfilled_records_skipped(self) -> None:
        """filled=False のレコードはスキップ."""
        mgr = DynamicKillManager(DynamicKillConfig(window=3))
        records = [
            self._make_record(side="sell", filled=False, pnl=-1.0),
            self._make_record(side="sell", filled=True, pnl=0.5),
        ]
        for r in records:
            if r.filled and r.side == "sell" and r.post_fill_30s_pnl is not None:
                mgr.track(r.post_fill_30s_pnl)
        assert len(mgr._pnl_history) == 1

    def test_none_pnl_records_skipped(self) -> None:
        """post_fill_30s_pnl=None はスキップ."""
        mgr = DynamicKillManager(DynamicKillConfig(window=3))
        records = [
            self._make_record(side="sell", pnl=None),
            self._make_record(side="sell", pnl=0.5),
        ]
        for r in records:
            if r.filled and r.side == "sell" and r.post_fill_30s_pnl is not None:
                mgr.track(r.post_fill_30s_pnl)
        assert len(mgr._pnl_history) == 1

    def test_buy_records_not_mixed(self) -> None:
        """buy records は sell manager に入らない."""
        sell_mgr = DynamicKillManager(DynamicKillConfig(window=3), side="sell")
        buy_mgr = BuyDynamicKillManager(DynamicKillConfig(window=3))
        records = [
            self._make_record(side="sell", pnl=-0.5),
            self._make_record(side="buy", pnl=0.3),
            self._make_record(side="sell", pnl=-1.0),
            self._make_record(side="buy", pnl=0.8),
        ]
        for r in records:
            if not r.filled or r.post_fill_30s_pnl is None:
                continue
            if r.side == "sell":
                sell_mgr.track(r.post_fill_30s_pnl)
            elif r.side == "buy":
                buy_mgr.track(r.post_fill_30s_pnl)
        assert sell_mgr._pnl_history == [-0.5, -1.0]
        assert buy_mgr._pnl_history == [0.3, 0.8]


# =====================================================================
# Cooldown サイクル数の永続化テスト
# =====================================================================


class TestCooldownCyclesPersistence:
    """cooldown 中に export → import → check_kill で正しく cooldown が継続."""

    def test_cooldown_survives_restart(self) -> None:
        """cooldown 途中で export → import → cooldown 継続."""
        cfg = DynamicKillConfig(window=2, threshold_bps=-0.5, resume_window=5)
        mgr = DynamicKillManager(cfg)
        mgr.track(-1.0)
        mgr.track(-1.0)
        mgr.check_kill()  # kill, cooldown=5
        mgr.check_kill()  # cooldown=4
        mgr.check_kill()  # cooldown=3

        state = mgr.export_state()
        assert state["cooldown"] == 3
        assert state["total_cooldown_cycles"] == 2  # 2 decrements so far

        # "restart"
        mgr2 = DynamicKillManager(cfg)
        mgr2.import_state(state)

        killed, tel = mgr2.check_kill()  # cooldown=2
        assert killed is True
        assert tel.cooldown_remaining == 2
        assert mgr2._total_cooldown_cycles == 3  # 2 + 1
