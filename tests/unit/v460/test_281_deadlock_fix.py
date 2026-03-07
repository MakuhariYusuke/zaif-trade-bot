"""281# テスト: balance_forced + per-side halt デッドロック修正.

修正内容:
- 273# I3 の untick_side_halt() を balance_forced_halt_block / both_sides_halted
  の両パスから除去 → halt が自然にカウントダウン → 永久デッドロック解消
- Inventory Escape を双方向化 (sell のみ → buy/sell 両方向)
  → BTC=0 + buy halt パターンでもデッドロック脱出可能

バグの再現状況:
- daily_pnl_bps_buy: -33.9bps → buy サイド per-side halt 発動
- BTC 残高: 0 → sell 実行不能 (残高不足で balance_forced → buy に戻される)
- halt_remaining_buy: 12 で永久停止 (tick + untick 補償で変化なし)
- 8 時間以上の完全デッドロック (236 回の skip ループ)
"""

from __future__ import annotations

import inspect

import pytest

from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
from tests.unit.v460._fill_test_source import ORCHESTRATOR_BALANCE, read_source_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1: ソースコード構造検証 — untick_side_halt 除去の確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUntickRemoval:
    """281#: orchestrator から untick_side_halt() の呼出しが除去されている."""

    def _get_source(self) -> str:
        return read_source_text(ORCHESTRATOR_BALANCE)

    def test_no_untick_in_balance_forced_halt_block(self) -> None:
        """balance_forced_halt_block パスに untick_side_halt() 呼出しがない."""
        src = self._get_source()
        # balance_forced_halt_block セクションを抽出
        block_start = src.index("balance_forced_halt_block")
        # orchestrator_balance では caller に True を返す
        block_section = src[block_start:block_start + 1500]
        return_idx = block_section.index("return True")
        halt_block_region = block_section[:return_idx]
        # コメント行を除外して実際のメソッド呼出しを検証
        code_lines = [
            line for line in halt_block_region.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "untick_side_halt" not in code_only, (
            "281# fix: balance_forced_halt_block 内に untick_side_halt 呼出しが残存"
        )

    def test_no_untick_in_both_sides_halted(self) -> None:
        """per_side_dd_both_halt パスに untick_side_halt() 呼出しがない.

        330#: run_continuous → _resolve_side_vetos に抽出。
        continue → return True に変更。
        """
        from scripts.v460.lib.orchestrator_pre_cycle import OrchestratorPreCycleMixin
        src = inspect.getsource(OrchestratorPreCycleMixin._resolve_side_vetos)
        # per_side_dd_both_halt セクションを抽出
        block_start = src.index("per_side_dd_both_halt")
        block_section = src[block_start:block_start + 1000]
        return_idx = block_section.index("return True")
        both_halt_region = block_section[:return_idx]
        # コメント行を除外して実際のメソッド呼出しを検証
        code_lines = [
            line for line in both_halt_region.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "untick_side_halt" not in code_only, (
            "281# fix: per_side_dd_both_halt 内に untick_side_halt 呼出しが残存"
        )

    def test_281_fix_comment_present(self) -> None:
        """281# fix コメントが存在する."""
        src = self._get_source()
        assert "balance_forced_halt_block" in src
        assert "balance_forced deadlock breakout" in src


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2: Inventory Escape 双方向化の検証
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInventoryEscapeBidirectional:
    """281#: Inventory Escape が buy/sell 両方向で作動する."""

    def _get_source(self) -> str:
        return read_source_text(ORCHESTRATOR_BALANCE)

    def test_ie_condition_not_sell_only(self) -> None:
        """Inventory Escape の条件に next_side == 'sell' がない."""
        src = self._get_source()
        # balance_forced 後の Inventory Escape セクションを抽出
        # (複数の "Inventory Escape Mode" があるので balance_forced_halt_block 手前を特定)
        ie_anchor = src.index("balance_forced deadlock breakout")
        ie_section = src[max(0, ie_anchor - 1500):ie_anchor]
        # 旧条件: `_ie_enabled and next_side == "sell"` が存在しないこと
        assert 'next_side == "sell"' not in ie_section, (
            "281# fix: Inventory Escape が sell 限定のままになっている"
        )
        # 新条件: `if _ie_enabled:` が存在すること
        assert "if _ie_enabled:" in ie_section, (
            "281# fix: Inventory Escape の条件が _ie_enabled のみであるべき"
        )

    def test_ie_bidirectional_comment_present(self) -> None:
        """双方向化のコメントが存在する."""
        src = self._get_source()
        assert "balance_forced deadlock breakout" in src
        assert "for {next_side}" in src


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3: DailyDrawdownGuard — halt カウントダウン動作テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHaltCountdownWithoutUntick:
    """281#: untick なしで halt が自然に満了する."""

    def _make_guard(self, halt_cycles: int = 15) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_hard_limit_bps=-5.0,
            per_side_halt_cycles=halt_cycles,
        )

    def test_halt_expires_after_n_ticks_without_untick(self) -> None:
        """N 回の tick_side_halt() で halt が自然解除される (untick なし)."""
        guard = self._make_guard(halt_cycles=15)
        guard.update_pnl(-6.0, side="buy")
        assert guard.is_side_halted("buy")
        assert guard._state.side_halt_remaining_buy == 15

        # untick なしで 15 回 tick → halt 解除
        for i in range(15):
            guard.tick_side_halt()

        assert not guard.is_side_halted("buy")
        assert guard._state.side_halt_remaining_buy == 0

    def test_halt_stuck_with_tick_untick_pair(self) -> None:
        """tick + untick ペアでは halt が解除されない (旧バグの再現)."""
        guard = self._make_guard(halt_cycles=15)
        guard.update_pnl(-6.0, side="buy")
        assert guard.is_side_halted("buy")

        # tick + untick を 30 回 → halt 永続 (旧デッドロックパターン)
        for _ in range(30):
            guard.tick_side_halt()
            guard.untick_side_halt()

        assert guard.is_side_halted("buy"), "tick + untick ペアでは halt が永続する"
        assert guard._state.side_halt_remaining_buy == 15

    def test_both_sides_halt_expires_after_n_ticks(self) -> None:
        """両サイド halt も N 回の tick で自然解除される."""
        guard = self._make_guard(halt_cycles=10)
        guard.update_pnl(-6.0, side="buy")
        guard.update_pnl(-6.0, side="sell")
        assert guard.is_side_halted("buy")
        assert guard.is_side_halted("sell")

        # 10 回の tick (untick なし) → 両サイド解除
        for _ in range(10):
            guard.tick_side_halt()

        assert not guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

    def test_reanchor_set_on_halt_release(self) -> None:
        """halt 解除時に reanchor PnL が設定される (269#)."""
        guard = self._make_guard(halt_cycles=5)
        guard.update_pnl(-6.0, side="buy")
        assert guard.is_side_halted("buy")

        # 5 回 tick → halt 解除
        for _ in range(5):
            guard.tick_side_halt()

        assert not guard.is_side_halted("buy")
        # reanchor PnL が設定されている
        assert guard._state.side_reanchor_pnl_buy == guard._state.daily_pnl_bps_buy

    def test_partial_countdown_not_stuck(self) -> None:
        """途中まで進んだカウントダウンが止まらない (281# デッドロック防止)."""
        guard = self._make_guard(halt_cycles=15)
        guard.update_pnl(-6.0, side="buy")
        assert guard._state.side_halt_remaining_buy == 15

        # 3 回 tick → remaining 12 (今日のデッドロック再現値)
        for _ in range(3):
            guard.tick_side_halt()
        assert guard._state.side_halt_remaining_buy == 12
        assert guard.is_side_halted("buy")

        # 残り 12 回 tick (untick なし) → halt 解除
        for _ in range(12):
            guard.tick_side_halt()
        assert not guard.is_side_halted("buy")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4: デッドロック・シナリオ再現テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeadlockScenario:
    """281#: BTC=0 + buy halt のデッドロック・シナリオ再現."""

    def _make_guard(self) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_hard_limit_bps=-30.0,
            per_side_halt_cycles=15,
        )

    def test_realistic_deadlock_scenario_old_behavior(self) -> None:
        """旧挙動: tick+untick ペアで halt 永続 (デッドロック再現).

        実際に発生したシナリオ:
        - buy PnL: -33.9bps → buy halt 発動 (halt_remaining=15)
        - 3 サイクル正常消費 → halt_remaining=12
        - BTC=0 で sell 不可 → balance_forced → buy に戻される
        - 223# refuse bypass → tick + untick ペア → halt_remaining=12 固定
        - 236 回繰返し → 8 時間以上のデッドロック
        """
        guard = self._make_guard()
        guard.update_pnl(-33.9, side="buy")
        guard.update_pnl(+58.8, side="sell")
        assert guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

        # 3 サイクルの正常消費
        for _ in range(3):
            guard.tick_side_halt()
        assert guard._state.side_halt_remaining_buy == 12

        # 旧挙動: 236 回の tick + untick → halt 永続
        for _ in range(236):
            guard.tick_side_halt()
            guard.untick_side_halt()

        assert guard.is_side_halted("buy"), "旧挙動では halt が永続する"
        assert guard._state.side_halt_remaining_buy == 12

    def test_realistic_deadlock_scenario_new_behavior(self) -> None:
        """新挙動: tick のみ (untick なし) で halt が 12 サイクルで解除.

        281# 修正後の予想される動作:
        - halt_remaining=12 → 12 回の tick → halt 解除
        - reanchor が PnL 基準リセット → 安全に取引再開
        """
        guard = self._make_guard()
        guard.update_pnl(-33.9, side="buy")
        guard.update_pnl(+58.8, side="sell")
        assert guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

        # 3 サイクルの正常消費
        for _ in range(3):
            guard.tick_side_halt()
        assert guard._state.side_halt_remaining_buy == 12

        # 新挙動: 12 回の tick (untick なし) → halt 解除
        for _ in range(12):
            guard.tick_side_halt()

        assert not guard.is_side_halted("buy"), "281# fix: 12 回の tick で halt 解除"
        assert guard._state.side_halt_remaining_buy == 0
        # reanchor が設定されている
        assert guard._state.side_reanchor_pnl_buy == guard._state.daily_pnl_bps_buy

    def test_sell_still_not_halted_during_buy_halt(self) -> None:
        """buy halt 中も sell は halt されない."""
        guard = self._make_guard()
        guard.update_pnl(-33.9, side="buy")
        guard.update_pnl(+58.8, side="sell")
        assert guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

        # tick を進めても sell は影響なし
        for _ in range(15):
            guard.tick_side_halt()
        assert not guard.is_side_halted("sell")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5: untick_side_halt メソッド自体は健在
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUntickMethodStillExists:
    """281#: untick_side_halt() メソッド自体は削除されていない.

    orchestrator からの呼出しは除去されたが、メソッド自体は
    将来の使用のため残存する。
    """

    def test_untick_method_exists(self) -> None:
        """DailyDrawdownGuard.untick_side_halt() が存在する."""
        guard = DailyDrawdownGuard(enabled=True, per_side_enabled=True)
        assert hasattr(guard, "untick_side_halt")
        assert callable(guard.untick_side_halt)

    def test_untick_method_still_functional(self) -> None:
        """untick_side_halt() は通常通り動作する."""
        guard = DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_hard_limit_bps=-5.0,
            per_side_halt_cycles=10,
        )
        guard.update_pnl(-6.0, side="sell")
        guard.tick_side_halt()
        remaining_after_tick = guard._state.side_halt_remaining_sell
        guard.untick_side_halt()
        remaining_after_untick = guard._state.side_halt_remaining_sell
        assert remaining_after_untick == remaining_after_tick + 1
