"""218# DynamicKillManager anti-stagnation (probe cycle) tests."""
from __future__ import annotations

import pytest

from ztb.risk.sell_dynamic_kill import (
    BuyDynamicKillManager,
    DynamicKillConfig,
    DynamicKillManager,
)


class TestAntiStagnation218:
    """218# anti-stagnation: stale kill 時のプローブサイクル."""

    def _make_kill_manager(
        self,
        threshold_bps: float = -0.5,
        window: int = 5,
        resume_window: int = 3,
        max_stale_kill_cycles: int = 10,
    ) -> DynamicKillManager:
        return DynamicKillManager(
            DynamicKillConfig(
                enabled=True,
                window=window,
                threshold_bps=threshold_bps,
                resume_window=resume_window,
                max_stale_kill_cycles=max_stale_kill_cycles,
            ),
            side="buy",
        )

    def test_probe_fires_after_max_stale_cycles(self) -> None:
        """stale counter が max_stale_kill_cycles に達するとプローブが発動."""
        mgr = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=3, max_stale_kill_cycles=10
        )
        # window 分のデータを投入 (mean = -1.0 < -0.5)
        for _ in range(5):
            mgr.track(-1.0)

        # 最初の check_kill: kill 発動 (cooldown=3 セット)
        killed, tel = mgr.check_kill()
        assert killed
        assert tel.cooldown_remaining == 3

        # cooldown 3 サイクル消化 + 再発動 → stale counter が溜まる
        # check_kill を max_stale_kill_cycles 回呼ぶ (track なし)
        kill_count = 0
        probe_happened = False
        for i in range(30):
            killed, tel = mgr.check_kill()
            if not killed:
                probe_happened = True
                break
            kill_count += 1

        assert probe_happened, "probe should fire within 30 cycles"
        # probe は max_stale_kill_cycles (10) 前後で発動するはず
        assert kill_count <= 15, f"probe took too long: {kill_count} cycles"

    def test_track_resets_stale_counter(self) -> None:
        """track() 呼び出しで stale counter がリセットされる."""
        mgr = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=2, max_stale_kill_cycles=5
        )
        for _ in range(5):
            mgr.track(-1.0)

        # kill 発動
        killed, _ = mgr.check_kill()
        assert killed

        # cooldown 2 サイクル消化
        mgr.check_kill()
        mgr.check_kill()

        # ここで track → stale リセット
        mgr.track(-0.1)

        # 再度 kill 発動 (mean はまだ < threshold)
        killed, _ = mgr.check_kill()
        assert killed

        # stale counter はリセットされているので、直後にはまだ probe されない
        kills_without_probe = 0
        for _ in range(4):
            killed, _ = mgr.check_kill()
            if killed:
                kills_without_probe += 1
            else:
                break
        # stale counter リセット後は 4 回以内にはまだ probe されない (resume_window + kill)
        assert kills_without_probe >= 3, f"expected at least 3 kills after reset, got {kills_without_probe}"

    def test_max_stale_zero_disables_probe(self) -> None:
        """max_stale_kill_cycles=0 でプローブ無効 (従来互換)."""
        mgr = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=2, max_stale_kill_cycles=0
        )
        for _ in range(5):
            mgr.track(-1.0)

        # 永遠に kill が続く (probe なし)
        for _ in range(50):
            killed, _ = mgr.check_kill()
            assert killed, "with max_stale=0, kill should persist indefinitely"

    def test_probe_allows_single_cycle_then_re_kills(self) -> None:
        """プローブ後にデータ投入がなければ次回再度 kill."""
        mgr = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=2, max_stale_kill_cycles=5
        )
        for _ in range(5):
            mgr.track(-1.0)

        # kill 発動 → stale まで進める
        probe_idx = None
        for i in range(20):
            killed, _ = mgr.check_kill()
            if not killed:
                probe_idx = i
                break

        assert probe_idx is not None, "probe should fire"

        # probe 後、track なしで check_kill → 再度 kill
        killed, _ = mgr.check_kill()
        assert killed, "should re-kill after probe without new data"

    def test_export_import_preserves_stale_counter(self) -> None:
        """export/import で stale_counter と total_probe_cycles が保存・復元."""
        mgr = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=2, max_stale_kill_cycles=5
        )
        for _ in range(5):
            mgr.track(-1.0)

        # kill → stale 進行
        for _ in range(3):
            mgr.check_kill()

        state = mgr.export_state()
        assert "stale_counter" in state
        assert "total_probe_cycles" in state

        # 新しいマネージャに復元
        mgr2 = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=2, max_stale_kill_cycles=5
        )
        mgr2.import_state(state)

        # stale counter が引き継がれている → あと 2 サイクルで probe
        kills = 0
        for _ in range(10):
            killed, _ = mgr2.check_kill()
            if not killed:
                break
            kills += 1
        # stale_counter=3 で復元、max=5 なので 2 回以内にprobe
        assert kills <= 4, f"expected probe shortly after import, got {kills} kills"

    def test_buy_dynamic_kill_manager_inherits_anti_stagnation(self) -> None:
        """BuyDynamicKillManager でもプローブが機能する."""
        mgr = BuyDynamicKillManager(
            DynamicKillConfig(
                enabled=True,
                window=5,
                threshold_bps=-0.5,
                resume_window=2,
                max_stale_kill_cycles=5,
            )
        )
        for _ in range(5):
            mgr.track(-1.0)

        probe_found = False
        for _ in range(20):
            killed, _ = mgr.check_kill()
            if not killed:
                probe_found = True
                break
        assert probe_found

    def test_total_probe_cycles_increments(self) -> None:
        """プローブ発動ごとに total_probe_cycles がインクリメント."""
        mgr = self._make_kill_manager(
            threshold_bps=-0.5, window=5, resume_window=1, max_stale_kill_cycles=3
        )
        for _ in range(5):
            mgr.track(-1.0)

        probes = 0
        for _ in range(30):
            killed, _ = mgr.check_kill()
            if not killed:
                probes += 1

        assert probes >= 2, "should see multiple probes"
        state = mgr.export_state()
        assert state["total_probe_cycles"] == probes
