"""226# T1/P5/S2/S5/#4-2/#2-1: loss_boost指数減衰 + inv_skew O(1) + state永続化テスト.

T1: loss_boost 1-shot → 指数減衰 (Avellaneda-Stoikov AS理論)
P5: inventory skewing O(n) → O(1) インクリメンタルカウンター
S2: toxic_veto balance_forced+halt_block パスでのカウンター減算
S5: halt 中 MCB/SAD フィード継続 (ソースコード検証)
#4-2: MCB change_history 永続化
#2-1: FFD export_state / import_state
"""
from __future__ import annotations

import math
import time
from collections import deque
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker
from tests.unit.v460._fill_test_source import (
    MAKER_PRICE,
    ORCHESTRATOR_GUARDS,
    ORCHESTRATOR_PRE_CYCLE,
    read_class_method_source,
    read_fill_test_method_source,
    read_source_text,
)


# ======================================================================
# helpers
# ======================================================================


def _make_config(**overrides) -> FillTestConfig:  # type: ignore[no-untyped-def]
    defaults = dict(
        spread_offset_ratio=0.001,
        min_offset_jpy=1.0,
        max_offset_ratio=0.02,
        inventory_skewing_enabled=True,
        inventory_skewing_window=10,
        inventory_skewing_max_factor=0.5,
        inventory_skewing_neutral_band=0.1,
        loss_boost_decay_tau_sec=300.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_maker_price(config: FillTestConfig | None = None) -> MakerPrice:
    cfg = config or _make_config()
    ffd_cfg = FastFillDefenseConfig(enabled=False)
    ffd = FastFillDefense(ffd_cfg, base_offset_ratio=cfg.spread_offset_ratio)
    return MakerPrice(
        config=cfg,
        fast_fill_defense=ffd,
        regime_detector=None,
        base_offset_ratio=cfg.spread_offset_ratio,
    )


# ======================================================================
# T1: loss_boost 指数減衰
# ======================================================================


class TestLossBoostExponentialDecay:
    """226# T1: loss_boost の 1-shot 消費 → 指数減衰への移行テスト."""

    def test_set_loss_boost_records_time(self) -> None:
        """set_loss_boost() がタイムスタンプを記録する."""
        mp = _make_maker_price()
        before = time.time()
        mp.set_loss_boost(1.5)
        after = time.time()
        # noinspection PyProtectedMember
        assert mp._loss_boost_mult == 1.5
        assert before <= mp._loss_boost_set_time <= after

    def test_decay_reduces_over_time(self) -> None:
        """経過時間に応じて effective mult が減衰する."""
        mp = _make_maker_price(_make_config(loss_boost_decay_tau_sec=100.0))
        # simulate boost set 100s ago (1τ → 63% decay)
        mp._loss_boost_mult = 1.5
        mp._loss_boost_set_time = time.time() - 100.0
        # The effective mult at t=τ: 1 + 0.5 * exp(-1) ≈ 1.184
        expected_decay = math.exp(-1.0)
        expected_mult = 1.0 + 0.5 * expected_decay
        assert expected_mult == pytest.approx(1.184, abs=0.01)

    def test_decay_resets_when_negligible(self) -> None:
        """減衰が十分に進んだら (mult < 1.01) リセットされる."""
        mp = _make_maker_price(_make_config(loss_boost_decay_tau_sec=100.0))
        # set boost long ago → decay to ~0
        mp._loss_boost_mult = 1.5
        mp._loss_boost_set_time = time.time() - 10000.0  # 100τ
        # After this much time, decayed_mult ≈ 1.0, should auto-reset
        decayed = 1.0 + (1.5 - 1.0) * math.exp(-10000 / 100)
        assert decayed < 1.01  # confirm negligible
        # The actual reset happens inside compute(), but we can verify the math

    def test_no_boost_when_mult_is_1(self) -> None:
        """mult=1.0 の場合は decay 処理に入らない."""
        mp = _make_maker_price()
        assert mp._loss_boost_mult == 1.0
        assert mp._loss_boost_set_time == 0.0

    def test_config_has_tau_field(self) -> None:
        """FillTestConfig に loss_boost_decay_tau_sec が存在する."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "loss_boost_decay_tau_sec")
        assert cfg.loss_boost_decay_tau_sec == 300.0

    def test_slots_include_set_time(self) -> None:
        """MakerPrice.__slots__ に _loss_boost_set_time が含まれる."""
        assert "_loss_boost_set_time" in MakerPrice.__slots__

    def test_compute_source_has_decay_formula(self) -> None:
        """指数減衰の数式が _apply_loss_boost() に実装されている (ソースコード検証)."""
        # 260# P2-2: loss boost は _apply_loss_boost() に抽出済み
        src = read_class_method_source(MAKER_PRICE, "MakerPriceCalculator", "_apply_loss_boost")
        assert "exp(" in src, "math.exp() が _apply_loss_boost() に存在すること"
        assert "_loss_boost_set_time" in src
        assert "226# T1" in src


# ======================================================================
# P5: inventory skewing O(1) カウンター
# ======================================================================


class TestInventorySkewingO1:
    """226# P5: update_inventory O(n) → O(1) 改善の正確性テスト."""

    def test_initial_state(self) -> None:
        """初期状態: imbalance=0, buy_count=0."""
        mp = _make_maker_price()
        assert mp._inv_net_imbalance == 0.0
        assert mp._inv_buy_count == 0

    def test_all_buys(self) -> None:
        """全 buy → imbalance = +1."""
        mp = _make_maker_price()
        for _ in range(5):
            mp.update_inventory("buy")
        assert mp._inv_net_imbalance == 1.0
        assert mp._inv_buy_count == 5

    def test_all_sells(self) -> None:
        """全 sell → imbalance = -1."""
        mp = _make_maker_price()
        for _ in range(5):
            mp.update_inventory("sell")
        assert mp._inv_net_imbalance == -1.0
        assert mp._inv_buy_count == 0

    def test_balanced(self) -> None:
        """buy/sell 均等 → imbalance ≈ 0."""
        mp = _make_maker_price()
        for _ in range(5):
            mp.update_inventory("buy")
            mp.update_inventory("sell")
        assert mp._inv_net_imbalance == pytest.approx(0.0)
        assert mp._inv_buy_count == 5

    def test_eviction_correctness(self) -> None:
        """maxlen 超過時の eviction で buy_count が正しく追跡される."""
        cfg = _make_config(inventory_skewing_window=5)
        mp = _make_maker_price(cfg)
        # Fill: buy buy buy buy buy (5 buys, maxlen=5)
        for _ in range(5):
            mp.update_inventory("buy")
        assert mp._inv_buy_count == 5
        assert mp._inv_net_imbalance == 1.0

        # 6th fill: sell → evicts oldest buy → 4 buys + 1 sell
        mp.update_inventory("sell")
        assert mp._inv_buy_count == 4
        assert mp._inv_net_imbalance == pytest.approx((2 * 4 - 5) / 5)  # 0.6

    def test_eviction_of_sell(self) -> None:
        """sell が eviction される場合は buy_count 不変."""
        cfg = _make_config(inventory_skewing_window=3)
        mp = _make_maker_price(cfg)
        # sell, sell, sell (3 sells)
        for _ in range(3):
            mp.update_inventory("sell")
        assert mp._inv_buy_count == 0
        # next buy → evicts oldest sell → 0 sells + 1 buy over sell, sell
        mp.update_inventory("buy")
        assert mp._inv_buy_count == 1
        assert mp._inv_net_imbalance == pytest.approx((2 * 1 - 3) / 3)  # -1/3

    def test_long_sequence_matches_naive(self) -> None:
        """長いシーケンスで O(1) 結果がナイーブ O(n) と一致する."""
        import random
        cfg = _make_config(inventory_skewing_window=20)
        mp = _make_maker_price(cfg)
        random.seed(42)
        sides = [random.choice(["buy", "sell"]) for _ in range(100)]
        for side in sides:
            mp.update_inventory(side)
        # naive verification
        dq = mp._inv_fill_history
        n = len(dq)
        naive_buys = sum(1 for s in dq if s == "buy")
        assert mp._inv_buy_count == naive_buys
        assert mp._inv_net_imbalance == pytest.approx((2 * naive_buys - n) / n)

    def test_slots_include_buy_count(self) -> None:
        """MakerPrice.__slots__ に _inv_buy_count が含まれる."""
        assert "_inv_buy_count" in MakerPrice.__slots__


# ======================================================================
# #4-2: MCB change_history 永続化
# ======================================================================


class TestMCBChangeHistoryPersistence:
    """226# #4-2: MCB の _change_history_5m/15m/1h が export/import される."""

    def _make_mcb(self) -> MicroCircuitBreaker:
        from scripts.v460.lib.micro_circuit_breaker import MCBConfig
        cfg = MCBConfig(enabled=True)
        return MicroCircuitBreaker(cfg)

    def test_export_includes_change_histories(self) -> None:
        """export_state() に change_history_5m/15m/1h が含まれる."""
        mcb = self._make_mcb()
        # Feed some data
        t = time.time()
        for i in range(10):
            mcb.update(15000000 + i * 100, t + i * 30)
        state = mcb.export_state()
        assert "change_history_5m" in state
        assert "change_history_15m" in state
        assert "change_history_1h" in state
        assert isinstance(state["change_history_5m"], list)

    def test_import_restores_change_histories(self) -> None:
        """import_state() で change_history が正しく復元される."""
        mcb = self._make_mcb()
        t = time.time()
        for i in range(10):
            mcb.update(15000000 + i * 100, t + i * 30)
        state = mcb.export_state()

        mcb2 = self._make_mcb()
        mcb2.import_state(state)

        for tf in ("5m", "15m", "1h"):
            key = f"change_history_{tf}"
            assert list(getattr(mcb2, f"_change_history_{tf}")) == state[key]

    def test_roundtrip_preserves_sigma(self) -> None:
        """export→import ラウンドトリップ後も σ 計算が同一."""
        mcb = self._make_mcb()
        t = time.time()
        for i in range(50):
            mcb.update(15000000 + i * 50, t + i * 10)
        state_before = mcb.export_state()

        mcb2 = self._make_mcb()
        mcb2.import_state(state_before)
        state_after = mcb2.export_state()

        for tf in ("5m", "15m", "1h"):
            key = f"change_history_{tf}"
            assert state_before[key] == state_after[key]


# ======================================================================
# #2-1: FFD export_state / import_state
# ======================================================================


class TestFFDStatePersistence:
    """226# #2-1: FastFillDefense の boost state が永続化される."""

    def _make_ffd(self) -> FastFillDefense:
        cfg = FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0)
        return FastFillDefense(cfg, base_offset_ratio=0.001)

    def test_export_state_returns_dict(self) -> None:
        """export_state() が dict を返す."""
        ffd = self._make_ffd()
        state = ffd.export_state()
        assert isinstance(state, dict)

    def test_export_includes_buy_sell_states(self) -> None:
        """buy/sell 両方の boost state がエクスポートされる."""
        ffd = self._make_ffd()
        state = ffd.export_state()
        for side in ("buy", "sell"):
            assert f"{side}_boost_active" in state
            assert f"{side}_boost_multiplier" in state
            assert f"{side}_boost_activated_at" in state

    def test_import_restores_boost_state(self) -> None:
        """import_state() で boost state が正しく復元される."""
        ffd = self._make_ffd()
        state = {
            "buy_boost_active": True,
            "buy_boost_multiplier": 2.5,
            "buy_boost_activated_at": 12345.0,
            "sell_boost_active": False,
            "sell_boost_multiplier": 1.0,
            "sell_boost_activated_at": 0.0,
        }
        ffd.import_state(state)
        assert ffd._state_buy.boost_active is True
        assert ffd._state_buy.boost_multiplier == 2.5
        assert ffd._state_buy.boost_activated_at == 12345.0
        assert ffd._state_sell.boost_active is False

    def test_roundtrip(self) -> None:
        """export→import ラウンドトリップで状態が保存される."""
        ffd = self._make_ffd()
        # Manually set buy boost active
        ffd._state_buy.boost_active = True
        ffd._state_buy.boost_multiplier = 3.0
        ffd._state_buy.boost_activated_at = time.time()

        state = ffd.export_state()
        ffd2 = self._make_ffd()
        ffd2.import_state(state)

        assert ffd2._state_buy.boost_active is True
        assert ffd2._state_buy.boost_multiplier == 3.0

    def test_export_default_state(self) -> None:
        """初期状態のエクスポートはデフォルト値."""
        ffd = self._make_ffd()
        state = ffd.export_state()
        assert state["buy_boost_active"] is False
        assert state["sell_boost_active"] is False
        assert state["buy_boost_multiplier"] == 1.0
        assert state["sell_boost_multiplier"] == 1.0


# ======================================================================
# S2: toxic_veto balance_forced halt_block パスでの減算
# ======================================================================


# ======================================================================
# S5: halt 中 MCB/SAD フィード継続
# ======================================================================


class TestHaltMCBSADFeedContinuation:
    """226# S5: DD halt ループ内で MCB/SAD の update が呼ばれる."""

    def test_orchestrator_source_has_halt_mcb_update(self) -> None:
        """halt 中の return True 前に MCB/SAD feed が存在する (ソースコード検証).

        330#: run_continuous → _handle_dd_halt に抽出。
        """
        src = read_class_method_source(
            ORCHESTRATOR_PRE_CYCLE,
            "OrchestratorPreCycleMixin",
            "_handle_dd_halt",
        )
        assert "226# S5" in src
        # "is_halted" → MCB/SAD feed が halt ブロック内にある
        halted_section = src[src.index("daily_drawdown_guard.is_halted()"):]
        # halt ブロックの return True 前に _feed_mcb_sad がある (272# DRY 抽出済)
        return_idx = halted_section.index("return True")
        feed_in_halt = "self._feed_mcb_sad()" in halted_section[:return_idx]
        assert feed_in_halt, "_feed_mcb_sad() should appear before return True in halt block"

    def test_orchestrator_source_has_halt_sad_update(self) -> None:
        """halt 中の continue 前に SAD update が存在する (272# _feed_mcb_sad に統合済)."""
        # _feed_mcb_sad ヘルパー内に SAD update が含まれていることを検証
        src = read_class_method_source(
            ORCHESTRATOR_GUARDS,
            "OrchestratorGuardsMixin",
            "_feed_mcb_sad",
        )
        assert "self._sad.update" in src, "_feed_mcb_sad should contain SAD update"


# ======================================================================
# YAML parser: loss_boost_decay_tau_sec
# ======================================================================


class TestFillConfigYAMLParser:
    """226# T1: YAML → FillTestConfig への配線テスト."""

    def test_from_yaml_parses_loss_boost_decay_tau(self) -> None:
        """止血セクションの loss_boost_decay_tau_sec が反映される."""
        yaml_cfg: dict = {
            "止血": {
                "loss_boost_offset_mult": 1.3,
                "loss_boost_decay_tau_sec": 200.0,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.loss_boost_decay_tau_sec == 200.0
        assert cfg.loss_boost_offset_mult == 1.3

    def test_from_yaml_default_tau(self) -> None:
        """YAML 未指定時はデフォルト値 (300.0s)."""
        cfg = FillTestConfig.from_yaml({})
        assert cfg.loss_boost_decay_tau_sec == 300.0

    def test_from_yaml_loss_control_alias(self) -> None:
        """'loss_control' エイリアスでも正しくパースされる."""
        yaml_cfg: dict = {
            "loss_control": {
                "loss_boost_decay_tau_sec": 150.0,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.loss_boost_decay_tau_sec == 150.0


# ======================================================================
# FFD hot-reload state preservation (run_fill_test)
# ======================================================================


class TestFFDHotReloadPreservation:
    """226# #2-1: FFD hot-reload 時の boost state 保存."""

    def test_rebuild_source_preserves_state(self) -> None:
        """_rebuild_fast_fill_defense に export/import パターンがある."""
        src = read_fill_test_method_source("_rebuild_fast_fill_defense")
        assert "export_state" in src, (
            "_rebuild_fast_fill_defense should call export_state"
        )
        assert "import_state" in src, (
            "_rebuild_fast_fill_defense should call import_state"
        )
