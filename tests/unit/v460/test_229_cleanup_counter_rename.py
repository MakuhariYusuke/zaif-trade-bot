"""229# テスト: import整理・hasattr排除・M-5 unknown counter fix・M-2 rename.

変更概要:
  Q1: getattr(self._config, "inv_decay_tau_sec", 0.0) → 直接アクセス
  H-1/Q5: maker_price _apply_regime_boosts 5x hasattr 排除
  H-3: orchestrator getattr(self, "_soft_drawdown_interval_multiplier") → 直接アクセス
  H-4: fast_fill_defense inline import time 排除 (module-level import)
  H-5: orchestrator inline import time 排除
  M-5: unknown regime counter Gate 2-3 early return で reset 漏れ修正
  M-2: get_recovery_lot_scale() → consume_recovery_cycle() rename
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import pytest

from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator
from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from tests.unit.v460._fill_test_source import ORCHESTRATOR_MID_CYCLE, read_source_text
from tests.unit.v460.conftest import (
    make_gate_config as _make_gate_config,
    make_maker_price_config as _make_fill_config,
)

_FAST_FILL_DEFENSE_SOURCE = Path(
    inspect.getsourcefile(FastFillDefense) or "",
).read_text(encoding="utf-8")
_MAKER_PRICE_SOURCE = Path(
    inspect.getsourcefile(MakerPrice) or "",
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gate(**overrides) -> CycleGateAggregator:
    return CycleGateAggregator(_make_gate_config(**overrides))


def _default_ctx(**overrides) -> dict:
    """CycleGateAggregator.evaluate() デフォルト引数."""
    ctx: dict = {
        "side": "buy",
        "regime": "ranging",
        "vol_ratio": 1.0,
        "inv_net_imbalance": 0.0,
        "is_buy_killed": False,
        "is_sell_killed": False,
    }
    ctx.update(overrides)
    return ctx


def _make_mp(config: FillTestConfig | None = None, regime=None) -> MakerPrice:
    """テスト用 MakerPriceCalculator."""
    if config is None:
        config = _make_fill_config()
    ffd = FastFillDefense(
        FastFillDefenseConfig(enabled=False, threshold_sec=60.0, offset_boost=1.2),
        base_offset_ratio=0.005,
    )
    return MakerPrice(
        config=config,
        fast_fill_defense=ffd,
        regime_detector=regime,
        base_offset_ratio=0.005,
    )


# ===========================================================================
# H-4: FastFillDefense module-level import time
# ===========================================================================

class TestFFDModuleLevelImport:
    """H-4: fast_fill_defense.py に inline import time が残っていないことを検証."""

    def test_no_inline_import_time(self):
        """get_boost_multiplier / evaluate_fill に inline import time がない."""
        # module-level "import time" は OK だが "import time as _time" は NG
        assert "import time as _time" not in _FAST_FILL_DEFENSE_SOURCE

    def test_time_in_module_globals(self):
        """time が module-level でインポートされている."""
        import scripts.v460.lib.fast_fill_defense as ffd_mod
        assert hasattr(ffd_mod, "time")
        assert ffd_mod.time is time

    def test_get_boost_multiplier_uses_time_time(self):
        """TTL decay で time.time() を正しく使用.

        236# CQS 分離: maybe_expire_boost() で TTL decay を先に実行後、
        get_boost_multiplier() は純粋 getter として boost 値を返す。
        """
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=1.0,
            offset_boost=2.0, boost_ttl_sec=1.0,  # TTL=1sec
        )
        ffd = FastFillDefense(cfg, base_offset_ratio=0.005)
        state = ffd._get_state("buy")
        state.boost_active = True
        state.boost_multiplier = 2.0
        state.boost_activated_at = time.time() - 2.0  # 2秒前 > TTL 1秒 → expired
        # 236# maybe_expire_boost で TTL decay → その後 getter は 1.0
        ffd.maybe_expire_boost("buy")
        assert ffd.get_boost_multiplier("buy") == 1.0


# ===========================================================================
# H-1/Q5: maker_price hasattr removal
# ===========================================================================

class TestMakerPriceNoHasattr:
    """H-1/Q5: maker_price._apply_regime_boosts に hasattr が残っていない."""

    def test_no_hasattr_in_apply_regime_boosts(self):
        """_apply_regime_boosts のソースに hasattr が含まれない."""
        src = inspect.getsource(MakerPrice._apply_regime_boosts)
        assert "hasattr" not in src

    def test_regime_detector_none_no_error(self):
        """regime_detector=None で _apply_regime_boosts がエラーなく動作."""
        mp = _make_mp(regime=None)
        result = mp._apply_regime_boosts("buy", 0.01)
        assert result == 0.01  # regime_detector=None → 変更なし

    def test_regime_detector_with_current_regime(self):
        """regime_detector が current_regime を持つ場合正しく動作."""
        from scripts.v460.lib.regime_detector import FillTestRegime
        from unittest.mock import MagicMock
        rd = MagicMock()
        rd.current_regime = FillTestRegime.RANGING
        rd.last_volatility_ratio = 1.0
        config = _make_fill_config(regime_ranging_offset_discount=0.8)
        mp = _make_mp(config=config, regime=rd)
        result = mp._apply_regime_boosts("buy", 0.01)
        assert result < 0.01  # ranging discount applied


# ===========================================================================
# Q1: inv_decay_tau_sec direct access
# ===========================================================================

class TestInvDecayTauDirectAccess:
    """Q1: getattr → self._config.inv_decay_tau_sec 直接アクセス."""

    def test_config_has_inv_decay_tau_sec_attr(self):
        """FillTestConfig に inv_decay_tau_sec 属性が存在する."""
        cfg = _make_fill_config()
        assert hasattr(cfg, "inv_decay_tau_sec")

    def test_default_value(self):
        """344#: デフォルト値は 1800.0 (30分減衰)."""
        cfg = _make_fill_config()
        assert cfg.inv_decay_tau_sec == 1800.0

    def test_maker_price_source_no_getattr_inv_decay(self):
        """maker_price に getattr(..., "inv_decay_tau_sec", ...) が残っていない."""
        assert (
            'getattr' not in _MAKER_PRICE_SOURCE
            or 'inv_decay_tau_sec' not in _MAKER_PRICE_SOURCE.split('getattr')[1]
        ) if 'getattr' in _MAKER_PRICE_SOURCE else True

    def test_decayed_imbalance_with_tau(self):
        """inv_decay_tau_sec を設定して _decayed_imbalance が減衰を適用する."""
        import math
        cfg = _make_fill_config(inv_decay_tau_sec=60.0)
        mp = _make_mp(config=cfg)
        mp._inv_net_imbalance = 1.0
        mp._inv_last_update_time = time.time() - 60.0  # 1τ 経過
        decayed = mp._decayed_imbalance(time.time())
        expected = 1.0 * math.exp(-1.0)
        assert abs(decayed - expected) < 0.01


# ===========================================================================
# H-3: orchestrator _soft_drawdown_interval_multiplier direct access
# ===========================================================================

class TestOrchestratorNoGetattr:
    """H-3: orchestrator に getattr(self, "_soft_drawdown_interval_multiplier") が残っていない."""

    def test_no_getattr_soft_drawdown(self):
        """fill_loop_orchestrator のソースに getattr(self, '_soft_drawdown_interval_multiplier' がない."""
        import scripts.v460.lib.fill_loop_orchestrator as orch_mod
        src = inspect.getsource(orch_mod)
        assert 'getattr(self, "_soft_drawdown_interval_multiplier"' not in src

    def test_no_inline_import_time_as_time(self):
        """H-5: fill_loop_orchestrator に inline import time as _time が残っていない."""
        import scripts.v460.lib.fill_loop_orchestrator as orch_mod
        src = inspect.getsource(orch_mod)
        assert "import time as _time" not in src


# ===========================================================================
# M-5: unknown regime counter reset bug fix
# ===========================================================================

class TestUnknownCounterResetOnGate2:
    """M-5: Gate 2/3 early return 時に _consecutive_unknown_blocks がリセットされる."""

    def test_ranging_gate2_resets_counter(self):
        """unknown→ranging(Gate 2 block)→unknown: カウンタは1からリスタート.

        Bug (修正前): unknown 2回ブロック → ranging Gate2 ブロック → unknown 1回ブロック
        → counter=3 (本来は1) → 偽のバイパス発動リスク.
        """
        gate = _make_gate()

        # Step 1: unknown regime で Gate 1 ブロック (buy side, skip_buy_unknown=True)
        r1 = gate.evaluate(**_default_ctx(side="buy", regime="unknown", vol_ratio=1.0))
        assert r1.blocked
        assert gate._consecutive_unknown_blocks["buy"] == 1

        # Step 2: もう1回 unknown
        r2 = gate.evaluate(**_default_ctx(side="buy", regime="unknown", vol_ratio=1.0))
        assert r2.blocked
        assert gate._consecutive_unknown_blocks["buy"] == 2

        # Step 3: regime=ranging, vol_ratio=0.5 (< threshold=0.75) → Gate 2 ブロック
        r3 = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=0.5,
        ))
        assert r3.blocked
        assert r3.blocking_reason == "ranging_low_vol_skip"
        # 229# M-5 fix: カウンタは 0 にリセットされているはず
        assert gate._consecutive_unknown_blocks["buy"] == 0

        # Step 4: unknown に戻る → カウンタは 1 から (修正前は 3 だった)
        r4 = gate.evaluate(**_default_ctx(side="buy", regime="unknown", vol_ratio=1.0))
        assert r4.blocked
        assert gate._consecutive_unknown_blocks["buy"] == 1

    def test_trending_gate3_resets_counter(self):
        """unknown→trending(Gate 3 block)→unknown: カウンタリセット確認."""
        gate = _make_gate()

        # Step 1: unknown ブロック
        gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 1

        # Step 2: trending_up → sell → Gate 3 ブロック (trending_sell_skip)
        r2 = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
            inv_net_imbalance=0.0,  # bypass threshold 未満
        ))
        assert r2.blocked
        assert r2.blocking_reason == "trending_sell_skip"
        # 229# M-5: カウンタリセット
        assert gate._consecutive_unknown_blocks["sell"] == 0

    def test_unknown_blocked_counter_increments_with_balance_forced(self):
        """234#: unknown + balance_forced → Gate 1 ブロック → カウンタ増加."""
        gate = _make_gate()

        # Step 1: unknown ブロック
        gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 1

        # Step 2: 234# balance_forced でも Gate 1 はブロック → カウンタ増加
        r2 = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown",
        ))
        assert r2.blocked
        assert gate._consecutive_unknown_blocks["buy"] == 2

    def test_non_unknown_always_resets(self):
        """non-unknown regime は Gate 1 通過時にカウンタリセット."""
        gate = _make_gate()
        gate._consecutive_unknown_blocks["sell"] = 5  # 手動設定

        # ranging, vol_ratio=1.0 (Gate 2 通過) → カウンタリセット
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging", vol_ratio=1.0,
        ))
        assert gate._consecutive_unknown_blocks["sell"] == 0

    def test_bypass_threshold_not_falsely_triggered(self):
        """unknown→ranging→unknown の遷移でバイパス閾値に偽到達しない.

        UNKNOWN_REGIME_MAX_CONSECUTIVE=5 に対し、
        unknown 3回 → ranging 1回(Gate 2 block) → unknown 3回 で
        合計6回にならないことを確認。
        """
        gate = _make_gate()
        assert gate.UNKNOWN_REGIME_MAX_CONSECUTIVE == 5

        # unknown 3回
        for _ in range(3):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 3

        # ranging で Gate 2 ブロック → リセット
        gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=0.5,
        ))
        assert gate._consecutive_unknown_blocks["buy"] == 0

        # unknown 3回 → カウンタは 3 (MAX=5 に到達しない)
        for _ in range(3):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 3


# ===========================================================================
# M-2: get_recovery_lot_scale → consume_recovery_cycle rename
# ===========================================================================

class TestConsumeRecoveryCycleRename:
    """M-2: 副作用のある getter を consume_ に改名."""

    def test_method_exists(self):
        """consume_recovery_cycle がメソッドとして存在する."""
        assert hasattr(DailyDrawdownGuard, "consume_recovery_cycle")

    def test_old_name_removed(self):
        """get_recovery_lot_scale はもう存在しない."""
        assert not hasattr(DailyDrawdownGuard, "get_recovery_lot_scale")

    def test_consume_decrements_counter(self):
        """consume_recovery_cycle が残カウンタをデクリメントする."""
        guard = DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_halt_cycles=2,
            per_side_recovery_cycles=3,
            per_side_recovery_lot_scale=0.5,
        )
        # リカバリ状態を手動セット
        guard._state.side_recovery_remaining_buy = 3

        scale1 = guard.consume_recovery_cycle("buy")
        assert scale1 == 0.5
        assert guard._state.side_recovery_remaining_buy == 2

        scale2 = guard.consume_recovery_cycle("buy")
        assert scale2 == 0.5
        assert guard._state.side_recovery_remaining_buy == 1

        scale3 = guard.consume_recovery_cycle("buy")
        assert scale3 == 0.5
        assert guard._state.side_recovery_remaining_buy == 0

        # 残カウンタ = 0 → 1.0 (通常)
        scale4 = guard.consume_recovery_cycle("buy")
        assert scale4 == 1.0

    def test_restore_recovery_counter_docstring_updated(self):
        """restore_recovery_counter の docstring が新メソッド名を参照."""
        doc = DailyDrawdownGuard.restore_recovery_counter.__doc__
        assert "consume_recovery_cycle" in doc
        assert "get_recovery_lot_scale" not in doc

    def test_orchestrator_calls_new_name(self):
        """orchestrator_mid_cycle のソースが consume_recovery_cycle を呼んでいる."""
        src = read_source_text(ORCHESTRATOR_MID_CYCLE)
        assert "consume_recovery_cycle" in src
        assert "get_recovery_lot_scale" not in src


# ===========================================================================
# Regression: 既存テストとの整合性
# ===========================================================================

class TestRegressionGuards:
    """229# 変更が既存機能を壊していないことを確認."""

    def test_ffd_evaluate_fill_works(self):
        """FFD evaluate_fill が import time 変更後も正常動作."""
        cfg = FastFillDefenseConfig(
            enabled=True,
            threshold_sec=5.0,
            offset_boost=2.0,
        )
        ffd = FastFillDefense(cfg, base_offset_ratio=0.005)
        # 通常の fill (fast fill ではない)
        ffd.evaluate_fill(
            side="buy",
            queue_wait_sec=10.0,
            fill_price=14_000_000,
            mid_at_fill=14_000_000,
        )
        assert not ffd.is_boost_active("buy")

    def test_maker_price_regime_detector_none(self):
        """regime_detector=None で MakerPrice が正常初期化される."""
        mp = _make_mp(regime=None)
        assert mp._regime_detector is None

    def test_cycle_gate_full_pass(self):
        """全ゲート通過の基本パスが壊れていない."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging", vol_ratio=1.0,
        ))
        assert not r.blocked
