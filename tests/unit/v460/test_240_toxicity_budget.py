"""240# Toxicity Budget テスト (232# §2.2 Glosten-Milgrom).

DynamicKillManager.assess_toxicity() の正規化スコア計算、
ToxicityLevel 判定、CycleGateAggregator の段階的応答、
Orchestrator の participation_rate チェック、
Executor の toxicity_offset_mult 適用を検証する。
"""

from __future__ import annotations

import inspect
import re
import textwrap
from pathlib import Path

import pytest
from tests.unit.v460._fill_test_source import ORCHESTRATOR_MID_CYCLE, read_source_text

from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from ztb.risk.sell_dynamic_kill import (
    BuyDynamicKillManager,
    DynamicKillConfig,
    DynamicKillManager,
)
from ztb.risk.toxicity_types import ToxicityAssessment, ToxicityLevel

_FILL_LOOP_ORCHESTRATOR_SOURCE = Path(
    "scripts/v460/lib/fill_loop_orchestrator.py",
).read_text(encoding="utf-8")
_FILL_CYCLE_EXECUTOR_SOURCE = Path(
    "scripts/v460/lib/fill_cycle_executor.py",
).read_text(encoding="utf-8")
_OFFSET_PIPELINE_SOURCE = Path(
    "scripts/v460/lib/offset_pipeline.py",
).read_text(encoding="utf-8")
_ORCHESTRATOR_MID_CYCLE_SOURCE = read_source_text(ORCHESTRATOR_MID_CYCLE)
_RUN_SINGLE_CYCLE_SIG = inspect.signature(FillCycleExecutorMixin.run_single_cycle)


# ═══════════════════════════════════════════════════════
# 1. ToxicityLevel / ToxicityAssessment 基盤テスト
# ═══════════════════════════════════════════════════════


class TestToxicityLevel:
    """ToxicityLevel enum の基本属性."""

    def test_has_four_levels(self) -> None:
        assert len(ToxicityLevel) == 4

    def test_level_values(self) -> None:
        assert ToxicityLevel.GREEN.value == "green"
        assert ToxicityLevel.YELLOW.value == "yellow"
        assert ToxicityLevel.ORANGE.value == "orange"
        assert ToxicityLevel.KILL.value == "kill"


class TestToxicityAssessment:
    """ToxicityAssessment dataclass の基本属性."""

    def test_is_frozen(self) -> None:
        a = ToxicityAssessment(
            level=ToxicityLevel.GREEN, score=0.0,
            offset_mult=1.0, participation_rate=1.0,
            threshold_used=-0.5, rolling_mean=None,
        )
        with pytest.raises(AttributeError):
            a.score = 0.5  # type: ignore[misc]

    def test_has_slots(self) -> None:
        assert hasattr(ToxicityAssessment, "__slots__")

    def test_green_defaults(self) -> None:
        a = ToxicityAssessment(
            level=ToxicityLevel.GREEN, score=0.0,
            offset_mult=1.0, participation_rate=1.0,
            threshold_used=-0.5, rolling_mean=0.1,
        )
        assert a.level == ToxicityLevel.GREEN
        assert a.offset_mult == 1.0
        assert a.participation_rate == 1.0


# ═══════════════════════════════════════════════════════
# 2. DynamicKillConfig — Toxicity Budget フィールド
# ═══════════════════════════════════════════════════════


class TestDynamicKillConfigToxicity:
    """DynamicKillConfig の toxicity budget フィールド検証."""

    def test_toxicity_fields_exist(self) -> None:
        cfg = DynamicKillConfig()
        assert hasattr(cfg, "toxicity_budget_enabled")
        assert hasattr(cfg, "toxicity_warn_level")
        assert hasattr(cfg, "toxicity_caution_level")
        assert hasattr(cfg, "toxicity_warn_offset_mult")
        assert hasattr(cfg, "toxicity_caution_offset_mult")
        assert hasattr(cfg, "toxicity_kill_offset_mult")
        assert hasattr(cfg, "toxicity_caution_min_participation")

    def test_default_disabled(self) -> None:
        """デフォルトでは toxicity budget は無効."""
        cfg = DynamicKillConfig()
        assert cfg.toxicity_budget_enabled is False

    def test_zone_ordering(self) -> None:
        """warn_level < caution_level (ゾーン境界の順序)."""
        cfg = DynamicKillConfig()
        assert cfg.toxicity_warn_level < cfg.toxicity_caution_level


# ═══════════════════════════════════════════════════════
# 3. assess_toxicity() — 正規化スコアとゾーン判定
# ═══════════════════════════════════════════════════════


def _make_mgr(
    *,
    threshold: float = -0.5,
    window: int = 5,
    warn: float = 0.3,
    caution: float = 0.7,
    warn_offset: float = 1.0,
    caution_offset: float = 2.0,
    kill_offset: float = 3.0,
    min_participation: float = 0.33,
    side: str = "sell",
) -> DynamicKillManager:
    """テスト用 DynamicKillManager を構築."""
    cfg = DynamicKillConfig(
        enabled=True,
        window=window,
        threshold_bps=threshold,
        resume_window=5,
        toxicity_budget_enabled=True,
        toxicity_warn_level=warn,
        toxicity_caution_level=caution,
        toxicity_warn_offset_mult=warn_offset,
        toxicity_caution_offset_mult=caution_offset,
        toxicity_kill_offset_mult=kill_offset,
        toxicity_caution_min_participation=min_participation,
    )
    return DynamicKillManager(cfg, side=side)


class TestAssessToxicityGreen:
    """GREEN ゾーンの検証."""

    def test_insufficient_data_returns_green(self) -> None:
        mgr = _make_mgr(window=10)
        for _ in range(5):
            mgr.track(0.0)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.GREEN
        assert a.score == 0.0

    def test_positive_rolling_mean_returns_green(self) -> None:
        mgr = _make_mgr()
        for _ in range(5):
            mgr.track(1.0)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.GREEN
        assert a.score == 0.0
        assert a.offset_mult == 1.0
        assert a.participation_rate == 1.0

    def test_budget_disabled_returns_green(self) -> None:
        cfg = DynamicKillConfig(
            enabled=True, window=5, threshold_bps=-0.5,
            toxicity_budget_enabled=False,
        )
        mgr = DynamicKillManager(cfg, side="sell")
        for _ in range(5):
            mgr.track(-0.4)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.GREEN

    def test_slightly_negative_stays_green(self) -> None:
        """score < warn_level (0.3) → GREEN."""
        mgr = _make_mgr(threshold=-1.0, window=5)
        # rolling_mean = -0.2 → score = -0.2 / -1.0 = 0.2 < 0.3
        for _ in range(5):
            mgr.track(-0.2)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.GREEN
        assert abs(a.score - 0.2) < 1e-6


class TestAssessToxicityYellow:
    """YELLOW ゾーン (warn_level ≤ score < caution_level) の検証."""

    def test_yellow_zone_entry(self) -> None:
        mgr = _make_mgr(threshold=-1.0, window=5)
        # rolling_mean = -0.3 → score = 0.3 (exactly warn_level)
        for _ in range(5):
            mgr.track(-0.3)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.YELLOW
        assert a.participation_rate == 1.0  # YELLOW has full participation
        assert a.offset_mult >= 1.0

    def test_yellow_zone_mid(self) -> None:
        mgr = _make_mgr(threshold=-1.0, window=5)
        # rolling_mean = -0.5 → score = 0.5 (mid of YELLOW: 0.3-0.7)
        for _ in range(5):
            mgr.track(-0.5)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.YELLOW
        assert a.participation_rate == 1.0
        # offset_mult should be between warn_offset (1.0) and caution_offset (2.0)
        assert 1.0 <= a.offset_mult <= 2.0

    def test_yellow_offset_linear_interpolation(self) -> None:
        """warn→caution 間でoffset_multが線形補間される."""
        mgr = _make_mgr(
            threshold=-1.0, window=5,
            warn=0.3, caution=0.7,
            warn_offset=1.0, caution_offset=2.0,
        )
        # score = 0.5 → t = (0.5 - 0.3) / (0.7 - 0.3) = 0.5
        # offset_mult = 1.0 + 0.5 * (2.0 - 1.0) = 1.5
        for _ in range(5):
            mgr.track(-0.5)
        a = mgr.assess_toxicity()
        assert abs(a.offset_mult - 1.5) < 1e-6


class TestAssessToxicityOrange:
    """ORANGE ゾーン (caution_level ≤ score < 1.0) の検証."""

    def test_orange_zone_entry(self) -> None:
        mgr = _make_mgr(threshold=-1.0, window=5)
        # rolling_mean = -0.7 → score = 0.7 (exactly caution_level)
        for _ in range(5):
            mgr.track(-0.7)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.ORANGE
        assert a.participation_rate <= 1.0
        assert a.offset_mult >= 2.0

    def test_orange_zone_mid(self) -> None:
        mgr = _make_mgr(
            threshold=-1.0, window=5,
            caution=0.7, caution_offset=2.0, kill_offset=3.0,
            min_participation=0.33,
        )
        # rolling_mean = -0.85 → score = 0.85
        # t = (0.85 - 0.7) / (1.0 - 0.7) = 0.5
        # participation = 1.0 - 0.5 * (1.0 - 0.33) = 1.0 - 0.335 = 0.665
        # offset = 2.0 + 0.5 * (3.0 - 2.0) = 2.5
        for _ in range(5):
            mgr.track(-0.85)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.ORANGE
        assert abs(a.offset_mult - 2.5) < 1e-6
        assert abs(a.participation_rate - 0.665) < 1e-6

    def test_orange_never_below_min_participation(self) -> None:
        """score → 1.0 でも participation >= min_participation."""
        mgr = _make_mgr(threshold=-1.0, window=5, min_participation=0.33)
        # rolling_mean = -0.99 → score = 0.99
        for _ in range(5):
            mgr.track(-0.99)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.ORANGE
        assert a.participation_rate >= 0.33


class TestAssessToxicityKill:
    """KILL ゾーン (score ≥ 1.0) の検証."""

    def test_kill_zone(self) -> None:
        mgr = _make_mgr(threshold=-0.5, window=5)
        # rolling_mean = -0.5 → score = 1.0
        for _ in range(5):
            mgr.track(-0.5)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.KILL
        assert a.participation_rate == 0.0
        assert a.offset_mult == 3.0

    def test_kill_beyond_threshold(self) -> None:
        mgr = _make_mgr(threshold=-0.5, window=5)
        # rolling_mean = -1.0 → score = 2.0 (超過)
        for _ in range(5):
            mgr.track(-1.0)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.KILL
        assert a.score >= 1.0

    def test_cooldown_returns_kill(self) -> None:
        """cooldown 中は check_kill() 相当で KILL を返す."""
        mgr = _make_mgr(threshold=-0.5, window=5)
        # kill を発動させて cooldown に入れる
        for _ in range(5):
            mgr.track(-1.0)
        mgr.check_kill()  # cooldown 開始
        # rolling_mean が回復しても cooldown 中は KILL
        for _ in range(5):
            mgr.track(1.0)
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.KILL


class TestAssessToxicityRegime:
    """レジーム別閾値の動作検証."""

    def test_regime_threshold_changes_score(self) -> None:
        cfg = DynamicKillConfig(
            enabled=True, window=5, threshold_bps=-0.5,
            regime_thresholds={"trending": -0.3},
            toxicity_budget_enabled=True,
        )
        mgr = DynamicKillManager(cfg, side="sell")
        for _ in range(5):
            mgr.track(-0.21)  # default: -0.21/-0.5=0.42, trending: -0.21/-0.3=0.7

        a_default = mgr.assess_toxicity(regime=None)
        a_trending = mgr.assess_toxicity(regime="trending")

        # default では YELLOW、trending では ORANGE (caution_level=0.7)
        assert a_default.score < a_trending.score
        assert a_trending.level == ToxicityLevel.ORANGE


class TestAssessToxicityNoSideEffect:
    """assess_toxicity() は副作用を持たない."""

    def test_no_state_mutation(self) -> None:
        mgr = _make_mgr(threshold=-0.5, window=5)
        for _ in range(5):
            mgr.track(-0.4)

        # assess を複数回呼んでも状態は変わらない
        a1 = mgr.assess_toxicity()
        a2 = mgr.assess_toxicity()
        assert a1.score == a2.score
        assert a1.level == a2.level

    def test_does_not_affect_check_kill(self) -> None:
        """assess_toxicity() は check_kill() の cooldown に影響しない."""
        mgr = _make_mgr(threshold=-0.5, window=5)
        for _ in range(5):
            mgr.track(-1.0)

        # assess を先に呼ぶ
        mgr.assess_toxicity()

        # check_kill はまだ正常に動作する (cooldown 未消費)
        killed, _ = mgr.check_kill()
        assert killed


# ═══════════════════════════════════════════════════════
# 4. CycleGateAggregator — 段階的応答
# ═══════════════════════════════════════════════════════


class TestGateAggregatorToxicityFields:
    """CycleGateResult の toxicity フィールド検証."""

    def test_default_values(self) -> None:
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult

        r = CycleGateResult()
        assert r.toxicity_offset_mult == 1.0
        assert r.participation_rate == 1.0

    def test_participation_skip_in_cancel_reasons(self) -> None:
        from scripts.v460.lib.cycle_gate_aggregator import _GATE_TO_CANCEL_REASON

        assert "toxicity_participation_skip" in _GATE_TO_CANCEL_REASON


class TestGateAggregatorGradedResponse:
    """Gate 4/5 で YELLOW/ORANGE の段階的応答が適用される検証.

    241# C-1 fix: 段階的応答は gate 非 block 時 (pre-kill ゾーン) に適用.
    check_kill()=False (not killed) だが assess_toxicity()=YELLOW/ORANGE のとき、
    offset 拡大 + 参加率制限が CycleGateResult に反映される。
    """

    def _make_gate(self) -> object:
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = self._make_config()
        return CycleGateAggregator(cfg)

    def _make_config(self) -> object:
        """最小構成のFillTestConfig."""
        from scripts.v460.lib.fill_config import FillTestConfig

        return FillTestConfig(
            buy_dynamic_kill_enabled=True,
            sell_dynamic_kill_enabled=True,
            degraded_liquidation_enabled=False,
        )

    def test_yellow_buy_applies_offset(self) -> None:
        """YELLOW toxicity (pre-kill) → is_buy_killed=False + offset 適用."""
        gate = self._make_gate()
        yellow = ToxicityAssessment(
            level=ToxicityLevel.YELLOW, score=0.5,
            offset_mult=1.5, participation_rate=1.0,
            threshold_used=-0.5, rolling_mean=-0.25,
        )
        result = gate.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0, inv_net_imbalance=0.0,
            is_buy_killed=False, is_sell_killed=False,
            buy_toxicity=yellow,
        )
        assert not result.blocked
        assert result.toxicity_offset_mult == 1.5
        assert result.participation_rate == 1.0

    def test_orange_sell_reduces_participation(self) -> None:
        """ORANGE toxicity (pre-kill) → is_sell_killed=False + participation < 1."""
        gate = self._make_gate()
        orange = ToxicityAssessment(
            level=ToxicityLevel.ORANGE, score=0.85,
            offset_mult=2.5, participation_rate=0.5,
            threshold_used=-0.5, rolling_mean=-0.425,
        )
        result = gate.evaluate(
            side="sell", regime="ranging", vol_ratio=1.0, inv_net_imbalance=0.0,
            is_buy_killed=False, is_sell_killed=False,
            sell_toxicity=orange,
        )
        assert not result.blocked
        assert result.toxicity_offset_mult == 2.5
        assert result.participation_rate == 0.5

    def test_kill_still_blocks(self) -> None:
        """is_buy_killed=True → gate blocked (KILL 時は binary kill 優先)."""
        gate = self._make_gate()
        kill = ToxicityAssessment(
            level=ToxicityLevel.KILL, score=1.2,
            offset_mult=3.0, participation_rate=0.0,
            threshold_used=-0.5, rolling_mean=-0.6,
        )
        result = gate.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0, inv_net_imbalance=0.0,
            is_buy_killed=True, is_sell_killed=False,
            buy_toxicity=kill,
        )
        assert result.blocked
        assert result.blocking_reason == "buy_dynamic_kill"

    def test_no_toxicity_falls_through_to_legacy(self) -> None:
        """toxicity=None → 従来の blocked 動作."""
        gate = self._make_gate()
        result = gate.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0, inv_net_imbalance=0.0,
            is_buy_killed=True, is_sell_killed=False,
            buy_toxicity=None,
        )
        assert result.blocked

    def test_green_toxicity_no_effect(self) -> None:
        """241# GREEN toxicity → offset/participation 変更なし."""
        gate = self._make_gate()
        green = ToxicityAssessment(
            level=ToxicityLevel.GREEN, score=0.1,
            offset_mult=1.0, participation_rate=1.0,
            threshold_used=-0.5, rolling_mean=-0.05,
        )
        result = gate.evaluate(
            side="buy", regime="ranging", vol_ratio=1.0, inv_net_imbalance=0.0,
            is_buy_killed=False, is_sell_killed=False,
            buy_toxicity=green,
        )
        assert not result.blocked
        assert result.toxicity_offset_mult == 1.0
        assert result.participation_rate == 1.0




# ═══════════════════════════════════════════════════════
# 5. cancel_reasons 定数
# ═══════════════════════════════════════════════════════


class TestCancelReasons:
    """cancel_reasons に TOXICITY_PARTICIPATION_SKIP が存在."""

    def test_constant_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR

        assert hasattr(CR, "TOXICITY_PARTICIPATION_SKIP")
        assert CR.TOXICITY_PARTICIPATION_SKIP == "toxicity_participation_skip"

    def test_in_audit_cancel_reasons(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR

        assert CR.TOXICITY_PARTICIPATION_SKIP in CR.AUDIT_CANCEL_REASONS


# ═══════════════════════════════════════════════════════
# 6. Executor — toxicity_offset_mult パラメータ
# ═══════════════════════════════════════════════════════


class TestExecutorToxicityParam:
    """fill_cycle_executor.run_single_cycle に toxicity_offset_mult がある."""

    def test_run_single_cycle_has_toxicity_param(self) -> None:
        assert "toxicity_offset_mult" in _RUN_SINGLE_CYCLE_SIG.parameters
        p = _RUN_SINGLE_CYCLE_SIG.parameters["toxicity_offset_mult"]
        assert p.default == 1.0

    def test_toxicity_offset_applied_in_source(self) -> None:
        """240# toxicity_offset が offset_pipeline.py で _apply_offset_multiplier に適用される."""
        src = _OFFSET_PIPELINE_SOURCE
        assert "toxicity_offset_mult" in src
        assert "_apply_offset_multiplier" in src


# ═══════════════════════════════════════════════════════
# 7. Orchestrator — assess_toxicity メソッド
# ═══════════════════════════════════════════════════════


class TestOrchestratorToxicityAssess:
    """Orchestrator に _assess_buy/sell_toxicity() が存在."""

    def test_assess_methods_exist(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        assert hasattr(FillLoopOrchestratorMixin, "_assess_buy_toxicity")
        assert hasattr(FillLoopOrchestratorMixin, "_assess_sell_toxicity")

    def test_unified_assess_method_exists(self) -> None:
        """241# S-2 DRY fix: 統一 _assess_toxicity() メソッドが存在."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        assert hasattr(FillLoopOrchestratorMixin, "_assess_toxicity")

    def test_toxicity_passed_to_gate_evaluate(self) -> None:
        """gate_aggregator.evaluate() に buy_toxicity/sell_toxicity を渡している."""
        src = _ORCHESTRATOR_MID_CYCLE_SOURCE
        assert "buy_toxicity=" in src
        assert "sell_toxicity=" in src

    def test_participation_skip_in_orchestrator(self) -> None:
        """participation_rate による確率的スキップロジックが存在."""
        src = _ORCHESTRATOR_MID_CYCLE_SOURCE
        assert "participation_rate" in src
        assert "toxicity_participation_skip" in src

    def test_evaluation_order_toxicity_before_check_kill(self) -> None:
        """241# C-2 fix: assess_toxicity が check_kill より先に評価される."""
        src = _ORCHESTRATOR_MID_CYCLE_SOURCE
        # _assess_buy_toxicity() の呼び出しが _is_side_killed("buy") の前にある
        tox_pos = src.find("_assess_buy_toxicity()")
        # 275# DRY: _is_buy_killed() → _is_side_killed("buy") に統一
        kill_pos = src.find('_is_side_killed("buy")')
        if kill_pos < 0:
            kill_pos = src.find("_is_side_killed('buy')")
        assert tox_pos < kill_pos, (
            "assess toxicity must be called before check_kill "
            "to avoid stale cooldown observation"
        )


# ═══════════════════════════════════════════════════════
# 8. 市場理論: Glosten-Milgrom 適合性
# ═══════════════════════════════════════════════════════


class TestGlostenMilgromTheory:
    """市場理論 (Glosten-Milgrom) との整合性."""

    def test_adverse_selection_premium_monotonic(self) -> None:
        """逆選択リスク↑ → offset_mult↑ (単調増加)."""
        mgr = _make_mgr(threshold=-1.0, window=5)
        offsets: list[float] = []
        for pnl in [0.0, -0.3, -0.5, -0.7, -0.9]:
            mgr2 = _make_mgr(threshold=-1.0, window=5)
            for _ in range(5):
                mgr2.track(pnl)
            a = mgr2.assess_toxicity()
            offsets.append(a.offset_mult)
        # 各段階でoffsetが非減少
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1], (
                f"offset_mult not monotonic: {offsets}"
            )

    def test_participation_monotonic_decreasing(self) -> None:
        """逆選択リスク↑ → participation_rate↓ (単調減少)."""
        rates: list[float] = []
        for pnl in [0.0, -0.3, -0.5, -0.7, -0.9]:
            mgr = _make_mgr(threshold=-1.0, window=5)
            for _ in range(5):
                mgr.track(pnl)
            a = mgr.assess_toxicity()
            rates.append(a.participation_rate)
        for i in range(1, len(rates)):
            assert rates[i] <= rates[i - 1], (
                f"participation_rate not monotonic: {rates}"
            )

    def test_green_is_full_liquidity_provision(self) -> None:
        """GREEN = 完全な流動性提供 (offset=1, participation=1)."""
        mgr = _make_mgr(threshold=-1.0, window=5)
        for _ in range(5):
            mgr.track(0.5)
        a = mgr.assess_toxicity()
        assert a.offset_mult == 1.0
        assert a.participation_rate == 1.0

    def test_kill_is_zero_liquidity(self) -> None:
        """KILL = 流動性提供停止 (participation=0)."""
        mgr = _make_mgr(threshold=-0.5, window=5)
        for _ in range(5):
            mgr.track(-1.0)
        a = mgr.assess_toxicity()
        assert a.participation_rate == 0.0


# ═══════════════════════════════════════════════════════
# 9. BuyDynamicKillManager の対称性
# ═══════════════════════════════════════════════════════


class TestBuyManagerToxicity:
    """BuyDynamicKillManager も assess_toxicity() を継承."""

    def test_assess_toxicity_inherited(self) -> None:
        cfg = DynamicKillConfig(
            enabled=True, window=5, threshold_bps=-0.8,
            toxicity_budget_enabled=True,
        )
        mgr = BuyDynamicKillManager(cfg)
        for _ in range(5):
            mgr.track(-0.4)
        a = mgr.assess_toxicity()
        assert a.level in (
            ToxicityLevel.GREEN, ToxicityLevel.YELLOW,
            ToxicityLevel.ORANGE, ToxicityLevel.KILL,
        )
        assert a.threshold_used == -0.8


# ═══════════════════════════════════════════════════════
# 10. 241# S-4: DynamicKillConfig toxicity バリデーション
# ═══════════════════════════════════════════════════════


class TestDynamicKillConfigToxicityValidation:
    """241# S-4: toxicity config フィールドのバリデーション検証."""

    def test_valid_config_ok(self) -> None:
        """正常な設定は例外なしで作成可能."""
        DynamicKillConfig(
            toxicity_budget_enabled=True,
            toxicity_warn_level=0.3,
            toxicity_caution_level=0.7,
            toxicity_warn_offset_mult=1.0,
            toxicity_caution_offset_mult=2.0,
            toxicity_kill_offset_mult=3.0,
            toxicity_caution_min_participation=0.33,
        )

    def test_warn_ge_caution_raises(self) -> None:
        """warn_level >= caution_level → ValueError."""
        with pytest.raises(ValueError, match="warn_level < caution_level"):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_warn_level=0.7,
                toxicity_caution_level=0.3,
            )

    def test_warn_equals_caution_raises(self) -> None:
        """warn_level == caution_level → ValueError."""
        with pytest.raises(ValueError, match="warn_level < caution_level"):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_warn_level=0.5,
                toxicity_caution_level=0.5,
            )

    def test_caution_above_one_raises(self) -> None:
        """caution_level > 1.0 → ValueError."""
        with pytest.raises(ValueError, match="caution_level <= 1.0"):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_warn_level=0.3,
                toxicity_caution_level=1.5,
            )

    def test_warn_offset_below_one_raises(self) -> None:
        """warn_offset_mult < 1.0 → ValueError."""
        with pytest.raises(ValueError, match="warn_offset_mult must be >= 1.0"):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_warn_offset_mult=0.5,
            )

    def test_caution_offset_below_warn_offset_raises(self) -> None:
        """caution_offset_mult < warn_offset_mult → ValueError."""
        with pytest.raises(ValueError, match="caution_offset_mult must be >="):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_warn_offset_mult=2.0,
                toxicity_caution_offset_mult=1.5,
            )

    def test_kill_offset_below_caution_offset_raises(self) -> None:
        """kill_offset_mult < caution_offset_mult → ValueError."""
        with pytest.raises(ValueError, match="kill_offset_mult must be >="):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_caution_offset_mult=3.0,
                toxicity_kill_offset_mult=2.0,
            )

    def test_min_participation_zero_raises(self) -> None:
        """min_participation <= 0 → ValueError."""
        with pytest.raises(ValueError, match="caution_min_participation"):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_caution_min_participation=0.0,
            )

    def test_min_participation_above_one_raises(self) -> None:
        """min_participation > 1.0 → ValueError."""
        with pytest.raises(ValueError, match="caution_min_participation"):
            DynamicKillConfig(
                toxicity_budget_enabled=True,
                toxicity_caution_min_participation=1.5,
            )

    def test_disabled_skips_validation(self) -> None:
        """toxicity_budget_enabled=False → バリデーションをスキップ."""
        # 不正な値でも enabled=False なら例外なし
        cfg = DynamicKillConfig(
            toxicity_budget_enabled=False,
            toxicity_warn_level=0.9,
            toxicity_caution_level=0.1,
        )
        assert not cfg.toxicity_budget_enabled


# ═══════════════════════════════════════════════════════
# 11. 241# C-2: 評価順序回帰テスト
# ═══════════════════════════════════════════════════════


class TestEvaluationOrderRegression:
    """241# C-2 fix: assess_toxicity と check_kill の評価順序."""

    def test_cooldown_last_cycle_assess_sees_cooldown(self) -> None:
        """最終 cooldown サイクルでも assess_toxicity は KILL を返す.

        check_kill() 前に assess_toxicity() を呼ぶことで、
        cooldown デクリメント前の状態を正しく観測できる。
        """
        mgr = _make_mgr(threshold=-0.5, window=5)
        # kill を発動: cooldown = resume_window = 5
        for _ in range(5):
            mgr.track(-1.0)
        killed, _ = mgr.check_kill()
        assert killed  # cooldown=5

        # cooldown を 1 まで消費 (4回 check_kill)
        for _ in range(4):
            killed, _ = mgr.check_kill()
        # この時点で cooldown=1

        # 先に assess_toxicity → cooldown > 0 なので KILL
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.KILL

        # その後 check_kill → cooldown をデクリメント
        killed, _ = mgr.check_kill()
        assert killed  # cooldown=1 → 0 のデクリメント中は True

    def test_assess_before_check_kill_consistency(self) -> None:
        """assess_toxicity を check_kill 前に呼んだ場合の一貫性."""
        mgr = _make_mgr(threshold=-0.5, window=5)
        # YELLOW zone: rolling_mean = -0.3, score = 0.6
        for _ in range(5):
            mgr.track(-0.3)

        # assess 先
        a = mgr.assess_toxicity()
        assert a.level == ToxicityLevel.YELLOW

        # check_kill: rolling_mean (-0.3) >= threshold (-0.5) → not killed
        killed, _ = mgr.check_kill()
        assert not killed  # 矛盾なし: YELLOW ≠ KILL
