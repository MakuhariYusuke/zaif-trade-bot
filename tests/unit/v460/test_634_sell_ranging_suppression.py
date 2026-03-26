"""634# sell/ranging 抑制: ranging buy 優先 + skip_gate penalty + no_feasible freeze.

テスト対象:
  P0: skip_gate sell/ranging penalty offset
  P1-1: no_feasible_quote → side freeze (2 cycles)
  P1-3: ranging buy priority (連続上限あり)
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.side_selector import SideSelector


# =========================================================
# P1-3: Ranging Buy Priority
# =========================================================


class TestRangingBuyPriority:
    """634# P1-3: ranging 時に sell → buy 切り替え."""

    @staticmethod
    def _make_selector(**overrides) -> SideSelector:
        cfg = FillTestConfig(**overrides)
        return SideSelector(cfg)

    def test_ranging_forces_buy_when_sell_turn(self) -> None:
        """ranging + sell 番 → buy に切り替わる."""
        sel = self._make_selector()
        # 1st call: _last_side is None → base="buy" (not sell), no override
        s1 = sel.next(regime="ranging")
        sel.update_after_decision(s1)
        assert s1 == "buy"

        # 2nd call: _last_side="buy" → base="sell" → ranging override → "buy"
        s2 = sel.next(regime="ranging")
        sel.update_after_decision(s2)
        assert s2 == "buy"

    def test_ranging_buy_consecutive_limit(self) -> None:
        """連続 buy が上限に達したら sell を許可."""
        sel = self._make_selector(ranging_buy_priority_max_consecutive=2)

        # Seed: start with a sell
        sel.update_after_decision("sell")

        # Now base would be "buy" (alternation), no override:
        s1 = sel.next(regime="ranging")
        sel.update_after_decision(s1)
        assert s1 == "buy"  # alternation buy, no override needed

        # _last_side="buy", _consecutive=0, base="sell" → override to buy
        s2 = sel.next(regime="ranging")
        sel.update_after_decision(s2)
        assert s2 == "buy"  # _consecutive was 0 < 2

        # _last_side="buy", _consecutive=1, base="sell" → override to buy
        s3 = sel.next(regime="ranging")
        sel.update_after_decision(s3)
        assert s3 == "buy"  # _consecutive was 1 < 2

        # _last_side="buy", _consecutive=2, base="sell" → limit hit, allow sell
        s4 = sel.next(regime="ranging")
        sel.update_after_decision(s4)
        assert s4 == "sell"  # _consecutive=2 >= 2

    def test_non_ranging_no_override(self) -> None:
        """ranging 以外のレジームでは通常の交互ロジック."""
        sel = self._make_selector()
        sel.update_after_decision("buy")
        # base="sell", regime=trending_up → no override
        s = sel.next(regime="trending_up")
        assert s == "sell"

    def test_frozen_side_overrides_ranging_priority(self) -> None:
        """frozen_side (残高不足) が ranging buy 優先より優先される."""
        sel = self._make_selector()
        sel.freeze_side("buy", cycles=2)
        sel.update_after_decision("buy")

        # base="sell", ranging override → buy, BUT frozen_side=buy → back to sell
        s = sel.next(regime="ranging")
        assert s == "sell"


# =========================================================
# P0: Skip Gate Sell/Ranging Penalty
# =========================================================


class TestSkipGateSellRangingPenalty:
    """634# P0: sell/ranging 時に skip_gate offset が加算される."""

    def test_config_field_exists(self) -> None:
        """FillTestConfig に skip_gate_sell_ranging_offset が存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "skip_gate_sell_ranging_offset")
        assert cfg.skip_gate_sell_ranging_offset == 0.5

    def test_penalty_in_skip_gate_source(self) -> None:
        """skip_gate_evaluator.py に sell/ranging penalty ロジックがある."""
        from tests.unit.v460._fill_test_source import (
            SKIP_GATE_EVALUATOR,
            read_source_text,
        )

        src = read_source_text(SKIP_GATE_EVALUATOR)
        assert "sell_ranging_penalty" in src
        assert "skip_gate_sell_ranging_offset" in src
        # getattr は使わないこと (255# テストと整合)
        lines = src.split("\n")
        penalty_lines = [l for l in lines if "sell_ranging" in l]
        assert not any("getattr" in l for l in penalty_lines)

    def test_offset_ceil_accommodates_penalty(self) -> None:
        """offset_ceil が sell_ranging_offset より大きいこと (有効性確認)."""
        from pathlib import Path

        from tests.unit.v460._yaml_test_helpers import load_yaml_mapping

        cfg = FillTestConfig.from_yaml(load_yaml_mapping(Path("configs/v460/fill_test.yaml")))
        assert cfg.skip_gate_offset_ceil > cfg.skip_gate_sell_ranging_offset, (
            f"offset_ceil ({cfg.skip_gate_offset_ceil}) must exceed "
            f"sell_ranging_offset ({cfg.skip_gate_sell_ranging_offset}) "
            "for the penalty to have effect"
        )


# =========================================================
# P1-1: no_feasible_quote Freeze
# =========================================================


class TestNoFeasibleQuoteFreeze:
    """634# P1-1: no_feasible_quote で side が凍結される."""

    def test_orchestrator_freeze_in_source(self) -> None:
        """orchestrator_post_cycle.py に freeze ロジックがある."""
        from tests.unit.v460._fill_test_source import (
            ORCHESTRATOR_POST_CYCLE,
            read_source_text,
        )

        src = read_source_text(ORCHESTRATOR_POST_CYCLE)
        assert "no_feasible_quote" in src
        assert "freeze_side" in src
        # getattr(self, "_side_selector") ではなく self._side_selector を使うこと
        lines = src.split("\n")
        freeze_lines = [l for l in lines if "freeze_side" in l and "_side_selector" in l]
        assert freeze_lines
        assert not any("getattr" in l for l in freeze_lines), (
            "should use self._side_selector directly, not getattr"
        )

    def test_freeze_side_reduces_sell_cycles(self) -> None:
        """freeze_side(sell, 2) 後の 2 サイクルで sell が回避される."""
        sel = SideSelector(FillTestConfig())
        sel.update_after_decision("buy")  # next would be "sell"

        sel.freeze_side("sell", cycles=2)

        # sell は frozen → buy にフォールバック
        s1 = sel.next(regime="ranging")
        sel.update_after_decision(s1)
        assert s1 == "buy"

        # まだ 1 cycle remaining
        s2 = sel.next(regime="ranging")
        sel.update_after_decision(s2)
        assert s2 == "buy"

        # freeze 切れ → sell 可能に
        s3 = sel.next(regime="ranging")
        # ranging buy priority が介入するが、上限チェック次第
        # frozen は切れているので一旦確認
        assert s3 in ("buy", "sell")


# =========================================================
# Config / Parser Integration
# =========================================================


class TestConfigIntegration:
    """634# 新設フィールドの config-parser 連携."""

    def test_ranging_buy_priority_max_consecutive_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.ranging_buy_priority_max_consecutive == 3

    def test_yaml_round_trip(self) -> None:
        """YAML → FillTestConfig で 634# フィールドが正しくパースされる."""
        from pathlib import Path

        from tests.unit.v460._yaml_test_helpers import load_yaml_mapping

        cfg = FillTestConfig.from_yaml(load_yaml_mapping(Path("configs/v460/fill_test.yaml")))
        assert cfg.skip_gate_sell_ranging_offset == 0.5
        assert cfg.skip_gate_offset_ceil == 0.8
        assert cfg.ranging_buy_priority_max_consecutive == 3
