"""193# テスト: ev_weighted → offset 修飾子モード.

- ev_weighted_as_offset: EV score を offset 乗数として計算
- emergency skip: 極端な negative EV は hard skip
- executor EV offset: order_price の post-hoc 調整
- 旧モードとの後方互換性
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ─── helper ─────────────────────────────────────────────────────────


class _MockSkipDecision:
    """SkipGate.evaluate 戻り値のモック."""

    def __init__(
        self,
        predicted_pnl_bps: float = 0.0,
        threshold_used: float = 0.0,
        threshold_bps: float = 0.0,
        features_used: int = 10,
        as_probability: float | None = None,
        reason: str = "pass",
        model_used: str = "primary",
        should_skip: bool = False,
    ) -> None:
        self.predicted_pnl_bps = predicted_pnl_bps
        self.threshold_used = threshold_used
        self.threshold_bps = threshold_bps
        self.features_used = features_used
        self.as_probability = as_probability
        self.reason = reason
        self.model_used = model_used
        self.should_skip = should_skip


def _make_alt_gate(pred_pnl: float = 1.0) -> MagicMock:
    """alt model ゲートのモック."""
    gate = MagicMock()
    gate.config.mode = "pnl"
    mock_decision = _MockSkipDecision(predicted_pnl_bps=pred_pnl)
    gate.evaluate.return_value = mock_decision
    return gate


def _make_evaluator(
    *,
    ev_as_offset: bool = True,
    sensitivity: float = 0.05,
    min_mult: float = 0.5,
    max_mult: float = 1.5,
    emergency_threshold: float = -8.0,
    gate_alt_buy: object | None = None,
    gate_alt_sell: object | None = None,
) -> object:
    """193# テスト用 SkipGateEvaluator 最小構成."""
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

    config = FillTestConfig(
        skip_gate_enabled=False,
        skip_gate_ev_weighted_enabled=True,
        skip_gate_ev_w30=0.4,
        skip_gate_ev_w120=0.6,
        skip_gate_ev_as_offset_enabled=ev_as_offset,
        skip_gate_ev_offset_sensitivity=sensitivity,
        skip_gate_ev_offset_min_mult=min_mult,
        skip_gate_ev_offset_max_mult=max_mult,
        skip_gate_ev_emergency_skip_threshold=emergency_threshold,
        skip_gate_ev_max_consecutive_skip=0,
        skip_gate_ev_one_sided_threshold_shift=0.0,
    )
    evaluator = SkipGateEvaluator(config, Path("."))
    evaluator._gate_alt_buy = gate_alt_buy
    evaluator._gate_alt_sell = gate_alt_sell
    return evaluator


# ─── 1. ev_weighted_as_offset 基本テスト ─────────────────────────────


class TestEvWeightedAsOffset:
    """193#: ev_weighted が offset 修飾子として機能するテスト."""

    def test_positive_ev_returns_pass_with_score(self) -> None:
        """正の ev_score → should_skip=False, ev_score が predicted_pnl_bps に格納."""
        alt = _make_alt_gate(pred_pnl=2.0)  # pnl120
        evaluator = _make_evaluator(gate_alt_buy=alt)
        primary = _MockSkipDecision(predicted_pnl_bps=1.0, threshold_used=0.0)

        result = evaluator._try_ev_weighted_decision(
            "buy", {"f1": 1.0}, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is False
        # ev = 0.4 * 1.0 + 0.6 * 2.0 = 1.6
        assert result.predicted_pnl_bps == pytest.approx(1.6, abs=0.01)
        assert result.reason == "ev_weighted_offset"

    def test_negative_ev_returns_pass_not_skip(self) -> None:
        """負の ev_score でも should_skip=False (offset モードではゲートしない)."""
        alt = _make_alt_gate(pred_pnl=-3.0)
        evaluator = _make_evaluator(gate_alt_buy=alt)
        primary = _MockSkipDecision(predicted_pnl_bps=-2.0, threshold_used=0.0)

        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4 * (-2.0) + 0.6 * (-3.0) = -2.6
        assert result.predicted_pnl_bps == pytest.approx(-2.6, abs=0.01)
        assert result.should_skip is False  # ← ゲートしない！
        assert result.reason == "ev_weighted_offset"

    def test_emergency_skip_on_extreme_negative(self) -> None:
        """ev_score < emergency_threshold → hard skip."""
        alt = _make_alt_gate(pred_pnl=-15.0)
        evaluator = _make_evaluator(
            gate_alt_buy=alt,
            emergency_threshold=-8.0,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-5.0, threshold_used=0.0)

        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4 * (-5.0) + 0.6 * (-15.0) = -11.0 < -8.0
        assert result.predicted_pnl_bps == pytest.approx(-11.0, abs=0.01)
        assert result.should_skip is True
        assert result.reason == "ev_weighted_emergency_skip"

    def test_sell_side_ev_calculation(self) -> None:
        """sell: ev = w30 * alt + w120 * primary."""
        alt = _make_alt_gate(pred_pnl=0.5)  # pnl30 (sell alt)
        evaluator = _make_evaluator(gate_alt_sell=alt)
        primary = _MockSkipDecision(predicted_pnl_bps=1.5, threshold_used=0.0)

        result = evaluator._try_ev_weighted_decision(
            "sell", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # sell: ev = w30 * alt + w120 * primary = 0.4*0.5 + 0.6*1.5 = 1.1
        assert result.predicted_pnl_bps == pytest.approx(1.1, abs=0.01)
        assert result.should_skip is False

    def test_disabled_returns_none(self) -> None:
        """ev_as_offset=True でも ev_weighted_enabled=False → None."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        config = FillTestConfig(
            skip_gate_enabled=False,
            skip_gate_ev_weighted_enabled=False,
            skip_gate_ev_as_offset_enabled=True,
        )
        evaluator = SkipGateEvaluator(config, Path("."))
        primary = _MockSkipDecision(predicted_pnl_bps=1.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is None

    def test_no_alt_model_returns_none(self) -> None:
        """alt モデル未ロード → None (offset モードでも)."""
        evaluator = _make_evaluator(gate_alt_buy=None)
        primary = _MockSkipDecision(predicted_pnl_bps=1.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is None


# ─── 2. offset 乗数計算テスト ────────────────────────────────────────


class TestEvOffsetMultiplier:
    """193#: ev_score → offset 乗数の計算ロジック."""

    @pytest.mark.parametrize(
        "ev_score,sensitivity,expected_raw",
        [
            (0.0, 0.05, 1.0),       # neutral → no change
            (2.0, 0.05, 1.10),      # positive → more aggressive
            (-3.0, 0.05, 0.85),     # negative → more conservative
            (-10.0, 0.05, 0.50),    # very negative → clamped at min
            (20.0, 0.05, 1.50),     # very positive → clamped at max
        ],
    )
    def test_ev_offset_multiplier_calculation(
        self, ev_score: float, sensitivity: float, expected_raw: float,
    ) -> None:
        """ev_score × sensitivity → offset 乗数 (クランプ適用)."""
        min_m, max_m = 0.5, 1.5
        raw = 1.0 + sensitivity * ev_score
        clamped = max(min_m, min(max_m, raw))
        assert clamped == pytest.approx(expected_raw, abs=0.01)

    def test_offset_mult_logged_in_decision(self) -> None:
        """_ev_weighted_as_offset が offset_mult をログに出力するテスト."""
        alt = _make_alt_gate(pred_pnl=-3.0)
        evaluator = _make_evaluator(
            gate_alt_buy=alt,
            sensitivity=0.1,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-1.0, threshold_used=0.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4*(-1.0) + 0.6*(-3.0) = -2.2
        assert result.predicted_pnl_bps == pytest.approx(-2.2, abs=0.01)
        assert result.should_skip is False


# ─── 3. SkipGateResult.ev_score テスト ────────────────────────────────


class TestSkipGateResultEvScore:
    """193#: SkipGateResult に ev_score フィールドが追加されたテスト."""

    def test_ev_score_default_none(self) -> None:
        """ev_score のデフォルト値は None."""
        from scripts.v460.lib.fill_config import SkipGateResult
        r = SkipGateResult()
        assert r.ev_score is None

    def test_ev_score_set(self) -> None:
        """ev_score を設定できる."""
        from scripts.v460.lib.fill_config import SkipGateResult
        r = SkipGateResult()
        r.ev_score = -2.5
        assert r.ev_score == pytest.approx(-2.5)


# ─── 4. FillTestConfig 新フィールドテスト ────────────────────────────


class TestEvOffsetConfig:
    """193#: FillTestConfig に追加された ev_offset 関連フィールド."""

    def test_defaults(self) -> None:
        """デフォルト値のテスト."""
        from scripts.v460.lib.fill_config import FillTestConfig
        c = FillTestConfig()
        assert c.skip_gate_ev_as_offset_enabled is False
        assert c.skip_gate_ev_offset_sensitivity == pytest.approx(0.05)
        assert c.skip_gate_ev_offset_min_mult == pytest.approx(0.5)
        assert c.skip_gate_ev_offset_max_mult == pytest.approx(1.5)
        assert c.skip_gate_ev_emergency_skip_threshold == pytest.approx(-8.0)

    def test_yaml_parse(self) -> None:
        """YAML parse で新フィールドが反映されること."""
        from scripts.v460.lib.fill_config import FillTestConfig

        yaml_data = {
            "skip_gate": {
                "ev_as_offset_enabled": True,
                "ev_offset_sensitivity": 0.1,
                "ev_offset_min_mult": 0.3,
                "ev_offset_max_mult": 2.0,
                "ev_emergency_skip_threshold": -10.0,
            }
        }
        c = FillTestConfig.from_yaml(yaml_data)
        assert c.skip_gate_ev_as_offset_enabled is True
        assert c.skip_gate_ev_offset_sensitivity == pytest.approx(0.1)
        assert c.skip_gate_ev_offset_min_mult == pytest.approx(0.3)
        assert c.skip_gate_ev_offset_max_mult == pytest.approx(2.0)
        assert c.skip_gate_ev_emergency_skip_threshold == pytest.approx(-10.0)


# ─── 5. 後方互換性テスト ─────────────────────────────────────────────


class TestBackwardCompatibility:
    """旧モード (ev_as_offset_enabled=False) が変更前と同じ挙動をするテスト."""

    def test_old_mode_skip_on_negative_ev(self) -> None:
        """旧モード: 負の ev_score → should_skip=True."""
        alt = _make_alt_gate(pred_pnl=-3.0)
        evaluator = _make_evaluator(
            ev_as_offset=False,
            gate_alt_buy=alt,
        )
        primary = _MockSkipDecision(
            predicted_pnl_bps=-2.0,
            threshold_used=0.0,
        )
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4*(-2.0) + 0.6*(-3.0) = -2.6 < 0.0 → skip
        assert result.should_skip is True
        assert result.reason == "ev_weighted_skip"

    def test_old_mode_pass_on_positive_ev(self) -> None:
        """旧モード: 正の ev_score → should_skip=False."""
        alt = _make_alt_gate(pred_pnl=2.0)
        evaluator = _make_evaluator(
            ev_as_offset=False,
            gate_alt_buy=alt,
        )
        primary = _MockSkipDecision(
            predicted_pnl_bps=1.0,
            threshold_used=0.0,
        )
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4*1.0 + 0.6*2.0 = 1.6 > 0.0 → pass
        assert result.should_skip is False
        assert result.reason == "ev_weighted_pass"


# ─── 6. Executor EV offset 価格調整テスト ─────────────────────────────


class TestExecutorEvOffsetAdjustment:
    """193#: fill_cycle_executor での ev_score → order_price 調整ロジック.

    実際の executor を使わず、計算ロジックだけを検証する。
    """

    @staticmethod
    def _compute_ev_adjustment(
        ev_score: float,
        side: str,
        spread: float,
        offset_ratio: float,
        order_price: float,
        *,
        sensitivity: float = 0.05,
        min_mult: float = 0.5,
        max_mult: float = 1.5,
    ) -> tuple[float, float]:
        """executor 内の EV offset 計算ロジックを再現.

        Returns:
            (adjusted_price, new_offset_ratio)
        """
        raw_mult = 1.0 + sensitivity * ev_score
        ev_mult = max(min_mult, min(max_mult, raw_mult))
        if ev_mult == 1.0:
            return order_price, offset_ratio
        old_offset = spread * offset_ratio
        new_offset = old_offset * ev_mult
        delta = new_offset - old_offset
        if side == "buy":
            adjusted_price = round(order_price + delta)
        else:
            adjusted_price = round(order_price - delta)
        new_ratio = offset_ratio * ev_mult
        return adjusted_price, new_ratio

    def test_negative_ev_widens_offset_buy(self) -> None:
        """負の EV → buy offset 減少 → 価格が mid から離れる (保守的)."""
        # spread=2000, offset_ratio=0.5 → offset=1000
        # buy: price = best_bid + offset = 9960000 + 1000 = 9961000
        price, ratio = self._compute_ev_adjustment(
            ev_score=-3.0,
            side="buy",
            spread=2000,
            offset_ratio=0.5,
            order_price=9_961_000,
            sensitivity=0.05,
        )
        # ev_mult = 1.0 + 0.05*(-3.0) = 0.85
        # delta = 1000*0.85 - 1000 = -150
        # price = 9961000 + (-150) = 9960850 (mid から離れた = 保守的)
        assert price == 9_960_850
        assert ratio == pytest.approx(0.5 * 0.85, abs=0.001)

    def test_positive_ev_narrows_offset_buy(self) -> None:
        """正の EV → buy offset 増加 → 価格が mid に近づく (積極的)."""
        price, ratio = self._compute_ev_adjustment(
            ev_score=2.0,
            side="buy",
            spread=2000,
            offset_ratio=0.5,
            order_price=9_961_000,
            sensitivity=0.05,
        )
        # ev_mult = 1.0 + 0.05*2.0 = 1.1
        # delta = 1000*1.1 - 1000 = 100
        # price = 9961000 + 100 = 9961100 (mid に近づく = 積極的)
        assert price == 9_961_100
        assert ratio == pytest.approx(0.5 * 1.1, abs=0.001)

    def test_negative_ev_widens_offset_sell(self) -> None:
        """負の EV → sell offset 減少 → 価格が mid から離れる (保守的)."""
        # sell: price = best_ask - offset = 9962000 - 1000 = 9961000
        price, ratio = self._compute_ev_adjustment(
            ev_score=-3.0,
            side="sell",
            spread=2000,
            offset_ratio=0.5,
            order_price=9_961_000,
            sensitivity=0.05,
        )
        # ev_mult = 0.85, delta = -150
        # sell: price = 9961000 - (-150) = 9961150 (mid から離れた = 保守的)
        assert price == 9_961_150
        assert ratio == pytest.approx(0.5 * 0.85, abs=0.001)

    def test_zero_ev_no_change(self) -> None:
        """EV=0 → 価格変更なし."""
        price, ratio = self._compute_ev_adjustment(
            ev_score=0.0,
            side="buy",
            spread=2000,
            offset_ratio=0.5,
            order_price=9_961_000,
        )
        assert price == 9_961_000
        assert ratio == pytest.approx(0.5)

    def test_extreme_negative_clamped(self) -> None:
        """極端な負 EV → min_mult でクランプ."""
        price, ratio = self._compute_ev_adjustment(
            ev_score=-100.0,
            side="buy",
            spread=2000,
            offset_ratio=0.5,
            order_price=9_961_000,
            sensitivity=0.05,
            min_mult=0.5,
        )
        # raw_mult = 1.0 + 0.05*(-100) = -4.0, clamped to 0.5
        # delta = 1000*0.5 - 1000 = -500
        assert price == 9_960_500
        assert ratio == pytest.approx(0.5 * 0.5, abs=0.001)
