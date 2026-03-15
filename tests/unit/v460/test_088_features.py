"""
088# 新機能テスト: SkipGate 動的較正 + sell ガード + side 分離適応 + データ品質.

087# レビュー対応で追加された 088# 機能の単体テスト。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from tests.unit.v460._fill_test_source import read_fill_test_runner_source  # 163# mixin 分割対応
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.ml.skip_gate import SkipGate, SkipGateConfig, SkipDecision
from scripts.v460.lib.param_adapter import (
    AdaptationConfig,
    SideAdaptationResult,
    compute_side_adaptation,
)


class _ProbPipelineStub:
    def __init__(self, prob: float = 0.5) -> None:
        self._prob = prob
        self.steps: list[tuple[str, object]] = []

    def set_output(self, *, transform: str) -> "_ProbPipelineStub":
        assert transform == "pandas"
        return self

    def predict_proba(self, _x: object) -> np.ndarray:
        return np.array([[1.0 - self._prob, self._prob]], dtype=float)


# =====================================================================
# SkipGate 動的閾値較正テスト
# =====================================================================


class TestSkipGateAdaptiveThreshold:
    """088# P0-1: SkipGate 動的閾値較正のテスト."""

    def _make_gate(
        self,
        *,
        adaptive: bool = True,
        as_threshold_buy: float = 0.65,
        as_threshold_sell: float = 0.65,
        target_skip_rate_buy: float = 0.10,
        target_skip_rate_sell: float = 0.20,
        adaptive_window: int = 50,
        adaptive_min_samples: int = 20,
        adaptive_step: float = 0.02,
        adaptive_floor: float = 0.35,
        adaptive_ceiling: float = 0.80,
    ) -> SkipGate:
        """AS モード + 動的較正の SkipGate を作成 (モデルはモック)."""
        config = SkipGateConfig(
            mode="as",
            enabled=True,
            as_threshold=0.60,
            as_threshold_buy=as_threshold_buy,
            as_threshold_sell=as_threshold_sell,
            adaptive_threshold=adaptive,
            target_skip_rate_buy=target_skip_rate_buy,
            target_skip_rate_sell=target_skip_rate_sell,
            adaptive_window=adaptive_window,
            adaptive_min_samples=adaptive_min_samples,
            adaptive_step=adaptive_step,
            adaptive_floor=adaptive_floor,
            adaptive_ceiling=adaptive_ceiling,
        )
        pipeline = _ProbPipelineStub()
        gate = SkipGate(
            model=object(),
            scaler=object(),
            feature_cols=["spread_jpy", "offset_ratio", "regime_trending"],
            config=config,
            pipeline=pipeline,
        )
        return gate

    def _make_features(self) -> dict[str, float]:
        """テスト用の特徴量辞書."""
        return {
            "spread_jpy": 3000.0,
            "offset_ratio": 0.05,
            "regime_trending": 0.0,
        }

    def test_warmup_uses_static_threshold(self) -> None:
        """サンプル不足時は静的閾値を使用."""
        gate = self._make_gate(adaptive_min_samples=20)
        # 18 回分の履歴を直接挿入 → +1 で 19 (< min_samples=20)
        gate._pas_history_buy = [0.50] * 18
        # _calibrate_threshold は base_threshold をそのまま返すはず
        result = gate._calibrate_threshold("buy", 0.50, 0.65)
        assert result == 0.65  # 静的閾値がそのまま返る
        assert len(gate._pas_history_buy) == 19  # current_prob が追加された

    def test_calibration_lowers_threshold_for_high_probs(self) -> None:
        """P(AS) 分布が全体的に高い場合、閾値が下がる."""
        gate = self._make_gate(
            as_threshold_buy=0.65,
            target_skip_rate_buy=0.20,
            adaptive_min_samples=10,
            adaptive_step=0.02,
        )
        # P(AS) が全て 0.50 前後 → 20% skip 率達成には閾値を 0.50 付近にすべき
        gate._pas_history_buy = [0.48 + i * 0.002 for i in range(20)]
        new_th = gate._calibrate_threshold("buy", 0.50, 0.65)
        # 0.65 > target(≈0.50) なので step で下がる
        assert new_th < 0.65
        assert new_th == pytest.approx(0.63, abs=0.01)

    def test_calibration_raises_threshold_for_low_probs(self) -> None:
        """P(AS) 分布が全体的に低い場合、閾値が上がる."""
        gate = self._make_gate(
            as_threshold_sell=0.35,
            target_skip_rate_sell=0.20,
            adaptive_min_samples=10,
            adaptive_step=0.02,
        )
        # P(AS) が全て 0.55-0.60 → 閾値 0.35 は低すぎ
        gate._pas_history_sell = [0.55 + i * 0.002 for i in range(20)]
        new_th = gate._calibrate_threshold("sell", 0.58, 0.35)
        # 0.35 < target(≈0.56) なので step で上がる
        assert new_th > 0.35
        assert new_th == pytest.approx(0.37, abs=0.01)

    def test_floor_ceiling_clamp(self) -> None:
        """閾値が floor / ceiling でクランプされる."""
        gate = self._make_gate(
            adaptive_floor=0.40,
            adaptive_ceiling=0.70,
            adaptive_min_samples=5,
            adaptive_step=0.50,  # 大きなステップで一気に移動
        )
        # 非常に低い分布 → 閾値が floor にクランプ
        gate._pas_history_buy = [0.10] * 10
        new_th = gate._calibrate_threshold("buy", 0.10, 0.60)
        assert new_th >= 0.40  # floor

        # 非常に高い分布 → 閾値が ceiling にクランプ
        gate._pas_history_sell = [0.95] * 10
        new_th = gate._calibrate_threshold("sell", 0.95, 0.60)
        assert new_th <= 0.70  # ceiling

    def test_window_truncation(self) -> None:
        """ウィンドウサイズを超えた古い履歴が削除される."""
        gate = self._make_gate(adaptive_window=10)
        gate._pas_history_buy = [0.50] * 15  # 超過
        gate._calibrate_threshold("buy", 0.50, 0.60)
        assert len(gate._pas_history_buy) <= 10

    def test_updates_config_threshold(self) -> None:
        """較正後、config の as_threshold_buy/sell が更新される."""
        gate = self._make_gate(
            as_threshold_buy=0.65,
            as_threshold_sell=0.65,
            adaptive_min_samples=5,
            adaptive_step=0.02,
        )
        # buy 側較正
        gate._pas_history_buy = [0.50] * 10
        gate._calibrate_threshold("buy", 0.50, 0.65)
        assert gate.config.as_threshold_buy != 0.65  # 変更された

        # sell 側較正
        gate._pas_history_sell = [0.50] * 10
        gate._calibrate_threshold("sell", 0.50, 0.65)
        assert gate.config.as_threshold_sell != 0.65  # 変更された

    def test_evaluate_calls_calibrate_in_as_mode(self) -> None:
        """evaluate() が AS モード + adaptive で _calibrate_threshold を呼ぶ."""
        gate = self._make_gate(adaptive=True)
        # Pipeline モックの predict_proba 設定
        gate._pipeline._prob = 0.55
        features = self._make_features()

        decision = gate.evaluate(features, side="buy")
        # P(AS)=0.55 が履歴に記録される
        assert len(gate._pas_history_buy) == 1
        assert gate._pas_history_buy[0] == pytest.approx(0.55)
        assert decision.as_probability == pytest.approx(0.55)

    def test_evaluate_no_calibrate_when_disabled(self) -> None:
        """adaptive_threshold=False ならば較正しない."""
        gate = self._make_gate(adaptive=False)
        gate._pipeline._prob = 0.60
        features = self._make_features()

        decision = gate.evaluate(features, side="buy")
        # 履歴に追加されない
        assert len(gate._pas_history_buy) == 0

    def test_evaluate_no_calibrate_when_no_side(self) -> None:
        """side=None ならば較正しない."""
        gate = self._make_gate(adaptive=True)
        gate._pipeline._prob = 0.60
        features = self._make_features()

        decision = gate.evaluate(features, side=None)
        # side=None なので較正なし
        assert len(gate._pas_history_buy) == 0
        assert len(gate._pas_history_sell) == 0

    def test_pas_history_independent_per_side(self) -> None:
        """buy/sell の P(AS) 履歴が独立."""
        gate = self._make_gate(adaptive=True)
        gate._pipeline._prob = 0.60
        features = self._make_features()

        gate.evaluate(features, side="buy")
        gate._pipeline._prob = 0.70
        gate.evaluate(features, side="sell")

        assert len(gate._pas_history_buy) == 1
        assert len(gate._pas_history_sell) == 1
        assert gate._pas_history_buy[0] == pytest.approx(0.60)
        assert gate._pas_history_sell[0] == pytest.approx(0.70)


# =====================================================================
# SkipGateConfig テスト
# =====================================================================


class TestSkipGateConfig:
    """SkipGateConfig の 088# 新フィールドテスト."""

    def test_defaults(self) -> None:
        """デフォルト値が 088# 仕様通り."""
        cfg = SkipGateConfig()
        assert cfg.adaptive_threshold is False
        assert cfg.target_skip_rate_buy == 0.10
        assert cfg.target_skip_rate_sell == 0.20
        assert cfg.adaptive_window == 50
        assert cfg.adaptive_min_samples == 20
        assert cfg.adaptive_step == 0.05  # 100# 0.02→0.05
        assert cfg.adaptive_floor == 0.35
        assert cfg.adaptive_ceiling == 0.80

    def test_custom_values(self) -> None:
        """カスタム値が正しく設定される."""
        cfg = SkipGateConfig(
            adaptive_threshold=True,
            target_skip_rate_buy=0.15,
            target_skip_rate_sell=0.25,
            adaptive_window=100,
        )
        assert cfg.adaptive_threshold is True
        assert cfg.target_skip_rate_buy == 0.15
        assert cfg.target_skip_rate_sell == 0.25
        assert cfg.adaptive_window == 100


# =====================================================================
# compute_side_adaptation テスト
# =====================================================================


class TestComputeSideAdaptation:
    """088# P1-3: side 分離適応のテスト."""

    def _make_config(self, current: float = 0.05) -> AdaptationConfig:
        return AdaptationConfig(
            current_offset_ratio=current,
            min_fill_rate=0.80,
            max_as_ratio=0.15,
            step_ratio=0.01,
            min_offset_ratio=0.01,
            max_offset_ratio=0.30,
            min_samples=50,
        )

    def test_both_hold(self) -> None:
        """buy/sell 両方正常 → 両方 hold."""
        result = compute_side_adaptation(
            buy_fill_rate=0.90, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.85, sell_as_ratio=0.12, sell_sample_count=100,
            buy_config=self._make_config(), sell_config=self._make_config(),
        )
        assert result.buy.action == "hold"
        assert result.sell.action == "hold"
        assert not result.any_changed

    def test_buy_increase_sell_hold(self) -> None:
        """buy は fill_rate 低下で増加、sell は正常 → 独立判定."""
        result = compute_side_adaptation(
            buy_fill_rate=0.60, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.90, sell_as_ratio=0.10, sell_sample_count=100,
            buy_config=self._make_config(), sell_config=self._make_config(),
        )
        assert result.buy.action == "increase"
        assert result.buy.new_offset == pytest.approx(0.06)
        assert result.sell.action == "hold"
        assert result.any_changed

    def test_sell_decrease_buy_hold(self) -> None:
        """sell は AS 超過で減少、buy は正常 → 独立判定."""
        result = compute_side_adaptation(
            buy_fill_rate=0.90, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.90, sell_as_ratio=0.25, sell_sample_count=100,
            buy_config=self._make_config(), sell_config=self._make_config(),
        )
        assert result.buy.action == "hold"
        assert result.sell.action == "decrease"
        assert result.sell.new_offset == pytest.approx(0.04)
        assert result.any_changed

    def test_independent_configs(self) -> None:
        """buy/sell で異なる config が適用される."""
        buy_cfg = self._make_config(current=0.10)
        sell_cfg = self._make_config(current=0.03)
        result = compute_side_adaptation(
            buy_fill_rate=0.60, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.60, sell_as_ratio=0.10, sell_sample_count=100,
            buy_config=buy_cfg, sell_config=sell_cfg,
        )
        assert result.buy.new_offset == pytest.approx(0.11)
        assert result.sell.new_offset == pytest.approx(0.04)

    def test_sample_insufficient_one_side(self) -> None:
        """片方のサンプル不足 → その側のみ hold."""
        result = compute_side_adaptation(
            buy_fill_rate=0.90, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.60, sell_as_ratio=0.10, sell_sample_count=10,
            buy_config=self._make_config(), sell_config=self._make_config(),
        )
        assert result.buy.action == "hold"
        assert result.sell.action == "hold"
        assert "サンプル不足" in result.sell.reason

    def test_any_changed_property(self) -> None:
        """any_changed プロパティが正しく動作."""
        result = compute_side_adaptation(
            buy_fill_rate=0.60, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.90, sell_as_ratio=0.10, sell_sample_count=100,
            buy_config=self._make_config(), sell_config=self._make_config(),
        )
        assert result.any_changed is True

    def test_default_configs(self) -> None:
        """config=None でデフォルト設定が使われる."""
        result = compute_side_adaptation(
            buy_fill_rate=0.90, buy_as_ratio=0.10, buy_sample_count=100,
            sell_fill_rate=0.90, sell_as_ratio=0.10, sell_sample_count=100,
        )
        assert isinstance(result, SideAdaptationResult)
        assert result.buy.action == "hold"
        assert result.sell.action == "hold"

    def test_deadlock_prevention_side_independent(self) -> None:
        """084# デッドロック防止が side 別に適用される."""
        result = compute_side_adaptation(
            buy_fill_rate=0.70, buy_as_ratio=0.25, buy_sample_count=100,
            sell_fill_rate=0.90, sell_as_ratio=0.10, sell_sample_count=100,
            buy_config=self._make_config(), sell_config=self._make_config(),
        )
        # buy: AS+fill 両方異常 → デッドロック防止
        assert result.buy.action == "hold"
        assert "デッドロック防止" in result.buy.reason
        # sell: 正常
        assert result.sell.action == "hold"


# =====================================================================
# _calibrate_threshold 境界値テスト
# =====================================================================


class TestCalibrateThresholdEdgeCases:
    """_calibrate_threshold の境界ケーステスト."""

    def _make_gate(self, **kwargs: float) -> SkipGate:
        defaults = dict(
            adaptive_threshold=True,
            target_skip_rate_buy=0.10,
            target_skip_rate_sell=0.20,
            adaptive_window=50,
            adaptive_min_samples=5,
            adaptive_step=0.02,
            adaptive_floor=0.35,
            adaptive_ceiling=0.80,
        )
        defaults.update(kwargs)  # type: ignore[arg-type]
        config = SkipGateConfig(mode="as", **defaults)  # type: ignore[arg-type]
        return SkipGate(
            model=object(),
            scaler=object(),
            feature_cols=["f1"],
            config=config,
            pipeline=_ProbPipelineStub(),
        )

    def test_all_same_probability(self) -> None:
        """全て同じ P(AS) → 閾値は近似的にその値に収束."""
        gate = self._make_gate(adaptive_step=0.50)
        gate._pas_history_buy = [0.50] * 10
        result = gate._calibrate_threshold("buy", 0.50, 0.50)
        assert result == pytest.approx(0.50, abs=0.01)

    def test_zero_skip_rate_target(self) -> None:
        """target_skip_rate=0 → 分位点は最大値 → 閾値は ceiling 方向."""
        gate = self._make_gate(target_skip_rate_buy=0.0, adaptive_step=0.02)
        gate._pas_history_buy = [0.50 + i * 0.01 for i in range(10)]
        result = gate._calibrate_threshold("buy", 0.55, 0.50)
        # quantile_idx = len * 1.0 → 最大値に近い → 閾値上昇
        assert result >= 0.50

    def test_full_skip_rate_target(self) -> None:
        """target_skip_rate=1.0 → 分位点はゼロ近傍 → 閾値は floor 方向."""
        gate = self._make_gate(target_skip_rate_buy=1.0, adaptive_step=0.50)
        gate._pas_history_buy = [0.50 + i * 0.01 for i in range(10)]
        result = gate._calibrate_threshold("buy", 0.55, 0.60)
        # quantile_idx = 0 → 最小値 → 閾値下降 → floor にクランプ
        assert result >= 0.35  # floor

    def test_step_limits_movement(self) -> None:
        """adaptive_step がジャンプ幅を制限する."""
        gate = self._make_gate(adaptive_step=0.01)
        gate._pas_history_buy = [0.30] * 10  # 低い分布
        result = gate._calibrate_threshold("buy", 0.30, 0.60)
        # 0.60 からの変動は最大 0.01
        assert abs(result - 0.60) <= 0.01 + 1e-9


# =====================================================================
# データ品質テスト (run_id / git_sha 存在チェック)
# =====================================================================


class TestDataQualityFillRecord:
    """088# P0-4: FillRecord 早期 return パスのフィールド検証.

    145# §9-#5: _make_skip_record に一元化されたため、
    (a) ヘルパ内の FillRecord に run_id/git_sha が含まれること、
    (b) cancel_reason を含む全 FillRecord 生成箇所に run_id が存在すること
    の 2 段階で検証する。
    """

    def test_early_return_paths_have_run_id(self) -> None:
        """全 skip record 生成箇所に run_id が含まれる (ソース検査)."""
        import re
        # 163# mixin 分割: 全ソースを連結して検索
        content = read_fill_test_runner_source()
        # 145# §9-#5 → 216# §7: _make_skip_record / build_skip_fill_record に集約済み
        # 集約先の関数自体が run_id を受け取る or 伝播していることを確認
        # _make_skip_record は build_skip_fill_record を呼ぶ wrapper
        assert "run_id" in content, "run_id reference missing from source"
        # build_skip_fill_record の呼び出し箇所で run_id が引数に含まれる
        for m in re.finditer(r'build_skip_fill_record\(', content):
            start = m.start()
            # 括弧対応でブロック抽出
            depth = 0
            end = start
            for i in range(start, min(start + 3000, len(content))):
                if content[i] == '(':
                    depth += 1
                elif content[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            block = content[start:end]
            assert "run_id" in block, (
                f"build_skip_fill_record missing run_id: ...{block[:100]}..."
            )
            assert "git_sha" in block, (
                f"build_skip_fill_record missing git_sha: ...{block[:100]}..."
            )

    def test_make_skip_record_used_for_all_skips(self) -> None:
        """145# §9-#5: cancel_reason 付きスキップは skip_record wrapper 経由であること."""
        import re
        # 163# mixin 分割: 全ソースを連結して検索
        content = read_fill_test_runner_source()
        # 216# §7: _make_skip_record + _make_loop_skip_record の合計を検査
        # _make_loop_skip_record は _make_skip_record の wrapper
        skip_calls = len(re.findall(r'_make_(?:loop_)?skip_record\(', content))
        assert skip_calls >= 8, (
            f"Expected ≥8 skip_record wrapper calls, found {skip_calls}"
        )
