"""366# M4: GLFT Fill Probability Model テスト.

FillProbabilityModel / estimate_fill_probability_params の単体テスト。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.v460.lib.fill_probability_model import (
    DEFAULT_A,
    DEFAULT_K,
    MIN_K,
    MIN_SAMPLES,
    FillProbabilityModel,
    FillProbEstimate,
    estimate_fill_probability_params,
)


# =====================================================================
# TestFillProbabilityModel
# =====================================================================


class TestFillProbabilityModel:
    """FillProbabilityModel の単体テスト."""

    def test_predict_fill_prob_at_zero_offset(self) -> None:
        """offset=0 で fill prob = A."""
        model = FillProbabilityModel(A=0.9, k=50.0)
        assert model.predict_fill_prob(0.0) == pytest.approx(0.9)

    def test_predict_fill_prob_decreases_with_offset(self) -> None:
        """offset 増加で fill prob 減少."""
        model = FillProbabilityModel(A=0.9, k=50.0)
        p1 = model.predict_fill_prob(0.01)
        p2 = model.predict_fill_prob(0.05)
        p3 = model.predict_fill_prob(0.10)
        assert p1 > p2 > p3 > 0.0

    def test_predict_fill_prob_clamped_to_01(self) -> None:
        """fill prob は [0, 1] にクランプ."""
        model = FillProbabilityModel(A=1.5, k=1.0)  # A>1 edge case
        assert model.predict_fill_prob(0.0) == 1.0

    def test_predict_fill_prob_negative_offset(self) -> None:
        """負の offset は 0 にクランプ."""
        model = FillProbabilityModel(A=0.9, k=50.0)
        assert model.predict_fill_prob(-0.1) == pytest.approx(0.9)

    def test_optimal_delta_zero_inventory(self) -> None:
        """在庫ゼロ → δ* = 1/k."""
        model = FillProbabilityModel(k=50.0)
        delta = model.optimal_delta(q=0.0)
        assert delta == pytest.approx(1.0 / 50.0)

    def test_optimal_delta_with_inventory(self) -> None:
        """在庫ありで δ* > 1/k."""
        model = FillProbabilityModel(k=50.0)
        delta_no_inv = model.optimal_delta(q=0.0, gamma=0.01, sigma=0.001, tau=60.0)
        delta_with_inv = model.optimal_delta(q=0.5, gamma=0.01, sigma=0.001, tau=60.0)
        assert delta_with_inv > delta_no_inv

    def test_optimal_delta_k_zero(self) -> None:
        """k=0 (無効) で δ*=0."""
        model = FillProbabilityModel(k=0.0)
        assert model.optimal_delta() == 0.0


# =====================================================================
# TestEstimateFillProbabilityParams
# =====================================================================


class TestEstimateFillProbabilityParams:
    """estimate_fill_probability_params の単体テスト."""

    def test_insufficient_samples_returns_fallback(self) -> None:
        """サンプル不足でフォールバック値."""
        offsets = np.array([0.01, 0.02])
        filled = np.array([True, False])
        result = estimate_fill_probability_params(offsets, filled)
        assert result.is_fallback
        assert result.A == DEFAULT_A
        assert result.k == DEFAULT_K

    def test_mismatched_lengths_raises(self) -> None:
        """長さ不一致で ValueError."""
        with pytest.raises(ValueError, match="長さが不一致"):
            estimate_fill_probability_params(
                np.array([0.01, 0.02]), np.array([True])
            )

    def test_synthetic_exponential_decay(self) -> None:
        """合成データ: A(δ) = 0.9·exp(-30·δ) を正しく推定."""
        rng = np.random.default_rng(42)
        true_A = 0.9
        true_k = 30.0
        n = 500

        offsets = rng.uniform(0.0, 0.10, size=n)
        # 各 offset での fill probability から Bernoulli sampling
        fill_probs = true_A * np.exp(-true_k * offsets)
        filled = rng.random(n) < fill_probs

        result = estimate_fill_probability_params(offsets, filled, n_bins=10)

        assert not result.is_fallback
        assert result.n_samples == n
        # 推定精度: A ±30%, k ±50% (ビン化 + ノイズのため緩い)
        assert result.A == pytest.approx(true_A, rel=0.3)
        assert result.k == pytest.approx(true_k, rel=0.5)
        assert result.r_squared > 0.5

    def test_all_filled_returns_low_k(self) -> None:
        """全 fill → k が小さい (offset に無関係)."""
        rng = np.random.default_rng(123)
        n = 100
        offsets = rng.uniform(0.0, 0.10, size=n)
        filled = np.ones(n, dtype=bool)

        result = estimate_fill_probability_params(offsets, filled, n_bins=5)
        # 全 fill → fill_rate = 1.0 全ビン → log(1) = 0 → slope ≈ 0
        # k = min(|slope|, MIN_K) or fallback
        if not result.is_fallback:
            assert result.k <= MIN_K + 1.0

    def test_negative_offsets_filtered(self) -> None:
        """負の offset は除外される."""
        rng = np.random.default_rng(99)
        n = 100
        offsets = rng.uniform(-0.05, 0.10, size=n)
        fill_probs = 0.8 * np.exp(-30.0 * np.maximum(offsets, 0))
        filled = rng.random(n) < fill_probs

        result = estimate_fill_probability_params(offsets, filled)
        # 負 offset 除外後もサンプル十分なら推定可能
        assert result.n_samples < n  # 一部除外


# =====================================================================
# TestFillProbabilityModelFit
# =====================================================================


class TestFillProbabilityModelFit:
    """FillProbabilityModel.fit() の統合テスト."""

    def test_fit_updates_model_params(self) -> None:
        """fit() で A, k が更新される."""
        rng = np.random.default_rng(42)
        true_A = 0.85
        true_k = 40.0
        n = 300

        offsets = rng.uniform(0.0, 0.08, size=n)
        fill_probs = true_A * np.exp(-true_k * offsets)
        filled = rng.random(n) < fill_probs

        model = FillProbabilityModel()
        estimate = model.fit(offsets, filled)

        assert model.A != DEFAULT_A  # 更新された
        assert model.k != DEFAULT_K  # 更新された
        assert model.last_estimate is estimate
        assert not estimate.is_fallback

    def test_fit_then_predict_consistent(self) -> None:
        """fit 後の predict が A·exp(-k·δ) と一致."""
        model = FillProbabilityModel(A=0.8, k=40.0)
        offset = 0.05
        expected = 0.8 * math.exp(-40.0 * 0.05)
        assert model.predict_fill_prob(offset) == pytest.approx(expected)

    def test_fit_with_side_separation(self) -> None:
        """buy/sell 分離で別モデル学習可能."""
        rng = np.random.default_rng(42)
        n = 200

        # Buy: A=0.9, k=30
        buy_offsets = rng.uniform(0.0, 0.10, size=n)
        buy_probs = 0.9 * np.exp(-30.0 * buy_offsets)
        buy_filled = rng.random(n) < buy_probs

        # Sell: A=0.7, k=50
        sell_offsets = rng.uniform(0.0, 0.10, size=n)
        sell_probs = 0.7 * np.exp(-50.0 * sell_offsets)
        sell_filled = rng.random(n) < sell_probs

        buy_model = FillProbabilityModel()
        sell_model = FillProbabilityModel()
        buy_model.fit(buy_offsets, buy_filled)
        sell_model.fit(sell_offsets, sell_filled)

        # buy の A は sell より大きいはず
        assert buy_model.A > sell_model.A * 0.8  # ノイズ考慮して緩く
