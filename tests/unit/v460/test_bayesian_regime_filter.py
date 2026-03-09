"""366# M2: Bayesian Regime Filter — unit tests.

Hamilton (1989) online Bayesian filter のテストスイート。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.v460.lib.bayesian_regime_filter import (
    BayesianRegimeConfig,
    BayesianRegimeFilter,
    BayesianRegimeResult,
    EmissionParams,
    RegimeState,
    _N_STATES,
    _REGIME_STR_TO_STATE,
    _STATE_TO_REGIME_STR,
)


# =====================================================================
# RegimeState
# =====================================================================


class TestRegimeState:
    """RegimeState enum."""

    def test_has_4_states(self) -> None:
        assert len(RegimeState) == 4

    def test_int_values(self) -> None:
        assert RegimeState.TRENDING_UP == 0
        assert RegimeState.TRENDING_DOWN == 1
        assert RegimeState.RANGING == 2
        assert RegimeState.HIGH_VOL == 3

    def test_numpy_indexing(self) -> None:
        arr = np.array([0.1, 0.2, 0.5, 0.2])
        assert arr[RegimeState.RANGING] == 0.5


# =====================================================================
# EmissionParams
# =====================================================================


class TestEmissionParams:
    """EmissionParams validation."""

    def test_default(self) -> None:
        p = EmissionParams()
        assert p.mu == 0.0
        assert p.sigma == 1e-4

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be > 0"):
            EmissionParams(sigma=-1.0)

    def test_zero_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be > 0"):
            EmissionParams(sigma=0.0)


# =====================================================================
# BayesianRegimeConfig
# =====================================================================


class TestBayesianRegimeConfig:
    """BayesianRegimeConfig defaults."""

    def test_default_config(self) -> None:
        cfg = BayesianRegimeConfig()
        assert len(cfg.emission_params) == 4
        assert 0.5 <= cfg.transition_stickiness <= 0.99
        assert cfg.adaptive_emission is True
        assert cfg.reestimate_interval > 0

    def test_offset_multipliers_present(self) -> None:
        cfg = BayesianRegimeConfig()
        assert len(cfg.offset_multipliers) == 4
        assert cfg.offset_multipliers[RegimeState.RANGING] < cfg.offset_multipliers[RegimeState.HIGH_VOL]


# =====================================================================
# BayesianRegimeFilter — 初期状態
# =====================================================================


class TestBayesianFilterInit:
    """BayesianRegimeFilter initialization."""

    def test_uniform_prior(self) -> None:
        f = BayesianRegimeFilter()
        np.testing.assert_allclose(f.posterior, 0.25, atol=1e-10)

    def test_custom_prior(self) -> None:
        prior = np.array([0.5, 0.1, 0.3, 0.1])
        cfg = BayesianRegimeConfig(prior=prior)
        f = BayesianRegimeFilter(cfg)
        np.testing.assert_allclose(f.posterior, prior, atol=1e-10)

    def test_transition_matrix_shape(self) -> None:
        f = BayesianRegimeFilter()
        A = f.transition_matrix
        assert A.shape == (4, 4)

    def test_transition_matrix_row_sums(self) -> None:
        f = BayesianRegimeFilter()
        A = f.transition_matrix
        np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-10)

    def test_transition_matrix_diagonal_dominant(self) -> None:
        f = BayesianRegimeFilter()
        A = f.transition_matrix
        diag = np.diag(A)
        assert (diag > 0.5).all(), "Diagonal should be > 0.5 (sticky)"


# =====================================================================
# BayesianRegimeFilter — update
# =====================================================================


class TestBayesianFilterUpdate:
    """Core Hamilton filter update."""

    def test_single_update_returns_result(self) -> None:
        f = BayesianRegimeFilter()
        result = f.update(1e-4)
        assert isinstance(result, BayesianRegimeResult)
        assert result.posterior.shape == (4,)
        assert abs(result.posterior.sum() - 1.0) < 1e-10

    def test_positive_return_favors_trending_up(self) -> None:
        """大きな正の return は trending_up の確率を上げるべき."""
        f = BayesianRegimeFilter()
        # 強い正の return を繰り返し投入
        for _ in range(10):
            result = f.update(5e-4)
        assert result.posterior[RegimeState.TRENDING_UP] > result.posterior[RegimeState.TRENDING_DOWN]
        assert result.map_state == RegimeState.TRENDING_UP

    def test_negative_return_favors_trending_down(self) -> None:
        """大きな負の return は trending_down の確率を上げるべき."""
        f = BayesianRegimeFilter()
        for _ in range(10):
            result = f.update(-5e-4)
        assert result.posterior[RegimeState.TRENDING_DOWN] > result.posterior[RegimeState.TRENDING_UP]
        assert result.map_state == RegimeState.TRENDING_DOWN

    def test_small_return_favors_ranging(self) -> None:
        """小さい return は ranging の確率を上げるべき."""
        f = BayesianRegimeFilter()
        for _ in range(20):
            result = f.update(1e-6)
        # ranging の μ=0, σ=1e-4 が一番 likelihood が高い
        assert result.posterior[RegimeState.RANGING] > 0.3

    def test_large_volatile_returns_favor_high_vol(self) -> None:
        """大きな変動 (正負交互) は high_vol の確率を上げるべき."""
        f = BayesianRegimeFilter()
        for i in range(30):
            sign = 1.0 if i % 2 == 0 else -1.0
            result = f.update(sign * 2e-3)
        assert result.posterior[RegimeState.HIGH_VOL] > 0.2

    def test_nan_observation_unchanged(self) -> None:
        """NaN は posterior を変更しない."""
        f = BayesianRegimeFilter()
        f.update(1e-4)
        post_before = f.posterior.copy()
        result = f.update(float("nan"))
        np.testing.assert_allclose(result.posterior, post_before)

    def test_inf_observation_unchanged(self) -> None:
        """Inf は posterior を変更しない."""
        f = BayesianRegimeFilter()
        f.update(1e-4)
        post_before = f.posterior.copy()
        result = f.update(float("inf"))
        np.testing.assert_allclose(result.posterior, post_before)

    def test_update_count_increments(self) -> None:
        f = BayesianRegimeFilter()
        assert f.update_count == 0
        f.update(0.0)
        assert f.update_count == 1
        f.update(0.0)
        assert f.update_count == 2

    def test_posterior_sums_to_one(self) -> None:
        """100 回更新しても posterior は確率分布."""
        f = BayesianRegimeFilter()
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = f.update(rng.normal(0, 5e-4))
        assert abs(result.posterior.sum() - 1.0) < 1e-10
        assert (result.posterior >= 0).all()


# =====================================================================
# BayesianRegimeResult
# =====================================================================


class TestBayesianRegimeResult:
    """BayesianRegimeResult properties."""

    def test_regime_probabilities(self) -> None:
        f = BayesianRegimeFilter()
        result = f.update(0.0)
        probs = result.regime_probabilities
        assert set(probs.keys()) == {"trending_up", "trending_down", "ranging", "high_vol"}
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_map_regime_str(self) -> None:
        f = BayesianRegimeFilter()
        result = f.update(0.0)
        assert result.map_regime_str in {"trending_up", "trending_down", "ranging", "high_vol"}

    def test_offset_multiplier_range(self) -> None:
        f = BayesianRegimeFilter()
        result = f.update(0.0)
        # 確率加重なので、min(offset_mults) <= mult <= max(offset_mults)
        assert 0.7 <= result.offset_multiplier <= 1.6

    def test_predicted_is_probability(self) -> None:
        f = BayesianRegimeFilter()
        result = f.update(0.0)
        assert result.predicted.shape == (4,)
        assert abs(result.predicted.sum() - 1.0) < 1e-10
        assert (result.predicted >= 0).all()

    def test_to_dict(self) -> None:
        f = BayesianRegimeFilter()
        result = f.update(1e-4)
        d = result.to_dict()
        assert "map_regime" in d
        assert "map_probability" in d
        assert "offset_multiplier" in d
        assert "probabilities" in d


# =====================================================================
# Adaptive Emission
# =====================================================================


class TestAdaptiveEmission:
    """Emission パラメータのオンライン更新."""

    def test_emission_adapts_to_observations(self) -> None:
        """正の return を繰り返すと trending_up の μ が上方修正される."""
        f = BayesianRegimeFilter()
        initial_mu_up = f.emission_mu[RegimeState.TRENDING_UP]
        for _ in range(50):
            f.update(1e-3)
        # trending_up の μ が上方に適応しているはず
        assert f.emission_mu[RegimeState.TRENDING_UP] >= initial_mu_up

    def test_emission_sigma_adapts(self) -> None:
        """大きな変動で σ が上方修正される."""
        f = BayesianRegimeFilter()
        initial_sigma_hv = f.emission_sigma[RegimeState.HIGH_VOL]
        for i in range(100):
            sign = 1.0 if i % 2 == 0 else -1.0
            f.update(sign * 5e-3)
        # 何らかの σ が変化しているはず
        changed = not np.allclose(f.emission_sigma, [3e-4, 3e-4, 1e-4, 1e-3])
        assert changed

    def test_no_adaptation_when_disabled(self) -> None:
        cfg = BayesianRegimeConfig(adaptive_emission=False)
        f = BayesianRegimeFilter(cfg)
        initial_mu = f.emission_mu.copy()
        initial_sigma = f.emission_sigma.copy()
        for _ in range(20):
            f.update(1e-3)
        np.testing.assert_allclose(f.emission_mu, initial_mu)
        np.testing.assert_allclose(f.emission_sigma, initial_sigma)


# =====================================================================
# A 行列再推定
# =====================================================================


class TestTransitionReestimation:
    """遷移行列の定期再推定."""

    def test_reestimation_fires_at_interval(self) -> None:
        """reestimate_interval 回更新後に A 行列が変化する."""
        cfg = BayesianRegimeConfig(reestimate_interval=20, adaptive_emission=False)
        f = BayesianRegimeFilter(cfg)
        initial_A = f.transition_matrix.copy()

        # 20 回の更新で再推定が発火
        rng = np.random.default_rng(123)
        for _ in range(20):
            f.update(rng.normal(0, 5e-4))

        # A 行列が変化しているはず
        assert not np.allclose(f.transition_matrix, initial_A, atol=1e-6)

    def test_reestimation_preserves_row_sums(self) -> None:
        cfg = BayesianRegimeConfig(reestimate_interval=10)
        f = BayesianRegimeFilter(cfg)
        for _ in range(15):
            f.update(1e-4)
        A = f.transition_matrix
        np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-10)


# =====================================================================
# weighted_value
# =====================================================================


class TestWeightedValue:
    """確率加重ユーティリティ."""

    def test_uniform_posterior(self) -> None:
        f = BayesianRegimeFilter()
        values = {
            RegimeState.TRENDING_UP: 1.0,
            RegimeState.TRENDING_DOWN: 1.0,
            RegimeState.RANGING: 0.8,
            RegimeState.HIGH_VOL: 1.5,
        }
        result = f.weighted_value(values)
        expected = 0.25 * (1.0 + 1.0 + 0.8 + 1.5)
        assert abs(result - expected) < 1e-10

    def test_deterministic_posterior(self) -> None:
        """事後確率が 1 状態に集中している場合."""
        prior = np.array([0.0, 0.0, 1.0, 0.0])
        cfg = BayesianRegimeConfig(prior=prior)
        f = BayesianRegimeFilter(cfg)
        values = {
            RegimeState.TRENDING_UP: 10.0,
            RegimeState.TRENDING_DOWN: 20.0,
            RegimeState.RANGING: 30.0,
            RegimeState.HIGH_VOL: 40.0,
        }
        result = f.weighted_value(values)
        assert abs(result - 30.0) < 1e-10


# =====================================================================
# State persistence
# =====================================================================


class TestStatePersistence:
    """get_state / restore_state."""

    def test_round_trip(self) -> None:
        f1 = BayesianRegimeFilter()
        for _ in range(30):
            f1.update(1e-4)
        state = f1.get_state()

        f2 = BayesianRegimeFilter()
        assert f2.restore_state(state)
        np.testing.assert_allclose(f2.posterior, f1.posterior)
        np.testing.assert_allclose(f2.transition_matrix, f1.transition_matrix)
        assert f2.update_count == f1.update_count

    def test_restore_invalid_state(self) -> None:
        f = BayesianRegimeFilter()
        assert f.restore_state({}) is False

    def test_restore_wrong_shape(self) -> None:
        f = BayesianRegimeFilter()
        state = {"posterior": [0.5, 0.5], "transition": [[1]], "emission_mu": [0], "emission_sigma": [1]}
        assert f.restore_state(state) is False


# =====================================================================
# Reset
# =====================================================================


class TestReset:
    """Filter reset."""

    def test_reset_to_uniform(self) -> None:
        f = BayesianRegimeFilter()
        for _ in range(50):
            f.update(1e-3)
        f.reset()
        np.testing.assert_allclose(f.posterior, 0.25, atol=1e-10)
        assert f.update_count == 0

    def test_reset_with_custom_prior(self) -> None:
        prior = np.array([0.4, 0.1, 0.4, 0.1])
        cfg = BayesianRegimeConfig(prior=prior)
        f = BayesianRegimeFilter(cfg)
        for _ in range(50):
            f.update(1e-3)
        f.reset()
        np.testing.assert_allclose(f.posterior, prior, atol=1e-10)


# =====================================================================
# 相互変換マップ
# =====================================================================


class TestConversionMaps:
    """STATE_TO_REGIME_STR / REGIME_STR_TO_STATE."""

    def test_all_states_mapped(self) -> None:
        for state in RegimeState:
            assert state in _STATE_TO_REGIME_STR

    def test_reverse_map(self) -> None:
        for name, state in _REGIME_STR_TO_STATE.items():
            assert _STATE_TO_REGIME_STR[state] == name

    def test_n_states_constant(self) -> None:
        assert _N_STATES == 4
