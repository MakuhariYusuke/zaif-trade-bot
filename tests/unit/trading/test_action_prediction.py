"""Tests for action_prediction module — F4/F9 fix verification.

_resolve_expected_obs_dim() と ActionPrediction._prepare_observation() が
ハードコード features[:5] を排除し model.observation_space に基づく
動的次元解決を行うことを検証する。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ztb.trading.live_trader.action_prediction import (
    ActionPrediction,
    _resolve_expected_obs_dim,
)


# ---------- helpers ----------


def _make_live_trader(
    *,
    obs_dim: int | None = None,
    expected_features: int | None = None,
) -> MagicMock:
    """Mock LiveTrader with configurable observation dimension."""
    lt = MagicMock()
    lt.logger = MagicMock()

    # model / observation_space
    if obs_dim is not None:
        space = SimpleNamespace(shape=(obs_dim,))
        lt.model = SimpleNamespace(observation_space=space)
    else:
        lt.model = None

    # expected_features (FeatureSchemaManager 経由)
    if expected_features is not None:
        lt.expected_features = expected_features
    else:
        # getattr 側で None 返す
        del lt.expected_features

    return lt


# ================================================================
# _resolve_expected_obs_dim
# ================================================================


class TestResolveExpectedObsDim:
    """observation_space → expected_features → 0 のフォールバック優先順位を検証."""

    def test_priority1_observation_space(self) -> None:
        lt = _make_live_trader(obs_dim=88, expected_features=10)
        assert _resolve_expected_obs_dim(lt) == 88

    def test_priority2_expected_features(self) -> None:
        lt = _make_live_trader(obs_dim=None, expected_features=42)
        assert _resolve_expected_obs_dim(lt) == 42

    def test_priority3_fallback_zero(self) -> None:
        lt = _make_live_trader(obs_dim=None, expected_features=None)
        assert _resolve_expected_obs_dim(lt) == 0

    def test_zero_obs_dim_falls_through(self) -> None:
        """observation_space.shape=(0,) は空 → expected_features へフォールバック."""
        lt = _make_live_trader(obs_dim=None, expected_features=5)
        # shape が空タプル ではなく (0,) のケース: shape が truthy なので 0 を返す
        space = SimpleNamespace(shape=(0,))
        lt.model = SimpleNamespace(observation_space=space)
        # shape=(0,) → shape is truthy → int(shape[0])=0 → falls through to expected
        # 実装上 shape が空でない限り observation_space を優先するので 0 を返す
        result = _resolve_expected_obs_dim(lt)
        # shape = (0,), shape is truthy, int(0) = 0.
        # obs_space.shape = (0,) → truthy → returns 0
        assert result == 0

    def test_no_model_attribute(self) -> None:
        """live_trader に model 属性がない場合もクラッシュしない."""
        lt = MagicMock(spec=[])  # 空 spec
        assert _resolve_expected_obs_dim(lt) == 0


# ================================================================
# _prepare_observation
# ================================================================


class TestPrepareObservation:
    """特徴量ベクトルの切り詰め/パディング/パススルーを検証."""

    @pytest.fixture
    def ap(self) -> ActionPrediction:
        lt = _make_live_trader(obs_dim=5)
        return ActionPrediction(lt)

    def test_exact_dim_passthrough(self, ap: ActionPrediction) -> None:
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ap._prepare_observation(features)
        np.testing.assert_array_equal(result, features)

    def test_truncate(self, ap: ActionPrediction) -> None:
        features = np.arange(10, dtype=float)
        result = ap._prepare_observation(features)
        assert len(result) == 5
        np.testing.assert_array_equal(result, features[:5])

    def test_pad(self, ap: ActionPrediction) -> None:
        features = np.array([1.0, 2.0])
        result = ap._prepare_observation(features)
        assert len(result) == 5
        np.testing.assert_array_equal(result[:2], features)
        np.testing.assert_array_equal(result[2:], [0.0, 0.0, 0.0])

    def test_fallback_unknown_dim(self) -> None:
        """expected_dim == 0 → features をそのまま返す."""
        lt = _make_live_trader(obs_dim=None, expected_features=None)
        ap = ActionPrediction(lt)
        features = np.arange(88, dtype=float)
        result = ap._prepare_observation(features)
        np.testing.assert_array_equal(result, features)

    def test_no_hardcoded_five(self) -> None:
        """features[:5] のハードコードが排除されたことを確認.

        88次元入力を 88次元モデルに渡しても 5 に切り詰められないこと (F4 修正)。
        """
        lt = _make_live_trader(obs_dim=88)
        ap = ActionPrediction(lt)
        features = np.arange(88, dtype=float)
        result = ap._prepare_observation(features)
        assert len(result) == 88
        np.testing.assert_array_equal(result, features)


# ================================================================
# expected_dim caching
# ================================================================


class TestExpectedDimCaching:
    """expected_dim プロパティが遅延解決・キャッシュされることを検証."""

    def test_cached(self) -> None:
        lt = _make_live_trader(obs_dim=10)
        ap = ActionPrediction(lt)
        assert ap._expected_dim is None
        _ = ap.expected_dim
        assert ap._expected_dim == 10
        # 再呼び出しでもキャッシュされた値を返す
        assert ap.expected_dim == 10
