"""Tests for StrictMaskedPolicy with backend-light tensor assertions."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import torch

    if not getattr(torch, "__file__", None):
        pytest.skip(
            "lightweight torch stub active; strict masked policy requires full torch backend",
            allow_module_level=True,
        )

    from ztb.training.policies.strict_masked_policy import StrictMaskedPolicy
except ImportError:
    pytest.skip(
        "torch or ztb.training.policies.strict_masked_policy module not available",
        allow_module_level=True,
    )



def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "_arr"):
        return np.asarray(getattr(value, "_arr"))
    return np.asarray(value)



def _tensor(value: object) -> object:
    return torch.tensor(value, dtype=getattr(torch, "float32", None))


class _CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)
        self.recorded_features_dim = features_dim

    def forward(self, observations: object) -> object:
        return observations


@pytest.fixture
def simple_observation_space() -> spaces.Box:
    return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)


@pytest.fixture
def simple_action_space() -> spaces.Discrete:
    return spaces.Discrete(3)


@pytest.fixture
def policy(simple_observation_space: spaces.Box, simple_action_space: spaces.Discrete) -> StrictMaskedPolicy:
    lr_schedule = lambda _: 3e-4  # noqa: E731
    return StrictMaskedPolicy(
        observation_space=simple_observation_space,
        action_space=simple_action_space,
        lr_schedule=lr_schedule,
    )


class TestStrictMaskedPolicyInit:
    def test_initialization(self, policy: StrictMaskedPolicy) -> None:
        assert policy is not None
        assert hasattr(policy, "action_net")
        assert hasattr(policy, "value_net")
        assert hasattr(policy, "mlp_extractor")
        assert hasattr(policy, "features_extractor")

    def test_policy_structure(self, policy: StrictMaskedPolicy) -> None:
        sample_obs = torch.randn(1, 10)
        features = policy.extract_features(sample_obs, policy.features_extractor)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)

        assert _as_numpy(features).shape == (1, 10)
        assert _as_numpy(logits).shape == (1, 3)

    def test_custom_feature_extractor_receives_features_dim(
        self,
        simple_observation_space: spaces.Box,
        simple_action_space: spaces.Discrete,
    ) -> None:
        policy = StrictMaskedPolicy(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            lr_schedule=lambda _: 3e-4,
            features_extractor_class=_CustomExtractor,
        )

        assert isinstance(policy.features_extractor, _CustomExtractor)
        assert policy.features_extractor.recorded_features_dim == 10


class TestStrictMaskedPolicyForward:
    def test_forward_without_mask(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(4, 10)
        actions, values, log_probs = policy.forward(obs, deterministic=False)

        assert _as_numpy(actions).shape == (4,)
        assert _as_numpy(values).shape == (4, 1)
        assert _as_numpy(log_probs).shape == (4,)
        assert np.all((_as_numpy(actions) >= 0) & (_as_numpy(actions) < 3))

    def test_forward_with_partial_mask(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(4, 10)
        action_masks = _tensor([[1, 0, 0]] * 4)
        actions, values, log_probs = policy.forward(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )

        assert np.all(_as_numpy(actions) == 0)
        assert _as_numpy(values).shape == (4, 1)
        assert _as_numpy(log_probs).shape == (4,)

    def test_forward_illegal_actions_never_sampled(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(128, 10)
        action_masks = _tensor([[1, 1, 0]] * 128)
        actions, _, _ = policy.forward(obs, deterministic=False, action_masks=action_masks)

        assert np.all(_as_numpy(actions) != 2)

    def test_forward_deterministic_is_stable(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(1, 10)
        action_masks = _tensor([[1, 1, 1]])

        actions = [
            int(_as_numpy(policy.forward(obs, deterministic=True, action_masks=action_masks)[0])[0])
            for _ in range(5)
        ]

        assert len(set(actions)) == 1


class TestStrictMaskedPolicyEvaluateActions:
    def test_evaluate_actions_without_mask(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 1, 2, 0])
        values, log_probs, entropy = policy.evaluate_actions(obs, actions)

        assert _as_numpy(values).shape == (4, 1)
        assert _as_numpy(log_probs).shape == (4,)
        assert _as_numpy(entropy).shape == (4,)

    def test_illegal_action_log_prob_is_very_low(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(4, 10)
        actions = torch.tensor([2, 2, 2, 2])
        action_masks = _tensor([[1, 1, 0]] * 4)
        _, log_probs, _ = policy.evaluate_actions(obs, actions, action_masks)

        assert np.all(_as_numpy(log_probs) < -10)

    def test_restricted_entropy_is_lower(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 0, 0, 0])
        only_hold = _tensor([[1, 0, 0]] * 4)
        all_legal = _tensor([[1, 1, 1]] * 4)

        _, _, entropy_low = policy.evaluate_actions(obs, actions, only_hold)
        _, _, entropy_high = policy.evaluate_actions(obs, actions, all_legal)

        assert np.all(_as_numpy(entropy_low) < _as_numpy(entropy_high))


class TestStrictMaskedPolicyPredictValues:
    def test_predict_values_matches_evaluate_actions(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(4, 10)
        actions = torch.tensor([0, 1, 2, 0])

        values_from_predict = policy.predict_values(obs)
        values_from_evaluate, _, _ = policy.evaluate_actions(obs, actions)

        assert np.allclose(_as_numpy(values_from_predict), _as_numpy(values_from_evaluate), atol=1e-6)


class TestStrictMaskedPolicyIntegration:
    def test_training_step_simulation(self, policy: StrictMaskedPolicy) -> None:
        batch_size = 32
        obs = torch.randn(batch_size, 10)
        action_masks = _tensor(np.random.randint(0, 2, size=(batch_size, 3)).astype(np.float32))
        action_masks_arr = _as_numpy(action_masks)
        action_masks_arr[:, 0] = 1.0
        action_masks = _tensor(action_masks_arr)

        actions, values_forward, log_probs_forward = policy.forward(
            obs,
            deterministic=False,
            action_masks=action_masks,
        )
        values_eval, log_probs_eval, entropy = policy.evaluate_actions(
            obs,
            actions,
            action_masks,
        )

        assert _as_numpy(actions).shape == (batch_size,)
        assert _as_numpy(values_forward).shape == (batch_size, 1)
        assert _as_numpy(log_probs_forward).shape == (batch_size,)
        assert _as_numpy(values_eval).shape == (batch_size, 1)
        assert _as_numpy(log_probs_eval).shape == (batch_size,)
        assert _as_numpy(entropy).shape == (batch_size,)
        assert np.allclose(_as_numpy(values_forward), _as_numpy(values_eval), atol=1e-5)


class TestStrictMaskedPolicyEdgeCases:
    def test_batch_size_one(self, policy: StrictMaskedPolicy) -> None:
        obs = torch.randn(1, 10)
        action_masks = _tensor([[1, 1, 0]])
        actions, values, log_probs = policy.forward(
            obs,
            deterministic=False,
            action_masks=action_masks,
        )

        assert _as_numpy(actions).shape == (1,)
        assert _as_numpy(values).shape == (1, 1)
        assert _as_numpy(log_probs).shape == (1,)
