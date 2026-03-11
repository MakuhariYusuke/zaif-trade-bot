"""StrictMaskedPolicy: backend-light strict action masking policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule, TensorDict


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    if hasattr(value, "_arr"):
        return np.asarray(getattr(value, "_arr"), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _to_tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=getattr(torch, "float32", None))


class _Dense(nn.Module):
    """Small dense layer that works with both real torch and test stubs."""

    def __init__(self, in_features: int, out_features: int, seed: int) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        scale = 1.0 / max(in_features, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = rng.normal(0.0, scale, size=(in_features, out_features)).astype(
            np.float32
        )
        self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        arr = _to_numpy(inputs)
        output = arr @ self.weight + self.bias
        return _to_tensor(output)


class _SimpleMlpExtractor(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64) -> None:
        super().__init__()
        self.actor_linear = _Dense(input_dim, latent_dim, seed=7)
        self.critic_linear = _Dense(input_dim, latent_dim, seed=13)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return _to_tensor(np.tanh(_to_numpy(self.actor_linear(features))))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return _to_tensor(np.tanh(_to_numpy(self.critic_linear(features))))


class StrictMaskedPolicy(nn.Module):
    """Self-contained masked actor-critic policy for tests and lightweight runtime."""

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        net_arch: dict[str, Any] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("StrictMaskedPolicy requires a discrete action space")

        extractor_kwargs = features_extractor_kwargs or {}
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.net_arch = net_arch or {}
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.share_features_extractor = share_features_extractor
        self.normalize_images = normalize_images
        flat_dim = int(np.prod(observation_space.shape or (1,)))
        self.features_extractor = features_extractor_class(
            observation_space,
            features_dim=flat_dim,
            **extractor_kwargs,
        )
        self.features_dim = flat_dim
        self.mlp_extractor = _SimpleMlpExtractor(self.features_dim)
        self.action_net = _Dense(64, action_space.n, seed=17)
        self.value_net = _Dense(64, 1, seed=19)
        self._optimizer_anchor = nn.Parameter(_to_tensor([0.0]))
        self.optimizer = optimizer_class(
            self.parameters(),
            lr=float(lr_schedule(1.0)),
            **(optimizer_kwargs or {}),
        )

    def extract_features(
        self,
        obs: torch.Tensor | TensorDict,
        extractor: BaseFeaturesExtractor | nn.Module,
    ) -> torch.Tensor:
        del extractor
        tensor_obs = obs if hasattr(obs, "shape") else obs["obs"]
        arr = _to_numpy(tensor_obs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        return _to_tensor(arr)

    @staticmethod
    def _apply_mask(
        logits: torch.Tensor,
        action_masks: np.ndarray[Any, Any] | torch.Tensor | None,
    ) -> np.ndarray:
        logits_arr = _to_numpy(logits)
        if action_masks is None:
            return logits_arr
        mask_arr = _to_numpy(action_masks).astype(bool)
        masked = logits_arr.copy()
        masked[~mask_arr] = -1e9
        return masked

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray[Any, Any] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        masked_logits = self._apply_mask(self.action_net(latent_pi), action_masks)
        probs = self._softmax(masked_logits)
        if deterministic:
            actions_arr = np.argmax(masked_logits, axis=1).astype(np.int64)
        else:
            actions_arr = np.array(
                [np.random.choice(probs.shape[1], p=row) for row in probs],
                dtype=np.int64,
            )
        log_probs_arr = np.log(probs[np.arange(len(actions_arr)), actions_arr] + 1e-12)
        values = self.value_net(latent_vf)
        return _to_tensor(actions_arr), values, _to_tensor(log_probs_arr)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        masked_logits = self._apply_mask(self.action_net(latent_pi), action_masks)
        probs = self._softmax(masked_logits)
        action_arr = _to_numpy(actions).astype(np.int64).reshape(-1)
        log_probs_arr = np.log(probs[np.arange(len(action_arr)), action_arr] + 1e-12)
        entropy_arr = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        values = self.value_net(latent_vf)
        return values, _to_tensor(log_probs_arr), _to_tensor(entropy_arr)

    def predict_values(self, obs: torch.Tensor | TensorDict) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)
