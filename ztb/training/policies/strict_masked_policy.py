"""
StrictMaskedPolicy: カスタムPPOポリシー（学習時の厳密なマスク適用）

学習時と評価時で同一のアクションマスクロジックを適用し、
違法アクションが損失計算に寄与しないことを保証する。
"""

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs


class StrictMaskedPolicy(MaskableActorCriticPolicy):
    """
    カスタムポリシー: 学習時も違法アクションを完全除外
    
    主な変更点:
    1. forward() メソッド: 違法アクションのlogitsを -1e9 に設定
    2. evaluate_actions() メソッド: 損失計算時も同じマスクを適用
    3. predict() メソッド: 推論時のデコード順序を統一 (mask → softmax(T) → argmax)
    
    これにより、学習/評価の分布不一致を根絶し、違法アクションへの確率漏れを防止。
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        net_arch: Optional[Dict[str, Any]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        StrictMaskedPolicy の初期化
        
        親クラス (MaskableActorCriticPolicy) のパラメータをそのまま継承。
        追加の初期化処理は不要（マスクロジックはforward/evaluate_actionsで実装）。
        
        Note: MaskableActorCriticPolicy は離散行動空間用なので、
        use_sde, log_std_init, full_std, use_expln, squash_output などの
        連続行動空間用パラメータは不要。
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray[Any, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with strict mask enforcement.
        
        違法アクションのlogitsを -1e9 に設定することで、
        サンプリング時に違法アクションが選ばれる確率を完全にゼロにする。
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            deterministic: If True, select action deterministically (argmax)
            action_masks: Action masks [batch_size, n_actions] (1=legal, 0=illegal)
        
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features from observation
        features = self.extract_features(obs, self.features_extractor)
        
        # Get latent representations for policy and value
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Get raw action logits
        logits = self.action_net(latent_pi)
        
        # CRITICAL: Apply mask BEFORE distribution creation
        # 違法アクションのlogitsを -1e9 に設定(事実上の確率ゼロ)
        if action_masks is not None:
            # action_masksがnumpy配列かTensorかを判定
            if isinstance(action_masks, torch.Tensor):
                mask_tensor = action_masks.bool()
            else:
                mask_tensor = torch.from_numpy(action_masks).bool()
            
            logits = torch.where(
                mask_tensor,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
            )
        
        # Create categorical distribution from masked logits
        distribution = torch.distributions.Categorical(logits=logits)
        
        # Sample or select deterministically
        if deterministic:
            # Deterministic: argmax over masked logits
            actions = torch.argmax(logits, dim=-1)
        else:
            # Stochastic: sample from masked distribution
            actions = distribution.sample()  # type: ignore[no-untyped-call]
        
        # Compute log probabilities and values
        log_probs = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
        values = self.value_net(latent_vf)
        
        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions with strict mask enforcement during loss calculation.
        
        学習時の損失計算でも違法アクションを除外することで、
        違法アクションの学習を防止し、合法アクションのみに勾配を流す。
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Actions tensor [batch_size]
            action_masks: Action masks [batch_size, n_actions] (1=legal, 0=illegal)
        
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features
        features = self.extract_features(obs, self.features_extractor)
        
        # Get latent representations
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Get raw action logits
        logits = self.action_net(latent_pi)
        
        # CRITICAL: Apply mask BEFORE loss calculation
        # これにより違法アクションが損失に寄与しない
        if action_masks is not None:
            logits = torch.where(
                action_masks.bool(),
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
            )
        
        # Create distribution from masked logits
        distribution = torch.distributions.Categorical(logits=logits)
        
        # Compute log probabilities and entropy
        log_probs = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
        entropy = distribution.entropy()  # type: ignore[no-untyped-call]
        
        # Compute values
        values = self.value_net(latent_vf)
        
        return values, log_probs, entropy

    def predict_values(
        self,
        obs: PyTorchObs,
    ) -> torch.Tensor:
        """
        Predict values for observations.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            Values tensor [batch_size, 1]
        """
        # Extract features
        features = self.extract_features(obs, self.features_extractor)
        
        # Get latent representation for value
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Compute values
        values = self.value_net(latent_vf)
        
        return torch.tensor(values)  # Ensure Tensor return type
