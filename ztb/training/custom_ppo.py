#!/usr/bin/env python3
"""
Custom PPO with Action Bias Mitigation.

Extends MaskablePPO to integrate:
1. PAN (Per-Action Advantage Normalization)
2. Target Entropy Controller
3. Stratified Mini-batch Sampler
4. Enhanced logging and monitoring

This class overrides the train() method to inject custom components
at the appropriate points in the learning loop.
"""

import sys
from typing import Any, Dict, Optional, Type, TypeVar, Union
import warnings
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor

from ztb.training.adv_norm import PerActionAdvantageNormalizer
from ztb.training.entropy_temperature import TargetEntropyController
from ztb.training.stratified_sampler import StratifiedSampler
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

SelfCustomPPO = TypeVar("SelfCustomPPO", bound="CustomPPO")


class CustomPPO(MaskablePPO):
    """
    Custom PPO with integrated bias mitigation components.
    
    Key modifications from standard MaskablePPO:
    1. train() method overridden to apply PAN to advantages
    2. Stratified sampling replaces uniform batch sampling
    3. Target Entropy Controller dynamically adjusts ent_coef
    4. Enhanced logging for bias monitoring
    """
    
    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        # Standard PPO params
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[Union[float, Schedule]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: Optional[Type] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # Custom params for bias mitigation
        enable_pan: bool = True,
        enable_target_entropy: bool = True,
        enable_stratified_sampling: bool = False,  # Disabled by default (complex integration)
        pan_epsilon: float = 1e-8,
        target_entropy_ratio: float = 0.7,
        lr_temperature: float = 3e-4,
        initial_temperature: float = 0.01,
    ):
        # Initialize parent (note: MaskablePPO doesn't support use_sde and sde_sample_freq)
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        # Store bias mitigation settings
        self.enable_pan = enable_pan
        self.enable_target_entropy = enable_target_entropy
        self.enable_stratified_sampling = enable_stratified_sampling
        
        # Initialize custom components
        self.pan_normalizer: Optional[PerActionAdvantageNormalizer] = None
        self.entropy_controller: Optional[TargetEntropyController] = None
        self.stratified_sampler: Optional[StratifiedSampler] = None
        
        if enable_pan:
            # Determine number of actions from action space
            if isinstance(self.action_space, spaces.Discrete):
                n_actions = self.action_space.n
            else:
                n_actions = 3  # Default for trading (HOLD, BUY, SELL)
            
            self.pan_normalizer = PerActionAdvantageNormalizer(
                n_actions=n_actions,
                epsilon=pan_epsilon,
            )
            logger.info(f"✓ PAN enabled (n_actions={n_actions}, epsilon={pan_epsilon})")
        
        if enable_target_entropy:
            if isinstance(self.action_space, spaces.Discrete):
                n_actions = self.action_space.n
            else:
                n_actions = 3
            
            self.entropy_controller = TargetEntropyController(
                n_actions=n_actions,
                target_entropy_ratio=target_entropy_ratio,
                lr_temperature=lr_temperature,
                initial_temperature=initial_temperature,
            )
            logger.info(f"✓ Target Entropy Controller enabled (target={target_entropy_ratio * np.log(n_actions):.3f})")
        
        if enable_stratified_sampling:
            warnings.warn(
                "Stratified sampling is experimental and may not work correctly with SB3's rollout buffer. "
                "Consider using PAN and Target Entropy first."
            )
            self.stratified_sampler = StratifiedSampler(
                n_actions=n_actions if isinstance(self.action_space, spaces.Discrete) else 3,
                regime_window=20,
                regime_threshold=0.001,
            )
            logger.info("✓ Stratified Sampler enabled (experimental)")
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        
        This method is overridden to integrate:
        1. PAN (Per-Action Advantage Normalization) - before policy loss computation
        2. Target Entropy Controller - dynamic ent_coef adjustment
        3. Stratified Sampling - balanced batch selection (if enabled)
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        
        continue_training = True
        
        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # ★ CUSTOM: Get rollout data (potentially with stratified sampling)
            # Note: Stratified sampling integration is complex with SB3's buffer
            # For now, we use standard sampling but apply PAN to advantages
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                
                # ★ CUSTOM: Apply PAN (Per-Action Advantage Normalization)
                advantages = rollout_data.advantages
                
                if self.enable_pan and self.pan_normalizer is not None:
                    # Convert to numpy for PAN processing
                    advantages_np = advantages.cpu().numpy()
                    actions_np = actions.cpu().numpy()
                    
                    # Apply PAN
                    advantages_np = self.pan_normalizer.normalize(advantages_np, actions_np)
                    
                    # Convert back to torch
                    advantages = th.tensor(advantages_np, device=self.device, dtype=th.float32)
                    
                    # Log PAN statistics
                    pan_stats = self.pan_normalizer.get_statistics()
                    for key, value in pan_stats.items():
                        self.logger.record(f"train/pan_{key}", value)
                else:
                    # Standard advantage normalization (SB3 default)
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Evaluate policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                
                # ★ CUSTOM: Target Entropy Controller
                if self.enable_target_entropy and self.entropy_controller is not None:
                    # Compute current entropy from policy distribution
                    with th.no_grad():
                        # Get action logits for entropy computation
                        distribution = self.policy.get_distribution(rollout_data.observations)
                        action_logits = distribution.distribution.logits
                        
                        # Compute entropy (keep as tensor)
                        current_entropy = self.entropy_controller.compute_entropy(action_logits)
                    
                    # Update temperature (ent_coef) - returns loss and new alpha
                    _, new_alpha = self.entropy_controller.update(current_entropy)
                    
                    # Use controlled entropy coefficient
                    ent_coef_current = new_alpha
                    
                    # Log entropy controller statistics
                    entropy_stats = self.entropy_controller.get_statistics()
                    for key, value in entropy_stats.items():
                        self.logger.record(f"train/entropy_{key}", value)
                else:
                    # Use fixed entropy coefficient
                    ent_coef_current = self.ent_coef
                
                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Value loss
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                
                # Entropy loss (using potentially controlled coefficient)
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                
                entropy_losses.append(entropy_loss.item())
                
                # Total loss
                loss = policy_loss + ent_coef_current * entropy_loss + self.vf_coef * value_loss
                
                # Calculate approximate KL divergence
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                
                # Early stopping based on KL divergence
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            self._n_updates += 1
            if not continue_training:
                break
        
        # Log training statistics
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )
        
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    
    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


if __name__ == "__main__":
    print("CustomPPO module loaded successfully")
    print(f"✓ PAN integration: train() line ~220")
    print(f"✓ Target Entropy integration: train() line ~240")
    print(f"✓ Stratified Sampling: Experimental (not yet integrated)")
