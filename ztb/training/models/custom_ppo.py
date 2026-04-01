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
from __future__ import annotations

import warnings
from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch.nn import functional as F

from ztb.training.config.lagrange_defaults import LAGRANGE_DEFAULTS
from ztb.training.entropy_temperature import TargetEntropyController
from ztb.training.optimization.adv_norm import PerActionAdvantageNormalizer
from ztb.training.optimization.lagrange_constraint import LagrangeConstraint
from ztb.training.optimization.stratified_sampler import StratifiedSampler

LAGRANGE_DEFAULT_WARMUP_STEPS = int(LAGRANGE_DEFAULTS["warmup_steps"])
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

SelfCustomPPO = TypeVar("SelfCustomPPO", bound="CustomPPO")
ActionSpace: TypeAlias = spaces.Space[Any]


def _resolve_action_space(
    env: GymEnv | str, current_action_space: ActionSpace | None
) -> ActionSpace:
    """Return a concrete action space even when the base class leaves it unset."""
    if current_action_space is not None:
        return current_action_space
    if hasattr(env, "action_space"):
        action_space = getattr(env, "action_space")
        if isinstance(action_space, spaces.Space):
            return cast(ActionSpace, action_space)
    return spaces.Discrete(3)


def _num_actions(action_space: ActionSpace) -> int:
    """Get the discrete action count, or fall back to the trading default."""
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    return 3


def _record_stats(logger_obj: Any, prefix: str, stats: dict[str, float]) -> None:
    for key, value in stats.items():
        logger_obj.record(f"{prefix}_{key}", value)


def _all_legal_action_masks(batch_size: int, n_actions: int) -> NDArray[np.float64]:
    return np.ones((batch_size, n_actions), dtype=np.float64)

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
        policy: str | type[MaskableActorCriticPolicy],
        env: GymEnv | str,
        # Standard PPO params
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: float | Schedule | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: type[Any] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        # Custom params for bias mitigation
        enable_pan: bool = True,
        enable_target_entropy: bool = True,
        enable_stratified_sampling: bool = False,  # Disabled by default (complex integration)
        enable_lagrange: bool = False,  # Lagrange constraint for action balance
        pan_epsilon: float = 1e-8,
        target_entropy_ratio: float = 0.7,
        lr_temperature: float = 3e-4,
        initial_temperature: float = 0.01,
        # Lagrange constraint parameters
        lagrange_target_action: str = "SELL",
        lagrange_r_target: float = LAGRANGE_DEFAULTS["r_target"],
        lagrange_tolerance: float = LAGRANGE_DEFAULTS["tolerance"],
        lagrange_eta: float = LAGRANGE_DEFAULTS["eta"],
        lagrange_lambda_max: float = LAGRANGE_DEFAULTS["lambda_max"],
        lagrange_warmup_steps: int = LAGRANGE_DEFAULT_WARMUP_STEPS,
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
        self.enable_lagrange = enable_lagrange

        # Some local shims and SB3 stubs do not populate action_space reliably.
        self.action_space = _resolve_action_space(
            env, getattr(self, "action_space", None)
        )

        # Initialize custom components
        self.pan_normalizer: PerActionAdvantageNormalizer | None = None
        self.entropy_controller: TargetEntropyController | None = None
        self.stratified_sampler: StratifiedSampler | None = None
        self.lagrange: LagrangeConstraint | None = None

        if enable_pan:
            n_actions = _num_actions(self.action_space)

            self.pan_normalizer = PerActionAdvantageNormalizer(
                n_actions=n_actions,
                epsilon=pan_epsilon,
            )
            logger.info(f"✓ PAN enabled (n_actions={n_actions}, epsilon={pan_epsilon})")

        if enable_target_entropy:
            n_actions = _num_actions(self.action_space)

            self.entropy_controller = TargetEntropyController(
                n_actions=n_actions,
                target_entropy_ratio=target_entropy_ratio,
                lr_temperature=lr_temperature,
                initial_temperature=initial_temperature,
            )
            logger.info(
                f"✓ Target Entropy Controller enabled (target={target_entropy_ratio * np.log(n_actions):.3f})"
            )

        if enable_stratified_sampling:
            warnings.warn(
                "Stratified sampling is experimental and may not work correctly with SB3's rollout buffer. "
                "Consider using PAN and Target Entropy first."
            )
            self.stratified_sampler = StratifiedSampler(
                n_actions=_num_actions(self.action_space),
                regime_window=20,
                regime_threshold=0.001,
            )
            logger.info("✓ Stratified Sampler enabled (experimental)")

        if enable_lagrange:
            self.lagrange = LagrangeConstraint(
                target_action=lagrange_target_action,
                r_target=lagrange_r_target,
                tolerance=lagrange_tolerance,
                eta=lagrange_eta,
                lambda_max=lagrange_lambda_max,
                warmup_steps=lagrange_warmup_steps,
            )
            logger.info(
                f"✓ Lagrange Constraint enabled (target={lagrange_target_action}, r_target={lagrange_r_target:.1%})"
            )

    def _normalize_advantages(
        self,
        advantages: th.Tensor,
        actions: th.Tensor,
    ) -> th.Tensor:
        if self.enable_pan and self.pan_normalizer is not None:
            advantages_np = advantages.cpu().numpy()
            actions_np = actions.cpu().numpy()
            normalized_advantages = self.pan_normalizer.normalize(
                advantages_np, actions_np
            )
            _record_stats(self.logger, "train/pan", self.pan_normalizer.get_statistics())
            return th.tensor(
                normalized_advantages,
                device=self.device,
                dtype=th.float32,
            )

        if self.normalize_advantage and len(advantages) > 1:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _compute_lagrange_penalty(
        self,
        rollout_data: Any,
        actions: th.Tensor,
    ) -> float:
        if not self.enable_lagrange or self.lagrange is None:
            return 0.0

        actions_np = actions.cpu().numpy()
        if hasattr(rollout_data, "action_masks"):
            action_masks_np = rollout_data.action_masks.cpu().numpy()
        else:
            action_masks_np = _all_legal_action_masks(
                batch_size=len(actions_np),
                n_actions=_num_actions(self.action_space),
            )

        penalty_value, lagrange_info = self.lagrange.compute_penalty(
            actions=actions_np,
            legal_masks=action_masks_np,
        )
        _record_stats(self.logger, "train/lagrange", lagrange_info)
        return float(penalty_value)

    def _resolve_entropy_coefficient(
        self,
        observations: th.Tensor,
    ) -> float:
        if not self.enable_target_entropy or self.entropy_controller is None:
            return float(self.ent_coef)

        with th.no_grad():
            distribution = self.policy.get_distribution(observations)
            action_logits = distribution.distribution.logits
            current_entropy = self.entropy_controller.compute_entropy(action_logits)

        _, new_alpha = self.entropy_controller.update(current_entropy)
        _record_stats(
            self.logger,
            "train/entropy",
            self.entropy_controller.get_statistics(),
        )
        return float(new_alpha)

    def _compute_value_loss(
        self,
        *,
        values: th.Tensor,
        rollout_data: Any,
        clip_range_vf: float | None,
    ) -> th.Tensor:
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        return F.mse_loss(rollout_data.returns, values_pred)

    def _compute_entropy_loss(
        self,
        *,
        entropy: th.Tensor | None,
        log_prob: th.Tensor,
    ) -> th.Tensor:
        if entropy is None:
            return -th.mean(-log_prob)
        return -th.mean(entropy)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer with custom enhancements.

        This overridden method integrates several bias mitigation techniques:
        1. PAN (Per-Action Advantage Normalization): Normalizes advantages per action type
        2. Target Entropy Controller: Dynamically adjusts entropy coefficient for optimal exploration
        3. Lagrange Constraint: Applies action balance constraints during training
        4. Enhanced logging: Records detailed statistics for all custom components

        The method processes the rollout buffer in batches, applying custom normalizations
        and constraints before computing policy and value losses. Early stopping is supported
        based on KL divergence thresholds.

        Note:
            This method modifies the standard PPO training loop to include bias mitigation
            components while maintaining compatibility with the base MaskablePPO implementation.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Log training start with memory information
        if th.cuda.is_available():
            memory_allocated = th.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = th.cuda.memory_reserved() / 1024**2  # MB
            self.logger.record("train/gpu_memory_allocated_mb", memory_allocated)
            self.logger.record("train/gpu_memory_reserved_mb", memory_reserved)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

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

                advantages = rollout_data.advantages
                advantages = self._normalize_advantages(advantages, actions)
                lagrange_penalty = self._compute_lagrange_penalty(
                    rollout_data,
                    actions,
                )

                # Evaluate policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                ent_coef_current = self._resolve_entropy_coefficient(
                    rollout_data.observations
                )

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                value_loss = self._compute_value_loss(
                    values=values,
                    rollout_data=rollout_data,
                    clip_range_vf=clip_range_vf if self.clip_range_vf is not None else None,
                )
                value_losses.append(value_loss.item())

                entropy_loss = self._compute_entropy_loss(
                    entropy=entropy,
                    log_prob=log_prob,
                )
                entropy_losses.append(entropy_loss.item())

                # Total loss
                loss = (
                    policy_loss
                    + ent_coef_current * entropy_loss
                    + self.vf_coef * value_loss
                )

                # ★ CUSTOM: Add Lagrange penalty to total loss
                # Penalty is positive when constraint is violated, so we ADD it to minimize
                if self.enable_lagrange and lagrange_penalty != 0.0:
                    lagrange_penalty_tensor = th.tensor(
                        lagrange_penalty, device=self.device, dtype=th.float32
                    )
                    loss = loss + lagrange_penalty_tensor

                # Calculate approximate KL divergence
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                # Early stopping based on KL divergence
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Log training statistics
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(th.tensor(self.policy.log_std)).mean().item()
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

def explained_variance(
    y_pred: NDArray[np.float32], y_true: NDArray[np.float32]
) -> float:
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
    return float(np.nan) if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

if __name__ == "__main__":
    print("CustomPPO module loaded successfully")
    print("✓ PAN integration: train() line ~220")
    print("✓ Target Entropy integration: train() line ~240")
    print("✓ Stratified Sampling: Experimental (not yet integrated)")
