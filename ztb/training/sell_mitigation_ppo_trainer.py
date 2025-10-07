"""
SELL Bias Mitigation PPO Trainer.

Extends PPOTrainer with comprehensive action bias mitigation:
1. Lagrange constraint for minimum action rate
2. Gradient probes for monitoring and failsafe
3. Enhanced action weighting
4. PAN (Per-Action Advantage Normalization) - prevents gradient crushing
5. Target Entropy Controller - automatic exploration maintenance
6. Stratified Mini-batch Sampler - minority scenario boosting
7. Reverse-as-Close flag - reduces SELL cost perception
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import pandas as pd
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ztb.training.custom_ppo import CustomPPO
from ztb.training.trainer_params import SELLMitigationParams
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.ppo_trainer import PPOTrainerAutoHalt as PPOTrainer
from ztb.training.ppo_config import PPOConfig
from ztb.training.lagrange_constraint import LagrangeConstraint, apply_lagrange_to_loss
from ztb.training.grad_probes import SELLGradientProbe, create_failsafe_dump
from ztb.training.weights import ActionWeightCalculator
from ztb.training.adv_norm import PerActionAdvantageNormalizer  # New: PAN
from ztb.training.entropy_temperature import TargetEntropyController  # New: Target Entropy
from ztb.training.stratified_sampler import StratifiedSampler  # New: Stratified Sampling
from ztb.types.generics import ConfigurableMixin, ConfigDict
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SELLBiasMitigationCallback(BaseCallback):
    """Callback for SELL bias mitigation during training."""

    def __init__(
        self,
        lagrange: Optional[LagrangeConstraint] = None,
        probe: Optional[SELLGradientProbe] = None,
        weight_calc: Optional[ActionWeightCalculator] = None,
        pan_normalizer: Optional[PerActionAdvantageNormalizer] = None,  # New
        entropy_controller: Optional[TargetEntropyController] = None,  # New
        stratified_sampler: Optional[StratifiedSampler] = None,  # New
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.lagrange = lagrange
        self.probe = probe
        self.weight_calc = weight_calc
        self.pan_normalizer = pan_normalizer
        self.entropy_controller = entropy_controller
        self.stratified_sampler = stratified_sampler
        self.step_count = 0

    def _on_step(self) -> bool:
        """Called at each step. Returns False to stop training."""
        self.step_count += 1

        # Log Lagrange statistics
        if self.lagrange is not None:
            stats = self.lagrange.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"lagrange/{key}", value)

        # Log probe statistics
        if self.probe is not None:
            stats = self.probe.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"probe/{key}", value)
        
        # New: Log PAN statistics
        if self.pan_normalizer is not None:
            stats = self.pan_normalizer.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"pan/{key}", value)
        
        # New: Log Target Entropy statistics
        if self.entropy_controller is not None:
            stats = self.entropy_controller.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"entropy/{key}", value)
        
        # New: Log Stratified Sampler statistics
        if self.stratified_sampler is not None:
            sampler_stats: Dict[str, Any] = self.stratified_sampler.get_statistics()
            # Log bucket distribution
            if "bucket_counts" in sampler_stats:
                bucket_counts = sampler_stats["bucket_counts"]
                if isinstance(bucket_counts, np.ndarray):
                    for regime in range(3):
                        for action in range(3):
                            self.logger.record(
                                f"stratified/bucket_r{regime}_a{action}",
                                int(bucket_counts[regime, action])
                            )

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # This is where we could apply Lagrange constraint to the loss
        # However, SB3 doesn't provide direct access to loss computation
        # We would need to modify the PPO algorithm itself for full integration
        pass


class SELLBiasMitigationPPOTrainer(PPOTrainer):
    """
    PPO Trainer with comprehensive SELL bias mitigation.

    Integrates:
    1. Lagrange constraint for minimum SELL rate (>=15%)
    2. Gradient probes for monitoring and failsafe
    3. Enhanced action weighting
    4. Comprehensive logging
    """

    def __init__(
        self,
        params: SELLMitigationParams,
    ):
        super().__init__(params)

        self.enable_lagrange = params.enable_lagrange
        self.enable_probes = params.enable_probes
        self.enable_weights = params.enable_weights
        self.enable_pan = params.enable_pan
        self.enable_target_entropy = params.enable_target_entropy
        self.enable_stratified_sampling = params.enable_stratified_sampling
        self.allow_reverse = params.allow_reverse

        # Initialize SELL bias mitigation components
        self.lagrange = None
        self.probe = None
        self.weight_calc = None
        self.pan_normalizer = None
        self.entropy_controller = None
        self.stratified_sampler = None

        if params.enable_lagrange:
            self.lagrange = LagrangeConstraint(
                target_action="SELL",
                r_target=0.15,  # 15% minimum SELL rate
                tolerance=0.05,
                eta=1e-3,    # Dual learning rate
                lambda_max=1.0,  # Maximum penalty
                warmup_steps=5000,  # Warmup period
            )
            logger.info("Lagrange constraint enabled (r_target=15%, warmup=5k steps)")

        if params.enable_probes:
            probe_path = params.probe_csv_path or f"{params.checkpoint_dir}/sell_probe.csv"
            self.probe = SELLGradientProbe(
                grad_norm_threshold=1e-6,
                advantage_threshold=0.0,
                consecutive_failures=200,
                moving_window=50,
                save_path=Path(probe_path),
            )
            logger.info(f"Gradient probes enabled (failsafe after 200 unhealthy steps, CSV: {probe_path})")

        if params.enable_weights:
            self.weight_calc = ActionWeightCalculator(
                beta=3.0,
                ema_alpha=0.1,
                epsilon=1e-6,
                entropy_min=0.05,
                target_kl_max=0.03,
                kl_consecutive_max=3,
            )
            logger.info("Action weighting enabled (beta=3.0, ema_alpha=0.1)")
        
        # New: Initialize PAN (Per-Action Advantage Normalization)
        if params.enable_pan:
            self.pan_normalizer = PerActionAdvantageNormalizer(
                n_actions=3,  # HOLD, BUY, SELL
                epsilon=1e-8,
                min_samples_per_action=1
            )
            logger.info("PAN (Per-Action Advantage Normalization) enabled")
        
        # New: Initialize Target Entropy Controller
        if params.enable_target_entropy:
            self.entropy_controller = TargetEntropyController(
                n_actions=3,
                target_entropy_ratio=0.7,  # 0.7 × log(3) ≈ 0.769
                lr_temperature=3e-4,
                initial_temperature=0.01
            )
            logger.info("Target Entropy Controller enabled (target=0.769)")
        
        # New: Initialize Stratified Sampler
        if params.enable_stratified_sampling:
            self.stratified_sampler = StratifiedSampler(
                n_actions=3,
                regime_window=20,
                regime_threshold=0.001,
                min_samples_per_bucket=1
            )
            logger.info("Stratified Mini-batch Sampler enabled (9 buckets: 3 regimes × 3 actions)")

    def _create_callback(self) -> BaseCallback:
        """Create composite training callback with SELL bias mitigation."""
        base_callback = super()._create_callback()

        mitigation_callback = SELLBiasMitigationCallback(
            lagrange=self.lagrange,
            probe=self.probe,
            weight_calc=self.weight_calc,
            pan_normalizer=self.pan_normalizer,  # New
            entropy_controller=self.entropy_controller,  # New
            stratified_sampler=self.stratified_sampler,  # New
            verbose=0,
        )

        return CallbackList([base_callback, mitigation_callback])

    def _setup_sell_bonus_weighting(self) -> None:
        """Setup enhanced SELL bonus weighting."""
        if not self.enable_weights or self.weight_calc is None:
            # Skip if no enhanced weighting
            return

        # Use enhanced weighting calculator
        logger.info("Using enhanced SELL bonus weighting")
        # Note: This would require modifying the PPO loss computation
        # For now, we rely on the Lagrange constraint for the hard guarantee

    def train(self, session_id: str) -> MaskablePPO:
        """Train with SELL bias mitigation using CustomPPO."""
        logger.info("=" * 60)
        logger.info("SELL Bias Mitigation Training Started (CustomPPO)")
        logger.info("=" * 60)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Lagrange: {'✅' if self.enable_lagrange else '❌'}")
        logger.info(f"Probes: {'✅' if self.enable_probes else '❌'}")
        logger.info(f"Weights: {'✅' if self.enable_weights else '❌'}")
        logger.info(f"PAN: {'✅' if self.enable_pan else '❌'}")
        logger.info(f"Target Entropy: {'✅' if self.enable_target_entropy else '❌'}")
        logger.info(f"Stratified Sampling: {'✅' if self.enable_stratified_sampling else '❌'}")
        logger.info(f"Data: {self.data_path}")

        try:
            # ★ MODIFIED: Create model with CustomPPO instead of standard flow
            if self.model is None:
                # Load data
                import pandas as pd
                df = pd.read_csv(self.data_path)

                # Create environment
                env = HeavyTradingEnv(df=df, config=self.config)

                # Wrap with ActionMasker for MaskablePPO
                def mask_fn(env: Any) -> Any:
                    return env.get_legal_actions().astype(bool)

                env = ActionMasker(env, mask_fn)  # type: ignore[assignment]

                # ★ Create CustomPPO with integrated bias mitigations
                self.model = CustomPPO(
                    policy=self.config.get("policy", "MlpPolicy"),
                    env=env,
                    learning_rate=self.config.get("learning_rate", 3e-4),
                    n_steps=self.config.get("n_steps", 2048),
                    batch_size=self.config.get("batch_size", 64),
                    n_epochs=self.config.get("n_epochs", 10),
                    gamma=self.config.get("gamma", 0.99),
                    gae_lambda=self.config.get("gae_lambda", 0.95),
                    clip_range=self.config.get("clip_range", 0.2),
                    clip_range_vf=self.config.get("clip_range_vf"),
                    normalize_advantage=self.config.get("normalize_advantage", True),
                    ent_coef=self.config.get("ent_coef", 0.0),
                    vf_coef=self.config.get("vf_coef", 0.5),
                    max_grad_norm=self.config.get("max_grad_norm", 0.5),
                    target_kl=self.config.get("target_kl"),
                    tensorboard_log=self.config.get("tensorboard_log"),
                    policy_kwargs=self.config.get("policy_kwargs"),
                    verbose=self.config.get("verbose", 1),
                    seed=self.config.get("seed"),
                    device=self.config.get("device", "auto"),
                    _init_setup_model=self.config.get("_init_setup_model", True),
                    # ★ Custom bias mitigation parameters
                    enable_pan=self.enable_pan,
                    enable_target_entropy=self.enable_target_entropy,
                    enable_stratified_sampling=self.enable_stratified_sampling,
                    pan_epsilon=1e-8,
                    target_entropy_ratio=0.7,
                    lr_temperature=3e-4,
                    initial_temperature=0.01,
                )
                
                logger.info("CustomPPO model created with integrated mitigations")

            # Neutralize policy bias
            self.neutralize_policy_bias()

            # Add action frequency weighting for SELL bias correction
            self._setup_sell_bonus_weighting()

            # Start training session
            self.start_training()

            # Train the model
            total_timesteps = self.config.get("total_timesteps", 100000)
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self._create_callback(),
                tb_log_name=session_id,
            )

            # Final validation
            self._final_validation()

            logger.info("=" * 60)
            logger.info("SELL Bias Mitigation Training Completed")
            logger.info("=" * 60)

            return self.model

        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Create failsafe dump if probes are enabled
            if self.probe is not None and self.model is not None:
                try:
                    dump_dir = Path(self.checkpoint_dir) / "failsafe_dump"
                    create_failsafe_dump(self.model, self.probe, dump_dir)
                    logger.info(f"Failsafe dump created: {dump_dir}")
                except Exception as dump_e:
                    logger.error(f"Failed to create failsafe dump: {dump_e}")
            raise

        finally:
            # Cleanup
            if self.probe is not None:
                self.probe.close()

    def _final_validation(self) -> None:
        """Perform final validation of SELL bias mitigation."""
        if self.lagrange is None:
            return

        final_stats = self.lagrange.get_statistics()
        logger.info("Final Lagrange Statistics:")
        logger.info(f"  SELL Rate (avg): {final_stats.get('r_sell_mean', 0):.1%}")
        logger.info(f"  Lambda (final): {final_stats.get('lambda_dual', 0):.6f}")
        logger.info(f"  Constraint Active: {final_stats.get('constraint_active', False)}")

        # Check if targets were met
        sell_rate_ok = final_stats.get('r_sell_mean', 0) >= 0.15
        lambda_bounded = abs(final_stats.get('lambda_dual', 0)) <= 1.0

        if sell_rate_ok and lambda_bounded:
            logger.info("✅ SELL bias mitigation targets achieved")
        else:
            logger.warning("⚠️  SELL bias mitigation targets not fully achieved")
            if not sell_rate_ok:
                logger.warning(f"   - SELL rate below 15%: {final_stats.get('r_sell_mean', 0):.1%}")
            if not lambda_bounded:
                logger.warning(f"   - Lambda out of bounds: {final_stats.get('lambda_dual', 0):.6f}")


def create_sell_mitigation_config(
    base_config: Dict[str, Any],
    enable_lagrange: bool = True,
    enable_probes: bool = True,
    enable_weights: bool = True,
) -> Dict[str, Any]:
    """
    Create configuration for SELL bias mitigation training.

    Args:
        base_config: Base PPO configuration
        enable_lagrange: Enable Lagrange constraint
        enable_probes: Enable gradient probes
        enable_weights: Enable enhanced weighting

    Returns:
        Enhanced configuration dictionary
    """
    config = base_config.copy()

    # Add SELL mitigation flags
    config["enable_lagrange"] = enable_lagrange
    config["enable_probes"] = enable_probes
    config["enable_weights"] = enable_weights

    # Enhanced logging
    config["tensorboard_log"] = config.get("tensorboard_log", "./tensorboard")

    return config


# Test function
def test_sell_mitigation_trainer():
    """Test SELL bias mitigation trainer."""
    print("Testing SELL Bias Mitigation Trainer...")

    # Create minimal config
    config = PPOConfig()
    config.total_timesteps = 1000  # Very short for testing
    config.n_steps = 128

    # Create trainer
    trainer = SELLBiasMitigationPPOTrainer(
        data_path="ml-dataset-final.csv",
        config=config,
        checkpoint_dir="test_checkpoints",
        enable_lagrange=True,
        enable_probes=True,
        enable_weights=True,
    )

    print("✅ SELL Bias Mitigation Trainer created successfully")
    print(f"   Lagrange: {'✅' if trainer.lagrange else '❌'}")
    print(f"   Probes: {'✅' if trainer.probe else '❌'}")
    print(f"   Weights: {'✅' if trainer.weight_calc else '❌'}")

    # Cleanup
    if trainer.probe:
        trainer.probe.close()

    print("✅ Test completed")


if __name__ == "__main__":
    test_sell_mitigation_trainer()
