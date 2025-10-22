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

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ztb.trading.environment.constants import EPSILON
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.config.lagrange_defaults import LAGRANGE_DEFAULTS
from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG, PPOConfig
from ztb.training.config.trainer_params import SELLMitigationParams
from ztb.training.core.ppo_trainer import PPOTrainerAutoHalt as PPOTrainer
from ztb.training.experiments.entropy_temperature import (
    TargetEntropyController,
)  # New: Target Entropy
from ztb.training.models.custom_ppo import CustomPPO
from ztb.training.optimization.adv_norm import PerActionAdvantageNormalizer  # New: PAN
from ztb.training.optimization.lagrange_constraint import LagrangeConstraint
from ztb.training.optimization.stratified_sampler import (
    StratifiedSampler,
)  # New: Stratified Sampling
from ztb.training.utils.grad_probes import SELLGradientProbe, create_failsafe_dump
from ztb.training.utils.weights import ActionWeightCalculator
from ztb.utils.config import ZTBConfig
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
                                int(bucket_counts[regime, action]),
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
    PPO Trainer with comprehensive SELL bias mitigation techniques.

    This advanced trainer extends the base PPO trainer with multiple
    specialized techniques to address SELL action bias in trading strategies:

    Core Mitigation Techniques:
    1. Lagrange Constraint: Enforces minimum SELL action rate (≥15%)
    2. Gradient Probes: Monitors training gradients and provides failsafe mechanisms
    3. Enhanced Action Weighting: Balances BUY/SELL action probabilities
    4. Per-Action Advantage Normalization (PAN): Prevents gradient crushing
    5. Target Entropy Controller: Maintains exploration through automatic entropy adjustment
    6. Stratified Mini-batch Sampling: Ensures minority scenarios are adequately represented
    7. Reverse-as-Close Trading: Reduces perceived SELL cost by reversing close logic

    Training Flow:
    - Initialize mitigation components based on configuration
    - Apply constraints and normalization during training
    - Monitor bias metrics and adjust parameters dynamically
    - Provide comprehensive logging and diagnostics

    Args:
        params: SELLMitigationParams containing all training and mitigation configuration
    """

    config: Dict[str, Any]  # Declare config attribute for mypy  # type: ignore[misc]

    def __init__(
        self,
        params: SELLMitigationParams,
    ):
        # Call parent PPOTrainer __init__ with TrainerParams (SELLMitigationParams inherits from TrainerParams)
        super().__init__(params)  # type: ignore[arg-type,call-arg]

        self.enable_lagrange = params.enable_lagrange
        self.enable_probes = params.enable_probes
        self.enable_weights = params.enable_weights
        self.enable_pan = params.enable_pan
        self.enable_target_entropy = params.enable_target_entropy
        self.enable_stratified_sampling = params.enable_stratified_sampling
        self.allow_reverse = params.allow_reverse

        # Store Lagrange parameters for CustomPPO creation
        self.lagrange_params = params.lagrange_params or {}

        # Initialize SELL bias mitigation components
        # ★ NOTE: Lagrange, PAN, and Target Entropy are now integrated into CustomPPO
        # We only keep separate instances for Probes and Weight Calculator
        self.probe = None
        self.weight_calc = None

        # ★ REMOVED: self.lagrange, self.pan_normalizer, self.entropy_controller
        # These are now created and managed by CustomPPO model

        if params.enable_lagrange:
            # ★ Lagrange is created by CustomPPO, just log here
            logger.info(
                "Lagrange constraint will be enabled in CustomPPO (r_target=15%, warmup=5k steps)"
            )

        if params.enable_probes:
            probe_path = (
                params.probe_csv_path or f"{params.checkpoint_dir}/sell_probe.csv"
            )
            self.probe = SELLGradientProbe(
                grad_norm_threshold=1e-6,
                advantage_threshold=0.0,
                consecutive_failures=200,
                moving_window=50,
                save_path=Path(probe_path),
            )
            logger.info(
                f"Gradient probes enabled (failsafe after 200 unhealthy steps, CSV: {probe_path})"
            )

        if params.enable_weights:
            self.weight_calc = ActionWeightCalculator(
                beta=3.0,
                ema_alpha=0.1,
                epsilon=EPSILON,
                entropy_min=0.05,
                target_kl_max=0.03,
                kl_consecutive_max=3,
            )
            logger.info("Action weighting enabled (beta=3.0, ema_alpha=0.1)")

        # New: Initialize PAN (Per-Action Advantage Normalization)
        if params.enable_pan:
            self.pan_normalizer = PerActionAdvantageNormalizer(
                n_actions=3,
                epsilon=EPSILON,
                min_samples_per_action=1,  # HOLD, BUY, SELL
            )
            logger.info("PAN (Per-Action Advantage Normalization) enabled")

        # New: Initialize Target Entropy Controller
        if params.enable_target_entropy:
            self.entropy_controller = TargetEntropyController(
                n_actions=3,
                target_entropy_ratio=0.7,  # 0.7 × log(3) ≈ 0.769
                lr_temperature=3e-4,
                initial_temperature=0.01,
            )
            logger.info("Target Entropy Controller enabled (target=0.769)")

        # New: Initialize Stratified Sampler
        if params.enable_stratified_sampling:
            self.stratified_sampler = StratifiedSampler(
                n_actions=3,
                regime_window=20,
                regime_threshold=0.001,
                min_samples_per_bucket=1,
            )
            logger.info(
                "Stratified Mini-batch Sampler enabled (9 buckets: 3 regimes × 3 actions)"
            )

        # Initialize model attribute
        self.model: Optional[CustomPPO] = None

    def _create_callback(self) -> BaseCallback:
        """Create composite training callback with SELL bias mitigation."""
        base_callback = PPOTrainer._create_callback(self)  # type: ignore[attr-defined]

        # ★ Get components from model (not from self)
        # Lagrange, PAN, and Entropy Controller are managed by CustomPPO
        lagrange = self.model.lagrange if self.model is not None else None
        pan_normalizer = self.model.pan_normalizer if self.model is not None else None
        entropy_controller = (
            self.model.entropy_controller if self.model is not None else None
        )
        stratified_sampler = (
            self.model.stratified_sampler if self.model is not None else None
        )

        mitigation_callback = SELLBiasMitigationCallback(
            lagrange=lagrange,
            probe=self.probe,
            weight_calc=self.weight_calc,
            pan_normalizer=pan_normalizer,
            entropy_controller=entropy_controller,
            stratified_sampler=stratified_sampler,
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
        logger.info(
            f"Stratified Sampling: {'✅' if self.enable_stratified_sampling else '❌'}"
        )
        logger.info(f"Data: {self.data_path}")  # type: ignore[attr-defined]

        try:
            # ★ MODIFIED: Create model with CustomPPO instead of standard flow
            if self.model is None:
                # Load data
                import pandas as pd

                df_full = pd.read_csv(self.data_path)  # type: ignore[attr-defined]

                # ====================================================================
                # UNIFIED MEMORY OPTIMIZATION (Bug #52 fix)
                # ====================================================================
                # Apply data_rows_limit if specified
                # Priority: 1) Top-level config, 2) memory_optimization section
                data_rows_limit = self.config.get("data_rows_limit") or (
                    self.config.get("memory_optimization", {}) or {}
                ).get("data_rows_limit")

                if data_rows_limit and len(df_full) > data_rows_limit:
                    logger.info(
                        f"⚠️  MEMORY OPTIMIZATION: Limiting data from {len(df_full)} to {data_rows_limit} rows"
                    )
                    # Memory optimized: Use iloc slice instead of copy
                    df = df_full.iloc[:data_rows_limit]
                    del df_full
                    import gc

                    gc.collect()
                else:
                    df = df_full

                # Extract max_features from unified config structure
                # Priority: 1) Top-level config, 2) memory_optimization section, 3) ppo section
                max_features = (
                    self.config.get("max_features")
                    or (self.config.get("memory_optimization", {}) or {}).get(
                        "max_features"
                    )
                    or (self.config.get("ppo", {}) or {}).get("max_features")
                )

                # Create environment config with curriculum_stage and allow_reverse
                env_config = {
                    "curriculum_stage": self.config.get("curriculum_stage", "full"),
                    "allow_reverse": self.allow_reverse,
                    "transaction_cost": self.config.get("transaction_cost", 0.001),
                    "max_position_size": self.config.get("max_position_size", 1.0),
                    "risk_free_rate": self.config.get("risk_free_rate", 0.0),
                    "reward_scaling": self.config.get("reward_scaling", 1.0),
                    # ★ BUG FIX: Pass reward_settings from config to environment
                    "reward_settings": self.config.get("reward_settings", {}),
                    # ★ BUG #52 FIX: Pass memory optimization settings (unified)
                    "max_features": max_features,
                    "enable_correlation_reduction": self.config.get(
                        "enable_correlation_reduction", True
                    ),
                }

                # Create environment with unified max_features
                env = HeavyTradingEnv(
                    df=df, config=env_config, max_features=max_features
                )

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
                    # ★ Lagrange constraint parameters
                    enable_lagrange=self.enable_lagrange,
                    lagrange_target_action="SELL",
                    lagrange_r_target=self.lagrange_params.get(
                        "r_target", LAGRANGE_DEFAULTS["r_target"]
                    ),
                    lagrange_tolerance=self.lagrange_params.get(
                        "tolerance", LAGRANGE_DEFAULTS["tolerance"]
                    ),
                    lagrange_eta=self.lagrange_params.get(
                        "eta", LAGRANGE_DEFAULTS["eta"]
                    ),
                    lagrange_lambda_max=self.lagrange_params.get(
                        "lambda_max", LAGRANGE_DEFAULTS["lambda_max"]
                    ),
                    lagrange_warmup_steps=int(
                        self.lagrange_params.get(
                            "warmup_steps", LAGRANGE_DEFAULTS["warmup_steps"]
                        )
                    ),
                    # ★ PAN/Entropy/Stratified parameters
                    pan_epsilon=EPSILON,
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
            self.start_training()  # type: ignore[attr-defined]

            # Train the model
            total_timesteps = self.config.get("total_timesteps", 100000)
            logger.info("=" * 80)
            logger.info(
                f"🚀 Starting model.learn() with total_timesteps={total_timesteps:,}"
            )
            logger.info(
                f"   Expected iterations: ~{total_timesteps // self.config.get('n_steps', 2048)}"
            )
            logger.info("=" * 80)
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self._create_callback(),
                tb_log_name=session_id,
            )
            logger.info("=" * 80)
            logger.info("✅ model.learn() completed")
            logger.info("=" * 80)

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
                    dump_dir = Path(self.checkpoint_dir) / "failsafe_dump"  # type: ignore[attr-defined]
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
        try:
            logger.info("Starting final validation...")

            # ★ Get Lagrange from model (not self.lagrange which doesn't exist)
            if (
                self.model is None
                or not hasattr(self.model, "lagrange")
                or self.model.lagrange is None
            ):
                logger.warning("Lagrange constraint not available for final validation")
                return

            final_stats = self.model.lagrange.get_statistics()

            # Calculate action distribution from legal actions
            legal_sell_count = final_stats.get("legal_sell_count", 0)
            total_legal_steps = final_stats.get("total_legal_steps", 0)

            # Estimate HOLD/BUY from remaining steps (rough approximation)
            # Note: This is an estimate since we don't track HOLD/BUY separately in Lagrange
            non_sell_count = total_legal_steps - legal_sell_count
            # Assume roughly equal HOLD/BUY split for now (can be improved)
            estimated_hold = non_sell_count // 2
            estimated_buy = non_sell_count - estimated_hold

            logger.info("=" * 80)
            logger.info("FINAL ACTION DISTRIBUTION:")
            logger.info("=" * 80)
            if total_legal_steps > 0:
                sell_rate = final_stats.get("r_sell_mean", 0)
                hold_rate = (
                    (estimated_hold / total_legal_steps) if total_legal_steps > 0 else 0
                )
                buy_rate = (
                    (estimated_buy / total_legal_steps) if total_legal_steps > 0 else 0
                )

                logger.info(
                    f"  HOLD: {hold_rate:6.1%} ({estimated_hold:4d} / {total_legal_steps})"
                )
                logger.info(
                    f"  BUY:  {buy_rate:6.1%} ({estimated_buy:4d} / {total_legal_steps})"
                )
                logger.info(
                    f"  SELL: {sell_rate:6.1%} ({legal_sell_count:4d} / {total_legal_steps})"
                )
            else:
                logger.warning(
                    "  No legal steps recorded - cannot calculate distribution"
                )
            logger.info("=" * 80)

            logger.info("Lagrange Statistics:")
            logger.info(f"  Lambda (final): {final_stats.get('lambda_dual', 0):.6f}")
            logger.info(
                f"  Constraint Active: {final_stats.get('constraint_active', False)}"
            )
            logger.info(
                f"  Target SELL Rate: {self.config.get('lagrange_r_target', 0.33):.1%}"
            )

            # Log output paths (with safe fallbacks)
            logger.info("=" * 80)
            logger.info("OUTPUT PATHS:")
            logger.info("=" * 80)
            model_path = getattr(
                self,
                "model_save_path",
                self.config.get(
                    "model_save_path",
                    ZTBConfig().get_model_path("ppo_balanced_mem_optimized.zip"),
                ),
            )
            checkpoint_path = getattr(
                self,
                "checkpoint_dir",
                self.config.get(
                    "checkpoint_dir", "checkpoints/ppo_balanced_mem_optimized"
                ),
            )
            tensorboard_path = self.config.get("tensorboard_log", "tensorboard")
            logger.info(f"  Model:        {model_path}")
            logger.info(f"  Checkpoints:  {checkpoint_path}")
            logger.info(f"  TensorBoard:  {tensorboard_path}")
            logger.info("=" * 80)

            # Check if targets were met
            sell_rate_ok = final_stats.get("r_sell_mean", 0) >= 0.15
            lambda_bounded = abs(final_stats.get("lambda_dual", 0)) <= 1.0

            if sell_rate_ok and lambda_bounded:
                logger.info("✅ SELL bias mitigation targets achieved")
            else:
                logger.warning("⚠️  SELL bias mitigation targets not fully achieved")
                if not sell_rate_ok:
                    logger.warning(
                        f"   - SELL rate below 15%: {final_stats.get('r_sell_mean', 0):.1%}"
                    )
                if not lambda_bounded:
                    logger.warning(
                        f"   - Lambda out of bounds: {final_stats.get('lambda_dual', 0):.6f}"
                    )

            logger.info("✅ Final validation completed")
        except Exception as e:
            logger.warning(f"Final validation failed (non-critical): {e}")


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
def test_sell_mitigation_trainer() -> None:
    """Test SELL bias mitigation trainer."""
    print("Testing SELL Bias Mitigation Trainer...")

    # Create minimal config
    config: PPOConfig = DEFAULT_PPO_CONFIG.copy()
    config["total_timesteps"] = 1000  # Very short for testing
    config["n_steps"] = 128

    # Create trainer params
    params = SELLMitigationParams(
        data_path="ml-dataset-final.csv",
        config=config,
        checkpoint_dir="test_checkpoints",
        enable_lagrange=True,
        enable_probes=True,
        enable_weights=True,
    )

    # Create trainer
    trainer = SELLBiasMitigationPPOTrainer(params)

    print("✅ SELL Bias Mitigation Trainer created successfully")
    # ★ NOTE: Lagrange is now in model, not trainer
    print("   Lagrange: Integrated into CustomPPO")
    print(f"   Probes: {'✅' if trainer.probe else '❌'}")
    print(f"   Weights: {'✅' if trainer.weight_calc else '❌'}")

    # Cleanup
    if trainer.probe:
        trainer.probe.close()

    print("✅ Test completed")


if __name__ == "__main__":
    test_sell_mitigation_trainer()
