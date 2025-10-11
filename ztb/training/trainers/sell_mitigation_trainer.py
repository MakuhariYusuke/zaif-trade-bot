"""
SELL Bias Mitigation PPO Trainer.

Extends PPOTrainer with comprehensive action bias mitigation using CustomPPO.
This trainer integrates multiple specialized techniques to address SELL action
bias in trading strategies.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ztb.training.callbacks_lib import SELLBiasMitigationCallback
from ztb.training.models.custom_ppo import CustomPPO
from ztb.training.utils.grad_probes import SELLGradientProbe, create_failsafe_dump
from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG, PPOConfig
from ztb.training.config.lagrange_defaults import LAGRANGE_DEFAULTS
from ztb.training.core.ppo_trainer import PPOTrainerAutoHalt as PPOTrainer
from ztb.training.config.trainer_params import SELLMitigationParams
from ztb.training.utils.weights import ActionWeightCalculator
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


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

    Training Flow:
    - Initialize mitigation components based on configuration
    - Apply constraints and normalization during training
    - Monitor bias metrics and adjust parameters dynamically
    - Provide comprehensive logging and diagnostics

    Args:
        params: SELLMitigationParams containing all training and mitigation configuration
    """

    def __init__(
        self,
        params: SELLMitigationParams,
    ):
        """
        Initialize SELL bias mitigation trainer.
        
        Args:
            params: SELLMitigationParams with training and mitigation configuration
        """
        # Call parent PPOTrainer __init__
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
        # NOTE: Lagrange, PAN, and Target Entropy are now integrated into CustomPPO
        # We only keep separate instances for Probes and Weight Calculator
        self.probe = None
        self.weight_calc = None

        if params.enable_lagrange:
            # Lagrange is created by CustomPPO, just log here
            r_target = self.lagrange_params.get("r_target", LAGRANGE_DEFAULTS["r_target"])
            logger.info(f"Lagrange constraint will be enabled in CustomPPO (r_target={r_target:.1%})")

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
        
        if params.enable_pan:
            logger.info("PAN (Per-Action Advantage Normalization) enabled")
        
        if params.enable_target_entropy:
            logger.info("Target Entropy Controller enabled (target=0.769)")
        
        if params.enable_stratified_sampling:
            logger.info("Stratified Mini-batch Sampler enabled (9 buckets: 3 regimes × 3 actions)")

        # Initialize model attribute
        self.model: Optional[CustomPPO] = None

    def _create_callback(self) -> BaseCallback:
        """
        Create composite training callback with SELL bias mitigation.
        
        Returns:
            CallbackList: Combined callbacks for training monitoring
        """
        base_callback = PPOTrainer._create_callback(self)  # type: ignore[attr-defined]

        # Get components from model (not from self)
        # Lagrange, PAN, and Entropy Controller are managed by CustomPPO
        lagrange = self.model.lagrange if self.model is not None else None
        pan_normalizer = self.model.pan_normalizer if self.model is not None else None
        entropy_controller = self.model.entropy_controller if self.model is not None else None
        stratified_sampler = self.model.stratified_sampler if self.model is not None else None

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
            return

        logger.info("Using enhanced SELL bonus weighting")
        # Note: This would require modifying the PPO loss computation
        # For now, we rely on the Lagrange constraint for the hard guarantee

    def train(self, session_id: str) -> MaskablePPO:
        """
        Train with SELL bias mitigation using CustomPPO.
        
        Args:
            session_id: Unique identifier for this training session
            
        Returns:
            MaskablePPO: Trained model
        """
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
        logger.info(f"Data: {self.data_path}")  # type: ignore[attr-defined]

        try:
            # Create model with CustomPPO
            if self.model is None:
                # Load data
                df = pd.read_csv(self.data_path)  # type: ignore[attr-defined]

                # Create environment config
                env_config = {
                    "curriculum_stage": self.config.get("curriculum_stage", "full"),  # type: ignore[attr-defined]
                    "allow_reverse": self.allow_reverse,
                    "transaction_cost": self.config.get("transaction_cost", 0.001),  # type: ignore[attr-defined]
                    "max_position_size": self.config.get("max_position_size", 1.0),  # type: ignore[attr-defined]
                    "risk_free_rate": self.config.get("risk_free_rate", 0.0),  # type: ignore[attr-defined]
                    "reward_scaling": self.config.get("reward_scaling", 1.0),  # type: ignore[attr-defined]
                    # ★ BUG FIX #48: Pass reward_settings from config to environment
                    "reward_settings": self.config.get("reward_settings", {}),  # type: ignore[attr-defined]
                }

                # Create environment
                env = HeavyTradingEnv(df=df, config=env_config)

                # Wrap with ActionMasker for MaskablePPO
                def mask_fn(env: Any) -> Any:
                    return env.get_legal_actions().astype(bool)

                env = ActionMasker(env, mask_fn)  # type: ignore[assignment]

                # Create CustomPPO with integrated bias mitigations
                self.model = CustomPPO(
                    policy=self.config.get("policy", "MlpPolicy"),  # type: ignore[attr-defined]
                    env=env,
                    learning_rate=self.config.get("learning_rate", 3e-4),  # type: ignore[attr-defined]
                    n_steps=self.config.get("n_steps", 2048),  # type: ignore[attr-defined]
                    batch_size=self.config.get("batch_size", 64),  # type: ignore[attr-defined]
                    n_epochs=self.config.get("n_epochs", 10),  # type: ignore[attr-defined]
                    gamma=self.config.get("gamma", 0.99),  # type: ignore[attr-defined]
                    gae_lambda=self.config.get("gae_lambda", 0.95),  # type: ignore[attr-defined]
                    clip_range=self.config.get("clip_range", 0.2),  # type: ignore[attr-defined]
                    clip_range_vf=self.config.get("clip_range_vf"),  # type: ignore[attr-defined]
                    normalize_advantage=self.config.get("normalize_advantage", True),  # type: ignore[attr-defined]
                    ent_coef=self.config.get("ent_coef", 0.0),  # type: ignore[attr-defined]
                    vf_coef=self.config.get("vf_coef", 0.5),  # type: ignore[attr-defined]
                    max_grad_norm=self.config.get("max_grad_norm", 0.5),  # type: ignore[attr-defined]
                    target_kl=self.config.get("target_kl"),  # type: ignore[attr-defined]
                    tensorboard_log=self.config.get("tensorboard_log"),  # type: ignore[attr-defined]
                    policy_kwargs=self.config.get("policy_kwargs"),  # type: ignore[attr-defined]
                    verbose=self.config.get("verbose", 1),  # type: ignore[attr-defined]
                    seed=self.config.get("seed"),  # type: ignore[attr-defined]
                    device=self.config.get("device", "auto"),  # type: ignore[attr-defined]
                    _init_setup_model=self.config.get("_init_setup_model", True),  # type: ignore[attr-defined]
                    # Custom bias mitigation parameters
                    enable_pan=self.enable_pan,
                    enable_target_entropy=self.enable_target_entropy,
                    enable_stratified_sampling=self.enable_stratified_sampling,
                    # Lagrange constraint parameters
                    enable_lagrange=self.enable_lagrange,
                    lagrange_target_action="SELL",
                    lagrange_r_target=self.lagrange_params.get("r_target", LAGRANGE_DEFAULTS["r_target"]),
                    lagrange_tolerance=self.lagrange_params.get("tolerance", LAGRANGE_DEFAULTS["tolerance"]),
                    lagrange_eta=self.lagrange_params.get("eta", LAGRANGE_DEFAULTS["eta"]),
                    lagrange_lambda_max=self.lagrange_params.get("lambda_max", LAGRANGE_DEFAULTS["lambda_max"]),
                    lagrange_warmup_steps=int(self.lagrange_params.get("warmup_steps", LAGRANGE_DEFAULTS["warmup_steps"])),
                    # PAN/Entropy/Stratified parameters
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
            self.start_training()  # type: ignore[attr-defined]

            # Train the model
            total_timesteps = self.config.get("total_timesteps", 100000)  # type: ignore[attr-defined]
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
        # Get Lagrange from model
        if self.model is None or not hasattr(self.model, 'lagrange') or self.model.lagrange is None:
            logger.warning("Lagrange constraint not available for final validation")
            return

        final_stats = self.model.lagrange.get_statistics()
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
    print(f"   Lagrange: Integrated into CustomPPO")
    print(f"   Probes: {'✅' if trainer.probe else '❌'}")
    print(f"   Weights: {'✅' if trainer.weight_calc else '❌'}")

    # Cleanup
    if trainer.probe:
        trainer.probe.close()

    print("✅ Test completed")


if __name__ == "__main__":
    test_sell_mitigation_trainer()
