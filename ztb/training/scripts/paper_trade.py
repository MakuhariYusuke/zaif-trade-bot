#!/usr/bin/env python3
"""
Paper Trading Evaluation for Trained RL Models.

Supports both PPO and SAC algorithms for trading evaluation.
Loads and simulates trading on test data to evaluate performance.

Note: this script intentionally adjusts sys.path to locate project modules;
the following file-level noqa silences E402 warnings from ruff for the
import placement that occurs after the sys.path modification.
"""

# ruff: noqa: E402

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import numpy as np

# Add project root to path before importing ztb modules
current = Path(__file__).resolve()
for parent in [current] + list(current.parents):
    if any(
        (parent / marker).exists()
        for marker in [
            "pyproject.toml",
            "setup.py",
            ".git",
            "requirements.txt",
            "package.json",
        ]
    ):
        project_root = parent
        break
else:
    project_root = current.parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.path_utils import ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

import pandas as pd
import torch
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(project_root))

from ztb.inference.decode import InferenceConfig, decode_action
from ztb.trading.env_config import TradingEnvConfig, get_trading_env_config
from ztb.trading.environment.constants import (
    DEFAULT_INITIAL_BALANCE_SMALL,
    PPO_DEFAULT_N_STEPS,
    continuous_to_discrete_action,
)
from ztb.trading.environment.environment import HeavyTradingEnv as TradingEnvironment
from ztb.training.config.ppo_config import get_ppo_config
from ztb.utils import DiscordNotifier
from ztb.utils.env_metrics import unwrap_env
from ztb.io.data_loader import DataLoader
from ztb.utils.file_utils import safe_json_dump, safe_json_load


def detect_algorithm(model_path: Path) -> str:
    """Detect the RL algorithm from model path or config.

    Args:
        model_path: Path to the model file

    Returns:
        Algorithm name ('ppo' or 'sac')
    """
    model_name = model_path.stem.lower()

    # Check filename for algorithm hints
    if "sac" in model_name:
        return "sac"
    elif "ppo" in model_name:
        return "ppo"

    # Default to PPO for backward compatibility
    return "ppo"


class TradeDict(TypedDict):
    """Trade information dictionary."""

    step: int
    action: int
    prev_portfolio: float
    new_portfolio: float
    prev_position: float
    new_position: float
    reward: float
    portfolio_change: float


class EpisodeResultDict(TypedDict):
    """Episode result dictionary."""

    total_reward: float
    length: int
    trades: List[TradeDict]
    final_portfolio: float
    total_trades: int


class TradingStatsDict(TypedDict, total=False):
    """Comprehensive trading statistics dictionary.

    Contains all key performance metrics and risk measures for trading evaluation.
    """

    # Episode-level metrics
    episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float

    # Trading performance metrics
    total_trades: int
    final_portfolio_value: float
    total_return_percent: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float

    # Action analysis
    action_distribution: Dict[int, int]

    # Performance stability
    consistency_score: float
    best_episode_return: float
    worst_episode_return: float


class PaperTrader:
    """Paper trading simulator for evaluating trained models."""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        algorithm: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.config = config or self._get_default_config()
        self.verbose = verbose
        self.algorithm = algorithm or detect_algorithm(self.model_path)
        print(f"PaperTrader verbose: {self.verbose}, algorithm: {self.algorithm}")
        self.logger = logging.getLogger(__name__)

        # Initialize instance variables
        self.test_df: Optional[pd.DataFrame] = None
        self.model: Optional[Union[MaskablePPO, SAC]] = None
        self.env: Optional[DummyVecEnv] = None
        self._base_env: Optional[TradingEnvironment] = None
        self.episode_results: List[EpisodeResultDict] = []
        self._normalization_stats: Optional[
            Any
        ] = None  # Store loaded normalization stats

        self._setup_common_config()

    def _setup_common_config(self) -> None:
        """Common configuration setup for both PPO and SAC."""
        # Load test data first
        self.logger.info(f"Loading test data from {self.test_data_path}")
        self.test_df = DataLoader.load_csv_optimized(str(self.test_data_path))
        self.logger.info(f"Loaded {len(self.test_df)} rows of test data")

        # Initialize environment
        self.env = self._create_env()

        # Load model
        self.logger.info(f"Loading model from {self.model_path}")
        self._load_model()
        self.logger.info("Model loaded successfully")
        print(f"DEBUG: Model observation space: {self.model.observation_space}")

        # Initialize schema attributes (only for PPO)
        if self.algorithm == "ppo":
            self._initialize_schema()
        else:
            # For SAC, skip schema validation
            self.schema_available = False
            obs_shape = getattr(self.model.observation_space, "shape", None)
            self.expected_features = int(obs_shape[0]) if obs_shape else None
            self.feature_names = None
            self.schema_hash = None

        # For SAC, disable correlation reduction to match model expectations
        if self.algorithm == "sac":
            # Recreate environment without correlation reduction
            self.env = self._create_env_sac()

        # Trading results
        self.trades: List[TradeDict] = []
        self.portfolio_value: float = DEFAULT_INITIAL_BALANCE_SMALL  # Starting capital
        self.position: float = 0.0  # Current position size

        # Inference configuration (only for PPO)
        if self.algorithm == "ppo":
            self.inference_config: Optional[InferenceConfig] = InferenceConfig(
                temperature=float(cast(float, self.config.get("temperature", 0.7))),
                tiebreaker_tau=float(
                    cast(float, self.config.get("tiebreaker_tau", 0.05))
                ),
                enable_tiebreaker=bool(
                    cast(bool, self.config.get("enable_tiebreaker", True))
                ),
                deterministic=bool(cast(bool, self.config.get("deterministic", False))),
            )
        else:
            self.inference_config = None

    def _get_default_config(self) -> TradingEnvConfig:
        """Get default configuration for paper trading."""
        return get_trading_env_config(
            {
                "reward_scaling": 1.0,  # Override for paper trading
                "risk_free_rate": 0.0,
                "initial_portfolio_value": 10000.0,
                "verbose": 1,
                "enable_correlation_reduction": False,  # Disable correlation reduction for SAC compatibility
                "correlation_reduction": False,  # Also set the actual config key
            }
        )

    def _get_base_env(self) -> Optional[TradingEnvironment]:
        """Return the unwrapped TradingEnvironment for the current env."""
        if self._base_env is not None:
            return self._base_env
        if self.env is None:
            return None
        base_env = unwrap_env(self.env)
        if isinstance(base_env, TradingEnvironment):
            self._base_env = base_env
        return self._base_env

    def _create_env(self) -> DummyVecEnv:
        """Create evaluation environment."""
        env = TradingEnvironment(
            df=self.test_df,
            config={
                "reward_scaling": self.config.get("reward_scaling", 1.0),
                "transaction_cost": self.config.get("transaction_cost", 0.001),
                "max_position_size": self.config.get("max_position_size", 1.0),
                "risk_free_rate": self.config.get("risk_free_rate", 0.0),
                "curriculum_stage": self.config.get("curriculum_stage", "full"),
                "initial_portfolio_value": self.config.get(
                    "initial_portfolio_value", 1000000.0
                ),
                "verbose": self.config.get("verbose", 1),
            },
        )

        # Store base environment reference for initial_portfolio_value access
        self._base_env = env

        return DummyVecEnv([lambda: env])

    def _create_env_sac(self) -> DummyVecEnv:
        """Create environment for SAC models (without correlation reduction)."""
        self.logger.info("Creating SAC environment with correlation_reduction=False")
        config = {
            "reward_scaling": self.config.get("reward_scaling", 1.0),
            "transaction_cost": self.config.get("transaction_cost", 0.001),
            "max_position_size": self.config.get("max_position_size", 1.0),
            "risk_free_rate": self.config.get("risk_free_rate", 0.0),
            "curriculum_stage": self.config.get("curriculum_stage", "full"),
            "initial_portfolio_value": self.config.get(
                "initial_portfolio_value", 1000000.0
            ),
            "verbose": self.config.get("verbose", 1),
            "correlation_reduction": True,  # Enable for SAC compatibility with this model
            "enable_correlation_reduction": True,  # Also set the actual config key
        }
        expected_dim = self.expected_features
        if expected_dim is None:
            obs_shape = getattr(self.model.observation_space, "shape", None)
            expected_dim = int(obs_shape[0]) if obs_shape else None
        if expected_dim:
            config["target_feature_count"] = int(expected_dim)
        self.logger.info(f"SAC config: {config}")
        env_kwargs: Dict[str, Any] = {
            "df": self.test_df,
            "config": config,
        }
        if expected_dim:
            env_kwargs["max_features"] = int(expected_dim)
        env = TradingEnvironment(**env_kwargs)

        # Store base environment reference for initial_portfolio_value access
        self._base_env = env

        return DummyVecEnv([lambda: env])

    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        self.logger.info(
            f"Loading {self.algorithm.upper()} model from {self.model_path}"
        )

        if self.algorithm == "ppo":
            self._load_ppo_model()
        elif self.algorithm == "sac":
            self._load_sac_model()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _load_ppo_model(self) -> None:
        """Load PPO model with custom checkpoint support."""
        # Create a dummy model first, then load checkpoint
        dummy_env = self._create_env()

        # Get policy_kwargs from config
        policy_kwargs_raw = self.config.get("policy_kwargs", {})
        policy_kwargs: Dict[str, Any] = (
            policy_kwargs_raw if isinstance(policy_kwargs_raw, dict) else {}
        )

        # Get PPO config from common configuration
        ppo_config = get_ppo_config()

        self.model = MaskablePPO(
            "MlpPolicy",
            dummy_env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", PPO_DEFAULT_N_STEPS),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.0),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            verbose=0,
            seed=42,
            policy_kwargs=policy_kwargs,
        )

        # Load model using Stable Baselines3's load method for zip files
        try:
            # Try loading as Stable Baselines3 zip format first
            self.model = MaskablePPO.load(
                str(self.model_path),
                env=dummy_env,
                custom_objects={"policy_kwargs": policy_kwargs},
            )
            print("Successfully loaded PPO model using Stable Baselines3 load method")
        except Exception as sb3_error:
            print(
                f"Stable Baselines3 load failed: {sb3_error}, trying custom checkpoint format..."
            )

            # Fallback to custom checkpoint loading (LZ4/ZSTD compressed)
            try:
                import pickle

                import lz4.frame
                import zstandard as zstd

                with open(self.model_path, "rb") as f:
                    compressed_data = f.read()

                # Try ZSTD first (newer compression used in training)
                try:
                    dctx = zstd.ZstdDecompressor()
                    decompressed_data = dctx.decompress(compressed_data)
                    compression_type = "ZSTD"
                except zstd.ZstdError:
                    # Fall back to LZ4 (older compression)
                    try:
                        decompressed_data = lz4.frame.decompress(compressed_data)
                        compression_type = "LZ4"
                    except Exception as lz4_error:
                        raise RuntimeError(
                            f"Failed to decompress with both ZSTD and LZ4: ZSTD error, LZ4: {lz4_error}"
                        )

                print(
                    f"Successfully decompressed model using {compression_type} compression"
                )

                # Load checkpoint data
                try:
                    checkpoint_data = pickle.loads(decompressed_data)
                except AttributeError as e:
                    if "CheckpointData" in str(e):
                        # Try loading again with the class available (already imported globally)
                        checkpoint_data = pickle.loads(decompressed_data)
                    else:
                        raise

                # Load state dict - checkpoint_data might be a dict or CheckpointData object
                if hasattr(checkpoint_data, "policy"):
                    # It's a CheckpointData object
                    policy_state = checkpoint_data.policy
                    value_state = getattr(checkpoint_data, "value_net", None)
                else:
                    # It's a dict
                    policy_state = checkpoint_data.get("policy")
                    value_state = checkpoint_data.get("value_net")

                if policy_state:
                    self.model.policy.load_state_dict(policy_state)
                if value_state and hasattr(self.model, "value_net"):
                    try:
                        self.model.value_net.load_state_dict(value_state)
                    except AttributeError:
                        pass  # Some PPO variants may not have value_net

                print("Successfully loaded PPO model using custom checkpoint format")

            except Exception as custom_error:
                raise RuntimeError(
                    f"Failed to load PPO model with both Stable Baselines3 and custom formats: SB3: {sb3_error}, Custom: {custom_error}"
                )

        assert self.model is not None, "PPO model failed to load"

    def _load_sac_model(self) -> None:
        """Load SAC model."""
        try:
            self.model = SAC.load(str(self.model_path))
            print("Successfully loaded SAC model")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAC model: {e}")

        assert self.model is not None, "SAC model failed to load"

    def _initialize_schema(self) -> None:
        """Initialize feature schema for PPO models."""
        # Load and validate feature schema using Phase 3 FeatureSchemaManager
        self.schema_available = False
        self.expected_features: Optional[int] = None
        self.feature_names: Optional[List[str]] = None
        self.schema_hash: Optional[str] = None

        try:
            from ztb.training.core.feature_schema_manager import FeatureSchemaManager

            model_name = self.model_path.stem
            schema_manager = FeatureSchemaManager(model_name)

            # Load schema metadata
            metadata = schema_manager.load_schema()

            self.expected_features = metadata.num_features
            self.feature_names = metadata.feature_names
            self.schema_hash = metadata.schema_hash
            self.schema_available = True

            self.logger.info("✅ Schema loaded for model: %s", model_name)
            self.logger.info("   Expected features: %d", self.expected_features)
            self.logger.info("   Schema hash: %s", self.schema_hash[:16])
            self.logger.info("   Created at: %s", metadata.created_at)

            # Display feature list summary
            if self.feature_names:
                self.logger.info("📋 Model feature requirements:")
                self.logger.info("   Total: %d features", len(self.feature_names))
                self.logger.info("   First 5: %s", self.feature_names[:5])
                self.logger.info("   Last 5: %s", self.feature_names[-5:])

        except FileNotFoundError:
            self.logger.warning(
                "⚠️  No schema found for model %s. Feature validation disabled.",
                self.model_path.stem,
            )
        except Exception as e:
            self.logger.warning(
                "⚠️  Failed to load schema for %s: %s", self.model_path.stem, str(e)
            )

    def _get_ppo_action(self, obs: np.ndarray) -> tuple:
        """Get action from PPO model using inference pipeline."""
        # Get legal actions mask for MaskablePPO
        base_env = self._get_base_env()
        if base_env is None:
            raise RuntimeError("Base environment not initialized for PPO action.")
        action_masks = cast(
            NDArray[np.bool_],
            base_env.get_legal_actions(),
        )

        # Get logits from policy network
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float()
            features = self.model.policy.extract_features(
                obs_tensor, self.model.policy.features_extractor
            )  # type: ignore[union-attr]
            if self.model.policy.share_features_extractor:  # type: ignore[union-attr]
                latent_pi, _ = self.model.policy.mlp_extractor(features)  # type: ignore[union-attr]
            else:
                latent_pi = self.model.policy.mlp_extractor.forward_actor(features[0])  # type: ignore[union-attr]
            logits = self.model.policy.action_net(latent_pi).cpu().numpy()  # type: ignore[union-attr]

        # Use unified decode_action for strict decode order
        action, decode_info = decode_action(
            logits[0] if logits.ndim > 1 else logits,
            action_masks,
            self.inference_config,
        )
        action = np.array([action])  # Wrap for env.step()

        return action, decode_info

    def _get_sac_action(self, obs: np.ndarray) -> tuple:
        """Get action from SAC model."""
        # SAC uses continuous actions, convert to discrete
        action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[union-attr]

        # Convert continuous action to discrete (assuming 3 actions: HOLD, BUY, SELL)
        # SAC outputs continuous values, map to discrete actions
        if isinstance(action, np.ndarray) and action.ndim > 0:
            action_value = action[0] if action.shape[0] > 0 else action.item()
        else:
            action_value = action

        # Map continuous action to discrete using centralized function
        discrete_action = continuous_to_discrete_action(action_value)
        action = np.array([discrete_action])

        # Create decode_info for compatibility
        decode_info = {
            "probabilities": [0.33, 0.33, 0.34],  # Placeholder
            "top2_actions": [discrete_action, discrete_action],
            "top2_probs": [1.0, 0.0],
            "margin": 0.0,
            "tiebreaker_activated": False,
        }

        return action, decode_info

    def _load_test_data(self) -> None:
        """Load test data for evaluation."""
        if self.test_data_path.exists():
            self.test_df = DataLoader.load_csv_optimized(self.test_data_path)
            # Use a subset for testing (e.g., last 20% of data)
            test_size = int(len(self.test_df) * 0.2)
            self.test_df = self.test_df.tail(test_size)
            self.logger.info(f"Using {len(self.test_df)} test samples")

            # Validate feature count against schema
            if self.schema_available and self.expected_features is not None:
                # Auto-detect feature columns (exclude meta columns)
                exclude_cols = {
                    "ts",
                    "timestamp",
                    "exchange",
                    "pair",
                    "episode_id",
                    "side",
                    "source",
                }
                feature_columns = [
                    col
                    for col in self.test_df.columns
                    if col not in exclude_cols
                    and pd.api.types.is_numeric_dtype(self.test_df[col])
                ]

                if len(feature_columns) != self.expected_features:
                    self.logger.error(
                        "❌ Feature count mismatch! Dataset has %d features, "
                        "but schema expects %d",
                        len(feature_columns),
                        self.expected_features,
                    )
                    raise ValueError(
                        f"Feature count mismatch: dataset={len(feature_columns)}, "
                        f"schema={self.expected_features}"
                    )
                else:
                    self.logger.info(
                        "✅ Feature count validated: %d features match schema",
                        len(feature_columns),
                    )
        else:
            self.test_df = None
            self.logger.warning(f"Test data not found: {self.test_data_path}")

    def simulate_trading(self, n_episodes: int = 5) -> TradingStatsDict:
        """Simulate paper trading for multiple episodes."""
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.test_df is None:
            raise ValueError("Test data not loaded")

        self.logger.info(
            f"Starting paper trading simulation with {n_episodes} episodes"
        )

        all_rewards = []
        all_lengths = []
        self.episode_results = []

        for episode in range(n_episodes):
            self.logger.info(f"Episode {episode + 1}/{n_episodes}")
            episode_result = self._simulate_episode()
            self.episode_results.append(episode_result)
            all_rewards.append(episode_result["total_reward"])
            all_lengths.append(episode_result["length"])

        # Calculate overall statistics
        stats = self._calculate_statistics(all_rewards, all_lengths)

        # Save detailed trade log
        self._save_trade_log(stats)

        return stats

    def _simulate_episode(self) -> EpisodeResultDict:
        """Simulate a single trading episode."""
        assert self.env is not None, "Environment not initialized"
        obs = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        episode_trades: List[TradeDict] = []

        while not done and steps < 10000:  # Max steps per episode
            # Get action from model
            predict_obs = obs[0] if isinstance(obs, tuple) else obs

            if self.algorithm == "ppo":
                action, decode_info = self._get_ppo_action(predict_obs)
            elif self.algorithm == "sac":
                action, decode_info = self._get_sac_action(predict_obs)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

            # Debug: Log action distribution for first few steps AND environment state
            if self.verbose and steps < 10:
                base_env = self._get_base_env()
                if base_env is None:
                    raise RuntimeError("Base environment not available for debug.")
                curriculum_stage = getattr(base_env, "curriculum_stage", "UNKNOWN")
                print(f"\n{'='*60}")
                print(f"Step {steps} - Environment & Decode Diagnostics")
                print(f"{'='*60}")
                print("[Environment State]")
                print(f"  Curriculum Stage: {curriculum_stage}")
                print(f"  Current Position: {self.position:.3f}")
                print(f"  Portfolio Value: ${self.portfolio_value:.2f}")
                if self.algorithm == "ppo":
                    action_masks = cast(
                        NDArray[np.bool_],
                        base_env.get_legal_actions(),
                    )
                    print(f"  Legal Actions Mask: {action_masks}")
                print("\n[Decode Pipeline Results]")
                action_idx = int(action[0])
                action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(
                    action_idx, f"UNKNOWN({action_idx})"
                )
                print(f"  Action Selected: {action_idx} ({action_name})")
                print(
                    f"  Probabilities: HOLD={decode_info['probabilities'][0]:.4f}, "
                    f"BUY={decode_info['probabilities'][1]:.4f}, "
                    f"SELL={decode_info['probabilities'][2]:.4f}"
                )
                print(f"  Top2 Actions: {decode_info['top2_actions']} (indices)")
                print(
                    f"  Top2 Probs: [{decode_info['top2_probs'][0]:.4f}, {decode_info['top2_probs'][1]:.4f}]"
                )
                print(f"  Margin (p1-p2): {decode_info['margin']:.4f}")
                print(f"  Tiebreaker Activated: {decode_info['tiebreaker_activated']}")
                if decode_info.get("tiebreaker_reason"):
                    print(f"  Tiebreaker Reason: {decode_info['tiebreaker_reason']}")
                print(f"{'='*60}")

            # Record state before action
            prev_portfolio = self.portfolio_value
            prev_position = self.position

            # Execute action
            obs, reward, done_vec, _ = self.env.step(action)
            done = done_vec[0]
            reward = reward[0]

            # Update from environment
            base_env = self._get_base_env()
            if base_env is None:
                raise RuntimeError("Base environment not available for state update.")
            self.portfolio_value = base_env.portfolio_value
            self.position = base_env.position

            # Record trade if position changed
            if (
                abs(self.position - prev_position) > 0.0001
            ):  # Position changed significantly
                trade: TradeDict = {
                    "step": steps,
                    "action": int(action[0]),
                    "prev_portfolio": prev_portfolio,
                    "new_portfolio": self.portfolio_value,
                    "prev_position": prev_position,
                    "new_position": self.position,
                    "reward": reward,
                    "portfolio_change": self.portfolio_value - prev_portfolio,
                }
                episode_trades.append(trade)

                # Log detailed trade information with CORRECT discrete action mapping
                action_idx = int(action[0])
                action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(
                    action_idx, f"UNKNOWN({action_idx})"
                )
                self.logger.info(
                    f"Trade #{len(episode_trades)}: {action_name} (action_idx={action_idx}) | "
                    f"Position: {prev_position:.3f} -> {self.position:.3f} | "
                    f"Portfolio: ${prev_portfolio:.2f} -> ${self.portfolio_value:.2f} | "
                    f"Change: ${trade['portfolio_change']:.2f}"
                )

            total_reward += reward
            steps += 1

        episode_result: EpisodeResultDict = {
            "total_reward": total_reward,
            "length": steps,
            "trades": episode_trades,
            "final_portfolio": self.portfolio_value,
            "total_trades": len(episode_trades),
        }

        # Log episode summary
        self.logger.info(
            f"Episode completed: {len(episode_trades)} trades, "
            f"Final Portfolio: ${self.portfolio_value:.2f}, "
            f"Total Reward: {total_reward:.2f}"
        )

        self.trades.extend(episode_trades)
        return episode_result

    def _calculate_statistics(
        self, rewards: List[float], lengths: List[int]
    ) -> TradingStatsDict:
        """Calculate comprehensive trading statistics."""
        # Get initial portfolio value from environment (not hardcoded)
        initial_portfolio = float(
            getattr(self._base_env, "initial_portfolio_value", 10000.0)
        )

        # Calculate average final portfolio across episodes
        if self.episode_results:
            final_portfolio_values = [
                r["final_portfolio"] for r in self.episode_results
            ]
            avg_final_portfolio = float(np.mean(final_portfolio_values))
        else:
            avg_final_portfolio = initial_portfolio

        stats: TradingStatsDict = {
            "episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "total_trades": len(self.trades),
            "final_portfolio_value": avg_final_portfolio,
            "total_return_percent": (
                (avg_final_portfolio - initial_portfolio) / initial_portfolio
            )
            * 100,
        }

        # Calculate win/loss ratio
        if self.trades:
            profitable_trades = [t for t in self.trades if t["portfolio_change"] > 0]
            stats["win_rate"] = len(profitable_trades) / len(self.trades)
            stats["avg_win"] = (
                float(np.mean([t["portfolio_change"] for t in profitable_trades]))
                if profitable_trades
                else 0
            )
            stats["avg_loss"] = (
                float(
                    np.mean(
                        [
                            t["portfolio_change"]
                            for t in self.trades
                            if t["portfolio_change"] <= 0
                        ]
                    )
                )
                if any(t["portfolio_change"] <= 0 for t in self.trades)
                else 0
            )

        # Sharpe ratio (simplified)
        if len(rewards) > 1:
            returns = np.array(rewards)
            stats["sharpe_ratio"] = (
                float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
            )

        # Action distribution (CORRECT: discrete action indices)
        action_counts: Dict[int, int] = {}
        for trade in self.trades:
            action = trade["action"]
            action_idx = action
            action_counts[action_idx] = action_counts.get(action_idx, 0) + 1
        stats["action_distribution"] = action_counts

        return stats

    def _save_trade_log(self, stats: TradingStatsDict) -> None:
        """Save detailed trade log and statistics."""
        results_dir = Path("results") / "paper_trading"
        ensure_dir(results_dir)

        # Save statistics
        stats_file = results_dir / "trading_stats.json"
        safe_json_dump(stats, str(stats_file), indent=2, default=str)

        # Save trade log
        trades_file = results_dir / "trade_log.json"
        safe_json_dump(self.trades, str(trades_file), indent=2, default=str)

        self.logger.info(f"Results saved to {results_dir}")
        self.logger.info(f"Statistics: {stats}")

    def close(self) -> None:
        """Clean up resources to prevent memory leaks."""
        try:
            # Close environment
            if hasattr(self, "env") and self.env is not None:
                # Close the underlying vectorized environment
                if hasattr(self.env, "close"):
                    self.env.close()
                # Clear the environment reference
                self.env = None
                self.logger.debug("PaperTrader environment closed")

            # Clear model reference and break potential circular references
            if hasattr(self, "model") and self.model is not None:
                # Clear model references to environment
                if hasattr(self.model, "env") and self.model.env is not None:
                    self.model.env = None
                if (
                    hasattr(self.model, "_last_obs")
                    and self.model._last_obs is not None
                ):
                    self.model._last_obs = None
                self.model = None
                self.logger.debug("PaperTrader model references cleared")

            # Clear data references
            if hasattr(self, "test_df"):
                self.test_df = None

            # Clear episode results and trading history
            if hasattr(self, "episode_results"):
                self.episode_results.clear()

            if hasattr(self, "trades"):
                self.trades.clear()

            # Clear normalization stats
            if hasattr(self, "_normalization_stats"):
                self._normalization_stats = None

            # Clear inference config
            if hasattr(self, "inference_config"):
                self.inference_config = None

        except Exception as e:
            self.logger.warning(f"Error during PaperTrader cleanup: {e}")
            import traceback

            self.logger.debug(f"Cleanup traceback: {traceback.format_exc()}")



def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper trading evaluation")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test-data",
        default="ml-dataset.csv",
        help="Path to test data (default: ml-dataset.csv)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        help="RL algorithm to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
        help="Reward scaling factor (default: 1.0)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per trade (default: 0.001)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=1.0,
        help="Maximum position size (default: 1.0)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Risk-free rate (default: 0.0)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON file (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    # Initialize Discord notifier
    notifier = DiscordNotifier()

    trader = None  # Initialize to avoid unbound variable issues
    try:
        # Create custom config from args
        custom_config = {
            "reward_scaling": args.reward_scaling,
            "transaction_cost": args.transaction_cost,
            "max_position_size": args.max_position_size,
            "risk_free_rate": args.risk_free_rate,
            "initial_portfolio_value": 10000.0,
            "curriculum_stage": "full",
        }

        # Load config file if provided
        if args.config:
            file_config = safe_json_load(Path(args.config))
            # Merge configs, file config takes precedence
            custom_config.update(file_config.get("environment", {}))
            custom_config.update(file_config.get("data", {}))

        # Create paper trader
        trader = PaperTrader(
            args.model_path,
            custom_config.get("test_data", args.test_data),
            config=custom_config,
            verbose=args.verbose,
            algorithm=args.algorithm,
        )

        # Send start notification
        schema_status = (
            f"{trader.expected_features} features (schema-validated ✅)"
            if trader.schema_available and trader.expected_features
            else "schema not available ⚠️"
        )

        notifier.send_notification(
            title="📈 Paper Trading Started",
            message=f"Evaluating {trader.algorithm.upper()} model: {Path(args.model_path).name}",
            color="info",
            fields={
                "Algorithm": trader.algorithm.upper(),
                "Model": Path(args.model_path).name,
                "Test Data": args.test_data,
                "Episodes": str(args.episodes),
                "Features": schema_status,
                "Reward Scaling": str(args.reward_scaling),
                "Transaction Cost": f"{args.transaction_cost:.4f}",
                "Max Position Size": str(args.max_position_size),
                "Risk-free Rate": str(args.risk_free_rate),
            },
        )

        # Run simulation
        logger.info("Starting paper trading simulation...")
        results = trader.simulate_trading(args.episodes)

        # Send completion notification
        notifier.send_notification(
            title="✅ Paper Trading Completed",
            message=f"{trader.algorithm.upper()} model evaluation completed: {Path(args.model_path).name}",
            color="success",
            fields={
                "Algorithm": trader.algorithm.upper(),
                "Total Return": f"{results.get('total_return_percent', 0):.2f}%",
                "Win Rate": f"{results.get('win_rate', 0):.2%}",
                "Total Trades": str(results.get("total_trades", 0)),
                "Final Portfolio": f"${results.get('final_portfolio_value', 0):.2f}",
                "Action Distribution": str(results.get("action_distribution", {})),
            },
        )

        # Print summary
        print("\n" + "=" * 50)
        print("PAPER TRADING RESULTS")
        print("=" * 50)
        print(f"Total Return: {results.get('total_return_percent', 0):.2f}%")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Final Portfolio: ${results.get('final_portfolio_value', 0):.2f}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        if "action_distribution" in results:
            print(f"Action Distribution: {results['action_distribution']}")
        print("=" * 50)

        # Clean up resources
        trader.close()

        return 0

    except Exception as e:
        logger.error(f"Paper trading failed: {e}", exc_info=True)

        # Clean up resources even on failure
        try:
            if "trader" in locals() and trader is not None:
                trader.close()
        except NameError:
            pass  # trader was not defined yet

        # Send failure notification
        notifier.send_notification(
            title="❌ Paper Trading Failed",
            message=f"Model evaluation failed: {Path(args.model_path).name}",
            color="error",
            fields={
                "Error": str(e),
                "Model": Path(args.model_path).name,
            },
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
