#!/usr/bin/env python3
"""
Paper Trading Evaluation for Trained PPO Models.

Loads and simulates trading on test data to evaluate performance.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.utils.path_utils import ensure_dir, get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

from ztb.trading.environment.environment import HeavyTradingEnv as TradingEnvironment
from ztb.training.ppo_config import get_ppo_config
from ztb.trading.env_config import get_trading_env_config, TradingEnvConfig
from ztb.utils import DiscordNotifier
from ztb.utils.data_utils import load_csv_data_optimized
from ztb.utils.file_utils import safe_json_load
from ztb.inference.decode import decode_action, InferenceConfig


class PaperTrader:
    """Paper trading simulator for evaluating trained models."""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.config = config or self._get_default_config()
        self.verbose = verbose
        print(f"PaperTrader verbose: {self.verbose}")
        self.logger = logging.getLogger(__name__)

        # Initialize instance variables
        self.test_df: Optional[pd.DataFrame] = None
        self.model: Optional[MaskablePPO] = None
        self.env: DummyVecEnv
        self.episode_results: List[Dict[str, Any]] = []
        self._normalization_stats: Optional[Any] = None  # Store loaded normalization stats

        # Load test data first
        self.logger.info(f"Loading test data from {self.test_data_path}")
        self.test_df = load_csv_data_optimized(str(self.test_data_path))
        self.logger.info(f"Loaded {len(self.test_df)} rows of test data")

        # Initialize environment
        self.env = self._create_env()

        # Load model
        self.logger.info(f"Loading model from {self.model_path}")
        self._load_model()
        self.logger.info("Model loaded successfully")

        # Trading results
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_value: float = 10000.0  # Starting capital
        self.position: float = 0.0  # Current position size

        # Inference configuration
        self.inference_config = InferenceConfig(
            temperature=float(cast(float, self.config.get("temperature", 0.7))),
            tiebreaker_tau=float(cast(float, self.config.get("tiebreaker_tau", 0.05))),
            enable_tiebreaker=bool(cast(bool, self.config.get("enable_tiebreaker", True))),
            deterministic=bool(cast(bool, self.config.get("deterministic", False))),
        )

    def _get_default_config(self) -> TradingEnvConfig:
        """Get default configuration for paper trading."""
        return get_trading_env_config({
            "reward_scaling": 1.0,  # Override for paper trading
            "risk_free_rate": 0.0,
            "initial_portfolio_value": 10000.0,
            "verbose": 1,
        })

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
                    "initial_portfolio_value", 10000.0
                ),
                "verbose": self.config.get("verbose", 1),
            },
        )

        return DummyVecEnv([lambda: env])

    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        self.logger.info(f"Loading model from {self.model_path}")
        # Create a dummy model first, then load checkpoint
        dummy_env = self._create_env()

        # Get policy_kwargs from config
        policy_kwargs_raw = self.config.get("policy_kwargs", {})
        policy_kwargs: Dict[str, Any] = policy_kwargs_raw if isinstance(policy_kwargs_raw, dict) else {}

        # Get PPO config from common configuration
        ppo_config = get_ppo_config()

        self.model = MaskablePPO(
            "MlpPolicy",
            dummy_env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
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
            print("Successfully loaded model using Stable Baselines3 load method")
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
                    self.model.value_net.load_state_dict(value_state)

                print("Successfully loaded model using custom checkpoint format")

            except Exception as custom_error:
                raise RuntimeError(
                    f"Failed to load model with both Stable Baselines3 and custom formats: SB3: {sb3_error}, Custom: {custom_error}"
                )

        assert self.model is not None, "Model failed to load"

    def _load_test_data(self) -> None:
        """Load test data for evaluation."""
        if self.test_data_path.exists():
            self.test_df = load_csv_data_optimized(self.test_data_path)
            # Use a subset for testing (e.g., last 20% of data)
            test_size = int(len(self.test_df) * 0.2)
            self.test_df = self.test_df.tail(test_size)
            self.logger.info(f"Using {len(self.test_df)} test samples")

            # Auto-detect feature columns (exclude meta columns) - used for both validations
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

            # Validate feature schema against training schema
            try:
                from ztb.utils.feature_schema import load_and_validate_schema

                # Assume model is in models/ directory, schema is alongside model zip
                model_dir = self.model_path.parent
                schema_path = model_dir / "features_schema.json"

                if schema_path.exists():
                    # Validate schema (strict=True will raise on mismatch)
                    schema = load_and_validate_schema(
                        model_dir, self.test_df, feature_columns, strict=True
                    )
                    self.logger.info(
                        f"Feature schema validated successfully "
                        f"({len(feature_columns)} features, "
                        f"hash: {schema.compute_hash()[:16]}...)"
                    )
                else:
                    self.logger.warning(
                        f"Feature schema not found: {schema_path}. "
                        "Skipping validation (may cause silent errors!)"
                    )

                # Validate normalization statistics
                try:
                    from ztb.utils.normalization import load_scaler
                    import numpy as np

                    scaler_path = model_dir / "scaler.npz"
                    if scaler_path.exists():
                        # Load saved normalization stats
                        saved_stats = load_scaler(model_dir, strict=True)

                        # Compute stats from test data (using feature_columns from above)
                        feature_data = self.test_df[feature_columns].values
                        test_mean = np.mean(feature_data, axis=0)
                        test_std = np.std(feature_data, axis=0)

                        # Log comparison (info only, not strict validation)
                        mean_diff = np.max(np.abs(saved_stats.mean - test_mean))
                        std_diff = np.max(np.abs(saved_stats.std - test_std))

                        self.logger.info(
                            f"Normalization stats loaded "
                            f"(hash: {saved_stats.compute_hash()[:16]}...)"
                        )
                        self.logger.info(
                            f"Train vs Test stats difference: "
                            f"mean Δ={mean_diff:.6f}, std Δ={std_diff:.6f}"
                        )

                        # Store for later use if needed
                        self._normalization_stats = saved_stats
                    else:
                        self.logger.warning(
                            f"Normalization stats not found: {scaler_path}. "
                            "Evaluation may use different normalization than training!"
                        )
                except FileNotFoundError as e:
                    self.logger.error(f"Normalization stats loading FAILED: {e}")
                    raise RuntimeError(
                        f"Normalization stats missing. Evaluation aborted to prevent "
                        f"silent errors. Please retrain model with stats persistence.\n{e}"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not validate normalization stats: {e}")

            except ValueError as e:
                # Schema validation failed - this is CRITICAL
                self.logger.error(f"Feature schema validation FAILED: {e}")
                raise RuntimeError(
                    f"Feature schema mismatch detected. Evaluation aborted to prevent "
                    f"silent errors. Please retrain model or fix data schema.\n{e}"
                )
            except Exception as e:
                self.logger.warning(f"Could not validate feature schema: {e}")
        else:
            self.test_df = None
            self.logger.warning(f"Test data not found: {self.test_data_path}")

    def simulate_trading(self, n_episodes: int = 5) -> Dict[str, Any]:
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

    def _simulate_episode(self) -> Dict[str, Any]:
        """Simulate a single trading episode."""
        obs = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        episode_trades = []

        while not done and steps < 10000:  # Max steps per episode
            # Get action from model
            predict_obs = obs[0] if isinstance(obs, tuple) else obs

            # Get legal actions mask for MaskablePPO
            action_masks = cast(
                TradingEnvironment, self.env.envs[0]
            ).get_legal_actions()

            # Get logits from policy network
            with torch.no_grad():
                obs_tensor = torch.from_numpy(predict_obs).float()
                features = self.model.policy.extract_features(obs_tensor, self.model.policy.features_extractor)  # type: ignore[union-attr]
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

            # Debug: Log action distribution for first few steps
            if self.verbose and steps < 10:
                print(f"\nStep {steps} (Unified Decode Diagnostics):")
                print(f"  Action selected: {action[0]} ({'HOLD' if action[0] == 0 else 'BUY' if action[0] == 1 else 'SELL'})")
                print(f"  Probabilities: HOLD={decode_info['probabilities'][0]:.4f}, "
                      f"BUY={decode_info['probabilities'][1]:.4f}, "
                      f"SELL={decode_info['probabilities'][2]:.4f}")
                print(f"  Top2 actions: {decode_info['top2_actions']}")
                print(f"  Top2 probs: [{decode_info['top2_probs'][0]:.4f}, {decode_info['top2_probs'][1]:.4f}]")
                print(f"  Margin: {decode_info['margin']:.4f}")
                print(f"  Tiebreaker activated: {decode_info['tiebreaker_activated']}")
                print(f"  Legal actions mask: {action_masks}")

            # Record state before action
            prev_portfolio = self.portfolio_value
            prev_position = self.position

            # Execute action
            obs, reward, done_vec, _ = self.env.step(action)
            done = done_vec[0]
            reward = reward[0]

            # Update from environment
            self.portfolio_value = cast(
                TradingEnvironment, self.env.envs[0]
            ).portfolio_value
            self.position = cast(TradingEnvironment, self.env.envs[0]).position

            # Record trade if position changed
            if (
                abs(self.position - prev_position) > 0.01
            ):  # Position changed significantly
                trade = {
                    "step": steps,
                    "action": action[0],
                    "prev_portfolio": prev_portfolio,
                    "new_portfolio": self.portfolio_value,
                    "prev_position": prev_position,
                    "new_position": self.position,
                    "reward": reward,
                    "portfolio_change": self.portfolio_value - prev_portfolio,
                }
                episode_trades.append(trade)

                # Log detailed trade information
                action_name = (
                    "BUY" if action[0] > 0.1 else "SELL" if action[0] < -0.1 else "HOLD"
                )
                self.logger.info(
                    f"Trade #{len(episode_trades)}: {action_name} | "
                    f"Position: {prev_position:.3f} -> {self.position:.3f} | "
                    f"Portfolio: ${prev_portfolio:.2f} -> ${self.portfolio_value:.2f} | "
                    f"Change: ${trade['portfolio_change']:.2f}"
                )

            total_reward += reward
            steps += 1

        episode_result = {
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
    ) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics."""
        initial_portfolio = float(cast(float, self.config.get("initial_portfolio_value", 10000.0)))

        # Calculate average final portfolio across episodes
        if self.episode_results:
            final_portfolio_values = [
                r["final_portfolio"] for r in self.episode_results
            ]
            avg_final_portfolio = float(np.mean(final_portfolio_values))
        else:
            avg_final_portfolio = initial_portfolio

        stats: Dict[str, Any] = {
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

        # Action distribution
        action_counts: Dict[str, int] = {}
        for trade in self.trades:
            action = trade["action"]
            if isinstance(action, (list, np.ndarray)):
                action = action[0]
            action_name = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        stats["action_distribution"] = action_counts

        return stats

    def _save_trade_log(self, stats: Dict[str, Any]) -> None:
        """Save detailed trade log and statistics."""
        results_dir = Path("results") / "paper_trading"
        ensure_dir(results_dir)

        # Save statistics
        stats_file = results_dir / "trading_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Save trade log
        trades_file = results_dir / "trade_log.json"
        with open(trades_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_dir}")
        self.logger.info(f"Statistics: {stats}")


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
        default="scalping-config.json",
        help="Path to config JSON file (default: scalping-config.json)",
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
        )

        # Send start notification
        notifier.send_notification(
            title="📈 Paper Trading Started",
            message=f"Evaluating model: {Path(args.model_path).name}",
            color="info",
            fields={
                "Model": Path(args.model_path).name,
                "Test Data": args.test_data,
                "Episodes": str(args.episodes),
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
            message=f"Model evaluation completed: {Path(args.model_path).name}",
            color="success",
            fields={
                "Total Return": f"{results['total_return_percent']:.2f}%",
                "Win Rate": f"{results.get('win_rate', 0):.2%}",
                "Total Trades": str(results["total_trades"]),
                "Final Portfolio": f"${results['final_portfolio_value']:.2f}",
                "Action Distribution": str(results.get("action_distribution", {})),
            },
        )

        # Print summary
        print("\n" + "=" * 50)
        print("PAPER TRADING RESULTS")
        print("=" * 50)
        print(f"Total Return: {results['total_return_percent']:.2f}%")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Final Portfolio: ${results['final_portfolio_value']:.2f}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        if "action_distribution" in results:
            print(f"Action Distribution: {results['action_distribution']}")
        print("=" * 50)

        return 0

    except Exception as e:
        logger.error(f"Paper trading failed: {e}", exc_info=True)

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
