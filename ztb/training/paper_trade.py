#!/usr/bin/env python3
"""
Paper Trading Evaluation for Trained PPO Models.

Loa            # Load state dict - checkpoint_data might be a dict or CheckpointData object
            try:
                # Try as dict first
                policy_state = checkpoint_data["policy"]
                value_state = checkpoint_data.get("value_net")
            except (TypeError, KeyError):
                # Try as object
                try:
                    policy_state = checkpoint_data.policy
                    value_state = getattr(checkpoint_data, 'value_net', None)
                except AttributeError:
                    raise ValueError("Unable to parse checkpoint data format")

            if policy_state:
                self.model.policy.load_state_dict(policy_state)
            if value_state and hasattr(self.model, "value_net"):
                self.model.value_net.load_state_dict(value_state)  # type: ignoreel and simulates trading on test data to evaluate performance.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment import HeavyTradingEnv as TradingEnvironment
from ztb.training.entrypoints.base_ml_reinforcement import CheckpointData, StepResult
from ztb.utils import DiscordNotifier


class PaperTrader:
    """Paper trading simulator for evaluating trained models."""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize instance variables
        self.test_df = None
        self.model: Optional[PPO] = None

        # Load test data first
        self._load_test_data()

        # Load model
        self._load_model()

        # Initialize environment
        self.env = self._create_env()

        # Trading results
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_value = 10000.0  # Starting capital
        self.position = 0.0  # Current position size

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for paper trading."""
        return {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
            "initial_portfolio_value": 10000.0,
            "verbose": 1,
        }

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
            },
        )

        return DummyVecEnv([lambda: env])

    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        self.logger.info(f"Loading model from {self.model_path}")
        # Create a dummy model first, then load checkpoint
        dummy_env = self._create_env()
        self.model = PPO(
            "MlpPolicy",
            dummy_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=42,
        )

        # Load model using Stable Baselines3's load method for zip files
        try:
            # Try loading as Stable Baselines3 zip format first
            self.model = PPO.load(str(self.model_path), env=dummy_env)
            print("Successfully loaded model using Stable Baselines3 load method")
        except Exception as sb3_error:
            print(
                f"Stable Baselines3 load failed: {sb3_error}, trying custom checkpoint format..."
            )

            # Fallback to custom checkpoint loading (LZ4/ZSTD compressed)
            import pickle

            try:
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
                    self.model.value_net.load_state_dict(value_state)  # type: ignore

                print("Successfully loaded model using custom checkpoint format")

            except Exception as custom_error:
                raise RuntimeError(
                    f"Failed to load model with both Stable Baselines3 and custom formats: SB3: {sb3_error}, Custom: {custom_error}"
                )
            # Try to load checkpoint data
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
                self.model.value_net.load_state_dict(value_state)  # type: ignore

        except ImportError:
            raise ImportError("lz4 is required to load compressed checkpoints")

        assert self.model is not None, "Model failed to load"

    def _load_test_data(self) -> None:
        """Load test data for evaluation."""
        if self.test_data_path.exists():
            self.test_df = pd.read_csv(self.test_data_path)
            # Use a subset for testing (e.g., last 20% of data)
            test_size = int(len(self.test_df) * 0.2)
            self.test_df = self.test_df.tail(test_size)
            self.logger.info(f"Using {len(self.test_df)} test samples")
        else:
            self.test_df = None
            self.logger.warning(f"Test data not found: {self.test_data_path}")

    def simulate_trading(self, n_episodes: int = 5) -> Dict[str, Any]:
        """Simulate paper trading for multiple episodes."""
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
            action, _ = self.model.predict(
                predict_obs, deterministic=False
            )  # Use stochastic for scalping

            # Debug: Log action distribution for first few steps
            if steps < 3:
                try:
                    # Get action probabilities if available
                    dist = self.model.policy.get_distribution(predict_obs)
                    if hasattr(dist, "distribution") and hasattr(
                        dist.distribution, "probs"
                    ):
                        probs = dist.distribution.probs.detach().cpu().numpy()
                        print(f"Step {steps}: Action probabilities: {probs}")
                    print(f"Step {steps}: Selected action: {action}")
                except Exception as e:
                    print(f"Could not get action distribution: {e}")
                    print(f"Step {steps}: Selected action: {action}")

            # Record state before action
            prev_portfolio = self.portfolio_value
            prev_position = self.position

            # Execute action
            obs, reward, done_vec, info = self.env.step(action)
            done = done_vec[0]
            reward = reward[0]

            # Update from environment
            self.portfolio_value = self.env.envs[0].portfolio_value
            self.position = self.env.envs[0].position

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
        initial_portfolio = self.config.get("initial_portfolio_value", 10000.0)

        # Calculate average final portfolio across episodes
        if self.episode_results:
            final_portfolio_values = [
                r["final_portfolio"] for r in self.episode_results
            ]
            avg_final_portfolio = float(np.mean(final_portfolio_values))
        else:
            avg_final_portfolio = initial_portfolio

        stats = {
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
        action_counts = {}
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
        results_dir.mkdir(parents=True, exist_ok=True)

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
            with open(args.config, "r") as f:
                file_config = json.load(f)
            # Merge configs, file config takes precedence
            custom_config.update(file_config.get("environment", {}))
            custom_config.update(file_config.get("data", {}))

        # Create paper trader
        trader = PaperTrader(
            args.model_path,
            custom_config.get("test_data", args.test_data),
            config=custom_config,
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
