#!/usr/bin/env python3
"""
Quick Train SAC v444 Configurable - Direct Environment Training

Fast training script for SAC v444 with direct environment usage.
Supports verbose output and quick testing of different configurations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import SAC
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.utils.constants import DEFAULT_SEED
    from ztb.training.constants import DEFAULT_BUFFER_SIZE_SAC, DEFAULT_BATCH_SIZE_SAC, DEFAULT_LEARNING_RATE_SAC, DEFAULT_LEARNING_STARTS_SAC, DEFAULT_GAMMA, DEFAULT_TAU, DEFAULT_ENT_COEF_AUTO, DEFAULT_TARGET_UPDATE_INTERVAL, DEFAULT_MAX_TRAIN_STEPS, DEFAULT_BUFFER_STEPS
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Attempting to continue with available modules...")


class DirectTrainer:
    """Direct trainer without unified trainer complexity."""

    def __init__(self, config_path: str, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self.logger = self._setup_logging()

    def _load_config(self) -> dict:
        """Load config directly from JSON."""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Config: {self.config_path}")
        return logger

    def _load_data(self) -> pd.DataFrame:
        """Load sample data."""
        np.random.seed(DEFAULT_SEED)
        dates = pd.date_range("2023-01-01", periods=2000, freq="1h")
        base_price = 5000000
        price_changes = np.random.normal(0, 0.005, 2000).cumsum()
        close = pd.Series(base_price * (1 + price_changes), index=dates)
        high = close * (1 + np.abs(np.random.normal(0, 0.002, 2000)))
        low = close * (1 - np.abs(np.random.normal(0, 0.002, 2000)))
        open_price = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(np.random.uniform(1000, 10000, 2000), index=dates)

        df = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": dates,
        })

        # Add technical indicators
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["RSI"] = 50
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["BB_Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
        df["BB_Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()

        return df.ffill().bfill()

    def _prepare_env_config(self) -> dict:
        """Prepare environment config by expanding nested parameters."""
        env_config = self.config['environment'].copy()

        # Expand nested configs - but keep behavior_optimization nested for proper mapping
        # if 'behavior_optimization' in env_config:
        #     env_config.update(env_config['behavior_optimization'])

        if 'action_bonuses' in env_config:
            env_config.update(env_config['action_bonuses'])

        return env_config

    def train(self) -> bool:
        """Execute training."""
        try:
            self.logger.info("="*80)
            self.logger.info("🚀 Starting direct SAC v444 training")
            self.logger.info("="*80)

            # Load data
            df = self._load_data()

            # Setup environment
            env_config = self._prepare_env_config()
            env = HeavyTradingEnv(df, env_config)

            # Create model
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=DEFAULT_LEARNING_RATE_SAC,
                buffer_size=DEFAULT_BUFFER_SIZE_SAC,
                learning_starts=DEFAULT_LEARNING_STARTS_SAC,
                batch_size=DEFAULT_BATCH_SIZE_SAC,
                tau=DEFAULT_TAU,
                gamma=DEFAULT_GAMMA,
                ent_coef=DEFAULT_ENT_COEF_AUTO,
                target_update_interval=DEFAULT_TARGET_UPDATE_INTERVAL,
                verbose=2 if self.verbose else 0,
            )

            # Train for 2000 steps
            self.logger.info("Training for 2000 timesteps...")
            
            # Track training history
            training_history = []
            obs, _ = env.reset()
            
            # Get maximum available steps from environment
            max_steps = env.data_manager.n_steps
            train_steps = min(DEFAULT_MAX_TRAIN_STEPS, max_steps - DEFAULT_BUFFER_STEPS)  # Leave some buffer
            
            self.logger.info(f"Training for {train_steps} timesteps (data allows up to {max_steps})")
            
            for step in range(train_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record step data
                # Get balance_penalty from reward_calculator if available
                balance_penalty = 0.0
                if hasattr(env, 'reward_calculator') and hasattr(env.reward_calculator, '_get_behavior_opt'):
                    try:
                        balance_penalty = env.reward_calculator._get_behavior_opt('balance_penalty', 0.0)
                        # Debug: log what the method returns
                        self.logger.info(f"Step {step}: _get_behavior_opt returned balance_penalty={balance_penalty}")
                    except Exception as e:
                        self.logger.info(f"Could not get balance_penalty from reward_calculator: {e}")
                        self.logger.info(f"Exception type: {type(e).__name__}")
                
                # Additional debug: log balance_penalty before creating step_data
                self.logger.info(f"Step {step}: balance_penalty before step_data creation: {balance_penalty}")
                
                step_data = {
                    'step': step,
                    'action': int(action),
                    'reward': float(reward),
                    'balance_penalty': float(balance_penalty),
                    'portfolio_return': float(info.get('portfolio_return', 0.0)),
                    'position': float(info.get('position', 0.0))
                }
                
                # Additional debug: log step_data balance_penalty
                self.logger.info(f"Step {step}: step_data balance_penalty: {step_data['balance_penalty']}")
                
                training_history.append(step_data)
                
                if step % 500 == 0:
                    self.logger.info(f"Step {step}: action={action}, reward={reward:.2f}, balance_penalty={step_data['balance_penalty']:.2f}")
                    
                    # Check SAC model parameters for exploration analysis
                    try:
                        if hasattr(model, 'actor') and hasattr(model.actor, 'log_std'):
                            log_std = model.actor.log_std.detach().cpu().numpy()
                            log_std_mean = float(np.mean(log_std))
                            log_std_std = float(np.std(log_std))
                            self.logger.info(f"Step {step}: SAC log_std mean={log_std_mean:.4f}, std={log_std_std:.4f}")
                            
                            # Check if exploration is collapsed (log_std values close to -20)
                            if log_std_mean < -15:
                                self.logger.warning(f"Step {step}: SAC exploration may be collapsed (log_std={log_std_mean:.4f})")
                        
                        # Check entropy coefficient if available
                        if hasattr(model, 'ent_coef') and model.ent_coef is not None:
                            if hasattr(model.ent_coef, 'item'):
                                ent_coef = model.ent_coef.item()
                            else:
                                ent_coef = float(model.ent_coef)
                            self.logger.info(f"Step {step}: SAC ent_coef={ent_coef:.6f}")
                            
                            if ent_coef < 0.001:
                                self.logger.warning(f"Step {step}: SAC ent_coef very low ({ent_coef:.6f}), exploration may be limited")
                        
                        # Check actor mu (mean action) if available
                        if TORCH_AVAILABLE and hasattr(model, 'actor') and hasattr(model.actor, 'mu'):
                            try:
                                # Get a sample observation to check mu output
                                obs_sample = env.reset()
                                if isinstance(obs_sample, tuple):
                                    obs_sample = obs_sample[0]
                                mu_output = model.actor.mu(torch.tensor(obs_sample, dtype=torch.float32).unsqueeze(0))
                                mu_mean = float(torch.mean(mu_output).detach().cpu().numpy())
                                mu_std = float(torch.std(mu_output).detach().cpu().numpy())
                                self.logger.info(f"Step {step}: SAC actor mu mean={mu_mean:.4f}, std={mu_std:.4f}")
                            except Exception as e:
                                self.logger.debug(f"Could not check actor mu: {e}")
                                
                    except Exception as e:
                        self.logger.debug(f"Could not check SAC parameters: {e}")
            
            # Save training results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"results/sac_v444_training_results_{timestamp}.json"
            
            results_data = {
                'config': self.config,
                'training_history': training_history,
                'total_steps': train_steps,
                'timestamp': timestamp
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Training completed - results saved to {results_file}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            return False


def main() -> bool:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Quick Train SAC v444 - Direct Environment Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        print("🚀 Quick Train SAC v444 - Direct Environment Training")
        print(f"Configuration: {args.config}")
        if args.verbose:
            print("Verbose mode enabled")

        trainer = DirectTrainer(args.config, verbose=args.verbose)
        success = trainer.train()

        if success:
            print("✅ SAC v444 training completed successfully!")
            return True
        else:
            print("❌ SAC v444 training failed!")
            return False

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)