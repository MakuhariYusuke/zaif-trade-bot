import sys
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import (
    extract_env_config,
    extract_sac_params,
    load_config_dict,
)
from ztb.optimization.parallel.window_evaluator import ParallelWindowEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Parallel Training (Walk-Forward Analysis)...")

    # Load config once (shared by env and algorithm)
    config_path = PROJECT_ROOT / "config/v457/base/config.yaml"
    env_config_dict = {}
    sac_params = {}

    if config_path.exists():
        try:
            full_config = load_config_dict(config_path)
            env_config_dict = extract_env_config(full_config)
            sac_params = extract_sac_params(full_config)
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}. Using defaults.")

    if not sac_params:
        sac_params = {
            "learning_rate": 5e-5,
            "buffer_size": 100000,
            "batch_size": 2048,
            "ent_coef": 0.05,
            "gamma": 0.8,
            "tau": 0.005,
        }

    def env_factory(df: pd.DataFrame) -> Any:
        """Create environment from DataFrame."""
        env = create_fast_intraday_env_v456(df=df, env_config=env_config_dict)
        if env is None:
            raise RuntimeError("Failed to create FastIntradayEnvV456.")
        return env

    def algorithm_factory(env: Any) -> SAC:
        """Create SAC model."""
        return SAC(
            "MlpPolicy",
            env,
            verbose=0,
            **sac_params,
        )
    
    # 1. Load Data
    data_path = PROJECT_ROOT / "data/yahoo_finance/btc_jpy_1m.csv"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col=0)
    logger.info(f"Loaded {len(df)} rows.")

    # 2. Pre-calculate features (v457 Requirement)
    # The Parallel Evaluator slices the DF, so we must calculate features on the FULL DF first
    # to avoid edge effects at window boundaries and ensure columns exist.
    logger.info("Calculating base features...")
    df = calculate_base_features(df, copy=False)
    
    # 3. Define Windows (Expanding Anchor)
    # Dataset is small (7000), so we use small steps
    total_len = len(df)
    test_len = 500
    val_len = 500
    
    # Generate 4 windows
    # Window 0: Train [0:3000], Val [3000:3500], Test [3500:4000]
    # Window 1: Train [0:4000], Val [4000:4500], Test [4500:5000]
    # ...
    
    windows = []
    start_train_size = 3000
    step_size = 1000
    
    for i in range(4):
        train_end = start_train_size + (i * step_size)
        if train_end + val_len + test_len > total_len:
            break
            
        val_end = train_end + val_len
        test_end = val_end + test_len
        
        windows.append((train_end, val_end, test_end))
        
    logger.info(f"Defined {len(windows)} windows for parallel execution.")
    
    # 4. Initialize Parallel Evaluator
    evaluator = ParallelWindowEvaluator(
        num_workers=4, # Use 4 cores
        enable_error_collection=True
    )
    
    # 5. Run Evaluation
    # Training each window for 20,000 steps (Total equivalent to ~80k serial steps if sequential)
    timesteps = 20000 
    
    logger.info(f"Starting execution: {timesteps} steps per window...")
    results, errors = evaluator.evaluate_windows_parallel(
        df=df,
        windows=windows,
        timesteps=timesteps,
        env_factory=env_factory,
        algorithm_factory=algorithm_factory
    )
    
    # 6. Report
    logger.info("=" * 60)
    logger.info("Parallel Training Results")
    logger.info("=" * 60)
    
    if errors:
        logger.error(f"Errors occurred in {len(errors)} windows.")
        for wid, err in errors.items():
            logger.error(f"Window {wid}: {err}")
            
    for wid, perf in results.items():
        logger.info(f"Window {wid}: Sharpe={perf.sharpe_ratio:.4f}, Sortino={perf.sortino_ratio:.4f}, MaxDD={perf.max_drawdown:.4f} Return={perf.total_return:.4f}")
        
    # Combine results logic if needed, or pick best
    # For now, just logging validates the process.

if __name__ == "__main__":
    # Windows Safe Multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
