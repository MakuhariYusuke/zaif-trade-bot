import os
import sys
from pathlib import Path


# Import torch first to avoid DLL initialization errors on Windows

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.backtest.unified_backtest.unified_backtester import UnifiedBacktester


def run_backtest():
    """
    Run optimized v451 backtest using unified backtester.
    """
    # Initialize unified backtester
    backtester = UnifiedBacktester()

    # Define paths
    config_path = os.path.join(
        project_root, "config", "v451", "sac_v451_optimized.json"
    )
    model_path = os.path.join(
        project_root, "models", "sac_v451_phase7_regime_aware.zip"
    )
    # Try alternative model path if primary doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(
            project_root, "checkpoints", "v451", "phase7", "best_model.zip"
        )

    data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    results_dir = os.path.join(project_root, "backtest_results", "v451_optimized")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Run standard backtest
    try:
        result = backtester.run_standard_backtest(
            config_path=config_path,
            model_path=model_path,
            data_path=data_path,
            results_dir=results_dir,
            algorithm="SAC",
        )
        print(
            f"Backtest completed successfully. Final portfolio value: ${result.portfolio_values[-1]:.2f}"
        )
        print(f"Results saved to {results_dir}")

    except Exception as e:
        print(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    run_backtest()
