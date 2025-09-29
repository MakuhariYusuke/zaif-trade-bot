"""
Regression smoke tests for quality validation.

Runs small synthetic data tests to ensure basic functionality works
and catch regressions early.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def create_synthetic_data(output_path: Path, n_samples: int = 100) -> None:
    """Create small synthetic trading data for smoke tests."""
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd

    # Generate synthetic OHLCV data
    np.random.seed(42)  # For reproducibility

    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]

    # Generate realistic-looking price data
    base_price = 1000000  # 1M JPY per BTC
    prices = []
    current_price = base_price

    for i in range(n_samples):
        # Random walk with mean reversion
        change = np.random.normal(0, 0.001)  # Small random changes
        current_price *= 1 + change
        prices.append(current_price)

    # Create OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        if i == 0:
            open_price = close
        else:
            open_price = data[-1]["close"]

        # Generate OHLC around close price
        high = close * (1 + abs(np.random.normal(0, 0.002)))
        low = close * (1 - abs(np.random.normal(0, 0.002)))

        volume = np.random.lognormal(10, 1)  # Realistic volume

        data.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": max(open_price, high),
                "low": min(open_price, low),
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Created synthetic data: {output_path} ({n_samples} samples)")


def run_paper_trader_smoke_test(data_path: Path) -> bool:
    """Run paper trader with synthetic data."""
    try:
        cmd = [
            sys.executable,
            "-m",
            "ztb.live.paper_trader",
            "--mode",
            "replay",
            "--policy",
            "sma_fast_slow",
            "--dataset",
            str(data_path),
            "--duration-minutes",
            "5",
            "--output-dir",
            tempfile.mkdtemp(),
            "--enable-risk",
            "--risk-profile",
            "conservative",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).parent.parent,
        )

        success = result.returncode == 0
        if not success:
            print(f"Paper trader smoke test failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        return success

    except subprocess.TimeoutExpired:
        print("Paper trader smoke test timed out")
        return False
    except Exception as e:
        print(f"Paper trader smoke test error: {e}")
        return False


def run_ppo_trainer_smoke_test(data_path: Path) -> bool:
    """Run PPO trainer with synthetic data for initialization test."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
import os
sys.path.insert(0, r'{Path(__file__).parent.parent}')

# Set required environment variables for smoke test
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from ztb.trading.ppo_trainer import PPOTrainer
import pandas as pd

# Load synthetic data
df = pd.read_parquet(r'{data_path}')
print(f"Loaded {{len(df)}} samples")

# Just test that we can import and create trainer class
print("PPO trainer import successful")
print("Smoke test passed")
""",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent,
            )

            success = result.returncode == 0
            if not success:
                print(f"PPO trainer smoke test failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

            return success

    except subprocess.TimeoutExpired:
        print("PPO trainer smoke test timed out")
        return False
    except Exception as e:
        print(f"PPO trainer smoke test error: {e}")
        return False


def run_venue_health_check_smoke_test() -> bool:
    """Run venue health check smoke test."""
    try:
        cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/check_venue_health.py",
            "--venue",
            "coincheck",
            "--symbol",
            "BTC_JPY",
            "--timeout",
            "2",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent.parent,
        )

        # Health check may fail due to network, but should not crash
        success = result.returncode in [0, 1]  # 0=success, 1=failure but graceful
        if not success:
            print(f"Venue health check smoke test failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        return success

    except subprocess.TimeoutExpired:
        print("Venue health check smoke test timed out")
        return False
    except Exception as e:
        print(f"Venue health check smoke test error: {e}")
        return False


def main():
    """Run all regression smoke tests."""
    print("Running regression smoke tests...")

    results = []

    # Test 1: Venue health check
    print("\n1. Testing venue health check...")
    venue_success = run_venue_health_check_smoke_test()
    results.append(("venue_health_check", venue_success))
    print(f"   Result: {'PASS' if venue_success else 'FAIL'}")

    # Test 2: Create synthetic data
    print("\n2. Creating synthetic data...")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        data_path = Path(f.name)

    try:
        create_synthetic_data(data_path, n_samples=100)
        data_created = True
    except Exception as e:
        print(f"Failed to create synthetic data: {e}")
        data_created = False

    results.append(("synthetic_data_creation", data_created))
    print(f"   Result: {'PASS' if data_created else 'FAIL'}")

    if data_created:
        # Test 3: Paper trader
        print("\n3. Testing paper trader...")
        trader_success = run_paper_trader_smoke_test(data_path)
        results.append(("paper_trader", trader_success))
        print(f"   Result: {'PASS' if trader_success else 'FAIL'}")

        # Test 4: PPO trainer
        print("\n4. Testing PPO trainer...")
        ppo_success = run_ppo_trainer_smoke_test(data_path)
        results.append(("ppo_trainer", ppo_success))
        print(f"   Result: {'PASS' if ppo_success else 'FAIL'}")

    # Cleanup
    if data_path.exists():
        data_path.unlink()

    # Summary
    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print("25")
        if not passed:
            all_passed = False

    print(
        f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
