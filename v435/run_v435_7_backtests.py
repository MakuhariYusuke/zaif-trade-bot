#!/usr/bin/env python3
"""
SAC v435.7 Backtest Runner
v435.7モデルのバックテスト実行
"""
import subprocess
import sys
from pathlib import Path


def run_backtest(model_name, config_path):
    """バックテストを実行"""
    print(f"\n🔍 Running backtest for {model_name}")

    try:
        # バックテストコマンドを実行
        cmd = [
            sys.executable,
            "-m",
            "ztb.analysis.sac_backtester",
            "--config",
            str(config_path),
            "--model",
            f"models/{model_name}.zip",
            "--data",
            "data/btc_jpy_yahoo_real_20251021_featured.csv",
            "--episodes",
            "10",
            "--deterministic",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            print(f"✅ Backtest completed for {model_name}")
            print(result.stdout[-500:])  # 最後の500文字を表示
            return True
        else:
            print(f"❌ Backtest failed for {model_name}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running backtest for {model_name}: {e}")
        return False


def main():
    print("🚀 SAC v435.7 Backtest Runner")
    print("=" * 40)

    config_dir = Path("v435/v435.7")
    models = [
        ("sac_v435.7a", config_dir / "sac_v435_7a_config.json"),
        ("sac_v435.7b", config_dir / "sac_v435_7b_config.json"),
        ("sac_v435.7c", config_dir / "sac_v435_7c_config.json"),
    ]

    results = []
    for model_name, config_path in models:
        if config_path.exists():
            success = run_backtest(model_name, config_path)
            results.append((model_name, success))
        else:
            print(f"❌ Config not found: {config_path}")
            results.append((model_name, False))

    print("\n📊 Backtest Summary:")
    for model_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {model_name}: {status}")

    successful_models = [model for model, success in results if success]
    if successful_models:
        print(
            f"\n✅ {len(successful_models)}/{len(results)} models backtested successfully"
        )
        print("📁 Results saved to v435/backtest_results_*.json")
    else:
        print("\n❌ All backtests failed")


if __name__ == "__main__":
    main()
