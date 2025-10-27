#!/usr/bin/env python3
"""
SAC v436 All Variants Backtest - Backtest all three trained models
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.backtest.backtest_engine import BacktestEngine
from ztb.utils.logging_utils import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def backtest_variant(model_path: str, config_path: str, variant_name: str) -> bool:
    """Backtest a single variant."""
    print(f"\n🚀 Backtesting {variant_name}...")
    print("-" * 40)

    try:
        # 設定読み込み
        config = load_config(config_path)

        # モデルパス設定
        config["backtest"]["model_path"] = model_path

        print("📋 Backtest Configuration:")
        print(f"  - Model: {model_path}")
        print(f"  - Data: {config['backtest']['data_path']}")
        print(f"  - Initial Balance: {config['backtest']['initial_balance']:,}")

        # バックテスト実行
        engine = BacktestEngine(config)
        results = engine.run_backtest()

        # 結果保存
        output_dir = Path("backtest_results") / "v436_variants"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"backtest_results_{variant_name}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 主要指標表示
        metrics = results.get("metrics", {})
        print("📊 Backtest Results:")
        print(f"  - Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  - Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"  - Total Trades: {metrics.get('total_trades', 0)}")

        # アクション分布分析
        action_dist = results.get("action_distribution", {})
        print("🎯 Action Distribution:")
        total_actions = sum(action_dist.values())
        if total_actions > 0:
            buy_pct = action_dist.get("BUY", 0) / total_actions * 100
            sell_pct = action_dist.get("SELL", 0) / total_actions * 100
            hold_pct = action_dist.get("HOLD", 0) / total_actions * 100
            print(f"  - BUY: {buy_pct:.1f}%")
            print(f"  - SELL: {sell_pct:.1f}%")
            print(f"  - HOLD: {hold_pct:.1f}%")

            # BUY biasチェック
            if buy_pct > 80:
                print("⚠️  HIGH BUY BIAS DETECTED!")
            elif buy_pct < 20:
                print("⚠️  HIGH SELL BIAS DETECTED!")
            else:
                print("✅ Balanced action distribution")

        print(f"💾 Results saved to: {results_file}")
        return True

    except Exception as e:
        print(f"❌ {variant_name} backtest failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="SAC v436 All Variants Backtest")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ログレベル設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    print("🚀 SAC v436 All Variants Backtest")
    print("=" * 50)
    print("Backtesting all three trained signal guidance variants")
    print()

    # モデルと設定のマッピング
    variants = [
        (
            "models/sac_v436_signal_guided.zip",
            "config/sac_v436_signal_guided_config.json",
            "full_guidance",
        ),
        (
            "models/sac_v436_no_guidance.zip",
            "config/sac_v436_no_guidance_config.json",
            "no_guidance",
        ),
        (
            "models/sac_v436_fade_out.zip",
            "config/sac_v436_fade_out_config.json",
            "fade_out",
        ),
    ]

    results = []
    for model_path, config_path, variant_name in variants:
        # モデルファイル存在チェック
        if not Path(model_path).exists():
            print(f"⚠️  Model not found: {model_path} - skipping {variant_name}")
            results.append((variant_name, False))
            continue

        success = backtest_variant(model_path, config_path, variant_name)
        results.append((variant_name, success))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("🎯 Backtest Summary:")
    for variant_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {variant_name}: {status}")

    successful_count = sum(1 for _, success in results if success)
    print(f"\n📊 {successful_count}/{len(results)} variants backtested successfully")

    if successful_count == len(results):
        print("🎉 All variants backtested successfully!")
        return 0
    else:
        print("⚠️  Some variants failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
