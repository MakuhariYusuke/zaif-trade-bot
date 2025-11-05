#!/usr/bin/env python3
"""
SAC v444 Improved Training Script - Multi-Configuration Testing
テスト複数のパラメータセット: Balance Penalty Scale (200, 300, 500)
アクションバイアス改善と報酬最適化の段階的検証
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys

# Configuration sets to test
CONFIG_SETS = {
    "scale_200": {
        "config_path": "config/sac_v444_3_balanced_penalty_scale_200.json",
        "description": "Minimal balance penalty (200) with increased action bonuses",
        "expected_reward_range": (-5000, -2000),
        "expected_buy_ratio": (0.25, 0.40),
    },
    "scale_300": {
        "config_path": "config/sac_v444_4_balanced_penalty_scale_300.json",
        "description": "Moderate balance penalty (300) with higher action bonuses",
        "expected_reward_range": (-4000, -1500),
        "expected_buy_ratio": (0.30, 0.45),
    },
    "scale_500": {
        "config_path": "config/sac_v444_5_balanced_penalty_scale_500.json",
        "description": "Higher balance penalty (500) with maximum action bonuses",
        "expected_reward_range": (-3000, -500),
        "expected_buy_ratio": (0.35, 0.50),
    },
}


def run_training(config_path: str, description: str) -> dict:
    """
    Run training with specified configuration
    設定されたコンフィグでtraining実行
    """
    print("\n" + "="*80)
    print(f"🚀 Starting Training: {description}")
    print(f"📋 Config: {config_path}")
    print("="*80)
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return {"status": "failed", "reason": "config_not_found"}
    
    try:
        # Run quick_train_v444 with config
        cmd = [
            sys.executable,
            "quick_train_v444.py",
            "--config", config_path,
            "--verbose"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Training completed successfully")
            return {"status": "success"}
        else:
            print(f"❌ Training failed with return code {result.returncode}")
            print(f"stderr: {result.stderr[:500]}")
            return {"status": "failed", "reason": "training_error"}
            
    except subprocess.TimeoutExpired:
        print("⏱️ Training timeout (1 hour)")
        return {"status": "failed", "reason": "timeout"}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"status": "failed", "reason": str(e)}


def analyze_results(config_name: str) -> dict:
    """
    Analyze training results for a configuration
    結果の分析
    """
    print(f"\n📊 Analyzing results for {config_name}...")
    
    # Load training results if available
    results_path = f"results/sac_v444_{config_name}_analysis.json"
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            print(f"✅ Analysis loaded from {results_path}")
            return results
        except Exception as e:
            print(f"⚠️ Could not load results: {str(e)}")
            return {}
    else:
        print(f"⚠️ Results not found at {results_path}")
        return {}


def create_comparison_report(all_results: dict) -> str:
    """
    Create comparison report for all configurations
    すべての設定の比較レポート作成
    """
    report = []
    report.append("\n" + "="*80)
    report.append("📈 TRAINING RESULTS COMPARISON REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().isoformat()}\n")
    
    for config_name, result in all_results.items():
        report.append(f"\n{'='*40}")
        report.append(f"Configuration: {config_name}")
        report.append(f"{'='*40}")
        
        config_info = CONFIG_SETS.get(config_name, {})
        report.append(f"Description: {config_info.get('description', 'N/A')}")
        
        if result.get("status") == "success":
            report.append("Status: ✅ SUCCESS")
            
            # Show key metrics if available
            analysis = result.get("analysis", {})
            if analysis:
                report.append(f"\nKey Metrics:")
                report.append(f"  • Mean Reward: {analysis.get('mean_reward', 'N/A')}")
                report.append(f"  • BUY Ratio: {analysis.get('buy_ratio', 'N/A')}")
                report.append(f"  • SELL Ratio: {analysis.get('sell_ratio', 'N/A')}")
                report.append(f"  • HOLD Ratio: {analysis.get('hold_ratio', 'N/A')}")
                report.append(f"  • Total Return: {analysis.get('total_return', 'N/A')}")
        else:
            report.append(f"Status: ❌ FAILED")
            report.append(f"Reason: {result.get('reason', 'Unknown')}")
    
    report.append("\n" + "="*80)
    report.append("NEXT STEPS:")
    report.append("="*80)
    report.append("""
1. Review the metrics for each configuration
2. Select the configuration with:
   - Most balanced BUY/SELL ratio
   - Highest mean reward
   - Stable training progress
3. Run backtest on the best model
4. Fine-tune hyperparameters if needed
5. Deploy to production after validation
    """)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="SAC v444 Multi-Config Training")
    parser.add_argument("--config", help="Test specific config (scale_200, scale_300, scale_500)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only analyze")
    parser.add_argument("--compare", action="store_true", help="Compare all configs")
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.config:
        # Test specific configuration
        if args.config not in CONFIG_SETS:
            print(f"❌ Unknown config: {args.config}")
            print(f"Available: {', '.join(CONFIG_SETS.keys())}")
            return
        
        configs_to_test = {args.config: CONFIG_SETS[args.config]}
    else:
        # Test all configurations
        configs_to_test = CONFIG_SETS
    
    # Run training for each configuration
    if not args.skip_training:
        for config_name, config_info in configs_to_test.items():
            result = run_training(
                config_info["config_path"],
                config_info["description"]
            )
            all_results[config_name] = result
            
            # Analyze results
            if result.get("status") == "success":
                analysis = analyze_results(config_name)
                result["analysis"] = analysis
    else:
        # Just analyze existing results
        for config_name in configs_to_test:
            analysis = analyze_results(config_name)
            all_results[config_name] = {"status": "analyzed", "analysis": analysis}
    
    # Create comparison report
    report = create_comparison_report(all_results)
    print(report)
    
    # Save report
    report_path = f"results/training_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("results", exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n💾 Report saved to {report_path}")


if __name__ == "__main__":
    main()
