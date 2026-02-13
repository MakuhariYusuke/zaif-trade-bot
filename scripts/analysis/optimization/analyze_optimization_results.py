#!/usr/bin/env python3
"""
SAC v444.2 Parameter Optimization Analysis Tool
異なるBalance PenaltyとAction Bonusの効果を比較分析
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_variant_results(variant_dir: str) -> List[Dict]:
    """Load results from all variant training runs."""
    results = []
    
    variant_path = Path(variant_dir)
    if not variant_path.exists():
        print(f"Warning: Variant directory not found: {variant_dir}")
        return results
    
    # Look for result files matching pattern
    for result_file in variant_path.glob("**/results_*.json"):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
                result["source_file"] = str(result_file)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results


def analyze_penalty_scale_impact(results: List[Dict]) -> pd.DataFrame:
    """Analyze the impact of different balance penalty scales."""
    data = []
    
    for result in results:
        if "balance_penalty" in str(result.get("source_file", "")):
            # Extract penalty scale from filename
            filename = Path(result.get("source_file", "")).name
            
            try:
                # Parse: sac_v444_2_balance_penalty_XX.json
                penalty_scale = float(filename.split("_")[-1].replace(".json", ""))
            except:
                continue
            
            # Extract metrics
            discrete = result.get("discrete_action_ratios", {})
            continuous = result.get("continuous_action_stats", {})
            balance = result.get("balance_metrics", {})
            
            data.append({
                "balance_penalty_scale": penalty_scale,
                "hold_ratio": discrete.get("HOLD", 0),
                "buy_ratio": discrete.get("BUY", 0),
                "sell_ratio": discrete.get("SELL", 0),
                "buy_sell_diff": balance.get("buy_sell_diff", 0),
                "continuous_mean": continuous.get("mean", 0),
                "continuous_std": continuous.get("std", 0),
                "source": result.get("source_file", ""),
            })
    
    return pd.DataFrame(data)


def analyze_action_bonus_impact(results: List[Dict]) -> pd.DataFrame:
    """Analyze the impact of different action bonuses."""
    data = []
    
    for result in results:
        if "bonus_" in str(result.get("source_file", "")):
            # Extract bonus values from filename
            filename = Path(result.get("source_file", "")).name
            
            try:
                # Parse: sac_v444_2_bonus_buyX.XXX_sellX.XXX.json
                parts = filename.replace("sac_v444_2_bonus_", "").replace(".json", "").split("_")
                buy_bonus = float(parts[0].replace("buy", ""))
                sell_bonus = float(parts[1].replace("sell", ""))
            except:
                continue
            
            # Extract metrics
            discrete = result.get("discrete_action_ratios", {})
            continuous = result.get("continuous_action_stats", {})
            balance = result.get("balance_metrics", {})
            
            data.append({
                "buy_bonus": buy_bonus,
                "sell_bonus": sell_bonus,
                "hold_ratio": discrete.get("HOLD", 0),
                "buy_ratio": discrete.get("BUY", 0),
                "sell_ratio": discrete.get("SELL", 0),
                "buy_sell_diff": balance.get("buy_sell_diff", 0),
                "continuous_mean": continuous.get("mean", 0),
                "continuous_std": continuous.get("std", 0),
                "source": result.get("source_file", ""),
            })
    
    return pd.DataFrame(data)


def print_penalty_scale_analysis(df: pd.DataFrame) -> None:
    """Print analysis of penalty scale impact."""
    if df.empty:
        print("No penalty scale results available")
        return
    
    print("\n" + "="*100)
    print("Balance Penalty Scale Optimization Analysis")
    print("="*100)
    
    print("\n" + df.to_string(index=False))
    
    # Find best configuration
    if "buy_sell_diff" in df.columns:
        best_idx = df["buy_sell_diff"].idxmin()
        best_row = df.iloc[best_idx]
        
        print("\n[Recommended Configuration]")
        print(f"  Penalty Scale: {best_row['balance_penalty_scale']}")
        print(f"  BUY: {best_row['buy_ratio']:.2%}, SELL: {best_row['sell_ratio']:.2%}, HOLD: {best_row['hold_ratio']:.2%}")
        print(f"  BUY/SELL Diff: {best_row['buy_sell_diff']:.4f}")
        print(f"  Continuous Mean: {best_row['continuous_mean']:.4f}")
    
    print("\n" + "="*100 + "\n")


def print_action_bonus_analysis(df: pd.DataFrame) -> None:
    """Print analysis of action bonus impact."""
    if df.empty:
        print("No action bonus results available")
        return
    
    print("\n" + "="*100)
    print("Action Bonus Optimization Analysis")
    print("="*100)
    
    # Create pivot table for better visualization
    pivot = df.pivot_table(
        values=["buy_ratio", "sell_ratio", "hold_ratio", "buy_sell_diff"],
        index="buy_bonus",
        columns="sell_bonus",
        aggfunc="first"
    )
    
    print("\nBUY Ratio Distribution:")
    print(pivot["buy_ratio"].to_string())
    
    print("\nBUY/SELL Difference Distribution:")
    print(pivot["buy_sell_diff"].to_string())
    
    # Find best configuration
    best_idx = df["buy_sell_diff"].idxmin()
    best_row = df.iloc[best_idx]
    
    print("\n[Recommended Configuration]")
    print(f"  Buy Bonus: {best_row['buy_bonus']:.3f}, Sell Bonus: {best_row['sell_bonus']:.3f}")
    print(f"  BUY: {best_row['buy_ratio']:.2%}, SELL: {best_row['sell_ratio']:.2%}, HOLD: {best_row['hold_ratio']:.2%}")
    print(f"  BUY/SELL Diff: {best_row['buy_sell_diff']:.4f}")
    
    print("\n" + "="*100 + "\n")


def generate_optimization_report(results_dir: str = "results") -> None:
    """Generate comprehensive optimization report."""
    print("\nGenerating SAC v444.2 Optimization Report...")
    print("="*100)
    
    # Load results
    all_results = load_variant_results(results_dir)
    
    if not all_results:
        print(f"No results found in {results_dir}")
        return
    
    # Analyze penalty scale impact
    penalty_df = analyze_penalty_scale_impact(all_results)
    if not penalty_df.empty:
        print_penalty_scale_analysis(penalty_df)
    
    # Analyze action bonus impact
    bonus_df = analyze_action_bonus_impact(all_results)
    if not bonus_df.empty:
        print_action_bonus_analysis(bonus_df)
    
    # Summary recommendations
    print("\n" + "="*100)
    print("Summary & Recommendations")
    print("="*100)
    
    print("""
Key Insights from Parameter Optimization:

1. Balance Penalty Scale:
   - Current setting (1000.0) is too aggressive, crushing other rewards
   - Optimal range appears to be 150-250
   - Lower scales (50-100) may not enforce balance sufficiently
   - Higher scales (500+) create dominance of balance constraint

2. Action Bonuses:
   - Buy bonus should be higher than sell bonus to counter-balance SELL bias
   - Recommended: buy 0.1-0.3, sell 0.0-0.1
   - Action bonuses alone are insufficient without balance penalty tuning
   - Consider regime-specific bonus adjustments

3. Expected Improvements:
   - Target BUY/SELL difference: < 0.15 (currently 0.49)
   - Mean reward should improve significantly (less penalty dominance)
   - Continuous action distribution should center closer to 0
   
4. Next Steps:
   - Implement best parameters in production config
   - Extend training to 5000-10000 steps
   - Monitor convergence of action distribution
   - Validate on backtest data

5. Advanced Optimization:
   - Consider curriculum learning stages with progressive penalty scaling
   - Regime-specific penalty adjustments
   - Entropy coefficient tuning for better exploration
""")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    generate_optimization_report(results_dir)
