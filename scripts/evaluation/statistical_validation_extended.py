"""
Statistical Validation Script - T-test and P-mean Method for Model Performance

Compares the performance of multiple models using t-test and p-mean method
to determine statistical significance across multiple metrics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def load_backtest_results(result_path: Path) -> Dict:
    """Load backtest results from JSON file"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def p_mean_method(p_values: List[float], method: str = 'geometric') -> float:
    """
    p平均法による総合p値の計算

    p平均法は、複数の独立した統計検定のp値を統合し、
    全体として統計的有意性があるかを評価する手法です。

    Args:
        p_values: p値のリスト（0.0 ~ 1.0の範囲）
        method: 平均化手法
            - 'arithmetic': 算術平均（単純平均）
            - 'geometric': 幾何平均（対数変換後平均）

    Returns:
        総合p値（0.0 ~ 1.0）
    """
    if not p_values:
        return 1.0

    p_array = np.array(p_values)

    if method == 'arithmetic':
        # 算術平均
        return float(np.mean(p_array))
    elif method == 'geometric':
        # 幾何平均 (0を避けるため小さな値を加算)
        p_array = np.clip(p_array, 1e-10, 1.0)
        return float(np.exp(np.mean(np.log(p_array))))
    else:
        raise ValueError(f"Unknown method: {method}")

def perform_comparison_with_single_results(model_a_results: List[Dict], model_b_results: List[Dict],
                                               metrics: List[str]) -> Dict[str, Any]:
    """
    Compare models when only single results are available (no statistical test possible)
    """
    results = {}

    for metric in metrics:
        val_a = model_a_results[0].get(metric, 0)
        val_b = model_b_results[0].get(metric, 0)
        diff = val_b - val_a
        percent_diff = (diff / val_a * 100) if val_a != 0 else 0

        results[metric] = {
            'model_a_value': val_a,
            'model_b_value': val_b,
            'difference': diff,
            'percent_difference': percent_diff,
            'model_b_better': val_b > val_a
        }

    return results

def interpret_results(p_value: float, cohens_d: float, alpha: float) -> str:
    """Interpret t-test results"""
    significance = "significant" if p_value < alpha else "not significant"

    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    return f"The difference is {significance} (p={p_value:.4f}) with {effect_size} effect size (d={cohens_d:.3f})"

def main():
    print("=" * 80)
    print("Statistical Validation - T-test and P-mean Method")
    print("=" * 80)

    # Define models to compare
    models = {
        'v420_forced_balance': 'results/sac_v420_forced_balance_backtest.json',
        'v501_fine_tune_1': 'results/sac_v501_fine_tune_1_backtest.json',
        'v502_fine_tune_2': 'results/sac_v502_fine_tune_2_backtest.json',
        'v503_fine_tune_3': 'results/sac_v503_fine_tune_3_backtest.json'
    }

    # Metrics to compare
    metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'total_pnl']

    # Load all results
    results_data = {}
    for model_name, result_path in models.items():
        full_path = project_root / result_path
        if full_path.exists():
            results_data[model_name] = [load_backtest_results(full_path)]
            print(f"✅ Loaded {model_name} results: {full_path}")
        else:
            print(f"❌ {model_name} results not found: {full_path}")
            return

    # Compare v420 (production) vs each research model
    baseline_model = 'v420_forced_balance'
    research_models = ['v501_fine_tune_1', 'v502_fine_tune_2', 'v503_fine_tune_3']

    all_comparisons = {}

    for research_model in research_models:
        print(f"\n🔬 Comparing {baseline_model} vs {research_model}...")

        comparison_results = perform_comparison_with_single_results(
            results_data[baseline_model],
            results_data[research_model],
            metrics
        )

        all_comparisons[f"{baseline_model}_vs_{research_model}"] = comparison_results

        # Print results
        print(f"\n📊 Comparison Results for {baseline_model} vs {research_model}:")
        for metric, result in comparison_results.items():
            print(f"  {metric}:")
            print(f"    {baseline_model}: {result['model_a_value']:.2f}")
            print(f"    {research_model}: {result['model_b_value']:.2f}")
            print(f"    Difference: {result['difference']:.2f} ({result['percent_difference']:.1f}%)")
            better = "better" if result['model_b_better'] else "worse"
            print(f"    {research_model} is {better} than {baseline_model}")

        print("\n⚠️ Note: Statistical tests require multiple runs with different seeds for valid results.")
        print("   Current comparison is based on single backtest runs only.")

    # Save all results
    output_path = project_root / "results" / "statistical_validation_v420_vs_research.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)

    print(f"\n💾 All results saved to: {output_path}")

    # Summary
    print("\n📋 SUMMARY:")
    for comparison_name, results in all_comparisons.items():
        model_b = comparison_name.split('_vs_')[1]
        better_metrics = sum(1 for r in results.values() if r['model_b_better'])
        total_metrics = len(results)
        print(f"  {comparison_name}: Research model {model_b} is better in {better_metrics}/{total_metrics} metrics compared to production v420")

if __name__ == "__main__":
    main()