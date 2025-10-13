"""
Statistical Validation Script - T-test for Model Performance

Compares the performance of two models using t-test to determine
if the improvement is statistically significant.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def load_backtest_results(result_path: Path) -> Dict:
    """Load backtest results from JSON file"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_portfolio_values(results: Dict) -> List[float]:
    """Extract portfolio values from backtest results"""
    # Assuming results contain portfolio history
    # This would need to be adjusted based on actual result format
    if 'portfolio_history' in results:
        return results['portfolio_history']
    elif 'final_portfolio' in results:
        # If only final value is available, return it
        return [results['final_portfolio']]
    else:
        raise ValueError("No portfolio data found in results")

def perform_ttest(model_a_results: List[Dict], model_b_results: List[Dict],
                 alpha: float = 0.05) -> Dict:
    """
    Perform t-test comparison between two models

    Args:
        model_a_results: List of backtest results for model A
        model_b_results: List of backtest results for model B
        alpha: Significance level

    Returns:
        Dictionary with t-test results
    """

    # Extract final portfolio values
    model_a_values = [r.get('final_portfolio', 200000) for r in model_a_results]
    model_b_values = [r.get('final_portfolio', 200000) for r in model_b_results]

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(model_a_values, model_b_values)

    # Calculate effect size (Cohen's d)
    mean_a = np.mean(model_a_values)
    mean_b = np.mean(model_b_values)
    std_a = np.std(model_a_values, ddof=1)
    std_b = np.std(model_b_values, ddof=1)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'model_a': {
            'mean': mean_a,
            'std': std_a,
            'n': len(model_a_values),
            'values': model_a_values
        },
        'model_b': {
            'mean': mean_b,
            'std': std_b,
            'n': len(model_b_values),
            'values': model_b_values
        },
        'effect_size': cohens_d,
        'interpretation': interpret_results(p_value, cohens_d, alpha)
    }

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
    print("Statistical Validation - T-test for Model Performance")
    print("=" * 80)

    # For demonstration, we'll compare the current v396 result with a baseline
    # In practice, you'd want multiple runs of each model

    # Load current v396 result
    v396_result_path = project_root / "docs" / "evaluation" / "backtest_sac_v397i_hold_reduction_results.json"
    if v396_result_path.exists():
        v396_result = load_backtest_results(v396_result_path)
        print(f"✅ Loaded v396 results: {v396_result_path}")
    else:
        print(f"❌ v396 results not found: {v396_result_path}")
        return

    # For baseline, we'll use a hypothetical previous model performance
    # In real scenario, you'd load actual baseline results
    baseline_results = [{
        'final_portfolio': 2500,  # From previous v397i run
        'total_return': -0.9875
    }]

    v396_results = [v396_result]

    # Perform t-test
    print("\n🔬 Performing statistical comparison...")
    results = perform_ttest(baseline_results, v396_results)

    print("\n📊 T-test Results:")
    print(f"  Model A (Baseline): μ={results['model_a']['mean']:,.0f}, σ={results['model_a']['std']:,.0f}, n={results['model_a']['n']}")
    print(f"  Model B (v396): μ={results['model_b']['mean']:,.0f}, σ={results['model_b']['std']:,.0f}, n={results['model_b']['n']}")
    print(f"  t-statistic: {results['t_statistic']:.4f}")
    print(f"  p-value: {results['p_value']:.4f}")
    print(f"  Significant (α={results['alpha']}): {results['significant']}")
    print(f"  Effect size (Cohen's d): {results['effect_size']:.3f}")
    print(f"  Interpretation: {results['interpretation']}")

    # Save results (convert numpy types to Python types)
    output_path = project_root / "docs" / "evaluation" / "statistical_validation_results.json"
    results_serializable = json.loads(json.dumps(results, default=str))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_path}")

    # Summary
    if results['significant'] and results['model_b']['mean'] > results['model_a']['mean']:
        print("\n✅ CONCLUSION: v396 shows statistically significant improvement!")
    elif results['significant'] and results['model_b']['mean'] < results['model_a']['mean']:
        print("\n❌ CONCLUSION: v396 shows statistically significant degradation!")
    else:
        print("\n⚠️ CONCLUSION: No statistically significant difference detected.")

if __name__ == "__main__":
    main()