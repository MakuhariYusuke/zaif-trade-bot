"""
ハイパーパラメータ最適化手法の比較実験

Grid Search, Random Search, Bayesian Optimization, Binary Searchを
同じ条件で比較し、それぞれの性能を評価します。
"""

from pathlib import Path
import json
import time
from typing import Dict, List

from ztb.optimization.base import OptimizationResult
from ztb.optimization.methods.grid_search import GridSearchOptimizer
from ztb.optimization.methods.random_search import RandomSearchOptimizer
from ztb.optimization.methods.bayesian_optimization import BayesianOptimizer
from ztb.optimization.methods.binary_search import BinarySearchOptimizer
from ztb.optimization.sac_utils import (
    get_sac_parameter_spaces,
    create_mock_objective_function
)


def compare_optimization_methods(
    preset: str = 'essential',
    n_trials_per_method: int = 20,
    output_dir: Path = Path('ztb/optimization/results')
):
    """
    複数の最適化手法を比較
    
    Args:
        preset: パラメータプリセット ('essential', 'learning', 'buffer')
        n_trials_per_method: 各手法の試行回数
        output_dir: 結果の保存先ディレクトリ
    """
    print("=" * 80)
    print("  ハイパーパラメータ最適化手法の比較実験")
    print("=" * 80)
    print()
    print(f"パラメータプリセット: {preset}")
    print(f"試行回数: {n_trials_per_method}")
    print()
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # パラメータ空間を取得
    param_spaces_dict = get_sac_parameter_spaces(preset)
    param_spaces = list(param_spaces_dict.values())
    
    print(f"探索パラメータ: {list(param_spaces_dict.keys())}")
    print()
    
    # モック目的関数を作成（テスト用）
    objective_func = create_mock_objective_function(noise_level=0.1)
    
    results = {}
    
    # 1. Random Search
    print("\n" + "=" * 80)
    print("1️⃣  Random Search")
    print("=" * 80 + "\n")
    
    random_search = RandomSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective_func,
        n_trials=n_trials_per_method,
        random_state=42
    )
    
    result_random = random_search.optimize()
    result_random.save(output_dir / 'random_search_result.json')
    results['random_search'] = result_random
    
    # 2. Grid Search
    print("\n" + "=" * 80)
    print("2️⃣  Grid Search")
    print("=" * 80 + "\n")
    
    # Grid Searchは組み合わせ爆発を避けるため、各パラメータ3-4点に制限
    grid_resolution = {}
    for param_name, param_space in param_spaces_dict.items():
        if param_space.param_type.value == 'categorical':
            grid_resolution[param_name] = param_space.choices[:3]  # 最大3つ
        elif param_space.param_type.value in ['continuous', 'log_uniform']:
            # 3点のみ
            import numpy as np
            if param_space.param_type.value == 'log_uniform':
                log_values = np.linspace(np.log10(param_space.low), 
                                        np.log10(param_space.high), 3)
                grid_resolution[param_name] = [10 ** x for x in log_values]
            else:
                grid_resolution[param_name] = list(
                    np.linspace(param_space.low, param_space.high, 3)
                )
    
    grid_search = GridSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective_func,
        grid_resolution=grid_resolution,
        random_state=42
    )
    
    result_grid = grid_search.optimize()
    result_grid.save(output_dir / 'grid_search_result.json')
    results['grid_search'] = result_grid
    
    # 3. Bayesian Optimization（オプション: scikit-optimizeが必要）
    print("\n" + "=" * 80)
    print("3️⃣  Bayesian Optimization")
    print("=" * 80 + "\n")
    
    try:
        bayesian_opt = BayesianOptimizer(
            parameter_spaces=param_spaces,
            objective_function=objective_func,
            n_trials=n_trials_per_method,
            n_initial_points=5,
            random_state=42
        )
        
        result_bayesian = bayesian_opt.optimize()
        result_bayesian.save(output_dir / 'bayesian_optimization_result.json')
        results['bayesian_optimization'] = result_bayesian
        
    except ImportError as e:
        print(f"⚠️  Bayesian Optimizationをスキップ: {e}")
        print("   インストール: pip install scikit-optimize")
        results['bayesian_optimization'] = None
    
    # 4. Binary Search（learning_rateのみ）
    if 'learning_rate' in param_spaces_dict:
        print("\n" + "=" * 80)
        print("4️⃣  Binary Search (learning_rate のみ)")
        print("=" * 80 + "\n")
        
        binary_search = BinarySearchOptimizer(
            parameter_space=param_spaces_dict['learning_rate'],
            objective_function=lambda params: objective_func(
                {**{k: v.default for k, v in param_spaces_dict.items() if k != 'learning_rate'},
                 **params}
            ),
            tolerance=1e-5,
            max_iterations=15,
            random_state=42
        )
        
        result_binary = binary_search.optimize()
        result_binary.save(output_dir / 'binary_search_result.json')
        results['binary_search'] = result_binary
    
    # 結果の比較
    print("\n" + "=" * 80)
    print("  📊 結果の比較")
    print("=" * 80 + "\n")
    
    comparison = []
    
    for method_name, result in results.items():
        if result is None:
            continue
        
        comparison.append({
            'method': method_name,
            'best_objective': result.best_objective_value,
            'n_trials': result.n_trials,
            'success_rate': result.success_rate,
            'duration_seconds': result.total_duration_seconds,
            'avg_time_per_trial': result.total_duration_seconds / result.n_trials if result.n_trials > 0 else 0,
            'best_params': result.best_parameters
        })
    
    # ベストオブジェクティブでソート
    comparison.sort(key=lambda x: x['best_objective'])
    
    # 表形式で表示
    print(f"{'手法':<25} {'ベスト目的値':>15} {'試行回数':>10} {'成功率':>10} {'総時間(秒)':>12} {'試行当たり(秒)':>15}")
    print("-" * 100)
    
    for item in comparison:
        print(f"{item['method']:<25} "
              f"{item['best_objective']:>15.6f} "
              f"{item['n_trials']:>10} "
              f"{item['success_rate']:>9.1%} "
              f"{item['duration_seconds']:>12.1f} "
              f"{item['avg_time_per_trial']:>15.3f}")
    
    print()
    
    # ベストパラメータの比較
    print("\n🏆 各手法のベストパラメータ:")
    print("-" * 80)
    for item in comparison:
        print(f"\n【{item['method']}】")
        for param, value in item['best_params'].items():
            print(f"  {param}: {value}")
    
    # 比較結果を保存
    comparison_path = output_dir / 'comparison_summary.json'
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 比較結果を保存: {comparison_path}")
    
    return results, comparison


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ハイパーパラメータ最適化手法の比較')
    parser.add_argument('--preset', type=str, default='essential',
                       choices=['essential', 'learning', 'buffer', 'full'],
                       help='パラメータプリセット')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='各手法の試行回数')
    parser.add_argument('--output-dir', type=str, 
                       default='ztb/optimization/results',
                       help='結果の保存先')
    
    args = parser.parse_args()
    
    results, comparison = compare_optimization_methods(
        preset=args.preset,
        n_trials_per_method=args.n_trials,
        output_dir=Path(args.output_dir)
    )
    
    print("\n✅ 比較実験が完了しました！")


if __name__ == '__main__':
    main()
