"""
ハイパーパラメータ最適化のサンプルスクリプト

このスクリプトは、最適化フレームワークの使い方を示します。
モック目的関数を使用するため、実際の訓練なしで動作を確認できます。
"""

from pathlib import Path

from ztb.optimization.base import ParameterSpace, ParameterType
from ztb.optimization.methods.grid_search import GridSearchOptimizer
from ztb.optimization.methods.random_search import RandomSearchOptimizer
from ztb.optimization.methods.binary_search import BinarySearchOptimizer
from ztb.optimization.sac_utils import (
    get_sac_parameter_spaces,
    create_mock_objective_function
)


def example_1_random_search():
    """例1: Random Searchの基本的な使い方"""
    print("\n" + "=" * 80)
    print("例1: Random Search - SAC Learning Rate最適化")
    print("=" * 80 + "\n")
    
    # パラメータ空間を定義
    param_spaces = [
        ParameterSpace(
            name='learning_rate',
            param_type=ParameterType.LOG_UNIFORM,
            low=1e-5,
            high=1e-2,
            default=3e-4
        ),
        ParameterSpace(
            name='batch_size',
            param_type=ParameterType.CATEGORICAL,
            choices=[64, 128, 256],
            default=128
        )
    ]
    
    # モック目的関数（実際の訓練の代わり）
    objective_func = create_mock_objective_function(noise_level=0.05)
    
    # Random Search最適化
    optimizer = RandomSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective_func,
        n_trials=15,
        random_state=42
    )
    
    result = optimizer.optimize()
    
    # 結果を保存
    output_path = Path('ztb/optimization/results/example_1_random_search.json')
    result.save(output_path)
    
    return result


def example_2_grid_search():
    """例2: Grid Searchの使い方"""
    print("\n" + "=" * 80)
    print("例2: Grid Search - Learning RateとBatch Sizeの組み合わせ")
    print("=" * 80 + "\n")
    
    # SAC用のパラメータ空間（learning プリセット）
    param_spaces_dict = get_sac_parameter_spaces('learning')
    param_spaces = list(param_spaces_dict.values())
    
    # グリッド解像度を指定
    grid_resolution = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'learning_starts': [100, 500, 1000],
        'train_freq': [1, 2],
        'gradient_steps': [1, 2]
    }
    
    objective_func = create_mock_objective_function(noise_level=0.05)
    
    optimizer = GridSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective_func,
        grid_resolution=grid_resolution,
        random_state=42
    )
    
    result = optimizer.optimize()
    
    output_path = Path('ztb/optimization/results/example_2_grid_search.json')
    result.save(output_path)
    
    return result


def example_3_binary_search():
    """例3: Binary Searchで単一パラメータを最適化"""
    print("\n" + "=" * 80)
    print("例3: Binary Search - Learning Rate の精密最適化")
    print("=" * 80 + "\n")
    
    # Learning Rateのみ
    param_space = ParameterSpace(
        name='learning_rate',
        param_type=ParameterType.LOG_UNIFORM,
        low=1e-5,
        high=1e-2,
        default=3e-4
    )
    
    objective_func = create_mock_objective_function(noise_level=0.02)
    
    optimizer = BinarySearchOptimizer(
        parameter_space=param_space,
        objective_function=objective_func,
        tolerance=1e-6,
        max_iterations=15,
        random_state=42
    )
    
    result = optimizer.optimize()
    
    output_path = Path('ztb/optimization/results/example_3_binary_search.json')
    result.save(output_path)
    
    return result


def example_4_bayesian_optimization():
    """例4: Bayesian Optimizationの使い方"""
    print("\n" + "=" * 80)
    print("例4: Bayesian Optimization - 効率的探索")
    print("=" * 80 + "\n")
    
    try:
        from ztb.optimization.methods.bayesian_optimization import BayesianOptimizer
    except ImportError:
        print("⚠️  scikit-optimizeがインストールされていません。")
        print("   インストール: pip install scikit-optimize")
        return None
    
    # Essential パラメータ
    param_spaces = list(get_sac_parameter_spaces('essential').values())
    objective_func = create_mock_objective_function(noise_level=0.05)
    
    optimizer = BayesianOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective_func,
        n_trials=25,
        n_initial_points=8,
        acquisition_function='EI',
        random_state=42
    )
    
    result = optimizer.optimize()
    
    output_path = Path('ztb/optimization/results/example_4_bayesian.json')
    result.save(output_path)
    
    return result


def example_5_staged_optimization():
    """例5: 段階的最適化 - 実践的なアプローチ"""
    print("\n" + "=" * 80)
    print("例5: 段階的最適化戦略")
    print("=" * 80 + "\n")
    
    print("Stage 1: Random Searchで大まかに探索")
    print("-" * 80)
    
    # Stage 1: Random Searchで広く探索
    param_spaces = list(get_sac_parameter_spaces('essential').values())
    objective_func = create_mock_objective_function(noise_level=0.1)
    
    random_opt = RandomSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective_func,
        n_trials=20,
        random_state=42
    )
    result_stage1 = random_opt.optimize()
    
    print("\nStage 1 完了！")
    print(f"ベストパラメータ: {result_stage1.best_parameters}")
    print(f"ベスト目的値: {result_stage1.best_objective_value:.6f}")
    
    # Stage 2: Learning Rateを Binary Searchで微調整
    print("\n\nStage 2: Learning Rate を Binary Searchで微調整")
    print("-" * 80)
    
    best_lr_from_stage1 = result_stage1.best_parameters['learning_rate']
    
    # Stage 1の結果の周辺を探索
    lr_space = ParameterSpace(
        name='learning_rate',
        param_type=ParameterType.LOG_UNIFORM,
        low=best_lr_from_stage1 * 0.1,  # 10分の1から
        high=best_lr_from_stage1 * 10,   # 10倍まで
        default=best_lr_from_stage1
    )
    
    binary_opt = BinarySearchOptimizer(
        parameter_space=lr_space,
        objective_function=lambda params: objective_func({
            **result_stage1.best_parameters,
            **params
        }),
        tolerance=1e-6,
        max_iterations=12,
        random_state=42
    )
    result_stage2 = binary_opt.optimize()
    
    print("\nStage 2 完了！")
    print(f"最終Learning Rate: {result_stage2.best_parameters['learning_rate']:.6e}")
    print(f"最終目的値: {result_stage2.best_objective_value:.6f}")
    
    # 最終結果
    final_params = {
        **result_stage1.best_parameters,
        'learning_rate': result_stage2.best_parameters['learning_rate']
    }
    
    print("\n" + "=" * 80)
    print("  最終結果")
    print("=" * 80)
    print(f"\n最適パラメータ:")
    for param, value in final_params.items():
        print(f"  {param}: {value}")
    print(f"\n目的値改善: {result_stage1.best_objective_value:.6f} → {result_stage2.best_objective_value:.6f}")
    
    return final_params


def main():
    """全ての例を実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='最適化フレームワークのサンプル')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5], 
                       help='実行する例の番号（指定しない場合は全て実行）')
    
    args = parser.parse_args()
    
    examples = {
        1: ("Random Search", example_1_random_search),
        2: ("Grid Search", example_2_grid_search),
        3: ("Binary Search", example_3_binary_search),
        4: ("Bayesian Optimization", example_4_bayesian_optimization),
        5: ("段階的最適化", example_5_staged_optimization),
    }
    
    if args.example:
        # 特定の例のみ実行
        name, func = examples[args.example]
        print(f"\n実行: 例{args.example} - {name}")
        func()
    else:
        # 全て実行
        print("\n" + "=" * 80)
        print("  ハイパーパラメータ最適化 - 全サンプル実行")
        print("=" * 80)
        
        for num, (name, func) in examples.items():
            print(f"\n\n{'=' * 80}")
            print(f"  例{num}: {name}")
            print(f"{'=' * 80}")
            try:
                func()
            except Exception as e:
                print(f"❌ エラー: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("  全てのサンプル実行が完了しました！")
    print("=" * 80)
    print("\n結果は ztb/optimization/results/ に保存されています。")


if __name__ == '__main__':
    main()
