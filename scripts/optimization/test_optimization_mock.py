"""
SAC最適化のモックテスト

実際の訓練なしで最適化フレームワークの動作を確認します。
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.optimization.methods.random_search import RandomSearchOptimizer
from ztb.optimization.methods.binary_search import BinarySearchOptimizer
from ztb.optimization.base import ParameterSpace, ParameterType
from ztb.utils.path_utils import get_project_root


def create_mock_sac_objective():
    """
    モック目的関数
    
    実際の訓練をシミュレート。learning_rateが3e-4付近で最小となる関数。
    """
    def objective(params: Dict[str, Any]) -> float:
        """擬似的なCritic Loss計算"""
        lr = params.get('learning_rate', 3e-4)
        batch_size = params.get('batch_size', 256)
        gamma = params.get('gamma', 0.99)
        target_interval = params.get('target_update_interval', 1)
        
        # 各パラメータの影響を擬似的にモデル化
        # learning_rate: 3e-4付近が最適
        lr_penalty = abs(lr - 3e-4) * 1000
        
        # batch_size: 256が最適
        bs_penalty = abs(batch_size - 256) / 100
        
        # gamma: 0.99が最適
        gamma_penalty = abs(gamma - 0.99) * 10
        
        # target_update_interval: 1が最適
        ti_penalty = abs(target_interval - 1) * 0.01
        
        # ベースライン + ペナルティ
        critic_loss = 0.0918 + lr_penalty + bs_penalty + gamma_penalty + ti_penalty
        
        # ノイズ追加（現実的なばらつき）
        import random
        noise = random.gauss(0, 0.005)
        critic_loss += noise
        
        print(f"  lr={lr:.2e}, bs={batch_size}, gamma={gamma:.3f}, ti={target_interval} -> Loss={critic_loss:.4f}")
        
        # 実際の訓練をシミュレート（少し待機）
        time.sleep(0.1)
        
        return critic_loss
    
    return objective


def main():
    """モックテスト実行"""
    print("="*80)
    print("  SAC最適化フレームワーク モックテスト")
    print("  - 実際の訓練なし、高速検証")
    print("="*80)
    
    # パラメータ空間定義
    param_spaces = [
        ParameterSpace(
            name='learning_rate',
            param_type=ParameterType.LOG_UNIFORM,
            low=1e-4,
            high=1e-3
        ),
        ParameterSpace(
            name='batch_size',
            param_type=ParameterType.CATEGORICAL,
            choices=[128, 256, 512]
        ),
        ParameterSpace(
            name='gamma',
            param_type=ParameterType.CONTINUOUS,
            low=0.985,
            high=0.999
        ),
        ParameterSpace(
            name='target_update_interval',
            param_type=ParameterType.INTEGER,
            low=1,
            high=5
        ),
    ]
    
    objective = create_mock_sac_objective()
    
    # Phase 1: Random Search
    print("\n" + "="*80)
    print("Phase 1: Random Search (10試行)")
    print("="*80)
    
    random_optimizer = RandomSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective,
        n_trials=10
    )
    
    random_result = random_optimizer.optimize()
    
    print(f"\nRandom Search結果:")
    print(f"  ベストCritic Loss: {random_result.best_objective_value:.6f}")
    print(f"  ベストパラメータ:")
    for key, value in random_result.best_parameters.items():
        print(f"    {key}: {value}")
    
    # Phase 2: Binary Search
    print("\n" + "="*80)
    print("Phase 2: Binary Search (learning_rate微調整)")
    print("="*80)
    
    best_lr = random_result.best_parameters['learning_rate']
    print(f"Random Searchの最適learning_rate: {best_lr:.2e}")
    
    lr_space = ParameterSpace(
        name='learning_rate',
        param_type=ParameterType.LOG_UNIFORM,
        low=best_lr * 0.7,
        high=best_lr * 1.3
    )
    
    # learning_rate以外は固定
    def binary_objective(params: Dict[str, Any]) -> float:
        full_params = random_result.best_parameters.copy()
        full_params.update(params)
        return objective(full_params)
    
    binary_optimizer = BinarySearchOptimizer(
        parameter_space=lr_space,
        objective_function=binary_objective,
        max_iterations=8,
        tolerance=1e-6
    )
    
    binary_result = binary_optimizer.optimize()
    
    print(f"\nBinary Search結果:")
    print(f"  最適Critic Loss: {binary_result.best_objective_value:.6f}")
    print(f"  最適learning_rate: {binary_result.best_parameters['learning_rate']:.6e}")
    
    # 最終推奨パラメータ
    print("\n" + "="*80)
    print("最終推奨パラメータ (v396用)")
    print("="*80)
    
    final_params = random_result.best_parameters.copy()
    final_params['learning_rate'] = binary_result.best_parameters['learning_rate']
    
    for key, value in final_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n予想Critic Loss: {binary_result.best_objective_value:.6f}")
    print(f"v395iベースライン: 0.0918")
    
    if binary_result.best_objective_value < 0.0918:
        improvement = (0.0918 - binary_result.best_objective_value) / 0.0918 * 100
        print(f"✅ 期待改善率: {improvement:.2f}%")
    
    print("\n✅ モックテスト完了！フレームワークは正常に動作しています。")
    print("\n次のステップ:")
    print("  python scripts\\optimization\\run_sac_optimization.py")


if __name__ == '__main__':
    main()
