"""
SAC v395i ハイパーパラメータ最適化実行スクリプト

既存の最適化フレームワークを使用して、SAC v395iの性能をさらに向上させる
最適なハイパーパラメータを探索します。

戦略:
1. Random Search: 広範囲探索（15-20試行）
2. Binary Search: 微調整（learning_rateを集中的に）
3. 段階的実行: 各試行5kステップで高速検証
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import time

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.optimization.methods.random_search import RandomSearchOptimizer
from ztb.optimization.methods.binary_search import BinarySearchOptimizer
from ztb.optimization.base import ParameterSpace, ParameterType
from ztb.utils.path_utils import get_project_root, ensure_dir


class SACOptimizer:
    """SAC v395i用の最適化実行クラス"""
    
    def __init__(self, base_config_path: Path, output_dir: Path):
        """
        初期化
        
        Args:
            base_config_path: ベース設定ファイル（v395i）
            output_dir: 結果出力ディレクトリ
        """
        self.base_config = self._load_config(base_config_path)
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        # v395iの現在の成果
        self.baseline = {
            'critic_loss': 0.0918,
            'actor_loss': -5.61,
            'ent_coef': 0.598  # 平均
        }
    
    def _load_config(self, path: Path) -> Dict:
        """設定ファイル読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_config(self, config: Dict, path: Path):
        """設定ファイル保存"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def create_parameter_spaces(self) -> List[ParameterSpace]:
        """
        最適化するパラメータ空間を定義
        
        v395iで良好だった範囲を中心に、さらなる改善を探る
        """
        return [
            # Learning Rate: v395iは3e-4、この周辺を探索
            ParameterSpace(
                name='learning_rate',
                param_type=ParameterType.LOG_UNIFORM,
                low=1e-4,
                high=1e-3
            ),
            # Batch Size: v395iは256、これも探索
            ParameterSpace(
                name='batch_size',
                param_type=ParameterType.CATEGORICAL,
                choices=[128, 256, 512]
            ),
            # Gamma: v395iは0.99、微調整
            ParameterSpace(
                name='gamma',
                param_type=ParameterType.CONTINUOUS,
                low=0.985,
                high=0.999
            ),
            # Target Update Interval: v395iは1、探索
            ParameterSpace(
                name='target_update_interval',
                param_type=ParameterType.INTEGER,
                low=1,
                high=5
            ),
        ]
    
    def create_objective_function(self, trial_steps: int = 5000):
        """
        目的関数を作成
        
        各パラメータセットで短期訓練を実行し、Critic Lossを評価
        
        Args:
            trial_steps: 各試行のステップ数（デフォルト5000）
        """
        def objective(params: Dict[str, Any]) -> float:
            """
            パラメータを評価
            
            Returns:
                評価値（Critic Loss、小さいほど良い）
            """
            print(f"\n{'='*80}")
            print(f"試行開始: {params}")
            print(f"{'='*80}")
            
            # 設定ファイル作成
            trial_config = self.base_config.copy()
            
            # パラメータ適用
            sac_key = 'sac_params'
            if sac_key not in trial_config and 'sac_hyperparameters' in trial_config:
                sac_key = 'sac_hyperparameters'
            if sac_key not in trial_config:
                trial_config[sac_key] = {}
            
            trial_config[sac_key]['learning_rate'] = params['learning_rate']
            trial_config[sac_key]['batch_size'] = params['batch_size']
            trial_config[sac_key]['gamma'] = params['gamma']
            trial_config[sac_key]['target_update_interval'] = params['target_update_interval']
            trial_config['total_timesteps'] = trial_steps
            
            # 一時設定ファイル保存
            trial_id = f"trial_{int(time.time())}"
            config_path = self.output_dir / f"{trial_id}_config.json"
            self._save_config(trial_config, config_path)
            
            # 訓練実行
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        '-m', 'ztb.training.train',
                        '--config', str(config_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30分タイムアウト
                )
                
                # 結果解析（簡易版：標準出力から最終Critic Lossを抽出）
                critic_loss = self._extract_final_critic_loss(result.stdout)
                
                print(f"\n結果: Critic Loss = {critic_loss:.6f}")
                
                # 結果保存
                result_data = {
                    'trial_id': trial_id,
                    'parameters': params,
                    'critic_loss': critic_loss,
                    'stdout': result.stdout[-2000:],  # 最後の2000文字のみ
                    'returncode': result.returncode
                }
                
                result_path = self.output_dir / f"{trial_id}_result.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                return critic_loss
                
            except subprocess.TimeoutExpired:
                print("タイムアウト！")
                return 1e6  # ペナルティ
            except Exception as e:
                print(f"エラー: {e}")
                return 1e6  # ペナルティ
        
        return objective
    
    def _extract_final_critic_loss(self, stdout: str) -> float:
        """
        標準出力から最終Critic Lossを抽出
        
        Args:
            stdout: 訓練スクリプトの標準出力
        
        Returns:
            Critic Loss（抽出失敗時は大きな値）
        """
        lines = stdout.split('\n')
        
        # 最後の方から "critic_loss" を含む行を探す
        for line in reversed(lines):
            if 'critic_loss' in line.lower():
                try:
                    # "critic_loss: 0.123" のようなフォーマットを想定
                    parts = line.split(':')
                    if len(parts) >= 2:
                        loss_str = parts[-1].strip()
                        # カンマや他の文字を除去
                        loss_str = loss_str.split(',')[0].split()[0]
                        return float(loss_str)
                except (ValueError, IndexError):
                    continue
        
        # 抽出失敗
        print("⚠️ Critic Lossを抽出できませんでした")
        return 1e6
    
    def run_random_search(self, n_trials: int = 15) -> Dict:
        """
        Random Searchを実行
        
        Args:
            n_trials: 試行回数
        
        Returns:
            最適化結果
        """
        print("\n" + "="*80)
        print("Phase 1: Random Search")
        print(f"試行回数: {n_trials}")
        print("="*80)
        
        param_spaces = self.create_parameter_spaces()
        objective = self.create_objective_function(trial_steps=5000)
        
        optimizer = RandomSearchOptimizer(
            parameter_spaces=param_spaces,
            objective_function=objective,
            n_trials=n_trials
        )
        
        result = optimizer.optimize()
        
        # 結果保存
        result.save(self.output_dir / 'random_search_result.json')
        
        print(f"\n{'='*80}")
        print("Random Search完了")
        print(f"{'='*80}")
        print(f"ベストパラメータ: {result.best_parameters}")
        print(f"ベストCritic Loss: {result.best_objective_value:.6f}")
        print(f"ベースライン比較: {self.baseline['critic_loss']:.6f}")
        
        if result.best_objective_value < self.baseline['critic_loss']:
            improvement = (self.baseline['critic_loss'] - result.best_objective_value) / self.baseline['critic_loss'] * 100
            print(f"✅ 改善: {improvement:.2f}%")
        
        return result
    
    def run_binary_search(self, center_value: float, param_name: str = 'learning_rate') -> Dict:
        """
        Binary Searchで微調整
        
        Args:
            center_value: 中心値（Random Searchの結果）
            param_name: 最適化するパラメータ名
        
        Returns:
            最適化結果
        """
        print("\n" + "="*80)
        print("Phase 2: Binary Search（微調整）")
        print(f"パラメータ: {param_name}")
        print(f"中心値: {center_value}")
        print("="*80)
        
        # 中心値の±30%の範囲で探索
        low = center_value * 0.7
        high = center_value * 1.3
        
        param_space = ParameterSpace(
            name=param_name,
            param_type=ParameterType.LOG_UNIFORM,
            low=low,
            high=high
        )
        
        objective = self.create_objective_function(trial_steps=5000)
        
        optimizer = BinarySearchOptimizer(
            parameter_space=param_space,
            objective_function=objective,
            max_iterations=10,
            tolerance=1e-5
        )
        
        result = optimizer.optimize()
        
        # 結果保存
        result.save(self.output_dir / 'binary_search_result.json')
        
        print(f"\n{'='*80}")
        print("Binary Search完了")
        print(f"{'='*80}")
        print(f"最適{param_name}: {result.best_parameters[param_name]:.6e}")
        print(f"Critic Loss: {result.best_objective_value:.6f}")
        
        return result


def main():
    """メイン実行"""
    print("="*80)
    print("  SAC v395i ハイパーパラメータ最適化")
    print("  - 高速検証: 各試行5000ステップ")
    print("  - 2段階最適化: Random → Binary Search")
    print("="*80)
    
    # 設定
    root = get_project_root()
    base_config = root / 'configs' / 'sac_v395i_complete_fix.json'
    output_dir = root / 'ztb' / 'optimization' / 'results' / f'sac_v395i_opt_{int(time.time())}'
    
    if not base_config.exists():
        print(f"エラー: ベース設定ファイルが見つかりません: {base_config}")
        return
    
    # 最適化実行
    optimizer = SACOptimizer(base_config, output_dir)
    
    # Phase 1: Random Search（15試行）
    random_result = optimizer.run_random_search(n_trials=15)
    
    # Phase 2: Binary Search（learning_rateを微調整）
    best_lr = random_result.best_parameters.get('learning_rate', 3e-4)
    binary_result = optimizer.run_binary_search(center_value=best_lr, param_name='learning_rate')
    
    # 最終レポート
    print("\n" + "="*80)
    print("  最適化完了レポート")
    print("="*80)
    print(f"\n結果ディレクトリ: {output_dir}")
    print(f"\nRandom Search:")
    print(f"  ベストCritic Loss: {random_result.best_objective_value:.6f}")
    print(f"  パラメータ: {random_result.best_parameters}")
    print(f"\nBinary Search (learning_rate):")
    print(f"  最適化後Critic Loss: {binary_result.best_objective_value:.6f}")
    print(f"  最適learning_rate: {binary_result.best_parameters['learning_rate']:.6e}")
    
    # v396用の推奨設定を生成
    recommended_params = random_result.best_parameters.copy()
    recommended_params['learning_rate'] = binary_result.best_parameters['learning_rate']
    
    print(f"\n🎯 v396用推奨パラメータ:")
    for key, value in recommended_params.items():
        print(f"  {key}: {value}")
    
    # v396設定ファイル生成
    v396_config = optimizer.base_config.copy()
    if 'sac_params' not in v396_config:
        v396_config['sac_params'] = {}
    
    for key, value in recommended_params.items():
        v396_config['sac_params'][key] = value
    
    v396_config_path = root / 'configs' / 'sac_v396_optimized.json'
    optimizer._save_config(v396_config, v396_config_path)
    
    print(f"\n✅ v396設定ファイル保存: {v396_config_path}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
