"""
SAC v395i 軽量版ハイパーパラメータ最適化

実際の訓練で最適化を実行（軽量版）:
- Random Search: 8試行
- 各試行: 3000ステップ（約10-15分）
- Total: 約2時間
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.optimization.methods.random_search import RandomSearchOptimizer
from ztb.optimization.base import ParameterSpace, ParameterType
from ztb.utils.path_utils import get_project_root, ensure_dir


def create_sac_objective_lightweight(base_config_path: Path, output_dir: Path, trial_steps: int = 3000):
    """
    軽量版目的関数
    
    各パラメータセットで短期訓練を実行し、Critic Lossを評価
    """
    ensure_dir(output_dir)
    
    # ベース設定読み込み
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    def objective(params: Dict[str, Any]) -> float:
        """パラメータを評価"""
        print(f"\n{'='*80}")
        print(f"試行: {params}")
        print(f"{'='*80}")
        
        # 設定ファイル作成
        trial_config = base_config.copy()
        
        # パラメータ適用（sac_hyperparameters/sac_paramsキーを使用）
        sac_key = 'sac_hyperparameters'
        if sac_key not in trial_config and 'sac_params' in trial_config:
            sac_key = 'sac_params'
        if sac_key not in trial_config:
            trial_config[sac_key] = {}

        for key, value in params.items():
            trial_config[sac_key][key] = value
        
        # 最適化用の設定調整
        trial_config['total_timesteps'] = trial_steps
        trial_config[sac_key]['learning_starts'] = 100  # 早期にメトリクス記録開始
        trial_config[sac_key]['buffer_size'] = max(5000, trial_steps)  # バッファサイズを調整
        
        # 一時設定ファイル保存
        trial_id = f"trial_{int(time.time()*1000)}"
        config_path = output_dir / f"{trial_id}_config.json"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(trial_config, f, indent=2, ensure_ascii=False)
        
        # 訓練実行
        print(f"訓練開始: {trial_steps}ステップ...")
        start_time = time.time()
        
        try:
            # 環境変数を設定（matplotlibエラー回避）
            import os
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            
            result = subprocess.run(
                [
                    'python',
                    'scripts/optimization/train_with_config.py',
                    '--config', str(config_path)
                ],
                capture_output=True,
                text=True,
                timeout=2400,  # 40分タイムアウト
                cwd=str(get_project_root()),
                env=env  # 環境変数を渡す
            )
            
            duration = time.time() - start_time
            print(f"訓練完了: {duration/60:.1f}分")
            
            # TensorBoardログディレクトリを特定
            # session_idがあれば使用、なければsac_sessionをデフォルトとする
            session_id = trial_config.get('session_id', 'sac_session')
            tensorboard_log_dir = get_project_root() / 'checkpoints' / session_id
            
            # Critic Loss抽出（stdout優先、失敗時はTensorBoard）
            critic_loss = extract_critic_loss(result.stdout, tensorboard_log_dir)
            
            print(f"Critic Loss: {critic_loss:.6f}")
            
            # 結果保存（stderrも含める）
            result_data = {
                'trial_id': trial_id,
                'parameters': params,
                'critic_loss': critic_loss,
                'duration_minutes': duration / 60,
                'stdout_tail': result.stdout[-3000:] if result.stdout else "",  # 最後の3000文字
                'stderr_tail': result.stderr[-3000:] if result.stderr else "",  # エラー出力も保存
                'returncode': result.returncode
            }
            
            result_path = output_dir / f"{trial_id}_result.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            return critic_loss
            
        except subprocess.TimeoutExpired:
            print("⚠️ タイムアウト（40分超過）")
            return 1e6
        except Exception as e:
            print(f"❌ エラー: {e}")
            return 1e6
    
    return objective


def extract_critic_loss(stdout: str, tensorboard_log_dir: Path = None) -> float:
    """
    標準出力またはTensorBoardから最終Critic Lossを抽出
    
    Args:
        stdout: 訓練の標準出力
        tensorboard_log_dir: TensorBoardログディレクトリ（オプション）
    
    Returns:
        Critic Loss（抽出失敗時は1e6）
    """
    # 1. まず標準出力から抽出を試みる
    lines = stdout.split('\n')
    
    # "Final Critic Loss:" 形式を検索
    for line in reversed(lines[-100:]):
        if 'final critic loss' in line.lower():
            try:
                # "Final Critic Loss: 0.123456" 形式
                parts = line.split(':')
                if len(parts) >= 2:
                    loss_str = parts[-1].strip()
                    return float(loss_str)
            except (ValueError, IndexError):
                continue
    
    # "critic_loss=" 形式を検索
    for line in reversed(lines[-200:]):
        if 'critic_loss=' in line.lower():
            try:
                # "critic_loss=0.123456" 形式
                for sep in ['=']:
                    if sep in line:
                        parts = line.split(sep)
                        if len(parts) >= 2:
                            loss_str = parts[-1].strip()
                            # カンマや括弧、パイプを除去
                            loss_str = loss_str.split(',')[0].split(')')[0].split('|')[0].split()[0]
                            return float(loss_str)
            except (ValueError, IndexError):
                continue
    
    # 2. 標準出力から抽出できない場合、TensorBoardから読み取る
    if tensorboard_log_dir and tensorboard_log_dir.exists():
        try:
            # extract_tensorboard_loss.pyをインポート
            import sys
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            
            from extract_tensorboard_loss import extract_critic_loss_from_tensorboard
            
            critic_loss = extract_critic_loss_from_tensorboard(tensorboard_log_dir, verbose=False)
            if critic_loss is not None:
                print(f"✅ TensorBoardからCritic Loss抽出: {critic_loss:.6f}")
                return critic_loss
        except Exception as e:
            print(f"⚠️ TensorBoard読み取りエラー: {e}")
    
    print("⚠️ Critic Lossを抽出できませんでした（stdout, TensorBoard両方失敗）")
    return 1e6


def main():
    """メイン実行"""
    print("="*80)
    print("  SAC v395i 軽量版ハイパーパラメータ最適化")
    print("  - Random Search: 8試行")
    print("  - 各試行: 3000ステップ（約10-15分）")
    print("  - 推定所要時間: 約1.5-2時間")
    print("="*80)
    
    # 設定
    root = get_project_root()
    base_config = root / 'config' / 'sac_v395i_complete_fix.json'
    output_dir = root / 'ztb' / 'optimization' / 'results' / f'sac_opt_{int(time.time())}'
    
    if not base_config.exists():
        print(f"❌ エラー: ベース設定ファイルが見つかりません: {base_config}")
        print("\n利用可能な設定ファイル:")
        configs_dir = root / 'configs'
        if configs_dir.exists():
            for f in configs_dir.glob('sac*.json'):
                print(f"  - {f.name}")
        return
    
    # 確認
    print(f"\nベース設定: {base_config.name}")
    print(f"出力先: {output_dir}")
    
    confirm = input("\n最適化を開始しますか？ (yes/no): ")
    if confirm.lower() != 'yes':
        print("キャンセルしました。")
        return
    
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
    
    objective = create_sac_objective_lightweight(base_config, output_dir, trial_steps=3000)
    
    # Random Search実行
    print("\n" + "="*80)
    print("Random Search開始")
    print("="*80)
    
    start_time = time.time()
    
    optimizer = RandomSearchOptimizer(
        parameter_spaces=param_spaces,
        objective_function=objective,
        n_trials=8
    )
    
    result = optimizer.optimize()
    
    total_duration = time.time() - start_time
    
    # 結果保存
    result.save(output_dir / 'optimization_result.json')
    
    # レポート
    print("\n" + "="*80)
    print("  最適化完了レポート")
    print("="*80)
    print(f"\n総所要時間: {total_duration/3600:.2f}時間")
    print(f"結果ディレクトリ: {output_dir}")
    print(f"\nベストCritic Loss: {result.best_objective_value:.6f}")
    print(f"v395iベースライン: 0.0918")
    
    if result.best_objective_value < 0.0918:
        improvement = (0.0918 - result.best_objective_value) / 0.0918 * 100
        print(f"✅ 改善: {improvement:.2f}%")
    else:
        degradation = (result.best_objective_value - 0.0918) / 0.0918 * 100
        print(f"⚠️ 悪化: {degradation:.2f}%")
    
    print(f"\n🎯 ベストパラメータ:")
    for key, value in result.best_parameters.items():
        print(f"  {key}: {value}")
    
    # v396設定ファイル生成
    with open(base_config, 'r', encoding='utf-8') as f:
        v396_config = json.load(f)
    
    if 'sac_params' not in v396_config:
        v396_config['sac_params'] = {}
    
    for key, value in result.best_parameters.items():
        v396_config['sac_params'][key] = value
    
    v396_config_path = root / 'configs' / 'sac_v396_optimized.json'
    with open(v396_config_path, 'w', encoding='utf-8') as f:
        json.dump(v396_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ v396設定ファイル保存: {v396_config_path.name}")
    print("\n次のステップ:")
    print(f"  python -m ztb.training.train --config configs\\sac_v396_optimized.json")
    print("="*80)


if __name__ == '__main__':
    main()
