# Main script for Heavy Trading RL Project
# 重特徴量取引RLプロジェクトのメインスクリプト

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent))
from train_ppo import PPOTrainer
from evaluate_model import TradingEvaluator
from optimize_params import HyperparameterOptimizer
from discord_notifications import notify_session_start, notify_session_end, notify_error


def load_config(config_path: str = 'rl_config.json') -> dict:
    """設定ファイルの読み込み"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
        print(f"Using default config (config file not found: {config_path})")

    return config


def get_default_config() -> dict:
    """デフォルト設定を取得"""
    return {
        'data': {
            'train_data': './data/train_features.parquet',
            'test_data': './data/test_features.parquet',
            'validation_data': './data/val_features.parquet',
        },
        'training': {
            'total_timesteps': 200000,
            'eval_freq': 5000,
            'n_eval_episodes': 5,
            'batch_size': 64,
            'n_steps': 2048,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
        },
        'environment': {
            'reward_scaling': 1.0,
            'transaction_cost': 0.001,
            'max_position_size': 1.0,
            'risk_free_rate': 0.0,
        },
        'optimization': {
            'n_trials': 100,
            'timeout': 3600,
            'metric': 'mean_reward',
            'retrain_best': True,
            'full_timesteps': 200000,
        },
        'evaluation': {
            'n_episodes': 20,
            'max_steps_per_episode': 10000,
            'deterministic': True,
        },
        'paths': {
            'log_dir': './logs/',
            'model_dir': './models/',
            'results_dir': './results/',
            'opt_dir': './optimization/',
            'tensorboard_log': './tensorboard/',
        },
        'experiment': {
            'name': f'heavy_trading_rl_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'description': 'Heavy feature trading RL with PPO and risk-adjusted rewards',
            'seed': 42,
        }
    }


def setup_directories(config: dict) -> None:
    """必要なディレクトリの作成"""
    paths = config['paths']
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("Directories setup complete")


def run_training_pipeline(config: dict, data_path: Optional[str] = None) -> None:
    """トレーニングパイプラインの実行"""
    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)

    # Discord通知: セッション開始
    config_name = Path(config.get('__file__', 'rl_config.json')).stem
    session_id = notify_session_start("training", config_name)

    try:
        # データパスの設定
        if data_path is None:
            data_path = config['data']['train_data']

        assert data_path is not None, "data_path must not be None"
        if not Path(data_path).exists():
            error_msg = f"Training data not found: {data_path}"
            print(f"Error: {error_msg}")
            notify_error(error_msg)
            return

        # トレーニング設定の更新
        train_config = config['training'].copy()
        train_config.update(config['paths'])
        train_config['seed'] = config['experiment']['seed']

        # PPOトレーニングの実行
        trainer = PPOTrainer(data_path, train_config)
        model = trainer.train()

        # 評価の実行
        print("\n" + "=" * 40)
        print("EVALUATION")
        print("=" * 40)

        eval_config = config['evaluation'].copy()
        eval_config['results_dir'] = config['paths']['results_dir']

        evaluator = TradingEvaluator(
            str(Path(config['paths']['model_dir']) / 'best_model'),
            data_path,
            eval_config
        )
        stats = evaluator.evaluate_model()

        # 結果の表示
        print("\nTraining Results:")
        print(f"Mean Reward: {stats['reward_stats']['mean_total_reward']:.4f}")
        print(f"Mean PnL: {stats['pnl_stats']['mean_total_pnl']:.4f}")
        print(f"Sharpe Ratio: {stats['pnl_stats']['sharpe_ratio']:.4f}")
        print(f"Total Trades: {stats['trading_stats']['total_trades']}")

        # Discord通知: セッション終了（成功）
        notify_session_end(stats, "training")

    except Exception as e:
        error_msg = f"Training pipeline failed: {str(e)}"
        print(f"Error: {error_msg}")
        notify_error(error_msg, str(e))
        raise

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)


def run_optimization_pipeline(config: dict, data_path: Optional[str] = None) -> None:
    """最適化パイプラインの実行"""
    print("=" * 60)
    print("STARTING OPTIMIZATION PIPELINE")
    print("=" * 60)

    # データパスの設定
    if data_path is None:
        data_path = config['data']['train_data']

    assert data_path is not None, "data_path must not be None"
    if not Path(data_path).exists():
        print(f"Error: Training data not found: {data_path}")
        return

    # 最適化設定の更新
    opt_config = config['optimization'].copy()
    opt_config['opt_dir'] = config['paths']['opt_dir']

    # ハイパーパラメータ最適化の実行
    optimizer = HyperparameterOptimizer(data_path, opt_config)
    study = optimizer.optimize()

    # 結果の表示
    print("\nOptimization Results:")
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")

    # 可視化の作成
    optimizer.create_optimization_plots(study)

    # 分析の実行
    analysis = optimizer.analyze_optimization_results(study)
    print(f"\nOptimization Analysis:")
    print(f"Completed trials: {analysis.get('completed_trials', 0)}")
    print(f"Mean value: {analysis.get('mean_value', 0):.4f}")

    # 本番トレーニング（オプション）
    if opt_config['retrain_best']:
        print("\nRetraining best model...")
        model = optimizer.retrain_best_model(study, opt_config['full_timesteps'])

        # 最終評価
        print("\nFinal model evaluation...")
        eval_config = config['evaluation'].copy()
        eval_config['results_dir'] = config['paths']['results_dir']

        evaluator = TradingEvaluator(
            str(Path(config['paths']['opt_dir']) / 'best_model_final'),
            data_path,
            eval_config
        )
        stats = evaluator.evaluate_model()

        print("\nFinal Model Results:")
        print(f"Mean Reward: {stats['reward_stats']['mean_total_reward']:.4f}")
        print(f"Mean PnL: {stats['pnl_stats']['mean_total_pnl']:.4f}")
        print(f"Sharpe Ratio: {stats['pnl_stats']['sharpe_ratio']:.4f}")

    print("\n" + "=" * 60)
    print("OPTIMIZATION PIPELINE COMPLETE")
    print("=" * 60)


def run_evaluation_pipeline(config: dict, model_path: str, data_path: Optional[str] = None) -> None:
    """評価パイプラインの実行"""
    print("=" * 60)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 60)

    # Discord通知: セッション開始
    config_name = Path(config.get('__file__', 'rl_config.json')).stem
    session_id = notify_session_start("evaluation", config_name)

    try:
        # データパスの設定
        if data_path is None:
            data_path = config['data']['test_data']

        assert data_path is not None, "data_path must not be None"
        if not Path(data_path).exists():
            error_msg = f"Test data not found: {data_path}"
            print(f"Error: {error_msg}")
            notify_error(error_msg)
            return

        if not Path(model_path).exists():
            error_msg = f"Model not found: {model_path}"
            print(f"Error: {error_msg}")
            notify_error(error_msg)
            return

        # 評価の実行
        eval_config = config['evaluation'].copy()
        eval_config['results_dir'] = config['paths']['results_dir']

        evaluator = TradingEvaluator(model_path, data_path, eval_config)
        stats = evaluator.evaluate_model()

        # 可視化の作成
        try:
            evaluator.create_visualizations()
        except Exception as viz_error:
            print(f"Warning: Visualization failed: {viz_error}")

        # 結果の表示
        print("\nEvaluation Results:")
        print(f"Mean Reward: {stats['reward_stats']['mean_total_reward']:.4f}")
        print(f"Mean PnL: {stats['pnl_stats']['mean_total_pnl']:.4f}")
        print(f"Sharpe Ratio: {stats['pnl_stats']['sharpe_ratio']:.4f}")
        print(f"Total Trades: {stats['trading_stats']['total_trades']}")

        # Discord通知: セッション終了（成功）
        notify_session_end(stats, "evaluation")

    except Exception as e:
        error_msg = f"Evaluation pipeline failed: {str(e)}"
        print(f"Error: {error_msg}")
        notify_error(error_msg, str(e))
        raise
    print(f"Win Rate: {stats['reward_stats'].get('win_rate', 0):.2%}")

    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE COMPLETE")
    print("=" * 60)


def run_comparison_pipeline(config: dict, model_paths: List[str], model_names: Optional[List[str]] = None,
                          data_path: Optional[str] = None) -> None:
    """モデル比較パイプラインの実行"""
    print("=" * 60)
    print("STARTING MODEL COMPARISON PIPELINE")
    print("=" * 60)

    # データパスの設定
    if data_path is None:
        data_path = config['data']['test_data']

    assert data_path is not None, "data_path must not be None"
    if not Path(data_path).exists():
        print(f"Error: Test data not found: {data_path}")
        return

    # モデルパスの検証
    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"Error: Model not found: {model_path}")
            return

    # モデル名の設定
    if model_names is None:
        model_names = [f'Model_{i+1}' for i in range(len(model_paths))]

    # 比較の実行
    eval_config = config['evaluation'].copy()
    eval_config['results_dir'] = config['paths']['results_dir']

    evaluator = TradingEvaluator(model_paths[0], data_path, eval_config)
    evaluator.compare_models(model_paths, model_names)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON PIPELINE COMPLETE")
    print("=" * 60)


def save_experiment_config(config: dict) -> None:
    """実験設定の保存"""
    experiment_dir = Path(config['paths']['log_dir']) / config['experiment']['name']
    experiment_dir.mkdir(exist_ok=True)

    config_file = experiment_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Experiment config saved to {config_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Heavy Trading RL Main Script')
    parser.add_argument('--config', type=str, default='rl_config.json',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'optimize', 'evaluate', 'compare'],
                       default='train', help='Pipeline mode')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--model', type=str, help='Path to model file (for evaluation)')
    parser.add_argument('--models', nargs='+', help='Paths to models for comparison')
    parser.add_argument('--model-names', nargs='+', help='Names for compared models')

    args = parser.parse_args()

    # 設定の読み込み
    config = load_config(args.config)

    # ディレクトリのセットアップ
    setup_directories(config)

    # 実験設定の保存
    save_experiment_config(config)

    # パイプラインの実行
    if args.mode == 'train':
        run_training_pipeline(config, args.data)

    elif args.mode == 'optimize':
        run_optimization_pipeline(config, args.data)

    elif args.mode == 'evaluate':
        if not args.model:
            print("Error: --model required for evaluation mode")
            return
        run_evaluation_pipeline(config, args.model, args.data)

    elif args.mode == 'compare':
        if not args.models:
            print("Error: --models required for comparison mode")
            return
        run_comparison_pipeline(config, args.models, args.model_names, args.data)


if __name__ == '__main__':
    main()