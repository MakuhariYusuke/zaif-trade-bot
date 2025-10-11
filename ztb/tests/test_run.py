# Test Run Script for Heavy Trading RL Project
# 重特徴量取引RLプロジェクトのテスト実行スクリプト

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ztb.utils.file_utils import safe_json_load
from ztb.utils.path_utils import ensure_dir, get_project_root
from ztb.utils.config import ZTBConfig

# Add project root to path for imports
sys.path.append(str(get_project_root()))

# ローカルモジュールのインポート
from ztb.training.config.ppo_config import get_ppo_config
from ztb.training.core.ppo_trainer import PPOTrainer
from ztb.utils import DiscordNotifier, LoggerManager
from ztb.utils.cli_common import CLIFormatter, CLIValidator, create_standard_parser


def load_config() -> dict:
    """テスト用設定ファイルの読み込み（固定）"""
    config_path = "config/training/test.json"
    config_file = Path(config_path)
    if config_file.exists():
        config = safe_json_load(config_file)
        # テスト実行時は必ず1000ステップに固定
        config["training"]["total_timesteps"] = 1000
        # experiment セクションを追加
        config["experiment"] = {
            "name": f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Test run with 1000 timesteps",
            "seed": 42,
        }
        print(f"Loaded test config from {config_path} (forced 1000 timesteps)")
    else:
        raise FileNotFoundError(f"Test config file not found: {config_path}")

    # 環境設定の読み込み
    env_config_path = Path("config/environment/dev.json")
    if env_config_path.exists():
        env_config = safe_json_load(env_config_path)
        config.update(env_config)
        print(f"Loaded environment config from {env_config_path}")

    return config


def get_default_config() -> dict:
    """デフォルト設定を取得"""
    return {
        "data": {
            "train_data": "./data/train_features.parquet",
            "test_data": "./data/test_features.parquet",
            "validation_data": "./data/val_features.parquet",
        },
        "training": {
            "total_timesteps": 200000,
            "eval_freq": 5000,
            "n_eval_episodes": 5,
            **get_ppo_config({
                "ent_coef": 0.01,  # Override for testing
            }),
        },
        "environment": {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
        },
        "optimization": {
            "n_trials": 100,
            "timeout": 3600,
            "metric": "mean_reward",
            "retrain_best": True,
            "full_timesteps": 200000,
        },
        "evaluation": {
            "n_episodes": 20,
            "max_steps_per_episode": 10000,
            "deterministic": True,
        },
        "paths": {
            "log_dir": "./logs/",
            "model_dir": str(ZTBConfig().get_model_dir()),
            "results_dir": "./results/",
            "opt_dir": "./optimization/",
            "tensorboard_log": "./tensorboard/",
            "checkpoint_dir": str(ZTBConfig().get_model_path("checkpoints")),
        },
        "experiment": {
            "name": f"heavy_trading_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Heavy feature trading RL with PPO and risk-adjusted rewards",
            "seed": 42,
        },
    }


def setup_directories(config: dict) -> None:
    """必要なディレクトリの作成"""
    paths = config["paths"]
    for path in paths.values():
        ensure_dir(Path(path))
    print("Directories setup complete")


def run_training_pipeline(
    config: dict,
    data_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """トレーニングパイプラインの実行"""
    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)

    # Discord通知: セッション開始
    config_name = "test_config"
    logger = LoggerManager(experiment_id=f"test_run_{config_name}")
    logger.log_experiment_start("test_run", config)

    # DiscordNotifierインスタンス作成（テストモード）
    notifier = DiscordNotifier(test_mode=True)

    try:
        # データパスの設定
        if data_path is None:
            data_path = config["data"]["train_data"]

        assert data_path is not None, "data_path must not be None"
        if not Path(data_path).exists():
            error_msg = f"Training data not found: {data_path}"
            print(f"Error: {error_msg}")
            logger.log_error(error_msg)
            return

        # トレーニング設定の更新
        train_config = config["training"].copy()
        train_config.update(config["paths"])
        train_config["seed"] = config["experiment"]["seed"]

        # PPOトレーニングの実行
        trainer = PPOTrainer(
            data_path,
            train_config,
            checkpoint_interval=1000,
            checkpoint_dir=config["paths"]["checkpoint_dir"],
        )

        # 設定ログ出力
        cache_config = config.get("memory", {})
        ckpt_config = config.get("training", {})
        print(
            f"CACHE: compressor={cache_config.get('compressor', 'auto')}(level={cache_config.get('compression_level', 3)}), dir={cache_config.get('cache_dir', 'data/cache')}, max={cache_config.get('cache_max_mb', 1000)}MB, ttl={cache_config.get('max_age_days', 7)}d, proc=pid_{os.getpid()}"
        )
        print(
            f"CKPT: light={ckpt_config.get('checkpoint_light', False)}, compressor={ckpt_config.get('checkpoint_compressor', 'auto')}, keep_last={cache_config.get('keep_ckpt', 5)}, interval=1000"
        )

        model = trainer.train(notifier=notifier)

        # 評価の実行
        print("\n" + "=" * 40)
        print("EVALUATION")
        print("=" * 40)

        eval_config = config["evaluation"].copy()
        eval_config["results_dir"] = config["paths"]["results_dir"]

        # evaluator = TradingEvaluator(
        #     str(Path(config['paths']['model_dir']) / 'best_model'),
        #     data_path,
        #     eval_config
        # )
        # stats = evaluator.evaluate_model()

        # 結果の表示
        print("\nTraining Results:")
        # print(f"Mean Reward: {stats['reward_stats']['mean_total_reward']:.4f}")
        # print(f"Mean PnL: {stats['pnl_stats']['mean_total_pnl']:.4f}")
        # print(f"Sharpe Ratio: {stats['pnl_stats']['sharpe_ratio']:.4f}")
        # print(f"Total Trades: {stats['trading_stats']['total_trades']}")

        # Discord通知: セッション終了（成功）
        logger.log_experiment_end({})
    except Exception as e:
        logging.exception(f"Training pipeline failed: {e}")
        error_msg = f"Training pipeline failed: {str(e)}"
        print(f"Error: {error_msg}")
        logger.log_error(error_msg, str(e))
        raise

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)


def run_optimization_pipeline(config: dict, data_path: Optional[str] = None) -> None:
    """最適化パイプラインの実行"""
    print("=" * 60)
    print("STARTING OPTIMIZATION PIPELINE")
    print("=" * 60)
    print(
        "Optimization pipeline is currently disabled (missing HyperparameterOptimizer)"
    )
    print("=" * 60)


def run_evaluation_pipeline(
    config: dict, model_path: str, data_path: Optional[str] = None
) -> None:
    """評価パイプラインの実行"""
    print("=" * 60)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 60)
    print("Evaluation pipeline is currently disabled (missing TradingEvaluator)")
    print("=" * 60)


def run_comparison_pipeline(
    config: dict,
    model_paths: List[str],
    model_names: Optional[List[str]] = None,
    data_path: Optional[str] = None,
) -> None:
    """モデル比較パイプラインの実行"""
    print("=" * 60)
    print("STARTING MODEL COMPARISON PIPELINE")
    print("=" * 60)
    print("Model comparison pipeline is currently disabled (missing TradingEvaluator)")
    print("=" * 60)


def save_experiment_config(config: dict) -> None:
    """実験設定の保存"""
    experiment_dir = Path(config["paths"]["log_dir"]) / config["experiment"]["name"]
    ensure_dir(experiment_dir)

    config_file = experiment_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Experiment config saved to {config_file}")


def main():
    """テスト実行メイン関数"""
    parser = create_standard_parser("Test Run Script")
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable feature caching"
    )
    parser.add_argument(
        "--checkpoint-light",
        action="store_true",
        help="Save only policy in checkpoints (faster, lighter)",
    )
    parser.add_argument(
        "--cache-compressor",
        type=str,
        choices=["auto", "zstd", "lz4", "zlib"],
        default="auto",
        help=CLIFormatter.format_help(
            "Cache compression algorithm", "auto", ["auto", "zstd", "lz4", "zlib"]
        ),
    )
    parser.add_argument(
        "--cache-access-pattern",
        type=str,
        choices=["frequent", "balanced", "archival"],
        default="balanced",
        help=CLIFormatter.format_help(
            "Cache access pattern hint",
            "balanced",
            ["frequent", "balanced", "archival"],
        ),
    )
    parser.add_argument(
        "--cache-max-mb",
        type=lambda x: CLIValidator.validate_positive_int(x, "cache-max-mb"),
        help="Maximum cache size in MB",
    )
    parser.add_argument(
        "--cache-ttl-days",
        type=lambda x: CLIValidator.validate_positive_int(x, "cache-ttl-days"),
        help="Cache TTL in days",
    )
    parser.add_argument(
        "--checkpoint-compressor",
        type=str,
        choices=["auto", "zstd", "lz4", "zlib"],
        default="auto",
        help="Checkpoint compression algorithm (default: auto)",
    )
    args = parser.parse_args()

    print("🧪 Starting Test Run (1000 timesteps)")

    # 設定の読み込み
    config = load_config()

    # --no-cacheオプションの処理
    if args.no_cache:
        if "memory" not in config:
            config["memory"] = {}
        config["memory"]["enable_cache"] = False
        print("Cache disabled via --no-cache option")

    # --checkpoint-lightオプションの処理
    if args.checkpoint_light:
        if "training" not in config:
            config["training"] = {}
        config["training"]["checkpoint_light"] = True
        print("Checkpoint light mode enabled via --checkpoint-light option")

    # キャッシュ設定の処理
    if "memory" not in config:
        config["memory"] = {}

    if args.cache_compressor != "auto":
        config["memory"]["compressor"] = args.cache_compressor
    if args.cache_access_pattern != "balanced":
        config["memory"]["access_pattern"] = args.cache_access_pattern
    if args.cache_max_mb:
        config["memory"]["cache_max_mb"] = args.cache_max_mb
    if args.cache_ttl_days:
        config["memory"]["max_age_days"] = args.cache_ttl_days

    # チェックポイント設定の処理
    if "training" not in config:
        config["training"] = {}

    if args.checkpoint_compressor != "auto":
        config["training"]["checkpoint_compressor"] = args.checkpoint_compressor

    # テスト実行用のパス設定
    config["paths"]["log_dir"] = "../logs/test/"
    config["paths"]["model_dir"] = str(ZTBConfig().get_model_path("test"))
    config["paths"]["results_dir"] = "../results/test/"
    config["paths"]["checkpoint_dir"] = str(ZTBConfig().get_model_path("test/checkpoints"))

    # ディレクトリのセットアップ
    setup_directories(config)

    # 実験設定の保存
    save_experiment_config(config)

    # トレーニング実行
    run_training_pipeline(config, None, None)


if __name__ == "__main__":
    main()
