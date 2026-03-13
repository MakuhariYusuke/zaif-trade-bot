"""5000ステップで取引が実行されるかテスト"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    compute_balance_roi,
    extract_env_metrics,
    resolve_env,
    unwrap_env,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger("quick_trade_test")

def main():
    """5000ステップで取引テスト"""
    config = {
        "training": {
            "algorithm": "SAC",
            "total_timesteps": 5000,  # 短時間テスト
            "eval_freq": 1000,
            "n_eval_episodes": 1,
            "log_interval": 100,
            "seed": 42,
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "learning_starts": 100,  # 早めに学習開始
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
            },
            "data_config": {
                "data_path": "data/btc_jpy_1m_v451_optimized_features.parquet",
                "window_size": 60,
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,
            },
        },
        "model_name": "quick_trade_test",
        "output_dir": "temp/quick_test",
    }

    logger.info("="*60)
    logger.info("5000ステップ 取引テスト開始")
    logger.info("="*60)

    start_time = time.time()
    
    try:
        # トレーナー初期化
        trainer = SACTrainer(config=config, logger=logger)
        
        # トレーニング実行
        logger.info("トレーニング開始...")
        result = trainer.train()
        
        elapsed = time.time() - start_time
        logger.info(f"トレーニング完了: {elapsed:.1f}秒")
        logger.info(f"戻り値タイプ: {type(result)}")
        logger.info(f"戻り値: {result}")
        
        # トレーナーから環境にアクセス
        try:
            env = resolve_env(trainer)
            if env is None:
                logger.error("❌ 環境へのアクセスパスが見つかりません")
                return 1

            logger.info("\n環境情報:")
            logger.info(f"  型: {type(env)}")

            target_env = unwrap_env(env)
            if target_env is None:
                logger.error("❌ 環境のunwrapに失敗しました")
                return 1

            logger.info(f"  最終環境型: {type(target_env)}")

            metrics = extract_env_metrics(env, include_optional=True)
            roi = compute_balance_roi(metrics)

            if "final_balance" in metrics:
                logger.info(
                    f"  ✅ 現在のポートフォリオ: {metrics['final_balance']:.2f}"
                )
            else:
                logger.warning(
                    f"  ❌ final_balanceが見つかりません（環境属性: {dir(target_env)[:20]}...）"
                )

            if "initial_balance" in metrics:
                logger.info(f"  ✅ 初期資本: {metrics['initial_balance']:.2f}")
            else:
                logger.warning("  ❌ initial_balanceが見つかりません")

            if "total_trades" in metrics:
                logger.info(f"  ✅ 総取引回数: {metrics['total_trades']}")
            else:
                logger.warning("  ❌ total_trades属性が見つかりません")

            if "buy_count" in metrics:
                logger.info(f"  ✅ 買い取引: {metrics['buy_count']}")
            else:
                logger.warning("  ❌ buy_count属性が見つかりません")

            if "sell_count" in metrics:
                logger.info(f"  ✅ 売り取引: {metrics['sell_count']}")
            else:
                logger.warning("  ❌ sell_count属性が見つかりません")

            if roi is not None:
                logger.info(f"  ✅ ROI: {roi:.4f}%")
            else:
                logger.warning("  ❌ ROI計算に必要な属性が見つかりません")
        except Exception as env_error:
            logger.error(f"環境情報の取得中にエラー: {env_error}", exc_info=True)
        
        # モデルから情報取得
        try:
            if hasattr(trainer, 'model'):
                model = trainer.model
                logger.info("\nモデル情報:")
                logger.info(f"  型: {type(model)}")
                logger.info(f"  学習ステップ: {model.num_timesteps}")
        except Exception as model_error:
            logger.error(f"モデル情報の取得中にエラー: {model_error}", exc_info=True)
            
        logger.info("\n="*60)
        logger.info("✅ テスト完了")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ エラー: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
