"""
Phase 4 Day 5: 単一実験テスト（5000ステップ）
メトリクス抽出の動作検証用
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    compute_balance_roi,
    extract_env_metrics,
    resolve_env,
    unwrap_env,
)
from ztb.utils.logging_utils import setup_logger

# ログ設定
logger = setup_logger("test_metrics_extraction", level=logging.INFO)

def create_test_config(data_path: str, seed: int = 42, timesteps: int = 5000) -> dict:
    """テスト用設定を作成"""
    return {
        "experiment_name": "test_metrics_extraction",
        "algorithm": "sac",
        "training": {
            "seed": seed,
            "total_timesteps": timesteps,
            "learning_starts": 100,
            "save_interval": 50000,  # 保存しない
            "data_config": {
                "data_path": data_path,
                "validation_split": 0.2,
            },
            "sac_hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 50000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
            },
        },
    }

def main():
    start_time = time.time()
    
    # 8特徴量データで実験
    data_path = "data/btc_jpy_1m_v451_optimized_features.parquet"
    config = create_test_config(data_path, seed=42, timesteps=5000)
    
    logger.info("="*80)
    logger.info("メトリクス抽出テスト開始")
    logger.info(f"データ: {data_path}")
    logger.info(f"ステップ数: {config['training']['total_timesteps']}")
    logger.info("="*80)
    
    try:
        # トレーナー作成
        logger.info("\nSACTrainer作成中...")
        trainer = SACTrainer(config=config, logger=logger)
        
        # トレーニング実行
        logger.info("\nトレーニング開始...")
        result = trainer.train()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nトレーニング完了 ({elapsed_time:.1f}秒)")
        logger.info(f"戻り値: {result} (型: {type(result)})")
        
        # 環境からメトリクス抽出
        logger.info("\n" + "="*80)
        logger.info("環境からメトリクス抽出中...")
        logger.info("="*80)
        
        metrics = {}
        env = resolve_env(trainer)

        if env is None:
            logger.error("❌ 環境へのアクセス失敗")
            return 1

        logger.info("✅ 環境へのアクセス成功")

        logger.info(f"\n環境型: {type(env)}")
        actual_env = unwrap_env(env)
        if actual_env is None:
            logger.error("❌ 環境のunwrapに失敗")
            return 1

        logger.info(f"\n最終環境型: {type(actual_env)}")
        logger.info(
            "環境属性 (最初の20個): %s",
            [attr for attr in dir(actual_env) if not attr.startswith("_")][:20],
        )

        # メトリクス取得
        logger.info("\n" + "-" * 80)
        logger.info("メトリクス取得結果:")
        logger.info("-" * 80)

        metrics.update(extract_env_metrics(env, include_optional=True))
        roi = compute_balance_roi(metrics)
        if roi is not None:
            metrics["roi"] = roi

        if "final_balance" in metrics:
            logger.info(f"✅ final_balance: {metrics['final_balance']:.2f}")
        else:
            logger.warning("❌ final_balanceが見つかりません")

        if "initial_balance" in metrics:
            logger.info(f"✅ initial_balance: {metrics['initial_balance']:.2f}")
        else:
            logger.warning("❌ initial_balanceが見つかりません")

        if "roi" in metrics:
            logger.info(f"✅ ROI: {metrics['roi']:.4f}%")

        if "total_trades" in metrics:
            logger.info(f"✅ total_trades: {metrics['total_trades']}")
        else:
            logger.warning("❌ total_trades属性が見つかりません")

        if "buy_count" in metrics:
            logger.info(f"✅ buy_count: {metrics['buy_count']}")
        else:
            logger.warning("❌ buy_count属性が見つかりません")

        if "sell_count" in metrics:
            logger.info(f"✅ sell_count: {metrics['sell_count']}")
        else:
            logger.warning("❌ sell_count属性が見つかりません")
        
        # 結果サマリー
        logger.info("\n" + "="*80)
        logger.info("結果サマリー")
        logger.info("="*80)
        logger.info(f"実行時間: {elapsed_time:.1f}秒")
        logger.info(f"取得メトリクス数: {len(metrics)}")
        logger.info(json.dumps(metrics, indent=2, ensure_ascii=False))
        
        if len(metrics) >= 3:  # 最低でもfinal_balance, roi, total_tradesが必要
            logger.info("\n✅ メトリクス抽出成功！")
            return 0
        else:
            logger.error("\n❌ メトリクス抽出不足")
            return 1
            
    except Exception as e:
        logger.error(f"\n❌ エラー: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
