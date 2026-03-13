"""超短期テスト（1000ステップ）- メトリクス抽出確認用"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    compute_balance_roi,
    extract_env_metrics,
    resolve_env,
    unwrap_env,
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

config = {
    "experiment_name": "ultra_short_test",
    "algorithm": "sac",
    "training": {
        "seed": 42,
        "total_timesteps": 1000,
        "learning_starts": 50,
        "save_interval": 999999,
        "data_config": {
            "data_path": "data/btc_jpy_1m_v451_optimized_features.parquet",
            "validation_split": 0.2,
        },
        "sac_hyperparameters": {
            "learning_rate": 3e-4,
            "buffer_size": 10000,
            "batch_size": 128,
            "gamma": 0.99,
            "tau": 0.005,
        },
        # 環境設定を追加 - 連続アクションスペース必須
        "environment": {
            "action_space_type": "continuous",  # SACには連続アクションスペースが必要
        },
    },
}

print("\n" + "="*80)
print("超短期テスト開始（1000ステップ）")
print("="*80)

trainer = SACTrainer(config=config, logger=logger)
result = trainer.train()

print("\n" + "="*80)
print("メトリクス抽出開始")
print("="*80)

# 環境アクセス
env = resolve_env(trainer)
if env is None:
    print("❌ 環境アクセス失敗")
    sys.exit(1)
print("✅ 環境アクセス成功")

# Unwrap
print(f"\n環境型: {type(env).__name__}")
actual_env = unwrap_env(env)
if actual_env is None:
    print("❌ 環境unwrap失敗")
    sys.exit(1)

print(f"\n最終環境型: {type(actual_env).__name__}")

# メトリクス取得
print("\n" + "-"*80)
print("メトリクス:")
print("-"*80)

metrics = extract_env_metrics(env, include_optional=True)
roi = compute_balance_roi(metrics)
if roi is not None:
    metrics["roi"] = roi

if "final_balance" in metrics:
    print(f"✅ final_balance: {metrics['final_balance']:.2f}")
else:
    print("❌ final_balance 見つからず")

if "initial_balance" in metrics:
    print(f"✅ initial_balance: {metrics['initial_balance']:.2f}")
else:
    print("❌ initial_balance 見つからず")

if "roi" in metrics:
    print(f"✅ ROI: {metrics['roi']:.4f}%")

if "total_trades" in metrics:
    print(f"✅ total_trades: {metrics['total_trades']}")
else:
    print("❌ total_trades 見つからず")

if "buy_count" in metrics:
    print(f"✅ buy_count: {metrics['buy_count']}")
else:
    print("❌ buy_count 見つからず")

if "sell_count" in metrics:
    print(f"✅ sell_count: {metrics['sell_count']}")
else:
    print("❌ sell_count 見つからず")

print("\n" + "="*80)
print(f"結果: {len(metrics)}/5 メトリクス取得")
print("="*80)

if len(metrics) >= 3:
    print("✅ メトリクス抽出成功！")
    sys.exit(0)
else:
    print("❌ メトリクス抽出不足")
    sys.exit(1)
