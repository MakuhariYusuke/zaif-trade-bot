"""
SAC v396改善版ログシステムのテスト訓練

1,000ステップの短期訓練で以下を検証:
- CSVメトリクスファイルが100ステップごとに記録される
- TensorBoardログが改善される
- 標準出力が正常に表示される
"""

import json
from pathlib import Path

# テスト用の設定
config = {
    "model_name": "sac_v396_log_test",
    "session_id": "sac_session",
    "algorithm": "sac",
    "total_timesteps": 1000,
    "metrics_log_interval": 100,
    "checkpoint_interval": 500,
    "data_path": "btc_jpy_real_dataset.csv",
    "environment": {
        "initial_balance": 1000000,
        "commission_rate": 0.001,
        "price_column": "close",
        "use_continuous_actions": True,
        "enable_action_masking": False,
    },
    "sac_hyperparameters": {
        "learning_rate": 0.000574,
        "batch_size": 128,
        "gamma": 0.9948,
        "tau": 0.005,
        "ent_coef": "auto",
        "target_update_interval": 3,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 100,
        "buffer_size": 5000,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
}

# 設定を保存
config_path = Path("configs/sac_v396_log_test.json")
config_path.parent.mkdir(parents=True, exist_ok=True)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"✅ Test config saved: {config_path}")
print()
print("Run the following command to test:")
print(
    f"  $env:MPLBACKEND='Agg'; python scripts\\optimization\\train_with_config.py --config {config_path}"
)
print()
print("Expected outputs:")
print("  1. CSV file: checkpoints/sac_session/sac_v396_log_test_training_metrics.csv")
print("  2. ~10 rows in CSV (100, 200, ..., 1000)")
print("  3. TensorBoard logs with improved frequency")
