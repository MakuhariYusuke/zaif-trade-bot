# Phase 3: Environment & Backtest Integration - 実装指示書

## 概要

Feature Schema Management Reformの**Phase 3**として、Environment統合とBacktest対応を実装してください。

Phase 1-2で実装済みの`FeatureSchemaManager`を活用し、モデルごとの特徴量スキーマに基づいて環境を動的に構築できるようにします。これにより、バックテスト時の次元不一致エラーを完全に解消します。

## 背景

### 現在の問題
```python
# v381モデル（110特徴量）のバックテスト
env = HeavyTradingEnv(df=df, config=config)  # 環境は68特徴量を生成
model = MaskablePPO.load("models/ppo_reward_v381.zip")  # 110特徴量を期待

# 実行
action = model.predict(obs)
# ❌ エラー: ValueError: Unexpected observation shape (68,) for Box environment,
#            please use (110,) for the observation shape.
```

### Phase 1-2で実装済み

1. **FeatureSchemaManager** (`ztb/training/core/feature_schema_manager.py`)
   - モデルごとのスキーマ保存・読み込み
   - 互換性チェック
   - メタデータ管理

2. **UnifiedTrainer統合** (`ztb/training/core/unified_trainer.py`)
   - 訓練時の自動スキーマ保存
   - `_save_model_schema()`メソッド実装

3. **ディレクトリ構造**
   ```
   models/
   ├── ppo_reward_v381_revised_profit_focused.zip
   ├── ppo_reward_v384_curated_60.zip
   └── schemas/
       ├── ppo_reward_v381_revised_profit_focused/
       │   ├── metadata.json
       │   ├── features_schema.json
       │   └── scaler.npz
       └── ppo_reward_v384_curated_60/
           ├── metadata.json
           ├── features_schema.json
           └── scaler.npz
   ```

## Phase 3の目標

### 実装すべき機能

1. **スキーマベースの環境作成** ⭐ 最優先
2. **バックテストの自動スキーマ対応**
3. **live_trade/paper_tradeの対応**
4. **既存モデルのスキーマ移行ツール**

## 実装タスク

### Task 1: スキーマベースの環境作成関数

**ファイル**: `ztb/trading/environment/schema_env_factory.py` (新規作成)

**実装内容**:
```python
"""
Schema-based Environment Factory

スキーマ情報からEnvironmentを動的に構築します。
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ztb.training.core.feature_schema_manager import FeatureSchemaManager
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_env_from_schema(
    model_name: str,
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    models_dir: Path = Path("models")
) -> HeavyTradingEnv:
    """
    モデルのスキーマ情報から環境を作成

    Args:
        model_name: モデル名（例: "ppo_reward_v384_curated_60"）
        df: 市場データ
        config: 環境設定（Noneの場合はデフォルト）
        models_dir: モデルディレクトリ

    Returns:
        HeavyTradingEnv: スキーマに基づいた環境

    Raises:
        FileNotFoundError: スキーマが見つからない場合
        ValueError: データに必要な特徴量がない場合
    """
    # スキーマ読み込み
    manager = FeatureSchemaManager(model_name, models_dir)
    metadata = manager.load_schema()
    scaler = manager.load_scaler()

    logger.info(f"Creating environment from schema: {model_name}")
    logger.info(f"  Expected features: {metadata.num_features}")
    logger.info(f"  Feature names: {metadata.feature_names[:5]}... (showing first 5)")

    # データに必要な特徴量があるか確認
    missing_features = set(metadata.feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Dataset is missing required features: {missing_features}\n"
            f"Dataset has: {len(df.columns)} columns\n"
            f"Model expects: {metadata.num_features} features"
        )

    # 設定を構築
    env_config = config or {}

    # スキーマ情報を設定に追加
    env_config.update({
        "feature_names": metadata.feature_names,
        "num_features": metadata.num_features,
        "schema_hash": metadata.schema_hash,
        "model_name": model_name,
    })

    # スケーラー情報を追加
    if scaler:
        env_config.update({
            "scaler_mean": scaler["mean"],
            "scaler_std": scaler["std"],
        })

    # 訓練設定から環境パラメータを抽出（可能な範囲で）
    training_config = metadata.training_config
    for key in ["reward_scaling", "transaction_cost", "max_position_size", "risk_free_rate"]:
        if key in training_config and key not in env_config:
            env_config[key] = training_config[key]

    # 環境作成
    env = HeavyTradingEnv(df=df, config=env_config)

    logger.info(f"✅ Environment created with {metadata.num_features} features")

    return env


def create_env_from_model_path(
    model_path: str,
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> HeavyTradingEnv:
    """
    モデルファイルパスから環境を作成

    Args:
        model_path: モデルファイルパス（例: "models/ppo_reward_v384_curated_60.zip"）
        df: 市場データ
        config: 環境設定

    Returns:
        HeavyTradingEnv: スキーマに基づいた環境
    """
    # モデル名を抽出
    model_name = Path(model_path).stem

    # models/ディレクトリを特定
    model_path_obj = Path(model_path)
    if model_path_obj.parent.name == "models":
        models_dir = model_path_obj.parent
    else:
        models_dir = Path("models")

    return create_env_from_schema(model_name, df, config, models_dir)
```

**テスト**:
```python
# テストコード (tests/test_schema_env_factory.py)
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from ztb.trading.environment.schema_env_factory import (
    create_env_from_schema,
    create_env_from_model_path
)

def test_create_env_from_schema():
    # モックデータ作成
    df = pd.DataFrame({
        'close': np.random.randn(100),
        'volume': np.random.randn(100),
        # ... 68特徴量
    })

    # 環境作成
    env = create_env_from_schema(
        model_name="ppo_reward_v384_curated_60",
        df=df
    )

    # 検証
    assert env is not None
    assert env.observation_space.shape[0] == 68  # v384は68特徴量
```

### Task 2: HeavyTradingEnvの拡張

**ファイル**: `ztb/trading/environment/environment.py`

**修正内容**:

1. **スキーマ情報の受け入れ**:
```python
class HeavyTradingEnv:
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        # 既存のコード...

        # スキーマ情報の取得（オプショナル）
        self.schema_hash = config.get("schema_hash")
        self.model_name = config.get("model_name")
        self.feature_names = config.get("feature_names")

        if self.feature_names:
            logger.info(f"Using schema-defined features: {len(self.feature_names)}")

        # 既存のコード...
```

2. **スケーラーの動的設定**:
```python
def _setup_scaler(self):
    """スケーラーのセットアップ"""
    # スキーマからのスケーラー情報があれば使用
    if "scaler_mean" in self.config and "scaler_std" in self.config:
        self.scaler = NormalizationStats(
            feature_names=self.feature_names,
            mean=self.config["scaler_mean"],
            std=self.config["scaler_std"],
            n_samples=len(self.df),
        )
        logger.info("Using scaler from schema")
    else:
        # 従来通りデータから計算
        self._compute_scaler()
```

### Task 3: バックテストスクリプトの改修

**ファイル**: `backtest_with_schema.py` (新規作成)

**実装内容**:
```python
#!/usr/bin/env python3
"""
Schema-aware Backtest Script

モデルのスキーマ情報を自動検出してバックテストを実行します。
v381（110特徴量）とv384（68特徴量）の両方に対応。
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from datetime import datetime

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.training.policies.policy_utils import predict_with_masks
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_backtest_with_schema(
    model_path: str,
    data_path: str,
    episodes: int = 10
) -> dict:
    """
    スキーマを考慮したバックテスト

    Args:
        model_path: モデルファイルパス
        data_path: データファイルパス
        episodes: エピソード数

    Returns:
        バックテスト結果
    """
    logger.info("="*80)
    logger.info("Schema-aware Backtest")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")

    # データ読み込み
    df = load_csv_data_optimized(data_path)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # スキーマベースで環境作成（自動的に特徴量数を調整）
    env = create_env_from_model_path(model_path, df)
    logger.info(f"Environment created with {env.observation_space.shape[0]} features")

    # モデル読み込み
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # バックテスト実行
    all_rewards = []
    all_pnls = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_pnl = 0.0
        steps = 0

        while not (done or truncated) and steps < 1000:
            action, _ = predict_with_masks(model, obs, env, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()

            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            ep_pnl += info.get('pnl', 0.0)

            # Count actions
            if action == 0:
                action_counts["HOLD"] += 1
            elif action == 1:
                action_counts["BUY"] += 1
            else:
                action_counts["SELL"] += 1

            steps += 1

        all_rewards.append(ep_reward)
        all_pnls.append(ep_pnl)

        logger.info(
            f"Episode {ep+1:2d}: Reward={ep_reward:7.2f}, "
            f"PnL={ep_pnl:10,.0f} JPY, Steps={steps:4d}"
        )

    # 結果サマリー
    total_actions = sum(action_counts.values())

    results = {
        "model_path": model_path,
        "data_path": data_path,
        "episodes": episodes,
        "avg_reward": float(np.mean(all_rewards)),
        "avg_pnl": float(np.mean(all_pnls)),
        "total_pnl": float(np.sum(all_pnls)),
        "action_distribution": {
            k: {"count": v, "pct": (v/total_actions*100 if total_actions > 0 else 0)}
            for k, v in action_counts.items()
        },
    }

    logger.info("\n" + "="*80)
    logger.info("Backtest Results")
    logger.info("="*80)
    logger.info(f"Average Reward: {results['avg_reward']:.2f}")
    logger.info(f"Average PnL: {results['avg_pnl']:,.0f} JPY")
    logger.info(f"Total PnL: {results['total_pnl']:,.0f} JPY")
    logger.info(f"Action Distribution:")
    for action, stats in results['action_distribution'].items():
        logger.info(f"  {action}: {stats['count']:,} ({stats['pct']:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Schema-aware Backtest")
    parser.add_argument("--model", required=True, help="Model path (.zip)")
    parser.add_argument("--data", default="ml-dataset-enhanced.csv", help="Data path (.csv)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    try:
        results = run_backtest_with_schema(
            model_path=args.model,
            data_path=args.data,
            episodes=args.episodes
        )

        # 結果保存
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**使用例**:
```bash
# v381（110特徴量）のバックテスト
python backtest_with_schema.py --model models/ppo_reward_v381_revised_profit_focused.zip --data ml-dataset-enhanced.csv --episodes 20

# v384（68特徴量）のバックテスト
python backtest_with_schema.py --model models/ppo_reward_v384_curated_60.zip --data ml-dataset-enhanced.csv --episodes 20

# 両方とも次元不一致エラーなしで実行できる！
```

### Task 4: 既存モデルのスキーマ移行

**ファイル**: `scripts/migrate_legacy_schemas.py` (新規作成)

**実装内容**:
```python
#!/usr/bin/env python3
"""
Legacy Schema Migration Tool

既存のモデルのスキーマ情報を新しい管理システムに移行します。
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Optional, Dict, Any

from ztb.training.core.feature_schema_manager import FeatureSchemaManager, migrate_legacy_schema
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


# 既存モデルの設定情報（手動で定義）
KNOWN_MODELS = {
    "ppo_reward_v381_revised_profit_focused": {
        "num_features": 110,
        "config": {
            "curated_features_list": None,  # 全特徴量
            "enable_feature_filtering": False,
            "total_timesteps": 100000,  # 推定
            "learning_rate": 0.003,
            "vf_coef": 0.3,
            "target_kl": 0.01,
        }
    },
    "ppo_reward_v384_curated_60": {
        "num_features": 68,
        "config": {
            "curated_features_list": "curated_features.py::CURATED_FEATURES",
            "enable_feature_filtering": True,
            "feature_filter_mode": "whitelist",
            "total_timesteps": 50000,
            "learning_rate": 0.003,
            "vf_coef": 0.3,
            "target_kl": 0.01,
        }
    },
}


def migrate_model(model_name: str, force: bool = False):
    """単一モデルのスキーマを移行"""
    logger.info(f"Migrating schema for: {model_name}")

    # スキーマがすでに存在するか確認
    manager = FeatureSchemaManager(model_name)
    try:
        existing = manager.load_schema()
        if not force:
            logger.warning(f"Schema already exists for {model_name}")
            logger.warning(f"  Features: {existing.num_features}")
            logger.warning(f"  Use --force to overwrite")
            return False
    except FileNotFoundError:
        pass  # スキーマが存在しない（正常）

    # レガシースキーマを移行
    legacy_schema_path = Path("models/features_schema.json")
    legacy_scaler_path = Path("models/scaler.npz")

    if not legacy_schema_path.exists():
        logger.error(f"Legacy schema not found: {legacy_schema_path}")
        return False

    # 既知の設定を取得
    config = KNOWN_MODELS.get(model_name, {}).get("config", {})

    # 移行実行
    migrate_legacy_schema(
        model_name=model_name,
        legacy_schema_path=legacy_schema_path,
        legacy_scaler_path=legacy_scaler_path,
        config=config
    )

    logger.info(f"✅ Migration completed for {model_name}")
    return True


def migrate_all_models(force: bool = False):
    """すべての既知モデルを移行"""
    logger.info("="*80)
    logger.info("Migrating all known models")
    logger.info("="*80)

    success_count = 0
    for model_name in KNOWN_MODELS:
        if migrate_model(model_name, force):
            success_count += 1

    logger.info(f"\n✅ Successfully migrated {success_count}/{len(KNOWN_MODELS)} models")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate legacy schemas")
    parser.add_argument("--model", help="Specific model to migrate")
    parser.add_argument("--all", action="store_true", help="Migrate all known models")
    parser.add_argument("--force", action="store_true", help="Overwrite existing schemas")
    parser.add_argument("--list", action="store_true", help="List all available schemas")

    args = parser.parse_args()

    if args.list:
        # 利用可能なスキーマをリスト
        FeatureSchemaManager.print_schema_summary()
        return 0

    if args.all:
        migrate_all_models(args.force)
    elif args.model:
        migrate_model(args.model, args.force)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**使用例**:
```bash
# すべての既知モデルを移行
python scripts/migrate_legacy_schemas.py --all

# 特定モデルのみ移行
python scripts/migrate_legacy_schemas.py --model ppo_reward_v381_revised_profit_focused

# 既存スキーマを上書き
python scripts/migrate_legacy_schemas.py --all --force

# スキーマ一覧表示
python scripts/migrate_legacy_schemas.py --list
```

### Task 5: live_trade/paper_tradeの対応

**ファイル**: `live_trade.py`

**修正箇所**:
```python
# 既存のモデル読み込み部分を修正
def __init__(self, model_path: str, ...):
    # モデル読み込み
    self.model = MaskablePPO.load(model_path)

    # スキーマベースで環境作成（追加）
    from ztb.trading.environment.schema_env_factory import create_env_from_model_path

    # リアルタイムデータ取得
    df = self._fetch_latest_data()

    # スキーマベースで環境作成
    self.env = create_env_from_model_path(
        model_path=model_path,
        df=df,
        config=self.config
    )

    logger.info(f"Environment created with {self.env.observation_space.shape[0]} features")
```

## テスト計画

### 単体テスト

```bash
# FeatureSchemaManager (すでに実装済み)
pytest tests/test_feature_schema_manager.py

# Schema Environment Factory (新規)
pytest tests/test_schema_env_factory.py

# HeavyTradingEnv拡張 (新規)
pytest tests/test_environment_schema_integration.py
```

### 統合テスト

```bash
# v381バックテスト（110特徴量）
python backtest_with_schema.py \
    --model models/ppo_reward_v381_revised_profit_focused.zip \
    --data ml-dataset-enhanced.csv \
    --episodes 20 \
    --output results_v381.json

# v384バックテスト（68特徴量）
python backtest_with_schema.py \
    --model models/ppo_reward_v384_curated_60.zip \
    --data ml-dataset-enhanced.csv \
    --episodes 20 \
    --output results_v384.json

# 両方とも成功することを確認
```

### 回帰テスト

```bash
# 既存の訓練が動作することを確認
python run_training.py --config archived/configs/ppo_legacy/training/ppo_test.json

# スキーマが自動保存されることを確認
ls -R models/schemas/ppo_test/
```

## 成功基準

### 必須要件
- [ ] `create_env_from_schema()`関数が動作
- [ ] v381モデル（110特徴量）のバックテストが成功
- [ ] v384モデル（68特徴量）のバックテストが成功
- [ ] 次元不一致エラーが発生しない
- [ ] 既存の訓練フローが正常動作

### 追加要件
- [ ] レガシースキーマ移行ツールが動作
- [ ] live_trade/paper_tradeが対応
- [ ] 全テストがパス
- [ ] ドキュメント更新

## 注意事項

### 既存コードの互換性

**重要**: 既存の訓練・推論フローを壊さないこと

```python
# 従来のコード（引き続き動作すべき）
env = HeavyTradingEnv(df=df, config=config)

# 新しいコード（追加オプション）
env = create_env_from_schema(model_name, df, config)
```

### エラーハンドリング

- スキーマが見つからない場合: 明確なエラーメッセージ
- データに特徴量が不足している場合: 詳細なログ
- レガシーモデル（スキーマなし）: フォールバック動作

### パフォーマンス

- スキーマ読み込みはキャッシュ
- 環境作成時の追加オーバーヘッドを最小化
- バックテスト速度に影響を与えない

## 実装順序（推奨）

1. **Day 1**: Task 1（スキーマベース環境作成）
   - `schema_env_factory.py`実装
   - 単体テスト作成

2. **Day 2**: Task 2（HeavyTradingEnv拡張）
   - スキーマ情報の受け入れ
   - スケーラー動的設定

3. **Day 3**: Task 3（バックテスト改修）
   - `backtest_with_schema.py`実装
   - v381/v384でテスト

4. **Day 4**: Task 4（スキーマ移行）
   - `migrate_legacy_schemas.py`実装
   - 既存モデル移行

5. **Day 5**: Task 5 & 統合テスト
   - live_trade対応
   - 全体テスト
   - ドキュメント更新

## 参考資料

### 既存コード

- **FeatureSchemaManager**: `ztb/training/core/feature_schema_manager.py`
- **UnifiedTrainer**: `ztb/training/core/unified_trainer.py`
- **HeavyTradingEnv**: `ztb/trading/environment/environment.py`

### ドキュメント

- **Phase 1-2実装サマリー**: `docs/FEATURE_SCHEMA_IMPLEMENTATION_SUMMARY.md`
- **改修計画書**: `docs/FEATURE_SCHEMA_MANAGEMENT_REFORM.md`
- **検証状況**: `docs/V381_V384_VERIFICATION_STATUS.md`

### スキーマ例

```json
// models/schemas/ppo_reward_v384_curated_60/metadata.json
{
  "model_name": "ppo_reward_v384_curated_60",
  "num_features": 68,
  "feature_names": ["close", "volume", "sma_short", ...],
  "schema_hash": "f7be18533fa61876",
  "created_at": "2025-10-10T16:26:20",
  "training_config": {
    "curated_features_list": "curated_features.py::CURATED_FEATURES",
    "enable_feature_filtering": true,
    "feature_filter_mode": "whitelist",
    "total_timesteps": 50000
  },
  "curated_features_spec": "curated_features.py::CURATED_FEATURES",
  "feature_filtering_enabled": true,
  "feature_filter_mode": "whitelist"
}
```

## 完了報告

実装完了時に以下を含めて報告してください:

1. **実装したファイル一覧**
2. **テスト結果**（すべてパスしたか）
3. **バックテスト結果**（v381とv384の両方）
4. **既知の問題・制限事項**
5. **次のフェーズ（Phase 4）への引き継ぎ事項**

## 質問・問題が発生した場合

以下の情報を添えて報告してください:

1. エラーメッセージ全文
2. 実行したコマンド
3. 期待した動作
4. 実際の動作
5. 関連するログ

---

**作成日**: 2025-10-10
**対象フェーズ**: Phase 3 (Environment & Backtest Integration)
**前提**: Phase 1-2完了（FeatureSchemaManager + UnifiedTrainer統合）
**目標**: バックテスト時の次元不一致エラーを完全解消
