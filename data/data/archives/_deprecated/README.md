# Deprecated Scripts

このディレクトリには非推奨となった旧スクリプトが含まれています。
これらのスクリプトは後方互換性のために保持されていますが、新しいコードでは使用しないでください。

## 移行ガイド

### トレーニングスクリプトの移行

#### 旧: run_smoke_test.py
```bash
# 旧方法（非推奨）
python run_smoke_test.py --config smoke_test_10k_config.json
```

**新方法:**
```bash
# Unified Trainerを使用
python run_training.py --config configs/training/ppo_100k_optimized.json
```

#### 旧: run_optimized_validation.py
```bash
# 旧方法（非推奨）
python run_optimized_validation.py
```

**新方法:**
```bash
# Unified Trainerで最適化パラメータを使用
python run_training.py --config configs/training/ppo_100k_optimized.json --force
```

## Unified Trainerの利点

1. **統一されたインターフェース**: 全ての学習アルゴリズムを単一のエントリーポイントから実行
2. **設定ファイルベース**: JSONファイルで全てのパラメータを管理
3. **保守性**: バグ修正とテストが一箇所に集約
4. **拡張性**: 新しいアルゴリズムや機能を簡単に追加可能

## 設定ファイルの作成

### 最適化パラメータを使用した100kトレーニング

`configs/training/ppo_100k_optimized.json`:
```json
{
  "algorithm": "ppo",
  "total_timesteps": 100000,

  "learning_rate": 0.009375625,
  "gamma": 0.895,
  "n_steps": 1408,
  "ent_coef": 0.02575,

  "enable_sell_mitigation": true,
  "enable_lagrange": true,
  "lagrange_r_target": 0.175,
  "lagrange_tolerance": 0.0775,
  "lagrange_eta": 0.02575,

  "checkpoint_dir": "checkpoints/ppo_100k_optimized",
  "model_dir": "models"
}
```

### 実行方法

```bash
# ドライラン（設定確認のみ）
python run_training.py --config configs/training/ppo_100k_optimized.json --dry-run

# 実際のトレーニング
python run_training.py --config configs/training/ppo_100k_optimized.json --force
```

## 削除予定

これらのスクリプトは将来のバージョンで削除される予定です。
可能な限り早く新しいUnified Trainerに移行してください。

## サポート

移行に関する質問や問題がある場合は、プロジェクトのIssueトラッカーで報告してください。
