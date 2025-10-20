# 本番設定ファイル更新コマンド

**作成日**: 2025年10月10日
**対象**: 検証済みパラメータの本番適用

---

## 🎯 目的

2025年10月10日に完了した検証結果を本番設定ファイルに反映します。

**検証済みパラメータ**:
- ✅ `batch_size`: 256 (高信頼度)
- ✅ `learning_rate`: 0.007503 (高信頼度)
- ⚠️ `max_grad_norm`: 5.05 (暫定、追加検証推奨)

---

## 📋 更新対象ファイル

1. `configs/training/ppo_100k_optimized.json` - メイン設定ファイル
2. `configs/training/ppo_binary_search_validated.json` - 検証済み設定 (存在する場合)

---

## 🔄 更新方法

### 方法1: 手動更新 (推奨)

**ステップ1**: 設定ファイルを開く
```cmd
notepad configs\training\ppo_100k_optimized.json
```

**ステップ2**: 以下の行を変更

```json
{
  "_comment_ppo": "Optimized PPO hyperparameters from 2025-10-10 binary search validation",

  "learning_rate": 0.007503,     // 旧: 0.009375625 → 新: 0.007503
  "batch_size": 256,             // 旧: 64 → 新: 256
  "max_grad_norm": 5.05,         // 旧: 0.5 → 新: 5.05 (暫定)

  // 以下は既存の最適化済み値 (変更なし)
  "n_steps": 1024,
  "gamma": 0.8475,
  "n_epochs": 16,
  "vf_coef": 0.1
}
```

**ステップ3**: ファイルを保存

**ステップ4**: 設定ファイルの検証
```cmd
python -c "import json; print(json.load(open('configs/training/ppo_100k_optimized.json')))"
```

---

### 方法2: バックアップ付き更新 (安全)

**ステップ1**: バックアップを作成
```cmd
REM 現在の設定をバックアップ
copy configs\training\ppo_100k_optimized.json configs\training\ppo_100k_optimized_backup_20251010.json

REM バックアップ確認
dir configs\training\ppo_100k_optimized*.json
```

**ステップ2**: 設定ファイルを開いて更新
```cmd
notepad configs\training\ppo_100k_optimized.json
```

**変更内容**:
```json
{
  "_comment_ppo": "Optimized PPO hyperparameters from 2025-10-10 binary search validation",
  "learning_rate": 0.007503,
  "n_steps": 1024,
  "batch_size": 256,
  "n_epochs": 16,
  "gamma": 0.8475,
  "ent_coef": 0.02575,
  "vf_coef": 0.1,
  "max_grad_norm": 5.05,
  "clip_range": 0.3,
  "gae_lambda": 0.95,
  "target_kl": 0.15
}
```

**ステップ3**: 設定ファイルの検証
```cmd
python -c "import json; config = json.load(open('configs/training/ppo_100k_optimized.json')); print('learning_rate:', config['learning_rate']); print('batch_size:', config['batch_size']); print('max_grad_norm:', config['max_grad_norm'])"
```

**期待される出力**:
```
learning_rate: 0.007503
batch_size: 256
max_grad_norm: 5.05
```

---

### 方法3: Python スクリプトで自動更新

**更新スクリプト** (`update_config.py`):

```python
import json
from pathlib import Path

# 設定ファイルのパス
config_path = Path("configs/training/ppo_100k_optimized.json")
backup_path = Path("configs/training/ppo_100k_optimized_backup_20251010.json")

# バックアップ作成
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Backup created: {backup_path}")

    # 検証済みパラメータで更新
    config['learning_rate'] = 0.007503
    config['batch_size'] = 256
    config['max_grad_norm'] = 5.05
    config['_comment_ppo'] = "Optimized PPO hyperparameters from 2025-10-10 binary search validation"

    # 更新を保存
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Config updated: {config_path}")
    print(f"   learning_rate: {config['learning_rate']}")
    print(f"   batch_size: {config['batch_size']}")
    print(f"   max_grad_norm: {config['max_grad_norm']}")
else:
    print(f"❌ Config file not found: {config_path}")
```

**実行方法**:
```cmd
REM スクリプトを保存して実行
python update_config.py
```

---

## 📊 更新前後の比較

| パラメータ | 旧値 | 新値 | 変化 | 根拠 |
|-----------|------|------|------|------|
| `learning_rate` | 0.009375625 | **0.007503** | -20% | 50k検証、improvement 37.36 |
| `batch_size` | 64 | **256** | +300% | 50k検証、improvement 27.71 |
| `max_grad_norm` | 0.5 | **5.05** | +910% | 30k検証、暫定値 (追加検証推奨) |
| `n_steps` | 1024 | 1024 | 変更なし | 既存最適化済み |
| `gamma` | 0.8475 | 0.8475 | 変更なし | 既存最適化済み |
| `n_epochs` | 16 | 16 | 変更なし | 既存最適化済み |
| `vf_coef` | 0.1 | 0.1 | 変更なし | 既存最適化済み |

---

## ✅ 更新後の検証

### 1. 設定ファイルの妥当性チェック

```cmd
REM JSON構文チェック
python -c "import json; json.load(open('configs/training/ppo_100k_optimized.json')); print('✅ JSON valid')"

REM パラメータ値確認
python -c "import json; config = json.load(open('configs/training/ppo_100k_optimized.json')); assert config['batch_size'] == 256, 'batch_size mismatch'; assert config['learning_rate'] == 0.007503, 'learning_rate mismatch'; assert config['max_grad_norm'] == 5.05, 'max_grad_norm mismatch'; print('✅ All parameters correct')"
```

### 2. ドライラン (実行テスト)

```cmd
REM 設定ファイルでドライラン実行
python run_training.py --config configs/training/ppo_100k_optimized.json --dry-run

REM エラーがなければ成功
```

### 3. 短時間トレーニングテスト (推奨)

```cmd
REM 1000ステップのテスト実行
python run_training.py --config configs/training/ppo_100k_optimized.json --force

REM または、設定ファイルを一時的にコピーしてtotal_timestepsを1000に変更
copy configs\training\ppo_100k_optimized.json configs\training\ppo_test.json
REM ppo_test.json の total_timesteps を 1000 に変更
python run_training.py --config configs/training/ppo_test.json --force
```

---

## 🔄 ロールバック方法

更新後に問題が発生した場合:

```cmd
REM バックアップから復元
copy configs\training\ppo_100k_optimized_backup_20251010.json configs\training\ppo_100k_optimized.json

REM 復元確認
python -c "import json; print(json.load(open('configs/training/ppo_100k_optimized.json'))['learning_rate'])"
```

---

## 📝 更新履歴の記録

**CHANGELOG.md** または設定ファイル内のコメントに記録:

```json
{
  "_comment_history": [
    "2025-10-10: Updated batch_size (64→256), learning_rate (0.0094→0.0075), max_grad_norm (0.5→5.05) based on binary search validation",
    "Previous: Binary search optimization (n_steps, gamma, vf_coef)"
  ]
}
```

---

## 📚 関連ドキュメント

- **検証結果**: [`VALIDATION_RESULTS_2025-10-10.md`](./VALIDATION_RESULTS_2025-10-10.md)
- **次のステップ**: [`NEXT_STEPS_COMMANDS.md`](./NEXT_STEPS_COMMANDS.md)
- **進捗管理**: [`PARAMETER_VALIDATION_TRACKING.md`](./PARAMETER_VALIDATION_TRACKING.md)

---

## ⚠️ 注意事項

### max_grad_norm について

- **現在の推奨値**: 5.05 (暫定)
- **理由**: 30k検証で再現性に課題 (7.525 vs 5.05)
- **推奨アクション**: 100k×2シード検証完了後、確定値に更新

**100k検証完了後の再更新**:
```cmd
REM 検証結果に基づいて再度更新
notepad configs\training\ppo_100k_optimized.json
REM max_grad_norm を確定値に変更
```

---

**準備ができたら、バックアップを取ってから更新してください!** 📝
