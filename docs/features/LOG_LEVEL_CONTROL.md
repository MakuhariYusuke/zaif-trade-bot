# ログレベル制御ガイド

**目的**: unified_trainer実行時のログ出力を制御し、視認性を向上

---

## 🎯 問題

- DEBUGログが大量に出力され、重要な情報が埋もれる
- サードパーティライブラリ（torch、stable-baselines3等）のDEBUGログも表示される
- 長時間実行時にログが見づらい

---

## ✅ 解決策

`--log-level` 引数を追加し、ログレベルを制御可能に。

---

## 🚀 使用方法

### 基本構文

```bash
python -m ztb.training.unified_trainer --config <設定ファイル> --log-level <レベル>
```

### ログレベル一覧

| レベル | 説明 | 用途 |
|--------|------|------|
| **DEBUG** | すべてのログ（最も詳細） | デバッグ時のみ |
| **INFO** | 重要な情報のみ（**デフォルト、推奨**） | 通常実行 |
| **WARNING** | 警告とエラーのみ | 安定稼働時 |
| **ERROR** | エラーのみ | 本番環境 |
| **CRITICAL** | 致命的エラーのみ | 本番環境（最小ログ） |

---

## 📝 実行例

### 通常実行（推奨）

```bash
# INFOレベル（デフォルト）- 重要な情報のみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO
```

**出力例**:
```
2025-10-07 10:00:00 - ztb.training.unified_trainer - INFO - Loaded config from configs\train\ensemble_B_100k_test.json
2025-10-07 10:00:01 - ztb.training.unified_trainer - INFO - Algorithm: ppo
2025-10-07 10:00:01 - ztb.training.unified_trainer - INFO - Starting PPO training...
2025-10-07 10:00:05 - ztb.trading.training.ppo_trainer - INFO - Training started (total_timesteps=100000)
2025-10-07 10:05:00 - ztb.trading.training.ppo_trainer - INFO - Checkpoint saved: checkpoints/ensemble_B_100k_test/checkpoint_10000
```

---

### 静かに実行（WARNINGレベル）

```bash
# 警告とエラーのみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING
```

**出力例**:
```
2025-10-07 10:00:00 - ztb.trading.training.ppo_trainer - WARNING - SELL rate is low: 0.03
2025-10-07 10:05:00 - ztb.trading.training.ppo_trainer - WARNING - Gradient norm is zero for SELL action
```

**用途**: 長時間実行時、安定稼働時

---

### デバッグ実行（DEBUGレベル）

```bash
# すべてのログを表示（デバッグ時のみ）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level DEBUG
```

**出力例**:
```
2025-10-07 10:00:00 - ztb.training.unified_trainer - DEBUG - Args: Namespace(config='configs\\train\\ensemble_B_100k_test.json', log_level='DEBUG', ...)
2025-10-07 10:00:00 - ztb.training.unified_trainer - DEBUG - Loading config from configs\train\ensemble_B_100k_test.json
2025-10-07 10:00:00 - ztb.utils.file_utils - DEBUG - Opening file: configs\train\ensemble_B_100k_test.json
2025-10-07 10:00:01 - ztb.trading.training.ppo_trainer - DEBUG - Initializing PPOTrainer...
2025-10-07 10:00:01 - torch.nn.modules.module - DEBUG - Registered hook on module...
2025-10-07 10:00:02 - stable_baselines3.common.logger - DEBUG - Writing to TensorBoard...
```

**用途**: エラー調査、動作確認時のみ

---

### エラーのみ表示（ERRORレベル）

```bash
# エラーのみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level ERROR
```

**出力例**:
```
2025-10-07 10:00:00 - ztb.training.unified_trainer - ERROR - Failed to load config: FileNotFoundError
```

**用途**: 本番環境、CI/CD

---

## 🔧 100kテストでの推奨設定

### シーケンシャル実行（INFOレベル）

```bash
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level INFO
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level INFO
```

---

### 並列実行（WARNINGレベル推奨）

```powershell
# 3つのウィンドウで並列実行（WARNINGレベルで静かに）
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level WARNING"
```

**理由**: 3つのウィンドウでINFOログが大量に出ると混乱するため、WARNINGレベルが見やすい

---

## 📊 --verbose フラグとの関係

### 従来の方法（--verbose）

```bash
# DEBUGレベル
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --verbose

# INFOレベル（デフォルト）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json
```

### 新しい方法（--log-level）

```bash
# より柔軟な制御
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING
```

**優先順位**: `--log-level` > `--verbose`

---

## 🎯 推奨ログレベル

| 状況 | 推奨レベル | 理由 |
|------|------------|------|
| 初回実行・動作確認 | **INFO** | 重要な情報を確認 |
| 100kテスト（シーケンシャル） | **INFO** | 進捗とチェックポイントを確認 |
| 100kテスト（並列） | **WARNING** | 複数ウィンドウで混乱しないように |
| 1M学習（長時間） | **WARNING** | ログ量を抑制 |
| エラー調査 | **DEBUG** | 詳細な情報で原因特定 |
| 本番環境 | **ERROR** | エラーのみ記録 |

---

## 💡 Tips

### 1. ログをファイルに保存

```bash
# INFOログをファイルに保存（コンソールには表示）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO 2>&1 | Tee-Object -FilePath training.log
```

---

### 2. 特定のモジュールだけDEBUG

```python
# 設定ファイルに追加（将来実装）
"logging": {
  "root_level": "INFO",
  "modules": {
    "ztb.trading.training.ppo_trainer": "DEBUG",
    "stable_baselines3": "WARNING"
  }
}
```

**注**: 現在は未実装、将来的に対応予定

---

### 3. TensorBoardを活用

ログを減らしてもTensorBoardで詳細確認可能:

```bash
# WARNINGレベルで実行（ログ少ない）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING

# 別ターミナルでTensorBoard起動（詳細な指標確認）
tensorboard --logdir logs --port 6006
```

---

## 📚 関連ドキュメント

- **QUICKSTART_100K_TEST.md**: 100kテスト実行ガイド（ログレベル制御例あり）
- **UNIFIED_TRAINER_INTEGRATION_SUMMARY.md**: unified_trainer統合サマリー
- **CHECKPOINT_INTERVAL_EXTENSION.md**: checkpoint_interval拡張ガイド

---

## ✅ まとめ

1. **推奨**: `--log-level INFO` で通常実行
2. **並列実行時**: `--log-level WARNING` で静かに実行
3. **エラー調査**: `--log-level DEBUG` で詳細確認
4. **TensorBoard**: ログを減らしても指標は確認可能

**デフォルトはINFO**なので、引数を省略してもDEBUGログは出ません。
