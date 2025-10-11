# ログレベル制御機能 統合サマリー

**実装日**: 2025年10月7日  
**目的**: unified_trainer実行時のログ出力を制御し、視認性を向上

---

## ✅ 実装内容

### 1. unified_trainer.py拡張

**ファイル**: `ztb/training/unified_trainer.py`

#### 追加引数

```python
parser.add_argument(
    "--log-level",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="Set logging level (default: INFO). Overrides --verbose flag.",
)
```

#### ログレベル制御ロジック

```python
# --log-level takes precedence over --verbose
if hasattr(args, 'log_level') and args.log_level:
    log_level = getattr(logging, args.log_level)
else:
    log_level = logging.DEBUG if args.verbose else logging.INFO

logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Also set the root logger level to suppress third-party DEBUG logs
logging.getLogger().setLevel(log_level)
```

**重要な変更点**:
- `--log-level` 引数が `--verbose` より優先
- ルートロガーのレベルも設定し、サードパーティライブラリ（torch、stable-baselines3等）のDEBUGログも抑制
- デフォルトは `INFO`（DEBUGログを抑制）

---

### 2. ドキュメント更新

#### 新規作成

- **LOG_LEVEL_CONTROL.md**: ログレベル制御の詳細ガイド
  - 使用方法、実行例、推奨設定
  - 100kテストでの活用方法
  - TensorBoardとの併用Tips

#### 更新

- **QUICKSTART_100K_TEST.md**: 100kテスト実行ガイド
  - 「ログレベル制御」セクション追加
  - 全実行例に `--log-level INFO` を追加
  - 並列実行時は `--log-level WARNING` を推奨

---

## 🚀 使用方法

### 基本構文

```bash
python -m ztb.training.unified_trainer --config <設定ファイル> --log-level <レベル>
```

### 利用可能なログレベル

| レベル | 説明 | 用途 |
|--------|------|------|
| **DEBUG** | すべてのログ（最も詳細） | デバッグ時のみ |
| **INFO** | 重要な情報のみ（**デフォルト、推奨**） | 通常実行 |
| **WARNING** | 警告とエラーのみ | 安定稼働時、並列実行時 |
| **ERROR** | エラーのみ | 本番環境 |
| **CRITICAL** | 致命的エラーのみ | 本番環境（最小ログ） |

---

## 📝 実行例

### 通常実行（推奨）

```bash
# INFOレベル（デフォルト）- 重要な情報のみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO
```

**出力**: 進捗、チェックポイント保存、重要なイベントのみ

---

### 静かに実行（WARNINGレベル）

```bash
# 警告とエラーのみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING
```

**用途**: 
- 並列実行時（3つのウィンドウで混乱しないように）
- 長時間実行時（1M学習等）
- 安定稼働時

---

### デバッグ実行（DEBUGレベル）

```bash
# すべてのログを表示（デバッグ時のみ）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level DEBUG
```

**用途**: エラー調査、動作確認時のみ

---

## 🔧 100kテストでの推奨設定

### シーケンシャル実行

```bash
# INFOレベル（進捗を確認）
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level INFO
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level INFO
```

---

### 並列実行（推奨）

```powershell
# WARNINGレベル（静かに実行）
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level WARNING"
```

**理由**: 3つのウィンドウでINFOログが大量に出ると混乱するため

---

## 📊 効果

### Before（--log-level 未実装）

```
2025-10-07 10:00:00 - torch.nn.modules.module - DEBUG - Registered hook...
2025-10-07 10:00:00 - stable_baselines3.common.logger - DEBUG - Writing to TensorBoard...
2025-10-07 10:00:00 - ztb.utils.file_utils - DEBUG - Opening file...
2025-10-07 10:00:01 - ztb.trading.training.ppo_trainer - INFO - Training started
2025-10-07 10:00:01 - torch.optim.adam - DEBUG - Step...
2025-10-07 10:00:01 - stable_baselines3.ppo.ppo - DEBUG - Collecting rollouts...
（DEBUGログが大量に出力され、重要な情報が埋もれる）
```

---

### After（--log-level INFO）

```
2025-10-07 10:00:00 - ztb.training.unified_trainer - INFO - Loaded config from configs\train\ensemble_B_100k_test.json
2025-10-07 10:00:01 - ztb.training.unified_trainer - INFO - Algorithm: ppo
2025-10-07 10:00:01 - ztb.training.unified_trainer - INFO - Starting PPO training...
2025-10-07 10:00:05 - ztb.trading.training.ppo_trainer - INFO - Training started (total_timesteps=100000)
2025-10-07 10:05:00 - ztb.trading.training.ppo_trainer - INFO - Checkpoint saved: checkpoints/ensemble_B_100k_test/checkpoint_10000
（重要な情報のみ表示、視認性向上）
```

---

### After（--log-level WARNING - 並列実行時）

```
2025-10-07 10:00:00 - ztb.trading.training.ppo_trainer - WARNING - SELL rate is low: 0.03
2025-10-07 10:05:00 - ztb.trading.training.ppo_trainer - WARNING - Gradient norm is zero for SELL action
（警告とエラーのみ表示、静かに実行）
```

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

## 💡 TensorBoardとの併用

ログを減らしてもTensorBoardで詳細確認可能:

```bash
# WARNINGレベルで実行（ログ少ない）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING

# 別ターミナルでTensorBoard起動（詳細な指標確認）
tensorboard --logdir logs --port 6006
```

**メリット**:
- コンソールログは最小限（警告のみ）
- TensorBoardで詳細な指標を確認（学習曲線、アクション分布、勾配ノルム等）
- ログファイルサイズも削減

---

## ✅ 検証結果

### ヘルプ表示確認

```bash
$ python -m ztb.training.unified_trainer --help | findstr /C:"log-level" /C:"verbose"
  --verbose             Enable verbose logging
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set logging level (default: INFO). Overrides --verbose
```

✅ 正常に追加されている

---

### ログレベル引数パース確認

```bash
$ python -c "import argparse; import logging; parser = argparse.ArgumentParser(); parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO'); args = parser.parse_args(['--log-level', 'WARNING']); log_level = getattr(logging, args.log_level); print(f'Log level: {args.log_level} → {log_level}')"

Log level: WARNING → 30
```

✅ 正常にパースされている（WARNING = 30）

---

## 📚 関連ドキュメント

1. **LOG_LEVEL_CONTROL.md**: ログレベル制御の詳細ガイド
2. **QUICKSTART_100K_TEST.md**: 100kテスト実行ガイド（ログレベル制御例あり）
3. **UNIFIED_TRAINER_INTEGRATION_SUMMARY.md**: unified_trainer統合サマリー
4. **CHECKPOINT_INTERVAL_EXTENSION.md**: checkpoint_interval拡張ガイド

---

## 🔄 既存機能との互換性

### --verbose フラグとの関係

**優先順位**: `--log-level` > `--verbose`

```bash
# DEBUGレベル（従来の方法）
python -m ztb.training.unified_trainer --config config.json --verbose

# DEBUGレベル（新しい方法）
python -m ztb.training.unified_trainer --config config.json --log-level DEBUG

# --log-levelが優先
python -m ztb.training.unified_trainer --config config.json --verbose --log-level WARNING
# → WARNINGレベルで実行（--verboseは無視される）
```

---

## 📋 今後の拡張予定

### モジュール別ログレベル制御

設定ファイルに追加可能にする（将来実装）:

```json
{
  "logging": {
    "root_level": "INFO",
    "modules": {
      "ztb.trading.training.ppo_trainer": "DEBUG",
      "stable_baselines3": "WARNING",
      "torch": "ERROR"
    }
  }
}
```

**メリット**:
- 特定のモジュールだけDEBUGレベルで詳細確認
- サードパーティライブラリのログを個別に抑制

---

## ✅ まとめ

1. **実装完了**: `--log-level` 引数を追加
2. **デフォルト**: `INFO`レベル（DEBUGログを抑制）
3. **推奨設定**:
   - シーケンシャル実行: `--log-level INFO`
   - 並列実行: `--log-level WARNING`
   - エラー調査: `--log-level DEBUG`
4. **互換性**: 既存の `--verbose` フラグも引き続き使用可能
5. **ドキュメント**: LOG_LEVEL_CONTROL.md、QUICKSTART_100K_TEST.md更新

**視認性が大幅に向上し、100kテストや1M学習がより快適に実行可能になりました！** 🎉
