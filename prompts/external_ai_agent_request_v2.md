# 外部AIエージェントへの依頼プロンプト v2

## 問題の概要

Windowsシステム上でPython深層学習トレーニングを実行すると、**ユーザーがキーボード操作を一切していないにも関わらず**、`KeyboardInterrupt`または"Training interrupted by user"エラーが発生して実行が中断されます。

## 既に実施した対策（効果が不安定）

1. **scipy/sklearn/shapの遅延インポート**
   - `SKIP_HEAVY_IMPORTS=1`、`ZTB_SKIP_SCIPY=1`、`ZTB_SKIP_SKLEARN=1`環境変数設定済み
   - `ztb/metrics/metrics.py`: scipy conditional import実装
   - `ztb/adaptation/explainability/__init__.py`: TYPE_CHECKING lazy import実装
   - `ztb/adaptation/safety/anomaly_*.py`: sklearn optional imports実装

2. **テスト結果**
   - `test_debug_interrupt.py` (1000 steps): **成功** ✅
   - `run_ab_reward_experiments.py` (5000 steps, 1回目): **成功** ✅
   - `run_ab_reward_experiments.py` (5000 steps, 2回目以降): **失敗** ❌ "Training interrupted by user"

## 症状の詳細

### 失敗時の出力
```
01/26/2026 14:04:24:WARNING:Training interrupted by user
❌ Training failed
ValueError: Training failed (success=False)
```

### 成功時との違い
- 同じスクリプト、同じ設定で、1回目は成功、2回目以降は即座に失敗
- メモリ使用量: 成功時は1.7GB安定、失敗時は即時終了のため不明
- プロセス継続性: 何らかのリソース・ロック・状態が原因と推測

## システム情報

- **OS**: Windows 11
- **Python**: 3.11.9 (virtual environment)
- **主要ライブラリ**: PyTorch, Stable-Baselines3, NumPy, Pandas
- **問題の発生場所**: `ztb.training.unified_trainer.trainer.UnifiedTrainer.train()`内

## プロジェクト構造

```
zaif-trade-bot/
├── ztb/
│   ├── training/
│   │   └── unified_trainer/
│   │       └── trainer.py  # メインのトレーニングロジック
│   ├── adaptation/
│   │   ├── explainability/__init__.py  # 遅延インポート対応済み
│   │   ├── safety/anomaly_*.py  # sklearn optional対応済み
│   │   └── monitoring/evaluation_manager.py  # 対応済み
│   └── metrics/
│       └── metrics.py  # scipy optional対応済み
├── scripts/v459/
│   ├── run_ab_reward_experiments.py  # メイン実験スクリプト
│   └── test_debug_interrupt.py  # デバッグ用（成功例）
└── data/
    └── btc_jpy_1m_v451.csv  # 149,487行の時系列データ
```

## 依頼内容

以下の点について調査・修正案を提案してください：

### 1. **根本原因の特定**
   - なぜ同じスクリプトで1回目は成功、2回目以降は失敗するのか？
   - Windows固有のリソース管理（ハンドル、ロック、スレッド）に問題がないか？
   - PyTorchのマルチスレッディング設定に問題がないか？

### 2. **安定化対策の提案**
   - プロセス間のリソース解放が不完全な可能性
   - PyTorch worker プロセスのクリーンアップ
   - Windows特有のファイルロック・DLLロード問題
   - シグナルハンドリングの改善

### 3. **具体的な修正箇所の指摘**
   特に以下のファイルを重点的に確認：
   - `ztb/training/unified_trainer/trainer.py` (Line 1-2000): メイントレーニングループ
   - `scripts/v459/run_ab_reward_experiments.py` (Line 80-200): 実験実行ロジック
   - PyTorchのDataLoader/multiprocessing設定

### 4. **Windows環境での推奨設定**
   ```python
   # 以下のような設定が必要か？
   torch.set_num_threads(1)
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   multiprocessing.set_start_method('spawn', force=True)
   ```

## 期待する回答形式

1. **原因分析**: 根本原因の推定（技術的根拠付き）
2. **修正コード**: 具体的なファイル名・行番号・修正内容
3. **環境変数設定**: 追加すべき設定（あれば）
4. **テスト手順**: 修正後の検証方法

## 補足情報

### 成功したテストスクリプト例
`scripts/v459/test_debug_interrupt.py`:
- シグナルハンドラー設定済み
- 環境変数設定済み
- 1000ステップで成功
- 例外トラッキング実装

### 失敗する実験スクリプト
`scripts/v459/run_ab_reward_experiments.py`:
- 同じ環境変数設定
- 5000ステップ実行
- Walk-Forward validation有効
- 1回目は成功、2回目以降失敗

### 重要な制約
- **scipy/sklearn/shapは使用不可**（インポート時にKeyboardInterrupt発生）
- NumPy代替実装は既に適用済み
- メモリ使用量は問題ない（1.7GB程度）
- トレーニング自体のロジックは正常（成功時の出力は完璧）

---

**最優先課題**: 同じコードで繰り返し安定実行できるようにすること

よろしくお願いいたします。
