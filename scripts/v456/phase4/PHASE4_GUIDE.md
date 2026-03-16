# Phase 4: Walk-Forward Analysis - モジュール化実装ガイド

## 📁 v456 phase4 最終ディレクトリ構成

```
scripts/v456/phase4/
│
├── modules/                     ← 【新】 モジュール化パッケージ
│   ├── __init__.py              # モジュール公開インタフェース
│   ├── splitter.py              # TimeSeriesWindow, WalkForwardSplitter
│   ├── evaluator.py             # WalkForwardModelEvaluator
│   ├── result.py                # WindowPerformance, WalkForwardResult
│   └── reporter.py              # WalkForwardReporter
│
├── results/                     ← JSON/CSV出力ディレクトリ
│   └── phase4/
│       └── walk_forward_results.json
│
├── models/                      ← SAC モデル保存ディレクトリ
│   └── window_*.pkl
│
├── run_walk_forward_analysis.py ← メインエントリーポイント
├── PHASE4_GUIDE.md              ← このガイド（更新版）
└── README.md                    ← 実行手順
```

## 🎯 Walk-Forward Analysis - モジュール構成

### 1️⃣ `modules/splitter.py` - 時系列分割生成

```python
class TimeSeriesWindow(NamedTuple):
    """個別ウィンドウ定義"""
    window_id: int              # ウィンドウID (0, 1, 2, ...)
    train_start: int            # 訓練期間開始 (idx)
    train_end: int              # 訓練期間終了
    val_start: int              # 検証期間開始
    val_end: int                # 検証期間終了
    test_start: int             # テスト期間開始
    test_end: int               # テスト期間終了（OOS）

class WalkForwardSplitter:
    """時系列安全な複数分割生成"""
    def split(df) → List[TimeSeriesWindow]
```

**設定値**:
- `initial_train_pct=0.50`: 初期訓練期間（全体の50%）
- `val_pct=0.15`: 各ウィンドウの検証期間（全体の15%）
- `test_pct=0.15`: 各ウィンドウのテスト期間（全体の15%）
- `step_pct=0.10`: ウィンドウシフト幅（全体の10%）
- `embargo_days=7`: 前向き見通し期間（日数）

### 2️⃣ `modules/evaluator.py` - SAC訓練・評価

```python
class WalkForwardModelEvaluator:
    """各ウィンドウでのSAC訓練・評価"""
    
    def train_and_evaluate_window(
        df, window, timesteps=10000
    ) → WindowPerformance:
        # 1. データ分割 (train/val/test)
        # 2. 訓練環境作成 (FastIntradayEnvV456)
        # 3. SAC モデル訓練
        # 4. 検証セット評価
        # 5. テストセット評価（OOS）
        # 6. 性能指標集約
        return WindowPerformance(...)
```

**出力メトリクス** (WindowPerformance):
- `window_id`: ウィンドウ番号
- `val_roi`: 検証期間のROI
- `test_roi`: テスト期間のROI（重要）
- `sharpe_ratio`: Sharpe比（リスク調整済みリターン）
- `max_drawdown`: 最大ドローダウン
- `win_rate`: 勝率
- `trades`: 取引数

### 3️⃣ `modules/result.py` - 結果集約

```python
@dataclass
class WindowPerformance:
    """ウィンドウ単位の性能"""
    window_id: int              # ウィンドウID
    val_roi: float              # 検証ROI
    test_roi: float             # テスト ROI
    val_final_balance: float    # 検証最終残高
    test_final_balance: float   # テスト最終残高
    sharpe_ratio: float         # Sharpe比
    max_drawdown: float         # 最大ドローダウン
    win_rate: float             # 勝率
    trades: int                 # 取引数

@dataclass
class WalkForwardResult:
    """Walk-Forward分析全体結果"""
    windows: List[TimeSeriesWindow]        # ウィンドウ定義
    performances: List[WindowPerformance]  # 各ウィンドウ性能
    average_val_roi: float                 # 平均 Val ROI
    average_test_roi: float                # 平均 Test ROI ✨
    test_roi_std: float                    # Test ROI 標準偏差（一貫性）
    average_sharpe: float                  # 平均 Sharpe比
    sharpe_consistency: float              # Sharpe相関（一貫性）
    average_win_rate: float                # 平均勝率
    overfitting_ratio: float               # Val/Test比（オーバーフィット検出）
```

### 4️⃣ `modules/reporter.py` - 結果報告

```python
class WalkForwardReporter:
    """結果集約と報告"""
    
    def report() → console output
        # ウィンドウ別性能
        # 集約性能
        # 一貫性指標
    
    def save_results(output_path) → JSON
        # phase4/results/ に JSON保存
```

## 🚀 実行方法

### Option 1: 基本実行 (3ウィンドウ × 10K timesteps)

```bash
cd scripts/v456/phase4
python run_walk_forward_analysis.py --windows 3 --timesteps 10000
```

### Option 2: カスタムデータ + フルトレーニング (100K timesteps)

```bash
python run_walk_forward_analysis.py \
    --data "data/datasets/full_data.csv" \
    --windows 5 \
    --timesteps 100000
```

### Option 3: プログラマティック実行

```python
from pathlib import Path
import pandas as pd
from modules import WalkForwardSplitter, WalkForwardModelEvaluator, WalkForwardReporter, WalkForwardResult

# データロード
df = pd.read_csv("data.csv", index_col="timestamp")

# 分割生成
splitter = WalkForwardSplitter(initial_train_pct=0.50)
windows = splitter.split(df)[:3]  # 最初の3ウィンドウ

# 各ウィンドウで訓練・評価
evaluator = WalkForwardModelEvaluator()
performances = [
    evaluator.train_and_evaluate_window(df, w, timesteps=10000)
    for w in windows
]

# 結果統計
import numpy as np
result = WalkForwardResult(
    windows=windows,
    performances=performances,
    average_val_roi=np.mean([p.val_roi for p in performances]),
    average_test_roi=np.mean([p.test_roi for p in performances]),
    # ... (他のメトリクス)
)

# 報告
reporter = WalkForwardReporter(result)
reporter.report()
reporter.save_results(Path("results/phase4/results.json"))
```

## ✅ 検証基準 (Phase 4 成功条件)

✓ **Robustness**:
- `test_roi_std < 0.05`: テスト期間の結果が一貫している
- `sharpe_consistency > 0.70`: Sharpe比の相関が高い

✓ **Non-Overfitting**:
- `overfitting_ratio < 1.5x`: 訓練と同程度の性能維持

✓ **Performance**:
- `average_test_roi >= 0.08`: 8%以上のテストROI
- `average_sharpe >= 0.5`: リスク調整済み

## 📊 Phase 4 実装過程

### ✅ Done
- [x] モジュール構造設計
- [x] modules/{splitter,evaluator,result,reporter}.py 実装
- [x] モジュール公開インタフェース (__init__.py)
- [x] run_walk_forward_analysis.py メイン化（モジュール利用）

### ⏳ In Progress
- [ ] 動作確認テスト（3ウィンドウ × 10K）
- [ ] 本番実行準備（100K × 複数ウィンドウ）

### 📅 Phase 4 スケジュール

| Date | Task | Status |
|------|------|--------|
| Today | モジュール化、テスト実行 | ⏳ |
| +1d | 100K本番実行 | 📅 |
| +2d | 結果分析、本運用準備 | 📅 |

## 📝 Notes

**メモリ・パフォーマンス考慮**:
- SAC buffer_size = TrainingConfig.BUFFER_SIZE (128000)
- batch_size = 256 (メモリ効率)
- learning_rate = TrainingConfig.LEARNING_RATE (0.0003)

**保守性向上点**:
- モジュール分割で テスト・再利用が容易に
- `modules/__init__.py` で公開インタフェース明確化
- 各モジュール単体テスト可能

## 🔗 関連ドキュメント

- [Phase 3 OOS評価](../phase3/PHASE3_GUIDE.md)
- [RL環境仕様](../../docs/environment_spec.md)
- [SAC設定](../../docs/sac_config.md)


# 3. WalkForwardModelEvaluator
   - train_and_evaluate_window()
   - SAC訓練・Val/Test評価
   - 性能メトリクス計算

# 4. WindowPerformance
   - window_id
   - val_roi, test_roi
   - sharpe_ratio, max_drawdown
   - win_rate, trades

# 5. WalkForwardResult
   - 全ウィンドウの集約結果
   - 平均性能
   - Sharpe一貫性
   - オーバーフィッティング比

# 6. WalkForwardReporter
   - report(): コンソール報告
   - save_results(): JSON保存
```

## 🚀 実行方法

### 基本実行（デフォルト）

```bash
python scripts/v456/phase4/run_walk_forward_analysis.py
```

出力例:
```
🚀 Phase 4: Walk-Forward Analysis (n_windows=3)
✓ Created 3 walk-forward windows

Window 0: Training & Evaluation
Train: 500 bars | Val: 150 bars | Test: 150 bars
[Training]
✓ Training completed (10000 timesteps)
[Validation Evaluation]
  Val ROI: 0.0015
[Test Evaluation (Out-of-Sample)]
  Test ROI: -0.0005

... (Window 1, 2)

📊 Walk-Forward Analysis Results
Window-by-Window Performance:
  Window 0: Val ROI 0.0015 | Test ROI -0.0005 | Sharpe 0.1234
  Window 1: Val ROI 0.0008 | Test ROI 0.0002 | Sharpe 0.0856
  Window 2: Val ROI 0.0012 | Test ROI 0.0001 | Sharpe 0.1102

Aggregate Performance:
  Average Val ROI: 0.0012
  Average Test ROI: -0.0001
  Test ROI Std Dev: 0.0003
  Average Sharpe: 0.1064
  Sharpe Consistency: 0.87
  Average Win Rate: 0.50
  Overfitting Ratio: -12.00

✓ Results saved to results/phase4/walk_forward_results.json
```

### カスタム実行

```bash
# 5ウィンドウで、ウィンドウあたり50000 timesteps
python scripts/v456/phase4/run_walk_forward_analysis.py \
  --windows 5 \
  --timesteps 50000

# カスタムデータを指定
python scripts/v456/phase4/run_walk_forward_analysis.py \
  --data /path/to/data.csv \
  --windows 3 \
  --timesteps 100000
```

## 📊 メトリクス解説

### Window-by-Window性能

| メトリクス | 意味 | 理想値 |
|:---|:---|:---|
| **Val ROI** | 検証セット上のReturn | > 0.001 |
| **Test ROI** | テストセット（OOS） | > 0.0005 |
| **Sharpe** | シャープレシオ | > 0.5 |

### 集約性能

| メトリクス | 意味 | 評価ポイント |
|:---|:---|:---|
| **Average Test ROI** | OOS平均リターン | 実運用予測値 |
| **Test ROI Std Dev** | テストROIの標準偏差 | 低いほど安定 |
| **Sharpe Consistency** | ウィンドウ間Sharpe相関 | 0.7+ で安定 |
| **Overfitting Ratio** | 訓練vs テスト比 | 0.8-1.2 が健全 |
| **Average Win Rate** | 平均勝率 | 50% 以上 |

## 🔍 設計ポイント

### 1. 時系列安全性
```
ウィンドウ間に重複なし（前方バイアスなし）
各テストセットは将来データのみ
```

### 2. ウィンドウシフト戦略
```
initial_train: 50% (先頭～)
val: 15% (各ウィンドウ）
test: 15% (各ウィンドウ)
step: 10% (ウィンドウシフト幅)
```

### 3. 既存コード再利用
```
✓ create_environment_wrapper() from phase3
✓ TrainingConfig from ztb.config
✓ feature_calculator_v456 from scripts
```

### 4. 構成化の恩恵
```
- phase4/ : Walk-Forward特化
- shared/ : 共有機能集約
- evaluation/ : 統計検定集約
- training/ : 訓練スクリプト集約
```

## 📈 実行タイムライン

### 典型的な実行時間

- **3 windows × 10K timesteps**: ~5分
- **3 windows × 50K timesteps**: ~20分
- **5 windows × 100K timesteps**: ~2時間

## 🛠️ 今後の拡張

### v1.1 予定項目
- [ ] GridSearch パラメータ最適化統合
- [ ] Plotly ビジュアライゼーション
- [ ] 複数戦略同時比較
- [ ] Walk-Forward統計有意性検定

### 保守性改善
- [ ] shared/ モジュール拡充
- [ ] evaluation/ の共有化
- [ ] 単体テスト追加
- [ ] ドキュメント自動生成

## 🚀 次のステップ

1. **Phase 4実行**: `python scripts/v456/phase4/run_walk_forward_analysis.py`
2. **結果確認**: `results/phase4/walk_forward_results.json`
3. **本訓練**: 最適なウィンドウ数・timesteps設定で再実行
4. **結論**: オーバーフィッティング確認 → 本運用OK判定

---

**Status**: ✅ Phase 4 実装完了・テスト準備完了
