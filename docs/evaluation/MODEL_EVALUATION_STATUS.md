# モデル評価レポート - 検証状況

**作成日**: 2025年10月10日
**目的**: 実取引で儲かるモデルを特定する
**ステータス**: ⚠️ **検証中断（環境問題）**

---

## 📋 検証計画

### 対象モデル
1. **v385** (ppo_reward_v385_curated.zip) - 最新・68特徴量
2. **v384** (ppo_reward_v384_curated_60.zip) - 60k steps・68特徴量
3. **v381** (ppo_reward_v381_revised_profit_focused.zip) - ベースライン

### 評価指標
- 📈 総収益率 (Total Return %)
- 📊 シャープレシオ (Sharpe Ratio)
- 📉 最大ドローダウン (Max Drawdown)
- 🎯 勝率 (Win Rate)
- 💰 利益率 (Profit Factor)
- 🔄 取引回数 (Total Trades)

---

## ⚠️ 検証で発生した問題

### 問題1: paper_trade.py モデル読み込みエラー

**症状**:
```
File "ztb/training/scripts/paper_trade.py", line 187, in _load_model
    self.model = MaskablePPO(...)
...
KeyboardInterrupt (during PyTorch import)
```

**原因**: Python 3.13環境でPyTorchインポート時にクラッシュ
**影響**: paper_trade.pyが使用不可

### 問題2: backtest_with_schema.py ログ出力エラー

**症状**:
```
File "ztb/training/policies/policy_utils.py", line 74, in predict_with_masks
    logger.debug(f"MaskablePPO prediction: action={action}, masks={action_masks}")
...
KeyboardInterrupt (during numpy array formatting)
```

**原因**: ログ出力時のnumpy配列フォーマットでクラッシュ
**影響**: backtest_with_schema.pyが使用不可

### 問題3: live_trade.py 実行時エラー

**症状**:
```
File "live_trade.py", line 438, in _load_model
    model = MaskablePPO.load(str(self.model_path))
...
KeyboardInterrupt (during torch._dynamo import)
```

**原因**: PyTorch動的コンパイルモジュールのインポートエラー
**影響**: live_trade.pyが使用不可

### 問題4: 既存バックテスト結果の問題

**v384バックテスト結果** (`backtest_v384_20251010_164125.json`):
```json
{
  "avg_reward": -996.8,
  "total_pnl": 0.0,
  "action_distribution": {
    "HOLD": {"count": 19980, "pct": 100.0},
    "BUY": {"count": 0, "pct": 0.0},
    "SELL": {"count": 0, "pct": 0.0}
  },
  "total_trades": 0
}
```

**問題点**: HOLD 100%、取引なし
**原因**: 環境設定orモデルの問題（要調査）

---

## 🔧 環境問題の根本原因

### Python 3.13 + PyTorch 互換性問題

**スタックトレースのパターン**:
```
import torch._dynamo
  → import sympy
    → KeyboardInterrupt (インポート中断)
```

**推測される原因**:
1. Python 3.13はまだPyTorch/SymPyで完全サポートされていない
2. Windows環境でのインポートパフォーマンス問題
3. 環境変数の設定不足

### numpy.typing問題

**修正済み**:
```python
# ❌ 以前
from numpy.typing import NDArray, _32Bit

# ✅ 修正後
from numpy.typing import NDArray
```

---

## 🚨 即時対応が必要な項目

### 優先度HIGH

1. **Python環境の見直し**
   - Python 3.11への降格検討
   - PyTorch再インストール
   - 環境変数の確認（PYTORCH_*, CUDA_*）

2. **ログレベルの調整**
   - DEBUG → INFO/WARNING
   - numpy配列のlog出力を削除

3. **代替評価方法**
   - モデルのmetrics.json確認（トレーニング時の性能）
   - TensorBoardログの分析
   - 過去の評価結果の精査

### 優先度MEDIUM

4. **軽量バックテストスクリプト作成**
   - ログ出力最小限
   - 直接的な環境作成（schema_env_factoryを経由しない）
   - エラーハンドリング強化

5. **モデル比較分析**
   - トレーニングログからの性能抽出
   - config比較による特性理解

---

## 📊 代替アプローチ: トレーニングログ分析

### トレーニング時の性能指標

モデルのトレーニング時の性能を確認することで、実取引での挙動を予測できます。

**確認すべきファイル**:
```
models/
├── ppo_reward_v385_curated/
│   └── (TensorBoard logs or metrics)
├── ppo_reward_v384_curated_60/
│   └── (TensorBoard logs or metrics)
└── ppo_reward_v381_revised_profit_focused/
    └── (TensorBoard logs or metrics)
```

**確認すべき指標**:
- Episode Reward (平均・最大・最小)
- Action Distribution (HOLD/BUY/SELL比率)
- Policy Loss推移
- Value Loss推移
- Entropy推移（探索度合い）

---

## 🎯 推奨される次のステップ

### ステップ1: 環境修復（最優先）

```bash
# Python 3.11環境の作成を検討
py -3.11 -m venv .venv311

# または現在の環境でPyTorch再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ステップ2: 軽量テストスクリプト

```python
# minimal_test.py
import sys
sys.path.insert(0, ".")

from sb3_contrib import MaskablePPO

# モデル読み込みテスト（環境なし）
try:
    model = MaskablePPO.load("models/ppo_reward_v385_curated.zip", env=None)
    print("✅ Model loaded successfully")
    print(f"Observation space: {model.observation_space}")
    print(f"Action space: {model.action_space}")
except Exception as e:
    print(f"❌ Failed: {e}")
```

### ステップ3: TensorBoard確認

```bash
# トレーニングログの確認
tensorboard --logdir logs/
```

### ステップ4: 手動評価

もし自動バックテストが機能しない場合：
1. 各モデルのトレーニングログを比較
2. config差分を分析
3. 過去の開発ログ・コミットメッセージを確認
4. 最も安定したパフォーマンスを示したモデルを選択

---

## 💡 Phase 5機能の優先度再評価

元々提案したPhase 5機能：
1. ~~特徴量順序検証~~ → 既存分析ツールあり（優先度LOW）
2. **スキーマDrift検出** → 優先度LOW（まず動くシステムが必要）
3. **マルチモデル管理** → 優先度MEDIUM（環境修復後）
4. **スキーマバージョニング** → 優先度LOW（将来的に）

**現在の最優先事項**:
✅ **環境修復** → まず1つのモデルでバックテストを成功させる

---

## 📝 結論

現時点では、環境問題により直接的なモデル評価（バックテスト・live trade dry-run）が実行できない状況です。

**即座に取るべきアクション**:
1. Python環境の見直し（3.11へのダウングレード検討）
2. PyTorch再インストール
3. 軽量テストスクリプトで段階的に検証
4. トレーニングログからの間接的評価

**最終目標**:
- ✅ v385, v384, v381の3モデルでバックテスト成功
- ✅ 最も収益性の高いモデルを特定
- ✅ 実取引での検証開始

**現状**: ⚠️ **環境問題により検証中断**
**次のステップ**: 🔧 **環境修復 → 軽量テスト → 段階的評価**
