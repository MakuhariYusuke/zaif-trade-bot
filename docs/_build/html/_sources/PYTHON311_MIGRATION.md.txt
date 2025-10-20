# Python 3.11環境移行ガイド

**日付**: 2025年10月10日  
**理由**: Python 3.13でPyTorch/SymPy互換性問題が発生  
**対応**: Python 3.11環境へダウングレード

---

## ✅ 実施済み

### 1. 環境作成
```bash
py -3.11 -m venv .venv311
```

### 2. pip アップグレード
```bash
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
```

### 3. パッケージインストール
```bash
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🧪 検証手順

### ステップ1: Python バージョン確認
```bash
.\.venv311\Scripts\python.exe --version
# 期待: Python 3.11.x
```

### ステップ2: 基本インポートテスト
```bash
.\.venv311\Scripts\python.exe -c "import torch; import numpy; import pandas; print('✅ Basic imports OK')"
```

### ステップ3: Stable-Baselines3 テスト
```bash
.\.venv311\Scripts\python.exe -c "from sb3_contrib import MaskablePPO; print('✅ SB3 OK')"
```

### ステップ4: スキーママネージャーテスト
```bash
.\.venv311\Scripts\python.exe -c "from ztb.training.core.feature_schema_manager import FeatureSchemaManager; print('✅ Schema Manager OK')"
```

### ステップ5: モデル読み込みテスト
```bash
.\.venv311\Scripts\python.exe -c "from sb3_contrib import MaskablePPO; m=MaskablePPO.load('models/ppo_reward_v385_curated.zip'); print(f'✅ Model loaded: {m.observation_space}')"
```

### ステップ6: クイックバックテスト（v385）
```bash
.\.venv311\Scripts\python.exe quick_backtest.py --model models/ppo_reward_v385_curated.zip --data ml-dataset-enhanced.csv --episodes 3
```

### ステップ7: クイックバックテスト（v384）
```bash
.\.venv311\Scripts\python.exe quick_backtest.py --model models/ppo_reward_v384_curated_60.zip --data ml-dataset-enhanced.csv --episodes 3
```

### ステップ8: クイックバックテスト（v381）
```bash
.\.venv311\Scripts\python.exe quick_backtest.py --model models/ppo_reward_v381_revised_profit_focused.zip --data ml-dataset-enhanced.csv --episodes 3
```

### ステップ9: Live Trade Dry-Run（勝ちモデル）
```bash
.\.venv311\Scripts\python.exe live_trade.py --model-path models/[BEST_MODEL].zip --duration-hours 0.05 --dry-run
```

---

## 📊 モデル評価計画

### 評価指標
| 指標 | 重要度 | 説明 |
|------|--------|------|
| 平均リターン (%) | ⭐⭐⭐ | 収益性の直接指標 |
| 最大リターン (%) | ⭐⭐ | ベストケース |
| 最悪リターン (%) | ⭐⭐⭐ | リスク評価 |
| トレード数 | ⭐⭐ | 取引頻度 |
| 平均報酬 | ⭐ | 環境報酬 |

### 合格基準
- ✅ 平均リターン > 0%
- ✅ 最悪リターン > -5%
- ✅ トレード数 > 0 (HOLD 100%ではない)

### 比較マトリックス
```
モデル | 平均リターン | 最大 | 最悪 | トレード数 | 判定
-------|-------------|------|------|-----------|-----
v385   | ?           | ?    | ?    | ?         | 検証中
v384   | ?           | ?    | ?    | ?         | 検証中
v381   | ?           | ?    | ?    | ?         | 検証中
```

---

## 🚀 環境切り替え方法

### VS Code設定（推奨）
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. `.venv311\Scripts\python.exe` を選択

### ターミナル直接指定
```bash
# 個別コマンド実行
.\.venv311\Scripts\python.exe your_script.py

# 環境アクティベート（PowerShell）
.\.venv311\Scripts\Activate.ps1
```

### 永続的な切り替え
```bash
# 既存 .venv をリネーム（バックアップ）
ren .venv .venv_old_313

# .venv311 を .venv にリネーム
ren .venv311 .venv
```

---

## ⚠️ 注意事項

### Python 3.13との違い
- ✅ PyTorch/SymPy が安定動作
- ✅ numpy.typing 互換性問題なし
- ⚠️ 一部の新しい言語機能は使えない（type statements等）

### パッケージバージョン
requirements.txt のバージョンピン止めに従う:
- numpy==2.3.3
- scikit-learn==1.7.2
- stable-baselines3>=2.0.0
- PyTorch (CUDA 11.8対応)

---

## 🔄 ロールバック手順

もし3.11でも問題が発生した場合：
```bash
# .venv (旧3.13環境) に戻す
deactivate
ren .venv .venv311_failed
ren .venv_old_313 .venv
```

---

## 📝 次のステップ

1. ✅ インストール完了確認
2. ⏳ 検証ステップ1-5 実行
3. ⏳ quick_backtest.py で3モデル評価
4. ⏳ ベストモデル特定
5. ⏳ live_trade.py で実取引検証

**目標**: 儲かるモデルを見つけて実取引開始！
