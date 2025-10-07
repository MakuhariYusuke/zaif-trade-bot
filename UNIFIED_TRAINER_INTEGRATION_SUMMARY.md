# unified_trainer.py統合サマリー

**日付**: 2025年10月7日  
**目的**: 1M学習アンサンブルをunified_trainer.pyに統合し、既存システムとの整合性を確保

---

## ✅ 完了事項

### 1. 設定ファイル修正（3種）

**変更内容**:
- `algorithm`: "ensemble" → **"ppo"**（統合トレーナーのPPOアルゴリズムとして実行）
- `output`セクション削除 → `checkpoint_dir`/`log_dir`/`model_dir`をトップレベルに移動

**検証結果**:
```
✅ ensemble_A_1M.json: Valid
   algorithm: ppo
   total_timesteps: 1000000
   ent_coef: 0.6
   checkpoint_dir: checkpoints/ensemble_A_1M

✅ ensemble_B_1M.json: Valid
   ent_coef: 0.7 | SELL倍率: 0.9

✅ ensemble_C_1M.json: Valid
   ent_coef: 0.8 | SELL倍率: 1.0 | allow_reverse: True
```

### 2. run_1m_ensemble.py確認

**既存実装**:
- ✅ unified_trainer.py呼び出し（`python -m ztb.training.unified_trainer`）
- ✅ プレフライトチェック統合
- ✅ 並列/シーケンシャル実行対応

**変更不要**: 既に正しい構造で実装済み

### 3. QUICKSTART_1M_ENSEMBLE.md更新

**追加内容**:
- unified_trainer.pyとの統合説明
- Windows環境対応コマンド（PowerShell並列実行）
- 既存システムとの互換性セクション
- 設定ファイル構造の説明

---

## 🏗️ システム統合図

```
ユーザー実行
    ↓
scripts/run_1m_ensemble.py
    ↓
    ├─ プレフライトチェック (preflight_schema_scaler_check.py)
    ↓
    └─ unified_trainer.py --config ensemble_A_1M.json
           ↓
           algorithm='ppo' 判定
           ↓
           PPOTrainerAutoHalt (または SELLBiasMitigationPPOTrainer)
           ↓
           CustomPPO (PAN + Target Entropy統合)
           ↓
           学習実行 (1M timesteps)
           ↓
           チェックポイント保存 (25k毎)
           ↓
           最終モデル保存 (models/ensemble_A_1M/)
```

---

## 📋 unified_trainer.pyのアルゴリズム一覧

| Algorithm | 用途 | 今回の使用 |
|-----------|------|-----------|
| **ppo** | 標準PPO学習（CustomPPO統合） | ✅ **使用中** |
| base_ml | 基本ML強化学習 | ❌ |
| iterative | 反復学習（run_1m.py） | ❌ |
| ensemble | 既存モデル読み込み専用 | ❌ |
| curriculum | カリキュラム学習 | ❌ |

---

## 🎯 多様化軸の確認

| 軸 | Model A | Model B | Model C | 効果 |
|----|---------|---------|---------|------|
| **ent_coef** | 0.6 | 0.7 | 0.8 | 探索の保守性/積極性 |
| **SELL倍率** | 0.8 | 0.9 | 1.0 | アクションバイアス対策 |
| **allow_reverse** | False | False | **True** | 戦略の多様性 |
| **seed** | 101 | 202 | 303 | 初期化の多様性 |

**共通設定**:
- CustomPPO: `enable_pan=true`, `enable_target_entropy=true`
- n_steps: 2048
- batch_size: 64
- total_timesteps: 1,000,000

---

## 🚀 実行コマンド

### 推奨: 並列実行（Windows PowerShell）

```powershell
# 3つの別ウィンドウで並列実行
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts\run_1m_ensemble.py --model A"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts\run_1m_ensemble.py --model B"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts\run_1m_ensemble.py --model C"
```

### または: シーケンシャル実行

```bash
python scripts\run_1m_ensemble.py
```

### または: 個別実行（デバッグ用）

```bash
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_1M.json
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_1M.json
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_1M.json
```

---

## 📊 監視方法

### TensorBoard

```bash
tensorboard --logdir logs --port 6006
```

### 重要指標

**CustomPPO動作確認**:
- `train/pan_total_samples` > 0
- `train/entropy_num_updates` > 0

**アクションバイアス対策**:
- `train/legal_sell_rate` ≥ 0.15
- `train/grad_norm(SELL)` ≠ 0（ゼロ張り付き回避）

**パフォーマンス**:
- `eval/sharpe_proxy` > 0
- `eval/max_drawdown` > -20%

---

## 🔄 既存システムとの整合性

### unified_trainer.pyとの完全互換性

**設定ファイル構造**:
```json
{
  "algorithm": "ppo",           // ← unified_trainer.py準拠
  "data_path": "...",          // ← 必須
  "session_id": "...",         // ← 必須
  "total_timesteps": 1000000,  // ← 必須
  "training": { ... },         // ← PPOConfig互換
  "checkpoint_dir": "...",     // ← unified_trainer.py形式
  "log_dir": "...",           // ← unified_trainer.py形式
  "model_dir": "..."          // ← unified_trainer.py形式
}
```

**PPOTrainerAutoHalt経由でCustomPPO適用**:
- `ppo_trainer.py`: PPOTrainerAutoHalt（CustomPPO統合済み）
- `unified_trainer.py`: PPOTrainerAutoHalt呼び出し
- **自動的に**: PAN + Target Entropyが適用される

---

## 📦 出力構造

```
checkpoints/ensemble_A_1M/
├── checkpoint_25000/
├── checkpoint_50000/
└── ...

logs/ensemble_A_1M/
└── [session_id]/
    └── events.out.tfevents.*

models/ensemble_A_1M/
├── ensemble_A_1M_custom_ppo.zip
├── feature_schema.json
└── scaler_params.npz
```

---

## 🎉 次のステップ

### 即座に実行可能

```bash
# 1. 並列実行開始（推奨）
python scripts\run_1m_ensemble.py --model A
python scripts\run_1m_ensemble.py --model B
python scripts\run_1m_ensemble.py --model C

# 2. TensorBoard監視
tensorboard --logdir logs
```

### 将来の実装（オプション）

1. **τ×Tスイープツール** - パラメータ探索自動化
2. **ローリング評価自動化** - 25k毎の自動評価
3. **confidence-weighted集計** - アンサンブル集計の高度化
4. **ステージング実装** - 段階的パラメータ変更

---

## 🔒 整合性チェックリスト

- ✅ 設定ファイル: algorithm='ppo'
- ✅ 設定ファイル: 出力パス構造（checkpoint_dir/log_dir/model_dir）
- ✅ run_1m_ensemble.py: unified_trainer.py呼び出し
- ✅ CustomPPO: PPOTrainerAutoHalt経由で自動適用
- ✅ プレフライトチェック: 既存スクリプト利用
- ✅ ドキュメント: Windows環境対応
- ✅ 多様化軸: ent_coef/SELL倍率/seed/reverse

---

**結論**: 1M学習アンサンブルは既存の`unified_trainer.py`システムに完全統合され、即座に実行可能な状態です。

**推奨アクション**: `python scripts\run_1m_ensemble.py --model A` (並列実行)
