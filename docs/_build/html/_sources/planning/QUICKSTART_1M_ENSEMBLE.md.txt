# 1M学習アンサンブル - クイックスタートガイド

**CustomPPO横展開完了記念: すぐに始められる1M学習アンサンブル**

> **重要**: このガイドは既存の`unified_trainer.py`システムと完全に統合されています。
> アンサンブル学習は`ppo`アルゴリズムとして実行され、既存のワークフローに従います。

---

## 🎯 目標

- **最終目標**: 儲かるモデルを探す
- **中間目標**: 反復学習に最適な設定を見つける
- **短期実行**: 1M学習×3モデルのアンサンブルを回す

---

## ✅ 前提条件チェック

```bash
# データファイル確認
dir ml-dataset-enhanced.csv

# CustomPPO動作確認
python -c "from ztb.training.custom_ppo import CustomPPO; print('✅ CustomPPO OK')"

# 設定ファイル確認
dir configs\train\ensemble_*_1M.json

# unified_trainer.py確認
python -m ztb.training.unified_trainer --help
```

**期待結果**: 全て✅

---

## 🏗️ システム構成

### unified_trainer.pyとの統合

このアンサンブル学習は、既存の`unified_trainer.py`の**PPOアルゴリズム**として実装されています:

- **algorithm: "ppo"** - 標準PPO学習（CustomPPO統合済み）
- **algorithm: "ensemble"** - 既存モデル読み込み専用（今回は未使用）

3つのモデル(A/B/C)はそれぞれ独立したPPO学習として実行され、後でアンサンブル集計します。

---

## 🚀 実行方法（3パターン）

### Option 1: 全モデルシーケンシャル実行（簡単）

```bash
python scripts\run_1m_ensemble.py
```

**所要時間**: 3-5時間 × 3 = 9-15時間  
**リソース**: CPU/メモリ集中
**メリット**: 設定不要、完全自動

### Option 2: 並列実行（推奨）

```powershell
# PowerShell - 3つの別ウィンドウで実行
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts\run_1m_ensemble.py --model A"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts\run_1m_ensemble.py --model B"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts\run_1m_ensemble.py --model C"
```

または手動で3つのターミナルを開いて:

```bash
# ターミナル1
python scripts\run_1m_ensemble.py --model A

# ターミナル2
python scripts\run_1m_ensemble.py --model B

# ターミナル3
python scripts\run_1m_ensemble.py --model C
```

**所要時間**: 3-5時間（並列）  
**リソース**: 3倍のCPU/メモリ
**メリット**: 時間短縮、進捗確認しやすい

### Option 3: unified_trainer.py直接実行（デバッグ用）

```bash
# モデルA
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_1M.json

# モデルB  
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_1M.json

# モデルC
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_1M.json
```

**メリット**: 既存システムとの完全互換性、詳細なログ出力

---

## 📊 モデル構成（多様化軸）

| モデル | ent_coef | SELL倍率 | allow_reverse | seed | 特徴 |
|--------|----------|---------|---------------|------|------|
| **A** | 0.6 (控えめ) | 0.8 | ❌ | 101 | Conservative |
| **B** | 0.7 (標準) | 0.9 | ❌ | 202 | Moderate |
| **C** | 0.8 (積極) | 1.0 | ✅ | 303 | Aggressive |

**共通設定**:
- **CustomPPO**: PAN=True, Target Entropy=True
- **total_timesteps**: 1,000,000
- **checkpoint_interval**: 25,000 (40回のチェックポイント)
- **n_steps**: 2,048
- **batch_size**: 64
- **学習率**: 3e-4

**多様化の意図**:
- `ent_coef`の違い → 探索の保守性/積極性
- `SELL倍率`の違い → アクションバイアス対策
- `allow_reverse`の違い → 戦略の多様性
- `seed`の違い → 初期化の多様性

---

## 📈 監視方法

### TensorBoard（リアルタイム）

```bash
# 全モデルをまとめて監視
tensorboard --logdir logs --port 6006

# または個別に
tensorboard --logdir logs\ensemble_A_1M:A,logs\ensemble_B_1M:B,logs\ensemble_C_1M:C
```

**ブラウザ**: <http://localhost:6006>

### 重要指標

**必須チェック**:

- `train/legal_sell_rate` → 目標: ≥0.15
- `train/pan_total_samples` → 動作確認: >0
- `train/entropy_num_updates` → 動作確認: >0
- `train/grad_norm(SELL)` → ゼロ張り付き回避

**パフォーマンス**:

- `eval/sharpe_proxy` → 目標: >0
- `eval/max_drawdown` → 目標: >-20%
- `eval/trades_per_1k` → 確認用

---

## 🔍 トラブルシューティング

### SELL率が0%のまま

```bash
# 原因確認
findstr "legal_sell_rate" logs\ensemble_A_1M\*\events.out.tfevents.*

# 対策: SELL倍率を上げる
# configs\train\ensemble_A_1M.json の
# "reward_profit_bonus_multipliers": [1.0, 1.0, 0.8]
# を [1.0, 1.0, 0.9] に変更して再実行
```

### メモリ不足

```json
// configs/*.json で調整
"training": {
  "n_steps": 1024,      // 2048 → 1024
  "batch_size": 32      // 64 → 32
}
```

### プレフライトエラー

```bash
# スキップして実行（非推奨）
python scripts\run_1m_ensemble.py --skip-preflight
```

---

## 📦 出力ファイル

### チェックポイント（25k毎）

```text
checkpoints\ensemble_A_1M\
├── checkpoint_25000\
├── checkpoint_50000\
├── checkpoint_75000\
├── ...
├── checkpoint_975000\
└── checkpoint_1000000\
```

**チェックポイント数**: 40個（25k間隔）  
**ディスク容量**: 約400MB-2GB  
**用途**: 早期停止、モデル選択、過学習回避

### ログ

```text
logs\ensemble_A_1M\
└── [session_id]\
    └── events.out.tfevents.*
```

### 最終モデル

```text
models\ensemble_A_1M\
├── ensemble_A_1M_custom_ppo.zip
├── feature_schema.json
└── scaler_params.npz
```

---

## 🎉 完了後の次のステップ

### 1. アンサンブル集計（実装予定）

```bash
python scripts\ensemble_aggregation.py ^
  --models ^
    models\ensemble_A_1M\final ^
    models\ensemble_B_1M\final ^
    models\ensemble_C_1M\final ^
  --weights confidence ^
  --output artifacts\ensemble_final_eval.json
```

### 2. バックテスト

```bash
python backtest_model.py ^
  --model models\ensemble_A_1M\ensemble_A_1M_custom_ppo.zip ^
  --data btc_jpy_real_dataset.csv
```

### 3. 本番デプロイ（慎重に）

```bash
# まずペーパートレード
python ztb\trading\paper_trade.py ^
  --model models\ensemble_A_1M\ensemble_A_1M_custom_ppo.zip ^
  --steps 1000
```

---

## 📚 詳細ドキュメント

- `1M_ENSEMBLE_OPERATIONS_MANUAL.md`: 完全運用マニュアル
- `CUSTOM_PPO_ROLLOUT_REPORT.md`: CustomPPO横展開詳細
- `FINAL_ROLLOUT_AND_ROADMAP.md`: ロードマップ
- `ztb/training/unified_trainer.py`: 統合トレーナーのソースコード

---

## 🔧 既存システムとの統合

### unified_trainer.pyのアルゴリズム

```python
# 利用可能なアルゴリズム
algorithms = [
    "ppo",         # ← 今回使用（CustomPPO統合済み）
    "base_ml",     # 基本ML強化学習
    "iterative",   # 反復学習（run_1m.py）
    "ensemble",    # 既存モデル読み込み専用
    "curriculum"   # カリキュラム学習
]
```

### 設定ファイルの互換性

既存の`unified_trainer.py`設定と完全互換:

```json
{
  "algorithm": "ppo",           // 必須
  "data_path": "...",          // 必須
  "session_id": "...",         // 必須
  "total_timesteps": 1000000,  // 必須
  "checkpoint_interval": 25000,// チェックポイント間隔（オプション、デフォルト: 25000）
  "training": { ... },         // PPOConfig互換
  "checkpoint_dir": "...",     // unified_trainer.py形式
  "log_dir": "...",           // unified_trainer.py形式
  "model_dir": "..."          // unified_trainer.py形式
}
```

**checkpoint_intervalの推奨値**:
- **1M学習**: 25000（40回のチェックポイント）
- **500k学習**: 25000（20回のチェックポイント）
- **100k学習**: 10000（10回のチェックポイント）
- **カスタム**: `total_timesteps / checkpoint_interval` = 10-50回を目安

---

## 🤝 サポート

**問題が発生したら**:

1. ログ確認: `logs\ensemble_*_1M\`
2. エラーメッセージをコピー
3. 該当する設定ファイル確認
4. TensorBoardで指標確認

**よくある質問**:

Q: 途中で中断した場合は?  
A: チェックポイントから再開可能（将来対応予定）

Q: 3モデル全部必要?  
A: 1モデルでも可。アンサンブルは2+推奨

Q: GPUは必要?  
A: 不要。CPUで動作（ただし遅い）

---

**作成日**: 2025年10月7日  
**ステータス**: unified_trainer.py統合完了 ✅  
**推奨実行方法**: `python scripts\run_1m_ensemble.py --model A` (並列実行)
