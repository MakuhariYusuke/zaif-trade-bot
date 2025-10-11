# 100k テスト実行ガイド

**目的**: 1M学習前の動作確認・パラメータ調整  
**所要時間**: 約15-30分（並列実行時）  
**チェックポイント**: 10個（10k毎）

---

## 🎯 100kテストの目的

1. **動作確認**: unified_trainer.py + CustomPPOの統合検証
2. **パラメータ調整**: ent_coef、SELL倍率の効果確認
3. **問題早期発見**: メモリリーク、エラー、バイアスの検出
4. **時間短縮**: 1M学習（数時間）の前に素早く検証

---

## ✅ 準備

### データ確認

```powershell
# データファイル確認
if (Test-Path "ml-dataset-enhanced.csv") {
    echo "✅ Data file exists"
} else {
    echo "❌ Data file not found"
}
```

### 設定ファイル確認

```powershell
# 3つのテスト設定ファイル
dir configs\train\ensemble_*_100k_test.json
```

**期待出力**:
```
ensemble_A_100k_test.json  (Conservative, ent_coef=0.6, SELL=0.8)
ensemble_B_100k_test.json  (Moderate, ent_coef=0.7, SELL=0.9)
ensemble_C_100k_test.json  (Aggressive, ent_coef=0.8, SELL=1.0, reverse=True)
```

---

## 🚀 実行方法

### ログレベル制御（推奨）

**DEBUGログが多すぎて見づらい場合**:

```bash
# INFOレベル（推奨、デフォルト）- 重要な情報のみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO

# WARNINGレベル - 警告とエラーのみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level WARNING

# ERRORレベル - エラーのみ表示
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level ERROR

# DEBUGレベル - すべてのログ表示（デバッグ時のみ）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level DEBUG
```

**利用可能なログレベル**:
- `DEBUG`: すべてのログ（最も詳細、デバッグ時のみ）
- `INFO`: 重要な情報のみ（推奨、デフォルト）
- `WARNING`: 警告とエラーのみ
- `ERROR`: エラーのみ
- `CRITICAL`: 致命的エラーのみ

**Tips**: 
- 通常実行は `--log-level INFO`（デフォルト）で十分
- エラー調査時のみ `--log-level DEBUG` を使用
- 安定稼働時は `--log-level WARNING` で静かに実行

---

### Option 1: シーケンシャル実行（簡単）

```bash
# モデルA（約15-20分）- INFOレベルで実行
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level INFO

# モデルB（約15-20分）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO

# モデルC（約15-20分）
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level INFO
```

**所要時間**: 45-60分（3モデル直列）

---

### Option 2: 並列実行（推奨）

```powershell
# PowerShell - 3つの別ウィンドウで並列実行（INFOレベル）
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json --log-level INFO"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json --log-level INFO"
```

**所要時間**: 15-30分（3モデル並列）

---

### Option 3: 1モデルだけテスト（最速）

```bash
# モデルBだけ実行（最もバランスが良い設定でテスト）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json --log-level INFO
```

**所要時間**: 15-20分

---

## 📊 監視方法

### TensorBoard起動

```bash
# 全モデルをまとめて監視
tensorboard --logdir logs --port 6006
```

**ブラウザ**: http://localhost:6006

### 重要指標（リアルタイム確認）

#### CustomPPO動作確認
- `train/pan_total_samples` → **>0** であればPAN動作中
- `train/entropy_num_updates` → **>0** であればTarget Entropy動作中

#### アクションバイアス確認
- `train/legal_sell_rate` → **目標: ≥0.10**（100kでは低めでもOK）
- `train/action_distribution/BUY` → 約0.4-0.5
- `train/action_distribution/SELL` → **目標: ≥0.05**
- `train/action_distribution/HOLD` → 約0.4-0.5

#### パフォーマンス確認
- `eval/sharpe_proxy` → **>0** が理想（負でも学習中なら許容）
- `eval/max_drawdown` → **>-30%** 程度
- `train/entropy` → **>0.5**（探索が続いているか）

#### 異常検出
- `train/grad_norm(SELL)` → **≠0**（ゼロ張り付きなら問題）
- `train/loss` → 発散していないか
- `train/explained_variance` → **>0** が理想

---

## 🔍 チェックポイント確認

### 保存先

```
checkpoints\ensemble_A_100k_test\
├── checkpoint_10000\
├── checkpoint_20000\
├── ...
└── checkpoint_100000\
```

**チェックポイント数**: 10個（10k毎）

### 評価方法

各チェックポイントのTensorBoard指標を比較:

1. **最良モデル選択**: 最終チェックポイントが最良とは限らない
2. **過学習検出**: `eval/sharpe_proxy`が途中から悪化していないか
3. **バイアス推移**: `legal_sell_rate`が改善しているか

---

## ✅ 成功基準（100kテスト）

### 必須条件（これが満たされないと1M学習NGの判断）

- ✅ **学習完了**: エラーなく100kステップ完了
- ✅ **CustomPPO動作**: `pan_total_samples > 0` かつ `entropy_num_updates > 0`
- ✅ **SELL発生**: `legal_sell_rate ≥ 0.05`（5%以上）
- ✅ **勾配正常**: `grad_norm(SELL) ≠ 0`（ゼロ張り付きなし）

### 推奨条件（これが満たされると1M学習に期待大）

- ⭐ **SELL率向上**: `legal_sell_rate ≥ 0.10`（10%以上）
- ⭐ **パフォーマンス**: `eval/sharpe_proxy > 0`
- ⭐ **探索継続**: `train/entropy > 0.5`
- ⭐ **多様性確保**: 3モデルで異なる行動分布

---

## 🔧 トラブルシューティング

### SELL率が低い（<5%）

**原因**:
- SELL報酬倍率が低すぎる
- ent_coefが低すぎて探索不足

**対策**:
```json
// configs/train/ensemble_A_100k_test.json
{
  "reward_profit_bonus_multipliers": [1.0, 1.0, 0.9],  // 0.8 → 0.9
  "training": {
    "ent_coef": 0.7  // 0.6 → 0.7
  }
}
```

**再テスト**: 設定変更後、もう一度100k実行

---

### grad_norm(SELL)がゼロ張り付き

**原因**:
- SELL勾配が消失している
- データ不均衡が激しい

**対策**:
```json
// Stratified Samplingを有効化
{
  "custom_ppo": {
    "enable_stratified_sampling": true  // false → true
  }
}
```

---

### メモリ不足

**対策**:
```json
// バッチサイズ削減
{
  "training": {
    "n_steps": 1024,  // 2048 → 1024
    "batch_size": 32  // 64 → 32
  }
}
```

---

### 学習が遅い

**対策**:
- GPUが無効になっているか確認
- `n_steps`を1024に削減
- 並列実行を避けてシーケンシャル実行

---

## 📈 結果分析

### TensorBoard分析

1. **Scalarsタブ**:
   - `train/legal_sell_rate` の推移グラフ
   - `eval/sharpe_proxy` の推移グラフ
   - 3モデルを比較（A/B/C）

2. **Distributionsタブ**:
   - `train/action_distribution` でBUY/SELL/HOLDの分布確認

3. **Histogramsタブ**:
   - `train/grad_norm` でSELL勾配の推移確認

### レポート作成（オプション）

```markdown
# 100kテスト結果レポート

## 実行日: 2025-10-07

### モデルA（Conservative）
- legal_sell_rate: 0.08 (8%)
- sharpe_proxy: -0.15
- entropy（最終）: 0.62
- 判定: ✅ 合格（SELL発生、勾配正常）

### モデルB（Moderate）
- legal_sell_rate: 0.12 (12%)
- sharpe_proxy: 0.05
- entropy（最終）: 0.68
- 判定: ⭐ 優秀（SELL率良好、Sharpe正）

### モデルC（Aggressive）
- legal_sell_rate: 0.15 (15%)
- sharpe_proxy: -0.05
- entropy（最終）: 0.75
- 判定: ⭐ 優秀（SELL率高い、探索継続）

### 総合判定
✅ 1M学習に進む準備完了

### 推奨パラメータ（1M学習用）
- モデルB/Cの設定をベースにする
- ent_coef: 0.7-0.8（探索重視）
- SELL倍率: 0.9-1.0（バイアス対策）
```

---

## 🎉 次のステップ

### 100kテスト成功時

```bash
# 1M学習開始（並列実行）
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_1M.json
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_1M.json
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_1M.json
```

### 100kテスト失敗時

1. トラブルシューティングセクションを参照
2. パラメータ調整
3. 再度100kテスト
4. 成功するまで繰り返し

---

## 📚 参考ドキュメント

- `QUICKSTART_1M_ENSEMBLE.md` - 1M学習ガイド
- `CHECKPOINT_INTERVAL_EXTENSION.md` - チェックポイント詳細
- `CODE_QUALITY_IMPROVEMENT_PLAN.md` - 型安全性・保守性向上計画

---

**所要時間**: 15-30分  
**推奨**: モデルB（Moderate）から開始  
**次の目標**: 1M学習で儲かるモデルを探す 🚀
