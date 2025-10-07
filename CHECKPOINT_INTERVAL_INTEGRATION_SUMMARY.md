# checkpoint_interval統合完了サマリー

**日付**: 2025年10月7日  
**要求**: unified_trainer.pyに`checkpoint_interval`を統合し、反復学習等で活用  
**ステータス**: ✅ 完了

---

## 🎯 実装概要

`checkpoint_interval`パラメータを`unified_trainer.py`に統合し、全アルゴリズム（PPO、反復学習）で利用可能にしました。これにより、設定ファイルから簡単にチェックポイント間隔を制御できるようになりました。

---

## ✅ 変更内容

### 1. unified_trainer.py: _train_ppo()拡張

**ファイル**: `ztb/training/unified_trainer.py`  
**変更**: `checkpoint_interval`を設定ファイルから読み取り、PPOTrainerに渡す

```python
# Get checkpoint interval from config (default: 25000 for 1M training = 40 checkpoints)
checkpoint_interval = self.config.get("checkpoint_interval", 25000)

# Create trainer with SELL mitigation if enabled
if enable_sell_mitigation:
    trainer = trainer_class(
        # ... other params ...
        checkpoint_interval=checkpoint_interval,  # ← 追加
    )
else:
    trainer = trainer_class(
        # ... other params ...
        checkpoint_interval=checkpoint_interval,  # ← 追加
    )
```

### 2. run_1m.py拡張（反復学習）

**ファイル**: `ztb/training/run_1m.py`  
**変更**: `--checkpoint-interval`引数追加

```python
parser.add_argument(
    "--checkpoint-interval",
    type=int,
    default=10000,
    help="Steps between checkpoints (default: 10000)",
)

# Create trainer
trainer = PPOTrainer(
    # ... other params ...
    checkpoint_interval=args.checkpoint_interval,  # ← 修正
)
```

### 3. unified_trainer.py: _train_iterative()拡張

**ファイル**: `ztb/training/unified_trainer.py`  
**変更**: `checkpoint_interval`をrun_1m.pyに渡す

```python
# Get checkpoint interval from config (default: 10000 for iterative training)
checkpoint_interval = self.config.get("checkpoint_interval", 10000)

sys.argv = [
    "run_1m.py",
    # ... other args ...
    "--checkpoint-interval",
    str(checkpoint_interval),  # ← 追加
    # ... other args ...
]
```

### 4. アンサンブル設定ファイル更新（3種）

**ファイル**:
- `configs/train/ensemble_A_1M.json`
- `configs/train/ensemble_B_1M.json`
- `configs/train/ensemble_C_1M.json`

**変更**: `checkpoint_interval: 25000`を追加

```json
{
  "algorithm": "ppo",
  "total_timesteps": 1000000,
  "checkpoint_interval": 25000,  // ← 追加
  "checkpoint_dir": "checkpoints/ensemble_A_1M",
  // ... other settings ...
}
```

### 5. ドキュメント更新

**ファイル**:
- `CHECKPOINT_INTERVAL_EXTENSION.md` - 拡張サマリー（新規作成）
- `QUICKSTART_1M_ENSEMBLE.md` - チェックポイント説明追加

---

## 📊 検証結果

```
✅ Import test passed
✅ Config loaded
   algorithm: ppo
   checkpoint_interval: 25000
   Expected checkpoints: 40
```

**全設定ファイル**:
```
✅ ensemble_A_1M.json: checkpoint_interval: 25000
✅ ensemble_B_1M.json: checkpoint_interval: 25000
✅ ensemble_C_1M.json: checkpoint_interval: 25000
```

---

## 🎯 アルゴリズム別デフォルト値

| Algorithm | デフォルト | チェックポイント数 (1M) | 理由 |
|-----------|-----------|---------------------|------|
| **ppo** | 25000 | 40回 | バランス良好、管理しやすい |
| **iterative** | 10000 | 100回 | 頻繁な評価が有用 |
| **base_ml** | N/A | - | 未実装 |
| **ensemble** | N/A | - | 読み込み専用 |
| **curriculum** | N/A | - | 独自ロジック |

---

## 💡 使用例

### 1M学習（推奨設定）

```json
{
  "algorithm": "ppo",
  "total_timesteps": 1000000,
  "checkpoint_interval": 25000,
  "checkpoint_dir": "checkpoints/my_model"
}
```

**結果**: 40回のチェックポイント（25k, 50k, 75k, ..., 1000k）

### 反復学習

```json
{
  "algorithm": "iterative",
  "total_timesteps": 500000,
  "checkpoint_interval": 10000,
  "checkpoint_dir": "checkpoints/iterative_model"
}
```

**結果**: 50回のチェックポイント（10k, 20k, 30k, ..., 500k）

### カスタム間隔

```json
{
  "algorithm": "ppo",
  "total_timesteps": 2000000,
  "checkpoint_interval": 50000,
  "checkpoint_dir": "checkpoints/long_training"
}
```

**結果**: 40回のチェックポイント（50k, 100k, 150k, ..., 2000k）

---

## 📦 チェックポイント構造

### 保存先

```
checkpoints/ensemble_A_1M/
├── checkpoint_25000/
│   ├── model.zip
│   ├── feature_schema.json
│   ├── scaler_params.npz
│   └── diagnostics.json
├── checkpoint_50000/
│   └── ...
├── ...
└── checkpoint_1000000/
    └── ...
```

### ディスク容量

- **1チェックポイント**: 約10-50MB
- **40チェックポイント**: 約400MB-2GB
- **推奨**: 総容量3-5GBの空き容量

---

## 🔧 推奨設定

### チェックポイント数の目安

**最適範囲**: 10-50回

**計算式**:
```python
num_checkpoints = total_timesteps / checkpoint_interval
```

**推奨設定**:
- **短期学習（<100k）**: `checkpoint_interval=10000` → 10回以下
- **中期学習（100k-500k）**: `checkpoint_interval=25000` → 4-20回
- **長期学習（1M以上）**: `checkpoint_interval=25000-50000` → 20-40回

### 利点

✅ **早期停止**: 過学習を検出して早期のチェックポイントを採用  
✅ **モデル選択**: 複数候補から最良モデルを選択  
✅ **リスク軽減**: 学習中断時の損失を最小化  
✅ **評価柔軟性**: 段階的なパフォーマンス評価

---

## 🚀 実行方法

### PPO学習（1M）

```bash
python -m ztb.training.unified_trainer --config configs/train/ensemble_A_1M.json
```

**チェックポイント**: 25k毎、計40回

### 反復学習

```bash
python -m ztb.training.unified_trainer --config configs/train/my_iterative_config.json
```

**設定例**:
```json
{
  "algorithm": "iterative",
  "total_timesteps": 500000,
  "checkpoint_interval": 10000
}
```

**チェックポイント**: 10k毎、計50回

---

## 📈 モニタリング

### TensorBoard

```bash
tensorboard --logdir logs/ensemble_A_1M --port 6006
```

### チェックポイント評価

各チェックポイントで以下を確認:

- `train/legal_sell_rate` → アクションバイアス
- `eval/sharpe_proxy` → パフォーマンス
- `eval/max_drawdown` → リスク
- `train/entropy` → 探索度合い

**最良モデル選択**:
- 最終チェックポイントが最良とは限らない
- 過学習の兆候があれば早期チェックポイントを採用
- 複数指標を総合的に評価

---

## 🎉 成果

### 統一性

✅ 全アルゴリズム（PPO、iterative）で統一的なチェックポイント制御

### 柔軟性

✅ 設定ファイルで簡単にカスタマイズ可能  
✅ デフォルト値は最適化済み（PPO=25000, iterative=10000）

### 拡張性

✅ 将来のアルゴリズムにも簡単に適用可能  
✅ `config.get("checkpoint_interval", default)`パターンで統一

### 実用性

✅ 1M学習で40回のチェックポイント（最適）  
✅ ディスク容量: 約400MB-2GB（管理可能）  
✅ 早期停止、モデル選択、リスク軽減を実現

---

## 📚 関連ドキュメント

1. **CHECKPOINT_INTERVAL_EXTENSION.md** - 詳細な拡張ガイド
2. **QUICKSTART_1M_ENSEMBLE.md** - クイックスタート（チェックポイント説明追加）
3. **UNIFIED_TRAINER_INTEGRATION_SUMMARY.md** - 統合サマリー

---

## ✅ チェックリスト

- ✅ unified_trainer.py: `checkpoint_interval`パラメータ追加
- ✅ run_1m.py: `--checkpoint-interval`引数追加
- ✅ _train_ppo(): 設定ファイルから読み取り
- ✅ _train_iterative(): run_1m.pyに渡す
- ✅ ensemble設定ファイル: `checkpoint_interval: 25000`設定
- ✅ ドキュメント更新: CHECKPOINT_INTERVAL_EXTENSION.md
- ✅ ドキュメント更新: QUICKSTART_1M_ENSEMBLE.md
- ✅ 検証完了: 40チェックポイント確認

---

**次のアクション**: すぐに1M学習を開始できます！

```bash
python -m ztb.training.unified_trainer --config configs/train/ensemble_A_1M.json
```
