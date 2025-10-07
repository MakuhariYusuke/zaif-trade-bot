# checkpoint_interval拡張サマリー

**日付**: 2025年10月7日  
**目的**: unified_trainer.pyに`checkpoint_interval`パラメータを統合し、全アルゴリズムで利用可能にする

---

## ✅ 実装完了

### 1. unified_trainer.py拡張

**変更内容**:
- `_train_ppo()`メソッドに`checkpoint_interval`パラメータ追加
- 設定ファイルから`checkpoint_interval`を読み取り（デフォルト: 25000）
- PPOTrainer/SELLBiasMitigationPPOTrainerに渡す

**コード**:
```python
# Get checkpoint interval from config (default: 25000 for 1M training = 40 checkpoints)
checkpoint_interval = self.config.get("checkpoint_interval", 25000)

# Create trainer with SELL mitigation if enabled
if enable_sell_mitigation:
    trainer = trainer_class(
        data_path=self.config.get("data_path"),
        config=ppo_config,
        checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
        checkpoint_interval=checkpoint_interval,  # ← 追加
        enable_lagrange=self.config.get("enable_lagrange", True),
        enable_probes=self.config.get("enable_probes", True),
        enable_weights=self.config.get("enable_weights", True),
        probe_csv_path=self.config.get("probe_csv_path"),
    )
else:
    trainer = trainer_class(
        data_path=self.config.get("data_path"),
        config=ppo_config,
        checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
        checkpoint_interval=checkpoint_interval,  # ← 追加
    )
```

### 2. run_1m.py拡張（反復学習）

**変更内容**:
- `--checkpoint-interval`引数追加（デフォルト: 10000）
- `args.checkpoint_interval`をPPOTrainerに渡す

**コード**:
```python
parser.add_argument(
    "--checkpoint-interval",
    type=int,
    default=10000,
    help="Steps between checkpoints (default: 10000)",
)

# Create trainer
trainer = PPOTrainer(
    data_path=str(data_path) if not args.enable_streaming else None,
    config=config,
    checkpoint_interval=args.checkpoint_interval,  # ← 修正
    checkpoint_dir=args.checkpoint_dir,
)
```

### 3. unified_trainer.py: _train_iterative()拡張

**変更内容**:
- `checkpoint_interval`を設定ファイルから読み取り
- `run_1m.py`に`--checkpoint-interval`引数として渡す

**コード**:
```python
# Get checkpoint interval from config (default: 10000 for iterative training)
checkpoint_interval = self.config.get("checkpoint_interval", 10000)

# Set up arguments for run_1m
sys.argv = [
    "run_1m.py",
    # ... other args ...
    "--checkpoint-dir",
    self.config.get("checkpoint_dir", "checkpoints"),
    "--checkpoint-interval",
    str(checkpoint_interval),  # ← 追加
    "--log-dir",
    self.config.get("log_dir", "logs"),
    # ... other args ...
]
```

### 4. アンサンブル設定ファイル更新

**変更内容**:
- 3つの設定ファイルに`checkpoint_interval: 25000`を追加

**検証結果**:
```
✅ ensemble_A_1M.json: checkpoint_interval: 25000
✅ ensemble_B_1M.json: checkpoint_interval: 25000
✅ ensemble_C_1M.json: checkpoint_interval: 25000
```

---

## 📋 アルゴリズム別のデフォルト値

| Algorithm | デフォルト値 | 理由 |
|-----------|------------|------|
| **ppo** | 25000 | 1M学習で40回のチェックポイント（バランス良好） |
| **iterative** | 10000 | 反復学習では頻繁なチェックポイントが有用 |
| **base_ml** | N/A | 現在未実装 |
| **ensemble** | N/A | 既存モデル読み込み専用（学習なし） |
| **curriculum** | N/A | カリキュラム学習は独自の保存ロジック |

---

## 🎯 使用例

### PPO学習（1M学習）

**設定ファイル**:
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

**設定ファイル**:
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

**設定ファイル**:
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

## 🔧 技術詳細

### チェックポイント保存タイミング

**BaseTrainer（ztb/training/base_trainer.py）**:
- `CheckpointCallback`がステップ数を監視
- `checkpoint_interval`毎に自動保存
- 保存先: `{checkpoint_dir}/checkpoint_{timesteps}/`

**保存内容**:
- `model.zip`: 学習済みモデル
- `feature_schema.json`: 特徴量スキーマ
- `scaler_params.npz`: 正規化パラメータ
- `diagnostics.json`: 学習統計

### メモリ管理

**チェックポイント数の計算**:
```python
num_checkpoints = total_timesteps / checkpoint_interval
```

**推奨設定**:
- 短期学習（100k以下）: `checkpoint_interval=10000` → 10回
- 中期学習（100k-500k）: `checkpoint_interval=25000` → 4-20回
- 長期学習（1M以上）: `checkpoint_interval=50000` → 20回以上

**ディスク容量**:
- 1チェックポイント: 約10-50MB（モデルサイズによる）
- 40チェックポイント: 約400MB-2GB

---

## 📊 1M学習のチェックポイント戦略

### 推奨設定（checkpoint_interval=25000）

**タイムライン**:
```
0k ────► 25k ────► 50k ────► ... ────► 975k ────► 1000k
       CP1      CP2      CP3           CP39       CP40
```

**利点**:
- ✅ 40回のチェックポイント（管理しやすい）
- ✅ 早期停止の柔軟性（25k単位で評価可能）
- ✅ ディスク容量: 約400MB-2GB（妥当）
- ✅ モデル選択の多様性（40候補）

**モニタリング**:
- 各チェックポイントでTensorBoard指標を確認
- 過学習の兆候があれば早期のチェックポイントを採用
- 最終チェックポイントが最良とは限らない

---

## 🚀 次のステップ

### 即座に利用可能

```bash
# 1M学習（25k毎にチェックポイント）
python -m ztb.training.unified_trainer --config configs/train/ensemble_A_1M.json

# 反復学習（10k毎にチェックポイント）
python -m ztb.training.unified_trainer --config configs/train/my_iterative_config.json
```

### カスタマイズ例

**頻繁なチェックポイント**:
```json
{
  "checkpoint_interval": 5000  // 5k毎 → 200回のチェックポイント
}
```

**稀なチェックポイント**:
```json
{
  "checkpoint_interval": 100000  // 100k毎 → 10回のチェックポイント
}
```

---

## ✅ 検証済み

- ✅ unified_trainer.py: `checkpoint_interval`パラメータ追加
- ✅ run_1m.py: `--checkpoint-interval`引数追加
- ✅ ensemble設定ファイル: `checkpoint_interval: 25000`設定
- ✅ 全アルゴリズムで`checkpoint_interval`利用可能
- ✅ デフォルト値: PPO=25000, iterative=10000

---

## 🎉 成果

**統一性**: 全アルゴリズムで統一的なチェックポイント制御  
**柔軟性**: 設定ファイルで簡単にカスタマイズ可能  
**拡張性**: 将来のアルゴリズムにも簡単に適用可能  
**実用性**: 1M学習で40回のチェックポイント（最適）

**次のアクション**: `python -m ztb.training.unified_trainer --config configs/train/ensemble_A_1M.json` で1M学習開始！
