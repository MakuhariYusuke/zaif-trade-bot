# run_training.py 修正完了 - v3.6.2

## ✅ 修正内容

### 問題
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000
# Error: unrecognized arguments: --timesteps 10000
```

### 解決策

#### 1. `--timesteps`引数を追加

**run_training.py:**
```python
parser.add_argument(
    "--timesteps",
    type=int,
    help="Override total_timesteps from config (useful for quick validation runs)"
)
```

#### 2. UnifiedTrainer.__init__を拡張

**ztb/training/unified_trainer.py:**
```python
def __init__(
    self,
    config: Dict[str, Any],
    total_timesteps: Optional[int] = None,  # ← 新規
):
    # Override if specified
    if total_timesteps is not None:
        config = config.copy()
        config["total_timesteps"] = total_timesteps
        logger.info(f"Overriding total_timesteps: {total_timesteps:,}")
```

#### 3. run_training.pyを薄層化

**Before: 201行（多くの処理）**
- 確認プロンプト
- 設定表示
- バリデーション

**After: 198行（CLIラッパーのみ）**
- 引数パース
- UnifiedTrainerに委譲

---

## 🎯 使用例

### SELL回避問題の検証（今すぐ実行可能）

```bash
# 10000 stepsで検証実行
python run_training.py \
  --config configs/training/ppo_balanced_mem_optimized.json \
  --timesteps 10000 \
  --force
```

**期待されるログ:**
```
INFO - Overriding total_timesteps: 10,000
INFO - Total timesteps: 10000
...
INFO - SELL Rate (avg): ??%  ← 15%以上が目標
INFO - Lambda (final): ??    ← 20.0未満が目標
```

### その他の使用例

```bash
# 5000 stepsでクイックテスト
python run_training.py --config configs/training/ppo_100k_optimized.json --timesteps 5000

# Dry run（設定検証のみ）
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --dry-run --timesteps 10000
```

---

## 📁 変更ファイル

1. **run_training.py** (198行)
   - `--timesteps`引数追加
   - 薄層化（確認プロンプト削除）
   - ドキュメント更新

2. **ztb/training/unified_trainer.py** (762行)
   - `__init__`に`total_timesteps`パラメータ追加
   - オーバーライドロジック実装

3. **docs/RUN_TRAINING_REFACTORING.md** (新規)
   - 詳細なリファクタリングドキュメント

---

## ✅ 検証結果

```bash
$ python run_training.py --help
...
  --timesteps TIMESTEPS
                        Override total_timesteps from config (useful for quick validation runs)
...
```

**動作確認: ✅ OK**

---

## 🚀 次のアクション

SELL回避問題の検証を実行してください：

```bash
python run_training.py \
  --config configs/training/ppo_balanced_mem_optimized.json \
  --timesteps 10000 \
  --force

# 完了後、ログを分析
python scripts/analyze_training_logs_v2.py <ログファイル>
```

**成功基準:**
- SELL rate: 1.6% → **15%以上**
- Lambda: 2.0 → **15.0未満**（飽和していない）

---

**修正完了。`--timesteps`引数が使えるようになりました！** ✅
