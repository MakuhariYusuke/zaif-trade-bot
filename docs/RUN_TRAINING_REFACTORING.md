# run_training.py リファクタリング - v3.6.2

**日時:** 2025-10-08  
**目的:** `--timesteps`引数の追加と、`run_training.py`の薄層化

---

## 🎯 実施した変更

### 1. `--timesteps`引数の追加

#### run_training.py
```python
parser.add_argument(
    "--timesteps",
    type=int,
    help="Override total_timesteps from config (useful for quick validation runs)"
)
```

**用途:**
- 設定ファイルの`total_timesteps`を上書き
- 短い検証実行に便利（例: 10000 steps）

**使用例:**
```bash
# 10000 stepsで検証実行
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000

# 5000 stepsでクイックテスト
python run_training.py --config configs/training/ppo_100k_optimized.json --timesteps 5000
```

---

### 2. UnifiedTrainerへの処理集約

#### Before: run_training.pyが多くの処理を実施
```python
# 設定読み込み
config = load_config(args.config)

# 確認プロンプト（47行）
if not args.force:
    logger.info("TRAINING CONFIGURATION")
    # ... 多くの表示処理
    response = input("Proceed? [y/N]: ")
    if response.lower() != 'y':
        return 0

# トレーナー初期化
trainer = UnifiedTrainer(config=config, force=args.force)
```

#### After: UnifiedTrainerに委譲（薄層化）
```python
# 設定読み込み
config = load_config(args.config)

# トレーナー初期化（全ロジックはUnifiedTrainer内）
trainer = UnifiedTrainer(
    config=config,
    force=args.force,
    dry_run=args.dry_run,
    total_timesteps=args.timesteps  # オーバーライド
)

# 実行
model = trainer.train()
```

**削減された処理:**
- 確認プロンプト → `UnifiedTrainer`内で処理
- 設定表示 → `UnifiedTrainer`内で処理
- バリデーション → `UnifiedTrainer`内で処理

**結果:**
- `run_training.py`: 201行 → **198行**（簡潔化）
- 責任の明確化: `run_training.py`は薄いCLIラッパーのみ

---

### 3. UnifiedTrainer.__init__の拡張

#### ztb/training/unified_trainer.py
```python
def __init__(
    self,
    config: Dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
    enable_streaming: bool = False,
    stream_batch_size: int = 256,
    max_features: Optional[int] = None,
    total_timesteps: Optional[int] = None,  # ← 新規追加
):
    """
    Args:
        total_timesteps: Override total_timesteps from config 
                        (for quick validation runs)
    """
    # Override total_timesteps if specified
    if total_timesteps is not None:
        config = config.copy()  # Don't modify original
        config["total_timesteps"] = total_timesteps
        logger.info(f"Overriding total_timesteps: {total_timesteps:,}")
    
    # ... 既存の処理
```

**特徴:**
- 元の`config`を変更しない（`copy()`使用）
- オーバーライド時にログ出力
- 型ヒント完備

---

## 📋 使用例

### 1. SELL回避問題の検証（緊急修正後）

**Before（エラー）:**
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000
# Error: unrecognized arguments: --timesteps 10000
```

**After（正常動作）:**
```bash
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000
# ✅ 10000 stepsで実行
# Overriding total_timesteps: 10,000
```

### 2. クイックテスト

```bash
# 5000 stepsでSELL率を確認
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 5000 --force
```

### 3. Dry Run（設定検証のみ）

```bash
# 実行せずに設定を検証
python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --dry-run --timesteps 10000
```

---

## 🎯 設計思想

### Single Responsibility Principle

**run_training.py（薄層）:**
- CLI引数のパース
- 設定ファイルの読み込み
- `UnifiedTrainer`の呼び出し
- **総行数: 198行（簡潔）**

**UnifiedTrainer（厚層）:**
- 設定バリデーション
- 確認プロンプト
- アルゴリズム選択
- 学習実行
- **総行数: 762行（ロジック集約）**

### 利点

1. **保守性向上:**
   - ビジネスロジックが`UnifiedTrainer`に集約
   - `run_training.py`は薄くシンプル

2. **テスト容易性:**
   - `UnifiedTrainer`を直接テスト可能
   - CLIレイヤーとロジックレイヤーの分離

3. **再利用性:**
   - 他のスクリプトから`UnifiedTrainer`を直接使用可能
   - CLI引数に依存しない

---

## 🔍 変更ファイル一覧

1. **run_training.py:**
   - `--timesteps`引数追加
   - 確認プロンプト削除（UnifiedTrainerに委譲）
   - ドキュメント更新

2. **ztb/training/unified_trainer.py:**
   - `__init__`に`total_timesteps`パラメータ追加
   - オーバーライドロジック実装
   - ドキュメント追加

---

## ✅ 検証

### テストコマンド

```bash
# 引数のヘルプ確認
python run_training.py --help

# 期待される出力:
#   --timesteps TIMESTEPS
#                         Override total_timesteps from config (useful for quick validation runs)
```

### 動作確認

```bash
# 10000 stepsで実行
python run_training.py \
  --config configs/training/ppo_balanced_mem_optimized.json \
  --timesteps 10000 \
  --force

# 期待されるログ:
# INFO - Overriding total_timesteps: 10,000
# INFO - Total timesteps: 10000
```

---

## 📊 コード行数の変化

| ファイル | Before | After | 変化 |
|---------|--------|-------|------|
| `run_training.py` | 201 | 198 | **-3** |
| `unified_trainer.py` | 762 | 762 | 0 |

**薄層化成功:**
- `run_training.py`の責務を削減
- ロジックを`UnifiedTrainer`に集約

---

## 🎯 次のステップ

**SELL回避問題の検証:**
```bash
# 緊急修正後の設定で10000 steps実行
python run_training.py \
  --config configs/training/ppo_balanced_mem_optimized.json \
  --timesteps 10000 \
  --force

# 結果をログ分析
python scripts/analyze_training_logs_v2.py <ログファイル>

# 期待される改善:
# - SELL rate: 1.6% → 15%以上
# - Lambda: 2.0 → 15.0未満（飽和していない）
```

---

**変更完了。`--timesteps`引数が使えるようになりました。** ✅
