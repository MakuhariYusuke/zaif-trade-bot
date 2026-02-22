# メモリリーク修正サマリー

## 適用済みの修正

### 1. training_utils.py
```python
# Before:
def load_training_data_parallel(..., enable_memory_cache: bool = True)
def parallel_data_preprocessing(..., enable_memory_cache: bool = True)

# After:
def load_training_data_parallel(..., enable_memory_cache: bool = False)
def parallel_data_preprocessing(..., enable_memory_cache: bool = False)
```

### 2. memory_cache.py
```python
# Before:
self.data_cache = TTLCache(maxsize=500, ttl=600)  # 10 minutes

# After:
self.data_cache = TTLCache(maxsize=500, ttl=60)  # 1 minute
```

### 3. ab_test_runner.py
```python
# Added at start of execute() method:
def execute(self) -> ExperimentResult:
    # Clear memory cache before training to prevent leak
    import gc
    try:
        default_memory_manager.optimize_memory_usage()
        gc.collect()
    except Exception:
        pass
    ...
```

## 検証結果

✅ 修正の適用確認: PASSED
✅ メモリキャッシュ無効化: PASSED
✅ キャッシュTTL短縮: PASSED (60秒)
✅ ABテストのクリーンアップコード: PASSED

## 残存する問題

❌ Feature Engineering フェーズでのメモリ消費: 651MB → 675MB
   - 500MBメモリ制限を超過 (130%→135%)
   - 6つのタイムフレームで特徴生成中にメモリが累積
   - Quality filteringで削除した特徴のメモリが解放されない

## 推奨される追加対策

### 短期対策 (即座に実施可能)

1. **メモリ制限の一時的な引き上げ**
   ```python
   # memory_cache.py
   default_memory_manager = MemoryManager(max_memory_mb=800.0)  # 500→800
   ```

2. **タイムフレーム数の削減**
   - 設定ファイルで6個→4個に削減してメモリ消費を抑制

3. **Fast modeの活用**
   ```bash
   python tools\ab_test_runner.py --fast-mode --configs "..." --seeds 1
   ```

### 中期対策 (コード修正が必要)

1. **Feature Engineering後のgc.collect()追加**
   - 各タイムフレームの処理後に明示的にガベージコレクション

2. **Padding戦略の見直し**
   - 176個へのパディングを削減、またはon-demandで生成

3. **メモリ効率的な特徴生成**
   - numpy配列の再利用
   - 中間データの即座削除

## テスト実行コマンド

### 修正の検証
```powershell
python tools\test_memory_leak_fix.py
```

### 短いABテスト (500ステップ)
```powershell
python tools\ab_test_runner.py `
    --configs "config/v447/sac_v447_balance_05_penalty_4.json" `
    --seeds 1 --timesteps 500
```

### 本番ABテスト (2000ステップ、メモリ制限引き上げ後)
```powershell
python tools\ab_test_runner.py `
    --configs `
        "config/v447/sac_v447_balance_04_penalty_4.json" `
        "config/v447/sac_v447_balance_05_penalty_4.json" `
        "config/v447/sac_v447_balance_06_penalty_4.json" `
        "config/v447/sac_v447_balance_05_penalty_6.json" `
    --seeds 3
```

## ファイル一覧

修正済みファイル:
- ztb/training/utils/training_utils.py
- ztb/cache/memory_cache.py
- tools/ab_test_runner.py

作成したツール:
- tools/fix_memory_leak.py
- tools/test_memory_leak_fix.py
- tools/monitor_training_memory.py

## 次のステップ

1. メモリ制限を800MBに引き上げてABテスト実行
2. Feature Engineering最適化の計画策定
3. 成功したらreward_components分析に進行
