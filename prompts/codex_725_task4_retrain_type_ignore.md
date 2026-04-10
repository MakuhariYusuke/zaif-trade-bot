# 725# Task 4: retrain_scheduler type: ignore 3箇所修正

## 背景
`scripts/v460/ml/retrain_scheduler.py` に3箇所の `# type: ignore` がある。
各々原因が異なるため個別に対処。

## 修正対象

### 4A. Line 525: `possibly-undefined`
```python
# 現状
try:
    tmp_path = atomic_pickle_tmp_path(cache_path)
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(cache_path)
except Exception as e:
    logger.warning(f"E4: Cache save failed: {e}")
    try:
        tmp_path.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
    except Exception:
        pass
```

**原因**: `atomic_pickle_tmp_path()` が例外を投げた場合 `tmp_path` が未定義。
**修正**: try ブロック前に `tmp_path: Path | None = None` を初期化し、cleanup で None チェック。

```python
tmp_path: Path | None = None
try:
    tmp_path = atomic_pickle_tmp_path(cache_path)
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(cache_path)
except Exception as e:
    logger.warning(f"E4: Cache save failed: {e}")
    if tmp_path is not None:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
```

### 4B. Line 1712: `index` (eval_set tuple unpacking)
```python
es_X, _ = fit_kwargs["eval_set"][0]  # type: ignore[index]
```

**原因**: `fit_kwargs` が `dict[str, Any]` のため `["eval_set"]` の型が不明。
**修正**: `fit_kwargs["eval_set"]` を `list` として cast するか、ローカル変数で型ヒントを付ける。

```python
eval_set_list: list[tuple[Any, Any]] = fit_kwargs["eval_set"]
es_X, _ = eval_set_list[0]
```

### 4C. Line 2154: `import-untyped` (psutil)
```python
import psutil  # type: ignore[import-untyped]
```

**原因**: psutil にスタブパッケージがない。
**修正選択肢**:
- (推奨) `mypy.ini` に `[[tool.mypy.overrides]]` で `module = "psutil"` → `ignore_missing_imports = true` を追加
- (代替) そのまま維持（外部ライブラリの型スタブ不在は本質的に解決不可能）

**注意**: psutil の `import-untyped` は本質的に解消不可能なので、mypy.ini への追加が最もクリーン。
現行 mypy.ini にすでに psutil が設定済みの場合はスキップ。

## テスト対象ファイル
- `scripts/v460/ml/retrain_scheduler.py`

## 検証
```bash
mypy scripts/v460/ml/retrain_scheduler.py --config-file mypy.ini
python -m pytest tests/unit/v460/test_retrain_hot_reload.py -x --tb=short
python -m pytest tests/unit/v460/test_retrain_e4_cache.py -x --tb=short -k "cache" 2>/dev/null || true
```

## 制約
- ランタイム動作の変更は不可
- 既存テスト全パス必須
- 4C は mypy.ini 変更のみ可（psutil 本体への変更不可）
