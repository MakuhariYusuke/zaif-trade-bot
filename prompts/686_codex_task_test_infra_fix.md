# Codex Task: テスト基盤修正 — pytest-benchmark 競合解消 (686#)

## 目的
pytest 実行時の INTERNALERROR を解消し、CI 安定性を確保する。

## 背景

### 現象
```
INTERNALERROR> File ".../pytest_benchmark/plugin.py", line 517, in pytest_runtest_makereport
INTERNALERROR> TypeError: unexpected type for `benchmark` in funcargs,
INTERNALERROR>   <function benchmark.<locals>._run at 0x...> must be a BenchmarkFixture instance.
```

テスト自体は 2268 passed だが、exit code 1 が返り CI が失敗扱いになる。

### 原因
`tests/unit/risk/test_rules.py` L47-54 に定義されたフォールバック `benchmark` fixture が、
インストール済み pytest-benchmark プラグインの同名 fixture と競合している。

```python
# tests/unit/risk/test_rules.py L47-54 (問題箇所)
@pytest.fixture
def benchmark():
    """Fallback benchmark fixture when pytest-benchmark is not installed."""
    def _run(func, *args, **kwargs):
        return func(*args, **kwargs)
    return _run
```

## タスク

### Task 1: benchmark fixture 競合の解消

**対象ファイル**: `tests/unit/risk/test_rules.py`

1. L47-54 のフォールバック `benchmark` fixture を**削除**する
2. このファイル内で `benchmark` を使用しているテストを検索し、以下のいずれかで対応:
   - pytest-benchmark の本物の fixture をそのまま使う（推奨）
   - benchmark を使わないテストは引数から `benchmark` を除去
3. 他のテストファイルにも同様のフォールバック fixture がないか `grep -r "def benchmark" tests/` で確認し、あれば同様に修正

### Task 2: 修正後の検証

1. `python -m pytest tests/unit/risk/test_rules.py -x --tb=short` でエラーなし確認
2. `python -m pytest tests/ -x --tb=short` で INTERNALERROR が消えたことを確認
3. exit code 0 を確認

## 受け入れ基準

- [ ] `tests/unit/risk/test_rules.py` のフォールバック benchmark fixture が削除されている
- [ ] 全テストが pass し、INTERNALERROR が発生しない（exit code 0）
- [ ] 他ファイルに同様の競合がないことを確認済み
