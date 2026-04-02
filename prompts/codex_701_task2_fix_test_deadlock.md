# Codex Task: 701# T2 — test_codex_408 IdempotencyLock テスト分離修正

## ゴール

`tests/unit/v460/test_codex_408_409_fixes.py::TestT1IdempotencyLock` の 3 テストがクラス全体実行時にデッドロックする問題を修正する。

## 背景

701# 調査結果:
- **個別テスト実行**: 3 件全て PASS
- **クラス全体実行** (`::TestT1IdempotencyLock`): KeyboardInterrupt / ハング
- **T1 以外の 33 テスト**: 全 PASS (2.08s)
- **影響**: フルリグレッション (`tests/unit/v460/`) が 76% で必ず停止し、残りの 1,400 テスト (24%) が実行されない

## 根因分析

`IdempotencyStore._process_lock()` はファイルベースの排他ロック:
- `os.O_CREAT | os.O_EXCL | os.O_WRONLY` で lock ファイルを作成
- `contextmanager` の finally で `unlink()` → 解放

テスト間で lock ファイルの残骸が干渉している可能性:
1. `test_process_lock_is_exclusive` — owner が lock 保持中に waiter が TimeoutError → owner の finally で解放
2. `test_process_lock_releases_on_exit` — **前テストの cleanup と競合**
3. `test_stale_lock_recovery` — lock ファイルを手動作成 → recovery テスト

## 修正方針

### Option A (推奨): 各テストに明示的 cleanup fixture 追加

```python
@pytest.fixture(autouse=True)
def _cleanup_lock_files(self, tmp_path: Path) -> Generator[None, None, None]:
    yield
    # テスト後にlock ファイルを強制削除
    for lock_file in tmp_path.glob("*.lock"):
        try:
            lock_file.unlink()
        except OSError:
            pass
```

### Option B: 各テストの db_path を完全に分離

各テスト関数内で `tmp_path / f"test_{uuid}.sqlite"` のような一意の db_path を使用。
ただし pytest の `tmp_path` は既にテストごとに固有なので、これは実質同じ。

### 実際の修正

1. `TestT1IdempotencyLock` に `autouse=True` の cleanup fixture を追加
2. `test_process_lock_is_exclusive` で `owner` の with ブロック終了後に lock 解放を明示確認
3. 各テストの冒頭で stale lock ファイルが存在しないことを assert (防御的)

## テスト検証

修正後に以下を全て PASS させること:
```bash
# 個別
python -m pytest tests/unit/v460/test_codex_408_409_fixes.py::TestT1IdempotencyLock -x --tb=short
# クラス全体
python -m pytest tests/unit/v460/test_codex_408_409_fixes.py -x --tb=short
# フルリグレッション (test_codex_408 を含めて)
python -m pytest tests/unit/v460/ -x --tb=line -q
```

## 制約

- `IdempotencyStore` 本体の変更は不要 (テスト側のみ修正)
- 型注釈必須
- テスト実行速度は現在の 2s 以内を維持
- Windows 環境での `os.O_EXCL` 動作を考慮 (ファイルハンドルの解放タイミング)

## 成果物

- `tests/unit/v460/test_codex_408_409_fixes.py` の TestT1IdempotencyLock 修正
