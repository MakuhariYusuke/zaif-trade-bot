# 263# Optional[X] → X | None 統一

| 項目 | 値 |
|---|---|
| Issue | 263# |
| 種別 | impl (PEP 604 準拠) |
| フェーズ | phg (横断品質改善) |
| Commit | `96c3d137f` |
| テスト | 3616 passed, 32 skipped (変更なし) |
| 元チケット | 259# P3-1 |

---

## 背景

Python 3.10+ (PEP 604) で導入された `X | None` 構文に統一し、
コードベース全体の型アノテーション表記を簡潔化する。

- `Optional[X]` → `X | None` (87 箇所)
- `from typing import Optional` 不要化 (18 ファイル)
- `List[X]` → `list[X]` (PEP 585, 1 箇所)

## 対象ファイル (18 ファイル)

### `Optional` を唯一の import として使用 → import 行ごと削除 (5 ファイル)

| ファイル | 置換数 |
|---|---|
| `scripts/v460/lib/macro_regime.py` | 0 (未使用 import のみ) |
| `scripts/v460/lib/maker_price.py` | 0 (未使用 import のみ) |
| `scripts/v460/lib/skip_gate_v460.py` | 5 |
| `scripts/v460/lib/order_manager.py` | 4 |
| `scripts/v460/lib/position_tracker.py` | 3 |

### `Optional` を他の import と併用 → `Optional` のみ除去 (13 ファイル)

| ファイル | 置換数 |
|---|---|
| `scripts/v460/lib/__init__.py` | 1 |
| `scripts/v460/lib/abstract_cycle_runner.py` | 3 + `List→list` |
| `scripts/v460/lib/adaptation_engine.py` | 5 |
| `scripts/v460/lib/balance_adapter.py` | 2 |
| `scripts/v460/lib/config_access.py` | 5 |
| `scripts/v460/lib/data_loader.py` | 8 (ネスト含む) |
| `scripts/v460/lib/fill_test.py` | 9 |
| `scripts/v460/lib/lot_sizer.py` | 2 |
| `scripts/v460/lib/manifest.py` | 6 (ネスト含む) |
| `scripts/v460/lib/order_monitor.py` | 6 |
| `scripts/v460/lib/phg_runner.py` | 7 |
| `scripts/v460/lib/run_continuous.py` | 18 |
| `scripts/v460/lib/state_manager.py` | 2 |

## 変換ルール

```
# Before
from typing import Optional
x: Optional[int] = None

# After
x: int | None = None
```

ネストケース:
```
# Before
Optional[list[str]]

# After
list[str] | None
```

## 安全性

- 意味的に同一 (`Optional[X]` ≡ `X | None` ≡ `Union[X, None]`)
- Python 3.11.9 (本プロジェクト) で完全サポート
- 全テスト 3616 passed → 変更前後で挙動変化なし
