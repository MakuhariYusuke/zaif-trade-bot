# 279# fix: degraded_liquidation lot floor — config.min_lot → config.min_order_btc

**日付**: 2026-03-04
**種別**: bugfix (CRITICAL — 本番 ERROR)
**前提**: 277# (`d8aebac68`)
**コミット**: `04d9590eb`

---

## 1. 問題

277# デプロイ後、`degraded_liquidation` モード到達時に以下の ERROR が周期的に発生:

```
AttributeError: 'FillTestConfig' object has no attribute 'min_lot'
  File "fill_cycle_executor.py", line 895, in run_single_cycle
    self.config.min_lot,
```

### 発生条件

`degraded_liquidation = True` (Gate bypass + balance_forced 時の縮退清算) に到達した場合のみ。
通常サイクルでは到達しないパスのため、277# テストでは未検出。

### 影響

- ERROR 発生時、当該サイクルは `run_continuous` の `except` で捕捉されスキップ
- 3分間隔の retry で再到達、**4 サイクル連続** (5970–5973) で同一エラー
- 縮退清算モードの目的（在庫偏り解消）が **完全に機能不全**

---

## 2. 根本原因

234# で `degraded_liquidation` lot 縮小のフロア値として `self.config.min_lot` を参照するコードが書かれたが、
`FillTestConfig` には `min_lot` フィールドは存在しない。

- `min_lot` は `LotSizingConfig` (lot_sizer.py L71) のフィールド
- `FillTestConfig` の対応フィールドは `min_order_btc: float = 0.001` (fill_config.py L581)
- `_regime_adjusted_lot()` (fill_record_helpers.py L158) では正しく `self.config.min_order_btc` を使用

234# 実装時のコンテキストスイッチにより、異なる Config クラスの属性名が混在したもの。

---

## 3. 修正

```python
# fill_cycle_executor.py L895
# Before:
self.config.min_lot,

# After:
self.config.min_order_btc,
```

1 ファイル、1 行の修正。

---

## 4. 検証

- 全テスト: **3827 passed, 32 skipped** (変化なし)
- 278# デプロイ後の初動ログ: ERROR なし、Cycle 5967 (buy) 正常進行
- `git_sha=04d9590eb3bd` が schema_health に記録

---

## 5. 教訓

| 項目 | 内容 |
|---|---|
| **直接原因** | `FillTestConfig` と `LotSizingConfig` の類似フィールド名 (`min_lot` vs `min_order_btc`) の取違え |
| **構造的原因** | `degraded_liquidation` パスは通常サイクルではほぼ到達しないため、テストカバレッジが不足していた |
| **予防策** | mypy strict モードで `object has no attribute` は検出可能だが、現状 `FillTestConfig` が 200+ フィールドの巨大 dataclass であるため `--strict` は実用的でない。277# の `__post_init__` バリデーション強化は同種のミスを防ぐ方向 |
| **277# セルフレビューで検出できたか** | セルフレビューの対象は 271#–276# の変更であり、234# の既存コードは対象外。ただし `min_lot` の grep で発見可能だった — 「config 参照の網羅 grep」をセルフレビュー手順に追加すべき |

---

## 6. 関連

- 234# [234_ph2_fix_gate_bypass_degraded_liquidation.md](234_ph2_fix_gate_bypass_degraded_liquidation.md) — degraded_liquidation 元実装
- 277# [277_magic_number_grounding.md](277_magic_number_grounding.md) — 直前のセルフレビュー
