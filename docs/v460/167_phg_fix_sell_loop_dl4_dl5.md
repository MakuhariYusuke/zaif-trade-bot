# 167# — sell ループ構造修正 (DL-4/DL-5) & カウンタ永続化

## 0. 背景

166# §11.5 の戦略的改善提案 4 項目を深堀り分析した結果、
**4 提案のうち 3 つが同一の構造的欠陥を共有** していることが判明。

## 1. 根本原因分析

### 1.1 sell ループのメカニズム (全提案の共通根因)

**発見**: `fill_loop_orchestrator.py` の `trending_sell_skip` パス (旧L734)
で `_last_side` を更新せず `continue` していたため、
`side_selector.next()` が次サイクルでも同じ side ("sell") を返し続ける。

```
[正常フロー]
buy fill → _last_side="buy" → next()→"sell" → sell fill → _last_side="sell" → next()→"buy" → ...

[バグ: trending_sell_skip で _last_side 未更新]
buy fill → _last_side="buy" → next()→"sell" → trending_sell_skip (continue, _last_side変更なし)
      → _last_side="buy" のまま → next()→"sell" → trending_sell_skip → ... 自己強化ループ
```

### 1.2 データ証拠

直近 SHA (3aba65e9 等, DL-3 修正後) での実測:

| 指標 | buy | sell | 差分 |
|------|-----|------|------|
| サイクル数 | 69 | 143 | sell 2.07x 過剰 |
| fill rate | 47.8% | 21.0% | -26.8pt |
| trending_sell_skip | 0 | 69 | sell skip の 61% |

after trending_sell_skip → 次のサイクルも sell: **99%** (自己強化率)

### 1.3 4 提案への影響マッピング

| 提案 | 本質 | 根本対応 |
|------|------|---------|
| ① sell fill rate 改善 | _last_side 未更新で sell 過剰蓄積 | **DL-4** |
| ② カウンタ永続化 | 再起動でリセット | **resume 復元** |
| ③ UTC 09-10 特別処理 | sell 蓄積の副次的症状 (ad-hoc 不要) | **DL-4 で自動解消** |
| ④ lot 自動調整 | rescue_enabled=true で bf_skip 到達不可。lot は既に最小値 | **対応不要** |

## 2. 修正内容

### 2.1 DL-4: trending_sell_skip の _last_side 更新

```python
# 167# DL-4: _last_side を更新して buy 側も試行可能に
self._last_side = next_side  # = "sell"
await asyncio.sleep(self.config.cycle_interval_sec)
continue
```

**効果**: sell-skip → buy → sell-skip → buy の交互パターンに正常化。
従来の sell-skip × 10 連続 → buy が sell-skip × 1 → buy × 1 に改善。

### 2.2 DL-5: balance_forced_skip の _last_side 更新 (防御的)

```python
# 167# DL-5: _last_side を更新 (rescue=true 時は到達しないが防御的に)
self._last_side = next_side
```

`balance_forced_rescue_enabled=true` により L632 は通常到達不可だが、
config 変更時の安全性確保のため防御的に追加。

### 2.3 P2: カウンタ復元 (resume_from_existing)

`resume_from_existing()` で末尾連続 skip レコードから
`_trending_sell_skip_count` と `_balance_forced_skip_count` を復元。

```python
# 167# DL-4/P2: 末尾連続 skip カウンタを復元 (再起動耐性)
for rec in reversed(existing):
    cr = rec.cancel_reason
    if cr == "trending_sell_skip":
        _tss_count += 1
    elif cr == "balance_forced_skip":
        _bfs_count += 1
    else:
        break
```

## 3. 設計思想

### 3.1 汎用原則: 「全 continue パスは `_last_side` を更新する」

166# DL-1/2/3 で確立されたパターンを **全** skip パスに拡張。
11 個の continue パスのうち、side 決定後の 6 パスすべてが `_last_side` を更新:

| パス | side | _last_side 更新 | 修正 |
|------|------|----------------|------|
| unknown_regime_buy_skip | buy | ✅ `"buy"` | 166# DL-1 |
| buy_dynamic_kill | buy | ✅ `"buy"` | 166# DL-2 |
| sell_dynamic_kill | sell | ✅ `"sell"` | 166# DL-3 |
| exception handler | any | ✅ `next_side` | 166# SR-4 |
| **trending_sell_skip** | sell | ✅ `next_side` | **167# DL-4** |
| **balance_forced_skip** | any | ✅ `next_side` | **167# DL-5** |

### 3.2 ad-hoc 回避

- UTC 09-10 特別処理は **不要** — sell 蓄積が解消されれば時間帯依存は消失
- lot 自動調整は **不要** — rescue mode が bf_skip を吸収済み
- 全修正は「skip パスの交互保証」という **単一の汎用原則** に基づく

## 4. テスト

```
189 passed, 0 failed (対象領域: fill_loop, fill_record, fill_cycle, side_selector, etc.)
458 passed, 0 failed (fill テスト全体)
```

## 5. 変更ファイル

```
scripts/v460/lib/fill_loop_orchestrator.py  |  4 ++++  (DL-4, DL-5)
scripts/v460/lib/fill_record_helpers.py     | 18 +++++++++++++++++-  (P2)
```
