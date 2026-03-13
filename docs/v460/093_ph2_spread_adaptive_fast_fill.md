# 093# spread_adaptive / fast_fill_defense サイド別パラメータ追加

| 項目 | 内容 |
|------|------|
| 日付 | 2025-06-17 |
| 根拠 | 084# 盲点A (spread_adaptive 全件 2.0× boost), 盲点C (短 wait AS 防衛), 090# sell fast fill -1.66bps |
| 対象 | `run_fill_test.py`, `fill_test.yaml`, `test_093_side_params.py` |
| 方針 | 既存実装の side 別拡張。新規モジュール追加なし。DRY 原則遵守。|

---

## §1 背景・課題

### 1.1 spread_adaptive の問題 (084# 盲点A)
- Median spread = 2.08 bps << `narrow_spread_bps=10.0`
- **全注文が `narrow_spread_boost=2.0` の適用対象** → 実効 offset は yaml 値の 2 倍
- buy 0.05 → 0.10、sell 0.12 → 0.24 で運用
- 084# Tier2 #4: "narrow_spread_boost 2.0 は最適か? 1.5, 2.5 等と比較"

### 1.2 fast_fill_defense の問題 (084# 盲点C)
- Wait Q1-Q3 (5-25s) は AS 率 41-47%、PnL 最悪帯
- 現行 `threshold_sec=5.0` では Q1 (5-6s) の一部しか捕捉できない
- 090# 分析: sell fast fill は **-1.66 bps** と破滅的
- 084# Tier3 #8: "timeout 短縮ではなく短 wait 側を制御"

---

## §2 実装内容

### 2.1 spread_adaptive — サイド別 narrow_spread_boost

**Config 追加フィールド**:
```python
narrow_spread_boost_buy: float | None = None   # None = 共通値使用
narrow_spread_boost_sell: float | None = None   # None = 共通値使用
```

**ロジック変更** (`_compute_maker_price`):
```python
sa_boost = self.config.narrow_spread_boost  # 共通フォールバック
if side == "buy" and self.config.narrow_spread_boost_buy is not None:
    sa_boost = self.config.narrow_spread_boost_buy
elif side == "sell" and self.config.narrow_spread_boost_sell is not None:
    sa_boost = self.config.narrow_spread_boost_sell
```

**チューニング値**:
| パラメータ | 旧値 | 新値 (buy) | 新値 (sell) | 根拠 |
|---|---|---|---|---|
| narrow_spread_boost | 2.0 (共通) | 1.5 | 2.0 | buy: fill 促進 (0.05→0.075)。sell: AS 構造的要因のため据え置き (084#) |

### 2.2 fast_fill_defense — サイド別 threshold / boost

**Config 追加フィールド**:
```python
fast_fill_threshold_sec_buy: float | None = None   # None = 共通値
fast_fill_threshold_sec_sell: float | None = None
fast_fill_offset_boost_buy: float | None = None
fast_fill_offset_boost_sell: float | None = None
```

**ロジック変更** (`run_continuous` 内の防御判定):
```python
ff_threshold = self.config.fast_fill_threshold_sec  # 共通フォールバック
if record.side == "buy" and self.config.fast_fill_threshold_sec_buy is not None:
    ff_threshold = self.config.fast_fill_threshold_sec_buy
elif record.side == "sell" and self.config.fast_fill_threshold_sec_sell is not None:
    ff_threshold = self.config.fast_fill_threshold_sec_sell
is_fast = record.queue_wait_sec <= ff_threshold
```

**チューニング値**:
| パラメータ | 旧値 | 新値 (buy) | 新値 (sell) | 根拠 |
|---|---|---|---|---|
| threshold_sec | 5.0 (共通) | 5.0 (null) | **15.0** | sell Q1-Q2+α を広く防御 (084# 盲点C) |
| offset_boost | 2.0 (共通) | 2.0 (null) | **2.5** | sell fast fill 破滅的 → 強い防御 (090#) |

---

## §3 実効 offset 変化まとめ

### 通常時 (narrow spread、全注文の ~100%)
```
          旧 (093# 前)           新 (093# 後)
buy:      0.05 × 2.0 = 0.10     0.05 × 1.5 = 0.075  ← fill 促進 (-25%)
sell:     0.12 × 2.0 = 0.24     0.12 × 2.0 = 0.24   ← 変更なし
```

### fast_fill_defense 発動時
```
                        旧                新 (sell fast fill)
buy 防御:               offset × 2.0     offset × 2.0 (変更なし)
sell 防御:              offset × 2.0     offset × 2.5 (+25%)
sell 検出範囲:          ≤ 5s              ≤ 15s (Q1-Q2, 一部 Q3 を捕捉)
```

---

## §4 設計判断

### Q1: narrow_spread_bps を下げないのか?
- 現行 10.0bps でほぼ全件が narrow 判定 (median 2.08bps)
- 閾値を下げると "normal" ゾーン (boost なし) の注文が出現
- 現時点では **boost の強度を side 別で調整する方が制御しやすい**
- narrow_spread_bps 調整は AB テストで後日探索 (087# P2 候補)

### Q2: sell の narrow_spread_boost を上げないのか?
- 084# の結論: "offset 0.24 で sell PnL がまだ -0.95 bps → offset 問題ではなく AS 構造的要因が大きい"
- sell offset 増加の限界効果は低い → 据え置きが合理的
- 真の対策は SkipGate 改善・time_filter・sell_guard (既に実装済み)

### Q3: fast_fill_defense の sell 15s は攻めすぎでは?
- 防御発動条件は `is_fast AND has_negative_edge` (両方成立が必要)
- negative edge がない正常 fill は 15s 以下でも防御非発動
- 防御は次サイクルのみ (1 回で解除) → 過剰阻害リスクは限定的

---

## §5 テスト

[test_093_side_params.py](../../tests/unit/v460/test_093_side_params.py): **29 テスト**
- A. Config フィールド存在 (5)
- B. fast_fill Config フィールド存在 (8)
- C. YAML パース spread_adaptive (3)
- D. YAML パース fast_fill_defense (3)
- E. ロジック構造 spread_adaptive (2)
- F. ロジック構造 fast_fill_defense (3)
- G. 実効値 spread_adaptive (3)
- H. 実効値 fast_fill_defense (2)

全 751 テスト PASS (v460 スコープ)

---

## §6 次ステップ

| # | 施策 | 優先度 | 状態 |
|---|---|---|---|
| 1 | 093# 適用後データ収集 (buy fill rate 変化、sell 防御頻度) | HIGH | 再起動後 |
| 2 | narrow_spread_bps 探索 (10→5→3) | MID | AB テスト候補 |
| 3 | wide_spread_ratio side 別化 | LOW | データ不足 |
| 4 | fast_fill_defense の持続時間制御 (1 サイクル → N サイクル) | LOW | 要検討 |
