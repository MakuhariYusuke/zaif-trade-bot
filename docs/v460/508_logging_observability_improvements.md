# 508# ログ・可観測性改善

## 概要

fill test 再起動前に反映できるログ改善。
de-meaning 関連の basis_bps / adjusted_spread が不可視だった問題と、
sell_age_cap / regime_timeout のログレベル不足を修正。

---

## 変更内容

### 1. Cross-Venue basis_bps / adjusted_spread_bps の可観測性 (508# P0)

**問題:** 506#/507# で導入した basis correction (de-meaning) が
ログにも fill_record にも記録されておらず、direction 判定の根拠が不可視。

**修正:**

| 対象 | 追加フィールド | 用途 |
|------|---------------|------|
| `CrossVenueLeadLagHint` | `basis_bps`, `adjusted_spread_bps` | hint 内部に補正情報保持 |
| `build_cross_venue_event_details()` | 同上 | event JSONL に記録 |
| `build_cross_venue_fill_fields()` | `cross_venue_basis_bps`, `cross_venue_adjusted_spread_bps` | fill_record に記録 |
| `FillRecord` (fill_quality.py) | 同上 | スキーマ定義 |
| fill_cycle_executor.py ログ | `basis=`, `adj=` | サイクルログに表示 |

**ログ出力例:**
```
[cross_venue] hint direction=up adverse_side=sell spread=-3.30bps
  velocity=+0.15bps/s age=0.50s conf=0.50 ... basis=-3.30bps adj=+0.12bps
```

### 2. sell_age_cap 発動ロギング (508# P1)

**問題:** 506# の `sell_age_cap_sec=25s` で timeout がキャップされても無音。

**修正:** キャップ発動時に info ログ出力:
```
[506#] sell_age_cap enforced: 30s → 25s (cap=25s)
```

### 3. regime_timeout ログレベル昇格 (508# P2)

**問題:** `regime_timeout` multiplier 適用が `debug` レベルで通常運用時に不可視。

**修正:** `logger.debug` → `logger.info` に昇格。
timeout 計算は運用判断に直結するため info が適切。

---

## 変更ファイル

| ファイル | 変更 |
|---------|------|
| `scripts/v460/lib/cross_venue_lead_lag.py` | Hint に `basis_bps`, `adjusted_spread_bps` 追加 + fill_fields / event_details 拡張 |
| `scripts/v460/lib/fill_cycle_executor.py` | hint ログに `basis=`, `adj=` 追加 |
| `scripts/v460/lib/order_monitor.py` | sell_age_cap 発動ログ + regime_timeout info 昇格 |
| `ztb/metrics/fill_quality.py` | FillRecord スキーマに 2 フィールド追加 |

---

## テスト

v460 全 3796 テスト パス (0 failed)。
既存 `test_builder_fields_accepted_by_fill_record` がスキーマ整合性を自動検証。

---

*実装日: 2026-03-20*
