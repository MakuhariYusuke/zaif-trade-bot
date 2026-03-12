# 056# 実装: レビュー指摘への対応

| key | value |
|---|---|
| 番号 | 056 |
| フェーズ | ph2 |
| 種別 | impl |
| 対象文書 | `055_ph2_rev_054.md` |
| 参照 | `054_ph2_plan_profitability_improvement.md` |
| 作成日 | 2025-02-15 |
| 前コミット | `255f94182` (054# impl) |

---

## §1 レビュー検証結果

055# rev の指摘6件を実コードで検証。

| # | 指摘 | 検証結果 | 対応 |
|---|---|---|---|
| 1 | `_rapid_exit_side` 未使用 (HIGH) | **CONFIRMED** — L1130で設定→L1517-18で即クリア→`_next_side()`で参照なし | **P0 修正済** |
| 2 | Smart Side 1サイクル遅延 (HIGH) | **CONFIRMED** — `_next_side()`(L852)→`_compute_maker_price()`(L866)の順 | **P0 修正済** |
| 3 | Round-trip buy→sell のみ (MEDIUM) | **CONFIRMED** — L490-491: sell→buy ペアは無視 | **P0 修正済** |
| 4 | E1 80% vs Gate 90% 不整合 (MEDIUM) | 054#§7.2で意図的に議論済み。分離記載は同意 | ドキュメント分離で対応 |
| 5 | 同時有効化で効果分離不能 (MEDIUM) | リスクは妥当だが、デッドロック中のA/Bは非現実的 | ログベースで対応 |
| 6 | テストが設定マッピングのみ (MEDIUM) | **CONFIRMED** — 18テスト全てがconfig/FillRecord検証 | **P0 修正済** |

### レビュアーの見落とし

- `_next_side()` は sync / `_compute_orderbook_imbalance()` は async → side判定前にimbalance取得は `run_single_cycle()` 側が必要
- `_rapid_exit_side` クリアの具体的タイミング: interval短縮ブロック(L1517-18)で消費前にクリアされるメカニズム

---

## §2 P0 修正内容

### Fix #1: `_rapid_exit_side` を `_next_side()` に接続

**ファイル**: `scripts/v460/run_fill_test.py`

**変更**:
1. `_next_side()` 冒頭に `_rapid_exit_side` チェックを追加
   - 設定されていれば優先返却 + フラグクリア
   - Smart Side ロジックよりも優先
2. interval短縮ブロック(L1517-18)での `_rapid_exit_side = None` を削除
   - `_next_side()` が消費するまで保持

**効果**: S3 損切りの「即反転で逆ポジション」が正しく機能

### Fix #2: Smart Side 最新板化

**ファイル**: `scripts/v460/run_fill_test.py`

**変更**:
1. `run_single_cycle()` で `_next_side()` 呼び出し前に imbalance を事前取得
   - `_compute_orderbook_imbalance()` を追加呼び出し
   - 失敗時は前回値フォールバック (graceful degradation)
2. `_compute_maker_price()` 内の imbalance 計算は維持 (offset 計算に使用)

**効果**: side 決定が最新の板情報に基づく

### Fix #3: Round-trip 双方向ペアリング

**ファイル**: `ztb/metrics/fill_quality.py`, `scripts/v460/monitor_fill_test.py`

**変更**:
1. `RoundTripRecord` に `entry_record`, `exit_record`, `direction` フィールド追加
   - `buy_record`/`sell_record` はプロパティで後方互換維持
2. `RoundTripMetrics` に `unpaired_sells`, `net_inventory` フィールド追加
3. `compute_round_trip_metrics()` を inventory-aware 双方向マッチングに改修
   - buy先行: sell が来たら close
   - sell先行: buy が来たら close
4. `monitor_fill_test.py` の表示を双方向対応に更新

**効果**: Smart Side による連続同side発注でも正しく損益評価

---

## §3 テスト追加

**ファイル**: `tests/unit/v460/test_fill_test_config.py`

### Test055NextSideBehavior (14テスト)

| テスト | 検証内容 |
|---|---|
| `test_alternates_buy_sell` | 基本交互ロジック |
| `test_start_side_sell` | start_side=sell |
| `test_rapid_exit_side_forces_side` | rapid_exit_side 優先返却 |
| `test_rapid_exit_side_overrides_smart_side` | Smart Side より優先 |
| `test_rapid_exit_side_clears_after_use` | 1回で消費 |
| `test_suppress_buy_on_strong_sell_pressure` | suppress: 売り圧力で buy 抑制 |
| `test_suppress_sell_on_strong_buy_pressure` | suppress: 買い圧力で sell 抑制 |
| `test_suppress_no_action_below_threshold` | 閾値以下で抑制なし |
| `test_suppress_max_consecutive_forces_base` | 連続上限で強制 |
| `test_follow_buy_on_positive_imbalance` | follow: 正imbalanceでbuy追従 |
| `test_follow_sell_on_negative_imbalance` | follow: 負imbalanceでsell追従 |
| `test_follow_max_consecutive_limits` | follow: 連続上限 |

### Test055RoundTripBidirectional (9テスト)

| テスト | 検証内容 |
|---|---|
| `test_buy_sell_pair` | 標準 buy→sell |
| `test_sell_buy_pair` | sell→buy (新機能) |
| `test_mixed_directions` | 混在ペアリング |
| `test_unpaired_sells_tracked` | 未ペア sell 追跡 |
| `test_unpaired_buys_tracked` | 未ペア buy 追跡 |
| `test_net_inventory` | 純在庫計算 |
| `test_backward_compat_buy_sell_record_properties` | 後方互換 buy_first |
| `test_backward_compat_sell_first_properties` | 後方互換 sell_first |
| `test_consecutive_same_side_then_close` | 連続same-side全ペアリング |

---

## §4 テスト結果

```
tests/unit/v460/ → 569 passed (548 既存 + 21 新規)
```

既存テスト (`Test051RoundTripMetrics` 5件含む) は後方互換プロパティにより全て合格。

---

## §5 変更ファイル一覧

| ファイル | 変更量 |
|---|---|
| `scripts/v460/run_fill_test.py` | +27/-2 |
| `scripts/v460/monitor_fill_test.py` | +8/-4 |
| `ztb/metrics/fill_quality.py` | +96/-29 |
| `tests/unit/v460/test_fill_test_config.py` | +298/-6 |
| `docs/v460/056_ph2_impl_review_response.md` | 新規 |
