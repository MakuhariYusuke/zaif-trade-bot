# 051# Phase 2 残課題の先行実装

- **番号**: 051
- **フェーズ**: Phase 2 (fill test 実行中)
- **作成日**: 2025-02-14
- **前提**: 050# Bug#1-7 修正済み (`c20925f96`)
- **fill test**: PID 53560 稼働中 (SHA `51c02be69`)。本変更は **再起動不要** で反映可能な項目のみ優先実装。

---

## §1 実装サマリ

| # | 項目 | 優先度 | 対象ファイル | 再起動 |
|---|------|--------|-------------|--------|
| 1 | P2-2: Round-trip 評価 (buy→sell FIFO ペアリング) | HIGH | `fill_quality.py`, `monitor_fill_test.py` | 不要 |
| 2 | UTC 時間帯別分析 (time_filter 検証) | MEDIUM | `fill_quality.py`, `monitor_fill_test.py` | 不要 |
| 3 | P2-4: レジーム別メトリクス | MEDIUM | `fill_quality.py`, `monitor_fill_test.py` | 不要 |
| 4 | Monitor clean_rate 表示拡張 | LOW | `monitor_fill_test.py` | 不要 |
| 5 | P2-3: Balance auto-shrink | HIGH | `run_fill_test.py` | **要** |

---

## §2 実装詳細

### 2.1 P2-2: Round-trip 評価

**目的**: 個別サイクルの 30 秒 PnL ではなく、buy→sell の往復実損益を投資家目線で把握。

**ロジック**:
- 時系列順の約定済みレコードを FIFO でペアリング (buy キュー → sell でマッチ)
- `pnl_bps = (sell_fill - buy_fill) / buy_fill × 10000`
- `pnl_jpy = (sell_fill - buy_fill) × min(qty_buy, qty_sell)`
- `hold_sec = sell_timestamp - buy_timestamp`

**新規データクラス**:
- `RoundTripRecord`: buy/sell レコード、PnL、保持時間
- `RoundTripMetrics`: 件数、PnL 統計、勝率、保持時間中央値、未ペア buy 数

**表示**: モニタに `🔄 Round-trip 評価` セクション追加。

### 2.2 UTC 時間帯別分析

**目的**: `time_filter.skip_utc_hours` の妥当性を検証。UTC 13 を追加すべきか判断するためのデータ基盤。

**ロジック**:
- 全レコードを UTC hour でグループ
- hour 毎の n / filled / PnL mean / AS ratio を算出
- AS > 45% の hour に ⚠ フラグ表示

**新規データクラス**: `HourlyMetrics`

**表示**: モニタに `🕐 UTC 時間帯別分析` セクション追加 (3 hour 以上のデータがある場合のみ)。

### 2.3 P2-4: レジーム別メトリクス

**目的**: trending / ranging / high_vol / unknown レジーム毎の fill quality 差異を把握。

**ロジック**:
- FillRecord.regime フィールドでグループ
- レジーム毎の n / fill% / PnL / AS% / wait 中央値を算出

**新規データクラス**: `RegimeMetrics`

**表示**: モニタに `🌊 レジーム別メトリクス` セクション追加 (unknown のみの場合はスキップ)。

### 2.4 Monitor clean_rate 表示拡張

**目的**: 049# §6.1 項目 4「データ品質 (clean_rate)」をモニタレポートに表示。

**変更**:
- `run_monitor()` が `filter_clean_records()` で clean/quarantine を分離
- **clean レコードのみ**でメトリクス算出 (quarantine 汚染防止)
- `print_report()` に `🧹 データ品質` セクション追加
- `print_report()` のシグネチャに `clean_count`, `quarantine_count` キーワード引数追加

### 2.5 P2-3: Balance auto-shrink

**目的**: 残高不足時に SAFE_STOP ではなくロット縮小で稼働を継続。

**ロジック**:
1. `_preflight_skip_count >= 3` かつ `_current_lot > order_quantity` かつ未発動 → ロット半減
2. `_balance_shrink_active = True` + `_pre_shrink_lot` に元ロットを保存
3. カウンタリセット → 縮小ロットで再試行
4. preflight 成功時 → `_pre_shrink_lot` に復元、`_balance_shrink_active = False`

**安全弁**: 
- order_quantity (最小ロット 0.001 BTC) 以下にはならない
- shrink 後も SAFE_STOP (max_preflight_skip=10) は残存
- soft_loss_cap 発動時は `_pre_shrink_lot` も更新 (復元先が過大にならない)

**注意**: この変更は **fill test 再起動後** に反映。

---

## §3 テスト

| クラス | テスト数 | 内容 |
|--------|---------|------|
| `Test051RoundTripMetrics` | 5 | FIFO ペアリング、複数ペア、未ペア buy、空リスト、sell 先行 |
| `Test051RegimeMetrics` | 3 | グルーピング、unknown マッピング、空リスト |
| `Test051HourlyMetrics` | 1 | UTC hour 別グルーピング + PnL/AS |
| `Test051BalanceAutoShrink` | 3 | フィールド存在、ロジック存在、閾値=3 |
| `Test051MonitorExtensions` | 3 | シグネチャ、clean_records 使用、インポート |
| **合計** | **15** | |

**全テスト**: 519 passed (504 → +15)

---

## §4 残課題 (次回以降)

| 項目 | 優先度 | 備考 |
|------|--------|------|
| P2-1: TP/SL tick path 検証 | HIGH | E3 データ蓄積待ち (sampling_ratio=0.33) |
| UTC 13 追加判断 | MEDIUM | 051# の hourly 分析結果を見て判断 → fill_test.yaml 更新 |
| fill test 再起動 | HIGH | 050# Bug#1-3 + 051# P2-3 を反映させるため |
| Round-trip 目標設定 | LOW | RT PnL mean > 0 bps を G1.2 以降の要件に |
