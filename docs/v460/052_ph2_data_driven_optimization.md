# 052# Fill Test 改善 Phase 2 — データ駆動パラメータ最適化

## 状態

- HEAD: `fa369b9c6` → 本コミット
- 前回: 051# `3fcce8e57` (proactive 5 items)
- テスト: 530 passed (525 → +5)

## データ分析結果 (n=491, filled=373)

### サイド別 PnL (30s)

| Side | n | PnL mean | PnL median | AS | AS_raw |
|------|---|----------|-----------|-----|--------|
| buy | 192 | -0.301 | -0.160 | 39.6% | 38.0% |
| sell | 181 | -0.958 | -0.094 | 38.7% | 41.4% |

### Multi-Timeframe PnL

| TF | n | mean | win% |
|----|---|------|------|
| 30s | 373 | -0.620 | 47.2% |
| 60s | 26 | -0.620 | 46.2% |
| 120s | 26 | +0.101 | 53.8% |

### Round-Trip

- 181 pairs, Mean +10.318 bps, WinRate 54.7%, Total ~2007.6 JPY

### UTC 時間帯 (WARNING = AS>40% or PnL<-1.0)

| UTC | n | PnL | AS% | Status |
|-----|---|-----|-----|--------|
| 00 | 25 | +0.337 | 28% | ✅ |
| 01 | 27 | -0.814 | 41% | ⚠ NEW |
| 02 | 16 | -1.272 | 44% | ⚠ NEW |
| 03 | 17 | +0.918 | 24% | ✅ |
| 08 | 15 | -3.805 | 47% | ⚠ 既存 |
| 13 | 28 | -1.070 | 29% | ⚠ NEW |
| 21 | 26 | -1.647 | 38% | ⚠ NEW |

### レジーム PnL

| Regime | n | PnL |
|--------|---|-----|
| ranging | 53 | -0.504 |
| trending | 19 | -1.209 |
| unknown | 34 | -1.709 |
| None | 267 | -0.462 |

### Daily Trend (悪化傾向)

| Date | n | PnL | win% |
|------|---|-----|------|
| 0213 | 163 | -0.441 | 48.5% |
| 0214 | 161 | -0.724 | 47.2% |
| 0215 | 49 | -0.875 | 42.9% |

## 実装内容

### 1. `_MIN_ORDER_BTC` 修正 (Critical Bug Fix)

- 0.0005 → **0.001** (Coincheck API 最小注文量)
- 前回 0.0005 で API rejection loop に陥った
- `_check_balance_for_side()` のコメントも修正

### 2. balance_shrink の最低ロット統一

- `run_continuous()` の P2-3 balance_shrink で `config.order_quantity` を使っていたが、
  `max(config.order_quantity, _MIN_ORDER_BTC)` に統一
- Coincheck 最低ロット以下への縮小を防止

### 3. UTC スキップ時間帯追加

- 既存: `[8, 9, 12, 14, 16, 17, 18, 19]`
- 追加: **UTC 1, 2, 13, 21** (全て PnL < -0.8bps or AS > 40%)
- 新: `[1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21]` (12/24時間スキップ)

### 4. トレンディング offset ブースト

- trending レジームの PnL = -1.209 bps (最悪)
- `_compute_maker_price()` で trending 検出時に offset × 1.5
- `regime_trending_offset_boost: 1.5` を YAML + Config に追加
- サイクル冒頭でレジーム情報がなくても、前サイクルの regime を使用

### 5. テスト (+5)

| Test | 内容 |
|------|------|
| `test_yaml_skip_utc_hours_includes_13_and_21` | UTC 1,2,13,21 スキップ確認 |
| `test_trending_offset_boost_in_code` | trending ブーストコード存在確認 |
| `test_trending_offset_boost_config` | Config フィールド 1.5 確認 |
| `test_yaml_trending_offset_boost` | YAML 設定確認 |
| `test_balance_shrink_uses_min_order_btc` | run_continuous に _MIN_ORDER_BTC 使用確認 |

## 残高状況

- JPY: 9,006.86 / BTC: 0.00093
- **デッドロック**: 両サイドとも Coincheck 最低注文 0.001 BTC を満たせない
- 再起動には追加入金が必要 (最低 ~2,000 JPY or 0.0002 BTC)

## 変更ファイル

- `scripts/v460/run_fill_test.py` — 4 変更 (MIN_ORDER fix, shrink guard, trending boost, config field)
- `configs/v460/fill_test.yaml` — 3 変更 (UTC skip, trending boost, MIN_ORDER コメント)
- `tests/unit/v460/test_fill_quality.py` — 5 テスト追加 + 1 修正
- `docs/v460/052_ph2_data_driven_optimization.md` — 本ドキュメント
