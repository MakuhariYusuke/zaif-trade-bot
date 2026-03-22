# 554# Raw Data Gap Fill + CalibrationMap Offline Batch

## 概要

553# で構築した OHLCV 更新パイプラインに raw trades からのギャップ充填機能を追加。
併せて 546# §B で推奨された CalibrationMap のオフラインバッチ構築を実装した。

## 1. Raw Trades → Parquet Gap Fill

### 背景

yfinance は直近 7 日分しか 1 分足を提供しない。
2026-02-13 ～ 03-15 の期間に parquet データの欠損があり、
`data/v460/raw/trades/` に保存されていた生取引 JSONL.gz ファイルを活用してギャップを埋めた。

### 実装: `update_training_data.py` に追加

| 関数 | 役割 |
|------|------|
| `_raw_trades_to_ohlcv_1min()` | JSONL.gz (ts, price, amount) → 1分足 OHLCV 変換 |
| `fill_gap_from_raw()` | parquet 内のギャップ検出 → raw trades → OHLCV → 特徴量計算 → マージ |

#### ギャップ検出ロジック

```
parquet 内の各日付の行数をカウント
  → 60 行 (= 1 時間分) 未満 → ギャップとして検出
  → raw trades に対応ファイルがあれば充填対象
```

#### 使い方

```bash
python scripts/v460/ml/update_training_data.py --raw-fill
```

### 実行結果

| メトリクス | 値 |
|------|------|
| ギャップ日数 | 29 日 (Feb 13 ～ Mar 15) |
| 追加行数 | 22,004 bars |
| 更新前行数 | 1,225,448 |
| 更新後行数 | 1,247,452 |
| カバレッジ | ほぼ連続 (Feb 13 以降) |

## 2. CalibrationMap Offline Batch

### 背景

546# §B (b) の推奨: fill_records からオフラインで CalibrationMap を構築し、
fill_test 起動時にロードすることで cold start 問題を回避する。

### 実装: `scripts/v460/ml/calibration_batch.py` (新規)

| 関数 | 役割 |
|------|------|
| `build_calibration_map()` | fill_records JSONL → CalibrationMap 構築 → JSON エクスポート |
| `load_calibration_state()` | JSON → CalibrationMap インスタンス復元 (fill_test 起動用) |
| `_side_to_action()` | buy→+1.0 / sell→-1.0 変換 |

#### データフロー

```
results/v460/fill_test/*.jsonl
  → iter_fill_records_glob() で読込
  → filled=True & pnl≠None をフィルタ
  → CalibrationMap.update(regime, action, gross_pnl, step)
  → JSON エクスポート (models/v460/entry_gate_calibration.json)
```

#### PnL ソース優先順位

1. `post_fill_30s_pnl` (primary)
2. `ev_weighted_pnl` (fallback)

#### 使い方

```bash
# 全期間で構築
python scripts/v460/ml/calibration_batch.py

# 直近 14 日のみ
python scripts/v460/ml/calibration_batch.py --days 14
```

### 実行結果

| メトリクス | 値 |
|------|------|
| 全レコード | 15,531 (38 日分) |
| filled レコード | 4,718 |
| Global n_eff | ≈ 200 |
| Global p_win_lcb | ≈ 0.38 |

#### Regime 別統計

| Regime | n_eff | p_win_lcb | 備考 |
|--------|-------|-----------|------|
| ranging | ≈ 121 | 0.348 | 主要レジーム |
| trending | ≈ 73 | 0.387 | |
| high_vol | ≈ 21 | — | n_min 未到達 → fallback |
| unknown | ≈ 10 | — | n_min 未到達 → fallback |

### CalibrationMap パラメータ

| パラメータ | 値 | 説明 |
|------|------|------|
| ewma_tau | 100.0 | EWMA 減衰定数 |
| n_min | 30.0 | 有効サンプル数の最小閾値 |
| prior (α, β) | (2, 2) | Beta 事前分布 (uninformative) |
| 階層 fallback | L1→L2→L3 | regime+action → regime → global |

### 出力

- `models/v460/entry_gate_calibration.json` (gitignored — ローカルのみ)
- メタデータ: 構築日時、レコード数、regime 分布、設定値を同梱

## テスト

### 新規テスト

`tests/unit/v460/test_554_calibration_batch.py` — 11 テスト全 PASS

- `build_calibration_map()` のレコード処理とフィルタリング
- `_side_to_action()` のサイド変換
- `load_calibration_state()` の JSON 復元
- `--days` フィルタの動作
- 欠損 PnL / 未 filled レコードの除外

### 既存テスト

553# の 15 テストも含め 26 テスト全 PASS (11 + 15)

## 変更ファイル

| ファイル | 変更 |
|------|------|
| `scripts/v460/ml/calibration_batch.py` | 新規 (248行) — CalibrationMap batch builder |
| `scripts/v460/ml/update_training_data.py` | `_raw_trades_to_ohlcv_1min()` + `fill_gap_from_raw()` 追加 (+155行) |
| `tests/unit/v460/test_554_calibration_batch.py` | 新規 (262行) — 11 テスト |
| `CHANGELOG.md` | エントリ追加 |
| `docs/v460/index.md` | エントリ追加 |

## 関連

- **546# §B**: CalibrationMap 推奨 (本実装の設計根拠)
- **552#**: SAC retrain 調査 (データ更新停止の発見)
- **553#**: OHLCV 自動更新パイプライン (本実装の前提)
- **555#**: CalibrationMap ランタイム統合 (本バッチ出力を起動時ロード)

## ztb 既存実装の活用

| コンポーネント | パス | 活用内容 |
|---|---|---|
| CalibrationMap | `ztb/trading/signal/calibration_map.py` | EWMA 統計 + 階層的 fallback |
| iter_fill_records_glob | `ztb/metrics/fill_quality.py` | fill_records JSONL の一括読込 |

## コミット

`447b2ec50` — feat: 554# Raw data gap fill + CalibrationMap offline batch
