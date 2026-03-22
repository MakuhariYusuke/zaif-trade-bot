# 553# OHLCV 自動更新パイプライン

## 概要

552# で特定された SAC sidecar 停止の根本原因 — OHLCV parquet の更新停止 (12日間) — を
解消するため、訓練データの自動更新パイプラインを構築した。

## 背景

| 問題 | 詳細 |
|------|------|
| 原因 | `data/btc_jpy_1m_full_registry_features.parquet` が 2026-03-11 以降未更新 |
| 影響 | retrain_scheduler が同一データで繰り返し OOS 失敗、sidecar が neutral fallback 継続 |
| 目標 | yfinance → FeatureRegistry → parquet の自動更新 + stale data guard |

## 実装

### 1. `scripts/v460/ml/update_training_data.py` (新規作成)

CLI + ライブラリ (retrain_scheduler 呼出) の二面対応。

#### 主要関数

| 関数 | 役割 |
|------|------|
| `update_training_parquet()` | yfinance BTC-JPY 1m ダウンロード → FeatureRegistry 特徴量計算 → parquet マージ |
| `ensure_data_fresh()` | データ鮮度チェック (48h 閾値) + 自動更新。retrain_scheduler から呼出 |
| `_download_ohlcv()` | yfinance API ラッパー |
| `_compute_features()` | FeatureRegistry 登録済み特徴量の一括計算 |
| `_merge_into_parquet()` | 既存 parquet との重複排除マージ + ソート |
| `_get_all_parquet_features()` | parquet 既存列と FeatureRegistry の交差 → 計算対象特徴量リスト |

#### 設計ポイント

- **ウォームアップ**: 既存 parquet 末尾 500 行を結合してから FeatureRegistry 計算 → RSI 等のローリング指標の初期値安定化
- **特徴量自動検出**: `_get_all_parquet_features()` で parquet 既存列 ∩ FeatureRegistry 登録名 → SAC 17 特徴量を必ず含む
- **重複排除**: timestamp ベースの drop_duplicates (keep="last") → yfinance の微妙なズレに対応
- **float32 統一**: SAC 推論との型互換性を維持

#### 使い方

```bash
# CLI — yfinance 更新
python scripts/v460/ml/update_training_data.py

# ライブラリ (retrain_scheduler から)
from scripts.v460.ml.update_training_data import ensure_data_fresh
updated = ensure_data_fresh(parquet_path, max_stale_hours=48)
```

### 2. `sac_retrain_scheduler.py` stale data guard

`retrain_once()` の先頭に `ensure_data_fresh()` 呼出を追加。
48 時間以上データが古い場合、retrain 前に自動更新を試行する (552# step 0)。

## 実行結果

| メトリクス | 値 |
|------|------|
| 更新前行数 | 1,216,930 |
| 更新後行数 | 1,225,300 |
| 追加行数 | 8,370 |
| 最終タイムスタンプ | 2026-03-22 |
| SAC 17 特徴量 NaN 率 | 0% (rolling window 内) |

## テスト

`tests/unit/v460/test_552_update_training_data.py` — 15 テスト全 PASS

主要テスト:
- `_download_ohlcv` のモック検証
- `_compute_features` のウォームアップ動作
- `_merge_into_parquet` の重複排除・ソート
- `ensure_data_fresh()` の鮮度判定 + 自動更新フロー
- `_get_all_parquet_features` の特徴量名フィルタリング

## 変更ファイル

| ファイル | 変更 |
|------|------|
| `scripts/v460/ml/update_training_data.py` | 新規 (357行) — OHLCV 更新パイプライン |
| `scripts/v460/ml/sac_retrain_scheduler.py` | stale data guard 追加 (+8行) |
| `tests/unit/v460/test_552_update_training_data.py` | 新規 (234行) — 15 テスト |
| `CHANGELOG.md` | エントリ追加 |
| `docs/v460/index.md` | エントリ追加 |

## 関連

- **552#**: SAC retrain OOS gate 持続失敗の根本原因調査 (本実装の動機)
- **554#**: raw trades ギャップ充填 (本モジュールに `fill_gap_from_raw()` を追加)
- **555#**: CalibrationMap ランタイム統合 (554# のオフラインバッチ出力を使用)

## コミット

`b907e8fa7` — feat: 553# OHLCV auto-update pipeline for SAC retrain
