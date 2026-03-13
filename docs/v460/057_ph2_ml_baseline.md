# 057# ML-1/ML-2 ベースライン分類器

**日付**: 2026-02-15  
**コミット**: `679773ea8`

## 概要

fill records から AS (Adverse Selection) 分類器と Fill/Timeout 分類器のベースラインを構築。

## 実装

| ファイル | 目的 |
|---|---|
| `scripts/v460/ml/data_loader.py` | fill_records → ML 特徴量 |
| `scripts/v460/ml/as_classifier.py` | AS 予測 (LogReg / GradientBoosting) |
| `scripts/v460/ml/fill_classifier.py` | Fill 予測 |
| `scripts/v460/ml/run_ml_pipeline.py` | パイプライン実行 |
| `tests/unit/v460/test_ml_pipeline.py` | 16 テスト |

## ベースライン結果 (284 AS-labeled samples)

| モデル | ROC-AUC | PR-AUC | 備考 |
|---|---|---|---|
| AS (GB) | 0.528 | 0.573 | Naive PR-AUC=0.521 |
| AS (LR) | 0.525 | 0.565 | |
| Fill (GB) | 0.454 | 0.853 | クラス不均衡駆動 |

### AS 特徴量重要度 (GB)
1. `log_queue_wait` (0.32)
2. `edge_bps` (0.28)
3. `spread_jpy` (0.24)

### Skip Policy (th=0.5)
- Skip 47%, PnL +0.91 bps improvement

## 判定

**弱い (ROC-AUC ≈ 0.53)** — 現在の特徴量 (fill record メタデータのみ) では情報不足。  
054# 新規フィールド (orderbook_imbalance 等) の実データ蓄積が必要。→ 058# で raw data を活用して改善。
