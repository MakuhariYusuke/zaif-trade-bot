# 444# Cross-Venue 閾値チューニング + ログ可視化

| 項目 | 内容 |
|---|---|
| 番号 | 444# |
| 分類 | ph2_fix |
| 前提 | 443# (初回閾値修正) |
| 目的 | 実測データに基づく微調整と hint=None 原因のロギング強化 |

---

## 1. 背景

443# で `velocity_bps_threshold` を 1.0 → 0.05 に修正したが、依然として hint 発火率が低い。
実運用ログを分析した結果、さらなる閾値調整が必要と判明。

## 2. 修正内容

### 2.1 spread_bps_threshold: 2.0 → 1.0
- CC-BF 間の実測スプレッドは通常 0.5〜1.7bps
- 閾値 2.0 では大半のケースで発火しない
- 1.0 に引き下げて捕捉率を向上

### 2.2 velocity_bps_threshold: 0.05 → 0.01
- 120s cycle 間隔での velocity は per-second 換算で 0.01〜0.07bps/s が典型値
- 5.98bps 乖離 + velocity 0.012bps/s のケースを捕捉するには 0.01 が必要

### 2.3 hint=None ログ: debug → info + 阻止理由表示
- `logger.info` に昇格し、具体的な阻止理由を表示:
  - `spread(+0.42)<1.0` — spread 不足
  - `vel(+0.012)<0.05, sign_disagree` — velocity 不足 or 符号不一致
  - `first_call` — 初回呼出 (previous snapshot なし)

## 3. 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | `spread_bps_threshold: 1.0`, `velocity_bps_threshold: 0.01` |
| `scripts/v460/lib/fill_cycle_executor.py` | hint=None ログを info + 具体理由付きに改修 |
