# 443# Cross-Venue 閾値修正 + ログ・FillRecord バグ修正

| 項目 | 内容 |
|---|---|
| 番号 | 443# |
| 分類 | ph2_fix |
| 前提 | 442# (cross-venue 有効化) |
| 目的 | 初回実運用で発覚した閾値・ログ・FillRecord の不具合修正 |

---

## 1. 修正内容

### 1.1 velocity_bps_threshold: 1.0 → 0.05
- 442# の初回運用で hint が一度も発火しなかった
- 原因: 120s cycle interval での velocity は per-second 換算で極小値 (0.01〜0.07 bps/s)
- `velocity_bps_threshold=1.0` は 1s 単位のトレーダー向け設定であり、120s cycle には不適切

### 1.2 ログレベル: debug → warning
- hint=None 時のログが `logger.debug` で本番環境では不可視
- `logger.warning` に昇格して原因の可視化を実現

### 1.3 FillRecord フィールド修正
- `cross_venue_lead_lag_velocity_bps` が正しく記録されていなかった

### 1.4 log_event インデントバグ修正
- `log_event()` 呼出の try-except ブロックのインデントが誤っており、一部ケースで実行されなかった

## 2. 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | `velocity_bps_threshold: 0.05` |
| `scripts/v460/lib/fill_cycle_executor.py` | ログレベル修正、FillRecord 修正、インデント修正 |
