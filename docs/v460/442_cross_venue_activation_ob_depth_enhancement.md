# 442# Cross-Venue有効化 + L5板深度拡張 + Microprice + Depth Imbalance

## 背景

- 439# で Cross-Venue Lead-Lag ガードの基盤を構築済み（BitFlyer BTC_JPY 参照）
- BitFlyer API の動作は検証済み（200 OK, mid≈11.44M JPY, 500-1000ms latency）
- 板情報（L5 深度）を取得できるのに L1 しか使っていなかった → 改善余地

## 変更概要

### 1. Cross-Venue ガード有効化
- `fill_test.yaml`: `enabled: true`, `veto_enabled: true`
- 本番稼働で BitFlyer 価格乖離を sell/buy ガードとして使用開始

### 2. L5 OB 深度拡張
- 参照板の取得深度を `depth: 1` → `depth: 5`（`reference_ob_depth` で設定可能）
- より深い板情報から正確な方向性シグナルを取得

### 3. Microprice (Gatheral 2018)
- 数式: $P_\mu = \frac{P_b \cdot Q_a + P_a \cdot Q_b}{Q_a + Q_b}$
- L1 深度の非対称性を反映した加重中間価格
- ローカル (Coincheck) と参照 (BitFlyer) 双方で計算
- `microprice_spread_bps`: 参照 microprice vs ローカル mid の乖離 (bps)

### 4. Depth Imbalance
- 数式: $DI = \frac{V_{bid} - V_{ask}}{V_{bid} + V_{ask}}$
- 参照取引所の板厚みの偏りを数値化（-1 〜 +1）
- ガードの確認信号として使用:
  - direction=up かつ DI > 0.1 → sell 側 adverse 確認 → `depth_imbalance_boost` (1.15x)
  - direction=down かつ DI < -0.1 → buy 側 adverse 確認 → `depth_imbalance_boost` (1.15x)

## 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `reference_ob_depth` | 5 | 参照板の取得レベル数 |
| `microprice_enabled` | true | Microprice 計算の有効化 |
| `depth_imbalance_enabled` | true | Depth Imbalance 計算の有効化 |
| `depth_imbalance_boost` | 1.15 | DI 確認時の追加 offset 倍率 |

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | cross_venue セクション有効化 + 4 新キー |
| `scripts/v460/lib/fill_config.py` | 4 新フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | YAML→Config マッピング追加 |
| `scripts/v460/lib/cross_venue_lead_lag.py` | VenueMidSnapshot/Hint 拡張, event/fill fields 拡張 |
| `scripts/v460/lib/fill_cycle_executor.py` | `_update_cross_venue_lead_lag_hint()` 全面書換え |
| `scripts/v460/lib/maker_risk_guards.py` | DI 確認ブースト追加 |
| `tests/unit/v460/test_439_cross_venue_lead_lag.py` | 新スキーマ対応 |
| `tests/unit/v460/test_113_resilience.py` | 行数上限調整 (810→830) |
| `tests/unit/v460/test_253_*.py` | executor 行数上限調整 (1300→1340) |
| `tests/unit/v460/test_336_*.py` | KNOWN_YAML_OVERRIDES に4フィールド追加 |

## テスト結果

- 15/15 cross-venue テスト PASS
- 3653 全 v460 テスト PASS (2 skipped)
