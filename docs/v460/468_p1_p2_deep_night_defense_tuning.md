# 468# P1/P2 deep-night 防御強化・閾値チューニング

**日時**: 2025-07-22
**前提**: 467# 残課題 (P1×2, P2×1) の解消
**データ根拠**: 461# §5.6 時間帯分析 (f840d0e, JST)

## 変更概要

467# で残課題として記録された3件を一括解消:

| # | 優先度 | 内容 | 対処 |
|---|--------|------|------|
| 461# | P1 | ranging_low_vol_skip 閾値最適化 | threshold 0.75→0.65, boost 1.5→1.8 |
| 461# | P1 | sell AS defense 時間帯拡張 | 5時間帯追加、UTC 13 強化 (1.3→2.0) |
| 461# | P2 | deep-night ceiling 緩和 | hour_ceiling_mult 5時間帯設定 + hour_offsets 3時間帯追加 |

## 1. P1: ranging_low_vol_skip 閾値最適化

### 変更
- `low_vol_threshold`: 0.75 → **0.65**
- `low_vol_offset_boost`: 1.5 → **1.8**

### 根拠
- 比例モード (`low_vol_boost_proportional: true`) が有効なため、閾値付近では boost ≒ 1.0x
- 0.75 は安全マージンとして広すぎ、vol_ratio 0.65-0.75 の帯域で偽陽性が発生
- 0.65 に下げることで真の低ボラ環境のみを対象とし、boost max を 1.8x に強化して防御力を確保
- 比例補間: vol_ratio=0.30 → boost=1.41x (旧: 1.20x), vol_ratio=0.50 → boost=1.21x (旧: 1.17x)

## 2. P1: sell AS defense 時間帯拡張

### sell_hour_offset_boost 変更

| UTC | JST | AS% | PnL 30s | 旧値 | 新値 | 備考 |
|-----|-----|-----|---------|------|------|------|
| 9 | 18h | 50.0% | -3.19 | — | **1.8** | 新規追加 |
| 11 | 20h | 33.3% | -1.37 | — | **1.3** | 新規追加 |
| 13 | 22h | 57.1% | -3.18 | 1.3 | **2.0** | 強化 (AS 57% に対して 1.3 は過小) |
| 15 | 00h | 18.2% | -1.93 | — | **1.3** | 新規追加 |
| 17 | 02h | 36.4% | -1.50 | — | **1.5** | 新規追加 |

### カバレッジ改善
- 旧: 8時間帯 (UTC 0,2,3,8,12,13,14,16)
- 新: **13時間帯** (UTC 0,2,3,8,9,11,12,13,14,15,16,17)
- 24h カバー率: 33% → **54%**

## 3. P2: hour_ceiling_mult 設定 + hour_offsets 拡張

### hour_ceiling_mult (新規セクション)

467# で実装済みの仕組みを初めて設定値投入:

| UTC | JST | AS% | ceiling mult | sell ceiling 実効値 |
|-----|-----|-----|-------------|-------------------|
| 13 | 22h | 57.1% | 1.5 | 0.50 → 0.75 |
| 14 | 23h | 100% | 2.0 | 0.50 → 1.00 |
| 15 | 00h | 18.2% | 1.3 | 0.50 → 0.65 |
| 17 | 02h | 36.4% | 1.5 | 0.50 → 0.75 |
| 18 | 03h | 100% | 2.0 | 0.50 → 1.00 |

> hard_skip_utc_hours 拡張ではなく ceiling 緩和を選択した理由:
> - AS 100% の時間帯 (JST 23h, 03h) はサンプル数が極小 (n=1〜2)
> - hard_skip は機会損失が大きい (467# で BTC=0 × ranging block の 7h+ デッドロックが確認済)
> - ceiling 緩和 + sell_boost + hour_offset の三層防御で十分な抑制力を確保

### hour_offsets 追加 (skip_gate 閾値厳格化)

| UTC | JST | AS% | offset | 備考 |
|-----|-----|-----|--------|------|
| 13 | 22h | 57.1% | **0.3** | 新規追加 |
| 15 | 00h | 18.2% | **0.2** | 新規追加 |
| 17 | 02h | 36.4% | **0.3** | 新規追加 |

### 三層防御の全体マトリクス (deep-night UTC 13-18)

| UTC | JST | hour_offset | sell_boost | ceiling_mult | hard_skip |
|-----|-----|-------------|------------|-------------|-----------|
| 13 | 22h | 0.3 | 2.0 | 1.5 | — |
| 14 | 23h | 0.5 | 1.5 | 2.0 | — |
| 15 | 00h | 0.2 | 1.3 | 1.3 | — |
| 16 | 01h | 0.5 | 1.5 | — | ✅ |
| 17 | 02h | 0.3 | 1.5 | 1.5 | — |
| 18 | 03h | 0.3 | — | 2.0 | — |

## テスト

- 全 3006+ テスト合格 (0 failed)
- `test_168_low_vol_offset_boost.py`: YAML parsing アサーション更新 (1.5→1.8, 0.75→0.65)
- `test_336_yaml_code_drift_prevention.py`: `low_vol_threshold` を KNOWN_YAML_OVERRIDES に追加

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/fill_test.yaml` | low_vol 閾値変更、sell_boost 5時間帯追加、hour_ceiling_mult 設定、hour_offsets 3時間帯追加 |
| `tests/unit/v460/test_168_low_vol_offset_boost.py` | YAML parsing テストのアサーション値更新 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | low_vol_threshold を KNOWN_YAML_OVERRIDES に追加 |
| `docs/v460/468_p1_p2_deep_night_defense_tuning.md` | 本ドキュメント |
