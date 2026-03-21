# 519# Sell Ceiling 引上げ + Favorable Tighten 有効化 + SDK Window 縮小

## 概要
518# 方向修正で特定された P0 アクションの実装。3 つの設定変更を同時適用。

## 背景・根拠

### 518# 検証結果の要約
- **offset_ceiling_ratio_sell=0.20** が pipeline の保護機能を打ち消し、native sell AS avg=−7.31 を引き起こしていた
- forced sell (ceiling=0.25) は AS avg=−4.42 → ceiling 5% 差で 65% の損失改善
- buy 側は ceiling=0.25 で PnL30=+39.0 と健全
- 512# favorable_tighten は実装済み未有効化のまま放置。buy が健全な現在が有効化の適時

## 変更内容

### A. `offset_ceiling_ratio_sell: 0.20 → 0.25` (P0-1)

| 項目 | 内容 |
|------|------|
| **理由** | native sell AS (ceil=0.20) avg=−7.31 vs forced sell AS (ceil=0.25) avg=−4.42 |
| **理論** | pipeline は offset を 0.274 まで正しくリスク評価。ceiling=0.20 が 0.074 分を切り捨て → 保護不足 |
| **効果予測** | sell AS avg PnL の改善 (−7.31 → −4.42 方向)。fill rate は微低下の可能性 |
| **buy との対称性** | buy ceiling は 491# で 0.20→0.25 済み。sell も同値に統一 |

### B. `sell_dynamic_kill.window: 50 → 30` (P0-2)

| 項目 | 内容 |
|------|------|
| **理由** | window=50 で sell kill は稼働中 (4 cancels) だが、直近の損失拡大への反応が遅い |
| **効果** | EWMA の effective window は alpha 依存だが、count-based 判定は 30 fill で早期に反応 |
| **リスク** | 過剰 kill → sell fill rate 低下。resume_window=10 が安全弁 |

### C. `favorable_tighten_enabled: false → true` (512# 有効化)

| 項目 | 内容 |
|------|------|
| **理由** | 512# で実装完了、安全デフォルト (false) で待機中。buy PnL30=+39.0 で健全 → 有効化の適時 |
| **メカニズム** | XV ヒントが favorable (有利方向) のとき、confidence 比例で offset を最大 10% 縮小 → fill rate 向上 |
| **安全設計** | confidence 比例 (低信頼度では影響微小)、hot-reload で即座に OFF 可能 |
| **mult** | 0.90 据置 (conf=1.0 時に 10% 縮小、conf=0.5 時に 5% 縮小) |

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `configs/v460/fill_test.yaml` | 3 パラメータ変更 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | KNOWN_YAML_OVERRIDES に 2 フィールド追加 |

## テスト結果
- `test_336_yaml_code_drift_prevention.py`: 26 passed (ドリフト検出テスト)
- `test_439_cross_venue_lead_lag.py`: 30 passed (favorable tighten テスト含む)
- `test_336_fill_config_parser.py`: 24 passed (config パーサテスト)
- `test_467_remaining_issues.py`: 22 passed (ceiling 関連テスト)
- 関連 4 テストファイル追加実行: 158 passed

## 観察ポイント (次回分析時)
1. **sell AS avg PnL**: −7.31 からの改善度合い
2. **sell fill rate**: ceiling 引上げによる fill rate 変化
3. **favorable tighten 発火率**: `[512# cross_venue]` サマリログで cv_tighten_count を確認
4. **sell_dynamic_kill cancel 数**: window=30 で kill 頻度がどう変化するか
5. **pipeline 出力 vs ceiling 衝突率**: 91/111 (82%) からの変化
