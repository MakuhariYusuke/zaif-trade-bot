# 512# Cross-venue ヒント機構スマート化

## 概要
Cross-venue lead-lag ヒント機構（bitFlyer API 情報利用）の活用度向上。
favorable-side tightening 新機能追加 + DRY 化 + 周期サマリ統合。

## 背景・動機
- ヒント発火率 6-12% のうち、実際に offset 変更されるのは 1-3% 程度
- adverse-side retreat のみ実装（439# 初期設計の保守的選択）
- favorable 側（信号方向に有利な側）は完全に無視されていた
- microprice 計算が fill_cycle_executor 内で 2 箇所重複
- fill_record_builder の `getattr` によるゆるい属性アクセス
- cross-venue 活動の周期サマリが存在せず、運用での把握が困難

## 変更内容

### A. Favorable-side offset tightening（新機能）
**理論的根拠**: Glosten (1994) — 方向性情報を持つ MM は有利側でより積極的に指値可能

- **adverse 側**: 既存の retreat/veto 機構はそのまま維持
- **favorable 側**: confidence 比例で offset を縮小し fill rate 向上
  - `tighten_factor = 1.0 - (1.0 - favorable_tighten_mult) × confidence`
  - conf=1.0, mult=0.90 → 10% 縮小
  - conf=0.5, mult=0.90 → 5% 縮小（控えめ）
- **安全設計**:
  - `favorable_tighten_enabled: false` デフォルト無効
  - confidence 比例で低信頼度信号は影響微小
  - hot-reload 対応で即座に ON/OFF 可能

### B. DRY microprice ヘルパー
- `compute_microprice(bids, asks) -> float | None` を `cross_venue_lead_lag.py` に追加
- Gatheral (2018) weighted midprice: `(Pb·Qa + Pa·Qb) / (Qa + Qb)`
- `fill_cycle_executor.py` の 2 箇所重複コード（local/reference）を統一

### C. getattr 排除
- `fill_record_builder.py` の 6 箇所 `getattr` → 直接属性参照
- 型安全性向上 + IDE 補完有効化

### D. Cross-venue 周期サマリカウンター
- `RunSessionState` に 5 カウンター追加:
  - `cv_hint_count`: ヒント発火数
  - `cv_retreat_count`: adverse retreat 適用数
  - `cv_tighten_count`: favorable tighten 適用数
  - `cv_veto_count`: veto 拒否数
  - `cv_cap_hit_count`: offset ceiling 到達数
- progress log に `[512# cross_venue]` サマリ出力

## 設定フィールド（新規）
| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `cross_venue_favorable_tighten_enabled` | bool | false | favorable 側 tightening 有効化 |
| `cross_venue_favorable_tighten_mult` | float | 0.90 | confidence=1.0 時の offset 乗率 |

## 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `cross_venue_lead_lag.py` | `compute_microprice()` ヘルパー追加 |
| `maker_risk_guards.py` | favorable-side tightening 分岐追加 |
| `fill_config.py` | 2 設定フィールド追加 |
| `fill_config_parser.py` | YAML マッピング追加 |
| `config_hot_reload.py` | hot-reload 対象追加 |
| `fill_cycle_executor.py` | DRY microprice 適用 + import |
| `fill_record_builder.py` | getattr → 直接参照 |
| `fill_loop_orchestrator.py` | RunSessionState に 5 カウンター |
| `orchestrator_post_cycle.py` | カウンター増分 + サマリログ |
| `fill_test.yaml` | favorable_tighten 設定追加 |

## テスト
- `test_439_cross_venue_lead_lag.py`:
  - `TestComputeMicroprice`: 5 テスト（基本計算、等量、空book、ゼロ数量）
  - `TestFavorableSideTightening`: 3 テスト（有効時縮小、無効時不変、confidence 比例）
  - `_make_hint` に `confidence` パラメータ追加
  - `_CrossVenueState` に private 属性追加（getattr 排除対応）

## 運用ガイド
1. まず `favorable_tighten_enabled: false` でデプロイ（サマリカウンターのみ有効）
2. `[512# cross_venue]` ログで hint/retreat/veto/cap_hit 比率を監視
3. hint 率が安定していることを確認後、`favorable_tighten_enabled: true` に切替
4. PnL と fill rate の変化を観察
5. 必要に応じて `favorable_tighten_mult` を 0.85-0.95 の範囲で調整
