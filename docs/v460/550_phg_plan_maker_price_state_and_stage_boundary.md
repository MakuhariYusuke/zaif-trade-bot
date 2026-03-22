# 550# PHG: MakerPrice state classification and compute stage boundary

## 目的

`scripts/v460/lib/maker_price.py` の carry-forward 設計を、

1. state の分類
2. `compute()` の stage 実行順序
3. 将来の分割境界

の 3 点で固定する。

本書は 521# の `maker_price` 設計メモを具体化した補助計画であり、
現時点では **state object 化を急がず**、stage orchestration を明示化するための基準文書とする。

## 前提

- public 契約は維持する
  - `MakerPriceCalculator.compute()`
  - `MakerPriceResult`
  - `last_offset_stages`
  - source-inspection 契約
- pure helper は継続して `ztb.trading.pricing.*` へ寄せる
- stateful orchestration は `scripts/v460/lib/maker_price.py` に残す

## 現状サマリ

2026-03-23 時点の `MakerPriceCalculator` state は **45 フィールド**。
547# の時点では 44 相当の想定だったが、その後

- `533#` veto state
- `366#` fill probability model

が明示 state として定着しており、最新 HEAD では 45 が正である。

### 既に helper 化済みの純ロジック

- `inventory_math.py`
- `offset_math.py`
- `offset_amount.py`
- `offset_ceiling.py`
- `boost_math.py`
- `spread_adaptive.py`
- `price_finalization.py`
- `stage_tracking.py`
- `ofi_lite.py`
- `contracts.py`

### 既に mixin 化済みの領域

- `maker_risk_guards.py`
- `maker_microstructure.py`
- `maker_regime_boost.py`

## State 分類

### 1. Pricing Core State

価格決定と pipeline 制御に直接必要な state。将来 split する場合も `MakerPriceCalculator` の中核 ownership に残す。

| field | 役割 | 将来の境界 |
|---|---|---|
| `_config` | pricing 全体の設定 | core ownership のまま |
| `_fast_fill_defense` | FFD stage 実行 | `risk adapter` 的 wrapper 候補 |
| `_regime_detector` | regime stage 入力 | `signal adapter` 候補 |
| `_base_offset_ratio` | 共通 base ratio | core |
| `_base_offset_ratio_buy` | buy base ratio override | core |
| `_base_offset_ratio_sell` | sell base ratio override | core |
| `_loss_boost_mult` | loss boost state | `pricing core state` |
| `_loss_boost_set_time` | loss boost decay 基準時刻 | `pricing core state` |
| `_consecutive_veto_count` | cross-venue veto deadlock 制御 | `risk/orchestration bridge` |
| `_veto_btc_balance` | veto 緩和用 BTC 残高 | `risk/orchestration bridge` |
| `_fill_prob_model` | GLFT dynamic-k 入力 | `microstructure adapter` 候補 |
| `_inv_fill_history` | inventory skew 履歴 | `inventory state` 候補 |
| `_inv_net_imbalance` | inventory skew 現在値 | `inventory state` 候補 |
| `_inv_buy_count` | O(1) buy count | `inventory state` 候補 |
| `_inv_last_update_time` | inventory decay 基準時刻 | `inventory state` 候補 |

### 2. Microstructure Cache

板/価格/流動性/ボラの cache。`compute()` の前半と各 `_apply_*` stage の入力として再利用される。

| field | 役割 | 将来の境界 |
|---|---|---|
| `_prev_mid_price` | velocity 計算の前回 mid | `market cache` |
| `_prev_mid_time` | velocity 計算の前回時刻 | `market cache` |
| `_last_mid_trend_bps` | 最新 velocity | `market cache` |
| `_last_imbalance` | imbalance cache | `market cache` |
| `_last_bid_depth` | bid depth cache | `market cache` |
| `_last_ask_depth` | ask depth cache | `market cache` |
| `_last_vpin` | VPIN cache | `market cache` |
| `_last_ob_snapshot` | current OB snapshot | `orderbook cache` |
| `_prev_ob_snapshot` | previous OB snapshot | `orderbook cache` |
| `_last_spread` | latest spread | `market cache` |
| `_last_spread_time` | spread staleness | `market cache` |
| `_smoothed_velocity_bps` | EMA velocity | `market cache` |
| `_last_amihud_illiq` | Amihud ILLIQ cache | `microstructure cache` |
| `_last_as_delta_star_ratio` | A-S delta* cache | `microstructure cache` |
| `_mid_high` | Parkinson sigma high | `volatility cache` |
| `_mid_low` | Parkinson sigma low | `volatility cache` |
| `_mid_hl_reset_time` | sigma window reset | `volatility cache` |
| `_last_sigma` | latest sigma | `volatility cache` |
| `_last_ofi_lite` | OFI-Lite latest value | `orderbook flow cache` |
| `_ofi_history` | OFI rolling history | `orderbook flow cache` |

### 3. Telemetry / Diagnostic

観測・FillRecord・post-analysis に使う state。計算本体の correctness より、説明責任と分析のための state。

| field | 役割 | 将来の境界 |
|---|---|---|
| `_last_vg_triggered` | VG 発動フラグ | `telemetry recorder` |
| `_last_vg_velocity_bps` | VG 判定時 velocity | `telemetry recorder` |
| `_last_vg_vpin` | VG 判定時 VPIN | `telemetry recorder` |
| `_last_vg_boost_factor` | VG boost factor | `telemetry recorder` |
| `_last_vg_reason` | VG 理由 | `telemetry recorder` |
| `_cross_venue_lead_lag_hint` | current hint snapshot | `telemetry + risk bridge` |
| `_cross_venue_lead_lag_vetoed` | veto 発火状態 | `telemetry + risk bridge` |
| `_cross_venue_lead_lag_veto_reason` | veto 理由 | `telemetry + risk bridge` |
| `_last_inv_skew_factor` | inventory skew 適用係数 | `telemetry recorder` |
| `_last_offset_stages` | serialized offset stage trace | `telemetry recorder` |

## 分割境界の見立て

### 今やってよい分割

1. pure helper の継続抽出
2. telemetry の schema/version 固定
3. stage apply / stage record / result serialization の重複削減

### まだ急がない分割

1. state object 化
2. `compute()` の multi-object orchestration 化
3. mixin の再シャッフル

理由:
- source-inspection 契約がまだ多い
- `compute()` は stage 順序の意味が強い
- 先に pure helper と telemetry 契約を固めるほうが破壊半径が小さい

## `compute()` stage 実行シーケンス

### 前処理 / bail-out

| step | 処理 | 入力 | 出力/影響 | 依存 |
|---|---|---|---|---|
| P1 | imbalance cache resolve | `_last_imbalance` | `imb` | 独立 |
| P2 | orderbook resolve | `_last_ob_snapshot` / adapter | `ob` | 独立 |
| P3 | spread / mid 計算 | `ob` | `spread`, `mid_price` | P2 |
| P4 | velocity update | `mid_price`, `_prev_mid_*` | `mid_trend_bps`, cache更新 | P3 |
| P5 | spread guard | `spread`, cfg | early raise | P3 |
| P6 | sell spread guard | `spread`, side, cfg | early raise | P3 |
| P7 | none/unknown passive MM bypass | regime, spread, mid | early return | P3 |

### Core offset pipeline

| step | stage | 主入力 | 主出力 | 依存 |
|---|---|---|---|---|
| S0 | base ratio resolve | side, base config | `effective_offset_ratio` | P3 |
| S1 | stage store seed | cfg, OFI cache, delta* cache | `_stages` | S0 |
| S2 | inventory skew pre-pass | inventory state | ratio 更新 | S0 |
| S3 | sell floor pre-pass | side, inv state | ratio floor | S2 |
| S4 | `as_shift` | side, spread, mid, sigma | ratio 更新 | S3 |
| S5 | `regime` | side, regime detector | ratio 更新 | S4 |
| S6 | `spread_adapt` | side, spread, mid, OFI | ratio 更新 | S5 |
| S7 | `kyle` | side, spread, mid | ratio 更新 | S6 |
| S8 | `amihud` | side, spread, mid | ratio 更新 | S7 |
| S9 | `vol_guard` | side, velocity, VPIN | ratio 更新 | S8 |
| S10 | `cross_venue` | side, hint, veto state | ratio 更新 / veto | S9 |
| S11 | `imb_risk` | side, imbalance | ratio 更新 | S10 |
| S12 | `buy_as_guard` | side, velocity | ratio 更新 | S11 |
| S13 | `sell_hour` | side, current hour | ratio 更新 | S12 |
| S14 | `loss_boost` | side, now, loss boost state | ratio 更新 | S13 |
| S15 | offset amount calc | spread, ratio | `offset` | S14 |
| S16 | `ffd` | side, spread, ratio, offset | ratio/offset 更新 | S15 |
| S17 | final stage record | ratio | `_last_offset_stages` | S16 |
| S18 | spread guard finalization | side, best bid/ask, offset | `MakerPriceResult` | S16 |

## stage の独立性と依存性

### 比較的独立な stage

- `regime`
- `sell_hour`
- `loss_boost`

これらは主に
- side
- regime/hour
- local state

に依存し、orderbook microstructure の詳細依存が薄い。

### microstructure 依存が強い stage

- `as_shift`
- `spread_adapt`
- `kyle`
- `amihud`
- `vol_guard`
- `cross_venue`
- `imb_risk`
- `buy_as_guard`

これらは
- spread / mid / sigma
- velocity
- VPIN / imbalance
- orderbook flow

への依存が強い。

### orchestration 上、順序を保つべき箇所

1. `inventory skew` / `sell floor` は stage 群の前に置く
2. `cross_venue` は veto を投げ得るので `imb_risk` より前
3. `loss_boost` は ratio pipeline の末尾寄りに置く
4. `ffd` は offset amount 計算後に置く
5. `finalization` は最後に固定

## Mixin との整合

### `RiskGuardsMixin`

担当:
- `vol_guard`
- `cross_venue`
- `buy_as_guard`
- FFD 連携の一部

評価:
- risk policy を抱える mixin として整合的
- ただし veto state は `MakerPriceCalculator` core ownership に残すのが妥当

### `MicrostructureMixin`

担当:
- `as_shift`
- `kyle`
- `amihud`
- sigma 推定
- fill probability model 参照

評価:
- 今後もっとも `microstructure cache` と結び付きやすい
- ただし cache object 分離はまだ早い

### `RegimeBoostMixin`

担当:
- `regime`
- `sell_hour`

評価:
- 比較的独立性が高い
- 将来 split するなら最も先に単体 module 化しやすい

## 今後の実装優先度

### P0: 継続してやる

1. pure helper の追加抽出
2. stage schema/version の固定
3. source-contract test を direct call 前提から stage 契約前提へ更新

### P1: 次の split-first 候補

1. `stage store` seed 部分の local helper 化
2. `veto telemetry` の local helper 化
3. `finalize + serialize` の local helper 化

### P2: future 対応

1. `MicrostructureCache` object
2. `TelemetryRecorder` object
3. `PricingCoreState` object

ただし P2 は、Wave5 broad を安定通過させてからでよい。

## 結論

- `maker_price` はまだ stateful orchestrator として残す
- ただし state は
  - Pricing Core State
  - Microstructure Cache
  - Telemetry / Diagnostic
  に整理して見通しを固定できた
- `compute()` は 19 step の pipeline として読み替えると、分割可能な境界がかなり明確である
- 今後は state object 化を急がず、stage 契約と pure helper を増やす方針が妥当
