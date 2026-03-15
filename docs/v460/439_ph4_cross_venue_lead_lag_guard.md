# 439# 433# §3 実装メモ: Cross-Venue Lead-Lag Guard

| 項目 | 内容 |
|---|---|
| 番号 | 439# |
| 分類 | ph4_impl |
| 対象 | 433# §3, 434# §4.2 |
| 前提 | 433# BitFlyer lead-lag 提案, 434# arb→lead-lag 補正 |
| 目的 | Cross-venue lead-lag guard の最小安全実装 (disabled-default) |

---

## 1. 背景

[433_ph4_advanced_microstructure_edge_ideas.md](docs/v460/433_ph4_advanced_microstructure_edge_ideas.md) §3 は、BitFlyer の価格変動を Coincheck の先行指標として使う案を提案している。

一方で [434_ph2_ph4_rev_426_432_433_multifaceted_validation.md](docs/v460/434_ph2_ph4_rev_426_432_433_multifaceted_validation.md) §4.2 は、これを厳密な arbitrage ではなく `cross-venue lead-lag / stale quote exploitation` のヒントとして扱うべきだと補正している。

この補正は妥当である。public API ベースでは以下の不確実性が残る。

- REST / public feed 遅延
- venue 間 clock skew
- Coincheck / BitFlyer の出来高構成差
- 非同期取得による stale fusion

したがって初手は:

- hard directional flip
- aggressive override

ではなく:

- adverse-side veto
- adverse-side offset boost / retreat
- fail-open

で入れるのが安全。

## 2. 現行コードの利用点

今回の実装では、既存の責務境界を崩さず、次の既存回路を再利用した。

### 2.1 参照市場データ

- `ztb.trading.live.registry.broker_registry.BrokerRegistry.create_adapter(...)`
- `ztb.trading.live.exchanges.bitflyer.adapter.BitFlyerAdapter.get_orderbook(...)`
- `scripts/v460/lib/ob_utils.py::best_bid_ask(...)`

BitFlyer adapter はすでに registry に登録済みで、追加の adapter レイヤーは不要だった。

### 2.2 maker 価格パイプライン

- `scripts/v460/lib/maker_price.py::MakerPriceCalculator.compute(...)`
- `scripts/v460/lib/maker_risk_guards.py::RiskGuardsMixin`

既存の `volatility_guard` / `imbalance_risk` と同じ「offset 調整ステージ」として差し込める構造になっていたため、新規ロジックは `RiskGuardsMixin` 側へ追加した。

### 2.3 実行時注入点

- `scripts/v460/lib/fill_cycle_executor.py::run_single_cycle(...)`

`run_single_cycle()` はすでに local orderbook を prefetch して `self._maker_price._last_ob_snapshot` を更新している。ここに参照市場の `depth=1` orderbook を追加取得し、hint を `MakerPriceCalculator` へ注入するのが最小差分だった。

### 2.4 設定 SSOT

- `scripts/v460/lib/fill_config.py`
- `scripts/v460/lib/fill_config_parser.py`
- `configs/v460/fill_test.yaml`

コード内 hidden flag にはせず、disabled default の YAML section を追加して SSOT を維持した。

## 3. 実装方針

### 3.1 safe-first policy

今回の live path は次だけを行う。

1. Coincheck local mid と reference venue mid の乖離を `spread_bps` で計測
2. reference venue の直近 mid 変化を `reference_velocity_bps` で計測
3. 乖離と速度の符号が一致し、かつ閾値超過なら adverse side を特定
4. adverse side のみ offset を拡大
5. `veto_enabled=true` かつ乖離が大きい場合のみ `InfeasibleQuoteError` で skip

以下は意図的に未実装とした。

- buy/sell の hard flip
- Sidecar bias への直接加算
- reference venue による順張り aggressive placement

### 3.2 fail-open

BitFlyer side の取得失敗、stale、欠損、時刻逆転では hint を捨てる。primary path は継続する。

## 4. 追加した構成要素

### 4.1 新規 helper

`scripts/v460/lib/cross_venue_lead_lag.py`

- `VenueMidSnapshot`
- `CrossVenueLeadLagHint`
- `compute_cross_venue_lead_lag_hint(...)`
- `build_reference_adapter(...)`

pure 判定ロジックと adapter 生成を分けたことで、runner 本体に新しい判定式や registry ロジックを埋め込まずに済んでいる。

### 4.2 設定

`FillTestConfig` と `fill_test.yaml` に追加:

- `cross_venue_lead_lag_enabled`
- `cross_venue_reference_exchange`
- `cross_venue_lead_lag_max_age_sec`
- `cross_venue_lead_lag_spread_bps_threshold`
- `cross_venue_lead_lag_velocity_bps_threshold`
- `cross_venue_lead_lag_offset_boost`
- `cross_venue_lead_lag_veto_enabled`
- `cross_venue_lead_lag_veto_threshold_bps`

### 4.3 価格決定パイプライン

`MakerPriceCalculator.compute(...)` に `cross_venue` ステージを追加した。

順序は:

1. `volatility_guard`
2. `cross_venue_lead_lag_guard`
3. `imbalance_risk`

とした。理由は:

- volatility/microstructure で local の危険信号を先に反映する
- cross-venue は「追加の retreat 票」として扱う
- imbalance は local board risk として最後にかける

## 5. 実装上の注意

### 5.1 cancel_reason

veto は新しい `cancel_reason`:

- `cross_venue_lead_lag_veto`

で既存の `InfeasibleQuoteError` 経路へ流している。これにより skip record, logging, fill-quality 集計の流れを壊していない。

### 5.2 cleanup

参照 adapter は optional で生成し、`_cleanup_sync()` で `close()` があれば呼ぶ。

### 5.3 dry-run

dry-run でも参照 adapter は primary adapter の `dry_run` flag を引き継いで生成する。ただし実際の BitFlyer public board は network 依存なので、取得失敗時は fail-open で無効化される。

## 6. 検証

追加した focused coverage:

- pure helper
  - 上昇 lead で sell adverse
  - stale / 符号不一致で `None`
- registry reuse
  - primary adapter の `dry_run` を継承
- maker guard
  - adverse side だけ retreat
  - safe side は不変
  - veto path は `InfeasibleQuoteError`
- executor wiring
  - local/ref orderbook から hint を inject
  - reference 失敗時に fail-open
- parser / YAML
  - new section の parse
  - production YAML round-trip

## 7. 今後の拡張余地

安全に広げるなら次の順が良い。

1. `offset_boost` の side/regime 別倍率化
2. BitFlyer WebSocket 化で stale を縮小
3. sidecar 特徴量化 (`bf_cc_spread_bps`, `bf_price_velocity_1s`)
4. multi-venue aggregation

### 7.1 追加実装: FillRecord / event log observability

今回、最初の拡張余地として `FillRecord` に以下を追加した。

- `cross_venue_reference_exchange`
- `cross_venue_lead_lag_direction`
- `cross_venue_lead_lag_adverse_side`
- `cross_venue_lead_lag_spread_bps`
- `cross_venue_lead_lag_velocity_bps`
- `cross_venue_lead_lag_age_sec`
- `cross_venue_lead_lag_applied`
- `cross_venue_lead_lag_vetoed`

実装位置は `FillRecordBuilderMixin._build_fill_cross_venue_fields(...)` とし、
sidecar / executor stage と同様に builder 1 箇所で組み立てる形に寄せた。

加えて、`FillCycleExecutorMixin._update_cross_venue_lead_lag_hint()` で
`cross_venue_hint` event を `fill_test_events.jsonl` に出すようにした。
payload は:

- `reference_exchange`
- `direction`
- `adverse_side`
- `spread_bps`
- `velocity_bps`
- `age_sec`

で、`run_id` / `git_sha` も既存 event logger 契約に合わせて付与する。

これにより、後続の分析では

- hint 自体が出ていたか
- 現在の side に対して guard が実際に効いたか
- veto まで発火したか
- hint が cycle 実行時に event log へ流れていたか

を fill record 単位で追える。

逆に、次はまだやらない方がよい。

- hard directional override
- reference venue だけでの aggressive front-run
- local queue / toxicity と独立した単独採用

## 8. 結論

433# §3 の案は、そのままでは強すぎる。だが 434# の補正に従えば、

- pure helper
- disabled-default config
- maker guard stage
- executor injection
- fail-open cleanup

までで、保守性を落とさずに小さく live path へ入れられる。

今回の実装はその最小版であり、将来の sidecar 化や multi-venue 化へそのまま伸ばせる形になっている。
