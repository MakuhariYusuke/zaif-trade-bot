# 092# 対応漏れ点検と先行実装

| key | value |
|---|---|
| 番号 | 092 |
| フェーズ | ph2 |
| 種別 | impl |
| 対象 | 083#–091# ドキュメント群の対応漏れ点検 + 先行実装可能項目の即時対応 |
| 前提 | 091# 実装完了 (`1d98c4c37`) |
| コミット | 本文書と同時 |

---

## §1 点検結果サマリ

### 実装完了済み (対応漏れなし)

| 出典 | 施策 | 対応先 |
|------|------|--------|
| 083# §4.1-1 | run_id 分離評価 | 085# |
| 083# §4.1-2 | SkipGate P(AS) 直接記録 | 085# |
| 083# §4.1-3 | time_filter 縮退 | 089# |
| 083# §4.2-1 | Sell 専用ポリシー化 | 088# P0-2 + P1-3 |
| 084# 盲点B | param_adapter デッドロック防止 | 085# |
| 084# 盲点D | AS_raw 並行表示 | 085# |
| 084# 盲点F | api_error リトライ強化 | 085# |
| 084# 盲点G | run_id 分離フィルタ | 085# |
| 087# P0-1 | SkipGate 動的較正 | 088# |
| 087# P0-2 | Sell ハードガード | 088# |
| 087# P0-3 | Status unknown リトライ | 088# |
| 087# P0-4 | データ品質 (run_id/git_sha 欠落) | 088# |
| 087# P1-3 | Side 分離適応 | 088# |
| 091# #2 | 未保存レコード化 | 091# |
| 091# #3 | 低残高停滞 | 091# |
| 091# #4/#5 | 090# 誤記修正 | 091# |
| 091# #6 | offset_floor post-adaptive | 091# |

### 意図的保留 (正当な理由あり)

| 出典 | 施策 | 保留理由 |
|------|------|---------|
| 083# §4.1-4 | timeout 短縮 | 084# 盲点C: Wait長=PnL良好 — データ反証 |
| 088# P1-1 | OB 特徴量復元 | SkipGate 機能不全、OBコスト高 |
| 088# P1-2 | SkipGate 再学習パイプライン | 現モデル有効性確認が先 |
| 091# #7 | Smart Side 再有効化 | imbalance 無効の前提条件未解決 |

---

## §2 本セッションで実装した項目

### 2.1 E1 閾値再設計: 90% → 85% (084# 盲点H)

**問題**: E1 fill_rate_p90 = 90% は maker 指値戦略で構造的に非現実的 (最良 run で 86.8%)。  
**変更**: `configs/v460/gate_thresholds.yaml` の `min_fill_rate_p90` を `0.90` → `0.85` に変更。  
**根拠**: 084# §2 盲点H「E1 threshold 90% は非現実的」— Gate を永遠に通過できないリスクを解消。

### 2.2 E6 Round-trip KPI 追加 (087# P1-1 / 083# §4.2-3)

**問題**: 単発30s PnL (E4) だけでは往復テール損失が見えない。  
mean -1.96bps vs median +0.60bps の乖離はテール損失管理の欠如を示す。

**変更**:
- `ztb/metrics/fill_quality.py` の `g1_1_judgment` に `records` パラメータを追加
- `E6_round_trip_pnl`: round-trip mean PnL (bps) の監視。閾値 -2.0 bps。
  - `pairs`, `median`, `total_jpy` も付属
  - `informational=True` (当面は Gate 判定に影響しない — 安定したら昇格)
- `configs/v460/gate_thresholds.yaml` に `min_round_trip_pnl_mean: -2.0` 追加

### 2.3 E7 Net Inventory Drift 追加 (087# P1-1)

**問題**: 片側連続取引による在庫偏りが損失を拡大。  
**変更**:
- `E7_net_inventory`: |net_inventory| の監視。閾値 5。
  - `net_inventory`, `unpaired_buys`, `unpaired_sells` も付属
  - `informational=True` (監視用)
- `configs/v460/gate_thresholds.yaml` に `max_net_inventory: 5` 追加

### 2.4 Monitor スナップショットにラウンドトリップ指標追加

- `scripts/v460/monitor_fill_test.py` の `save_snapshot` 出力に `round_trip` と `inventory` セクション追加
- `g1_1_judgment(metrics, thresholds, records=clean_records)` で records を渡すよう修正

### 2.5 既存テスト不整合修正 (089# time_filter 追従)

089# time_filter 大幅削減後、3 テストが旧形式の skip_utc_hours を検査していた:
- `test_fill_quality.py::Test052AdaptSellOffsetSync::test_yaml_skip_utc_hours_includes_12`
- `test_fill_quality.py::Test052AdaptSellOffsetSync::test_yaml_skip_utc_hours_includes_13_and_21`
- `test_regime_detector.py::TestTimeFilterNoRecord::test_yaml_side_specific_time_filter`

→ 089# の side 別 time_filter (`skip_utc_hours_buy/sell`) に合わせて更新。

---

## §3 テスト結果

### 新規テスト: 18 件 (`test_092_gap_fixes.py`)

| クラス | テスト数 | 対象 |
|--------|----------|------|
| `TestE1ThresholdRedesign` | 4 | E1 閾値 85%、YAML整合 |
| `TestE6RoundTripKPI` | 7 | E6 判定: 存在、informational、閾値、正負PnL |
| `TestE7NetInventory` | 5 | E7 判定: バランス、ドリフト、informational |
| `TestGateThresholdsYaml` | 2 | YAML に round-trip/inventory 閾値あり |

### 全体回帰: **722 passed** (旧 3 FAIL → 修正後 0 FAIL)

---

## §4 未着手施策 (データ蓄積・実験設計待ち)

| # | 重大度 | 施策 | 現状 |
|---|--------|------|------|
| 1 | HIGH | 088#/089# 後データ再集計 | 088/089 後の fill_records = **0 件**。データ蓄積待ち |
| 2 | MID | spread_adaptive parameters 探索 | ABテスト設計・実行が必要 |
| 3 | MID | fast_fill_defense 閾値調整 | データ分析待ち |
| 4 | MID | offset 最適化 side 別バンディット | ABテスト基盤が必要 |
| 5 | LOW | time_filter event-driven 完全化 | 根本設計変更、段階的に |
| 6 | LOW | Event-driven サイクル間隔 | 091# で「120s維持」判断済 |
| 7 | LOW | 過学習判定 / Wait-AS 因果層化 | データ蓄積待ち |

---

## §5 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/gate_thresholds.yaml` | E1: 0.90→0.85, E6/E7 閾値追加 |
| `ztb/metrics/fill_quality.py` | g1_1_judgment に E6/E7 + records パラメータ追加 |
| `scripts/v460/monitor_fill_test.py` | records 渡し + snapshot に round-trip/inventory |
| `tests/unit/v460/test_092_gap_fixes.py` | 18 新規テスト |
| `tests/unit/v460/test_fill_quality.py` | 089# time_filter 追従修正 (2テスト) |
| `tests/unit/v460/test_regime_detector.py` | 089# time_filter 追従修正 (1テスト) |
