# 107# Time Filter 分析 — 動的ゲーティングへの移行提案

| 区分 | 値 |
|------|------|
| 番号 | 107# |
| 日付 | 2026-02-18 |
| 前提 | 106# git=8ba101953, fill_test run_id=1771339803_3afba87b |
| 種別 | 分析・提案 |

---

## 1. 現状分析: 静的 time_filter の問題点

### 1.1 現在のフィルター設定

| Side | Blocked UTC hours | JST | 遮断率 |
|------|-------------------|-----|--------|
| BUY  | 1,2,8,12,16,18,21 | 10,11,17,21,01,03,06 | **7/24 = 29.2%** |
| SELL | 4,8,13,14,16,17 | 13,17,22,23,01,02 | **6/24 = 25.0%** |
| 共通 | 16 (JST01) | — | 全 side 一律遮断 |

### 1.2 根本的な欠陥

1. **反応性ゼロ**: UTC03 が歴史的に良好 (+1.48 bps BUY) でも、その時間帯内で突然の暴落が起きれば AS を食らう。フィルターは介入しない。
2. **過学習**: 過去の統計的パターン (n=7~15 程度) に基づく。市場構造変化で容易に無効化される。
3. **機会損失**: BUY 約86件、SELL 約70件の fill を推定逸失（active hour あたりの fill rate × blocked hours）。
4. **粒度の粗さ**: 1時間単位のオン/オフ。実際の AS リスクはミクロ秒単位のオーダーフロー変化に起因。

### 1.3 UTC 時間別 PnL 全データ (全期間)

**BUY**:
| UTC | n | avg_pnl | AS | AS% | 判定 |
|-----|---|---------|----|----- |------|
| 00 | 17 | +0.90 | 3 | 17.6% | ✅ |
| 01 | 14 | **-2.43** | 7 | 50.0% | ❌ blocked |
| 02 | 8 | **-2.78** | 5 | 62.5% | ❌ blocked |
| 03 | 8 | +1.48 | 1 | 12.5% | ✅ |
| 04 | 7 | +3.99 | 1 | 14.3% | ✅ |
| 05 | 14 | -0.35 | 4 | 28.6% | ✅ |
| 06 | 13 | -0.14 | 4 | 30.8% | ✅ |
| 07 | 14 | -0.06 | 2 | 14.3% | ✅ |
| 08 | 7 | **-0.47** | 3 | 42.9% | ❌ blocked |
| 09 | 8 | -0.67 | 4 | 50.0% | ✅ |
| 10 | 5 | -0.40 | 1 | 20.0% | ✅ |
| 11 | 6 | +0.04 | 1 | 16.7% | ✅ |
| 12 | 6 | **-2.19** | 4 | 66.7% | ❌ blocked |
| 13 | 16 | -0.14 | 4 | 25.0% | ✅ |
| 14 | 8 | +1.63 | 5 | 62.5% | ✅ |
| 15 | 15 | +1.22 | 7 | 46.7% | ✅ |
| 16 | 7 | **-3.60** | 5 | 71.4% | ❌ blocked |
| 17 | 11 | +0.47 | 7 | 63.6% | ✅ |
| 18 | 8 | **-3.24** | 5 | 62.5% | ❌ blocked |
| 19 | 11 | +0.40 | 6 | 54.5% | ✅ |
| 20 | 25 | +1.76 | 4 | 16.0% | ✅ best |
| 21 | 13 | **-2.19** | 6 | 46.2% | ❌ blocked |
| 22 | 16 | +1.11 | 3 | 18.8% | ✅ |
| 23 | 15 | +1.07 | 3 | 20.0% | ✅ |

**SELL**:
| UTC | n | avg_pnl | AS | AS% | 判定 |
|-----|---|---------|----|----- |------|
| 00 | 19 | +0.09 | 4 | 21.1% | ✅ |
| 01 | 13 | +0.93 | 4 | 30.8% | ✅ |
| 02 | 8 | +0.24 | 2 | 25.0% | ✅ |
| 03 | 9 | +0.42 | 3 | 33.3% | ✅ |
| 04 | 7 | **-5.56** | 4 | 57.1% | ❌ blocked |
| 05 | 15 | +0.57 | 5 | 33.3% | ✅ |
| 06 | 13 | +0.02 | 3 | 23.1% | ✅ |
| 07 | 13 | -1.85 | 7 | 53.8% | ⚠️ marginal |
| 08 | 8 | **-6.72** | 4 | 50.0% | ❌ blocked |
| 09 | 6 | -1.42 | 2 | 33.3% | ✅ |
| 10 | 7 | -0.50 | 2 | 28.6% | ✅ |
| 11 | 5 | -0.97 | 1 | 20.0% | ✅ |
| 12 | 6 | -0.03 | 1 | 16.7% | ✅ |
| 13 | 14 | **-1.91** | 4 | 28.6% | ❌ blocked |
| 14 | 8 | **-4.62** | 4 | 50.0% | ❌ blocked |
| 15 | 14 | +0.81 | 6 | 42.9% | ✅ |
| 16 | 8 | **-2.13** | 5 | 62.5% | ❌ blocked |
| 17 | 8 | +0.65 | 5 | 62.5% | ❌ blocked† |
| 18 | 5 | +1.37 | 2 | 40.0% | ✅ |
| 19 | 13 | -1.20 | 7 | 53.8% | ⚠️ marginal |
| 20 | 23 | +0.95 | 7 | 30.4% | ✅ best |
| 21 | 14 | -1.27 | 5 | 35.7% | ⚠️ marginal |
| 22 | 13 | -0.20 | 4 | 30.8% | ✅ |
| 23 | 14 | -0.32 | 4 | 28.6% | ✅ |

† UTC17 SELL は実際 **+0.65 bps** で被ブロック → 誤遮断 (旧データ -2.11 bps 時代の設定が残存)

---

## 2. 既存動的メカニズムの棚卸し

### 2.1 SkipGate (062#)
- **特徴量**: `hour_sin`, `hour_cos`, `spread_jpy`, `offset_ratio`, `regime_*`, `trade_count_60s`, `buy_ratio`, `trade_flow_imbalance_60s`, `avg_trade_size`, `price_velocity_60s`, `vpin_60s`, `side_aligned_tfi`, `side_aligned_velocity` (16特徴量)
- **時刻情報**: あり (`hour_sin/cos` — 24h周期の連続エンコーディング)
- **AS確率予測**: あり (`as_probability`, 閾値 `0.52`)
- **適応閾値**: あり (`skip_gate_adaptive_threshold` — 直近50サイクルの skip_rate に基づいて閾値を上下調整)
- **現状の問題**: SkipGate は 26/734 = 3.5% しかブロックしておらず、time_filter に比べて圧倒的に控えめ。時間帯情報より microstructure 信号を重視している。

### 2.2 Spread Adaptive (054#)
- **narrow spread boost**: スプレッドが `narrow_spread_bps` 未満 → offset を boost (side 別倍率)
- **wide spread reduction**: スプレッドが `wide_spread_bps` 超 → offset を縮小
- **限界**: 二値判定 (narrow/wide) のみ。連続的なリスク調整ではない。

### 2.3 Regime Detector (035#)
- **trending**: offset × 1.5 boost
- **high_vol**: `high_vol_multiplier` × base volatility
- **限界**: レジーム遷移にヒステリシス（3サイクル）あり、急変には遅延。

### 2.4 Sell Guard (088#)
- `sell_offset_floor: 0.10` — sell 側の最低 offset 保証
- `sell_max_spread_jpy` — 超過でスキップ
- **限界**: sell 専用。buy 側は保護なし。

---

## 3. 提案: 3段階の動的ゲーティング移行

### Phase 1: SkipGate 強化 (Low risk, 即時実行可能)

**目的**: SkipGate の既存 `hour_sin/cos` 特徴量が time_filter の役割を暗黙的に学習するようにモデルを再学習。

**具体策**:
1. **再学習データの蓄積**: 現在の 535 fills は最低限だが使用可能。time_filter OFF で走らせた全データが理想
2. **閾値の side 別最適化**: 現在 `as_threshold: 0.52` 一律 → BUY/SELL 別閾値を `skip_gate_as_threshold_buy/sell` で設定（既にコード実装済み）
3. **adaptive threshold の積極化**: `target_skip_rate_buy: 0.10 → 0.15`、`target_skip_rate_sell: 0.20 → 0.25` で skip 力を強化。time_filter の遮断率 (25-29%) に近づける

**検証方法**: time_filter を `enabled: false` にして SkipGate のみの 48h テスト実行。PnL と AS rate を比較。

### Phase 2: Volatility Guard 新設 (Medium risk)

**目的**: 急な値動きをリアルタイム検知し、time_filter では不可能だった「時間帯内の急変」に対応。

```yaml
# 107# volatility guard config (案)
volatility_guard:
  enabled: true
  # 短期ボラティリティ (直近N秒のmid_price変動率)
  short_vol_window_sec: 60      # 60秒ウィンドウ
  short_vol_threshold_bps: 15   # 15bps超で guard 発動
  # VPIN (Volume-synchronized Probability of Informed Trading)
  vpin_threshold: 0.7           # 0.7超で skip
  # 行動: skip | offset_boost | both
  action: "offset_boost"        # offsetを2倍に (skipは攻撃的すぎる)
  offset_boost_factor: 2.0
```

**実装概要**:
- `_calculate_short_vol()` メソッド: 直近60秒の mid_price 変動率 (bps) を計算。既に SkipGate の `build_features_from_market_state()` で `price_velocity_60s` を計算しているため、ロジック流用可能。
- 判定位置: `_compute_effective_offset()` 内、`spread_adaptive` の直後に挿入。
- VPIN: SkipGate features の `vpin_60s` をそのまま再利用。

**利点**: time_filter が「1時間全体」を遮断するのに対し、volatility_guard は「その瞬間」の市場状態に反応。静的→動的への本質的転換。

### Phase 3: Time Filter 漸進的廃止 (Low risk if Phase 1-2 成功後)

**段階的削減**:

| ステップ | BUY blocked | SELL blocked | 条件 |
|---------|-------------|--------------|------|
| 現状 | 7h | 6h | — |
| Step 1 | 3h (2,16,18) | 3h (4,8,14) | PnL ≤ -2.5 bps のみ残す |
| Step 2 | 1h (16) | 1h (8) | PnL ≤ -3.5 bps のみ |
| Step 3 | 0h | 0h | Phase 1+2 が十分機能した場合 |

各ステップで 48h 運用→PnL/AS 確認→次ステップへ。

---

## 4. リスクと留保

### 4.1 SkipGate の限界
- **学習データ偏り**: time_filter で遮断された時間帯のデータが存在しない → 遮断解除直後は SkipGate が「未知の時間帯」として不安定になる可能性。
- **対策**: Step 1 (3h→3h残し) での遮断解除で新データ収集 → SkipGate 再学習のフィードバックループ。

### 4.2 Volatility Guard の閾値設定
- 閾値が低すぎると機会損失、高すぎると AS 防止効果なし。
- **対策**: 初期は `action: "offset_boost"` (skip より穏健) で導入し、データ蓄積後に閾値を最適化。

### 4.3 SkipGate + Volatility Guard + time_filter の多重防御
- 移行期は3つが共存する。相互作用で過剰遮断のリスクあり。
- **対策**: 各ガードの発動を fill_record に記録し、事後分析で遮断理由の重複を検出。

---

## 5. 推奨実行順序

1. **即時**: SELL UTC17 を blocked から除外 (+0.65 bps を誤遮断中)
2. **107#**: Phase 1 — SkipGate adaptive 閾値調整 + side 別閾値有効化
3. **108#**: Phase 2 — Volatility Guard の実装 + テスト
4. **109#**: Phase 3 Step 1 — time_filter を最悪時間帯のみに縮小
5. **110#+**: Phase 3 Step 2-3 — データに基づく段階的廃止

---

## 6. 結論

time_filter は「平均的に悪い時間帯を遮断する」静的な手段であり、本質的に以下を解決できない:
- **良い時間帯内での突然の悪化** (false negative)
- **悪い時間帯内での良い機会の逸失** (false positive)

これに対し、SkipGate (ML ベースの AS 確率予測) + Volatility Guard (リアルタイム変動検知) は:
- 時間帯に依存せず、**その瞬間の市場状態**でリスクを判断
- SkipGate は既に `hour_sin/cos` で時刻情報を持つため、time_filter の知識を暗黙的に吸収可能
- Volatility Guard は time_filter では対応不可能だった「急な値動き」に直接対応

**最終目標**: time_filter 全廃、SkipGate + Volatility Guard による完全動的ゲーティング。

---

## 7. 実装記録 (107# commit)

### 7.1 実施内容

| 項目 | 変更 | ファイル |
|------|------|----------|
| SELL UTC17 解除 | `skip_utc_hours_sell` から 17 を除去 (6h→5h) | `configs/v460/fill_test.yaml` |
| SkipGate 強化 | `target_skip_rate_buy` 0.10→0.15, `sell` 0.20→0.25 | 同上 |
| Volatility Guard | 新セクション追加 (velocity 15bps / VPIN 0.70 / boost ×2) | 同上 |
| VG Config | `volatility_guard_*` 5フィールド追加 | `run_fill_test.py` FillTestConfig |
| VG YAML parsing | `from_yaml()` に `volatility_guard` セクション処理追加 | 同上 |
| VG 実装 | `_compute_maker_price()` に velocity + VPIN → offset boost ロジック | 同上 |
| VPIN キャッシュ | `_last_vpin` を SkipGate features から取得・保存 | 同上 |
| R1 batch flush | `_maybe_flush_batch()` ヘルパー + 3箇所の重複 flush 統合 | 同上 |
| R3 テスト | `_calibrate_threshold` 5件 + `warm_start` 3件 | `test_enricher_skip_gate.py` |
| 107# テスト | YAML/Config/コード検証 8件 | `test_fill_quality.py` |
| 既存テスト修正 | 091# / 089# テスト 4件を新構造に適合 | `test_091_fixes.py`, `test_regime_detector.py` |

### 7.2 テスト結果

- **827 passed** (811 baseline + 16 new)
- 既存テスト 4件を 107# 変更に追従修正

### 7.3 Phase 進捗

- [x] Phase 1: SkipGate 強化 (target_skip_rate 引上げ)
- [x] Phase 2: Volatility Guard 実装 (velocity + VPIN → offset boost)
- [x] SELL UTC17 即時解除
- [x] fill_test 再起動 — run_id=1771380856_b7d09bbf, PID=24884, git_sha=361c67f4e
- [ ] 48h 観察 → Phase 3 Step 1 実施判断
- [ ] Phase 3 Step 1: time_filter を最悪時間帯のみに縮小 (BUY 7h→3h, SELL 5h→3h)
