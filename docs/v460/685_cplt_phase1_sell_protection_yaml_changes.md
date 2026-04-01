# 685# Phase 1 Sell Protection — YAML パラメータ変更

| 項目 | 値 |
|------|-----|
| 作成日 | 2026-04-02 |
| 入力 | 684# 統合レビュー Phase 1 提案 (S1/S3/S4/S5) |
| 対象 | `configs/v460/fill_test.yaml`, `configs/v460/experiments/g2_sac_train.yaml` |
| 方式 | YAML hot-reload (fill_test 再起動不要) |

---

## 0. 背景

684# で 681#-683# の三者レビューをクロスバリデーションし、Phase 1（即時 YAML 変更）/ Phase 2（Codex タスク）/ Phase 3（中期）に層別した。本 685# は Phase 1 の実行記録。

**三者合意**: sell 側の品質が問題の核。buy は概ね健全。

---

## 1. 変更一覧

### 1.1 S1: JST 11h/13h Sell 防御強化

4/1 データで最も sell 損失が集中した JST 11h (UTC 2) と JST 13h (UTC 4) に対する防御。

| パラメータ | キー | 変更前 | 変更後 | 根拠 |
|-----------|------|:------:|:------:|------|
| sell_hour_offset_boost[2] | S1a | 2.0 | **2.5** | JST11h sell sum=-31.1bps(n=6)。sell損失88%を占める最悪帯 |
| sell_hour_offset_boost[4] | S1b | (なし) | **2.5** | JST13h sell PnL=-8.85(n=3) sum=-26.6bps。壊滅帯 |
| hour_ceiling_mult[2] | S1c | (なし) | **2.0** | JST11h ceiling 解放。offset boost と連動 |
| hour_ceiling_mult[4] | S1d | (なし) | **2.5** | JST13h ceiling 解放。最悪帯のため最大緩和 |

### 1.2 S3: Skip Gate Regime Thresholds 厳格化

trending regime での SG 排除強化。

| パラメータ | 変更前 | 変更後 | 根拠 |
|-----------|:------:|:------:|------|
| regime_thresholds.trending_up | 0.3 | **0.5** | 4/1 trending_up/sell PnL=-2.93(n=11) AS=45% |
| regime_thresholds.trending_down | 0.1 | **0.3** | 4/1 trending_down/sell PnL=-2.49(n=13) AS=31% |

### 1.3 S4/S5: Trending Offset Boost 調整

| パラメータ | 変更前 | 変更後 | 根拠 |
|-----------|:------:|:------:|------|
| trending_down_sell_offset_boost | 0.7 | **1.0** | S4: 176# の「aligned discount」理論が4/1データで否定（PnL=-2.49）|
| trending_up_sell_offset_boost | 1.8 | **2.2** | S5: 4/1 trending_up/sell PnL=-2.93, AS=45% → さらなる防御強化 |

---

## 2. S2 (Toxic Sell Veto) — SKIP

684# の S2 提案 `toxic_sell_veto_velocity_threshold: 0.0→1.0` はコード検証で前提誤りが判明：

- `toxic_sell_veto_velocity_threshold` は `price_velocity_bps`（60s ウィンドウ）を参照
- 684# は `mid_price_trend_5s`（5s）が使われると誤認していた
- 閾値引上げは veto をより選択的にする（発動減少 = 防御弱化）→ 意図と逆
- → **スキップ**。5s トレンドガードは Codex タスク（684# Phase 2）で別実装

---

## 3. 付随修正

### 3.1 g2_sac_train.yaml 特徴量修正

Codex が `features.selected` に追加した `mid_price_trend_5s` と `signed_obi` を**コメントアウト**。
理由: データパイプライン未整備で parquet に該当カラムが存在しない。

### 3.2 テスト修正（Codex trend_5s_guard 追加への対応）

Codex が `SkipFillRecordExtraFields` NamedTuple に `trend_5s_guard_triggered`, `trend_5s_guard_action`, `trend_5s_at_order` を追加（デフォルトなし）した影響で 4 テストが破損：

| テストファイル | 修正内容 |
|---------------|---------|
| test_176_trending_offset_asymmetry | assertion値を 2.2/1.0 に更新 |
| test_253_hot_reload_dead_config_getattr_bare_except | 行数上限 1560→1600 (PPO sidecar 追加) |
| test_516_skip_gate_result_fields_migration | SkipFillRecordExtraFields に 3 フィールド追加 (×2箇所) |
| test_642_observability_fields | _PreOrderPhaseResult に 3 フィールド追加 |

---

## 4. Expected Effects

| 指標 | 期待 | 検証方法 |
|------|------|---------|
| JST 11h/13h sell PnL | 改善（offset boost + ceiling 解放で逆選択 sell 抑制） | 翌日 fill_records 分析 |
| SG trending regime 排除率 | 上昇（threshold 厳格化） | skip_gate ログ集計 |
| trending_down sell 損失 | 縮小（discount 0.7→1.0 で中立化） | regime×side PnL |
| trending_up sell 損失 | 縮小（boost 1.8→2.2 で防御強化） | regime×side PnL |

---

## 5. 残課題（Phase 2 / Codex）

- [ ] SAC sell-aware training（684# Codex Task A）
- [ ] trend_5s sell guard（684# Codex Task B）
- [ ] sell offset ceiling 0.40→0.50（Phase 1 観察後に判断）
