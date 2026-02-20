# 121# モデル差し替え・次期改善計画

**Date**: 2026-02-20
**Session**: 124.1#
**Phase**: ph2 (G1.1-exec) → ph3 (G2-train) 移行準備
**前提**: 168h Gate 判定完了 — G1.2-full **WATCH** (全 12 チェック PASS)

---

## 目次

- [§1 エグゼクティブサマリ](#1-エグゼクティブサマリ)
- [§2 現行モデルの評価](#2-現行モデルの評価)
  - [§2.1 168h Gate 判定結果の要約](#21-168h-gate-判定結果の要約)
  - [§2.2 ボトルネック分析](#22-ボトルネック分析)
  - [§2.3 現行パラメータ構成](#23-現行パラメータ構成)
- [§3 改善アプローチ一覧](#3-改善アプローチ一覧)
- [§4 Track A: パラメータチューニング (低コスト・即時可能)](#4-track-a-パラメータチューニング)
  - [§4.1 time_filter 段階的緩和 (Phase 3 Step 1)](#41-time_filter-段階的緩和)
  - [§4.2 sell offset 引き上げ](#42-sell-offset-引き上げ)
  - [§4.3 spread_adaptive 再調整](#43-spread_adaptive-再調整)
  - [§4.4 regime warm-up 問題の解消](#44-regime-warm-up-問題の解消)
- [§5 Track B: SkipGate 再訓練 (中コスト・高インパクト)](#5-track-b-skipgate-再訓練)
  - [§5.1 現行 SkipGate の課題](#51-現行-skipgate-の課題)
  - [§5.2 759 サンプル再訓練計画](#52-759-サンプル再訓練計画)
  - [§5.3 特徴量変更方針](#53-特徴量変更方針)
  - [§5.4 buy/sell 分割モデル](#54-buysell-分割モデル)
  - [§5.5 訓練パイプライン実行手順](#55-訓練パイプライン実行手順)
  - [§5.6 デプロイ判定基準](#56-デプロイ判定基準)
- [§6 Track C: 新規 ML 導入 (高コスト・ph3 以降)](#6-track-c-新規-ml-導入)
  - [§6.1 SAC 4-seed 訓練 (G2-train)](#61-sac-4-seed-訓練)
  - [§6.2 Walk-Forward バグ修正 (前提)](#62-walk-forward-バグ修正)
  - [§6.3 Oracle テスト (理論上限の確認)](#63-oracle-テスト)
- [§7 Track D: 板情報 (OB) 特徴量の再活性化](#7-track-d-板情報-ob-特徴量の再活性化)
  - [§7.1 OB 無効化の経緯と再評価](#71-ob-無効化の経緯と再評価)
  - [§7.2 現行インフラの OB データ経路](#72-現行インフラの-ob-データ経路)
  - [§7.3 OB データ収集の即時有効化](#73-ob-データ収集の即時有効化)
  - [§7.4 OB 特徴量の段階的復活](#74-ob-特徴量の段階的復活)
  - [§7.5 マイクロストラクチャ特徴量の活用展望](#75-マイクロストラクチャ特徴量の活用展望)
- [§8 先送り事項の実装計画](#8-先送り事項の実装計画)
  - [§8.1 ph3 ブロッカー (即時着手必須)](#81-ph3-ブロッカー)
  - [§8.2 即時実行可能な未着手施策](#82-即時実行可能な未着手施策)
  - [§8.3 棚上げ妥当だが再訪タイミング注意](#83-棚上げ妥当だが再訪タイミング注意)
  - [§8.4 v461+/将来への先送り一覧](#84-v461将来への先送り一覧)
- [§9 見落とし・盲点分析](#9-見落とし盲点分析)
- [§10 フェーズ全体のロードマップ](#10-フェーズ全体のロードマップ)
- [§11 リスクと制約](#11-リスクと制約)
- [§12 意思決定フロー](#12-意思決定フロー)
- [Appendix A: 168h 判定 JSON 主要値](#appendix-a-168h-判定-json-主要値)
- [Appendix B: YAML パラメータ変更チェックリスト](#appendix-b-yaml-パラメータ変更チェックリスト)
- [Appendix C: 全先送り項目一覧](#appendix-c-全先送り項目一覧)

---

## §1 エグゼクティブサマリ

168h Gate 判定で G1.2-full **WATCH** (全チェック PASS、F4d_pnl_mean_floor が watch) を達成した。
これは ph2 の運用 Gate を通過したことを意味するが、PnL30s mean = **-0.161bps** が
依然として負であり、**大義（短期間での高収益性システム）に到達するには改善が必須**。

改善は **3 つの Track** で並行推進する:

| Track | 内容 | コスト | 期待効果 | 所要期間 |
|-------|------|--------|---------|---------|
| **A** | パラメータチューニング | 低 | +0.2〜0.4bps | 1-2日 + 168h 再測定 |
| **B** | SkipGate 再訓練 | 中 | +0.3〜0.5bps | 3-5日 + 168h 再測定 |
| **C** | SAC 新規 ML (ph3) | 高 | 戦略根本改善 | 2-4週間 |
| **D** | 板情報 (OB) 特徴量の再活性化 | 低-中 | データ基盤 + AS予測精度 | Track B に統合 |

**推奨**: Track A + B + D (OBデータ収集) を同時実施した上で fill_test を再起動し、168h 再測定。
結果に基づき Track C への移行を判断。

---

## §2 現行モデルの評価

### §2.1 168h Gate 判定結果の要約

**G1.2-full: WATCH** (全 12 チェック PASS)

| 指標 | 値 | 閾値 | 判定 | 備考 |
|------|-----|------|------|------|
| F1 attempted_fill_rate | 78.3% | ≥70% | ✅ PASS | |
| F1b overall_fill_rate | 70.7% | ≥62% | ✅ PASS | |
| F2 cancel_ratio | 21.7% | ≤30% | ✅ PASS | |
| F3 queue_wait_median | 12.8s | ≤60s | ✅ PASS | |
| F4 PnL30s | -0.161bps | p≥0.05 | ✅ PASS | p=0.163 (Holm: 0.490) |
| F4b PnL60s | -0.304bps | p≥0.05 | ✅ PASS | p=0.186 (Holm: 0.372) |
| F4c PnL120s | +0.186bps | p≥0.05 | ✅ PASS | p=0.656 → 正値 |
| F4d pnl_mean_floor | -0.161bps | ≥-0.50 | ✅ PASS | ⚠️ watch (-0.10 超過) |
| F5 AS_ratio | 28.1% | ≤30% | ✅ PASS | 1.9pt マージン |
| F6 skip_gate_ratio | 9.7% | ≤20% | ✅ PASS | |
| F7 calendar_days | 8 | ≥7 | ✅ PASS | |
| F8 n_attempted | 1,060 | ≥500 | ✅ PASS | |

**重要**: PnL120s = +0.186bps が正値 → **30s で逆行しても 120s で回復する特性は健在**。

### §2.2 ボトルネック分析

#### 2.2.1 PnL のドライバー分解

| 区分 | PnL30s | AS% | n | 改善余地 |
|------|--------|-----|---|---------|
| **全体** | **-0.161** | **28.1%** | **830** | — |
| ranging | -0.080 | 22.7% | 388 | 低 (最良レジーム) |
| trending | +0.016 | 29.2% | 171 | ほぼ均衡 |
| **unknown** | **-0.390** | **35.1%** | **271** | **最大** |

**発見**: `unknown` regime が全体の 32.6% を占め、PnL=-0.390bps / AS=35.1% と最悪。
これは再起動時の warm-up 20 サイクル + regime 未確定サイクルが主因。

#### 2.2.2 SkipGate の貢献度

| 区分 | PnL30s | n |
|------|--------|---|
| high_prob (skip 候補) | -0.093 | 226 |
| low_prob (keep) | +0.179 | 225 |
| **Δ (差分)** | **-0.271** | — |

SG は正しい方向に機能 (Δ < 0 = high_prob 群の方が悪い)。
ただし **AUC ≈ 0.5** の限界があり、改善余地は大きい。

#### 2.2.3 キャンセル理由内訳

| 理由 | 件数 | 全cancel内% |
|------|-----:|----------:|
| timeout | 135 | 39.2% |
| skip_gate | 114 | 33.1% |
| postonly_reject | 65 | 18.9% |
| status_unknown | 22 | 6.4% |
| stale_reprice_failed | 5 | 1.5% |
| その他 | 3 | 0.9% |

**注目**: timeout (39.2%) が最多。SG による skip (33.1%) は 9.7% の skip_gate_ratio に対応。
postonly_reject (18.9%) は narrow spread 時の構造的問題 (118# §8.2)。

#### 2.2.4 run 別トレンド (multi_track)

| 期間 | PnL30s | AS% | fill_rate | 備考 |
|------|--------|-----|-----------|------|
| all_run | -0.161 | 28.1% | 70.7% | 全体 |
| current_run | -0.600 | 30.2% | 74.9% | 直近 239 件 |
| trailing_200 | -0.316 | 26.2% | 74.5% | 直近 200 件 |

直近 run/trailing_200 の PnL が全体より悪化している点は留意。
ただし n が小さく分散が大きいため、統計的有意ではない可能性が高い。

### §2.3 現行パラメータ構成 (fill_test.yaml)

```yaml
# 主要パラメータ抜粋
cycle_interval_sec: 120.0
order_timeout_sec: 90.0
spread_offset_ratio: 0.05
side_offset.sell: 0.14
skip_gate.buy_enabled: true
skip_gate.sell_enabled: false      # 118# A3: sell 逆選別対策で無効化
skip_gate.as_threshold_buy: 0.50
skip_gate.target_skip_rate_buy: 0.15
skip_gate.target_skip_rate_sell: 0.25
skip_gate.model_path: models/v460/skip_gate_as.pkl
spread_adaptive.narrow_spread_bps: 2.0
volatility_guard.vpin_threshold: 0.70
time_filter.skip_utc_hours_buy: [1, 2, 8, 12, 16, 18, 21]   # 7時間
time_filter.skip_utc_hours_sell: [4, 8, 13, 14, 16]          # 5時間
regime.window: 20
```

---

## §3 改善アプローチ一覧

| # | アプローチ | 分類 | コスト | インパクト | 根拠 |
|---|-----------|------|--------|----------|------|
| A1 | time_filter 段階的緩和 | パラメータ | 低 | +0.1〜0.2bps (機会増) | 107# Ph3, 118# §5.6 |
| A2 | sell offset 引き上げ | パラメータ | 低 | sell AS 削減 | 118# §3.5, 105# |
| A3 | spread_adaptive 再調整 | パラメータ | 低 | postonly_reject 削減 | 118# §8.2 |
| A4 | regime warm-up state persistence | パラメータ/コード | 低 | unknown -0.390→改善 | 118# §3.3 |
| B1 | SkipGate 759 サンプル再訓練 | ML再訓練 | 中 | AUC 向上 → AS 削減 | 118# App F |
| B2 | regime 特徴量の復活 | ML再訓練 | 中 | regime 情報活用 | 118# F2 |
| B3 | buy/sell 分割モデル | ML再訓練 | 中-高 | sell 逆選別解消 | 118# §4.3 |
| C1 | SAC 4-seed 訓練 | 新規 ML | 高 | 戦略根本改善 | 000# §2 ph3 |
| C2 | Oracle テスト | 分析 | 中 | 理論上限確認 | 118# §8.5 |
| C3 | WF バグ修正 (6 件) | 前提作業 | 高 | ph3 の信頼性 | 111# §5 |
| **D1** | **OB 記録常時有効化** | **データ収集** | **低** | **"Microstructure Edge" の基盤** | **§7.3** |
| D2 | OB 特徴量で SG 再訓練 | ML再訓練 | 中 | 板情報による AS 予測改善 | 070#-072#, §7.4 |
| D3 | マイクロストラクチャ特徴量 (10種) | 将来 | 高 | SAC 観測空間拡張 | ztb/features/microstructure.py |

---

## §4 Track A: パラメータチューニング

### §4.1 time_filter 段階的緩和 (Phase 3 Step 1)

**根拠**: 118# §5.6, §8.8 で VG 有効性が確認済 (AS -7.7pt 改善)。
time_filter は BUY 7h / SELL 5h をブロックしており、**大量の機会損失**を生んでいる。

**変更案**:

```yaml
# Before (現行)
time_filter:
  skip_utc_hours_buy: [1, 2, 8, 12, 16, 18, 21]   # 7h block
  skip_utc_hours_sell: [4, 8, 13, 14, 16]           # 5h block

# After (Step 1)
time_filter:
  skip_utc_hours_buy: [8, 16, 18]                   # 3h block (-4h)
  skip_utc_hours_sell: [8, 14, 16]                   # 3h block (-2h)
```

**選定基準**: 各時間帯の PnL 最悪値 (mean ≤ -3.0bps) のみ残留。

| 解除候補 (buy) | 元 PnL | 理由 |
|---------------|--------|------|
| UTC01 (JST10) | -2.43bps | VG 補完で十分 |
| UTC02 (JST11) | -2.78bps | VG 補完で十分 |
| UTC12 (JST21) | -2.19bps | 軽度 |
| UTC21 (JST06) | -2.19bps | 軽度 |

| 解除候補 (sell) | 元 PnL | 理由 |
|----------------|--------|------|
| UTC04 (JST13) | -5.56bps | **要注意だが n=7** → 継続監視 |
| UTC13 (JST22) | -1.91bps | 閾値未満 |

**期待効果**: 機会損失 ~30% 削減 → fill 数増加 → PnL 分散低下。
**リスク**: AS 増加。VG で緩和されるが、48h 観察期間を設けて検証。

### §4.2 sell offset 引き上げ

**根拠**: 118# §3.5 — sell PnL30 = -0.632bps (買計 +0.281bps) の非対称性。

```yaml
# Before
side_offset:
  sell: 0.14

# After (候補 1: 保守)
side_offset:
  sell: 0.18

# After (候補 2: 積極)
side_offset:
  sell: 0.22
```

**考察**: sell offset 引き上げは fill_rate_sell を低下させるが、AS_ratio_sell を削減する。
0.14 → 0.18 で AS -3〜5pt、fill_rate -2〜3pt と推定。
fill_rate は F1=70% に対し現在 78.3% で 8.3pt のマージンあり。

**推奨**: **0.18 で開始** → 48h 後に AS/PnL を確認 → 必要に応じ 0.22 へ追加引き上げ。

### §4.3 spread_adaptive 再調整

**根拠**: 118# §8.2 — postonly_reject 65 件 (18.9%)、narrow spread 時の sell 偏り。

```yaml
# Before
spread_adaptive:
  narrow_spread_bps: 2.0

# After
spread_adaptive:
  narrow_spread_bps: 2.5      # P50 近辺に引き上げ → narrow boost 発動頻度適正化
```

**期待効果**: postonly_reject -20〜30% 削減 (n≈15-20 件減)。
fill_rate への正の影響 (+0.5〜1pt)。

### §4.4 regime warm-up 問題の解消

**根拠**: 118# §3.3, §8.6 — unknown regime が AS=35.1%, PnL=-0.390bps で最悪。

**実装**: `StatePersistence` (113#) に regime_detector の状態を追加保存。

```python
# regime_detector の state を persistence 対象に追加
# 再起動時に warm-up なしで前回の regime 状態を復元
state_persistence.register("regime_detector", regime_detector.get_state)
```

**期待効果**: 再起動後 20 cycle の unknown 解消 → PnL +0.05〜0.10bps。
実装コスト: 低 (StatePersistence は 113# で汎用設計済み)。

---

## §5 Track B: SkipGate 再訓練

### §5.1 現行 SkipGate の課題

| 指標 | 097# 訓練時 | 168h 実測 | 問題点 |
|------|-----------|----------|--------|
| ROC-AUC | ~0.442 | (未再計測) | **ランダム分類器水準** |
| skip_rate | 10% → 15% | 9.7% | adaptive 閾値で自動調整中 |
| sell 判定 | 逆選別発生 | sell_enabled=false | **sell 側で無効化中** |
| サンプル数 | 215 | **759 filled** | **3.5x に増加 → 再訓練可能** |
| regime 特徴量 | 全件 unknown (定数 0) | ranging=32%, trending=16% | **復活可能** |

### §5.2 759 サンプル再訓練計画

118# Appendix F の計画を踏襲し、以下の手順で実施:

**Step 1: データ準備**
- `results/v460/fill_test/` の 8 JSONL ファイルから enriched_df 構築
- `build_preorder_as_features()` で特徴量 X, ラベル y を生成
- 759 filled records + spread_at_order 552 件

**Step 2: ベースライン再訓練**
- 現行と同じ構成: k=10, C=0.01, Logistic Regression
- Walk-Forward 20-fold (min_train=50, step=30)
- 097# 結果 (AUC=0.442, Skip20%=+0.405bps) との比較

**Step 3: regime 特徴量強制 include**
- `regime_trending`, `regime_ranging` を SelectKBest 除外対象から除く
- k=12 で regime 2 + base 10 の 12 特徴量

**Step 4: buy/sell 分割訓練** (§5.4 で詳述)

**Step 5: 評価・デプロイ** (§5.6 で詳述)

### §5.3 特徴量変更方針

#### 現行 16 特徴量の評価

| ランク | 特徴量 | 安定性 | 097# 重要度 | 変更 |
|-------|--------|--------|------------|------|
| 1 | `spread_jpy` | **安定** | 0.0494 | 維持 |
| 2 | `avg_trade_size` | **安定** | 0.0484 | 維持 |
| 3 | `offset_ratio` | 不安定 | 0.0455 | 維持 (監視) |
| 4 | `trade_count_60s` | **安定** | 0.0440 | 維持 |
| 5 | `hour_cos` | **安定** | 0.0406 | 維持 |
| 6 | `side_aligned_velocity` | **安定** | 0.0354 | 維持 |
| 7 | `buy_ratio` | **安定** | 0.0327 | 維持 |
| 8 | `trade_flow_imbalance_60s` | **安定** | 0.0327 | 維持 |
| 9 | `price_velocity_60s` | 不安定 | 0.0298 | 維持 (監視) |
| 10 | `side_buy` | 不安定 | 0.0217 | 維持 |
| 11 | `regime_trending` | N/A (定数 0) | — | **復活** |
| 12 | `regime_ranging` | N/A (定数 0) | — | **復活** |
| 13 | `regime_high_vol` | N/A (定数 0) | — | **復活** |
| 14 | `hour_sin` | 不安定 | 0.0053 | 維持 |
| 15 | `vpin_60s` | 不安定 | — | 維持 |
| 16 | `side_aligned_tfi` | **安定** | — | 維持 |

**主な変更点**:
1. **regime 特徴量 3 種の復活**: 759 サンプルで regime 分布が十分 (ranging=32%, trending=16%)
   → 定数 0 問題は解消。unknown regime の AS=35.1% は有力な説明変数。
2. **k 値の引き上げ**: k=10 → k=12-14 (サンプル/特徴量比が改善したため)
3. **新規追加は見送り**: `vpin_acceleration` (060# 提案) は効果限定的。
   データ量優先で既存特徴量の精度向上に集中。

### §5.4 buy/sell 分割モデル

**根拠**:
- sell SkipGate は逆選別を起こしていた (098# §4.2)
- sell AS のドライバーが buy と異なる (spread_bps, hour 依存性)
- 現在 sell_enabled=false で無効化中

**分割訓練の実現性**:
- buy: 382 filled (最低ライン 200+ を大幅超過)
- sell: 377 filled (同上)
- 各々 WF 10-fold (min_train=50, step=30) で訓練可能

**sell 専用モデルのポイント**:
1. `spread_bps` の重要度を検証 (sell は narrow spread で AS 増加する傾向)
2. `hour_cos` の sell 固有パターンを捕捉
3. `recent_sell_pnl_mean` (直近 sell PnL running average) を追加候補として検討
4. 逆選別 (skip 群の PnL > keep 群の PnL) が解消しているか を最優先で確認

**成功判定**: sell モデルの Skip20% PnL 改善が正 (+0.0bps 以上)、かつ逆選別なし。

### §5.5 訓練パイプライン実行手順

```powershell
# Step 0: fill_test 停止 (差し替え作業中)
# ※ または別ディレクトリで訓練のみ実行 (fill_test は止めずに可能)

# Step 1: データ準備
.venv\Scripts\python.exe scripts/v460/train_skip_gate.py `
    --data-dir results/v460/fill_test `
    --output models/v460/skip_gate_as_v2.pkl `
    --mode baseline

# Step 2: regime 特徴量強制 include
.venv\Scripts\python.exe scripts/v460/train_skip_gate.py `
    --data-dir results/v460/fill_test `
    --output models/v460/skip_gate_as_v2_regime.pkl `
    --mode regime-force --k 12

# Step 3: buy/sell 分割
.venv\Scripts\python.exe scripts/v460/train_skip_gate.py `
    --data-dir results/v460/fill_test `
    --output-buy models/v460/skip_gate_buy_v2.pkl `
    --output-sell models/v460/skip_gate_sell_v2.pkl `
    --mode split

# Step 4: 評価比較
.venv\Scripts\python.exe scripts/v460/train_skip_gate.py `
    --data-dir results/v460/fill_test `
    --evaluate --compare models/v460/skip_gate_as.pkl
```

**注**: `train_skip_gate.py` は既存の 097# 訓練スクリプトを拡張する形で整備。
既存の `scripts/v460/` 内に enrich / build_features ロジックがあれば再利用。

### §5.6 デプロイ判定基準

| 基準 | 閾値 | 確認方法 |
|------|------|---------|
| ROC-AUC | > 0.55 (改善) | WF 平均 AUC |
| Skip20% PnL 改善 | > +0.3bps | WF Skip20% 平均 |
| sell 逆選別 | 解消 (skip PnL < keep PnL) | sell モデル評価 |
| 過学習チェック | train-val AUC gap < 0.05 | fold 間一貫性 |

**デプロイ**: 上記を満たした場合、YAML を更新して fill_test 再起動。

```yaml
# デプロイ時の YAML 変更例
skip_gate:
  model_path: models/v460/skip_gate_as_v2_regime.pkl  # or split model
  buy_enabled: true
  sell_enabled: true    # ← 逆選別解消なら再有効化
  # 分割モデルの場合:
  # model_path_buy: models/v460/skip_gate_buy_v2.pkl
  # model_path_sell: models/v460/skip_gate_sell_v2.pkl
```

---

## §6 Track C: 新規 ML 導入 (高コスト・ph3 以降)

000# §2 のフェーズ定義に基づき、ph3 は **G2-train (SAC 4-seed 訓練)** フェーズ。
Track A/B による fill_test 改善後、以下の条件を満たしてから進入する。

### §6.1 SAC 4-seed 訓練 (G2-train)

000# §3.4 Gate 定義:
- 4 seed の Walk-Forward → worst-seed ROI > -2%
- 少なくとも 1 seed が +0.5bps/trade 以上

**必要条件**:
1. G1.2-full PASS (= WATCH から PASS への昇格、つまり F4d watch 解消)
2. WF バグ 6 件修正 (§6.2)
3. Oracle テスト完了 (§6.3)
4. execute_trade() 実装 (013# D-1)
5. ph3 Stop 条件の明文化 (112# §3.4)

**SAC の action space**: {buy, sell, hold} の 3 値。
fill_test の maker-only 戦略を SAC のアクション実行関数にラップして使用。

### §6.2 Walk-Forward バグ修正 (前提)

111# §5 で特定された 6 件のバグ:

| # | バグ | 深刻度 | 影響 |
|---|------|--------|------|
| P0-1 | Entry Gate crash (np.nan 除算) | Critical | WF 全体停止 |
| P0-2 | Fee 二重カウント (0.1% + 0.1%) | Critical | reward 信号汚染 |
| P0-3 | Val/Test 汚染 (1-fold overlap) | Critical | 過学習未検出 |
| P1-1 | Trade 誤分類 (partial fill = cancel) | High | PF 過大評価 |
| P1-2 | Reporter 3 重定義 | Medium | メンテナンス困難 |
| P1-3 | CalibrationMap 未ロード | High | 実行時 crash |

**P0-1〜P0-3 は SAC 訓練の信頼性に直結** → ph3 進入前に必ず修正。

### §6.3 Oracle テスト (理論上限の確認)

**目的**: 完全予測 (Oracle) エージェントが maker 0% 手数料で達成する PnL を測定。
**意義**: Oracle PnL が AS コストを下回る場合、いかなる ML 改善も理論上限に到達しない。
ph3 に数週間投資する前に「天井があるか」を確認する。

**実装**: `backtest/` 内の既存フレームワークを活用。
Oracle agent = 将来の mid_price を完全に知っている前提で、buy/sell/hold を決定。

---

## §7 Track D: 板情報 (OB) 特徴量の再活性化

### §7.1 OB 無効化の経緯と再評価

**経緯**:

| 時期 | 事象 | 結論 |
|------|------|------|
| 070# | 72+ モデル徹底サーチ | 全 ROC-AUC ≤ 0.54。**ただしサンプル 284 件で SNR=0.11** |
| 071# | OB 3 特徴量除去 | `spread_bps_ob`, `depth_imbalance_ob`, `side_aligned_imbalance` 廃止 |
| 072# | OB トグル実装 | `use_ob_features: false`。YAML 1 行で復元可能 |
| 現在 | **759 filled records** | **サンプル数は 070# 時の 2.7 倍** |

**再評価**:
070# の結論は「OB 特徴量がゴミ」ではなく**「284 サンプルでは ML が信号を検出できない」**。
759 サンプルで SNR が改善している可能性があり、**再検証の条件は整った**。

**プロジェクト名 "Microstructure Edge" の理念**: v460 は板情報・約定フローのマイクロストラクチャ
特徴量による優位性を掲げている (000# §0)。OB 特徴量を無効のまま放置することは理念に反する。

### §7.2 現行インフラの OB データ経路

```
[Coincheck API]
  GET /api/order_books (public)
       │
       ▼
[CoincheckAdapter.get_orderbook(depth=10)]
  → OrderBookSnapshot(bids=[(price,size),...], asks=[(price,size),...])
       │
       ├──→ [MakerPriceCalculator] --- 価格発見 (常時使用✅)
       │    _compute_maker_price(): best_bid/ask → 指値価格算出
       │    _get_mid_price(): mid = (bid + ask) / 2 → PnL 計測
       │
       ├──→ [FillRecord] --- データ記録 (⚠️ imbalance_enabled=false で None)
       │    orderbook_imbalance, bid_depth_total, ask_depth_total
       │
       ├──→ [SkipGate] --- AS 予測特徴量 (❌ use_ob_features=false)
       │    _OB_FEATURE_COLS: spread_bps_ob, depth_imbalance_ob,
       │    side_aligned_imbalance
       │
       ├──→ [MarketDataCollector] --- raw OB アーカイブ (✅ 蓄積中)
       │    data/v460/raw/orderbook/*.jsonl.gz
       │
       └──→ [feature_enricher.py] --- 学習パイプライン用 OB エンリッチ (✅)
            load_raw_orderbook() → _find_nearest_ob() → enrich_fill_records()
```

**致命的な問題**: `imbalance.enabled: false` のため、fill_record に
`orderbook_imbalance`, `bid_depth_total`, `ask_depth_total` が**一切記録されていない**。
168h の 830 filled records には OB データが**ゼロ**。

### §7.3 OB データ収集の即時有効化

**目的**: SG 再訓練 (Track B) や将来の分析に使うため、fill_record への OB データ記録を再開。
**SG 判定やサイドセレクションには影響させず、記録のみ**を有効化する。

#### 方法 1: imbalance.enabled を true に (即座だがサイド影響あり)

```yaml
imbalance:
  enabled: true    # OB imbalance 計算 + FillRecord 記録が有効に
  # ただし smart_side.enabled: false のままなら side 選択には影響しない
  # しかし _compute_orderbook_imbalance() が毎サイクル get_orderbook 呼ぶ
  # → API コール増加 (4 req/s制限内に収まるか検証済: 118# §8.1 で 429=0件)
```

#### 方法 2: FillRecord への OB 記録を imbalance_enabled とは独立にする (推奨)

```python
# run_fill_test.py L779-781 を変更:
# Before: imbalance_enabled 判定で None を入れていた
# After:  常に OB imbalance を計算・記録 (SkipGate/SideSelection とは独立)
orderbook_imbalance=self._maker_price._last_imbalance,  # 常時記録
bid_depth_total=self._maker_price._last_bid_depth,       # 常時記録
ask_depth_total=self._maker_price._last_ask_depth,        # 常時記録
```

**推奨**: 方法 2。OB データ記録とフィルタリングロジックを分離。
imbalance/smart_side は引き続き disabled に保ち、記録のみ行う。

**実装コスト**: 低 (数行変更 + `_compute_orderbook_imbalance` を毎サイクル呼ぶよう変更)

### §7.4 OB 特徴量の段階的復活

Track B (SG 再訓練) と統合した 3 段階の OB 復活計画:

**Stage 1: データ収集 (即時 — fill_test 再起動時)**
- §7.3 の方法 2 で FillRecord に OB データを常時記録
- `spread_at_order` (既に記録中) + `orderbook_imbalance` + `bid/ask_depth_total` の 4 値
- 追加の API コールは `get_orderbook(depth=5)` 1 回/サイクル (120s 間隔 → 0.008 req/s。余裕)

**Stage 2: 学習パイプラインの OB 統合 (Track B と同時)**
- SG 再訓練時に `feature_enricher.py` の `enrich_fill_records()` で OB データを結合
- raw OB アーカイブ (`data/v460/raw/orderbook/`) からの時系列マッチングも活用
- 特徴量候補:

| 特徴量 | 算出元 | 根拠 | 期待寄与 |
|--------|--------|------|---------|
| `spread_bps_ob` | best_bid/ask | 071# で除去されたが 759 サンプルで再検証 | sell AS の narrow spread 相関 (118# §8.2) |
| `depth_imbalance_ob` | bid/ask_vol_5 | 070# IC≈0 だったが n=284→759 | informed trader 検知 |
| `side_aligned_imbalance` | 上記 × side | 070# で除去 | side 固有の板偏り捕捉 |
| `bid_depth_slope` | bid_vol / price_range | `ztb/features/microstructure.py` L108 | 板の厚み勾配 |
| `ask_depth_slope` | ask_vol / price_range | 同 L111 | 同上 |
| `depth_ratio` | bid_depth / ask_depth | FillRecord の既存列 | bid/ask 量比率 (OB 圧力の直接指標) |

**Stage 3: OB 特徴量パフォーマンス検証**
- SG 再訓練で OB あり/なしの AUC 比較
- Skip20% PnL 改善の差分
- OB 特徴量が有意なら `use_ob_features: true` に切替

### §7.5 マイクロストラクチャ特徴量の活用展望

`ztb/features/microstructure.py` に 10 特徴量が定義済み:

```python
MICROSTRUCTURE_FEATURES = [
    "bid_ask_spread",           # スプレッド
    "depth_imbalance",          # 板不均衡
    "trade_flow_imbalance",     # 約定フロー不均衡
    "vwap_deviation",           # VWAP 乖離
    "trade_intensity",          # 約定密度
    "order_flow_toxicity",      # VPIN 近似
    "price_impact",             # 価格インパクト
    "micro_return_vol",         # マイクロリターンボラティリティ
    "bid_depth_slope",          # 板の厚み勾配 (bid)
    "ask_depth_slope",          # 板の厚み勾配 (ask)
]
```

これらは 1 分集約 Parquet から算出される設計 (001# §2.2) で、
現在は **backtest/training パイプライン用**。fill_test のリアルタイム判定には未使用。

**ph3 以降の展望**:
- SAC の観測空間 (observation space) に組込む候補
- `MarketDataCollector` が蓄積中の raw OB データから 1 分集約を生成し、
  `add_microstructure_features()` で 10 特徴量を算出 → SAC state に注入
- これは v460 "Microstructure Edge" の最終形態

---

## §8 先送り事項の実装計画

### §8.1 ph3 ブロッカー (即時着手必須)

ph3 進入前に解決しなければ**訓練結果が汚染**される項目:

| # | 項目 | 出典 | コスト | 実装計画 |
|---|------|------|--------|---------|
| **P0-1** | WF Entry Gate crash (np.nan 除算) | 111# §5.1 | 低 | nan チェック追加。`np.isnan()` guard |
| **P0-2** | WF Fee 二重カウント (0.1%+0.1%) | 111# §5.1 | 低 | fee 計算を 1 箇所に集約 |
| **P0-3** | WF Val/Test 汚染 (1-fold overlap) | 111# §5.2 | 中 | embargo 追加。fold 境界の修正 |
| **P1-1** | Trade 誤分類 (partial fill = cancel) | 111# §5.1 | 低 | partial fill ステータス判定修正 |
| **P1-2** | Reporter 3 重定義 | 111# §5.2 | 中 | DRY化。1 Reporter に統合 |
| **P1-3** | CalibrationMap 未ロード | 111# §5.2 | 低 | 初期化時ロード追加 |
| **A-OT** | Oracle テスト | 118# §8.5 | 中 | backtest/ フレームワークで実装 |
| **A-ET** | execute_trade() TODO | 013# D-1 | 中 | fill_test maker ロジックをラップ |
| **A-SC** | ph3 Stop 条件明文化 | 112# §3.4 | 低 | ドキュメント整備 |
| **A-FM** | 運用失敗モードテスト | 112# §3.3, 113# §7 | 高 | 429 burst / OOM / 板薄化 注入テスト |

**優先順位**:
1. P0-1〜P0-3 (WF バグ) — SAC 訓練の絶対前提
2. A-OT (Oracle テスト) — ph3 投資判断の根拠
3. A-ET (execute_trade) — SAC アクション実行基盤
4. その他

**スケジュール**: Track A+B と並行して Day 5-15 で着手。
fill_test 再測定の 168h 待ち期間を有効活用。

### §8.2 即時実行可能な未着手施策

| # | 項目 | 出典 | コスト | 期待効果 | Track との関係 |
|---|------|------|--------|---------|--------------|
| B6 | Offset 体系的 AB 探索 | 095# M10, 118# §5.8 | 中 | 因果バイアス解消 | 168h 再測定後に実施 |
| B7 | minimum_spread_guard | 118# §8.2 | 低 | postonly_reject 根本対策 | Track A §4.3 を拡張 |
| B8 | sell SkipGate ルールベース代替 | 118# §4.3 選択肢 D | 低 | sell 逆選別への暫定対策 | Track B §5.4 の代替策 |

#### B7: minimum_spread_guard の詳細

168h データで postonly_reject=65 件の spread 平均=1,915 JPY (全体平均 2,409 JPY の 79%)。

```yaml
# fill_test.yaml に新規追加
minimum_spread_guard:
  enabled: true
  min_spread_jpy: 1500     # spread < 1500 JPY → skip (postonly_reject 回避)
```

**実装**: `run_single_cycle()` の preflight check に spread 下限判定を追加。
既に `_compute_maker_price()` で spread を取得しているため、追加コストはゼロ。

### §8.3 棚上げ妥当だが再訪タイミング注意

| # | 項目 | 出典 | Disposition | 再訪条件 |
|---|------|------|------------|---------|
| C1 | AS 判定 horizon 30s→60s 変更 | 098# §6 | 📋 棚上げ (B2 で PnL120 Gate 追加済) | — |
| C2 | Event-driven サイクル間隔 | 088# | 📋 棚上げ (v461+) | WebSocket 安定性確立後 |
| C3 | Round-trip を primary KPI | 088# | 📋 棚上げ (E6 informational 蓄積中) | ph3 SAC で hold 期間を action 化した段階 |
| C4 | Smart Side 再有効化 (OB imbalance) | 091# | 📋 棚上げ (side_aligned_velocity 代替) | **§7.4 Stage 2 で OB 特徴量検証時に再評価** |
| C5 | fast_fill 持続期間制御 | 093# §6#4 | 📋 棚上げ (100# L2 済) | sell 防御改善後 |
| C6 | Sell 保持期間延長 | 095# M7 | ⏳ ph3 以降 | **SAC action space に組込む — §9 盲点 E4 参照** |

### §8.4 v461+/将来への先送り一覧

| # | 項目 | 出典 | 備考 |
|---|------|------|------|
| D1 | `scripts/v460/lib/` → `ztb/` 移動 | 118# E3, App G: E11 | skip_gate.py は import 20+ 箇所変更要 |
| D2 | `utils/` 70+ ファイル分割 (God Package) | 118# E4 | v461+ |
| D3 | `config/` vs `configs/` 整理 | 118# E5 | LOW |
| D4 | ドキュメント命名違反 28 件修正 | 118# E2 | LOW |
| D5 | `UnifiedTrainer` 2835 行 God Object 分割 | 118# §7 DUP3 | v461+ |
| D6 | `ztb/adaptation/` Dead code 整理 | 118# E8 | テスト参照残存→完全 Dead ではない |
| D7 | RiskRuleEngine / Reconciliation 実装 | 118# §7 Tier-3 | ~2000 行, ph5 |
| D8 | 多取引所 fail-over 自動切替 | 118# §8.7 | ph5 本番で必要 |
| D9 | VG イベントの JSONL 構造化ログ出力 | App G: E12 | ph3-pre |
| D10 | MC CVaR Binding 化 (informational→必須条件) | App G: E13 | ph5 |
| D11 | 自動再起動 (systemd/nssm/TaskScheduler) | 118# §8.3 | ph5 |
| D12 | WebSocket API 活用 (REST polling→WS) | 013# D-4 | **1-2 日工数。レイテンシ改善に直結 → ph3-pre で検討** |
| D13 | Event-driven cycle (120s 固定→板変動トリガー) | 118# §5.2 | v461+。WebSocket 前提 |
| D14 | bitFlyer product_code 正規化統一 | 013# C-5 | Coincheck 主につき低優先 |
| D15 | SimBroker リネーム / StreamBuffer 組込み | 118# E6, E7 | LOW / 将来 |

---

## §9 見落とし・盲点分析

### E1: OB データが fill_record に記録されていない (Critical)

**問題**: `imbalance.enabled: false` のため、168h の 830 filled records には
`orderbook_imbalance`, `bid_depth_total`, `ask_depth_total` が**全て None**。
プロジェクト名 "Microstructure Edge" でありながら、マイクロストラクチャデータを
蓄積していないという根本的矛盾。

**影響**: SG 再訓練 (Track B) で OB 特徴量を使おうとしても、
168h データには OB カラムが存在しない → raw OB アーカイブ (`data/v460/raw/orderbook/`)
からの時系列マッチング (feature_enricher の `_find_nearest_ob()`) に頼る必要がある。

**対策**: §7.3 (OB データ収集の即時有効化) を fill_test 再起動時に適用。
以降の fill_record には OB データが記録される。

### E2: 30s 判定 vs 120s エッジの構造的矛盾 (Strategic)

**問題**: PnL30s = -0.161bps (負) だが PnL120s = +0.186bps (正)。
119# 分析で「EE 除去後の enriched 期間は +218.8bps の黒字」と確認済み。
**v460 の本質的エッジは 120s 以降に顕在化する** のに、
Gate (F4) は 30s を primary 指標とし、fill_test は 1-leg 設計で保持期間概念がない。

**影響**: PnL30s を改善しようとする全施策が、実はエッジの本質から離れている可能性。
fill_test で PnL30s=0 を目指すこと自体が、120s のリバージョン利益を
過小評価する構造になっている。

**短期対策**: Track A/B で 30s PnL 改善は継続 (Gate 通過のため)。
**中期対策**: ph3 SAC で保持期間 (30s/60s/120s) を action に組込む設計が**大義に最も直結**。
**長期展望**: fill_test Gate の F4 primary を PnL120s に変更するか、
round-trip PnL を primary KPI に昇格するか、ph3 結果を待って判断。

### E3: time_filter 緩和と SG 再訓練の順序依存 (Important)

**問題**: 118# §8.8 で「SG 再訓練は time_filter 緩和前が望ましい」とされたが、
逆の視点もある: time_filter がブロックしている時間帯にはデータがないため、
SG は未知の時間帯で不安定になるリスクがある。

**推奨順序**:
1. time_filter Step 1 緩和 (Track A §4.1)
2. 48h データ蓄積 (新時間帯のデータを含む)
3. 蓄積したデータを**含めて** SG 再訓練 (Track B §5.2)
4. 新 SG モデルのデプロイ

### E4: sell PnL 非対称性の本質が offset 調整では解決しない (Strategic)

**問題**: sell PnL30 = -0.678bps vs buy = +0.222bps。差 0.9bps。
sell offset 引き上げ (§4.2) は対症療法であり、根本原因は:
1. BTC/JPY bid 側に informed buyer が多い (構造的バイアス)
2. sell の 30s 逆行は 120s で回復するが、fill_test が 30s で判定

**本質的解決策**: sell 保持期間延長 → ph3 SAC で action = {buy+30s, buy+120s, sell+30s,
sell+120s, hold} のような拡張 action space が最も直接的。

### E5: fill_record に OB データがあるのに SG 特徴量に未使用 (Important)

**問題**: FillRecord スキーマには `orderbook_imbalance`, `bid_depth_total`, `ask_depth_total`
のフィールドが定義されている (054# で追加)。しかし:
1. `imbalance_enabled=false` で記録されていない (E1)
2. 仮に記録されていても、SkipGate の `build_features_from_market_state()` は
   `use_ob_features=false` で OB 特徴量を生成しない

fill_test の `run_single_cycle()` で毎サイクル `get_orderbook()` を呼んで
MakerPriceCalculator に渡しているのに、そのデータの大部分が捨てられている。

### E6: 4-seed 検証が未実施 (v459 教訓②)

**問題**: v459 教訓として「1 seed/1 設定での評価は不十分」が明記されているが、
fill_test は実質 1 設定のみで 168h 実施。市場環境依存性の評価が弱い。

**対策**: ph3 SAC で 4-seed 訓練が必須 (G2-train Gate 定義)。
fill_test 段階では 1 設定が妥当 (探索空間が狭い) だが、
Track A+B 後の**再測定 168h が異なる市場環境**になることで事実上の 2 条件検証になる。

### E7: Coincheck 署名不一致 (013# C-3) の E2E 検証未実施

**問題**: 013# App-D で Coincheck API の HMAC 署名は実装済みとされているが、
実 API での E2E 検証記録がない。fill_test は public API (order_books, trades) +
private API (orders, accounts) を使用するが、private API の認証テストが
ドキュメント上で確認できない。

**リスク**: ph5 本番移行時に認証エラーが発生する可能性。
**対策**: ph5 前に private API の E2E テスト実施。fill_test が正常動作しているため
暗黙的に検証済みだが、明示的な記録が必要。

### E8: WebSocket 未活用 (013# D-4) のレイテンシ影響

**問題**: 現在は REST polling (120s 間隔) で板情報を取得。
Coincheck は WebSocket API (`wss://ws-api.coincheck.com/`) を提供しており、
リアルタイムで板更新を受信可能。

**影響**: 120s サイクル冒頭の `get_orderbook()` は最新だが、
注文後の timeout 監視中の板変化はポーリング間隔依存で取得が遅れる。
stale order 検出 (094#) のドリフト判定精度に影響。

**対策優先度**: 中。ph3-pre (WF バグ修正と並行) での検討を推奨。
D12 として §8.4 にリスト済み。

---

## §10 フェーズ全体のロードマップ

```
[現在地] 168h Gate G1.2-full WATCH 達成
    │
    ├─── Track A: パラメータチューニング (Day 1-2)
    │    ├── A1: time_filter Step 1  (7h→3h / 5h→3h)
    │    ├── A2: sell offset 0.14→0.18
    │    ├── A3: narrow_spread_bps 2.0→2.5
    │    ├── A4: regime state persistence
    │    └── A4b: minimum_spread_guard 追加
    │
    ├─── Track B: SkipGate 再訓練 (Day 3-5、A完了後)
    │    ├── B1: データ準備 + ベースライン再訓練
    │    ├── B2: regime 特徴量復活
    │    ├── B3: buy/sell 分割モデル
    │    └── B4: 評価 + デプロイ判定
    │
    ├─── Track D: OB 復活 (Day 1、A と同時)
    │    ├── D1: FillRecord OB 記録の常時有効化
    │    └── D2: OB 特徴量を Track B に統合
    │
    ▼
fill_test 再起動 (Track A+B+D の結果をデプロイ) [Day 5]
    │
    ├─── 48h 中間チェック (time_filter 緩和のAS影響確認)
    │
    ▼
168h 再測定完了 [Day 12]
    │
    ├─── G1.2-full PASS (F4d watch 解消) ───┐
    │                                        ▼
    ├─── WATCH 継続 → 追加チューニング   Phase C 前提作業 [Day 5-15 並行]
    │                                    ├── WF バグ 6 件修正 (P0-1~P1-3)
    └─── FAIL → §8.2 施策              ├── Oracle テスト
                                          ├── execute_trade() 実装
                                          ├── ph3 Stop 条件明文化
                                          └── WebSocket API 検討 (D12)
                                               │
                                               ▼
                                         ph3: SAC 4-seed 訓練 [Day 20+]
                                               │
                                               ▼
                                         G2-train 判定
```

**タイムライン見積り**:

| マイルストーン | 目標日 | 条件 |
|--------------|--------|------|
| Track A + D (OB 記録有効化) | Day 2 | パラメータ変更 + OB 記録 |
| Track B (SG 再訓練) 完了 | Day 5 | 48h データ蓄積後に再訓練 |
| fill_test 再起動 | Day 5 | YAML 更新 + 新モデル適用 |
| 48h 中間チェック | Day 7 | time_filter 緩和の安全確認 |
| 168h 再測定完了 | Day 12 | G1.2-full 再判定 |
| Phase C 前提作業 | Day 5-15 | WF バグ修正 + Oracle テスト (168h 待ち期間を活用) |
| ph3 進入判断 | Day 15+ | G1.2-full PASS + Phase C 完了 |

---

## §11 リスクと制約

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| SG 再訓練で AUC が改善しない | Track B 効果ゼロ | ルールベース代替 (118# §4.3 選択肢 D) |
| time_filter 緩和で AS 悪化 | F5 FAIL | 48h 観察 → 即座にロールバック可能 |
| sell offset 引き上げで fill_rate 低下 | F1 マージン縮小 | 現在 78.3% → 70% まで 8.3pt 余裕あり |
| regime state persistence のバグ | 判定不整合 | テスト先行 (状態保存/復元の往復テスト) |
| OB 特徴量追加で過学習 | SG 精度低下 | WF cross-validation で train-val gap 監視 |
| 168h 再測定で市場環境変化 | 比較困難 | multi_track で期間別メトリクス分析 |
| OB 取得による API コスト増 | rate limit 抵触 | 118# §8.1 で 429=0 件確認済。120s 間隔で余裕 |
| ph3 WF バグが想定以上に深刻 | ph3 遅延 | P0 のみ先行修正、P1 は後回し |
| Oracle テストで天井が低い | 戦略自体の限界 | **テイカー戦略やマルチ取引所展開を検討 (000# R2)** |
| 30s PnL 改善が 120s エッジを破壊 | 逆効果 | PnL30/60/120 の全 horizon を監視 |

**最悪シナリオ**: Track A+B+D で PnL 改善なし → Track C (SAC) も理論上限で blocked
→ **maker-only 戦略に根本的な限界** → テイカー戦略やマルチ取引所展開を検討
(000# §6 リスク項目 R2 に該当)

---

## §12 意思決定フロー

```
[START]
    │
    ▼
Track A (パラメータ) + Track D (OB記録) → 即時デプロイ
    │
    ▼
48h データ蓄積 (time_filter 緩和後の新データ含む)
    │
    ▼
Track B (SG 再訓練、OB 特徴量含む)
    │
    ▼
SG デプロイ判定 (§5.6 基準)
    ├── PASS → 新モデルで fill_test 再起動
    └── FAIL → ルールベース代替 (§8.2 B8) で再起動
    │
    ▼
fill_test 168h 再測定
    │
    ▼
G1.2-full 再判定
    │
    ├─── PASS (F4d watch なし)
    │    │
    │    ▼
    │    PnL30s > 0 ?
    │    ├── YES → ph3 進入 (SAC 訓練)
    │    └── NO  → PnL120s 正なら ph3 で保持期間最適化を狙う
    │              追加パラメータ探索 (AB テスト — §8.2 B6)
    │
    ├─── WATCH 継続
    │    │
    │    ▼
    │    F4d 以外に watch/fail ?
    │    ├── NO  → sell offset 追加引き上げ or SG sell 再有効化 → 再測定
    │    └── YES → 該当施策を 118# §9 Phase D から選択
    │
    └─── FAIL
         │
         ▼
         FAIL 指標に応じた対応:
         ├── F4 (PnL) FAIL → sell offset + SG 再訓練 + param_adapter 検証
         ├── F5 (AS) FAIL  → sell offset + SG 再訓練 + time_filter ロールバック
         └── F1 (fill_rate) FAIL → time_filter ロールバック + offset 下げ
```

---

## Appendix A: 168h 判定 JSON 主要値

```json
{
  "gate_result": "WATCH",
  "total_orders": 1174,
  "filled_orders": 830,
  "fill_rate_p90": 0.634,
  "attempted_fill_rate": 0.783,
  "pnl_30s_mean": -0.161,
  "pnl_60s_mean": -0.304,
  "pnl_120s_mean": +0.186,
  "as_ratio": 0.281,
  "skip_gate_ratio": 0.097,
  "queue_wait_median": 12.8,
  "regime_ranging": { "n": 512, "pnl": -0.080, "as": 0.227 },
  "regime_trending": { "n": 226, "pnl": +0.016, "as": 0.292 },
  "regime_unknown": { "n": 436, "pnl": -0.390, "as": 0.351 },
  "sg_delta": -0.271,
  "trailing_200_pnl": -0.316
}
```

## Appendix B: YAML パラメータ変更チェックリスト

Track A+B+D デプロイ時に変更する fill_test.yaml の項目一覧:

| パラメータ | 現行値 | 変更後 | Track | 備考 |
|-----------|--------|--------|-------|------|
| `time_filter.skip_utc_hours_buy` | [1,2,8,12,16,18,21] | [8,16,18] | A1 | 7h→3h |
| `time_filter.skip_utc_hours_sell` | [4,8,13,14,16] | [8,14,16] | A1 | 5h→3h |
| `side_offset.sell` | 0.14 | 0.18 | A2 | sell AS 抑制 |
| `spread_adaptive.narrow_spread_bps` | 2.0 | 2.5 | A3 | postonly_reject 抑制 |
| `skip_gate.model_path` | skip_gate_as.pkl | skip_gate_as_v2.pkl | B1-B3 | 再訓練モデル |
| `skip_gate.sell_enabled` | false | (判定次第) | B3 | 逆選別解消なら true |
| (OB 記録常時有効化) | imbalance_enabled 依存 | 常時記録 | D1 | コード変更 |
| (minimum_spread_guard) | なし | min_spread_jpy: 1500 | A4b | 新規追加 |
| (regime state persistence) | — | コード変更 | A4 | StatePersistence 拡張 |

**変更手順**:
1. fill_test 停止
2. OB 記録常時有効化のコード変更 (§7.3 方法 2)
3. regime state persistence のコード変更 + テスト (§4.4)
4. minimum_spread_guard のコード変更 + テスト (§8.2 B7)
5. YAML 更新 (上表の変更を適用)
6. 新 SkipGate モデルを `models/v460/` に配置 (Track B 完了後)
7. fill_test 再起動 (`--hours 168`)
8. 48h 後に中間チェック (AS/PnL の急変がないか confirm)

## Appendix C: 全先送り項目一覧

118# Appendix A の 53 OPEN items のうち、本ドキュメントで扱う項目の追跡マトリクス:

| 118# ID | 項目 | 本書での配置 | ステータス |
|---------|------|------------|----------|
| B10 | sell offset 引き上げ | §4.2 Track A | ⏳ 計画済 |
| D1 | データ 500 蓄積で再訓練 | §5 Track B | ⏳ 759 filled 達成→実行可 |
| D6 | SkipGate 抜本見直し | §5 Track B + §7 Track D | ⏳ 計画済 |
| D7 | sell 専用 SkipGate モデル | §5.4 Track B | ⏳ 計画済 |
| A5 | time_filter Phase 3 Step 1 | §4.1 Track A | ⏳ 計画済 |
| C5 | v458 WF バグ 6 件 | §6.2 + §8.1 | ❌ 未着手 |
| C8 | ph3 Stop 条件明文化 | §8.1 A-SC | ❌ 未着手 |
| E9 | 運用失敗モードテスト | §8.1 A-FM | ⚠️ 実装済・テスト未実施 |
| — | OB 記録常時有効化 | §7.3 Track D (新規) | ❌ 未着手 |
| — | minimum_spread_guard | §8.2 B7 (新規) | ❌ 未着手 |
| — | Oracle テスト | §6.3 + §8.1 A-OT | ❌ 未着手 |

---

*本ドキュメントは 000# (プロジェクト提案)、118# (バックログ深層分析)、及び*
*070#-072# (OB 特徴量経緯)、097#-098# (SkipGate 再訓練)、119# (161h 分析) に基づき、*
*168h Gate 判定結果を踏まえて作成。*
*次回 fill_test の結果に応じて §12 の意思決定フローに従い更新する。*
