# 379# Pre-366# 市場理論 SAC 統合 + リファクタリング + ph3 次タスク設計

**Status**: P3-A/B/C 完了・P3-D 未着手  
**Parent**: 377# (ph3 統一方針), 378# (SAC stub 回避 + タイムライン文書化)  
**Date**: 2026-03-25 (updated: 2026-03-11)

---

## §1 目的

035# から 306# までの 10 個の pre-366# 市場理論システムを SAC の観測空間に接続し、
SAC が市場状態をより包括的に認識できるようにする。

## §2 現状分析

### §2.1 既存 SAC 特徴量 (12 → 17)

| # | Feature (FeatureRegistry) | 由来 | 内容 |
|---|---|---|---|
| 1 | `price_velocity` | scalping.py | 1-bar リターン |
| 2 | `micro_trend` | scalping.py | 5-bar 方向性指標 |
| 3 | `price_acceleration` | scalping.py | velocity の変化率 |
| 4 | `volume_surge` | scalping.py | 突発的出来高増 |
| 5 | `momentum_divergence` | scalping.py | fast/slow momentum 差分 |
| 6 | `tick_volume_ratio` | scalping.py | 出来高/平均出来高比率 |
| 7 | `order_flow_imbalance` | scalping.py | candle 形状ベース flow |
| 8 | `micro_volatility` | scalping.py | 短期 return std |
| 9 | `spread_pressure` | scalping.py | H-L range / body 比率 |
| 10 | `momentum_burst` | scalping.py | price×volume momentum |
| 11 | `liquidity_surge` | scalping.py | volume / recent max |
| 12 | `realized_volatility` | scalping.py | RV (return² 累積) |
| **13** | **`parkinson_sigma`** | **market_theory.py (379#)** | **305# Parkinson H/L σ** |
| **14** | **`vpin_proxy`** | **market_theory.py (379#)** | **107# VPIN toxicity** |
| **15** | **`kyle_lambda_proxy`** | **market_theory.py (379#)** | **266# Kyle λ proxy** |
| **16** | **`amihud_illiq`** | **market_theory.py (379#)** | **266# Amihud ILLIQ** |
| **17** | **`ema_velocity_bps`** | **market_theory.py (379#)** | **227#/200# EMA velocity** |

### §2.2 build_features.py 特徴量 (17 → 22)

| 区分 | 列数 | 内容 |
|---|---|---|
| V460 base | 10 | 10 microstructure proxy (OHLCV) |
| M2-M5 | 7 | BayesianRegime(4) + σ-Cluster(1) + GLFT fill(1) + VPIN(1) |
| **PRE366** | **5** | **parkinson_sigma + ema_velocity + kyle_λ + amihud + vpin** |
| **合計** | **22** | |

## §3 10 システムの SAC 接続マッピング

### §3.1 直接接続済み (5/10)

| 番号 | システム | 新 Feature | 接続方法 |
|---|---|---|---|
| 305# | Parkinson σ | `parkinson_sigma` | FeatureRegistry + build_features proxy |
| 107# | VPIN + Vol Guard | `vpin_proxy` | FeatureRegistry + build_features proxy |
| 266# | Kyle λ | `kyle_lambda_proxy` | FeatureRegistry + build_features proxy |
| 266# | Amihud ILLIQ | `amihud_illiq` | FeatureRegistry + build_features proxy |
| 227#/200# | EMA Velocity | `ema_velocity_bps` | FeatureRegistry + build_features proxy |

### §3.2 既存特徴量で間接カバー (3/10)

| 番号 | システム | カバーする既存 Feature | 根拠 |
|---|---|---|---|
| 035# | 4-state Regime | M2 `posterior_*` (build_features) | BayesianRegime は 035# の上位互換 |
| 054# | OB Imbalance (S1) | `order_flow_imbalance` | candle 形状 proxy で同等の情報 |
| 258# | AS Reservation | `parkinson_sigma` + `kyle_lambda_proxy` | AS δ* = f(σ², τ, k) → σ と λ で間接表現 |

### §3.3 Phase 3 future work (2/10) — env-internal state

| 番号 | システム | 理由 | 対策案 |
|---|---|---|---|
| 162#/228# | Inventory Skewing + Time-decay | fill event 依存 (OHLCV 不可) | ObservationBuilder 拡張 (§6.1) |
| 226# | Loss Boost | fill outcome 依存 (OHLCV 不可) | OptimizerTracker 拡張 (§6.2) |

## §4 実装詳細

### §4.1 ztb/features/market_theory.py (新規)

5 つの新 FeatureRegistry 特徴量を `@FeatureRegistry.register()` で登録。

```
parkinson_sigma:    ln(H/L) / (2·√(ln2)), rolling mean
vpin_proxy:         |CLV × volume| / Σvolume (rolling)
kyle_lambda_proxy:  (H-L) / (2·volume), z-score 正規化
amihud_illiq:       |return| / volume, z-score 正規化
ema_velocity_bps:   EMA(Δclose/close × 10000, span=5)
```

### §4.2 scripts/v460/build_features.py (修正)

`_add_pre366_proxy()` 関数を追加。`PRE366_FEATURES` (5列) を Parquet に追加。
`build_and_save()` で 22 列の存在を検証。

### §4.3 configs/v460/experiments/g2_sac_train.yaml (修正)

`features.selected` を 12 → 17 に拡張。

### §4.4 Feature import chain

```
unified_feature.py → import ztb.features.scalping (12)
                   → import ztb.features.market_theory (5)  # 379# 追加
update_medium_parquet.py → 同上
```

## §5 10 システム リファクタリング設計

### §5.1 リファクタリング方針

**原則**: 動作するコードの構造改善。機能変更は行わない。

| 対象 | 現状 | 改善案 | 優先度 |
|---|---|---|---|
| maker_microstructure.py | 360行 Mixin | σ推定/Kyle/Amihud を個別メソッドに明確分離 (済み) | ✅ 完了 |
| realized_volatility | O(n²) 可能性 | numpy cumsum で O(n) (366# T4 で既に実装済み) | ✅ 完了 |
| Welford online var | regime_detector | Bayesian filter 統合完了 (366# T5) | ✅ 完了 |
| velocity_math.py | 単一関数 | 完結・変更不要 | ✅ 完了 |
| fill_probability_model.py | GLFT k 推定 | 安定・変更不要 | ✅ 完了 |

### §5.2 追加リファクタリング (今回実施)

| # | 対象 | 内容 |
|---|---|---|
| R1 | market_theory.py features | scalping.py と同一パターン (numpy vectorized, FeatureRegistry decorator) で統一 |
| R2 | build_features.py | V460 + M2-M5 + PRE366 の 3 段構造にモジュール化 |
| R3 | g2_sac_train.yaml | コメントで feature 由来を明記 |
| R4 | Feature pipeline unification | FeatureRegistry 系と build_features 系の将来的統合設計 |

### §5.3 構造的改善: Feature Pipeline 統合設計

```
現在 (2パイプライン並行):
  SAC training → FeatureRegistry → full_registry_features.parquet (12→17列)
  v460 pipeline → build_features.py → v460_features.parquet (17→22列)

将来 (統合):
  FeatureRegistry に v460 microstructure proxy も登録
  → 単一 Parquet で全特徴量を管理
  → build_features.py は FeatureRegistry.compute_all() のラッパーに
```

## §6 Phase 3 次タスク設計

### §6.1 P3-A: Env-internal State Features (HIGH priority) — ✅ 完了

**目的**: 162#/228# (Inventory) / 226# (Loss Boost) を SAC 観測空間に追加

**実装**: `ztb/trading/environment/components/env_internal_trackers.py`
- `EnvInternalTracker` が `inventory_pressure`, `loss_risk`, `time_in_market` を追跡
- `HeavyTradingEnv` の観測空間に自動注入済み
- 既存テストで全機能のカバレッジ確認済み

### §6.2 P3-B: SAC Live Presence 完成 (CRITICAL)

| タスク | 内容 | 状態 |
|---|---|---|
| B1 | SB3 stub 回避 | ✅ 378# 完了 |
| B2 | `--once` 履歴記録 | ✅ 378# 完了 |
| B3 | 特徴量 Parquet 再生成 (17列) | ✅ z-score clipping 適用済 |
| B4 | SAC 訓練 + OOS gate | ✅ 軽量化 (20K steps, 1 seed) で実行・PyTorch DLL fix |
| B5 | sidecar_signal.json デプロイ確認 | ✅ orchestrator_mid_cycle.py に統合済 |
| B6 | fill_test live loop との統合テスト | ✅ 既存 Mixin 構造で CycleGateAggregator.evaluate() に注入済 |

### §6.3 P3-C: OOS Gate Tuning (MEDIUM priority) — ✅ 完了

```yaml
# g2_sac_train.yaml gate 条件
E1: positive_seed_ratio ≥ 0.75
E2: roi_seed_std ≤ 0.03
E3: convergence (30K以降 ROI変動 ≤ 5%)
E4: worst_seed_roi > -0.02
```

**実装内容 (379# P3-C):**

1. **Neutral Bias Fallback** (`sac_retrain_scheduler.py`)
   - OOS Gate 失敗時 (ROI 不足 or trade_count 不足) に `_push_neutral_fallback()` を自動実行
   - `directional_bias=0.0, confidence=0.0, model_version="neutral"` を sidecar に書き込み
   - 市場環境激変時にSACの介入を自動遮断する安全弁として機能

2. **Sidecar IO mtime キャッシュ** (`sidecar_signal_io.py`)
   - `read_sidecar_signal()` に `st_mtime` ベースのインメモリキャッシュを導入
   - ファイル内容未変更時はディスク I/O + JSON パースを完全スキップ
   - キャッシュヒットでも TTL チェックは動的に実施 (時間経過による stale 検出)
   - ファイル削除時はキャッシュも自動クリア

3. **テスト** (31 tests all PASSED)
   - `TestPushNeutralFallback`: neutral signal の書き込み検証
   - `TestReadSidecarCache`: mtime キャッシュヒット / 新規書き込み時の無効化 / ファイル削除時クリア
   - `TestRetrainOnce::test_oos_failed`: neutral fallback 呼び出し検証を追加

### §6.4 P3-D: Feature Importance Analysis (LOW priority)

訓練後に 17 特徴量の SHAP/permutation importance を計算し、
有効な特徴量サブセット選定。冗長特徴量の除去で SAC 効率向上。

### §6.5 実行順序

```
P3-B3 (Parquet 再生成)     ✅ 完了
P3-B4 (訓練)              ✅ 完了 (軽量化モード)
P3-B5/B6 (デプロイ)       ✅ 完了 (orchestrator_mid_cycle 統合)
P3-A (env-internal)        ✅ 完了 (env_internal_trackers.py)
P3-C (OOS gate)            ✅ 完了 (neutral fallback + IO cache)
P3-D (importance analysis) ⬜ 未着手 → NEXT
```

### §6.6 発見された課題一覧

| # | 課題 | 影響 | 対策 | 状態 |
|---|---|---|---|---|
| I1 | PyTorch DLL 競合 (`WinError 1114`) | Windows 環境で SAC 訓練がクラッシュ | `import torch` を SB3 より前に実行 | ✅ 修正済 |
| I2 | 毎サイクルの sidecar 同期 I/O | ライブループのレイテンシ劣化 | mtime キャッシュ導入 | ✅ 修正済 |
| I3 | OOS 失敗時の stale バイアス残留 | 危険なバイアスが残り続ける | neutral fallback 自動発行 | ✅ 修正済 |
| I4 | g2_sac_train.yaml の軽量化設定 | 本番訓練には不十分 (20K steps) | 本番運用時に復元必要 | ⚠ 要対応 |
| I5 | Feature Pipeline 二重管理 | FeatureRegistry と build_features の乖離リスク | §5.3 統合設計で将来解消 | ⬜ 技術負債 |
| I6 | `_small.parquet` データの代表性 | 80K行では市場全体を表現できない | 本番では full parquet 使用 | ⚠ 要対応 |

## §7 テスト

### §7.1 単体テスト

- `tests/unit/core/features/test_market_theory_features.py` — **23 tests** all PASSED
  - 各特徴量: shape, NaN-free, range, known values, edge cases
  - FeatureRegistry 統合テスト
- `tests/unit/v460/test_sidecar_sac_integration.py` — **63 tests** all PASSED
  - SidecarSignal round-trip, CycleGate injection, BPS offset, confidence scaling
- `tests/unit/v460/test_sac_retrain_scheduler.py` — **31 tests** all PASSED
  - SACRetrainConfig, Trigger, retrain_once (cold/warm/oos), neutral fallback, IO cache
- **合計: 117 tests all PASSED**

### §7.2 統合テスト

- [x] SAC env が 17 特徴量で obs 構築可能
- [x] 20K steps 軽量訓練完走 (1 seed)
- [x] sidecar_signal.json → orchestrator_mid_cycle → CycleGateAggregator 統合確認
- [ ] 50K steps × 4 seeds 本番訓練 (I4 解消後)

## §8 影響範囲

| ファイル | 変更内容 |
|---|---|
| `ztb/features/market_theory.py` | **新規**: 5 特徴量登録 |
| `scripts/v460/build_features.py` | `_add_pre366_proxy()` 追加, `PRE366_FEATURES` 追加 |
| `configs/v460/experiments/g2_sac_train.yaml` | `features.selected` 12→17 |
| `ztb/features/unified_feature.py` | market_theory import 追加 |
| `scripts/v460/update_medium_parquet.py` | market_theory import 追加 |
| `tests/unit/core/features/test_market_theory_features.py` | **新規**: 23 テスト |
| `docs/v460/379_pre366_sac_integration.md` | **新規**: 本設計書 |

## §9 リスク・制約

| リスク | 対策 |
|---|---|
| 特徴量 12→17 で SAC sample efficiency 低下 | 50K steps では十分。低下時は P3-D で feature selection |
| kyle_lambda_proxy と amihud_illiq の相関 | z-score 正規化で scale 分離済み。SHAP 分析で検証予定 |
| build_features.py と FeatureRegistry の二重管理 | §5.3 統合設計で将来解消 |

## §10 revision history

| 版 | 日付 | 内容 |
|---|---|---|
| v1.0 | 2026-03-25 | 初版: 10 システム SAC 接続 + リファクタリング + ph3 設計 |
| v2.0 | 2026-03-11 | P3-A/B (B3-B6)/C 完了。neutral fallback + mtime IO cache 実装。117 tests PASSED |
