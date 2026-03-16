# 371# ph2 366# 市場理論システム配線 (M2-M5)

| 項目 | 値 |
|---|---|
| 文書番号 | 371# |
| フェーズ | ph2 G1.1-exec |
| 前提文書 | 366# (M1-M5 実装), 370# (TUNE-4R + DD 緩和) |
| 作業日 | 2026-03-15 |
| 方針 | 366# で実装済みの M2-M5 を live 実行パスに配線 |

---

## §1 背景

366# で M1-M5 の市場理論システムを実装・テスト済みだったが、調査の結果:

| システム | 実装状態 | 配線状態 (before) |
|---|---|---|
| M1 Microprice L5 | ✅ `265e768de` | ✅ Active — `compute_microprice_bias_bps()` 直結 |
| M2 Bayesian Regime | ✅ Phase A 完了 | ❌ **未配線** — 単体テストのみ |
| M3 σ-Clustering | ✅ `1b8d8e55f` | ❌ **未配線** — 単体テストのみ |
| M4 GLFT Fill Prob | ✅ `59a30956c` | ❌ **未配線** — 単体テストのみ |
| M5 Volume-Sync VPIN | ✅ `cf28375b7` | ⚠ **部分配線** — ML pipeline のみ、live path は time-based |

M2-M5 の **4 システムが live 実行パスに接続されていなかった**。

---

## §2 配線設計

### §2.1 M2: Bayesian Regime Filter → `regime_detector.py`

**注入点**: `RegimeDetector.update()` 内、`RegimeResult` 作成後

```
update(timestamp, mid_price)
  ├─ 既存: 閾値ベース分類 → RegimeResult (confidence)
  └─ NEW: BayesianRegimeFilter.update(ret) → posterior
         confidence_blended = 0.6 × threshold_conf + 0.4 × bayes_conf
```

- 既存の閾値分類を**破壊しない** (フォールバック維持)
- confidence のみベイズ事後確率で**ブレンド**
- `set_bayesian_filter()` で遅延注入 (dependency inversion)
- `bayesian_offset_multiplier` プロパティで offset 乗数を公開

**State Persistence (バグ修正)**:
- `get_state()` に `bayesian_filter` キーを追加
- `restore_state()` で `BayesianRegimeFilter.restore_state()` を呼出し
- リスタート時の事後分布リセット問題を解消

### §2.2 M3: σ-Clustering → `adaptation_engine.py`

**注入点**: `_build_adapt_kwargs()` 内

```
_build_adapt_kwargs(regime_detector=...)
  ├─ 既存: YAML step_ratio を構築
  └─ NEW: VolatilityRegimeClassifier.classify(vol_ratio)
         step_ratio × σ_cluster_offset_mult
         LOW=×0.8 | MID=×1.0 | HIGH=×1.3 | EXTREME=×2.0
```

- `set_sigma_classifier()` で遅延注入
- 3 箇所の `_build_adapt_kwargs()` 呼出しに `regime_detector=` kwarg を追加

### §2.3 M4: GLFT Fill Probability → `maker_microstructure.py`

**注入点**: `_apply_as_reservation_shift()` L216

```
AS δ* = γσ²τ + (2/γ)·ln(1 + γ/k)
  ├─ 既存: k = config.as_delta_star_fill_rate_k (静的定数)
  └─ NEW: k = FillProbabilityModel.k (OLS 推定値、動的)
         フォールバック: 推定失敗時は config.k を使用
```

**Calibration Cycle (バグ修正)**:
- `AdaptationEngine` に `set_fill_prob_model()` を追加
- `try_auto_adapt()` 内でレコード取得後に `_calibrate_fill_prob_model()` を呼出し
- fill_records の `effective_offset_used` と `filled` から A/k を再推定
- `MakerPriceCalculator` と `AdaptationEngine` が**同一インスタンス**を共有

### §2.4 M5: Volume-Sync VPIN → `skip_gate_evaluator.py`

**注入点**: VPIN setter ブロック (L1115 付近)

```
VPIN 計算:
  ├─ 既存: vpin_60s = time-based VPIN (feature_enricher)
  └─ NEW: vpin_vol_sync_enabled → _compute_vol_sync_vpin(recent_trades)
         Easley (2012) 出来高バケット VPIN
         フォールバック: 例外時は time-based VPIN を維持
```

- `_compute_vol_sync_vpin()` ヘルパー追加
- 累積配列構築 → `compute_vpin_volume_sync()` 呼出し
- 負 amount ガード追加 (`max(amount, 0.0)`)

---

## §3 設定体系

### §3.1 新規 Config フィールド (fill_config.py)

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `bayesian_regime_enabled` | `bool` | `False` | M2 有効化 |
| `bayesian_regime_stickiness` | `float` | `0.90` | 遷移行列の対角成分 |
| `bayesian_regime_emission_lr` | `float` | `0.01` | emission 学習率 |
| `sigma_clustering_enabled` | `bool` | `False` | M3 有効化 |
| `sigma_clustering_low_threshold` | `float` | `0.6` | LOW/MID 境界 |
| `sigma_clustering_high_threshold` | `float` | `1.5` | MID/HIGH 境界 |
| `sigma_clustering_extreme_threshold` | `float` | `3.0` | HIGH/EXTREME 境界 |
| `glft_dynamic_k_enabled` | `bool` | `False` | M4 有効化 |
| `glft_dynamic_k_min_samples` | `int` | `20` | 推定最小サンプル数 |

### §3.2 YAML 設定 (fill_test.yaml)

```yaml
bayesian_regime:
  enabled: true
  stickiness: 0.90
  emission_lr: 0.01

sigma_clustering:
  enabled: true
  low_threshold: 0.6
  high_threshold: 1.5
  extreme_threshold: 3.0

glft_dynamic_k:
  enabled: true
  min_samples: 20

vpin_vol_sync:
  enabled: true
  bucket_btc: 0.05
  n_buckets: 50
```

---

## §4 バグ修正

| # | 問題 | 重要度 | 修正内容 |
|---|---|---|---|
| B1 | M4 GLFT `fit()` 未呼出し — k がフォールバック固定値のまま | **高** | `AdaptationEngine._calibrate_fill_prob_model()` を追加。`try_auto_adapt()` 内で定期呼出し |
| B2 | Bayesian filter state 未永続化 — リスタートで事後分布リセット | **中** | `regime_detector.get_state()` / `restore_state()` に Bayesian filter 状態を統合 |
| B3 | VPIN 負 amount ガードなし | **低** | `max(float(amount), 0.0)` ガード追加 |

---

## §5 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/fill_config.py` | M2-M4 config フィールド追加 (10 fields) |
| `scripts/v460/lib/fill_config_parser.py` | M2-M5 YAML セクション解析追加 |
| `scripts/v460/lib/regime_detector.py` | M2 Bayesian filter 統合 + state persistence |
| `scripts/v460/lib/adaptation_engine.py` | M3 σ-clustering + M4 GLFT calibration |
| `scripts/v460/lib/maker_microstructure.py` | M4 GLFT dynamic k |
| `scripts/v460/lib/skip_gate_evaluator.py` | M5 Volume-Sync VPIN + 負amount guard |
| `scripts/v460/run_fill_test.py` | M2-M4 コンポーネント初期化 |
| `configs/v460/fill_test.yaml` | M2-M5 設定セクション追加 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | KNOWN_YAML_OVERRIDES 追加 |

---

## §6 テスト結果

```
$ python -m pytest tests/unit/v460/ -x -q --tb=short --no-cov
4460 passed, 33 skipped, 12 warnings in 29.18s
```

- 全テスト pass、リグレッションなし
- M2-M5 固有テスト: 98 passed
- drift prevention テスト: 4 passed

---

## §7 366# ステータス更新

| # | 提案 | 実装 | テスト | **配線** | 状態 |
|---|---|---|---|---|---|
| M1 | Microprice L5 | ✅ 265e768de | ✅ | ✅ Active | **完了** |
| M2 | Bayesian Regime | ✅ Phase A | ✅ 42 tests | ✅ **371# 配線** | **完了** |
| M3 | σ-Clustering | ✅ 1b8d8e55f | ✅ | ✅ **371# 配線** | **完了** |
| M4 | GLFT Fill Prob | ✅ 59a30956c | ✅ | ✅ **371# 配線+calibration** | **完了** |
| M5 | Volume-Sync VPIN | ✅ cf28375b7 | ✅ | ✅ **371# 配線** | **完了** |

**M1-M5 全システム live 配線完了。**

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-15 | 1.0 | 初版 — M2-M5 配線 + バグ修正 3 件 |
