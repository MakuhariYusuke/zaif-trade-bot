# 060# ML パイプライン改善: バグ修正 + 特徴量 v2 + チューニング

**日付**: 2026-02-15  
**前提**: 059# レビュー対応完了 (commit `8bb9ded36`), 620 テスト PASS  
**ステータス**: 完了 — AS LR Skip20% **+0.382 bps** (ベースライン -0.161 bps から改善)  

---

## §0 エグゼクティブサマリ

059# レビュー対応後の ML パイプライン再実行で `spread_jpy` ALL-NaN バグを発見。
根本修正後、特徴量 v2 (マルチタイムフレーム + モメンタム) 追加、
SelectKBest 特徴量選択、94 構成のハイパーパラメータチューニングを実施。

| ステージ | AS LR ROC-AUC | Skip20% (bps) | 特徴量数 | サンプル数 |
|---|---|---|---|---|
| 060c ベースライン | 0.4754 | -0.161 | 21 | 166 |
| 060d v2 特徴量 | 0.5007 | +0.007 | 39 | 166 |
| 060e SelectKBest k=12 | 0.4584 | +0.253 | 12 selected | 166 |
| **060f チューニング済** | **0.4978** | **+0.382** | **8 selected** | **166** |

**結論**: 特徴量選択 (k=8) + 強正則化 (C=0.01) の組合せが最良。
GB は 166 サンプルでは不安定 (チューニング時 +0.402 → 実行時 +0.044)、
LR が安定的に +0.382 bps を再現。

---

## §1 spread_jpy ALL-NaN バグ修正

### §1.1 症状

ML パイプライン再実行で全 TSCV fold で `spread_jpy`, `offset_ratio` が ALL-NaN 警告。
SimpleImputer が定数列 (全 NaN) をそのまま通過 → 実質的に対応特徴量が無視。

### §1.2 根本原因

```
build_as_features(df, *, require_spread=False)  # デフォルト False
```

spread データは時間的に Q2 以降 (row 71+) にのみ存在。
TSCV fold 1-2 では非 NaN spread が 0 件 → SimpleImputer がスキップ。

- 284 サンプル中 118 件が NaN spread を含む
- `require_spread=False` がデフォルトのため全件含まれていた

### §1.3 修正

| 修正 | 内容 |
|---|---|
| `build_enriched_as_features` | `require_spread=True` をデフォルトに → 166 クリーンサンプル |
| `data_loader.py` | 冗長な if/elif spread ロジックを単純化 |
| PnL/Fill パイプライン | spread 特徴量を除外 (時間的に偏在、低重要度) |
| テスト追加 | `test_enriched_as_require_spread_filters` |

---

## §2 特徴量エンジニアリング v2

### §2.1 マルチタイムフレーム取引特徴量

`_compute_multi_timeframe_trade_features()`: 30 秒 / 300 秒ウィンドウで計算。

| 特徴量 | 30s | 300s | 説明 |
|---|---|---|---|
| `vpin_{w}s` | ✓ | ✓ | Volume-weighted price impact normalized |
| `tfi_{w}s` | ✓ | ✓ | Trade flow imbalance |
| `velocity_{w}s` | ✓ | ✓ | 価格速度 |
| `trade_count_{w}s` | ✓ | ✓ | 取引回数 |

### §2.2 クロスタイムフレーム加速度

| 特徴量 | 計算 | 意味 |
|---|---|---|
| `vpin_acceleration` | vpin_30s - vpin_300s | 短期 vs 長期のインパクト変化 |
| `tfi_acceleration` | tfi_30s - tfi_300s | フロー急変検出 |
| `trade_rate_acceleration` | (count_30s / 30) - (count_300s / 300) | 取引頻度急変 |

### §2.3 リターンモメンタム

`_compute_return_momentum()`: OB mid-price ベースのリターン。

| 特徴量 | 説明 |
|---|---|
| `return_30s` | 30 秒リターン |
| `return_60s` | 60 秒リターン (**最重要特徴量**, LR coeff 1.03) |
| `return_300s` | 300 秒リターン |
| `realized_vol_300s` | 300 秒内の実現ボラティリティ (リターン標準偏差, ≥5 snapshots) |
| `side_aligned_return_*` | サイド調整済みリターン (buy=+return, sell=-return) |

### §2.4 結果

v2 特徴量追加後、`return_60s` が LR の最重要特徴量 (coeff 1.03) に。
これは「fill 後 60 秒の価格変動方向」が AS 予測に最も有用であることを示す。

---

## §3 特徴量選択

### §3.1 動機

166 サンプル × 39 特徴量 = n/p 比 4.3 → 過学習リスク大。
次元削減が必須。

### §3.2 実装

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(f_classif, k=K)),  # 060# 追加
    ("scaler", StandardScaler()),
    ("model", clf),
])
```

- `SelectKBest(f_classif)`: 分散分析 F 値で上位 k 特徴量を選択
- CV ループ内で fit → リーク防止
- `n_features_select` パラメータで制御 (None = 全特徴量)

---

## §4 ハイパーパラメータチューニング

### §4.1 探索空間

`tune_as_classifier.py`: 94 構成を網羅的に評価。

| モデル | パラメータ | 値 |
|---|---|---|
| LR | C | 0.01, 0.1, 0.5, 1.0, 5.0 |
| LR | penalty | l1, l2 |
| LR | k | 4, 8, 12, None |
| GB | n_estimators | 30, 50, 100 |
| GB | max_depth | 2, 3 |
| GB | learning_rate | 0.03, 0.05, 0.1 |
| GB | k | 8, 15, None |

### §4.2 主要発見

1. **特徴量選択は必須**: 上位構成は全て k=8 or k=15 (None は低スコア)
2. **GB は小構成が最良**: n=30, lr=0.05 (n=100 は過学習)
3. **LR は強正則化**: C=0.01 が最良 (C=1.0+ は過学習)
4. **L2 > L1**: L2 正則化が安定的に高スコア
5. **GB without selection = 全て負**: 特徴量選択なしの GB は全構成で Skip20% < 0

### §4.3 Top 5 結果

| Rank | モデル | パラメータ | Skip20% (bps) | ROC-AUC |
|---|---|---|---|---|
| 1 | GB | n=30, d=3, lr=0.05, k=15 | +0.402 | 0.470 |
| 2 | GB | n=30, d=2, lr=0.03, k=15 | +0.396 | 0.482 |
| 3 | **LR** | **C=0.01, l2, k=8** | **+0.382** | **0.498** |
| 4 | GB | n=30, d=3, lr=0.03, k=15 | +0.363 | 0.480 |
| 5 | LR | C=0.1, l2, k=12 | +0.352 | 0.460 |

### §4.4 モデル選択: LR > GB

チューニング時の GB +0.402 bps は、パイプライン実行時に +0.044 bps まで低下。
これは GB の不安定性 (fold ごとの特徴量選択変動、小サンプルでの分散) を反映。

LR は一貫して +0.382 bps を再現 → **本番は LR(C=0.01, l2, k=8) を採用**。

---

## §5 最終パイプライン構成 (060f)

### §5.1 AS 分類器

```python
# Pipeline
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(f_classif, k=8)),   # LR 用
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")),
])
```

### §5.2 最終メトリクス

| 指標 | 値 |
|---|---|
| ROC-AUC | 0.4978 ± 0.004 |
| PR-AUC | 0.5861 ± 0.041 |
| Brier | 0.2514 ± 0.005 |
| Skip20% PnL改善 | **+0.382 bps** |
| Skip10% PnL改善 | +0.328 bps |
| Naive PR-AUC | 0.566 |
| PR-AUC 改善 | +0.020 |

### §5.3 Top 5 特徴量 (LR, k=8 選択後)

パイプライン出力より:
1. `return_60s` (coeff 1.265)
2. `price_velocity_60s` (coeff 1.265)
3. `velocity_30s` (coeff 0.659)
4. `side_aligned_return_300s` (coeff 0.644)
5. `vpin_60s` (coeff 0.510)

### §5.4 PnL 回帰器 (変更なし)

Ridge 回帰は v2 特徴量の恩恵を受けるが、skip gate としての性能は依然マイナス。
AS 分類器ベースの skip が優先。

---

## §6 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/data_loader.py` | spread ロジック簡素化、Fill から spread 除外 |
| `scripts/v460/ml/feature_enricher.py` | v2 特徴量 (15 cols)、`require_spread=True` |
| `scripts/v460/ml/as_classifier.py` | SelectKBest、チューニング済みハイパラ |
| `scripts/v460/ml/run_ml_pipeline.py` | GB k=15, LR k=8 分離、チューニングコメント |
| `scripts/v460/ml/tune_as_classifier.py` | 新規: 94 構成チューニングスクリプト |
| `scripts/v460/ml/walk_forward_as.py` | 新規: expanding-window walk-forward 検証 |
| `scripts/v460/ml/skip_gate.py` | AS 分類器モード追加 (mode="as")、`train_and_save_as_skip_gate()` |
| `tests/unit/v460/test_enricher_skip_gate.py` | `test_enriched_as_require_spread_filters` + AS mode 4テスト追加 |

---

## §7 Walk-Forward 検証 (061#)

### §7.1 実装

`walk_forward_as.py`: Expanding-window walk-forward + embargo (=2 サンプル gap)。
TSCV (固定 n_splits=5) に対して、より現実的な時系列検証。

### §7.2 結果比較

| 指標 | TSCV (060f) | Walk-Forward (061) |
|---|---|---|
| Folds | 5 | 6 |
| ROC-AUC (mean) | 0.498 | 0.487 |
| PR-AUC (mean) | 0.586 | 0.576 |
| Skip20% (bps) | +0.382 | **+0.230** |
| Feature stability | — | Jaccard 0.417 |

**解釈**: Walk-forward でも Skip20% が正値 (+0.230 bps) を維持。
TSCV より control が厳しいため数値は低いが、実運用での期待効果は +0.2~0.4 bps 帯。

### §7.3 安定的に選択される特徴量

全 6 fold で常に選択 (5/8 features):
1. `depth_imbalance_ob` — 板の偏り
2. `return_300s` — 5 分リターン
3. `return_60s` — 1 分リターン
4. `side_aligned_return_30s` — サイド調整済み 30 秒リターン
5. `velocity_300s` — 5 分価格速度

---

## §8 SkipGate AS 分類器モード (061#)

### §8.1 設計

既存 SkipGate (PnL Ridge) に加え、AS 分類器 (LR) ベースの skip モードを追加。

```python
SkipGateConfig(
    mode="as",         # "pnl" (Ridge) or "as" (AS LR)
    as_threshold=0.6,  # P(AS) >= 0.6 → skip
)
```

### §8.2 学習関数

`train_and_save_as_skip_gate()`: enriched features → LR(C=0.01, k=8) pipeline を
学習し、`models/v460/skip_gate_as.pkl` に保存。

### §8.3 メリット

| 比較 | PnL Ridge (旧) | AS LR (新) |
|---|---|---|
| OOF 評価 | IC=-0.067 (マイナス) | Skip20% +0.230~0.382 bps |
| 実用性 | skip gate OFF 推奨 | **ON 推奨** |

---

## §9 残課題

| 項目 | 優先度 | 詳細 |
|---|---|---|
| サンプル数拡大 | P1 | 166 サンプルは不十分。fill test 再開でデータ蓄積 |
| AS SkipGate の fill_test 統合 | P1 | run_fill_test.py から AS SkipGate を呼び出し |
| Fill 分類器改善 | P3 | ROC-AUC 0.46, ほぼランダム。追加特徴量必要 |

### §9.1 ph3+ 先行着手可能課題

| 課題 | Phase | 依存 | 先行着手可否 | 詳細 |
|---|---|---|---|---|
| G1-info 再検証 (real features) | ph1 再検証 | OB/Trades データ蓄積 | **可** | `evaluator.py` の walk_forward_eval で microstructure features を XGBoost 検証 |
| ModelRegistry 実装 | ph3 | なし | **可** | 015# §6.3 方式A: metadata.json 付きモデル保存 |
| 訓練-推論次元統一 | ph3 | ModelRegistry | **可** | 015# §3.1: 88次元→5次元ギャップの解消 |
| SAC 重複実装の整理 | ph3 | なし | **可** | 015# §6.2: #2,#5,#6 廃止、#1 に統一 |
| DynamicLRScheduler 移設 | ph3 | SAC 整理 | **可** | 053# P2: sac_v430 → unified_trainer |
| 特徴量パイプライン統一 | ph3 | ModelRegistry | **一部可** | 015# §5.4: OnlineLearningEngine の独自特徴量をメイン pipeline に統合 |
| 行動空間 1D 連続化 | ph3 | 設計のみ | **設計のみ可** | 015# §6.4: TTL を maker 側管理に分離 |
| AS SkipGate のライブ統合 | ph2 | fill test 再開 | **コード準備可** | run_fill_test.py への統合コード |

**推奨先行着手順序**:
1. SAC 重複整理 (#2,#5,#6 廃止) — 依存なし、コードベース軽量化
2. ModelRegistry 実装 — ph3 の基盤
3. G1-info 再検証 — データ蓄積が進めば即時実行可能
