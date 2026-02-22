# 38. 特徴生成最適化計画 + 既存実装監査レビュー

**対象**: `docs/v459/35_feature_generation_optimization_plan.md` / `docs/v459/37_existing_implementation_audit.md`  
**日付**: 2026-01-26  
**結論**: 方向性は良く、詰めも改善。ただし**既存実装の読み違いとAPI不整合が残っており、現状のまま実装すると失敗する**。  

---

## 1. サマリー（要点）

- **Tier 0: 再計算排除**が最優先という判断は適切  
- ただし**「既存実装でほぼ完結」「新規70行」**は現状のコードと合致しない  
- **FeatureRegistry / FeatureSetConfig / MTF制御**まわりの理解にズレがあり、ここを修正しないと再計算排除が成立しない  

---

## 2. 良くなった点（評価）

- **固定/変動コスト分離**で推定が現実的に修正された  
- **再計算排除を最優先**にした意思決定は正しい  
- **sklearn依存の削除**など、環境制約への配慮が反映された  
- **実装ボリューム削減**の姿勢は保守性の観点で良い  

---

## 3. 重大な不整合（修正必須）

### 3.1 FeatureRegistry APIの読み違い
`docs/v459/35_feature_generation_optimization_plan.md` と `docs/v459/37_existing_implementation_audit.md` の多くのサンプルが、**実在しないAPI**を前提にしています。

- `FeatureRegistry.get_fast_features()` / `get_standard_features()` / `get_all_features()` は存在しない  
- `FeatureRegistry.get_feature_names()` は存在しない  
- `FeatureRegistry.compute_features()` は `feature_names` / `return_timing` を受け取らない  

**実在するのは**:
- `FeatureRegistry.compute_features_batch(df, feature_names=..., return_timing=...)`
- `FeatureRegistry.list()`（全登録特徴量）
- `FeatureRegistry.get_optimized_feature_set()`（※ただし分析ファイルが必要）

**修正案**:
- サンプルコードは `compute_features_batch()` を前提に書き換える  
- “fast/standard/full” は **FeatureSetConfigに存在しない**ため、`feature_set` は `minimal` / `default` / `high_quality` などに置換  

### 3.2 MTF無効化が現状コードでは不可能
計画書では `include_multi_timeframe_features=False` で MTF を無効化できる想定ですが、  
現行コードでは `ztb/trading/environment/heavy_env/mixins/initialization.py` にて**強制的に MTF が有効化**されています。

```
include_mtf = feature_flags.get("include_multi_timeframe_features", False)
if not include_mtf:
    logger.info("Forcing enable of multi-timeframe features (v455 requirement)")
    include_mtf = True
```

**結論**: MTF削減を狙うなら**設定変更ではなくコード修正が必要**です。

### 3.3 Parquet/Featherの読み込み経路が未接続
計画書では **「Parquet保存 → ABRewardExperimentで読む」** だけで再利用できる想定ですが、  
UnifiedTrainerのデータロードは `load_csv_data()` 固定です。  
`ABRewardExperiment` 側で読み込んでもTrainer内部に渡る仕組みがないため、  
**実装が空回りする可能性が高い**です。

**必要な修正例**:
- `data_config.data_path` の拡張子を見て `DataLoader.load_parquet()` を使う分岐を追加  
  もしくは  
- Trainerへ **DataFrame直接注入**できる経路を追加  

### 3.4 `prepare_cached_data.py` は「特徴付き」ではない
`docs/v459/37_existing_implementation_audit.md` 内に矛盾があります。

- **正しい理解**: `prepare_cached_data.py` は **生データのFeather化のみ**  
- **誤記**: 「特徴生成済みデータを生成できる」扱い  

**Go基準に含めるなら修正が必要**です。

### 3.5 FeatureCacheの種類が混在
監査では `ztb.utils.cache.feature_cache.FeatureCache` を前提にしていますが、  
実際に特徴計算で使われているのは `ztb.features.processors.caching.cache.feature_cache`（メモリキャッシュ）です。  
**ディスクキャッシュ（utils側）とは別物**で統合されていません。

**結論**: “FeatureCache有効化”だけでは性能改善は起きないため、  
**どちらを使うかの整理と統合方針が必須**です。

---

## 4. 中程度の懸念（仕様修正推奨）

### 4.1 既存「相関削減」は分析ファイル前提
`FeatureRegistry.get_optimized_feature_set()` は  
`reports/feature_analysis_*.json` がない場合は全特徴量を返す設計です。  
現状の計画だと“削減されない”可能性が高いので、  
**分析ファイル生成の工程を明記**してください。

### 4.2 事前計算データの必須列が未定義
特徴生成済みデータを保存する場合、**OHLCV + timestamp は必須**です。  
`HeavyTradingEnv` は OHLC から価格配列を作るため、  
**特徴列のみの保存では訓練に失敗**します。

### 4.3 MTF事前計算のファイル形式不整合
MTF事前計算を Parquet にする想定ですが、  
`MultiTimeframeDataPipeline._load_single_timeframe()` は CSV のみ読込みです。  
Parquet利用には loader側の修正が必要です（またはCSV保存に合わせる）。

### 4.4 数値整合性検証の基準が混在
35番の「誤差<1e-5」→ 3.5で緩和されたのは良いが、  
FeatureRegistryは `feature_float_dtype=float16` が既定で、  
誤差基準が厳しすぎる可能性があります。  
**基準値の根拠（dtype依存）**を明記するとよいです。

---

## 5. 誤記・不整合（軽微だが修正推奨）

- セクション番号が重複（4.4 / 4.3 逆転）  
- 例示コードに `import` の誤記がある  
- `compute_ichimoku_pandas()` の例が未完成（`n` 未定義）  

---

## 6. 具体的な修正提案（最小で成立させる）

### ✅ 最短で成立させる修正案
1. **`FeatureRegistry.compute_features_batch()` を採用**  
2. **FeatureSetConfig の既存セットに統一**  
   - `minimal` / `default` / `high_quality` / `v451` など  
3. **MTF無効化をコードで許可**  
   - `include_mtf` を強制 True にしない  
4. **Parquet読み込み経路をTrainer側に追加**  
5. **“FeatureCache”は どちらを使うか明示**  

---

## 7. 最終判断

**現状のまま実装 → 高確率で失敗（API不一致 + MTF強制）**  
**修正を入れれば「1日で完結する最小実装」路線は成立可能**です。

---

**Reviewed by**: Codex  
**Status**: 要修正（API整合・MTF制御・読み込み経路の追加が必須）  
