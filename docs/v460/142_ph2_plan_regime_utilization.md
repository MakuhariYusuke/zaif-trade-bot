# レジーム機能活用計画 — 既存資産の統合と段階的発展

**作成日**: 2026-02-23 (142# セルフチェック時)  
**目的**: 既存のレジーム検知インフラを最大限活用し、収益性向上に直結する施策を計画

---

## §1 既存レジームインフラ棚卸し

### §1.1 現在稼働中

| コンポーネント | 場所 | 出力 | 用途 |
|---|---|---|---|
| **FillTestRegimeDetector** | `scripts/v460/lib/regime_detector.py` | `FillTestRegime` (4値: trending/ranging/high_vol/unknown) | fill_test ループ内で mid_price からリアルタイム分類 |
| **141# regime_thresholds** | `scripts/v460/ml/skip_gate.py` | PnL 閾値のオーバーライド | skip_gate の PnL 閾値をレジーム別に調整 |

### §1.2 未活用の既存資産

| コンポーネント | 場所 | 出力 | 特徴 |
|---|---|---|---|
| **MarketRegimeClassifier** | `ztb/analysis/regime/market_regime_classifier.py` | `MarketRegime` (多値: 20+ alias) | RL 環境用の精綻な分類。v444 由来 |
| **BasicRegimeDetector** | `ztb/analysis/regime/basic_regime_detector.py` | 基本分類 | 軽量版 |
| **AdvancedRegimeDetector** | `ztb/analysis/regime/advanced_regime_detector.py` | 拡張分類 | マルチシグナル |
| **RegimeAdaptiveTrainer** | `ztb/training/components/regime_adaptive_trainer.py` | regime 別学習 | レジーム重み付き学習 |
| **regime_clustering** | `ztb/features/generators/technical/trend/regime_clustering.py` | クラスタリング | 教師なし分類 |
| **RegimePerformanceAnalyzer** | `ztb/analysis/comparative/regime_performance_analyzer.py` | 分析レポート | レジーム別 PnL 分析 |

### §1.3 ギャップ

- `FillTestRegime` (4値) と `MarketRegime` (多値: 20+ alias) の**マッピング層がない**
- regime 情報は skip_gate 閾値のみ影響 — **lot/offset/reprice は regime 無関係**
- retrain データはレジーム重み付けなし — **全レジーム均等**
- `RegimePerformanceAnalyzer` はバッチ分析用 — **リアルタイム未接続**

---

## §2 段階的活用計画

### Phase R-1: レジーム別パラメータ適応 (即効性: 高)

**概要**: 現行の `FillTestRegimeDetector` 出力を活用し、skip_gate 以外のパラメータもレジーム連動にする

| 施策 | 対象 | 現状 | 提案 | 期待効果 |
|---|---|---|---|---|
| **R-1a** | offset_ratio | 固定値 (YAML) | high_vol: +20%, trending: -15% | ボラ上昇時の約定率改善、トレンド時の利幅確保 |
| **R-1b** | lot サイズ | 固定値 | high_vol: ×0.7, ranging: ×1.0, trending: ×1.2 ※ preflight は調整後 lot 基準 | リスク管理 + トレンド追従 |
| **R-1c** | reprice 上限 | sell=1, buy=2 | 共に regime 連動 (high_vol: +1) | ボラ時の粘り強さ |
| **R-1d** | timeout | 固定 120s | trending: 90s, ranging: 150s | トレンド逃し防止、レンジでの忍耐 |

**実装コスト**: 低 (FillConfig にレジーム別パラメータ dict を追加、run_fill_test で参照)  
**リスク**: 低 (パラメータ変動幅に安全マージンを設ける)

### Phase R-2: レジーム重み付き retrain (即効性: 中)

**概要**: 直近のレジーム状況に合わせた学習データ重み付け

| 施策 | 内容 | 期待効果 |
|---|---|---|
| **R-2a** | **retrain データのレジーム重み付け**: 現在のレジームに近いサンプルを upweight | 現行レジームへの特化 (特に sell の high_vol 対策) |
| **R-2b** | **レジーム別 retrain 頻度**: high_vol では retrain 間隔短縮 (市場変化が速い) | 適応速度改善 |
| **R-2c** | **RegimeAdaptiveTrainer 設計資産の再利用**: 既存の `ztb/training/components/regime_adaptive_trainer.py` の特徴量・重み付け仕様を retrain_scheduler 用に adapter 層で再利用 (SAC Mixin のため XGBoost 直結は不可) | 既存設計の活用 |

**実装コスト**: 中 (`retrain_model()` に `sample_weight` エンリッチ、R-2c は adapter 層追加が必要)  
**リスク**: 中 (過学習リスク — 重み付けの減衰係数が必要)

### Phase R-3: 精緻分類へのアップグレード (即効性: 低/長期)

**概要**: `FillTestRegime` 4値 → `MarketRegime` の部分的統合

| 施策 | 内容 | 前提条件 |
|---|---|---|
| **R-3a** | **マッピング層**: `MarketRegime` → `FillTestRegime` への集約マッピング関数 | Phase R-1/R-2 の効果検証後 |
| **R-3b** | **サブレジーム閾値**: `trending` を `strong_bull_trend` / `weak_bear_trend` に細分化してパラメータを最適化 | 十分な FillRecord データの蓄積 |
| **R-3c** | **OB 特徴量連携**: `MarketRegimeClassifier` に OB micro-feature を入力して分類精度向上 | P2-10 (OB micro-feature) 完了後 |

**実装コスト**: 高 (MarketRegimeClassifier の fill_test 統合、状態管理)  
**リスク**: 高 (多値 (20+ alias) に対するパラメータチューニングの組合せ爆発)

---

## §3 推奨実行順序

```
即時 (143#-144#)        中期 (145#-147#)         長期 (148#+)
├── R-1a: offset 連動   ├── R-2a: retrain 重み    ├── R-3a: マッピング層
├── R-1b: lot 連動      ├── R-2b: retrain 頻度    ├── R-3b: サブレジーム
└── R-1d: timeout 連動  └── R-2c: Adaptive 連携   └── R-3c: OB+regime
```

### 優先理由
1. **R-1 系が最高 ROI**: 実装コスト低、既に `FillTestRegimeDetector` が稼働中でデータも流れている
2. **R-1a (offset 連動) が最優先**: offset は PnL に直結する最大パラメータ。ボラ上昇時に offset を広げることで fill 率と PnL のバランス改善が即座に期待できる
3. **R-1b (lot 連動)**: high_vol でロット縮小は「守り」の最重要施策

---

## §4 検証基準

| Phase | 成功指標 | 測定方法 | 統計ゲート |
|---|---|---|---|
| R-1 | sell PnL 改善 (high_vol 時) ≥ +0.1bps | regime 別 PnL 集計 (FillRecord.regime フィールド) | bootstrap 95% CI が 0 跨ぎない / 各レジーム最低 50 samples / fee控除後指標 |
| R-1 | 全体 PnL ≥ 現状維持 | 24h 連続 run の total_pnl | 同上 |
| R-2 | retrain 後 skip_precision 改善 ≥ 5% | OnlineMonitor.skip_precision | 同上 |
| R-3 | サブレジーム別 PnL 分散の縮小 | RegimePerformanceAnalyzer | 同上 |

### §4.1 段階導入方針 (142# §6 #3 対応)

R-1a/b/c/d は同時投入ではなく **1 変更ずつ A/B 実施** し、寄与分解を行う:

```
Step 1: R-1a (offset) → 24h 実測 → 統計ゲート確認
Step 2: R-1b (lot)    → 24h 実測 → 統計ゲート確認
Step 3: R-1c (reprice)→ 24h 実測 → 統計ゲート確認
Step 4: R-1d (timeout)→ 24h 実測 → 統計ゲート確認
```

---

## §5 注意事項

1. **パラメータ連動の粒度制御**: 全 4 レジームでパラメータを個別設定すると組合せ爆発する。まず **high_vol のみ** の差分から始め、効果を見て拡大
2. **レジーム変更のレイテンシ**: `FillTestRegimeDetector` のヒステリシス (3 サイクル連続一致) があるため、急激な市場変化への追従に ~6 分のラグがある。R-1 で offset/lot を急変させすぎないよう変動幅にクランプを設ける
3. **unknown レジームの扱い**: 信頼度低時は unknown → 全パラメータをデフォルト値に固定する安全策を維持
4. **バックテスト不可**: fill_test はリアルタイム実測。R-1 の効果検証は live dry-run で行う必要がある

---

## §6 Codexレビュー追記 (2026-02-22)

### §6.1 総評

- R-1 を先に進める方針は妥当です。既存の `FillTestRegimeDetector` を使って execution パラメータへ反映する方が、R-3 先行より ROI が高いです。
- ただし R-2/R-3 には「既存資産をそのまま接続できる」という前提ズレがあり、ここを放置すると実装コスト見積もりを誤る可能性があります。

### §6.2 指摘事項 (重大度順)

| # | 重大度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `docs/v460/142_ph2_plan_regime_utilization.md:61`, `ztb/training/components/regime_adaptive_trainer.py:68` | R-2c で `RegimeAdaptiveTrainer` 連携を想定しているが、実体は SAC 向け Mixin であり、fill test の XGBoost retrain へ直結できない。 | R-2c は「直接流用」ではなく「特徴量・重み付け仕様のみ再利用」に再定義し、`retrain_scheduler` 用の adapter 層を別タスク化する。 |
| 2 | MEDIUM | `docs/v460/142_ph2_plan_regime_utilization.md:101` | 成功指標が平均値中心で、統計的有意性や分散を考慮していない。+0.1bps はノイズに埋もれる可能性が高い。 | ブートストラップ CI、side/regime 別サンプル下限、fee 控除後指標を R-1 から必須化する。 |
| 3 | MEDIUM | `docs/v460/142_ph2_plan_regime_utilization.md:45` | R-1a/b/c/d は同時に効く execution パラメータで、同時投入すると寄与分解が困難。 | 1変更ずつ A/B 実施し、`offset -> lot -> timeout -> reprice` の順で段階導入する。 |
| 4 | LOW | `docs/v460/142_ph2_plan_regime_utilization.md:21`, `ztb/analysis/regime/market_regime_types.py:13` | `MarketRegime` を「40+値」としているが、現コードの正規 enum は約20種 + 互換 alias。 | ドキュメント表現を「多値（20+ alias）」へ修正し、マッピング設計の複雑度見積もりを更新する。 |
| 5 | MEDIUM | `docs/v460/142_ph2_plan_regime_utilization.md:46` | lot 適応で buy 側を増量する場合、preflight 残高チェックとの整合が未定義。実装側で不足エラーを増やしやすい。 | R-1b の受け入れ条件に「調整後 lot で preflight 判定する」を明記する。 |

### §6.3 次アクション (短期)

1. R-1b 仕様に「preflight は調整後 lot 基準」を追記して実装拘束を入れる。
2. R-2c を「SAC資産の直接流用」から「設計資産の再利用」へ表現修正する。
3. R-1 検証基準へ統計ゲート (CI/最小サンプル/fee控除後PnL) を追加する。
