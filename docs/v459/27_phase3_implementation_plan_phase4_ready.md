# Phase 3 実装計画 - 既存実装活用・Phase 4接続考慮版

**Date**: 2026-01-25  
**Update**: 既存実装を最大限活用し、効率的に実装  
**Phase**: Phase 3 Implementation Plan with Phase 4+ Connectivity  
**Duration**: 5-7日（既存実装活用で短縮）

---

## 実装方針: 既存実装の直接活用（コード重複回避）

### 🎯 基本方針: Option C採用

**保守性重視**: 新規実装を避け、既存クラスを**直接呼び出す**ラッパー方式

1. **ABTestingFramework拡張** (ztb/training/unified_optimizer.py)
   - ✅ 既存API維持: create_ab_test, run_ab_test, get_ab_test_results
   - 🔄 **ResultComparator内部活用**: Mann-Whitney U + t-test + 効果量
   - 🔄 **StatisticalValidator内部活用**: Holm-Bonferroni補正
   - 🔄 **p_mean_method()内部活用**: 既存関数を直接呼び出し
   - ✅ **コード重複なし**: 統計ロジックは既存実装のみ

2. **活用する既存クラス**:
   - **ResultComparator** (ztb/trading/production/result_comparator.py):
     - `_run_statistical_tests()`: Mann-Whitney U, t-test, Levene検定
     - `_calculate_effect_size()`: Cohen's d効果量
     - `_calculate_confidence_intervals()`: 信頼区間計算
   
   - **StatisticalValidator** (ztb/metrics/statistical_validator.py):
     - `_apply_multiple_testing_correction()`: Holm-Bonferroni補正
     - `_compare_strategies_statistically()`: ANOVA戦略間比較
   
   - **p_mean_method()** (ztb/metrics/metrics.py):
     - 幾何平均/算術平均による総合p値計算

3. **既存のBacktestReporter活用** (ztb/evaluation/walk_forward/reporter.py)
   - ✅ Walk-Forward評価に完全統合済み
   - ✅ close_reason tracking完備
   - ✅ テストカバレッジ完了
   - ✅ 追加作業不要

4. **メトリクス統一は完了** (ztb/metrics/metrics.py)
   - ✅ sharpe_ratio, max_drawdown, calculate_metrics実装済み
   - ✅ 追加作業不要

---

## 実装優先順位（既存活用版）

### 🔥 最優先: ABTestingFramework拡張（Phase 4評価基盤）

**実装対象**: `ztb/training/unified_optimizer.py` の `ABTestingFramework`

**既存実装の状況**:
- ✅ create_ab_test(): テスト作成API
- ✅ run_ab_test(): テスト実行API
- ✅ get_ab_test_results(): 結果取得API
- ✅ _perform_significance_test(): t-test + Cohen's d実装済み
- ⚠️ Mann-Whitney U検定: 未実装
- ⚠️ Cliff's Delta効果量: 未実装
- ⚠️ Holm-Bonferroni補正: 未実装
- ⚠️ p平均法統合: 未実装

**Phase 3での拡張内容**:

#### 1. 既存クラス統合による統計検定（コード重複なし）

```python
class ABTestingFramework:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 既存の統計クラスを内部活用（新規実装なし）
        from ztb.trading.production.result_comparator import ResultComparator
        from ztb.metrics.statistical_validator import StatisticalValidator
        
        self.result_comparator = ResultComparator(
            confidence_level=0.95,
            min_sample_size=30
        )
        
        self.statistical_validator = StatisticalValidator({
            "multiple_test_method": "holm",  # Holm-Bonferroni
            "alpha_level": 0.05,
            "confidence_level": 0.95
        })
        
        # 既存の実装を保持
        self.confidence_level = 0.95
        self.min_sample_size = 30
        self.test_results = {}
        self.active_tests = {}

def _perform_significance_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
    """統計的有意差検定（既存ResultComparator活用、コード重複なし）"""
    control_scores = [r["score"] for r in test_config["control_results"]]
    variant_scores = [r["score"] for r in test_config["variant_results"]]
    
    # 既存のResultComparatorを直接活用（Mann-Whitney U + t-test + 効果量）
    # Note: _run_statistical_testsはasyncなので、同期ラッパー経由で呼び出し
    import asyncio
    statistical_tests = asyncio.run(
        self.result_comparator._run_statistical_tests(control_scores, variant_scores)
    )
    
    # 既存のp_mean_methodを活用
    p_mean_result = self._perform_p_mean_method(
        control_scores, variant_scores, test_config.get("n_splits", 4)
    )
    
    # 統合判定（新規実装、軽量ロジック）
    combined = self._make_combined_decision(
        statistical_tests.get("t_test", {}),
        statistical_tests.get("mann_whitney", {}),
        p_mean_result
    )
    
    return {
        "t_test": statistical_tests.get("t_test"),
        "mann_whitney": statistical_tests.get("mann_whitney"),
        "levene": statistical_tests.get("levene"),  # 等分散検定
        "p_mean": p_mean_result,
        "combined_decision": combined,
        "effect_size": self.result_comparator._calculate_effect_size(
            control_scores, variant_scores
        )  # Cohen's d
    }
```

#### 2. p平均法の統合

```python
def _perform_p_mean_method(
    self, 
    control_scores: List[float], 
    variant_scores: List[float],
    n_splits: int = 4
) -> Dict[str, Any]:
    """
    p平均法による統計検証（richmanbtc氏のオリジナル手法）
    
    資料: https://note.com/btcml/n/n0d9575882640
    
    手法:
    1. リターン時系列をN個の期間に分割（N = n_splits）
    2. 各期間でt検定してp値を計算
    3. N個のp値の平均を取る（幾何平均推奨）
    4. mean(p) < alpha (例: 0.03) ならOK
    
    特徴:
    - 全ての期間で安定して優位な場合のみ有意
    - 1つでも劣る期間があるとp値が大きくなる
    - 時系列の変化に強い（直近の劣化を検出）
    """
    from ztb.metrics.metrics import p_mean_method
    from scipy.stats import ttest_1samp
    
    # 各期間に分割
    n_control = len(control_scores)
    n_variant = len(variant_scores)
    
    p_values = []
    for i in range(n_splits):
        # 期間iのデータ抽出
        start = i * n_control // n_splits
        end = (i + 1) * n_control // n_splits
        
        control_split = control_scores[start:end]
        variant_split = variant_scores[start:end]
        
        # 差分のt検定（variant - control > 0?）
        diff = np.array(variant_split) - np.array(control_split)
        if np.std(diff) == 0:
            p_values.append(1.0)
        else:
            t, p = ttest_1samp(diff, 0)
            if t > 0:  # variant優位方向
                p_values.append(p)
            else:
                p_values.append(1.0)
    
    # p平均法適用（幾何平均）
    mean_p = p_mean_method(p_values, method="geometric")
    
    # エラー率計算（第一種過誤率）
    type1_error = (mean_p * n_splits) ** n_splits / math.factorial(n_splits)
    
    return {
        "test_name": "p_mean_method",
        "p_values": p_values,
        "mean_p": mean_p,
        "n_splits": n_splits,
        "is_significant": mean_p < 0.03,  # richmanbtc氏推奨閾値
        "type1_error_rate": type1_error,
        "interpretation": "全期間で安定優位" if mean_p < 0.03 else "不安定または非優位"
    }
```

#### 3. 多重比較補正（既存StatisticalValidator活用、コード重複なし）

```python
def compare_multiple_conditions(
    self,
    conditions: List[str],
    metric_name: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """複数条件の統計的比較（既存StatisticalValidator活用）
    
    Note: 
        - 既存のcreate_ab_test/run_ab_testでペアごとのp値取得
        - StatisticalValidator._apply_multiple_testing_correction()で補正
        - コード重複なし、保守性向上
    """
    from itertools import combinations
    
    # 全ペアの組み合わせ
    pairs = list(combinations(conditions, 2))
    
    # 各ペアでABテスト実行してp値収集
    p_values = []
    test_results = []
    
    for cond_a, cond_b in pairs:
        test_id = self.create_ab_test(
            f"{cond_a}_vs_{cond_b}",
            control_params={"condition": cond_a},
            variant_params={"condition": cond_b},
            evaluation_function=lambda p: self._get_metric_value(p["condition"], metric_name)
        )
        result = self.run_ab_test(test_id)
        test_results.append((cond_a, cond_b, result))
        
        # Mann-Whitney Uのp値を使用（ノンパラメトリックのため保守的）
        p_values.append(result["mann_whitney"]["p_value"])
    
    # 既存のStatisticalValidatorで多重比較補正（Holm-Bonferroni）
    # Note: _apply_multiple_testing_correctionは既存実装、コード重複なし
    correction_result = self.statistical_validator._apply_multiple_testing_correction(
        p_values
    )
    
    # 棄却されたペアを特定
    rejections = [
        {"pair": pairs[i], "p_value": p_values[i], "adjusted_p": correction_result["adjusted_p_values"][i]}
        for i, rejected in enumerate(correction_result["rejected"])
        if rejected
    ]
    
    return {
        "n_comparisons": len(pairs),
        "alpha": alpha,
        "pairwise_results": test_results,
        "correction": correction_result,  # 既存実装の結果
        "rejections": rejections,
        "method": "holm_bonferroni"
    }
```

#### 4. ベースライン比較の追加

```python
def compare_with_baseline(
    self,
    model_condition: str,
    baseline_condition: str,
    metric_name: str = "net_roi"
) -> Dict[str, Any]:
    """モデルとベースラインの統計的比較
    
    Note: 既存のztb.analysis.baseline_comparison.BaselineStrategyと連携
    """
    # 既存のcreate_ab_test/run_ab_test活用
    test_id = self.create_ab_test(
        f"{model_condition}_vs_{baseline_condition}",
        control_params={"type": "baseline", "name": baseline_condition},
        variant_params={"type": "model", "name": model_condition},
        evaluation_function=lambda p: self._evaluate_strategy(p, metric_name)
    )
    
    result = self.run_ab_test(test_id, num_iterations=30)
    
    # 判定
    mann_whitney = result["mann_whitney"]
    p_mean = result["p_mean"]
    
    is_superior = (
        mann_whitney["is_significant"] and 
        mann_whitney["is_meaningful"] and 
        p_mean["is_significant"]
    )
    
    return {
        "model": model_condition,
        "baseline": baseline_condition,
        "metric": metric_name,
        "is_superior": is_superior,
        "decision": "superior" if is_superior else "comparable_or_inferior",
        "detailed_results": result
    }
```

**メリット**:
- ✅ **コード重複なし**: 統計ロジックは既存実装のみ
- ✅ **保守性向上**: 修正は1箇所のみ（ResultComparator/StatisticalValidator）
- ✅ **テスト資産活用**: 既存テストがそのまま使える
- ✅ **バグリスク低減**: 実績ある実装を活用

**工数**: 0.5日（ラッパー実装 + 統合テスト）

---

### ✅ Reporter統合（既に完了、追加作業最小）

**実装対象**: なし（BacktestReporterは完全統合済み）

**既存実装の状況**:
- ✅ [BacktestReporter](c:\Users\Admin\dev\zaif-trade-bot\ztb\evaluation\walk_forward\reporter.py#L245): 完全統合済み
- ✅ [evaluator.py](c:\Users\Admin\dev\zaif-trade-bot\ztb\evaluation\walk_forward\evaluator.py#L53)での使用: import済み
- ✅ [types.py](c:\Users\Admin\dev\zaif-trade-bot\ztb\evaluation\walk_forward\types.py#L177-L178): val_reporter/test_reporterフィールド定義済み
- ✅ テストカバレッジ: tests/unit/v459/test_reporter_v459.py 完備
- 🔄 [TrainingReporter](c:\Users\Admin\dev\zaif-trade-bot\ztb\training\unified_trainer\components\reporter.py#L13): Compatibility Shim（実体は ztb.training.unified_trainer.reporting.TrainingReporter）

**Phase 3での対応**:
1. TrainingReporterのShim状況を確認
2. 重複実装がないかチェック（scripts/v456/phase4/modules/reporter.py は Phase 4モジュールのため除外）
3. 必要に応じてドキュメント更新のみ

**工数**: 0.5日（確認・ドキュメント更新）

---

### 🎯 報酬設計検証（Phase 4最適設定決定）

**実装対象**: Config 3種類 + AB実験

**Phase 4での重要性**:
- Paper Trading最適報酬設定の決定
- Live Trading移行前の最終チューニング
- Stage 3（Curriculum）が最有望ならPhase 4採用

**実装内容**:
1. **Stage 1: Pure PnL** (`config/v459/experiments/reward_stage1_pure_pnl.yaml`)
2. **Stage 2: 固定ガイダンス** (`reward_stage2_fixed_guidance.yaml`)
3. **Stage 3: Decayガイダンス** (`reward_stage3_curriculum.yaml`)
4. AB実験実行（4 seed × 4 windows × 3 stages = 48実験）
5. 拡張ABTestingFrameworkで統計検定（t-test + Mann-Whitney U + p平均法）

**工数**: 0.5日（Config作成） + 3.0日（実験実行） + 0.5日（分析） = 4.0日

---

### 🛡️ リスク管理統合（Phase 4本番運用準備）

**実装対象**:
- Circuit Breaker相当機能（Env内実装）
- MTF因果性検証強化
- Scaler境界厳格化

**Phase 4での重要性**:
- Paper Trading保護機能
- Live Trading移行前のリスク管理完成
- 本番運用の最終防衛線

**実装内容**:
1. **Circuit Breaker**: Env内実装（daily_loss, consecutive_loss）
2. **MTF因果性**: check_scaler.py拡張
3. **Scaler境界**: fit()でVal/Testエラー化

**工数**: 0.5日（各機能実装） × 3 = 1.5日

---

## Phase 4接続ポイント

### Phase 3 → Phase 4 成果物移行

| Phase 3成果物 | Phase 4での使用 |
|--------------|----------------|
| **ztb/metrics/metrics.py** | Paper Trading評価メトリクス計算（既存活用） |
| **ABTestingFramework拡張版** | Backtest vs Paper Trading統計比較 |
| **p平均法統合** | 時系列変化に強い安定性評価 |
| **統一Reporter** | Paper Trading評価ループ（既存活用） |
| **最適報酬設定** | Paper Trading学習/評価 |
| **Circuit Breaker** | Paper Trading保護 |

### Phase 4準備チェックリスト

**Phase 3完了時に確認**:
- [ ] ABTestingFramework拡張版テスト済み（t-test + Mann-Whitney U + p平均法）
- [ ] 報酬Stage 1/2/3でAB比較データ取得済み
- [ ] p平均法で全期間安定性確認済み
- [ ] Circuit Breaker発動テスト済み
- [ ] 統一Reporter学習ループ統合済み（既存）

### Phase 4移行時の注意点

1. **Paper Trading特有の要素**:
   - リアルタイムスリッページ: Phase 3では静的、Phase 4では動的
   - 遅延シミュレーション: Phase 4で新規実装
   - 24時間連続動作: Circuit Breaker重要性増

2. **Phase 3データの再利用**:
   - 報酬設計Stage 3が最優秀ならPhase 4で採用
   - ABTestingFramework拡張版でPaper Trading vs Backtest比較
   - p平均法で時系列安定性検証（直近劣化の早期検出）

3. **Go/No-Go Gate 2判定**:
   - Phase 3でNet ROI > 5%達成必須
   - p平均法で全期間安定性確認（mean(p) < 0.03）
   - 未達の場合Phase 4延期、Phase 3追加改善

---

## 実装スケジュール（6.0日間、Reporter確認スキップで最短化）

| Day | タスク | 成果物 | Phase 4接続 | ステータス |
|-----|--------|--------|-------------|---------|
| **1** | ABTestingFramework拡張（ラッパー実装） | 既存クラス統合完了 | ✅ Phase 4比較基盤 | ✅ **完了** (0.5日) |
| **—** | ~~Reporter確認・ドキュメント更新~~ | ~~統合状況確認完了~~ | ✅ **スキップ（Phase 2完了済み）** | ✅ **完了** (0日) |
| **2** | 報酬Stage 1-3 Config作成 | 3設定完成 | ✅ Phase 4最適設定 | ✅ **完了** (0.5日) |
| **3-4** | 報酬AB実験実行 | 48実験データ | ✅ Phase 4採用判断 | ⏳ 次タスク |
| **5** | リスク管理統合 | CB/MTF/Scaler | ✅ Phase 4保護機能 | ⏳ 待機 |
| **6** | Phase 3完了 | 完了レポート | ✅ Phase 4着手準備 | ⏳ 待機 |
| **7-8** | （余裕期間） | - | - | ⏳ 待機 |

**工数削減**: Day 2 Reporter確認がPhase 2で完了済みのためスキップ

**Phase 3完了後即座にPhase 4着手可能**

**コード重複回避により保守性向上**: 統計ロジック修正は既存クラスのみ

---

## Phase 4準備の最小要件（Phase 3出口条件）

### Gate 1: 技術検証（必須）
- [x] Walk-Forward評価エラーなし
- [x] Entry Gate動作確認
- [x] 指標二重計上なし
- [x] **メトリクス統一完了**（既存ztb/metrics/metrics.py活用）
- [x] **統一Reporter完成**（BacktestReporter統合済み）

### Gate 2: 収益性検証（Go/No-Go判定）
- [ ] **Net ROI > 5%** ← Phase 3報酬AB実験で検証
- [ ] **Sharpe Ratio > 1.0** ← Phase 3で達成必須
- [ ] **Max Drawdown < 15%** ← Phase 3で確認
- [ ] **BuyAndHold超過** ← Phase 3ベースライン比較で検証
- [ ] **p平均法安定性** (mean(p) < 0.03) ← Phase 3で確認

### Gate 3: リスク管理（必須）
- [ ] **Circuit Breaker実装** ← Phase 3で対応
- [ ] **連続損失制限** ← Phase 3テスト
- [ ] **日次損失上限** ← Phase 3テスト

**Phase 3でGate 1/2/3全通過 → Phase 4 Go判定**

---

## 次のアクション

1. **Day 1-2実装開始**: ABTestingFramework拡張
   - `_perform_mann_whitney_test()` メソッド追加
   - `_compute_cliffs_delta()` メソッド追加
   - `_perform_p_mean_method()` メソッド追加（既存p_mean_method()活用）
   - `compare_multiple_conditions()` メソッド追加（Holm-Bonferroni）
   - `compare_with_baseline()` メソッド追加

2. **並行タスク**: Phase 3完了報告テンプレート準備
   - Doc28: phase3_completion_report.md
   - Phase 4接続ポイント明記

3. **Phase 4先行準備**（Phase 3並行作業）:
   - Paper Trading評価設計書ドラフト
   - スリッページ/遅延モデル調査
   - 24時間運用監視設計

---

## 既存実装の直接活用まとめ（コード重複なし）

### ✅ 直接活用する既存実装

1. **ResultComparator** (ztb/trading/production/result_comparator.py)
   - `_run_statistical_tests()`: Mann-Whitney U + t-test + Levene検定
   - `_calculate_effect_size()`: Cohen's d効果量
   - `_calculate_confidence_intervals()`: 信頼区間計算
   - **活用方法**: ABTestingFramework内で直接呼び出し（コード重複なし）

2. **StatisticalValidator** (ztb/metrics/statistical_validator.py)
   - `_apply_multiple_testing_correction()`: Holm-Bonferroni補正
   - `_compare_strategies_statistically()`: ANOVA戦略間比較
   - **活用方法**: ABTestingFramework内で直接呼び出し（コード重複なし）

3. **p_mean_method()** (ztb/metrics/metrics.py)
   - richmanbtc氏のオリジナル手法
   - **活用方法**: ABTestingFramework内で直接呼び出し

4. **BacktestReporter** (ztb/evaluation/walk_forward/reporter.py)
   - ✅ Phase 2で完全統合済み
   - ✅ close_reason tracking実装済み（tp/sl/reversal/manual）
   - ✅ evaluator.py統合完了（Line 456-470）
   - ✅ テストカバレッジ完了（23 tests）
   - ✅ **Phase 3で追加作業不要**

5. **メトリクス** (ztb/metrics/metrics.py)
   - sharpe_ratio, max_drawdown, calculate_metrics
   - 追加作業不要

### 🆕 新規実装する部分（最小限）

1. **ABTestingFramework軽量拡張**:
   - `__init__()`: ResultComparator/StatisticalValidatorのインスタンス化
   - `_make_combined_decision()`: 3検定統合判定（軽量ロジック、~30行）
   - `compare_with_baseline()`: ベースライン比較ラッパー（既存メソッド呼び出し）
   - **特徴**: 統計ロジックは一切実装せず、既存クラスを呼び出すのみ

2. **報酬設計Config**:
   - Stage 1/2/3の3種類（YAML設定ファイル）

3. **リスク管理機能**:
   - Circuit Breaker（Env内）
   - MTF因果性検証強化
   - Scaler境界厳格化

### 📊 コード重複回避の効果

| 項目 | 新規実装方式 | 既存活用方式（Option C） |
|------|-------------|-------------------------|
| **統計ロジック実装** | Mann-Whitney U等を再実装 | ✅ 既存クラスを呼び出すのみ |
| **保守箇所** | ABTestingFramework + ResultComparator | ✅ ResultComparator のみ |
| **テスト作成** | 新規テスト全作成 | ✅ 統合テストのみ（既存テスト活用） |
| **バグリスク** | 中（新規実装） | ✅ 低（実績ある実装） |
| **工数** | 1.0-2.0日 | ✅ 0.5日 |
| **可読性** | コード重複 | ✅ 明確な依存関係 |

---

**Phase 3 → Phase 4シームレス移行を実現（既存実装最大活用で効率化）**
