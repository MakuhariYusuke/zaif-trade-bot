# v459 Phase 3: 統計検証・Reporter完全統合・報酬設計検証 仕様書 (24)

**Date**: 2026-01-25  
**Status**: 📋 Planning  
**Phase**: Phase 3 - Statistical Testing, Complete Reporter Integration, Reward Design Validation  
**Predecessor**: Phase 2 (P1-1, P1-2, P1-3 完了) - 16/16 tests passed

---

## 1. Executive Summary

### 1.1 Phase 2完了サマリー

v459 Phase 2を完了し、以下の基盤を確立しました：

**Phase 2実績**:
- ✅ P1-1: close_reason実装完了（7/7テスト）
- ✅ P1-2: Entry Price更新バグ修正完了（2/2テスト）
- ✅ P1-3: BacktestReporter統合完了（7/7テスト）
- ✅ Phase 2テスト全パス（16/16テスト、100%）
- ✅ 累積テスト維持（119/119テスト: Phase 0: 77件、Phase 1: 26件、Phase 2: 16件）
- ✅ Doc19実装レビュー対応完了（5件の実装修正）
- ✅ Doc21実装レビュー対応完了（7件の実装修正）

**Phase 2で延期された項目**:
- ⏸️ P1-4: AB Testing完全実装（記述統計のみ実装、統計検定は Phase 3へ）
- ⏸️ TrainingReporter統合（BacktestReporterのみ統合、TrainingReporterは Phase 3へ）
- ⏸️ entry_reason/hold_reason実装（履歴情報が必要なため Phase 3へ）
- ⏸️ MTF因果性検証強化（Phase 3へ）
- ⏸️ Scaler fit境界の厳密化（警告→エラー化、Phase 3へ）

### 1.2 Phase 3目的

Phase 2で確立した基盤の上に、**統計的妥当性の確保**、**Reporter完全統合**、**報酬設計の段階的検証**を実施します。

**Phase 3スコープ**:
- **P1-4完全実装**: AB Testing基盤（統計検定実装、4 seed対応、多重比較補正）
- **Reporter完全統合**: TrainingReporter統合、3実装→1実装への完全統一
- **報酬設計検証**: Stage 1/2/3の段階的実装とAB比較
- **リスク管理統合**: Circuit Breaker、Virtual Portfolio Manager連携
- **MTF/Scaler改善**: 因果性検証強化、fit境界厳密化

**完了条件**:
- [ ] P1-4完全実装（統計検定、4 seed対応、多重比較補正）
- [ ] TrainingReporter統合完了（互換API移植、2実装削除）
- [ ] 報酬設計3ステージ実装とAB比較データ取得
- [ ] 全テスト合格維持（Phase 0/1/2/3統合テスト）
- [ ] Phase 3完了報告作成

**工数見積もり**: 7-10日

**Doc00との整合性**:
- Doc00 Section 4 "Phase 3: 報酬設計の段階検証" に準拠
- Doc00 Section 5.2 "収益性検証基準" に準拠
- Doc00 Section 5.6 "統計検定仕様" に準拠
- Doc00 Section 3.3 "報酬設計の段階化" に準拠

---

## 2. Phase 3実装項目

### 2.1 P1-4: AB Testing完全実装

**Doc00準拠**: Section 5.6 統計検定仕様

**Phase 2での実装範囲**:
- ✅ ABTestingComparator基本クラス実装
- ✅ compute_descriptive_stats()実装（mean, std, median, min, max）
- ✅ 2 seed対応（基盤構築レベル）
- ⏸️ 統計検定（Phase 3へ延期）

**Phase 3での実装方針変更**:

**既存実装の直接活用（コード重複なし）**:

1. **ABTestingFramework軽量拡張** (ztb/training/unified_optimizer.py):
   - ✅ create_ab_test, run_ab_test, get_ab_test_results API実装済み
   - ✅ UnifiedOptimizerとの統合済み
   - 🔄 **ResultComparator活用**: Mann-Whitney U + t-test + 効果量（コード重複なし）
   - 🔄 **StatisticalValidator活用**: Holm-Bonferroni補正（コード重複なし）
   - 🔄 **p_mean_method活用**: 既存実装を直接呼び出し（ztb/metrics/metrics.py）

2. **活用する既存クラス**:
   - **ResultComparator** (ztb/trading/production/result_comparator.py):
     - `_run_statistical_tests()`: Mann-Whitney U, t-test, Levene検定
     - `_calculate_effect_size()`: Cohen's d効果量
     - `_calculate_confidence_intervals()`: 信頼区間計算
   
   - **StatisticalValidator** (ztb/metrics/statistical_validator.py):
     - `_apply_multiple_testing_correction()`: Holm-Bonferroni補正
     - `_compare_strategies_statistically()`: ANOVA戦略間比較
   
   - **p_mean_method()** (ztb/metrics/metrics.py):
     - richmanbtc氏のオリジナル手法、既に実装済み

3. **保守性重視**:
   - ✅ **コード重複なし**: 統計ロジックは既存実装のみ
   - ✅ **保守性向上**: 修正は1箇所のみ（ResultComparator/StatisticalValidator）
   - ✅ **テスト資産活用**: 既存テストがそのまま使える
   - ✅ **バグリスク低減**: 実績ある実装を活用

**Phase 3での実装**: 既存クラスを呼び出すラッパーメソッドのみ新規作成

#### 2.1.1 統計検定実装（既存クラス直接活用、コード重複なし）

**実装方針: ラッパーメソッドによる既存実装活用**

ABTestingFrameworkは統計ロジックを実装せず、既存のResultComparatorとStatisticalValidatorを**直接呼び出す**:

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

def _perform_significance_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
    """統計的有意差検定（既存ResultComparator活用、コード重複なし）
    
    Note:
        統計ロジックは一切実装せず、ResultComparator._run_statistical_tests()を
        直接呼び出すラッパーメソッド。保守性重視。
    """
    control_scores = [r["score"] for r in test_config["control_results"]]
    variant_scores = [r["score"] for r in test_config["variant_results"]]
    
    # 既存のResultComparatorを直接活用（Mann-Whitney U + t-test + 効果量）
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

**既存実装から得られる統計検定結果**:

ResultComparator._run_statistical_tests()の返り値:
```python
{
    "t_test": {
        "statistic": float,
        "p_value": float,
        "dof": int,
        "interpretation": str
    },
    "mann_whitney": {
        "statistic": float,
        "p_value": float,
        "interpretation": str
    },
    "levene": {
        "statistic": float,
        "p_value": float,
        "interpretation": str
    },
    "ks_test": {
        "statistic": float,
        "p_value": float
    }
}
```

**保守性メリット**:
- ✅ 統計ロジックはResultComparatorの1箇所のみ
- ✅ ABTestingFrameworkは呼び出すだけ
- ✅ バグ修正は1箇所で済む
- ✅ 既存テストがそのまま使える

**Cliff's Delta効果量計算**（新規実装、軽量）:
```python
def _compute_cliffs_delta(self, a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's Delta効果量計算
    
    Cliff's Delta = (dominance_ab - dominance_ba) / (n_a * n_b)
    
    where:
    - dominance_ab = #{(x,y) : x > y, x∈A, y∈B}
    - dominance_ba = #{(x,y) : x < y, x∈A, y∈B}
    
    Range: [-1, 1]
    - +1: A完全優位（全てのx > y）
    - 0: 差なし
    - -1: B完全優位（全てのx < y）
    
    Interpretation (Doc00準拠):
    - |d| < 0.147: negligible（無視可能）
    - 0.147 ≤ |d| < 0.33: small（小）
    - 0.33 ≤ |d| < 0.474: medium（中）- **実用的意義あり**
    - |d| ≥ 0.474: large（大）
    
    Performance:
        計算量: O(n_a * n_b)
        メモリ: O(1)
        
        Phase 3想定サイズ: 4 seed × 4 split = 16サンプル
        → 16 * 16 = 256比較（許容範囲）
        
        大規模データ（n > 1000）の場合:
        - ソートベース実装（O(n log n)）への切替を検討
        - numpy.searchsorted()を使用した最適化版
    
    Note:
        Phase 3では最大16サンプル程度のため、単純実装で十分
        Phase 4以降で大規模化する場合は最適化版に切替
    """
    n_a = len(a)
    n_b = len(b)
    
    if n_a == 0 or n_b == 0:
        return 0.0
    
    # すべてのペアについて優位性をカウント
    # Note: Phase 3の想定サイズ（16×16=256比較）では十分高速
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    
    delta = (greater - less) / (n_a * n_b)
    return float(delta)

def _interpret_effect_size(self, delta: float) -> str:
    """効果量の解釈"""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"
```

#### 2.1.2 p平均法統合（richmanbtc氏のオリジナル手法）

**資料**: https://note.com/btcml/n/n0d9575882640

**既存実装の活用**:
- `ztb/metrics/metrics.py`: `p_mean_method(p_values, method="geometric")` 実装済み
- `tools/analysis/sac_v438_deep_analysis.py`: `_analyze_p_average_method()` 使用例あり

**p平均法の特徴**:
1. リターン時系列をN個の期間に分割（N = n_splits）
2. 各期間でt検定してp値を計算
3. N個のp値の平均を取る（幾何平均推奨）
4. mean(p) < alpha（例: 0.03）ならOK

**強み**:
- 全ての期間で安定して優位な場合のみ有意
- 1つでも劣る期間があるとp値が大きくなる
- **時系列の変化に強い**（直近の劣化を検出）
- エラー率（第一種過誤率）: (mean(p) * N)^N / N!

**ABTestingFrameworkへの統合**:
```python
def _perform_p_mean_method(
    self,
    control_scores: List[float],
    variant_scores: List[float],
    n_splits: int = 4,
    alpha: float = 0.03  # richmanbtc氏推奨閾値
) -> Dict[str, Any]:
    """p平均法による統計検証（ABTestingFrameworkに追加）
    
    Note:
        既存のztb.metrics.metrics.p_mean_method()を活用し、
        ABTestingFrameworkの統計検定に統合
    
    Args:
        control_scores: コントロール群のスコア
        variant_scores: バリアント群のスコア
        n_splits: 期間分割数（デフォルト: 4、Phase 3は4 windows）
        alpha: 有意水準（デフォルト: 0.03、richmanbtc氏推奨）
    
    Returns:
        p平均法結果辞書:
        - p_values: 各期間のp値リスト
        - mean_p: p値の幾何平均
        - is_significant: mean_p < alphaかどうか
        - type1_error_rate: 第一種過誤率
        - interpretation: 安定性評価
    """
    from ztb.metrics.metrics import p_mean_method
    from scipy.stats import ttest_1samp
    import math
    
    # 各期間に分割
    n_control = len(control_scores)
    p_values = []
    
    for i in range(n_splits):
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
    
    # エラー率計算
    type1_error = (mean_p * n_splits) ** n_splits / math.factorial(n_splits)
    
    return {
        "test_name": "p_mean_method",
        "p_values": p_values,
        "mean_p": mean_p,
        "n_splits": n_splits,
        "is_significant": mean_p < alpha,
        "type1_error_rate": type1_error,
        "alpha": alpha,
        "interpretation": "全期間で安定優位" if mean_p < alpha else "不安定または非優位",
        "note": "時系列変化に強く、直近劣化を検出可能"
    }
```

**統合判定ロジック**:
```python
def _make_combined_decision(
    self,
    t_test_result: Dict[str, Any],
    mann_whitney_result: Dict[str, Any],
    p_mean_result: Dict[str, Any]
) -> Dict[str, Any]:
    """3つの統計検定結果を統合判定
    
    判定基準:
    1. p平均法で安定性確認（mean_p < 0.03）
    2. Mann-Whitney Uで有意差確認（p < 0.05）
    3. Cliff's Deltaで実用的意義確認（|d| > 0.33）
    4. t-testも参考（正規分布に近い場合は重視）
    
    Note:
        3検定全てで有意 → 強い証拠
        2検定で有意 → 中程度の証拠
        1検定のみ → 弱い証拠
    """
    significance_count = sum([
        t_test_result.get("is_significant", False),
        mann_whitney_result.get("is_significant", False),
        p_mean_result.get("is_significant", False)
    ])
    
    is_meaningful = mann_whitney_result.get("is_meaningful", False)
    
    if significance_count == 3 and is_meaningful:
        decision = "strong_evidence"
        confidence = "high"
    elif significance_count >= 2 and is_meaningful:
        decision = "moderate_evidence"
        confidence = "medium"
    elif significance_count >= 1:
        decision = "weak_evidence"
        confidence = "low"
    else:
        decision = "no_evidence"
        confidence = "none"
    
    return {
        "decision": decision,
        "confidence": confidence,
        "significance_count": significance_count,
        "is_meaningful": is_meaningful,
        "summary": self._generate_decision_summary(
            decision, t_test_result, mann_whitney_result, p_mean_result
        )
    }
```

#### 2.1.3 多重比較補正（Holm-Bonferroni法、ABTestingFrameworkに追加）

**Doc00準拠**: Section 5.6 多重比較補正

**既存実装の活用**:
- ABTestingFrameworkの `create_ab_test()` / `run_ab_test()` APIを活用
- 複数ペアのp値にHolm-Bonferroni補正を適用

```python
def compare_multiple_conditions(
    self,
    conditions: List[str],
    metric_name: str,
    alpha: float = 0.05,
    evaluation_function: Optional[Callable] = None
) -> Dict[str, Any]:
    """複数条件の統計的比較（多重比較補正付き、ABTestingFrameworkに追加）
    
    Note:
        既存のcreate_ab_test/run_ab_testを活用し、
        複数ペアの結果にHolm-Bonferroni補正を適用
    
    Args:
        conditions: 条件名リスト（例: ["stage1", "stage2", "stage3"]）
        metric_name: 比較する指標名
        alpha: 有意水準（デフォルト: 0.05）
        evaluation_function: 評価関数（Optiona、デフォルトは内部メトリクス取得）
    
    Returns:
        多重比較結果辞書:
        - pairwise_comparisons: ペアごとの比較結果
        - corrected_alpha: Holm-Bonferroni補正後の閾値リスト
        - rejections: 棄却された帰無仮説（有意な差）
    
    Note:
        Holm-Bonferroni法:
        1. 全ペアのp値を昇順ソート
        2. i番目（0-indexed）のp値に対し、修正閾値 = α/(m-i)
        3. p値が修正閾値を下回る限り棄却継続、上回ったら停止
        
        例: 3条件（3ペア）、α=0.05の場合
        - 1番目のp値: α/3 = 0.0167
        - 2番目のp値: α/2 = 0.0250
        - 3番目のp値: α/1 = 0.0500
    """
    from itertools import combinations
    
    # 全ペアの組み合わせを生成
    pairs = list(combinations(conditions, 2))
    
    # 各ペアでABテスト実行（既存APIを活用）
    pairwise_results = []
    p_values = []
    
    for cond_a, cond_b in pairs:
        # 既存のcreate_ab_test/run_ab_test活用
        test_id = self.create_ab_test(
            test_name=f"{cond_a}_vs_{cond_b}",
            control_params={"condition": cond_a},
            variant_params={"condition": cond_b},
            evaluation_function=evaluation_function or (lambda p: self._get_metric_value(p["condition"], metric_name))
        )
        
        result = self.run_ab_test(test_id, num_iterations=30)
        pairwise_results.append({
            "pair": (cond_a, cond_b),
            "result": result
        })
        
        # Mann-Whitney Uのp値を使用（ノンパラメトリックのため保守的）
        p_values.append(result["mann_whitney"]["p_value"])
    
    # Holm-Bonferroni補正
    n_comparisons = len(pairs)
    sorted_indices = np.argsort(p_values)
    corrected_alphas = [alpha / (n_comparisons - i) for i in range(n_comparisons)]
    
    # 棄却判定
    rejections = []
    for i, idx in enumerate(sorted_indices):
        if p_values[idx] < corrected_alphas[i]:
            rejections.append({
                "pair": pairs[idx],
                "p_value": p_values[idx],
                "corrected_alpha": corrected_alphas[i],
                "rank": i + 1
            })
        else:
            break  # 1つでも棄却失敗したら停止
    
    return {
        "metric": metric_name,
        "n_conditions": len(conditions),
        "n_comparisons": n_comparisons,
        "alpha": alpha,
        "pairwise_comparisons": pairwise_results,
        "corrected_alphas": corrected_alphas,
        "rejections": rejections,  # 有意な差が検出されたペア
        "n_rejections": len(rejections),
        "interpretation": self._interpret_multiple_comparison(rejections, len(conditions))
    }
```

#### 2.1.4 4 seed対応とサンプル数確保

**Doc00準拠**: Section 5.6 サンプル数要件（各条件 n ≥ 16）

**Phase 2実績**: 2 seed対応（基盤構築）  
**Phase 3目標**: 4 seed対応（統計検定要件）

**既存実装の活用**:
- ABTestingFrameworkの `run_ab_test(num_iterations=N)` パラメータ活用
- 4 seed × 4 windows = 16サンプル確保（Doc00要件達成）

**変更内容**:
- `seeds: [0, 1]` → `seeds: [0, 1, 2, 3]`（config/v459/experiments/ab_test_*.yaml）
- ABTestingFramework.run_ab_test()の num_iterations パラメータで調整可能

**Note**: Walk-Forward評価は各windowで「Train/Val/Test」の3期間に分割。
統計検定のサンプル数は「window単位」で計測（Val+Testを1つの測定値として集計）。

#### 2.1.5 ベースライン比較機能（ABTestingFrameworkに追加）

**Doc00準拠**: Section 5.5 ベースライン比較

**既存実装の活用**:
- `ztb/analysis/baseline_comparison.py`: 既存のベースライン実装を活用
  - `BuyAndHoldStrategy`: Buy-and-Hold戦略（実装済み）
  - `SMAStrategy`: SMA Crossover戦略（実装済み）
  - `BaselineResult`: ベースライン結果構造体（実装済み）
- ABTestingFrameworkの `create_ab_test/run_ab_test` APIを活用

**Phase 3での拡張**: ABTestingFrameworkに `compare_with_baseline()` メソッド追加

```python
def compare_with_baseline(
    self,
    model_condition: str,
    baseline_condition: str,
    metric_name: str = "net_roi",
    alpha: float = 0.05,
    evaluation_function: Optional[Callable] = None
) -> Dict[str, Any]:
    """モデルとベースラインの統計的比較（ABTestingFrameworkに追加）
    
    Note:
        既存のztb.analysis.baseline_comparison.BaselineStrategyと連携し、
        ベースライン戦略の結果を統計検定で比較
        
        3つの統計検定（t-test + Mann-Whitney U + p平均法）を統合判定
    
    Args:
        model_condition: モデル条件名（例: "sac_model"）
        baseline_condition: ベースライン条件名（例: "buy_and_hold"）
        metric_name: 比較する指標名
        alpha: 有意水準
        evaluation_function: 評価関数（Optional）
    
    Returns:
        比較結果辞書:
        - is_superior: モデルが統計的に有意に優れているか
        - is_inferior: モデルが統計的に有意に劣っているか
        - is_comparable: 統計的差異なし
        - detailed_results: 3検定の詳細結果（t-test, Mann-Whitney U, p平均法）
        - combined_decision: 統合判定結果
        - confidence: 証拠の強さ（high/medium/low/none）
    
    Note:
        Doc00 Section 5.5判定基準:
        - Buy-and-Hold: 必須超過
        - SMA Crossover: 必須超過
        - Random Action: 必須超過
        - Momentum (1h): 参考（判定外）
        
        判定ロジック:
        - 3検定全て有意 + 実用的意義 → strong_evidence (superior)
        - 2検定以上有意 + 実用的意義 → moderate_evidence (superior)
        - 1検定のみ有意 → weak_evidence (要追加検証)
        - 有意差なし → no_evidence (comparable)
    """
    # 既存のcreate_ab_test/run_ab_test活用
    test_id = self.create_ab_test(
        test_name=f"{model_condition}_vs_{baseline_condition}",
        control_params={"type": "baseline", "name": baseline_condition},
        variant_params={"type": "model", "name": model_condition},
        evaluation_function=evaluation_function or (lambda p: self._evaluate_strategy(p, metric_name))
    )
    
    # ABテスト実行（30 iterations推奨）
    result = self.run_ab_test(test_id, num_iterations=30)
    
    # 統合判定
    combined = result["combined_decision"]
    
    # 最終判定
    is_superior = combined["confidence"] in ["high", "medium"] and combined["is_meaningful"]
    is_inferior = False  # 逆方向検定が必要な場合は別途実装
    is_comparable = combined["confidence"] in ["low", "none"]
    
    return {
        "model": model_condition,
        "baseline": baseline_condition,
        "metric": metric_name,
        "is_superior": is_superior,
        "is_inferior": is_inferior,
        "is_comparable": is_comparable,
        "decision": "superior" if is_superior else "comparable_or_inferior",
        "detailed_results": result,
        "combined_decision": combined,
        "confidence": combined["confidence"],
        "note": "3検定統合判定（t-test + Mann-Whitney U + p平均法）"
    }
```

**テストケース追加**（Phase 3新規）:
1. t-test実装が正しく動作することを確認（既存実装のテスト）
2. Mann-Whitney U検定が正しく実行されることを確認
3. Cliff's Delta計算が正しいことを確認（既知データセット）
4. p平均法が正しく統合されることを確認（ztb.metrics.metrics.p_mean_method活用）
5. 3検定の統合判定が正しいことを確認
6. Holm-Bonferroni補正が正しく適用されることを確認
7. 4 seed結果が正しく集計されることを確認
8. ベースライン比較判定が正しいことを確認

**完了条件**: P1-4テスト全パス（既存2件 + 新規8件 = 10件）

---

### 2.2 TrainingReporter統合完了

**Doc00準拠**: Section 7 既存資産活用マップ - Reporter統一

**現状の問題**:
- BacktestReporter: 完全統合済み（Phase 2完了、ztb/evaluation/walk_forward/reporter.py）
- TrainingReporter: **2実装が残存**（重複リスク）
  1. `ztb/training/unified_trainer/components/reporter.py`: TrainingReporterクラス（32行）
  2. `ztb/training/unified_trainer/reporting.py`: TrainingReporterクラス（227行）
- **重複の具体的リスク**:
  - 同名クラス（TrainingReporter）が2箇所に存在
  - 指標定義のズレリスク（sharpe_ratio, win_rate等の計算式が異なる可能性）
  - メンテナンス負荷（2箇所の同期が必要）
  - Doc17指摘事項: メトリクス計算の共通化が未対応

**Phase 3目標**:
1. **メトリクス計算の共通化**（Doc17対応、最優先）
2. BacktestReporterをTraining用途にも使用可能にする（互換API追加）
3. TrainingReporter 2実装を削除
4. 単一実装への完全統一（ztb/evaluation/walk_forward/reporter.py）

#### 2.2.1 メトリクス計算統一（Doc17対応）

**既存実装の活用**: `ztb/metrics/metrics.py`を直接使用（新規ファイル不要）

**目的**: 
- Reporter、ABTestingComparator、baseline_comparisonで重複するメトリクス計算を統一
- 指標定義のズレ防止
- メンテナンス負荷削減

**既存関数の活用**:
```python
# 既存のztb/metrics/metrics.pyを直接使用
from ztb.metrics.metrics import sharpe_ratio, max_drawdown
from ztb.metrics.metrics import calculate_metrics

# BacktestReporterでの使用例
class BacktestReporter:
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        # 独自実装を削除し、統一関数を使用
        from ztb.metrics.metrics import sharpe_ratio
        return sharpe_ratio(
            returns,
            risk_free_rate=0.0,
            periods_per_year=252
        )
    
    Args:
        returns: リターン配列
        risk_free_rate: リスクフリーレート（年率）
        periods_per_year: 年間期間数（日次: 252、時間足: 252*24）
    
    Returns:
        Sharpe Ratio（年率換算）
    
    Raises:
        ValueError: returns配列にNaN/Infが含まれる場合
        ValueError: periods_per_year <= 0の場合
    
    Note:
        統一計算式: (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)
        
        使用箇所:
        - BacktestReporter._calculate_sharpe_ratio()
        - evaluator._calculate_sharpe()
        - baseline_comparison.BuyAndHoldStrategy.evaluate()
        - ABTestingComparator._compute_metrics_from_trades()
        
        エッジケース処理:
        - 空配列 → 0.0
        - std=0（全リターン同値） → 0.0
        - NaN/Inf含む → ValueError
        - サンプル数<2 → 0.0（std計算不可）
    """
    # 入力検証
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    
    if len(returns) == 0:
        return 0.0
    
    if len(returns) < 2:
        # サンプル数不足: std計算不可
        return 0.0
    
    # NaN/Inf検出
    if not np.all(np.isfinite(returns)):
        raise ValueError(
            f"returns contains NaN or Inf values. "
            f"Found {np.sum(~np.isfinite(returns))} invalid values."
        )
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0 or not np.isfinite(std_return):
        return 0.0
    
    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return
    result = sharpe * np.sqrt(periods_per_year)
    
    # 結果検証
    if not np.isfinite(result):
        return 0.0
    
    return float(result)


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Max Drawdown計算
    
    Args:
        cumulative_returns: 累積リターン配列（例: (1 + returns).cumprod()）
    
    Returns:
        Max Drawdown（負の値、例: -0.15 = 15%下落）
    
    Note:
        統一計算式: min((cumulative - running_max) / running_max)
        
        使用箇所:
        - BacktestReporter (現在未使用、Phase 3で追加)
        - evaluator._calculate_max_drawdown()
        - baseline_comparison.BuyAndHoldStrategy.evaluate()
        - ABTestingComparator._compute_metrics_from_trades()
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return float(np.min(drawdown))


def compute_win_rate(pnl_array: np.ndarray) -> float:
    """Win Rate計算
    
    Args:
        pnl_array: PnL配列
    
    Returns:
        Win Rate（0.0-1.0）
    
    Note:
        統一計算式: (PnL > 0の数) / 全取引数
        
        使用箇所:
        - BacktestReporter (現在は直接計算)
        - baseline_comparison.SMAStrategy.evaluate()
        - ABTestingComparator._compute_metrics_from_trades()
    """
    if len(pnl_array) == 0:
        return 0.0
    
    return float(np.sum(pnl_array > 0) / len(pnl_array))


def compute_profit_factor(pnl_array: np.ndarray) -> float:
    """Profit Factor計算
    
    Args:
        pnl_array: PnL配列
    
    Returns:
        Profit Factor（勝ちトレード合計 / 負けトレード合計）
    
    Note:
        統一計算式: sum(pnl[pnl > 0]) / abs(sum(pnl[pnl < 0]))
    """
    if len(pnl_array) == 0:
        return 0.0
    
    gross_profit = np.sum(pnl_array[pnl_array > 0])
    gross_loss = abs(np.sum(pnl_array[pnl_array < 0]))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def compute_trade_metrics(
    trades_df: pd.DataFrame,
    initial_capital: float = 200000.0
) -> Dict[str, float]:
    """トレードデータから全メトリクスを計算
    
    Args:
        trades_df: BacktestReporterが生成したTrade CSV
        initial_capital: 初期資本金
    
    Returns:
        全メトリクス辞書:
        - net_roi: ROI（コスト込み）
        - gross_roi: ROI（コスト抜き）
        - win_rate: 勝率
        - profit_factor: Profit Factor
        - avg_win: 平均勝ちトレード
        - avg_loss: 平均負けトレード
        - sharpe_ratio: Sharpe Ratio
        - max_drawdown: Max Drawdown
        - total_trades: 総トレード数
    
    Note:
        ABTestingComparator._compute_metrics_from_trades()の置き換え
        BacktestReporterでも同様のロジックで使用可能
    """
    if len(trades_df) == 0:
        return {
            "net_roi": 0.0,
            "gross_roi": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
        }
    
    net_pnl = trades_df["net_pnl"].values
    gross_pnl = trades_df.get("gross_pnl", net_pnl).values
    
    # 累積リターン計算
    cumulative = (1 + net_pnl / initial_capital).cumprod()
    
    return {
        "net_roi": float(net_pnl.sum() / initial_capital),
        "gross_roi": float(gross_pnl.sum() / initial_capital),
        "win_rate": compute_win_rate(net_pnl),
        "profit_factor": compute_profit_factor(net_pnl),
        "avg_win": float(net_pnl[net_pnl > 0].mean()) if np.any(net_pnl > 0) else 0.0,
        "avg_loss": float(net_pnl[net_pnl < 0].mean()) if np.any(net_pnl < 0) else 0.0,
        "sharpe_ratio": compute_sharpe_ratio(net_pnl / initial_capital),
        "max_drawdown": compute_max_drawdown(cumulative),
        "total_trades": len(trades_df),
    }
```

**マイグレーション計画**（段階的移行によるリスク低減）:

**Phase 1: 共通関数作成と検証**（1日目）
1. 既存`ztb/metrics/metrics.py`を直接使用（Reporter/Evaluator/baseline_comparisonで統一）
2. 単体テスト実装（既知データセットでの検証）
3. エッジケース検証（空配列、NaN、Inf、単一値等）

**Phase 2: 並行運用とA/Bテスト**（2日目）
1. 既存実装を残したまま、共通関数を並行呼び出し
2. 出力比較テスト実施:
   ```python
   # 既存実装
   old_sharpe = reporter._calculate_sharpe_ratio(returns)
   # 新実装
   new_sharpe = compute_sharpe_ratio(returns)
   # 数値一致確認（許容誤差: 1e-10）
   assert abs(old_sharpe - new_sharpe) < 1e-10
   ```
3. 全メトリクスで数値一致を確認（sharpe_ratio, max_drawdown, win_rate, profit_factor）

**Phase 3: 段階的置き換え**（3-4日目）
1. BacktestReporterから置き換え（影響範囲: 最小）
2. Evaluatorを置き換え（影響範囲: 中）
3. baseline_comparisonを置き換え（影響範囲: 大）
4. 各ステップで全テスト実行、リグレッション検出

**Phase 4: 既存実装削除**（5日目）
1. 旧実装を`@deprecated`マーク
2. 1週間の猶予期間（問題発見時のロールバック余地）
3. 問題なければ旧実装を完全削除

**ロールバック戦略**:
- Git tag作成: `v459-phase3-metrics-migration-start`
- 問題発生時:
  1. 共通関数の呼び出しを既存実装に戻す（1行変更のみ）
  2. 共通関数の修正と再検証
  3. 修正版で再度マイグレーション
- ロールバックトリガー:
  - テスト失敗
  - 数値不一致（> 1e-10）
  - 実行時エラー
  - パフォーマンス劣化（> 10%）

**完了条件**: 
- メトリクス共通化テスト全パス（新規5件）
- 既存実装との数値一致確認（全メトリクス、誤差 < 1e-10）
- 全Phase 0/1/2テスト維持（119/119件）
- パフォーマンス劣化なし（ベンチマーク: ± 5%以内）

#### 2.2.2 互換API実装

**BacktestReporterに追加するAPI**:
```python
# ztb/evaluation/walk_forward/reporter.py

class BacktestReporter:
    # ... 既存実装 ...
    
    def __init__(self, ...) -> None:
        # 既存初期化
        # ...
        
        # Phase 3追加: Training用統計記録
        self.episode_rewards: List[Dict[str, Any]] = []
    
    def record_episode_end(self, episode: int, total_reward: float) -> None:
        """エピソード終了時の記録（Training用API）
        
        Args:
            episode: エピソード番号
            total_reward: エピソード累積報酬
        
        Note:
            Phase 3追加: Training用互換APIとして実装
            BacktestReporterの内部状態を更新し、統計を記録
        
        Example:
            >>> reporter = BacktestReporter(...)
            >>> reporter.record_episode_end(episode=1, total_reward=15.3)
            >>> reporter.record_episode_end(episode=2, total_reward=22.1)
            >>> stats = reporter.get_episode_statistics()
            >>> print(stats['mean_reward'])  # 18.7
        """
        import time
        
        self.episode_rewards.append({
            "episode": episode,
            "total_reward": total_reward,
            "timestamp": time.time(),
        })
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """エピソード統計を取得（Training用API）
        
        Returns:
            統計辞書:
            - mean_reward: 平均報酬
            - std_reward: 報酬標準偏差
            - min_reward: 最小報酬
            - max_reward: 最大報酬
            - n_episodes: エピソード数
        
        Note:
            Phase 3追加: TrainingReporter互換APIとして実装
            学習中のエピソード統計を提供
        
        Example:
            >>> reporter = BacktestReporter(...)
            >>> # ... record_episode_end() calls ...
            >>> stats = reporter.get_episode_statistics()
            >>> assert 'mean_reward' in stats
            >>> assert stats['n_episodes'] > 0
        """
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "n_episodes": 0,
            }
        
        import numpy as np
        rewards = [r["total_reward"] for r in self.episode_rewards]
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n_episodes": len(rewards),
        }
    
    def reset_episode_statistics(self) -> None:
        """エピソード統計をリセット（Training用API）
        
        Note:
            Phase 3追加: TrainingReporter互換APIとして実装
            学習フェーズ切替時に使用
        
        Example:
            >>> reporter = BacktestReporter(...)
            >>> reporter.record_episode_end(1, 10.0)
            >>> reporter.reset_episode_statistics()
            >>> stats = reporter.get_episode_statistics()
            >>> assert stats['n_episodes'] == 0
        """
        self.episode_rewards = []
```

**互換性検証テスト**:
```python
# tests/unit/evaluation/test_reporter_training_api.py (Phase 3新規)

def test_training_api_compatibility():
    """既存TrainingReporterとBacktestReporterの互換性検証"""
    # 既存TrainingReporterと同じ入力で同じ出力を確認
    pass
    
    def record_episode_end(self, episode: int, total_reward: float) -> None:
        """エピソード終了時の記録（Training用API）
        
        Args:
            episode: エピソード番号
            total_reward: エピソード累積報酬
        
        Note:
            Phase 3追加: Training用互換APIとして実装
            BacktestReporterの内部状態を更新し、統計を記録
        """
        self.episode_rewards.append({
            "episode": episode,
            "total_reward": total_reward,
            "timestamp": time.time(),
        })
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """エピソード統計を取得（Training用API）
        
        Returns:
            統計辞書:
            - mean_reward: 平均報酬
            - std_reward: 報酬標準偏差
            - min_reward: 最小報酬
            - max_reward: 最大報酬
            - n_episodes: エピソード数
        
        Note:
            Phase 3追加: TrainingReporter互換APIとして実装
            学習中のエピソード統計を提供
        """
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "n_episodes": 0,
            }
        
        rewards = [r["total_reward"] for r in self.episode_rewards]
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "n_episodes": len(rewards),
        }
    
    def reset_episode_statistics(self) -> None:
        """エピソード統計をリセット（Training用API）
        
        Note:
            Phase 3追加: TrainingReporter互換APIとして実装
            学習フェーズ切替時に使用
        """
        self.episode_rewards = []
```

#### 2.2.2 TrainingReporter削除計画

**削除対象ファイル**:
1. `ztb/training/unified_trainer/components/reporter.py`
2. `ztb/training/unified_trainer/reporting.py`

**マイグレーション手順**:
1. TrainingReporterの使用箇所を特定（grep検索）
2. BacktestReporterの互換API実装
3. TrainingReporter呼び出しをBacktestReporterに置換
4. 既存テストの動作確認
5. TrainingReporter 2ファイル削除
6. 削除後の統合テスト実行

**テストケース追加**（Phase 3新規）:
1. record_episode_end()が正しく記録されることを確認
2. get_episode_statistics()が正しい統計を返すことを確認
3. reset_episode_statistics()が正しくリセットすることを確認
4. Trainingループでの使用が正常であることを確認

**完了条件**: Reporter統合テスト全パス（新規4件）、TrainingReporter 2ファイル削除

---

### 2.3 報酬設計の段階的検証

**Doc00準拠**: Section 3.3 報酬設計の段階化、Section 4 Phase 3実装計画

**Phase 3目標**: 純PnL → ガイダンス追加 → カリキュラム学習の段階的実装とAB比較

#### 2.3.1 報酬設計3ステージ

| Stage | 報酬関数 | 目的 | Doc00参照 |
|-------|----------|------|-----------|
| **Stage 1** | `R = PnL_net` | 純粋な収益性ベースライン | Section 3.3 |
| **Stage 2** | `R = PnL_net - 0.05 * TrendPenalty` | ガイダンス効果検証 | Section 3.3 |
| **Stage 3** | `R = PnL_net - W(t) * TrendPenalty` | カリキュラム効果検証 | Section 3.3 |

#### 2.3.2 実装詳細

**Stage 1: 純PnL（ベースライン）**
```python
# config/v459/experiments/reward_stage1_pure_pnl.yaml
reward:
  type: "pure_pnl"
  params: {}

# ztb/trading/environment/fast_intraday_env_v456.py
def _compute_reward_stage1(self) -> float:
    """Stage 1: 純PnL報酬
    
    R = (current_balance - previous_balance) / initial_balance
    
    Note:
        Doc00 Section 3.3 Stage 1準拠
        複雑な報酬設計を避け、純粋な収益性を学習
    """
    return (self.balance - self.previous_balance) / self.initial_balance
```

**Stage 2: 固定ガイダンス**
```python
# config/v459/experiments/reward_stage2_fixed_guidance.yaml
reward:
  type: "fixed_guidance"
  params:
    trend_penalty_weight: 0.05  # 固定重み

# ztb/trading/environment/fast_intraday_env_v456.py
def _compute_reward_stage2(self) -> float:
    """Stage 2: 固定ガイダンス報酬
    
    R = PnL_net - 0.05 * TrendPenalty
    
    where:
    - TrendPenalty = 1.0 if action opposes ichimoku else 0.0
    
    Note:
        Doc00 Section 3.3 Stage 2準拠
#### 2.3.1 報酬設計の段階化

**現行実装の活用**:
- `ztb/trading/rewards/fast_intraday.py`: `compute_hft_reward()`（Pure PnL）
- `ztb/trading/environment/fast_intraday_env_v456.py`: Trend Penalty統合済み（L747-770）

**Phase 3での3段階検証**:

**Stage 1: 純PnL（ベースライン）**
```yaml
# config/v459/experiments/reward_stage1_pure_pnl.yaml
reward_params:
  fee_penalty_weight: 0.0  # コストペナルティなし
  # compute_hft_reward()のPure PnL部分のみ使用

# Env側でのTrend Penaltyを無効化
use_trend_guidance: false
```

**Stage 2: 固定ガイダンス**
```yaml
# config/v459/experiments/reward_stage2_fixed_guidance.yaml
reward_params:
  fee_penalty_weight: 0.0

# Env側でのTrend Penaltyを有効化（固定重み）
use_trend_guidance: true
guidance_weight: 0.05  # 固定
guidance_decay_steps: 999999999  # 実質無効化（常に固定重み）
```

**Note**: fast_intraday_env_v456.py L747-770の既存実装を使用:
```python
# 既存実装（L747-770）
if trend_alignment < -0.1 and guidance_weight > 0:
    target_penalty_norm = 0.05 * abs(trend_alignment) * guidance_weight
    penalty_jpy = target_penalty_norm * self.reward_scale
    reward -= penalty_jpy
```

**Stage 3: Decay付きガイダンス（カリキュラム学習）**
```yaml
# config/v459/experiments/reward_stage3_curriculum.yaml
reward_params:
  fee_penalty_weight: 0.0

# Env側でのTrend Penaltyを有効化（Decay付き）
use_trend_guidance: true
guidance_weight: 0.05
guidance_decay_steps: 50000  # 50kステップで0に減衰
```

**Note**: fast_intraday_env_v456.py L747-770の既存実装を使用:
```python
# 既存実装（L747-770）
guidance_weight = max(0.0, 1.0 - (self.lifetime_steps / self.guidance_decay_steps))
if trend_alignment < -0.1 and guidance_weight > 0:
    target_penalty_norm = 0.05 * abs(trend_alignment) * guidance_weight
    penalty_jpy = target_penalty_norm * self.reward_scale
    reward -= penalty_jpy
```

**統合方針の明確化**:
- Pure PnL部分: `compute_hft_reward()`が担当（報酬関数側）
- Trend Penalty: `fast_intraday_env_v456.py`が担当（環境側）
- Stage切り替え: Config経由で`use_trend_guidance`と`guidance_decay_steps`を変更

#### 2.3.2 ABテスト比較
    """Stage 3: Decay付きガイダンス報酬（カリキュラム学習）
    
    R = PnL_net - W(t) * TrendPenalty
    
    where:
    - W(t) = max(0, 1 - lifetime_steps / 50000)
    
    Note:
        Doc00 Section 3.3 Stage 3準拠
        学習初期にガイダンスを提供し、50kステップで徐々に無効化
        最終的に純PnL最適化へ収束
    """
    pnl = (self.balance - self.previous_balance) / self.initial_balance
    trend_penalty = self._compute_trend_penalty()
    
    # Decay重み計算
    weight = max(0.0, 1.0 - self.lifetime_steps / 50000.0)
    
    return pnl - weight * 0.05 * trend_penalty
```

#### 2.3.3 AB比較計画

**実験設計**:
- 条件数: 3（Stage 1, Stage 2, Stage 3）
- シード数: 4（統計検定要件）
- 評価期間: Val + Test（Walk-Forward）
- 比較指標: net_roi, sharpe_ratio, win_rate, max_drawdown

**統計検定**:
- ペアワイズ比較: Stage 1 vs 2、Stage 2 vs 3、Stage 1 vs 3
- 多重比較補正: Holm-Bonferroni法（α=0.05）
- 効果量: Cliff's Delta（|d| > 0.33で実用的意義）

**成功基準**:
- Stage 2 vs Stage 1: 統計的有意差あり（ガイダンス効果検証）
- Stage 3 vs Stage 1: 統計的有意差あり（カリキュラム効果検証）
- Stage 3 vs Stage 2: 差異の有無で判断（カリキュラム優位性検証）

**テストケース追加**（Phase 3新規）:
1. Stage 1報酬計算が正しいことを確認
2. Stage 2報酬計算が正しいことを確認
3. Stage 3報酬計算が正しいことを確認（decay含む）
4. トレンドペナルティが正しく計算されることを確認
5. 3ステージのAB比較が実行できることを確認

**完了条件**: 報酬設計テスト全パス（新規5件）、AB比較データ取得

---

### 2.4 リスク管理統合

**Doc00準拠**: Section 7 既存資産活用マップ - Circuit Breaker、Risk Allocator

**Phase 3目標**: 既存のリスク管理コンポーネント（Circuit Breaker、Virtual Portfolio Manager）をWalk-Forward評価に統合

#### 2.4.1 Circuit Breaker統合

**既存資産**: `ztb/trading/production/circuit_breaker.py`

**既存CircuitBreaker API**:
- `CircuitBreaker(config: CircuitBreakerConfig)`: コンストラクタ
- `call_sync(func, *args, **kwargs)`: 保護された同期関数呼び出し
- `CircuitBreakerConfig`: failure_threshold, recovery_timeout_seconds等の設定

**Phase 3統合方針**:
既存のCircuitBreakerは非同期マイクロサービス向け設計のため、
**直接統合せず、同等機能をEnv内で実装**します。

```python
# ztb/trading/environment/fast_intraday_env_v456.py

class FastIntradayEnvV456:
    def __init__(self, config: Dict[str, Any], ...):
        # ... 既存初期化 ...
        
        # Circuit Breaker相当の保護ロジック
        self.use_circuit_breaker = config.get("use_circuit_breaker", False)
        if self.use_circuit_breaker:
            cb_config = config.get("circuit_breaker_config", {})
            self.max_daily_loss = cb_config.get("max_daily_loss", 10000)  # JPY
            self.max_consecutive_losses = cb_config.get("max_consecutive_losses", 5)
            self.cooldown_steps = cb_config.get("cooldown_steps", 100)
            
            # 状態変数
            self.daily_pnl_tracking = 0.0
            self.consecutive_losses = 0
            self.is_halted = False
            self.halt_cooldown_remaining = 0
        else:
            self.use_circuit_breaker = False
    
    def evaluate_step(self, obs: np.ndarray, action: np.ndarray) -> tuple:
        """Circuit Breaker統合評価ステップ
        
        Note:
            既存のztb.trading.production.circuit_breaker.CircuitBreakerとは
            設計が異なりますが、同等の保護機能を提供します。
        
        Returns:
            (obs, reward, done, truncated, info)
        """
        # Circuit Breaker判定
        if self.use_circuit_breaker:
            if self.is_halted:
                self.halt_cooldown_remaining -= 1
                if self.halt_cooldown_remaining <= 0:
                    self.is_halted = False
                else:
                    # クールダウン中は強制終了
                    info = {
                        "circuit_breaker": "halted",
                        "cooldown_remaining": self.halt_cooldown_remaining
                    }
                    return obs, 0.0, True, True, info
        
        # 通常のステップ実行
        obs, reward, done, truncated, info = self.step(action)
        
        # Circuit Breaker状態更新
        if self.use_circuit_breaker and not done:
            self.daily_pnl_tracking = self.accounting.net_pnl
            
            # 損失判定
            if self.daily_pnl_tracking < -self.max_daily_loss:
                self.is_halted = True
                self.halt_cooldown_remaining = self.cooldown_steps
                info["circuit_breaker"] = "triggered_daily_loss"
                done = True
                truncated = True
            
            # 連続損失判定
            if reward < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.max_consecutive_losses:
                    self.is_halted = True
                    self.halt_cooldown_remaining = self.cooldown_steps
                    info["circuit_breaker"] = "triggered_consecutive_losses"
                    done = True
                    truncated = True
            else:
                self.consecutive_losses = 0
        
        return obs, reward, done, truncated, info
```
            self.last_trade_day: Optional[pd.Timestamp] = None
    
    def evaluate_step(self, env, model, obs, step: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """1ステップ評価実行（Circuit Breaker統合版）"""
        # ... 既存評価ロジック ...
        
        action, _states = model.predict(obs, deterministic=True)
        
        # Circuit Breaker判定（有効時のみ）
        if self.enable_circuit_breaker:
            # 日次PnL追跡
            current_day = pd.Timestamp.now().date()
            if self.last_trade_day != current_day:
                self.daily_pnl = 0.0
                self.last_trade_day = current_day
            
            # Circuit Breaker発動チェック
            if self.circuit_breaker.should_halt(
                daily_loss_pct=abs(self.daily_pnl) / env.initial_balance,
                consecutive_losses=self.consecutive_losses
            ):
                # 取引停止: ポジションをクローズしてホールド
                action = np.array([0.0])  # Hold action
                logger.warning(
                    f"Circuit Breaker発動: daily_pnl={self.daily_pnl:.2f}, "
                    f"consecutive_losses={self.consecutive_losses}"
                )
                self.circuit_breaker.trigger_halt()
        
        obs, reward, done, info = env.step(action)
        
        # Circuit Breaker統計更新
        if self.enable_circuit_breaker:
            trade_pnl = info.get('trade_pnl', 0.0)
            self.daily_pnl += trade_pnl
            
            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        return obs, reward, done, info
        # ... 既存初期化 ...
        
        self.enable_circuit_breaker = enable_circuit_breaker
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                max_daily_loss=0.03,  # 3%日次損失上限
                max_consecutive_losses=5,  # 5連敗で発動
                position_scale_on_loss=0.5,  # 50%縮小
            )
    
    def evaluate_step(self, ...):
        # ... 既存評価ロジック ...
        
        # Circuit Breaker判定
        if self.enable_circuit_breaker:
            if self.circuit_breaker.should_halt(current_loss, consecutive_losses):
                # 取引停止
                action = 0.0  # Hold
                self.circuit_breaker.trigger_halt()
```

**テストケース追加**（Phase 3新規）:
1. Circuit Breaker無効時の動作確認
2. Circuit Breaker有効時、日次損失上限で停止することを確認
3. Circuit Breaker有効時、5連敗で停止することを確認
4. Circuit Breaker発動後のポジションサイズ縮小確認

**完了条件**: Circuit Breaker統合テスト全パス（新規4件）

#### 2.4.2 Virtual Portfolio Manager統合

**既存資産**: `ztb/trading/production/risk_based_allocator.py`、`virtual_portfolio_manager.py`

**統合内容**: Walk-Forward評価時に、仮想ポートフォリオでリスク配分をシミュレート

**完了条件**: Virtual Portfolio Manager統合テスト全パス（新規2件）

---

### 2.5 MTF因果性検証強化

**Doc00準拠**: Doc02 因果性設計

**Phase 3目標**: MTF特徴量の因果性保証を強化し、future leakageを完全排除

**実装内容**:
- `check_causality.py`スクリプト拡張
- MTF特徴量の厳密な検証（5m→15m→30m→1hの順序保証）
- 警告レベルエラーの導入

**完了条件**: MTF因果性検証テスト全パス（新規3件）

---

### 2.6 Scaler fit境界の厳密化

**Phase 3目標**: Scaler fit範囲がtrain期間に厳密に限定されることを保証（警告→エラー化）

**実装内容**:
```python
# ztb/processing/online_scaler.py

def fit(self, data: np.ndarray, period: str) -> None:
    """Scalerをfitする
    
    Args:
        data: fit対象データ
        period: 期間識別子（"train" / "val" / "test"）
    
    Raises:
        ValueError: period != "train"の場合（Phase 3厳密化）
    
    Note:
        Phase 3: 警告からエラーへ変更
        Val/Test期間でのfitは因果性違反のため禁止
    """
    if period != "train":
        raise ValueError(
            f"Scaler fit is only allowed on 'train' period, got '{period}'. "
            "Fitting on 'val' or 'test' data causes future leakage."
        )
    
    # ... fit処理 ...
```

**完了条件**: Scaler境界テスト全パス（新規2件）

---

## 3. Phase 3実装計画

### 3.1 実装順序（依存関係考慮）

| 順序 | タスク | 依存 | 工数 | 完了条件 |
|------|--------|------|------|----------|
| 1 | メトリクス計算統一 | なし | 1.0日 | ztb/metrics/metrics.pyをReporter/Evaluatorで直接使用、数値一致確認 |
| 2 | P1-4統計検定実装 | 1 | 1.5日 | Mann-Whitney U、Cliff's Delta、Holm-Bonferroni実装 |
| 3 | P1-4ベースライン比較 | 1, 2 | 0.5日 | ベースライン比較API実装、既存baseline_comparisonと連携 |
| 4 | P1-4 4 seed対応 | 2 | 0.5日 | 4 seed動作確認 |
| 5 | TrainingReporter互換API | 1 | 1.0日 | 互換API実装、動作確認 |
| 6 | TrainingReporter削除 | 5 | 0.5日 | マイグレーション、2ファイル削除 |
| 7 | 報酬Stage 1実装 | なし | 0.5日 | 純PnL報酬実装 |
| 8 | 報酬Stage 2実装 | 7 | 0.5日 | 固定ガイダンス実装 |
| 9 | 報酬Stage 3実装 | 8 | 0.5日 | Decay付きガイダンス実装 |
| 10 | 報酬AB比較実験 | 2-4, 7-9 | 1.0日 | 3ステージAB比較実行 |
| 11 | Circuit Breaker統合 | なし | 0.5日 | Walk-Forward統合 |
| 12 | MTF因果性強化 | なし | 0.5日 | 因果性検証強化 |
| 13 | Scaler境界厳密化 | なし | 0.5日 | エラー化実装 |
| 14 | 統合テスト作成 | 1-13 | 1.0日 | Phase 3テスト全パス |
| 15 | Phase 3完了報告 | 14 | 0.5日 | Doc25作成 |

**合計工数**: 10.5日

### 3.2 テスト計画

**Phase 3新規テスト数**: 38件（見込み）

| カテゴリ | テスト数 | 内容 |
|----------|----------|------|
| メトリクス共通化 | 5件 | sharpe_ratio、max_drawdown、win_rate、profit_factor、compute_trade_metrics |
| P1-4統計検定 | 6件 | Mann-Whitney U、Cliff's Delta、Holm-Bonferroni、ベースライン比較 |
| TrainingReporter | 4件 | 互換API、統計取得、リセット、Trainingループ |
| 報酬設計 | 5件 | Stage 1/2/3、トレンドペナルティ、AB比較 |
| リスク管理 | 6件 | Circuit Breaker統合、Virtual Portfolio Manager統合 |
| MTF因果性 | 3件 | 因果性検証強化 |
| Scaler境界 | 2件 | エラー化、境界チェック |
| 統合テスト | 7件 | エンドツーエンド統合テスト |

**累積テスト数**: 119（Phase 0-2） + 38（Phase 3） = 157件

---

## 4. 完了条件

### 4.1 技術検証（Phase 3完了条件）

- [ ] **メトリクス計算統一完了**（ztb/metrics/metrics.pyをReporter/Evaluatorで直接使用、数値一致確認）
- [ ] P1-4統計検定実装完了（Mann-Whitney U、Cliff's Delta、Holm-Bonferroni）
- [ ] P1-4 4 seed対応完了（統計検定要件達成）
- [ ] P1-4既存baseline_comparison.py連携完了
- [ ] TrainingReporter統合完了（2ファイル削除、単一実装統一）
- [ ] 報酬設計3ステージ実装完了（Stage 1/2/3）
- [ ] 報酬AB比較データ取得完了（3条件 × 4 seed）
- [ ] Circuit Breaker統合完了（Walk-Forward評価連携）
- [ ] MTF因果性検証強化完了
- [ ] Scaler境界厳密化完了（警告→エラー化）
- [ ] Phase 3テスト全パス（新規38件）
- [ ] 累積テスト維持（157/157件、100%）

### 4.2 収益性検証（Doc00準拠）

**Doc00 Section 5.2収益性検証基準**:

| 指標 | 最低基準 | 目標基準 | 測定条件 |
|------|----------|----------|----------|
| **Net ROI** | > 5% | > 15% | 年率換算、コスト込み |
| **Profit Factor** | > 1.20 | > 1.50 | 手数料・スリッページ後 |
| **Sharpe Ratio** | > 1.0 | > 1.5 | 日次リターン、年率換算 |
| **Max Drawdown** | < 15% | < 10% | 高値からの最大下落 |
| **Win Rate** | > 35% | > 45% | 手数料込み勝敗 |
| **期待値/取引** | > ¥500 | > ¥1,000 | コスト控除後 |

**Phase 3評価**: 報酬設計3ステージのうち、最低1ステージが最低基準をクリアすること

### 4.3 ベースライン比較（Doc00準拠）

**Doc00 Section 5.5ベースライン比較基準**:

| 比較対象 | 条件 | 判定 | 統計検定 |
|----------|------|------|----------|
| Buy-and-Hold | 同期間、同コスト | **必須超過** | 要 |
| SMA Crossover | 20/50期間、同条件 | **必須超過** | 要 |
| Random Action | 同頻度、同コスト | **必須超過** | 要 |
| Momentum (1h) | 1時間リターン追従 | **参考**（判定外） | 任意 |

**Phase 3評価**: 最良ステージがBuy-and-Hold、SMA Crossover、Random Actionを統計的に有意に超過すること（Mann-Whitney U、Cliff's Delta > 0.33）

---

## 5. Doc00との整合性チェックリスト

- [x] Section 3.3 報酬設計の段階化（Stage 1/2/3実装）
- [x] Section 4 Phase 3実装計画（報酬設計段階検証）
- [x] Section 5.2 収益性検証基準（最低基準・目標基準定義）
- [x] Section 5.5 ベースライン比較基準（BH/SMA/Random/Momentum）
- [x] Section 5.6 統計検定仕様（Mann-Whitney U、Cliff's Delta、Holm-Bonferroni、サンプル数要件）
- [x] Section 7 既存資産活用マップ（Reporter統一、Circuit Breaker、Risk Allocator）

---

## 6. リスク評価とレビュー指摘対応

### 6.1 Doc19/Doc21指摘事項の再発防止

**Phase 2レビューで指摘された問題点の予防策**:

| Doc19/21指摘 | Phase 3での予防策 |
|--------------|-------------------|
| Recorder二重記録 | 統合テストで記録回数を明示的に検証 |
| PnL計算不整合 | メトリクス計算を統一（ztb/metrics/metrics.pyを直接使用） |
| prev_entry_price未使用 | API使用箇所を grep検索で全箇所確認 |
| 反転PnL配賦の誤り | 反転ケースのエンドツーエンドテスト追加 |
| Add/Reduce時のentry_price更新 | 全ポジション変更パターンのテスト追加 |
| メトリクス計算の重複 | 共通ユーティリティ化（Doc17対応） |
| close_reasonデータフロー | env→info→reporter の一方向性をテストで保証 |

### 6.2 Phase 3固有リスク

#### 高リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| 統計検定実装の誤り | 誤った判定 | 既知データセットでの検証、scipy公式ドキュメント準拠 |
| TrainingReporter削除による影響 | Training失敗 | 段階的マイグレーション、互換APIの完全テスト |
| 報酬設計の学習不安定化 | 性能低下 | Stage 1ベースライン確保、段階的検証 |
| メトリクス共通化時のバグ混入 | 全評価指標が不正確化 | 既存実装との出力比較テスト、数値一致検証 |

### 中リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| 4 seed実験の時間超過 | スケジュール遅延 | 並列実行環境準備 |
| Circuit Breaker誤発動 | 取引機会損失 | 閾値の慎重な設定 |

---

## 7. 依存関係とバージョン要件

**Python**: 3.9以上（型ヒント、dataclassesサポート）

**必須ライブラリ**:
```
scipy>=1.9.0          # mannwhitneyu統計検定
numpy>=1.21.0         # 数値計算、配列操作
pandas>=1.3.0         # データフレーム、CSV読み込み
```

**既存依存関係**:
```
stable-baselines3>=2.0.0  # SAC実装
gym>=0.21.0               # 環境API
```

**バージョン互換性**:
- scipy 1.9.0以降: mannwhitneyu()のalternative引数サポート
- numpy 1.21.0以降: isfinite()の安定性向上
- pandas 1.3.0以降: DataFrame.to_dict(orient="records")サポート

**確認方法**:
```python
import scipy
import numpy as np
import pandas as pd

print(f"scipy: {scipy.__version__}")  # >= 1.9.0
print(f"numpy: {np.__version__}")    # >= 1.21.0
print(f"pandas: {pd.__version__}")   # >= 1.3.0
```

**Phase 3新規依存**:
- なし（既存ライブラリのみ使用）

---

## 8. 次のステップ

### Phase 3完了後

1. **Phase 4準備**: Paper Trading統合（Doc00 Section 4 Phase 5）
2. **Phase 5準備**: Go/No-Go判定（Doc00 Section 4 Phase 6）
3. **Doc25作成**: Phase 3完了報告

---

**Status**: 📋 Planning  
**Author**: GitHub Copilot  
**Date**: 2026-01-25  
**Version**: 1.1
