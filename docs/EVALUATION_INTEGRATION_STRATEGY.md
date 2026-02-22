# 統合評価フレームワーク戦略

## 概要

`unified_evaluation` と `walk_forward` を統合し、高収益性トレードシステム構築のための包括的評価フレームワークを実現する。

## 現状分析

### 既存モジュール
- **unified_evaluation**: 包括的な静的評価フレームワーク（Sharpe, Sortino, Calmar等）
- **walk_forward**: 時系列ウィンドウごとの動的評価（訓練/検証/テスト分割）

### 問題点
1. 2つの評価体系が独立している
2. walk_forward結果を包括的評価に反映できない
3. 過学習検出が体系的でない
4. ウィンドウ間の性能比較が不十分

## 統合設計

### レベル 1: データ型統合（完了 ✅）
```python
ComprehensiveEvaluation
├── model_name: str
├── evaluation_type: str ("backtest" | "walk_forward" | "cross_validation")
├── timestamp: str (ISO format)
├── results: Dict[str, EvaluationResult]  # String keys for metrics
│   └── metric names: "sharpe_ratio", "max_drawdown", etc.
├── risk_metrics: Dict[str, Any]
├── performance_metrics: Dict[str, Any]
├── market_regime_analysis: Dict[str, Any]
└── robustness_tests: Dict[str, Any]
```

**Status**: ✅ types.py, unified_evaluation.py, ztb/evaluation/unified_evaluation.py 全て同期完了

### レベル 2: walk_forward統合（実装中）

#### 2.1 WalkForwardEvaluationResult 拡張
```python
@dataclass
class WalkForwardEvaluationResult(ComprehensiveEvaluation):
    """Walk-Forward分析の統合評価結果"""

    window_count: int  # ウィンドウ数
    windows_results: List[WindowPerformance]  # 各ウィンドウの詳細結果

    # 統計的サマリー
    avg_in_sample_return: float
    avg_out_of_sample_return: float
    in_sample_std: float
    out_of_sample_std: float

    # 過学習検出
    overfitting_indicator: float  # out_of_sample / in_sample ratio
    overfitting_severity: str  # "none" | "mild" | "moderate" | "severe"

    # ウィンドウ横断的指標
    consistency_score: float  # ウィンドウ間の性能の一貫性 (0-1)
    robustness_score: float  # テストセット性能の堅牢性 (0-1)
    stability_index: float  # リターンの安定性指数
```

#### 2.2 統合評価メソッド
```python
class WalkForwardUnifiedEvaluator(UnifiedEvaluator):
    """Walk-Forward統合評価器"""

    def evaluate_walk_forward(
        self,
        windows: List[TimeSeriesWindow],
        df: pd.DataFrame,
        model_path: str,
    ) -> WalkForwardEvaluationResult:
        """全ウィンドウを評価し、統合評価結果を生成"""

        # 1. 各ウィンドウで訓練・評価
        windows_results = []
        for window in windows:
            result = self.evaluate_window(window, df, model_path)
            windows_results.append(result)

        # 2. ウィンドウ間の統計分析
        stats = self._analyze_cross_window_stats(windows_results)

        # 3. 過学習検出
        overfitting = self._detect_overfitting(windows_results)

        # 4. 統合評価結果を生成
        return WalkForwardEvaluationResult(
            model_name=Path(model_path).stem,
            evaluation_type="walk_forward",
            timestamp=datetime.now().isoformat(),
            results=self._aggregate_metrics(windows_results),
            risk_metrics=self._calculate_cross_window_risk(windows_results),
            windows_results=windows_results,
            avg_in_sample_return=stats['avg_in_sample'],
            avg_out_of_sample_return=stats['avg_out_of_sample'],
            in_sample_std=stats['in_sample_std'],
            out_of_sample_std=stats['out_of_sample_std'],
            overfitting_indicator=overfitting['indicator'],
            overfitting_severity=overfitting['severity'],
            consistency_score=stats['consistency'],
            robustness_score=stats['robustness'],
            stability_index=stats['stability'],
        )
```

#### 2.3 過学習検出ロジック
```python
def _detect_overfitting(self, windows: List[WindowPerformance]) -> Dict:
    """過学習指標を計算"""

    in_sample_returns = [w.val_roi for w in windows]
    out_of_sample_returns = [w.test_roi for w in windows]

    # 指標: Out-of-Sample / In-Sample比率
    # < 0.8: 優秀（テストで性能維持）
    # 0.8-1.0: 良好
    # 1.0-1.2: 若干の過学習
    # > 1.2: 重度の過学習

    ratio = np.mean(out_of_sample_returns) / np.mean(in_sample_returns)

    if ratio > 1.2:
        severity = "severe"
    elif ratio > 1.0:
        severity = "moderate"
    elif ratio > 0.8:
        severity = "mild"
    else:
        severity = "none"

    return {
        "indicator": ratio,
        "severity": severity,
        "threshold": 0.8,  # 推奨閾値
    }
```

### レベル 3: 統合分析インターフェース（計画中）

```python
class UnifiedEvaluationReport:
    """統合評価レポート生成"""

    def generate_report(
        self,
        backtest_eval: ComprehensiveEvaluation,
        walk_forward_eval: WalkForwardEvaluationResult,
        comparison_benchmark: Optional[ComprehensiveEvaluation] = None,
    ) -> str:
        """マルチレベル評価レポートを生成"""

        # 1. バックテスト評価
        # 2. Walk-Forward評価
        # 3. 過学習検出結果
        # 4. リスク分析
        # 5. 推奨事項
```

## 実装ロードマップ

### フェーズ1: 基礎統合（現在）
- [x] 型定義の統一化（types.py）
- [x] unified_evaluation と shim の同期
- [ ] walk_forward結果をComprehensiveEvaluationに統合
- [ ] 過学習検出ロジックの実装

### フェーズ2: 拡張機能
- [ ] ウィンドウ横断的統計分析
- [ ] 一貫性スコア計算
- [ ] ロバストネス指標の算出

### フェーズ3: レポート生成
- [ ] 統合分析レポート
- [ ] ビジュアライゼーション
- [ ] 推奨事項自動生成

## 高収益性への寄与

1. **過学習の可視化**: テストセット性能の劣化を数値化
2. **安定性評価**: ウィンドウ間の一貫性を定量化
3. **リスク調整**: Sharpe比等で報酬/リスクを最適化
4. **動的調整**: 不安定な期間の検出と対応

## ベストプラクティス

### メトリクス計算
- すべてのメトリクスを string キーで保存
- Enum は内部的にのみ使用し、保存時は value に変換
- JSON/YAML互換性を維持

### 型安全性
- ComprehensiveEvaluationClass は Any型を受け入れるが、メソッドで型チェック
- walk_forward の WindowPerformance は継承してなく、aggregation で変換

### テスト戦略
- ウィンドウ単位のテスト
- 統計分析の精度確認
- 過学習検出の閾値検証

## 関連ファイル

- `ztb/analysis/common/types.py`: 型定義
- `ztb/analysis/evaluation/unified_evaluation.py`: 実装
- `ztb/evaluation/unified_evaluation.py`: compatibility shim
- `ztb/evaluation/walk_forward/types.py`: walk-forward型
- `ztb/evaluation/walk_forward/evaluator.py`: walk-forward評価器
- `ztb/evaluation/walk_forward/__init__.py`: 統合インターフェース

---

**目標**: SAC学習の高収益性を統計的に検証可能なシステムの構築
