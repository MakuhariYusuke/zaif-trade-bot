# Extended Evaluation Framework

このドキュメントでは、取引戦略の包括的な評価を行うための拡張評価モジュールを説明します。これらのモジュールは、基本的なバックテスト結果を超えて、戦略の堅牢性、安定性、リスク特性を多角的に分析します。

## 概要

拡張評価フレームワークは以下の7つの評価モジュールで構成されています：

1. **Regime Evaluation** - 市場レジーム別のパフォーマンス分析
2. **Walkforward Analysis** - 時系列クロスバリデーション
3. **Cost Sensitivity Analysis** - 取引コスト感度分析
4. **Drawdown Recovery Analysis** - ドローダウン回復分析
5. **Feature Importance Analysis** - 特徴量重要度分析
6. **Bootstrap Confidence Intervals** - 統計的信頼区間推定
7. **Stress Testing** - 極端な市場条件での耐性テスト

## 各モジュールの詳細

### 1. Regime Evaluation (`regime_evaluation.py`)

市場の状態（トレンド、レンジ、高ボラティリティ、低ボラティリティ）別に戦略のパフォーマンスを分析します。

#### 主な機能
- 市場レジームの自動分類
- レジーム別パフォーマンス指標計算
- レジーム遷移分析
- 適応性評価

#### 使用例
```bash
python regime_evaluation.py --returns-path data/returns.csv --output-dir results/regime
```

#### 出力
- レジーム別Sharpe Ratio、勝率、最大ドローダウン
- レジーム遷移確率行列
- パフォーマンス比較チャート

### 2. Walkforward Analysis (`walkforward_analysis.py`)

時系列データを尊重したクロスバリデーションを行い、戦略の安定性を評価します。

#### 主な機能
- ローリングウィンドウ分析
- パフォーマンスの時系列安定性評価
- 過学習検出
- アンカーリングバイアス回避

#### 使用例
```bash
python walkforward_analysis.py --returns-path data/returns.csv --window-size 252 --step-size 21 --output-dir results/walkforward
```

#### 出力
- 各ウィンドウのパフォーマンス指標
- 安定性指標（標準偏差、変動係数）
- 時系列パフォーマンスチャート

### 3. Cost Sensitivity Analysis (`cost_sensitivity.py`)

取引コスト（スプレッド、手数料、市場影響）が戦略パフォーマンスに与える影響を分析します。

#### 主な機能
- コスト感度曲線の生成
- ブレークイーブンポイント計算
- コスト最適化推奨
- スケーラビリティ評価

#### 使用例
```bash
python cost_sensitivity.py --returns-path data/returns.csv --trades-path data/trades.csv --cost-range 0.0001 0.01 --output-dir results/cost
```

#### 出力
- コスト vs パフォーマンス曲線
- 感度指標（1bpsあたりの影響）
- 最適コスト範囲推奨

### 4. Drawdown Recovery Analysis (`drawdown_recovery.py`)

ドローダウンからの回復パターンとリスク特性を分析します。

#### 主な機能
- ドローダウン期間の特定
- 回復時間と効率の計算
- リスク指標（Pain Index、Recovery Ratio）
- 回復パターン分析

#### 使用例
```bash
python drawdown_recovery.py --returns-path data/returns.csv --min-drawdown 0.05 --output-dir results/drawdown
```

#### 出力
- ドローダウン統計（頻度、深度、回復時間）
- 回復効率指標
- リスクメトリクスサマリー

### 5. Feature Importance Analysis (`feature_importance.py`)

機械学習モデルの特徴量重要度をSHAPとPermutation Importanceで分析します。

#### 主な機能
- SHAP値計算
- Permutation Importance分析
- 特徴量安定性評価
- モデル解釈可能性向上

#### 使用例
```bash
python feature_importance.py --model-path models/trained_model.pkl --data-path data/features.csv --target-column target --output-dir results/features
```

#### 出力
- SHAP重要度ランキング
- Permutation重要度比較
- 特徴量安定性指標

### 6. Bootstrap Confidence Intervals (`bootstrap_confidence.py`)

ブートストラップ法により、パフォーマンス指標の統計的信頼区間を推定します。

#### 主な機能
- パフォーマンス指標の信頼区間計算
- BCa法によるバイアス補正
- 統計的有意性評価
- 不確実性定量化

#### 使用例
```bash
python bootstrap_confidence.py --returns-path data/returns.csv --trades-path data/trades.csv --n-bootstrap 1000 --output-dir results/bootstrap
```

#### 出力
- 各指標の信頼区間（パーセンタイル法/BCa法）
- 標準誤差とバイアス推定
- ブートストラップ分布ヒストグラム

### 7. Stress Testing (`stress_test.py`)

極端な市場条件での戦略耐性を評価します。

#### 主な機能
- 複数ストレスシナリオ実行
- 生存確率計算
- 極端イベント耐性評価
- リスク限界点特定

#### デフォルトシナリオ
- **Market Crash**: 突然の市場暴落と回復
- **Volatility Spike**: ボラティリティ急上昇
- **Liquidity Crisis**: 流動性枯渇
- **Black Swan**: 5σ以上の極端イベント
- **Correlation Breakdown**: 相関崩壊

#### 使用例
```bash
python stress_test.py --returns-path data/returns.csv --output-dir results/stress
```

#### 出力
- 各シナリオでのパフォーマンス劣化
- 生存確率分析
- ストレス耐性ランキング

## 統合評価ワークフロー

### 基本ワークフロー
```bash
# 1. 市場適応性評価
python regime_evaluation.py --returns-path data/returns.csv --output-dir results/

# 2. 安定性検証
python walkforward_analysis.py --returns-path data/returns.csv --output-dir results/

# 3. コスト影響評価
python cost_sensitivity.py --returns-path data/returns.csv --trades-path data/trades.csv --output-dir results/

# 4. 回復力分析
python drawdown_recovery.py --returns-path data/returns.csv --output-dir results/

# 5. モデル解釈（ML戦略の場合）
python feature_importance.py --model-path models/model.pkl --data-path data/features.csv --output-dir results/

# 6. 統計的信頼性
python bootstrap_confidence.py --returns-path data/returns.csv --output-dir results/

# 7. 極端条件耐性
python stress_test.py --returns-path data/returns.csv --output-dir results/
```

### 自動化スクリプト
```bash
#!/bin/bash
# run_full_evaluation.sh

echo "Starting comprehensive strategy evaluation..."

# 並列実行可能な評価
python regime_evaluation.py --returns-path $1 --output-dir results/ &
python walkforward_analysis.py --returns-path $1 --output-dir results/ &
python drawdown_recovery.py --returns-path $1 --output-dir results/ &
wait

# 順次実行が必要な評価
python cost_sensitivity.py --returns-path $1 --trades-path $2 --output-dir results/
python bootstrap_confidence.py --returns-path $1 --output-dir results/
python stress_test.py --returns-path $1 --output-dir results/

if [ -f "models/model.pkl" ]; then
    python feature_importance.py --model-path models/model.pkl --data-path $3 --output-dir results/
fi

echo "Evaluation complete. Check results/ directory for detailed reports."
```

## 評価指標の解釈

### パフォーマンス指標
- **Sharpe Ratio**: リスク調整後リターン。>1.0が良好
- **Win Rate**: 勝ちトレードの割合。>50%が最低基準
- **Profit Factor**: 総利益/総損失。>1.2が良好
- **Maximum Drawdown**: 最大損失割合。<20%が理想

### リスク指標
- **VaR (95%)**: 95%信頼区間での最大損失
- **CVaR (95%)**: VaRを超える損失の期待値
- **Tail Ratio**: 分布の裾の比率。正規分布では1.0

### 安定性指標
- **Regime Stability**: 異なる市場条件での一貫性
- **Walkforward Consistency**: 時系列安定性
- **Bootstrap CI Width**: 推定の不確実性

## カスタム評価モジュールの作成

### 基本構造
```python
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class CustomAnalysisResult:
    """カスタム分析結果"""
    metric_name: str
    value: float
    confidence_interval: tuple
    additional_data: Dict[str, Any]

class CustomAnalyzer:
    """カスタム分析クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze(self, returns: pd.Series) -> CustomAnalysisResult:
        # 分析ロジックを実装
        pass

    def plot_results(self, result: CustomAnalysisResult):
        # 可視化を実装
        pass

    def export_results(self, result: CustomAnalysisResult, path: str):
        # エクスポートを実装
        pass
```

### 統合方法
1. 分析クラスを実装
2. コマンドラインインターフェースを追加
3. ドキュメントを更新
4. ワークフローに組み込み

## 結果の統合レポート

### 総合評価ダッシュボード
```python
# evaluation_dashboard.py
import json
import pandas as pd
from pathlib import Path

class EvaluationDashboard:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

    def load_all_results(self):
        """全評価結果を読み込み"""
        results = {}
        for json_file in self.results_dir.glob("*_results.json"):
            module_name = json_file.stem.replace("_results", "")
            with open(json_file, 'r') as f:
                results[module_name] = json.load(f)
        return results

    def generate_summary_report(self, results: dict) -> str:
        """総合評価レポート生成"""
        report = ["# Strategy Evaluation Summary Report\n"]

        # 各モジュールのサマリー
        for module, data in results.items():
            report.append(f"## {module.replace('_', ' ').title()}\n")
            # 主要指標の抽出とレポート
            report.append(self._extract_key_metrics(module, data))
            report.append("")

        # 総合評価
        report.append("## Overall Assessment\n")
        report.append(self._generate_overall_assessment(results))

        return "\n".join(report)
```

## トラブルシューティング

### よくある問題
1. **メモリ不足**: 大規模データセットではチャンク処理を検討
2. **計算時間**: ブートストラップサンプル数を減らすか並列処理を検討
3. **依存関係**: SHAPなどのオプションライブラリはpip installでインストール
4. **データ形式**: CSV/JSON形式の統一を確保

### パフォーマンス最適化
- 計算集約的な分析は並列実行
- 大規模データはサンプリング
- 結果のキャッシュを活用
- メモリ使用量の監視

## 拡張性

このフレームワークは以下の点で拡張可能です：

- **新しい評価指標**: ドメイン固有の指標追加
- **カスタムストレスシナリオ**: 業界特化のリスク要因
- **機械学習統合**: 自動化された戦略最適化
- **リアルタイム監視**: ライブ取引での継続評価
- **マルチアセット対応**: 複数資産クラスの統合分析

各モジュールの詳細なAPIドキュメントは、各Pythonファイルのdocstringを参照してください。