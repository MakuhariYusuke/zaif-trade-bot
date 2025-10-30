# 統合分析・バックテストシステム

v441開発前に実装された、トレーニング・バックテスト・分析・レポート生成を統合管理するシステムです。

## 概要

このシステムは、以下の機能を統一的に提供します：

- **統合分析ツール**: モデルの包括的な分析を実行
- **統一レポート生成**: JSON/HTML/Markdown形式のレポート
- **設定管理**: JSONベースの柔軟な設定
- **モジュール化**: 再利用可能な分析コンポーネント

## ディレクトリ構造

```
ztb/
├── analysis/                    # 統合分析システム
│   ├── core/                   # コア分析機能
│   │   ├── analyzer.py         # メイン分析器
│   │   ├── metrics.py          # 分析指標計算
│   │   └── validators.py       # 検証機能
│   ├── performance/            # パフォーマンス分析
│   ├── reporting/              # レポート生成
│   │   ├── generator.py        # レポート生成器
│   │   └── templates/          # レポートテンプレート
│   └── __init__.py
├── tools/                      # 統合ツール
│   ├── unified_analyzer.py     # 統合分析ツール (メイン)
│   └── __init__.py
└── config/                     # 設定ファイル
    └── analysis/               # 分析設定
        └── default.json        # デフォルト設定
```

## 使用方法

### 1. 統合分析ツール

```bash
# 基本的な使用方法
python -m ztb.tools.unified_analyzer \
  --config config/analysis/default.json \
  --model models/sac_model.zip

# 出力ディレクトリ指定
python -m ztb.tools.unified_analyzer \
  --config config/analysis/default.json \
  --model models/sac_model.zip \
  --output-dir reports/custom_analysis
```

### 2. 設定ファイル

`config/analysis/default.json` で分析設定をカスタマイズ：

```json
{
  "analysis_type": "comprehensive",
  "log_level": "INFO",
  "output_dir": "reports/analysis",

  "analysis": {
    "performance_analysis": true,
    "risk_analysis": true,
    "behavioral_analysis": true,
    "comparison_analysis": false
  },

  "reporting": {
    "format": "html",
    "include_plots": true,
    "sections": {
      "summary": true,
      "performance": true,
      "risk": true,
      "behavioral": true
    }
  }
}
```

## 分析内容

### パフォーマンス分析
- 総リターン
- シャープレシオ
- 最大ドローダウン
- 勝率
- プロフィットファクター
- 総取引数
- 平均取引時間

### リスク分析
- VaR (95%, 99%)
- Expected Shortfall
- ベータ
- ボラティリティ
- リスク調整リターン

### 行動分析
- アクション分布 (HOLD/BUY/SELL)
- ポジション継続時間
- 取引頻度
- 市場レジーム適応度

## レポート形式

- **JSON**: 構造化データ、API連携用
- **HTML**: ブラウザ表示用、スタイリング付き
- **Markdown**: ドキュメント用、GitHub対応

## 拡張性

### 新しい分析指標の追加

```python
# ztb/analysis/core/analyzer.py にメソッドを追加
def analyze_custom_metric(self, model: Any, **kwargs) -> Dict[str, Any]:
    """カスタム分析指標"""
    # 実装
    return {"custom_metric": value}
```

### 新しいレポート形式の追加

```python
# ztb/analysis/reporting/generator.py にメソッドを追加
def _generate_custom_report(self, results: Dict[str, Any], ...) -> Path:
    """カスタムレポート生成"""
    # 実装
    return filepath
```

## 次のステップ

1. **ワークフロー自動化**: トレーニング→バックテスト→分析→レポートの自動実行
2. **旧スクリプト移行**: 既存の散らばったスクリプトを新構造に移行
3. **バックテスト統合**: バックテストエンジンの統一インターフェース実装
4. **CI/CD統合**: 自動分析パイプラインの構築

## 使用例

```bash
# SAC v438モデルの分析
python -m ztb.tools.unified_analyzer \
  --config config/analysis/default.json \
  --model models/sac_model.zip

# 出力: reports/analysis/analysis_report_YYYYMMDD_HHMMSS.html
```

レポートにはパフォーマンス指標、リスク指標、行動分析結果が含まれ、モデルの総合的な評価が可能になります。
