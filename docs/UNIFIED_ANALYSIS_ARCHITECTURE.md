"""
統合分析・バックテストシステムのアーキテクチャ概要

このシステムは、トレーニング、バックテスト、分析、レポート生成を統合的に管理します。
"""

# =============================================================================
# 統合分析・バックテストシステム アーキテクチャ
# =============================================================================

"""
推奨構成:

ztb/
├── analysis/                    # 統合分析システム
│   ├── core/                   # コア分析機能
│   │   ├── analyzer.py         # メイン分析器
│   │   ├── metrics.py          # 分析指標計算
│   │   └── validators.py       # 検証機能
│   ├── backtest/               # バックテストモジュール
│   │   ├── engine.py           # バックテストエンジン
│   │   ├── strategies.py       # 戦略テスト
│   │   └── validation.py       # バックテスト検証
│   ├── performance/            # パフォーマンス分析
│   │   ├── calculator.py       # パフォーマンス計算
│   │   ├── risk_metrics.py     # リスク指標
│   │   └── comparison.py       # 比較分析
│   ├── reporting/              # レポート生成
│   │   ├── generator.py        # レポート生成器
│   │   ├── templates/          # レポートテンプレート
│   │   └── exporters.py        # エクスポート機能
│   └── __init__.py
├── backtest/                   # バックテスト専用モジュール
│   ├── core/                   # コアバックテスト機能
│   ├── engines/                # バックテストエンジン
│   ├── strategies/             # 戦略実装
│   └── validation/             # 検証機能
├── tools/                      # 統合ツール
│   ├── unified_analyzer.py     # 統合分析ツール (メインエントリーポイント)
│   ├── unified_backtester.py   # 統合バックテストツール
│   └── unified_reporter.py     # 統合レポートツール
└── config/                     # 設定ファイル
    ├── analysis/               # 分析設定
    │   ├── default.json
    │   └── performance_metrics.json
    ├── backtest/               # バックテスト設定
    │   ├── default.json
    │   └── validation_rules.json
    └── reporting/              # レポート設定
        ├── templates.json
        └── formats.json

# =============================================================================
# 使用例
# =============================================================================

# 1. 統合分析ツールの使用
python -m ztb.tools.unified_analyzer --config config/analysis/default.json --model models/sac_model.zip

# 2. 統合バックテストツールの使用
python -m ztb.tools.unified_backtester --config config/backtest/default.json --model models/sac_model.zip

# 3. ワークフロー自動化
python -m ztb.tools.workflow_runner --config config/workflow/training_to_report.json

# =============================================================================
# 設定ファイル例
# =============================================================================

# config/analysis/default.json
{
  "analysis_type": "comprehensive",
  "metrics": {
    "performance": ["sharpe_ratio", "max_drawdown", "win_rate"],
    "risk": ["var_95", "expected_shortfall"],
    "behavioral": ["action_distribution", "position_duration"]
  },
  "comparison": {
    "baseline_models": ["ppo_baseline", "sac_v427"],
    "metrics": ["profit", "risk_adjusted_return"]
  },
  "reporting": {
    "format": "html",
    "include_plots": true,
    "export_path": "reports/analysis/"
  }
}

# =============================================================================
# ワークフロー自動化
# =============================================================================

ワークフロー例:
1. モデルトレーニング (SAC/PPO)
2. 自動バックテスト実行
3. パフォーマンス分析
4. 比較分析 (前バージョンとの比較)
5. レポート生成 (HTML/PDF)
6. 結果保存・通知

# =============================================================================
# 利点
# =============================================================================

1. 統一インターフェース: すべての分析が同じAPIで実行可能
2. 設定管理: JSONベースの設定で柔軟なカスタマイズ
3. 再利用性: モジュール化により機能の再利用が容易
4. 保守性: 構造化されたコードでメンテナンスが容易
5. 拡張性: 新しい分析機能を簡単に追加可能
6. 自動化: ワークフロー自動化で効率化

# =============================================================================
# 移行計画
# =============================================================================

Phase 1: 構造作成
- 新しいディレクトリ構造を作成
- コアモジュールを設計・実装

Phase 2: 機能移行
- 既存スクリプトを新構造に移行
- 統合インターフェースを実装

Phase 3: 設定統合
- 設定ファイルを統一
- ConfigManager拡張

Phase 4: ワークフロー自動化
- 自動化スクリプト実装
- CI/CD統合

Phase 5: テスト・検証
- 統合テスト実行
- 既存機能の互換性確認
"""</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\UNIFIED_ANALYSIS_ARCHITECTURE.md
