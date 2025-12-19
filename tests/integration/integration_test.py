"""
統合テスト: ウォークフォワード分析と統合最適化システムの連携テスト

Phase 3-2品質確保のための統合テストを実行します。
ウォークフォワード分析、Kelly基準、ATRリスク管理、適応型信頼度調整の
完全な統合動作を確認します。
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.analysis.walk_forward_analyzer import WalkForwardAnalyzer, ParameterSet
from ztb.analysis.integrated_optimizer import IntegratedParameterOptimizer, IntegratedOptimizationConfig
from ztb.analysis.strategy_evaluators import create_simple_strategy_evaluator
from ztb.analysis.atr_risk_manager import RiskManagementMode

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_market_data(days: int = 365) -> pd.DataFrame:
    """テスト用市場データ生成"""
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    np.random.seed(42)

    # トレンド + ノイズの価格生成
    trend = np.linspace(0, 50, days)
    noise = np.random.randn(days) * 5
    prices = 100 + trend + noise

    # OHLCデータ生成
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(days)) * 2,
        'low': prices - np.abs(np.random.randn(days)) * 2,
        'close': prices + np.random.randn(days) * 1,
        'volume': np.random.randint(1000, 10000, days)
    }, index=dates)

    return data


def create_test_trades_data(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """テスト用トレードデータ生成"""
    trades = []
    np.random.seed(123)

    for i in range(50):  # 50トレード生成
        entry_date = data.index[np.random.randint(30, len(data) - 30)]
        exit_date = entry_date + timedelta(days=np.random.randint(1, 30))

        # シンプルなエントリー/エグジットロジック
        entry_price = data.loc[entry_date, 'close']
        exit_price = data.loc[exit_date, 'close'] if exit_date in data.index else entry_price * (1 + np.random.randn() * 0.1)

        pnl = (exit_price - entry_price) / entry_price
        success = pnl > 0

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'success': success,
            'confidence': np.random.uniform(0.5, 0.9),
            'position_size': np.random.uniform(0.01, 0.05)
        })

    return trades


def test_walk_forward_integration():
    """ウォークフォワード分析の統合テスト"""
    logger.info("ウォークフォワード分析統合テスト開始")

    # テストデータ生成
    market_data = create_test_market_data(365)
    trades_data = create_test_trades_data(market_data)

    # ウォークフォワード分析器初期化
    analyzer = WalkForwardAnalyzer()

    # パラメータセット定義
    param_sets = [
        ParameterSet(
            stop_loss_atr_multiplier=1.5,
            take_profit_risk_multiplier=2.0,
            position_size_kelly_fraction=0.02,
            confidence_threshold=0.8,
            max_positions=5,
            name="conservative"
        ),
        ParameterSet(
            stop_loss_atr_multiplier=2.0,
            take_profit_risk_multiplier=3.0,
            position_size_kelly_fraction=0.05,
            confidence_threshold=0.7,
            max_positions=5,
            name="balanced"
        )
    ]

    # 戦略評価関数
    strategy_evaluator = create_simple_strategy_evaluator()

    # ウォークフォワード分析実行
    try:
        results = analyzer.walk_forward_optimization(
            data=market_data,
            strategy_func=strategy_evaluator,
            train_days=90,
            test_days=30,
            step_days=15,
            parameter_sets=param_sets,
            min_samples=30
        )

        logger.info(f"ウォークフォワード分析完了: {len(results)} 結果")
        assert len(results) > 0, "ウォークフォワード分析結果が空です"

        # 最適パラメータ確認
        best_result = max(results, key=lambda x: x.out_of_sample_performance.get('total_return', 0))
        logger.info(f"最適パラメータ: {best_result.best_parameters.name}")
        logger.info(f"総リターン: {best_result.out_of_sample_performance.get('total_return', 0):.4f}")

        return True

    except Exception as e:
        logger.error(f"ウォークフォワード分析エラー: {e}")
        return False


def test_integrated_optimizer():
    """統合最適化システムのテスト"""
    logger.info("統合最適化システムテスト開始")

    # テストデータ生成
    market_data = create_test_market_data(365)

    # 統合最適化設定
    config = IntegratedOptimizationConfig(
        train_days=60,
        test_days=20,
        step_days=10,
        min_samples=20,
        min_trades_for_kelly=5,
        kelly_risk_tolerance="half",
        max_position_size=0.05,
        atr_period=14,
        risk_management_mode=RiskManagementMode.DYNAMIC,
        base_confidence_threshold=0.7,
        adaptive_thresholds_enabled=True
    )

    # 統合最適化器初期化
    optimizer = IntegratedParameterOptimizer(config)

    # 最適化実行
    try:
        # モック戦略関数
        def mock_strategy_evaluator(data: pd.DataFrame, params: ParameterSet) -> Dict[str, float]:
            return {
                'total_return': 0.1,
                'sharpe_ratio': 1.5,
                'win_rate': 0.55,
                'max_drawdown': 0.15
            }

        results = optimizer.run_integrated_optimization(
            market_data=market_data,
            base_strategy_func=mock_strategy_evaluator
        )

        logger.info("統合最適化完了")
        assert results is not None, "統合最適化結果がNoneです"

        # 結果確認
        logger.info(f"平均Sharpe比率: {results.average_sharpe_ratio:.4f}")
        logger.info(f"平均勝率: {results.average_win_rate:.4f}")
        logger.info(f"総リターン: {results.total_return:.4f}")

        return True

    except Exception as e:
        logger.error(f"統合最適化エラー: {e}")
        return False


def test_component_integration():
    """コンポーネント統合テスト"""
    logger.info("コンポーネント統合テスト開始")

    # テストデータ生成
    market_data = create_test_market_data(100)

    # 各コンポーネントの連携テスト
    try:
        # 1. ATRリスクマネージャー
        from ztb.analysis.atr_risk_manager import ATRRiskManager
        atr_manager = ATRRiskManager()
        # 基本的な初期化テスト
        assert atr_manager is not None, "ATRマネージャー初期化失敗"

        # 2. Kellyポジションサイザー
        from ztb.analysis.kelly_position_sizer import KellyPositionSizer
        kelly_sizer = KellyPositionSizer()
        assert kelly_sizer is not None, "Kellyサイザー初期化失敗"

        # 3. 適応型信頼度調整
        from ztb.analysis.adaptive_confidence_adjuster import AdaptiveConfidenceAdjuster
        confidence_adjuster = AdaptiveConfidenceAdjuster()
        decision = confidence_adjuster.calculate_adaptive_threshold(market_data)
        assert decision.final_threshold > 0, "信頼度調整失敗"

        logger.info("コンポーネント初期化成功")

        return True

    except Exception as e:
        logger.error(f"コンポーネント統合エラー: {e}")
        return False


def main():
    """メイン実行関数"""
    logger.info("Phase 3-2統合テスト開始")

    import time
    start_time = time.time()

    test_results = []

    # テスト実行
    tests = [
        ("ウォークフォワード分析統合", test_walk_forward_integration),
        ("統合最適化システム", test_integrated_optimizer),
        ("コンポーネント統合", test_component_integration)
    ]

    for test_name, test_func in tests:
        logger.info(f"テスト実行: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
            logger.info(f"テスト結果: {test_name} - {'成功' if result else '失敗'}")
        except Exception as e:
            logger.error(f"テスト例外: {test_name} - {e}")
            test_results.append((test_name, False))

    end_time = time.time()
    execution_time = end_time - start_time

    # 結果サマリー
    success_count = sum(1 for _, result in test_results if result)
    total_count = len(test_results)

    logger.info(f"統合テスト完了: {success_count}/{total_count} 成功 (実行時間: {execution_time:.2f}秒)")

    # 詳細結果出力
    for test_name, result in test_results:
        status = "✓" if result else "✗"
        logger.info(f"{status} {test_name}")

    # 結果保存
    result_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_count,
        'successful_tests': success_count,
        'failed_tests': total_count - success_count,
        'execution_time_seconds': execution_time,
        'test_results': [
            {'name': name, 'success': result}
            for name, result in test_results
        ]
    }

    with open('integration_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)

    logger.info("統合テスト結果を integration_test_results.json に保存しました")

    # 最終判定
    if success_count == total_count:
        logger.info("🎉 全統合テスト成功！Phase 3-2品質確保完了")
        return 0
    else:
        logger.error(f"❌ {total_count - success_count} テスト失敗")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)