#!/usr/bin/env python3
"""
バックテスト分析の単体テスト
BacktestAnalyzerクラスの各機能を包括的にテスト
"""

import json
import sys
from pathlib import Path

import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.backtest.analyze_backtest import BacktestAnalyzer


class TestBacktestAnalyzer:
    """BacktestAnalyzerクラスの単体テスト"""

    def test_init_with_valid_file(self, tmp_path):
        """有効なファイルでの初期化テスト"""
        # テストデータ作成
        test_data = {
            "initial_balance": 10000,
            "final_portfolio_value": 12000,
            "total_steps": 1000,
            "portfolio_values": [10000, 11000, 12000],
        }

        test_file = tmp_path / "test_backtest.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        analyzer = BacktestAnalyzer(str(test_file))

        assert analyzer.data["initial_portfolio"] == 10000
        assert analyzer.data["final_portfolio"] == 12000
        assert analyzer.data["portfolio_history"] == [10000, 11000, 12000]

    def test_init_with_missing_required_fields(self, tmp_path):
        """必須フィールドが欠けている場合のテスト"""
        test_data = {"portfolio_values": [10000, 11000, 12000]}

        test_file = tmp_path / "test_backtest.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        with pytest.raises(ValueError, match="Missing required fields"):
            BacktestAnalyzer(str(test_file))

    def test_field_mapping(self):
        """フィールドマッピングのテスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {
            "initial_balance": 10000,
            "final_balance": 12000,
            "portfolio_values": [10000, 11000, 12000],
        }

        analyzer._validate_data()

        assert analyzer.data["initial_portfolio"] == 10000
        assert analyzer.data["final_portfolio"] == 12000
        assert analyzer.data["portfolio_history"] == [10000, 11000, 12000]

    def test_calculate_win_rate_with_trade_pnls(self):
        """trade_pnlsを使用した勝率計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {"trade_pnls": [100, -50, 200, -25, 150]}  # 勝ち3、負け2

        win_rate = analyzer._calculate_win_rate()
        assert win_rate == 0.6  # 3/5 = 0.6

    def test_calculate_win_rate_with_winning_trades(self):
        """winning_trades/total_tradesを使用した勝率計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {"winning_trades": 7, "total_trades": 10}

        win_rate = analyzer._calculate_win_rate()
        assert win_rate == 0.7  # 7/10 = 0.7

    def test_calculate_win_rate_with_trades_array(self):
        """trades配列を使用した勝率計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {
            "trades": [
                {"type": "BUY", "pnl": 100},
                {"type": "SELL", "pnl": -50},
                {"type": "BUY", "pnl": 200},
                {"type": "FINAL_CLOSE", "pnl": 0},  # FINAL_CLOSEは除外されるべき
            ]
        }

        win_rate = analyzer._calculate_win_rate()
        assert win_rate == 2 / 3  # 2 winning trades out of 3 actual trades

    def test_calculate_win_rate_no_data(self):
        """データがない場合の勝率計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {}

        win_rate = analyzer._calculate_win_rate()
        assert win_rate == 0.0

    def test_calculate_risk_metrics_with_valid_data(self):
        """有効なデータでのリスク指標計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {"portfolio_history": [10000, 10200, 10100, 10300, 10200]}

        metrics = analyzer.calculate_risk_metrics()

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
        assert "win_rate" in metrics

        # 総リターンの確認 (200/10000 = 0.02)
        assert abs(metrics["total_return"] - 0.02) < 0.001

    def test_calculate_risk_metrics_empty_portfolio(self):
        """空のportfolio_historyでのリスク指標計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {"portfolio_history": []}

        metrics = analyzer.calculate_risk_metrics()

        assert metrics == {}

    def test_analyze_method(self):
        """analyzeメソッドのテスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = {
            "initial_portfolio": 10000,
            "final_portfolio": 12000,
            "portfolio_history": [10000, 11000, 12000],
            "trade_pnls": [1000, 2000],
        }

        results = analyzer.analyze()

        assert "risk_metrics" in results
        assert "temporal_patterns" in results
        assert "market_conditions" in results
        assert "trading_frequency" in results

        assert results["risk_metrics"]["win_rate"] == 1.0  # 2 winning trades

    def test_sharpe_ratio_calculation(self):
        """シャープレシオ計算のテスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)

        import numpy as np

        returns = np.array([0.01, 0.02, -0.01, 0.015])

        sharpe = analyzer.sharpe_ratio(returns, risk_free_rate=0.0, annualize=False)
        expected = np.mean(returns) / np.std(returns, ddof=1)

        assert abs(sharpe - expected) < 0.001

    def test_max_drawdown_calculation(self):
        """最大ドローダウン計算のテスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)

        values = [10000, 10500, 10300, 10800, 10200, 10600]
        max_dd = analyzer.max_drawdown(values)

        # 10800 -> 10200 のドローダウン: (10800-10200)/10800 = 0.0556
        expected_dd = (10800 - 10200) / 10800

        assert abs(max_dd - expected_dd) < 0.001

    @pytest.mark.parametrize(
        "test_data,expected_win_rate",
        [
            ({"trade_pnls": [100, -50, 200]}, 2 / 3),
            ({"winning_trades": 5, "total_trades": 8}, 5 / 8),
            ({"trades": [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}]}, 2 / 3),
            ({}, 0.0),
        ],
    )
    def test_calculate_win_rate_parametrized(self, test_data, expected_win_rate):
        """パラメータ化された勝率計算テスト"""
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = test_data

        win_rate = analyzer._calculate_win_rate()
        assert abs(win_rate - expected_win_rate) < 0.001


def test_backtest_analysis():
    """バックテスト分析の統合テスト"""

    print("=== バックテスト分析テスト開始 ===")

    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent.parent.parent

    # テスト対象のバックテスト結果ファイル
    test_files = [
        "backtest_results_sac_v444_2.json",
        "backtest_results_sac_v444.json",
        "backtest_results.json",
    ]

    for test_file in test_files:
        file_path = project_root / test_file
        print(f"チェック中: {file_path} (存在: {file_path.exists()})")
        if not file_path.exists():
            print(f"⚠ テストファイルが見つかりません: {test_file}")
            continue

        print(f"\n--- {test_file} の分析 ---")

        try:
            # BacktestAnalyzerのインスタンス化
            analyzer = BacktestAnalyzer(str(file_path))

            # データ読み込み確認
            print(f"初期残高: {analyzer.data.get('initial_balance', 'N/A')}")
            print(f"最終残高: {analyzer.data.get('final_balance', 'N/A')}")
            print(f"総取引数: {analyzer.data.get('total_trades', 'N/A')}")
            print(f"勝ちトレード: {analyzer.data.get('winning_trades', 'N/A')}")
            print(f"負けトレード: {analyzer.data.get('losing_trades', 'N/A')}")

            # 勝率計算テスト
            win_rate = analyzer._calculate_win_rate()
            print(f"計算された勝率: {win_rate:.3f} ({win_rate*100:.1f}%)")

            # リスク指標計算テスト
            risk_metrics = analyzer.calculate_risk_metrics()
            print("\nリスク指標:")
            for key, value in risk_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            # 総合分析実行
            results = analyzer.analyze()
            print("\n分析完了 ✅")
            # 勝率の詳細確認
            if "risk_metrics" in results and "win_rate" in results["risk_metrics"]:
                final_win_rate = results["risk_metrics"]["win_rate"]
                print(f"最終勝率: {final_win_rate:.3f} ({final_win_rate*100:.1f}%)")

        except Exception as e:
            print(f"✗ エラー: {e}")
            import traceback

            traceback.print_exc()


def test_win_rate_calculation():
    """勝率計算の詳細テスト"""

    print("\n=== 勝率計算詳細テスト ===")

    # モックデータでテスト
    test_cases = [
        {
            "name": "trade_pnls使用",
            "data": {"trade_pnls": [0.01, -0.005, 0.02, -0.01, 0.015]},
            "expected_win_rate": 0.6,  # 3勝2敗
        },
        {
            "name": "winning_trades/total_trades使用",
            "data": {"winning_trades": 7, "total_trades": 10},
            "expected_win_rate": 0.7,
        },
        {
            "name": "trades配列使用",
            "data": {
                "trades": [
                    {"type": "BUY", "pnl": 0.01},
                    {"type": "SELL", "pnl": -0.005},
                    {"type": "BUY", "pnl": 0.02},
                    {"type": "FINAL_CLOSE", "pnl": 0.0},  # このトレードは除外されるべき
                ]
            },
            "expected_win_rate": 2 / 3,  # 2勝1敗（FINAL_CLOSE除外）
        },
        {"name": "データなし", "data": {}, "expected_win_rate": 0.0},
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")

        # 一時的なanalyzerインスタンスを作成
        analyzer = BacktestAnalyzer.__new__(BacktestAnalyzer)
        analyzer.data = test_case["data"]

        calculated_win_rate = analyzer._calculate_win_rate()
        expected_win_rate = test_case["expected_win_rate"]

        print(f"期待値: {expected_win_rate:.3f}")
        print(f"計算値: {calculated_win_rate:.3f}")

        if abs(calculated_win_rate - expected_win_rate) < 0.001:
            print("✅ 正しい")
        else:
            print("✗ 不正")


if __name__ == "__main__":
    test_win_rate_calculation()
    test_backtest_analysis()
