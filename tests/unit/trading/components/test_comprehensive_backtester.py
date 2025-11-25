"""Tests for Comprehensive Backtesting System component."""

import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.trading.comprehensive_backtester import (
    BacktestConfiguration,
    BacktestEngine,
    BacktestResult,
    ComprehensiveBacktestingSystem,
    MarketData,
    PerformanceAnalyzer,
    RiskManager,
    RiskMetrics,
    StrategyEvaluator,
    TradeRecord,
    TradingPerformanceMetrics,
)


@pytest.fixture
def mock_integration_manager():
    """Mock V433 Integration Manager"""
    manager = Mock()
    manager.component_manager = Mock()
    manager.component_manager.v433_system = Mock()
    manager.component_manager.v433_system.update_market_data = Mock(return_value=None)
    manager.component_manager.position_manager = Mock()
    manager.component_manager.position_manager.submit_signal = Mock(return_value=None)
    return manager


@pytest.fixture
def backtesting_system(mock_integration_manager):
    """Comprehensive Backtesting System instance"""
    return ComprehensiveBacktestingSystem(mock_integration_manager)


@pytest.fixture
def sample_backtest_config():
    """Sample backtest configuration"""
    return BacktestConfiguration(
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 12, 31),
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.1,
        risk_per_trade=0.02,
        max_drawdown_limit=0.2,
        benchmark_symbol="BTC/JPY",
        data_frequency="1H",
    )


@pytest.fixture
def sample_market_data():
    """Sample market data"""
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    data = pd.DataFrame(
        {
            "open": np.random.uniform(1000000, 2000000, 100),
            "high": np.random.uniform(1000000, 2000000, 100),
            "low": np.random.uniform(1000000, 2000000, 100),
            "close": np.random.uniform(1000000, 2000000, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )

    return MarketData(
        symbol="BTC/JPY",
        timeframe="1H",
        data=data,
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 1, 5),
    )


@pytest.fixture
def sample_trade_records():
    """Sample trade records"""
    return [
        TradeRecord(
            trade_id="trade_1",
            timestamp=datetime.datetime(2023, 1, 1, 10, 0),
            symbol="BTC/JPY",
            side="buy",
            quantity=0.01,
            price=1500000.0,
            commission=15.0,
        ),
        TradeRecord(
            trade_id="trade_2",
            timestamp=datetime.datetime(2023, 1, 1, 15, 0),
            symbol="BTC/JPY",
            side="sell",
            quantity=0.01,
            price=1550000.0,
            commission=15.5,
        ),
    ]


class TestComprehensiveBacktestingSystemInitialization:
    """Initialization tests for Comprehensive Backtesting System"""

    def test_initialization(
        self,
        backtesting_system: ComprehensiveBacktestingSystem,
        mock_integration_manager,
    ):
        """Test successful initialization"""
        assert backtesting_system.integration_manager == mock_integration_manager
        assert isinstance(backtesting_system.backtest_engine, BacktestEngine)
        assert isinstance(backtesting_system.strategy_evaluator, StrategyEvaluator)
        assert isinstance(backtesting_system.risk_manager, RiskManager)
        assert isinstance(backtesting_system.performance_analyzer, PerformanceAnalyzer)
        assert backtesting_system.backtest_results == []
        assert backtesting_system.is_running is False

    def test_initialization_with_config(
        self, mock_integration_manager, sample_backtest_config
    ):
        """Test initialization with configuration"""
        system = ComprehensiveBacktestingSystem(
            mock_integration_manager, sample_backtest_config
        )

        assert system.config == sample_backtest_config
        assert system.backtest_engine.config == sample_backtest_config


class TestComprehensiveBacktestingSystemOperations:
    """Operation tests for Comprehensive Backtesting System"""

    def test_run_comprehensive_backtest(
        self,
        backtesting_system: ComprehensiveBacktestingSystem,
        sample_backtest_config,
        sample_market_data,
    ):
        """Test comprehensive backtest execution"""
        with patch.object(
            backtesting_system.backtest_engine, "run_backtest"
        ) as mock_run, patch.object(
            backtesting_system.strategy_evaluator, "evaluate_strategy"
        ) as mock_evaluate, patch.object(
            backtesting_system.risk_manager, "assess_risk"
        ) as mock_risk, patch.object(
            backtesting_system.performance_analyzer, "analyze_performance"
        ) as mock_analyze:
            # Mock backtest result
            mock_result = BacktestResult(
                config=sample_backtest_config,
                trades=[],
                performance_metrics=TradingPerformanceMetrics(),
                risk_metrics=RiskMetrics(),
                execution_time=10.5,
                success=True,
            )
            mock_run.return_value = mock_result

            # Mock evaluation results
            mock_evaluate.return_value = {
                "strategy_score": 85.0,
                "confidence_level": 0.9,
                "recommendations": ["Good performance"],
            }

            mock_risk.return_value = {
                "risk_score": 75.0,
                "risk_adjusted_return": 1.2,
                "risk_warnings": [],
            }

            mock_analyze.return_value = {
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.15,
                "win_rate": 0.65,
            }

            result = backtesting_system.run_comprehensive_backtest(
                sample_backtest_config, sample_market_data
            )

            assert isinstance(result, BacktestResult)
            assert result.config == sample_backtest_config
            assert result.execution_time == 10.5
            assert result.success is True
            assert len(backtesting_system.backtest_results) == 1

    def test_run_multiple_backtests(
        self,
        backtesting_system: ComprehensiveBacktestingSystem,
        sample_backtest_config,
        sample_market_data,
    ):
        """Test running multiple backtests"""
        configs = [sample_backtest_config, sample_backtest_config]

        with patch.object(backtesting_system, "run_comprehensive_backtest") as mock_run:
            mock_run.return_value = BacktestResult(
                config=sample_backtest_config,
                trades=[],
                performance_metrics=TradingPerformanceMetrics(),
                risk_metrics=RiskMetrics(),
                execution_time=5.0,
                success=True,
            )

            results = backtesting_system.run_multiple_backtests(
                configs, sample_market_data
            )

            assert len(results) == 2
            assert all(isinstance(r, BacktestResult) for r in results)
            assert mock_run.call_count == 2

    def test_get_backtest_report(
        self, backtesting_system: ComprehensiveBacktestingSystem, sample_backtest_config
    ):
        """Test getting backtest report"""
        # Add mock results
        backtesting_system.backtest_results = [
            BacktestResult(
                config=sample_backtest_config,
                trades=[],
                performance_metrics=TradingPerformanceMetrics(
                    total_return=0.15, sharpe_ratio=1.2
                ),
                risk_metrics=RiskMetrics(max_drawdown=0.1, value_at_risk=0.05),
                execution_time=10.0,
                success=True,
            )
        ]

        report = backtesting_system.get_backtest_report()

        assert "summary" in report
        assert "performance_overview" in report
        assert "risk_analysis" in report
        assert "recommendations" in report
        assert report["summary"]["total_backtests"] == 1
        assert report["summary"]["successful_backtests"] == 1
        assert report["performance_overview"]["avg_total_return"] == 0.15

    def test_compare_strategies(
        self, backtesting_system: ComprehensiveBacktestingSystem, sample_backtest_config
    ):
        """Test strategy comparison"""
        # Add multiple mock results
        backtesting_system.backtest_results = [
            BacktestResult(
                config=sample_backtest_config,
                trades=[],
                performance_metrics=TradingPerformanceMetrics(
                    total_return=0.15, sharpe_ratio=1.2
                ),
                risk_metrics=RiskMetrics(max_drawdown=0.1),
                execution_time=10.0,
                success=True,
            ),
            BacktestResult(
                config=sample_backtest_config,
                trades=[],
                performance_metrics=TradingPerformanceMetrics(
                    total_return=0.20, sharpe_ratio=1.5
                ),
                risk_metrics=RiskMetrics(max_drawdown=0.12),
                execution_time=12.0,
                success=True,
            ),
        ]

        comparison = backtesting_system.compare_strategies()

        assert "best_performing" in comparison
        assert "comparison_metrics" in comparison
        assert "recommendations" in comparison
        assert comparison["best_performing"]["total_return"] == 0.20
        assert comparison["best_performing"]["sharpe_ratio"] == 1.5

    def test_validate_backtest_results(
        self, backtesting_system: ComprehensiveBacktestingSystem, sample_backtest_config
    ):
        """Test backtest result validation"""
        valid_result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(total_return=0.1),
            risk_metrics=RiskMetrics(max_drawdown=0.1),
            execution_time=5.0,
            success=True,
        )

        invalid_result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(
                total_return=-0.5
            ),  # Bad return
            risk_metrics=RiskMetrics(max_drawdown=0.3),  # High drawdown
            execution_time=5.0,
            success=True,
        )

        assert backtesting_system.validate_backtest_results(valid_result) is True
        assert backtesting_system.validate_backtest_results(invalid_result) is False


class TestBacktestEngine:
    """Tests for BacktestEngine"""

    def test_initialization(self, mock_integration_manager, sample_backtest_config):
        """Test BacktestEngine initialization"""
        engine = BacktestEngine(mock_integration_manager, sample_backtest_config)

        assert engine.integration_manager == mock_integration_manager
        assert engine.config == sample_backtest_config
        assert engine.current_capital == sample_backtest_config.initial_capital
        assert engine.trades == []

    def test_run_backtest(
        self, mock_integration_manager, sample_backtest_config, sample_market_data
    ):
        """Test backtest execution"""
        engine = BacktestEngine(mock_integration_manager, sample_backtest_config)

        with patch.object(
            engine, "_load_market_data", return_value=sample_market_data
        ), patch.object(
            engine, "_execute_trading_strategy"
        ) as mock_execute, patch.object(
            engine, "_calculate_performance_metrics"
        ) as mock_calc:
            mock_execute.return_value = []  # No trades
            mock_calc.return_value = TradingPerformanceMetrics(total_return=0.05)

            result = engine.run_backtest(sample_market_data)

            assert isinstance(result, BacktestResult)
            assert result.config == sample_backtest_config
            assert result.performance_metrics.total_return == 0.05
            assert result.success is True

    def test_execute_trading_strategy(
        self, mock_integration_manager, sample_backtest_config, sample_market_data
    ):
        """Test trading strategy execution"""
        engine = BacktestEngine(mock_integration_manager, sample_backtest_config)

        with patch.object(engine, "_generate_signals") as mock_signals, patch.object(
            engine, "_execute_signal"
        ) as mock_execute:
            mock_signals.return_value = [
                {
                    "timestamp": datetime.datetime(2023, 1, 1, 10),
                    "signal": "buy",
                    "price": 1500000.0,
                },
                {
                    "timestamp": datetime.datetime(2023, 1, 1, 15),
                    "signal": "sell",
                    "price": 1550000.0,
                },
            ]

            mock_execute.side_effect = [
                TradeRecord(
                    "trade_1",
                    datetime.datetime(2023, 1, 1, 10),
                    "BTC/JPY",
                    "buy",
                    0.01,
                    1500000.0,
                    15.0,
                ),
                TradeRecord(
                    "trade_2",
                    datetime.datetime(2023, 1, 1, 15),
                    "BTC/JPY",
                    "sell",
                    0.01,
                    1550000.0,
                    15.5,
                ),
            ]

            trades = engine._execute_trading_strategy(sample_market_data)

            assert len(trades) == 2
            assert trades[0].side == "buy"
            assert trades[1].side == "sell"
            assert mock_signals.call_count == 1
            assert mock_execute.call_count == 2

    def test_calculate_performance_metrics(
        self, mock_integration_manager, sample_backtest_config, sample_trade_records
    ):
        """Test performance metrics calculation"""
        engine = BacktestEngine(mock_integration_manager, sample_backtest_config)
        engine.trades = sample_trade_records
        engine.current_capital = 100000.0

        metrics = engine._calculate_performance_metrics()

        assert isinstance(metrics, TradingPerformanceMetrics)
        assert metrics.total_trades == 2
        # Profit calculation: sell_price - buy_price - commissions
        expected_profit = (1550000.0 - 1500000.0) - (
            15.0 + 15.5
        )  # 50000 - 30.5 = 49969.5
        assert (
            abs(metrics.total_return - (expected_profit / 100000.0)) < 0.001
        )  # Return percentage


class TestStrategyEvaluator:
    """Tests for StrategyEvaluator"""

    def test_initialization(self, mock_integration_manager):
        """Test StrategyEvaluator initialization"""
        evaluator = StrategyEvaluator(mock_integration_manager)

        assert evaluator.integration_manager == mock_integration_manager
        assert evaluator.evaluation_criteria == {}

    def test_evaluate_strategy(self, mock_integration_manager, sample_backtest_config):
        """Test strategy evaluation"""
        evaluator = StrategyEvaluator(mock_integration_manager)

        result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(
                total_return=0.15,
                sharpe_ratio=1.5,
                win_rate=0.7,
                max_consecutive_wins=5,
            ),
            risk_metrics=RiskMetrics(max_drawdown=0.1),
            execution_time=10.0,
            success=True,
        )

        evaluation = evaluator.evaluate_strategy(result)

        assert "strategy_score" in evaluation
        assert "confidence_level" in evaluation
        assert "recommendations" in evaluation
        assert evaluation["strategy_score"] >= 0 and evaluation["strategy_score"] <= 100
        assert (
            evaluation["confidence_level"] >= 0 and evaluation["confidence_level"] <= 1
        )

    def test_evaluate_strategy_weak_performance(
        self, mock_integration_manager, sample_backtest_config
    ):
        """Test strategy evaluation with weak performance"""
        evaluator = StrategyEvaluator(mock_integration_manager)

        result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(
                total_return=-0.1, sharpe_ratio=0.3, win_rate=0.4
            ),
            risk_metrics=RiskMetrics(max_drawdown=0.25),
            execution_time=10.0,
            success=True,
        )

        evaluation = evaluator.evaluate_strategy(result)

        assert evaluation["strategy_score"] < 50  # Should be low score
        assert "recommendations" in evaluation
        assert len(evaluation["recommendations"]) > 0


class TestRiskManager:
    """Tests for RiskManager"""

    def test_initialization(self, mock_integration_manager):
        """Test RiskManager initialization"""
        risk_manager = RiskManager(mock_integration_manager)

        assert risk_manager.integration_manager == mock_integration_manager
        assert risk_manager.risk_limits == {}

    def test_assess_risk(self, mock_integration_manager, sample_backtest_config):
        """Test risk assessment"""
        risk_manager = RiskManager(mock_integration_manager)

        result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(total_return=0.1),
            risk_metrics=RiskMetrics(
                max_drawdown=0.15, value_at_risk=0.05, expected_shortfall=0.08, beta=0.8
            ),
            execution_time=10.0,
            success=True,
        )

        assessment = risk_manager.assess_risk(result)

        assert "risk_score" in assessment
        assert "risk_adjusted_return" in assessment
        assert "risk_warnings" in assessment
        assert assessment["risk_score"] >= 0 and assessment["risk_score"] <= 100
        assert assessment["risk_adjusted_return"] > 0

    def test_assess_risk_high_risk(
        self, mock_integration_manager, sample_backtest_config
    ):
        """Test risk assessment with high risk"""
        risk_manager = RiskManager(mock_integration_manager)

        result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(total_return=0.05),
            risk_metrics=RiskMetrics(
                max_drawdown=0.35,
                value_at_risk=0.12,
                expected_shortfall=0.18,  # High drawdown  # High VaR
            ),
            execution_time=10.0,
            success=True,
        )

        assessment = risk_manager.assess_risk(result)

        assert assessment["risk_score"] < 50  # Should be low score
        assert len(assessment["risk_warnings"]) > 0


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer"""

    def test_initialization(self, mock_integration_manager):
        """Test PerformanceAnalyzer initialization"""
        analyzer = PerformanceAnalyzer(mock_integration_manager)

        assert analyzer.integration_manager == mock_integration_manager

    def test_analyze_performance(
        self, mock_integration_manager, sample_backtest_config, sample_trade_records
    ):
        """Test performance analysis"""
        analyzer = PerformanceAnalyzer(mock_integration_manager)

        result = BacktestResult(
            config=sample_backtest_config,
            trades=sample_trade_records,
            performance_metrics=TradingPerformanceMetrics(),
            risk_metrics=RiskMetrics(),
            execution_time=10.0,
            success=True,
        )

        analysis = analyzer.analyze_performance(result)

        assert "sharpe_ratio" in analysis
        assert "max_drawdown" in analysis
        assert "win_rate" in analysis
        assert "profit_factor" in analysis
        assert "avg_trade_duration" in analysis
        assert "monthly_returns" in analysis

    def test_calculate_advanced_metrics(
        self, mock_integration_manager, sample_trade_records
    ):
        """Test advanced metrics calculation"""
        analyzer = PerformanceAnalyzer(mock_integration_manager)

        metrics = analyzer._calculate_advanced_metrics(sample_trade_records)

        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "omega_ratio" in metrics
        assert isinstance(metrics["sharpe_ratio"], (int, float))


class TestBacktestConfiguration:
    """Tests for BacktestConfiguration dataclass"""

    def test_initialization(self):
        """Test BacktestConfiguration initialization"""
        config = BacktestConfiguration()

        assert config.start_date is None
        assert config.end_date is None
        assert config.initial_capital == 100000.0
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0005

    def test_custom_initialization(self, sample_backtest_config):
        """Test BacktestConfiguration with custom values"""
        assert sample_backtest_config.initial_capital == 100000.0
        assert sample_backtest_config.commission_rate == 0.001
        assert sample_backtest_config.max_position_size == 0.1
        assert sample_backtest_config.risk_per_trade == 0.02


class TestBacktestResult:
    """Tests for BacktestResult dataclass"""

    def test_initialization(self, sample_backtest_config):
        """Test BacktestResult initialization"""
        result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(),
            risk_metrics=RiskMetrics(),
            execution_time=10.5,
            success=True,
        )

        assert result.config == sample_backtest_config
        assert result.trades == []
        assert result.execution_time == 10.5
        assert result.success is True

    def test_result_summary(self, sample_backtest_config):
        """Test result summary property"""
        result = BacktestResult(
            config=sample_backtest_config,
            trades=[],
            performance_metrics=TradingPerformanceMetrics(
                total_return=0.15, total_trades=10
            ),
            risk_metrics=RiskMetrics(max_drawdown=0.1),
            execution_time=10.5,
            success=True,
        )

        summary = result.result_summary

        assert "Total Return" in summary
        assert "15.00%" in summary
        assert "Total Trades" in summary
        assert "10" in summary
        assert "Max Drawdown" in summary
        assert "10.00%" in summary


class TestTradeRecord:
    """Tests for TradeRecord dataclass"""

    def test_initialization(self):
        """Test TradeRecord initialization"""
        trade = TradeRecord(
            trade_id="test_trade",
            timestamp=datetime.datetime(2023, 1, 1, 10, 0),
            symbol="BTC/JPY",
            side="buy",
            quantity=0.01,
            price=1500000.0,
            commission=15.0,
        )

        assert trade.trade_id == "test_trade"
        assert trade.symbol == "BTC/JPY"
        assert trade.side == "buy"
        assert trade.quantity == 0.01
        assert trade.price == 1500000.0
        assert trade.commission == 15.0

    def test_trade_value(self):
        """Test trade value calculation"""
        trade = TradeRecord(
            trade_id="test_trade",
            timestamp=datetime.datetime(2023, 1, 1, 10, 0),
            symbol="BTC/JPY",
            side="buy",
            quantity=0.01,
            price=1500000.0,
            commission=15.0,
        )

        assert trade.trade_value == 15000.0  # 0.01 * 1500000
        assert trade.total_cost == 15015.0  # trade_value + commission


class TestMarketData:
    """Tests for MarketData dataclass"""

    def test_initialization(self, sample_market_data):
        """Test MarketData initialization"""
        assert sample_market_data.symbol == "BTC/JPY"
        assert sample_market_data.timeframe == "1H"
        assert isinstance(sample_market_data.data, pd.DataFrame)
        assert len(sample_market_data.data) == 100

    def test_data_validation(self, sample_market_data):
        """Test market data validation"""
        assert sample_market_data.validate_data() is True

        # Test with invalid data
        invalid_data = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [1, 2, 3],
                "low": [1, 2, 3],
                "close": [1, 2, 3],
            }  # High should be >= open
        )

        invalid_market_data = MarketData(
            symbol="BTC/JPY",
            timeframe="1H",
            data=invalid_data,
            start_date=datetime.datetime(2023, 1, 1),
            end_date=datetime.datetime(2023, 1, 1),
        )

        assert invalid_market_data.validate_data() is False


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass"""

    def test_initialization(self):
        """Test PerformanceMetrics initialization"""
        metrics = TradingPerformanceMetrics()

        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0

    def test_calculated_properties(self):
        """Test calculated properties"""
        metrics = TradingPerformanceMetrics(
            total_return=0.15, total_trades=10, winning_trades=7, losing_trades=3
        )

        assert metrics.win_rate == 0.7
        assert metrics.loss_rate == 0.3
        assert metrics.profit_factor == 0.0  # Would need profit/loss amounts


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass"""

    def test_initialization(self):
        """Test RiskMetrics initialization"""
        metrics = RiskMetrics()

        assert metrics.max_drawdown == 0.0
        assert metrics.value_at_risk == 0.0
        assert metrics.expected_shortfall == 0.0

    def test_risk_score_calculation(self):
        """Test risk score calculation"""
        metrics = RiskMetrics(
            max_drawdown=0.15, value_at_risk=0.05, expected_shortfall=0.08
        )

        # Risk score is a composite measure
        assert isinstance(metrics.risk_score, (int, float))
        assert metrics.risk_score >= 0
