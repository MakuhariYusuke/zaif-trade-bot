Evaluation Guide
===============

This section covers model evaluation, performance analysis, and backtesting methodologies for Zaif Trade Bot.

Evaluation Overview
-------------------

Model evaluation is crucial for understanding trading strategy performance. Zaif Trade Bot provides comprehensive evaluation tools for backtesting, performance metrics, and risk analysis.

Evaluation Workflow
-------------------

1. **Backtesting**: Test strategy on historical data
2. **Performance Analysis**: Calculate key metrics
3. **Risk Assessment**: Evaluate risk-adjusted returns
4. **Benchmarking**: Compare against baselines
5. **Validation**: Cross-validate results

Backtesting
-----------

Backtesting Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   evaluation:
     backtest:
       episodes: 100                    # Number of backtest episodes
       deterministic: false             # Use deterministic policy
       render: false                    # Render environment
       save_results: true               # Save results to file
       results_path: "results/backtest" # Results directory

Running Backtests
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.evaluation import Backtester
   from ztb.config import ConfigManager

   # Load configuration
   config = ConfigManager.load_config('config/trading.yaml')

   # Create backtester
   backtester = Backtester(config)

   # Run backtest
   results = backtester.run_backtest(
       data_path='data/market_data.csv',
       episodes=100
   )

   print(f"Total Return: {results.total_return:.2%}")
   print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
   print(f"Max Drawdown: {results.max_drawdown:.2%}")

Backtest Results
~~~~~~~~~~~~~~~~

Backtest results include comprehensive performance metrics:

.. code-block:: python

   # Access detailed results
   print("Backtest Results:")
   print(f"- Episodes: {results.num_episodes}")
   print(f"- Total Trades: {results.total_trades}")
   print(f"- Win Rate: {results.win_rate:.2%}")
   print(f"- Profit Factor: {results.profit_factor:.2f}")
   print(f"- Average Trade: {results.avg_trade_return:.2%}")

Performance Metrics
-------------------

Key Performance Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zaif Trade Bot calculates comprehensive performance metrics:

**Return Metrics**
* Total Return: Overall portfolio return
* Annualized Return: Return normalized to yearly basis
* Risk-Adjusted Return: Return adjusted for risk taken

**Risk Metrics**
* Volatility: Standard deviation of returns
* Sharpe Ratio: Risk-adjusted return measure
* Sortino Ratio: Downside risk-adjusted return
* Maximum Drawdown: Largest peak-to-trough decline
* Value at Risk (VaR): Potential loss at confidence level

**Trading Metrics**
* Win Rate: Percentage of profitable trades
* Profit Factor: Gross profit / Gross loss
* Average Win/Loss: Average profit/loss per trade
* Trade Frequency: Number of trades per period
* Holding Period: Average time in position

Calculating Metrics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.evaluation import PerformanceAnalyzer

   # Analyze performance
   analyzer = PerformanceAnalyzer(results)

   # Calculate all metrics
   metrics = analyzer.calculate_metrics()

   print("Performance Metrics:")
   for metric, value in metrics.items():
       print(f"{metric}: {value}")

Custom Metrics
~~~~~~~~~~~~~~

You can define custom performance metrics:

.. code-block:: python

   from ztb.evaluation import MetricCalculator

   class CustomMetric(MetricCalculator):
       def calculate(self, returns):
           # Custom calculation logic
           return custom_value

   # Register custom metric
   analyzer.add_metric('custom_metric', CustomMetric())

Risk Analysis
-------------

Risk Assessment
~~~~~~~~~~~~~~~

Comprehensive risk analysis includes:

.. code-block:: python

   from ztb.evaluation import RiskAnalyzer

   # Perform risk analysis
   risk_analyzer = RiskAnalyzer(results)

   # Calculate risk metrics
   risk_metrics = risk_analyzer.calculate_risk_metrics()

   print("Risk Metrics:")
   print(f"VaR (95%): {risk_metrics.var_95:.2%}")
   print(f"Expected Shortfall: {risk_metrics.expected_shortfall:.2%}")
   print(f"Tail Risk: {risk_metrics.tail_risk:.2f}")

Stress Testing
~~~~~~~~~~~~~~

Test strategy under extreme market conditions:

.. code-block:: python

   # Stress test scenarios
   stress_tests = {
       'market_crash': {'volatility_multiplier': 2.0, 'trend': -0.05},
       'high_volatility': {'volatility_multiplier': 3.0},
       'liquidity_crisis': {'spread_multiplier': 5.0}
   }

   for scenario, params in stress_tests.items():
       stress_results = backtester.stress_test(scenario, **params)
       print(f"{scenario}: Return = {stress_results.total_return:.2%}")

Benchmarking
------------

Benchmark Strategies
~~~~~~~~~~~~~~~~~~~~

Compare your strategy against common benchmarks:

.. code-block:: python

   from ztb.evaluation import BenchmarkComparator

   # Define benchmarks
   benchmarks = {
       'buy_hold': 'Buy and Hold',
       'random': 'Random Trading',
       'moving_average': 'Moving Average Crossover'
   }

   # Compare against benchmarks
   comparator = BenchmarkComparator(results)
   comparison = comparator.compare_benchmarks(benchmarks)

   for benchmark, metrics in comparison.items():
       print(f"{benchmark}: Sharpe = {metrics.sharpe_ratio:.2f}")

Custom Benchmarks
~~~~~~~~~~~~~~~~~

Create custom benchmark strategies:

.. code-block:: python

   from ztb.strategies import BenchmarkStrategy

   class MyBenchmark(BenchmarkStrategy):
       def generate_signals(self, data):
           # Custom benchmark logic
           return signals

   # Register and compare
   comparator.add_benchmark('my_benchmark', MyBenchmark())
   results = comparator.run_comparison()

Walk-Forward Analysis
---------------------

Walk-Forward Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Prevent overfitting with walk-forward analysis:

.. code-block:: python

   from ztb.evaluation import WalkForwardAnalyzer

   # Configure walk-forward analysis
   wf_config = {
       'train_window': 252,    # Training window (trading days)
       'test_window': 21,      # Test window
       'step_size': 21         # Step size
   }

   # Run walk-forward analysis
   wf_analyzer = WalkForwardAnalyzer(config, wf_config)
   wf_results = wf_analyzer.run_analysis()

   print(f"Walk-Forward Sharpe: {wf_results.avg_sharpe:.2f}")
   print(f"Out-of-Sample Return: {wf_results.oos_return:.2%}")

Cross-Validation
~~~~~~~~~~~~~~~~

Time-series cross-validation:

.. code-block:: python

   from ztb.evaluation import TimeSeriesCrossValidator

   # Configure cross-validation
   cv_config = {
       'n_splits': 5,
       'test_size': 0.2,
       'gap': 5
   }

   # Run cross-validation
   cv = TimeSeriesCrossValidator(config, cv_config)
   cv_results = cv.run_cross_validation()

   print(f"CV Score: {cv_results.mean_score:.2f} (+/- {cv_results.std_score:.2f})")

Visualization
-------------

Performance Charts
~~~~~~~~~~~~~~~~~~

Generate comprehensive performance visualizations:

.. code-block:: python

   from ztb.evaluation import PerformanceVisualizer

   # Create visualizer
   visualizer = PerformanceVisualizer(results)

   # Generate plots
   visualizer.plot_equity_curve()
   visualizer.plot_drawdown()
   visualizer.plot_monthly_returns()
   visualizer.plot_trade_analysis()

   # Save plots
   visualizer.save_plots('results/plots/')

Custom Visualizations
~~~~~~~~~~~~~~~~~~~~~

Create custom performance charts:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Custom equity curve
   fig, ax = plt.subplots(figsize=(12, 6))
   ax.plot(results.equity_curve, label='Strategy')
   ax.plot(results.benchmark_equity, label='Benchmark', alpha=0.7)
   ax.set_title('Equity Curve Comparison')
   ax.legend()
   plt.savefig('results/custom_equity.png')

Reporting
---------

Automated Reports
~~~~~~~~~~~~~~~~~

Generate comprehensive evaluation reports:

.. code-block:: python

   from ztb.evaluation import ReportGenerator

   # Generate report
   report_gen = ReportGenerator(results)
   report = report_gen.generate_report(
       format='html',
       include_plots=True,
       benchmark_comparison=True
   )

   # Save report
   report.save('results/evaluation_report.html')

Report Contents
~~~~~~~~~~~~~~~

Evaluation reports include:

* Executive Summary
* Performance Metrics Table
* Risk Analysis
* Benchmark Comparison
* Trade Analysis
* Visualizations
* Recommendations

Custom Reports
~~~~~~~~~~~~~~

Create custom report templates:

.. code-block:: python

   from ztb.evaluation import CustomReport

   class MyReport(CustomReport):
       def add_custom_section(self):
           # Add custom analysis section
           pass

   # Generate custom report
   custom_report = MyReport(results)
   custom_report.generate()

Model Validation
----------------

Validation Techniques
~~~~~~~~~~~~~~~~~~~~~

Comprehensive model validation:

.. code-block:: python

   from ztb.evaluation import ModelValidator

   # Validate model
   validator = ModelValidator(model, test_data)

   # Run validation tests
   validation_results = validator.run_validation()

   print("Validation Results:")
   print(f"Data Drift: {validation_results.data_drift}")
   print(f"Model Stability: {validation_results.model_stability}")
   print(f"Performance Decay: {validation_results.performance_decay}")

Out-of-Sample Testing
~~~~~~~~~~~~~~~~~~~~~

Test on unseen data:

.. code-block:: python

   # Split data for out-of-sample testing
   train_data, oos_data = split_data(data, test_size=0.3)

   # Train on training data
   model.train(train_data)

   # Evaluate on out-of-sample data
   oos_results = model.evaluate(oos_data)

   print(f"OOS Return: {oos_results.total_return:.2%}")
   print(f"OOS Sharpe: {oos_results.sharpe_ratio:.2f}")

Monte Carlo Simulation
~~~~~~~~~~~~~~~~~~~~~~

Assess strategy robustness:

.. code-block:: python

   from ztb.evaluation import MonteCarloSimulator

   # Run Monte Carlo simulation
   simulator = MonteCarloSimulator(strategy, data)
   mc_results = simulator.run_simulation(n_simulations=1000)

   print("Monte Carlo Results:")
   print(f"Mean Return: {mc_results.mean_return:.2%}")
   print(f"Return Std: {mc_results.return_std:.2%}")
   print(f"VaR (95%): {mc_results.var_95:.2%}")

Best Practices
--------------

1. **Multiple Metrics**: Use multiple performance metrics for comprehensive evaluation
2. **Out-of-Sample Testing**: Always test on unseen data
3. **Benchmarking**: Compare against relevant benchmarks
4. **Walk-Forward Analysis**: Use walk-forward optimization to prevent overfitting
5. **Risk Management**: Focus on risk-adjusted returns, not just raw returns
6. **Visualization**: Use charts to understand performance patterns
7. **Documentation**: Document evaluation methodology and assumptions

Common Pitfalls
---------------

* **Overfitting**: Testing on the same data used for optimization
* **Data Snooping**: Using future information in backtests
* **Survivorship Bias**: Only testing on currently successful strategies
* **Look-Ahead Bias**: Using information not available at decision time
* **Transaction Costs**: Ignoring realistic trading costs

Troubleshooting
---------------

Common Evaluation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Inconsistent Results**
* Check random seeds
* Verify data preprocessing
* Ensure deterministic evaluation

**Poor Out-of-Sample Performance**
* Reduce model complexity
* Use regularization
* Implement walk-forward analysis

**High Drawdown**
* Implement stop-loss rules
* Reduce position sizes
* Add risk management

Next Steps
----------

* :doc:`../deployment/index` - Learn about model deployment
* :doc:`../api/index` - Explore the API reference
* :doc:`../examples/index` - See evaluation examples
