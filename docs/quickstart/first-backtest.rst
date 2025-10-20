Running Your First Backtest
==========================

This tutorial will guide you through running your first backtest with Zaif Trade Bot.

What is Backtesting?
--------------------

Backtesting is the process of testing a trading strategy on historical data to evaluate its performance before deploying it in live markets. Zaif Trade Bot provides comprehensive backtesting capabilities with realistic market conditions.

Prerequisites
-------------

* Zaif Trade Bot installed (:doc:`installation`)
* Basic configuration created (:doc:`basic-usage`)
* Historical market data (CSV format)

Preparing Data
--------------

Zaif Trade Bot expects market data in CSV format with the following columns:

* ``timestamp``: Unix timestamp
* ``open``: Opening price
* ``high``: Highest price
* ``low``: Lowest price
* ``close``: Closing price
* ``volume``: Trading volume

Example data format:

.. code-block:: csv

   timestamp,open,high,low,close,volume
   1609459200,1000000,1050000,950000,1020000,100.5
   1609462800,1020000,1080000,1000000,1060000,95.2
   ...

Sample Data
~~~~~~~~~~~

If you don't have real market data, you can use the included sample data:

.. code-block:: bash

   # Check if sample data exists
   ls data/

   # Use synthetic data for testing
   python -c "
   import pandas as pd
   import numpy as np

   # Generate sample data
   dates = pd.date_range('2021-01-01', periods=1000, freq='1H')
   np.random.seed(42)

   data = pd.DataFrame({
       'timestamp': dates.astype(int) // 10**9,
       'open': 1000000 + np.random.randn(1000).cumsum() * 10000,
       'high': lambda x: x['open'] + abs(np.random.randn(1000)) * 5000,
       'low': lambda x: x['open'] - abs(np.random.randn(1000)) * 5000,
       'close': lambda x: x['open'] + np.random.randn(1000) * 8000,
       'volume': np.random.exponential(100, 1000)
   })

   # Fix OHLC relationships
   for i in range(len(data)):
       high = max(data.loc[i, ['open', 'close']].values) + abs(np.random.randn()) * 2000
       low = min(data.loc[i, ['open', 'close']].values) - abs(np.random.randn()) * 2000
       data.loc[i, 'high'] = high
       data.loc[i, 'low'] = low

   data.to_csv('data/sample_btc_data.csv', index=False)
   print('Sample data created: data/sample_btc_data.csv')
   "

Creating a Backtest Configuration
---------------------------------

Create a configuration file for your backtest:

.. code-block:: yaml

   # config/backtest.yaml
   trading:
     symbol: "BTC/JPY"
     initial_balance: 1000000  # 1,000,000 JPY
     position_size: 0.1        # 10% of balance per trade
     max_position_size: 0.5    # Maximum 50% of balance

   environment:
     transaction_cost: 0.0005  # 0.05% trading fee
     slippage: 0.0001          # 0.01% slippage
     max_steps: 1000           # Maximum steps per episode

   backtest:
     data_path: "data/sample_btc_data.csv"
     start_date: "2021-01-01"
     end_date: "2021-02-01"
     warmup_period: 24         # 24 hours warmup

   strategy:
     type: "simple_ma_crossover"
     fast_period: 10
     slow_period: 30

Running the Backtest
--------------------

Method 1: Using Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.backtesting import BacktestRunner
   from ztb.config import ConfigManager
   from ztb.data import DataLoader
   import matplotlib.pyplot as plt

   # Load configuration
   config = ConfigManager.load_config('config/backtest.yaml')

   # Load data
   loader = DataLoader()
   data = loader.load_csv(config.backtest.data_path)

   # Create backtest runner
   runner = BacktestRunner(config)

   # Run backtest
   print("Running backtest...")
   results = runner.run(data)

   # Print results
   print(f"Total Return: {results.total_return:.2f}%")
   print(f"Annual Return: {results.annual_return:.2f}%")
   print(f"Max Drawdown: {results.max_drawdown:.2f}%")
   print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
   print(f"Total Trades: {results.total_trades}")
   print(f"Win Rate: {results.win_rate:.2f}%")

   # Plot equity curve
   plt.figure(figsize=(12, 6))
   plt.plot(results.equity_curve)
   plt.title('Backtest Equity Curve')
   plt.xlabel('Time')
   plt.ylabel('Portfolio Value (JPY)')
   plt.grid(True)
   plt.show()

Method 2: Using Command Line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run backtest from command line
   ztb backtest --config config/backtest.yaml --output results/my_first_backtest.json

   # View results
   cat results/my_first_backtest.json

Method 3: Using Simple Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.strategies import SimpleMACrossover
   from ztb.backtesting import BacktestEngine

   # Create strategy
   strategy = SimpleMACrossover(fast_period=10, slow_period=30)

   # Create backtest engine
   engine = BacktestEngine(
       initial_balance=1000000,
       transaction_cost=0.0005
   )

   # Load data
   data = pd.read_csv('data/sample_btc_data.csv')
   data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
   data.set_index('timestamp', inplace=True)

   # Run backtest
   results = engine.run_backtest(strategy, data)

   print("Backtest Results:")
   print(f"Final Balance: {results.final_balance:,.0f} JPY")
   print(f"Total Return: {results.total_return:.2f}%")
   print(f"Max Drawdown: {results.max_drawdown:.2f}%")

Analyzing Results
-----------------

Understanding Key Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Total Return**: Overall percentage gain/loss
**Annual Return**: Annualized return rate
**Max Drawdown**: Largest peak-to-valley decline
**Sharpe Ratio**: Risk-adjusted return (higher is better)
**Win Rate**: Percentage of profitable trades
**Profit Factor**: Gross profit divided by gross loss

Example Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detailed analysis
   def analyze_backtest_results(results):
       print("=== Backtest Analysis ===")
       print(f"Initial Balance: {results.initial_balance:,.0f} JPY")
       print(f"Final Balance: {results.final_balance:,.0f} JPY")
       print(f"Total Return: {results.total_return:.2f}%")

       if hasattr(results, 'trades') and results.trades:
           profitable_trades = [t for t in results.trades if t.pnl > 0]
           losing_trades = [t for t in results.trades if t.pnl <= 0]

           print(f"Total Trades: {len(results.trades)}")
           print(f"Profitable Trades: {len(profitable_trades)}")
           print(f"Losing Trades: {len(losing_trades)}")
           print(f"Win Rate: {len(profitable_trades)/len(results.trades)*100:.1f}%")

           if profitable_trades:
               avg_win = sum(t.pnl for t in profitable_trades) / len(profitable_trades)
               print(f"Average Win: {avg_win:,.0f} JPY")

           if losing_trades:
               avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
               print(f"Average Loss: {avg_loss:,.0f} JPY")

       print(f"Max Drawdown: {results.max_drawdown:.2f}%")
       print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")

   # Analyze your results
   analyze_backtest_results(results)

Common Issues and Solutions
---------------------------

**No trades executed**
* Check your strategy parameters
* Verify data format and quality
* Ensure sufficient data points for indicators

**Poor performance**
* Start with simple strategies
* Adjust position sizing
* Consider transaction costs

**Memory errors**
* Reduce data size
* Use data sampling
* Check system resources

**Data format errors**
* Verify CSV column names
* Check timestamp format
* Ensure numeric data types

Next Steps
----------

* :doc:`training-your-first-model` - Learn how to train a reinforcement learning model
* :doc:`../guides/index` - Explore advanced backtesting techniques
* :doc:`../evaluation/index` - Learn about performance evaluation methods