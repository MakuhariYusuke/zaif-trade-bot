Zaif Trade Bot Documentation
============================

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style

Zaif Trade Bot is an advanced algorithmic trading system that uses reinforcement learning to optimize trading strategies for cryptocurrency markets.

Features
--------

* **Reinforcement Learning**: PPO and SAC algorithms for strategy optimization
* **Multi-asset Support**: Bitcoin and other cryptocurrency trading
* **Risk Management**: Advanced risk controls and position sizing
* **Backtesting**: Comprehensive backtesting with realistic market conditions
* **Live Trading**: Production-ready trading with monitoring and alerting
* **Performance Analytics**: Detailed performance metrics and visualization

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from ztb import ConfigManager, BacktestAnalyzer

   # Load configuration
   config = ConfigManager.load_config('config/trading.yaml')

   # Run backtest
   analyzer = BacktestAnalyzer(config)
   results = analyzer.run_backtest()

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart/index
   configuration
   evaluation
   deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Development:

   contributing/index
   changelog/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`