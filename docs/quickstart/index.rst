Quick Start Guide
================

This section provides step-by-step guides to get you started with Zaif Trade Bot quickly.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start:

   installation
   basic-usage
   first-backtest
   training-your-first-model

Installation
------------

Prerequisites
~~~~~~~~~~~~~

Zaif Trade Bot requires Python 3.11 or later. Make sure you have the following installed:

* Python 3.11+
* pip
* virtualenv (recommended)

Install from Source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/MakuhariYusuke/zaif-trade-bot.git
   cd zaif-trade-bot

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode
   pip install -e .

Install with Docker
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build the Docker image
   docker build -t zaif-trade-bot .

   # Run the container
   docker run -it zaif-trade-bot

Basic Usage
-----------

Configuration
~~~~~~~~~~~~~

Create a basic configuration file:

.. code-block:: yaml

   # config/trading.yaml
   trading:
     symbol: "BTC/JPY"
     initial_balance: 1000000
     position_size: 0.1

   training:
     algorithm: "PPO"
     total_timesteps: 100000

Running Your First Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb import ConfigManager, BacktestRunner

   # Load configuration
   config = ConfigManager.load_config('config/trading.yaml')

   # Run backtest
   runner = BacktestRunner(config)
   results = runner.run()

   print(f"Total Return: {results.total_return:.2f}%")
   print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.training import PPOTrainer

   # Initialize trainer
   trainer = PPOTrainer(config)

   # Train the model
   trainer.train()

   # Save the trained model
   trainer.save('models/my_first_model.zip')

Next Steps
----------

* :doc:`../configuration` - Learn about configuration options
* :doc:`../evaluation` - Understand performance evaluation
* :doc:`../deployment` - Deploy your trained models