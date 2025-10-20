Basic Usage
===========

This guide covers the basic usage of Zaif Trade Bot, including configuration and running simple operations.

Project Structure
-----------------

After installation, your project should have the following structure:

.. code-block:: text

   zaif-trade-bot/
   ├── ztb/                    # Main package
   ├── config/                 # Configuration files
   ├── models/                 # Trained models
   ├── data/                   # Training data
   ├── results/                # Backtest results
   ├── logs/                   # Log files
   └── docs/                   # Documentation

Configuration
-------------

Zaif Trade Bot uses YAML configuration files. Create a basic configuration:

.. code-block:: yaml

   # config/trading.yaml
   trading:
     symbol: "BTC/JPY"
     initial_balance: 1000000  # Starting balance in JPY
     position_size: 0.1        # Position size (10% of balance)
     max_position_size: 0.5    # Maximum position size

   environment:
     transaction_cost: 0.0005  # 0.05% trading fee
     max_steps: 1000          # Maximum steps per episode

   training:
     algorithm: "PPO"
     total_timesteps: 100000
     learning_rate: 0.0003

Configuration Files
~~~~~~~~~~~~~~~~~~~

**Trading Configuration** (``config/trading.yaml``):

.. code-block:: yaml

   trading:
     symbol: "BTC/JPY"          # Trading pair
     initial_balance: 1000000   # Starting capital
     position_size: 0.1         # Default position size
     max_position_size: 0.5     # Risk limit

**Training Configuration** (``config/training.yaml``):

.. code-block:: yaml

   training:
     algorithm: "PPO"           # Algorithm: PPO or SAC
     total_timesteps: 100000    # Training steps
     learning_rate: 0.0003      # Learning rate
     batch_size: 256           # Batch size
     n_epochs: 10              # Number of epochs

Basic Operations
----------------

Loading Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.config import ConfigManager

   # Load configuration
   config = ConfigManager.load_config('config/trading.yaml')
   print(f"Trading symbol: {config.trading.symbol}")
   print(f"Initial balance: {config.trading.initial_balance}")

Running a Simple Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.backtesting import BacktestRunner
   from ztb.config import ConfigManager

   # Load configuration
   config = ConfigManager.load_config('config/trading.yaml')

   # Create backtest runner
   runner = BacktestRunner(config)

   # Run backtest
   results = runner.run()

   # Print results
   print(f"Total Return: {results.total_return:.2f}%")
   print(f"Max Drawdown: {results.max_drawdown:.2f}%")
   print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")

Data Loading
~~~~~~~~~~~~

.. code-block:: python

   from ztb.data import DataLoader

   # Load market data
   loader = DataLoader()
   data = loader.load_csv('data/btc_jpy_data.csv')

   # Preprocess data
   processed_data = loader.preprocess(data)

   print(f"Loaded {len(data)} data points")
   print(f"Columns: {list(data.columns)}")

Model Training
~~~~~~~~~~~~~~

.. code-block:: python

   from ztb.training import PPOTrainer
   from ztb.config import ConfigManager

   # Load training configuration
   config = ConfigManager.load_config('config/training.yaml')

   # Initialize trainer
   trainer = PPOTrainer(config)

   # Train the model
   print("Starting training...")
   trainer.train()

   # Save the trained model
   trainer.save('models/my_trained_model.zip')
   print("Model saved!")

Command Line Interface
----------------------

Zaif Trade Bot also provides a command-line interface for common operations:

.. code-block:: bash

   # Show help
   ztb --help

   # Run backtest
   ztb backtest --config config/trading.yaml

   # Train model
   ztb train --config config/training.yaml --output models/my_model.zip

   # Evaluate model
   ztb evaluate --model models/my_model.zip --data data/test_data.csv

Logging
-------

Zaif Trade Bot uses Python's logging module. Configure logging level:

.. code-block:: python

   import logging

   # Set logging level
   logging.basicConfig(level=logging.INFO)

   # Or use the built-in logger
   from ztb.utils import setup_logging
   setup_logging(level='DEBUG')

Best Practices
--------------

1. **Always use virtual environments** to avoid dependency conflicts
2. **Start with small position sizes** when testing new strategies
3. **Validate your configuration** before running expensive operations
4. **Monitor resource usage** during training (CPU, memory, disk)
5. **Keep backups** of important models and results

Next Steps
----------

* :doc:`first-backtest` - Learn how to run your first backtest
* :doc:`training-your-first-model` - Train your first trading model
* :doc:`../guides/index` - Explore detailed guides for specific features